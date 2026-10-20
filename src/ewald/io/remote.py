"""Read files on other machines over SSH/SFTP, e.g. trajectories on an HPC cluster.

Remote locations are written like ``scp`` or as URLs::

    import_file("cluster:/scratch/me/run1/dump.lammpstrj")      # host alias from ~/.ssh/config
    import_file("me@login.hpc.edu:run1/XDATCAR")                # relative to the remote home
    import_file("ssh://me@login.hpc.edu:2222/scratch/me/dump.*.lammpstrj")

A saved profile (:mod:`ewald.io.profiles`, the Remotes side bar in the GUI) works as a host:
``import_file("nurion:/scratch/me/run3/dump.lammpstrj")`` uses its host, user, port, key, jump
host and the password or key passphrase saved in the OS keychain.

Connections follow ``~/.ssh/config`` (HostName, User, Port, IdentityFile,
IdentitiesOnly, ProxyJump, ProxyCommand, UserKnownHostsFile,
StrictHostKeyChecking) and use the SSH agent and default keys. Host keys are
checked against ``known_hosts``. When key authentication fails, the password or
one-time code is asked through :func:`set_password_prompt` (a terminal prompt
by default, a dialog in the GUI); only a profile keeps a password, in the keychain.

The native LAMMPS dump/data and XDATCAR readers read remote files in place:
frames are fetched on demand through a block cache, and a LAMMPS dump is
indexed by running ``grep -b`` on the remote side, so a large dump is never
copied. Every other format is downloaded once into a local cache
(:func:`cache_dir`) and reused while the remote size and mtime are unchanged.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import socket
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

_URL = re.compile(r"^(?:ssh|sftp)://(?:(?P<user>[^@/]+)@)?(?P<host>\[[^\]]+\]|[^:/]+)(?::(?P<port>\d+))?(?P<path>/.*)?$")
_SCP = re.compile(r"^(?:(?P<user>[^@/:\s\\]+)@)?(?P<host>\[[^\]]+\]|[^@/:\s\\]{2,}):(?P<path>.*)$")


@dataclass(frozen=True)
class RemotePath:
    """A file on another machine: ``host`` may be an alias from ``~/.ssh/config``."""

    host: str
    path: str
    user: str | None = None
    port: int | None = None

    def __str__(self) -> str:
        user = f"{self.user}@" if self.user else ""
        if self.port:
            path = self.path if self.path.startswith("/") else "/~/" + self.path
            return f"ssh://{user}{self.host}:{self.port}{path}"
        return f"{user}{self.host}:{self.path}"

    def with_path(self, path: str) -> RemotePath:
        return replace(self, path=path)

    @property
    def name(self) -> str:
        return posixpath.basename(self.path)


def parse_remote(path) -> RemotePath | None:
    """``RemotePath`` for ``ssh://``/``sftp://`` URLs and ``[user@]host:path``; ``None`` for local paths."""
    if isinstance(path, RemotePath):
        return path
    if not isinstance(path, str):
        return None
    m = _URL.match(path)
    if m:
        p = m["path"] or "/"
        if p.startswith("/~"):  # ssh://host/~/x -> relative to the remote home
            p = p[2:].lstrip("/") or "."
        return RemotePath(m["host"].strip("[]"), p, m["user"], int(m["port"]) if m["port"] else None)
    m = _SCP.match(path)
    if m and not os.path.exists(path):  # 'C:\x' never matches (host needs 2+ chars)
        p = m["path"] or "."
        if p == "~" or p.startswith("~/"):
            p = p[2:] or "."
        return RemotePath(m["host"].strip("[]"), p, m["user"])
    return None


def is_remote(path) -> bool:
    return parse_remote(path) is not None


# ----------------------------------------------------------------------- callbacks
def _terminal_prompt(prompt: str, echo: bool) -> str | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None
    if echo:
        return input(prompt)
    import getpass

    return getpass.getpass(prompt)


_password_prompt = _terminal_prompt
_host_key_prompt = None
_progress = None


def set_password_prompt(fn) -> None:
    """``fn(prompt, echo) -> str | None`` answers password / one-time-code prompts; ``None`` cancels."""
    global _password_prompt
    _password_prompt = fn or _terminal_prompt


def set_host_key_prompt(fn) -> None:
    """``fn(host, key_type, fingerprint) -> bool`` decides about hosts missing from known_hosts.

    Accepted keys are appended to known_hosts, like answering "yes" to ssh.
    Without a callback unknown hosts are refused (unless ``StrictHostKeyChecking``
    is ``no`` or ``accept-new`` in ``~/.ssh/config``).
    """
    global _host_key_prompt
    _host_key_prompt = fn


def set_progress_callback(fn) -> None:
    """``fn(done_bytes, total_bytes, label)`` is called while downloading into the cache."""
    global _progress
    _progress = fn


# ----------------------------------------------------------------------- connections
def ssh_config_path() -> Path:
    return Path(os.environ.get("EWALD_SSH_CONFIG") or Path.home() / ".ssh" / "config")


def _ssh_config():
    import paramiko

    cfg = paramiko.SSHConfig()
    path = ssh_config_path()
    if path.is_file():
        with open(path) as fh:
            cfg.parse(fh)
    return cfg


def config_hosts() -> list[str]:
    """Host aliases from ``~/.ssh/config`` without wildcards, for completion in the GUI."""
    try:
        hosts = _ssh_config().get_hostnames()
    except Exception:
        return []
    return sorted(h for h in hosts if not any(c in h for c in "*?!"))


def _require_paramiko():
    try:
        import paramiko  # noqa: F401
    except ImportError:
        raise ImportError("Reading files over SSH needs paramiko, which ewald installs by default; "
                          "reinstall ewald or run: pip install paramiko") from None


class RemoteError(OSError):
    pass


class Connection:
    """One authenticated SSH transport with an SFTP session, shared by every file on that host."""

    def __init__(self, host: str, user: str | None = None, port: int | None = None, proxyjump: str | None = None,
                 profile=None, password: str | None = None, passphrase: str | None = None, _depth: int = 0):
        """``host`` is a saved profile name (:mod:`ewald.io.profiles`), a ``~/.ssh/config`` alias or a
        host name. ``password``/``passphrase`` override the profile's saved ones."""
        import paramiko

        from . import profiles

        if _depth > 8:
            raise RemoteError(f"ProxyJump chain through {host!r} is too long (a loop?)")
        self.alias = host
        self._depth = _depth
        prof = profile if profile is not None else profiles.get(host)
        opts = _ssh_config().lookup(prof.host or host if prof else host)
        if prof is not None:
            opts = prof.ssh_options(opts)
            password = password if password is not None else profiles.secret(prof.name, "password")
            passphrase = passphrase if passphrase is not None else profiles.secret(prof.name, "passphrase")
        self.profile = prof
        self._password, self._passphrase = password or None, passphrase or None
        if proxyjump:  # the earlier hops of a 'ProxyJump a,b' chain
            opts["proxyjump"] = proxyjump
        self.hostname = opts.get("hostname", host)
        self.port = int(port or opts.get("port", 22))
        self.user = user or opts.get("user") or _local_user()
        self._jump = None
        sock = self._socket(opts)
        self.transport = paramiko.Transport(sock)
        try:
            known = self._known_host_entry(opts)
            if known:  # negotiate a host-key type known_hosts has, like OpenSSH does
                want = set(known) | ({"rsa-sha2-512", "rsa-sha2-256"} if "ssh-rsa" in known else set())
                sec = self.transport.get_security_options()
                sec.key_types = [k for k in sec.key_types if k in want] + [k for k in sec.key_types if k not in want]
            self.transport.start_client(timeout=float(opts.get("connecttimeout", 20)))
            self._check_host_key(opts, known)
            self._authenticate(opts)
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        except BaseException:
            self.close()
            raise
        self.lock = threading.RLock()
        self._grep = None  # unknown until first use

    def __repr__(self):
        return f"<Connection {self.user}@{self.hostname}:{self.port}>"

    # --- transport
    def _socket(self, opts):
        import paramiko

        jump = opts.get("proxyjump")
        if jump and jump.lower() != "none":
            *earlier, last = [h.strip() for h in jump.split(",")]
            hop = re.match(r"^(?:ssh://)?(?:([^@]+)@)?(\[[^\]]+\]|[^:]+)(?::(\d+))?$", last)
            if hop is None:
                raise RemoteError(f"cannot parse ProxyJump {jump!r}")
            self._jump = Connection(hop[2].strip("[]"), hop[1], int(hop[3]) if hop[3] else None,
                                    ",".join(earlier) or None, _depth=self._depth + 1)
            return self._jump.transport.open_channel("direct-tcpip", (self.hostname, self.port), ("127.0.0.1", 0))
        cmd = opts.get("proxycommand")
        if cmd and cmd.lower() != "none":
            return paramiko.ProxyCommand(cmd)
        return socket.create_connection((self.hostname, self.port), timeout=float(opts.get("connecttimeout", 20)))

    def _known_hosts_files(self, opts):
        files = opts.get("userknownhostsfile")
        files = files.split() if isinstance(files, str) else (files or [])
        files = [os.path.expanduser(f) for f in files] or [os.path.expanduser("~/.ssh/known_hosts")]
        return files

    @property
    def _host_name(self) -> str:
        return self.hostname if self.port == 22 else f"[{self.hostname}]:{self.port}"

    def _known_host_entry(self, opts) -> dict:
        """known_hosts keys (type -> key) for this host, from every UserKnownHostsFile."""
        import paramiko

        known = paramiko.HostKeys()
        for f in self._known_hosts_files(opts):
            if os.path.isfile(f):
                try:
                    known.load(f)
                except Exception:  # an unreadable line or file is skipped, as ssh does
                    pass
        entry = known.lookup(self._host_name) or (known.lookup(self.alias) if self.alias != self.hostname else None)
        return dict(entry or {})

    def _check_host_key(self, opts, entry):
        key = self.transport.get_remote_server_key()
        name = self._host_name
        files = self._known_hosts_files(opts)
        if key.get_name() in entry:
            if entry[key.get_name()] != key:
                raise RemoteError(f"Host key for {name} does not match known_hosts. "
                                  "It may have been reinstalled, or someone is intercepting the connection.")
            return
        fingerprint = _fingerprint(key)
        policy = str(opts.get("stricthostkeychecking", "ask")).lower()
        if policy == "no":
            return
        if entry:  # known, but the server offers another key type: refuse, never ask or add
            raise RemoteError(f"Host {name} is in known_hosts with a {', '.join(sorted(entry))} key, but the "
                              f"server offered {key.get_name()} {fingerprint}. Someone may be intercepting the "
                              f"connection; check with `ssh {self.alias}`.")
        if policy == "accept-new" or (policy != "yes" and _host_key_prompt
                                      and _host_key_prompt(name, key.get_name(), fingerprint)):
            _append_known_host(files[0], name, key)
            return
        raise RemoteError(f"Host {name} is not in known_hosts ({key.get_name()} {fingerprint}). "
                          f"Run `ssh {self.alias}` once to verify and add it.")

    def _authenticate(self, opts):
        import paramiko

        t = self.transport
        only = str(opts.get("identitiesonly", "no")).lower() == "yes"
        keys = []
        if not only:
            try:
                keys += list(paramiko.Agent().get_keys())
            except Exception:
                pass
        files = [os.path.expanduser(f) for f in opts.get("identityfile", [])]
        if not only:
            files += [os.path.expanduser(f"~/.ssh/id_{k}") for k in ("ed25519", "ecdsa", "rsa")]
        for f in dict.fromkeys(files):
            if os.path.isfile(f):
                key = _load_key(f, self._passphrase)
                if key is not None:
                    keys.append(key)
        allowed = None
        for key in keys:
            try:
                allowed = t.auth_publickey(self.user, key)
            except paramiko.BadAuthenticationType as exc:
                allowed = exc.allowed_types
                break
            except paramiko.AuthenticationException:
                continue
            if t.is_authenticated():
                return
            if allowed:  # partial success: a second factor follows
                break
        if t.is_authenticated():
            return
        self._interactive(allowed)

    def _interactive(self, allowed):
        t = self.transport
        target = f"{self.user}@{self.hostname}"
        cancelled = False
        if self._password:  # saved with the profile: try it first, ask if it is refused
            cancelled = self._try_password(allowed, self._password)
        if not t.is_authenticated() and not cancelled:
            cancelled = self._try_password(allowed, None)
        if not t.is_authenticated():
            raise RemoteError(f"Authentication to {target} failed" + (" (cancelled)" if cancelled else
                              "; check your SSH key or agent, or that the password is right"))

    def _try_password(self, allowed, saved) -> bool:
        """One password / keyboard-interactive attempt; ``saved`` answers password prompts
        (other prompts, e.g. one-time codes, are always asked). Returns True if cancelled."""
        import paramiko

        t = self.transport
        target = f"{self.user}@{self.hostname}"
        cancelled = False

        def ask(prompt, echo):
            if saved and not echo and "pass" in prompt.lower():
                return saved
            return _password_prompt(prompt, echo)

        def handler(title, instructions, prompts):
            nonlocal cancelled
            answers = []
            for text, echo in prompts:
                text = text.strip() or "Password:"
                ans = ask(f"{target} {text} ", echo)
                if ans is None:
                    cancelled = True
                    raise paramiko.AuthenticationException("cancelled")
                answers.append(ans)
            return answers

        try:
            if allowed is None or "keyboard-interactive" in allowed:
                t.auth_interactive(self.user, handler)
            elif "password" in allowed:
                pw = ask(f"{target} password: ", False)
                if pw is None:
                    cancelled = True
                else:
                    t.auth_password(self.user, pw)
        except paramiko.BadAuthenticationType as exc:
            if "password" in exc.allowed_types and not cancelled:
                pw = ask(f"{target} password: ", False)
                if pw is None:
                    cancelled = True
                else:
                    try:
                        t.auth_password(self.user, pw)
                    except paramiko.AuthenticationException:
                        pass
        except paramiko.AuthenticationException:
            pass
        return cancelled

    # --- operations
    def is_active(self) -> bool:
        return self.transport.is_active()

    def close(self):
        try:
            self.transport.close()
        finally:
            if self._jump is not None:
                self._jump.close()

    def home(self) -> str:
        with self.lock:
            return self.sftp.normalize(".")

    def stat(self, path):
        with self.lock:
            return self.sftp.stat(path)

    def listdir(self, path):
        with self.lock:
            return self.sftp.listdir_attr(path)

    def run(self, command: str, timeout: float = 600) -> tuple[int, bytes]:
        """Run ``command`` on the remote host; returns (exit status, stdout)."""
        chan = self.transport.open_session(timeout=timeout)
        try:
            chan.settimeout(timeout)
            chan.exec_command(command)
            chunks = []
            while True:
                data = chan.recv(1 << 20)
                if not data:
                    break
                chunks.append(data)
            return chan.recv_exit_status(), b"".join(chunks)
        finally:
            chan.close()


def _local_user() -> str:
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return "root"


def _load_key(path, passphrase: str | None = None):
    import paramiko

    def load(secret=None):  # passphrase is positional: 'passphrase=' in paramiko 3, 'password=' in 5
        return paramiko.PKey.from_path(path, secret.encode()) if secret else paramiko.PKey.from_path(path)

    try:
        return load()
    except Exception as exc:
        encrypted = isinstance(exc, paramiko.PasswordRequiredException) or (
            isinstance(exc, TypeError) and "encrypted" in str(exc))
        if not encrypted:
            return None
    if passphrase:
        try:
            return load(passphrase)
        except Exception:
            pass  # a stale saved passphrase: ask
    pw = _password_prompt(f"Passphrase for {path}: ", False)
    if pw is None:
        return None
    try:
        return load(pw)
    except Exception:
        return None


def _fingerprint(key) -> str:
    import base64

    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


def _append_known_host(path, name, key):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(f"{name} {key.get_name()} {key.get_base64()}\n")
    except OSError:
        pass  # accepted for this session only


_connections: dict[tuple, Connection] = {}
_connections_lock = threading.Lock()


def connect(host: str, user: str | None = None, port: int | None = None) -> Connection:
    """The cached connection to ``host`` (reconnects if it dropped)."""
    _require_paramiko()
    k = (host, user, port)
    with _connections_lock:
        conn = _connections.get(k)
        if conn is None or not conn.is_active():
            conn = _connections[k] = Connection(host, user, port)
        return conn


def disconnect(host: str) -> None:
    """Close the cached connections to ``host`` (e.g. after its profile changed)."""
    with _connections_lock:
        for k in [k for k in _connections if k[0] == host]:
            try:
                _connections.pop(k).close()
            except Exception:
                pass


def disconnect_all() -> None:
    with _connections_lock:
        for conn in _connections.values():
            try:
                conn.close()
            except Exception:
                pass
        _connections.clear()


def _conn(rp: RemotePath) -> Connection:
    return connect(rp.host, rp.user, rp.port)


# ----------------------------------------------------------------------- files
class RemoteFile(io.RawIOBase):
    """Seekable, read-only remote file with an LRU block cache and pipelined reads."""

    def __init__(self, conn: Connection, path: str, block_size: int = 1 << 18, cache_blocks: int = 128):
        super().__init__()
        self._conn = conn
        self._cache: OrderedDict[int, bytes] = OrderedDict()  # before open, so close() works if it fails
        with conn.lock:
            self._fh = conn.sftp.open(path, "rb")
        self.name = path
        self.size = self._fh.stat().st_size
        self._pos = 0
        self._bs = block_size
        self._max = cache_blocks
        self._last_block = -2

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _blocks(self, first: int, last: int) -> None:
        """Make blocks first..last resident, fetching the missing ones in one pipelined request."""
        missing = [b for b in range(first, last + 1) if b not in self._cache]
        if missing:
            ranges = [(b * self._bs, min(self._bs, self.size - b * self._bs)) for b in missing]
            with self._conn.lock:
                data = list(self._fh.readv(ranges))
            for b, chunk in zip(missing, data):
                self._cache[b] = chunk
        for b in range(first, last + 1):
            self._cache.move_to_end(b)
        while len(self._cache) > max(self._max, last - first + 1):
            self._cache.popitem(last=False)

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        first, last = self._pos // self._bs, (self._pos + n - 1) // self._bs
        ahead = last + 15 if first in (self._last_block, self._last_block + 1) else last  # sequential: read ahead
        self._blocks(first, min(ahead, (self.size - 1) // self._bs))
        self._last_block = last
        out = memoryview(buf)
        done = 0
        while done < n:
            b, off = divmod(self._pos, self._bs)
            chunk = self._cache[b][off:off + n - done]
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
            self._pos += len(chunk)
        return n

    def close(self):
        if not self.closed:
            try:
                self._fh.close()
            except Exception:  # never opened, or the connection is gone
                pass
            self._cache.clear()
        super().close()


def open_remote(path, buffering: int = 1 << 16):
    """Open a remote file for binary reading (seekable, buffered)."""
    rp = parse_remote(path)
    return io.BufferedReader(RemoteFile(_conn(rp), rp.path), buffering)


def stat(path):
    rp = parse_remote(path)
    return _conn(rp).stat(rp.path)


def exists(path) -> bool:
    try:
        stat(path)
        return True
    except FileNotFoundError:
        return False


def listdir(path):
    """``[(name, is_dir, size, mtime)]`` for a remote directory, directories first."""
    import stat as st

    rp = parse_remote(path)
    rows = [(a.filename, st.S_ISDIR(a.st_mode or 0), a.st_size or 0, a.st_mtime or 0)
            for a in _conn(rp).listdir(rp.path)]
    return sorted(rows, key=lambda r: (not r[1], r[0].lower()))


def home(path) -> str:
    """Absolute remote home directory of the host in ``path``."""
    rp = parse_remote(path)
    return _conn(rp).home()


def glob(pattern) -> list[str]:
    """Expand ``* ? [..]`` in a remote path (every component may contain wildcards)."""
    rp = parse_remote(pattern)
    conn = _conn(rp)
    absolute = rp.path.startswith("/")
    parts = [p for p in rp.path.split("/") if p]
    found = ["/" if absolute else ""]
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        nxt = []
        for base in found:
            if not any(c in part for c in "*?["):
                nxt.append(posixpath.join(base, part))
                continue
            try:
                entries = conn.listdir(base or ".")
            except OSError:
                continue
            import stat as st

            for a in entries:
                if fnmatch.fnmatchcase(a.filename, part) and (last or st.S_ISDIR(a.st_mode or 0)):
                    nxt.append(posixpath.join(base, a.filename))
        found = nxt
    out = []
    for p in sorted(found):
        try:
            conn.stat(p)
            out.append(str(rp.with_path(p)))
        except OSError:
            pass
    return out


def line_offsets(path, regex: str, start: int = 0) -> list[tuple[int, bytes]] | None:
    """Byte offset and text of every line matching the extended ``regex``, found by remote ``grep -b``.

    ``start`` (a line start) skips the bytes before it (``tail -c``), so a file that grows
    while it is being written is searched only from where the last search ended.

    Returns ``None`` when ``path`` is local or compressed, or the host has no
    usable grep, so the caller falls back to streaming the file.
    """
    rp = parse_remote(path)
    if rp is None or rp.path.endswith((".gz", ".bz2", ".xz")):
        return None
    conn = _conn(rp)
    if conn._grep is False:
        return None
    try:
        grep = f"LC_ALL=C grep -b -E {shlex.quote(regex)}"
        command = f"{grep} -- {shlex.quote(rp.path)}" if not start else \
            f"tail -c +{int(start) + 1} -- {shlex.quote(rp.path)} | {grep}"
        status, out = conn.run(command)
    except Exception:
        conn._grep = False
        return None
    if status not in (0, 1):
        if not start:  # a failing tail must not turn grep off for the whole connection
            conn._grep = False if status in (126, 127) else conn._grep
        return None
    conn._grep = True
    hits = []
    for line in out.splitlines():
        off, sep, text = line.partition(b":")
        if not sep or not off.isdigit():
            return None
        hits.append((int(off) + start, text))
    return hits


# ----------------------------------------------------------------------- local cache
def cache_dir() -> Path:
    """Where downloaded remote files are kept (``EWALD_CACHE_DIR`` overrides)."""
    env = os.environ.get("EWALD_CACHE_DIR")
    if env:
        return Path(env) / "remote"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ewald" / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "ewald"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ewald"
    return base / "remote"


def clear_cache() -> None:
    import shutil

    shutil.rmtree(cache_dir(), ignore_errors=True)


def fetch(path) -> str:
    """Local copy of a remote file, downloaded once and reused while size and mtime match."""
    rp = parse_remote(path)
    conn = _conn(rp)
    full = rp.path if rp.path.startswith("/") else posixpath.join(conn.home(), rp.path)
    st = conn.stat(full)
    key = hashlib.sha1(f"{conn.user}@{conn.hostname}:{conn.port}{full}".encode()).hexdigest()[:16]
    folder = cache_dir() / re.sub(r"[^\w.-]", "_", conn.hostname) / key
    local = folder / (posixpath.basename(full) or "file")
    meta_path = folder / "meta.json"
    meta = {"source": f"{conn.user}@{conn.hostname}:{full}", "size": st.st_size, "mtime": st.st_mtime}
    try:
        if local.is_file() and json.loads(meta_path.read_text()) == meta:
            return str(local)
    except (OSError, ValueError):
        pass
    folder.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    label = f"{rp.host}:{full}"
    callback = (lambda done, total: _progress(done, total, label)) if _progress else None
    with conn.lock:
        conn.sftp.get(full, str(tmp), callback=callback)
    os.replace(tmp, local)
    meta_path.write_text(json.dumps(meta))
    return str(local)


def download(path, dest) -> str:
    """Copy a remote file to ``dest`` (a file name, or a folder to keep the remote name); returns the local path."""
    rp = parse_remote(path)
    conn = _conn(rp)
    dest = Path(os.path.expanduser(str(dest)))
    if dest.is_dir():
        dest = dest / (posixpath.basename(rp.path.rstrip("/")) or "file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    label = f"{rp.host}:{rp.path}"
    callback = (lambda done, total: _progress(done, total, label)) if _progress else None
    try:
        with conn.lock:
            conn.sftp.get(rp.path, str(tmp), callback=callback)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return str(dest)


__all__ = ["RemotePath", "parse_remote", "is_remote", "connect", "disconnect", "disconnect_all", "open_remote", "stat",
           "exists", "listdir", "home", "glob", "fetch", "download", "cache_dir", "clear_cache", "config_hosts",
           "set_password_prompt", "set_host_key_prompt", "set_progress_callback", "RemoteError"]
