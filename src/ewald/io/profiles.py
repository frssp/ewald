"""Saved remote computers ("profiles"), like the session list of an SSH terminal app.

A profile's name works as a host name everywhere a remote path is accepted::

    from ewald.io import profiles
    profiles.save(profiles.Profile("nurion", host="nurion.ksc.re.kr", user="me",
                                   auth="key", key_file="~/.ssh/id_ed25519", folder="/scratch/me"),
                  passphrase="...")                       # passphrase goes to the OS keychain
    import_file("nurion:/scratch/me/run3/dump.lammpstrj")  # notebooks and .ewd keep just this

Profiles are stored in ``<config dir>/remote_profiles.json`` without secrets. A password or
key passphrase is saved with :mod:`ewald.keychain` (OS keychain, 0600 file fallback) under
``ssh-password:<name>`` / ``ssh-passphrase:<name>``. One-time codes are always asked.

Settings missing from a profile come from ``~/.ssh/config`` for its ``host``, so a profile
can also just name a config alias and add a saved password or a start folder. A profile
with the same name as a config alias takes precedence over the alias.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields, replace

from .. import keychain

AUTH = {"auto": "SSH agent / keys from ~/.ssh/config",
        "key": "Key file",
        "password": "Password"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")


@dataclass
class Profile:
    """One remote computer. ``host`` defaults to ``name`` (e.g. an alias from ``~/.ssh/config``)."""

    name: str
    host: str = ""
    user: str = ""
    port: int | None = None
    auth: str = "auto"
    key_file: str = ""
    jump: str = ""  # ProxyJump: another profile, a config alias or user@host[:port]; a,b chains
    folder: str = ""  # where the file browser starts (empty: the remote home)
    bookmarks: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def target(self) -> str:
        """``user@host:port`` for display."""
        host = self.host or self.name
        return (f"{self.user}@" if self.user else "") + host + (f":{self.port}" if self.port else "")

    def url(self, path: str = "") -> str:
        """The remote path ``name:path`` for :func:`ewald.import_file` and ``load``."""
        return f"{self.name}:{path or self.folder or '.'}"

    def ssh_options(self, opts: dict) -> dict:
        """``opts`` (an ``~/.ssh/config`` lookup of ``host``) with this profile's settings applied."""
        opts = dict(opts)
        if self.user:
            opts["user"] = self.user
        if self.port:
            opts["port"] = str(self.port)
        if self.jump:
            opts["proxyjump"] = self.jump
        if self.auth == "key" and self.key_file:
            opts["identityfile"] = [self.key_file]
            opts["identitiesonly"] = "yes"
        elif self.auth == "password":
            opts["identityfile"] = []
            opts["identitiesonly"] = "yes"
        elif self.key_file:
            opts["identityfile"] = [self.key_file] + list(opts.get("identityfile", []))
        return opts


def _path():
    from ..settings import config_dir

    return config_dir() / "remote_profiles.json"


def _from_dict(d: dict) -> Profile | None:
    known = {f.name for f in fields(Profile)}
    try:
        p = Profile(**{k: v for k, v in d.items() if k in known})
    except TypeError:
        return None
    p.port = int(p.port) if p.port else None
    p.bookmarks = [str(b) for b in (p.bookmarks or [])]
    return p if _NAME.match(str(p.name)) else None


def profiles() -> list[Profile]:
    """Saved profiles, in the order the user arranged them."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data.get("profiles", []) if isinstance(data, dict) else []
    return [p for p in (_from_dict(r) for r in rows if isinstance(r, dict)) if p is not None]


def names() -> list[str]:
    return [p.name for p in profiles()]


def get(name: str) -> Profile | None:
    return next((p for p in profiles() if p.name == name), None)


def _write(items: list[Profile]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in asdict(p).items() if v not in ("", None, [])} | {"name": p.name} for p in items]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "profiles": rows}, indent=1), encoding="utf-8", newline="\n")
    tmp.replace(path)


def validate(profile: Profile) -> None:
    if not _NAME.match(profile.name or ""):
        raise ValueError("A profile name needs 2+ characters: letters, digits, '.', '_' or '-' "
                         "(it is used as the host in name:/path).")
    if profile.auth not in AUTH:
        raise ValueError(f"auth must be one of {', '.join(AUTH)}")
    if profile.auth == "key" and not profile.key_file:
        raise ValueError("Choose a key file, or use 'SSH agent / keys' authentication.")
    if any(c in (profile.host or "") for c in " /:@"):
        raise ValueError("Host is a host name, IP address or ~/.ssh/config alias (put the user in User).")


def save(profile: Profile, password: str | None = None, passphrase: str | None = None,
         old_name: str | None = None) -> Profile:
    """Add or replace a profile (``old_name`` renames one, moving its saved secrets).

    ``password`` / ``passphrase``: a string saves it in the keychain, ``""`` forgets it,
    ``None`` keeps what is saved.
    """
    validate(profile)
    profile = replace(profile, bookmarks=list(dict.fromkeys(profile.bookmarks)))
    items = profiles()
    old = old_name or profile.name
    if old != profile.name and any(p.name == profile.name for p in items):
        raise ValueError(f"A profile named {profile.name!r} already exists.")
    idx = next((i for i, p in enumerate(items) if p.name == old), None)
    if idx is None:
        items.append(profile)
    else:
        items[idx] = profile
    _write(items)
    if old != profile.name:
        for kind in ("password", "passphrase"):
            value = secret(old, kind)
            set_secret(old, kind, "")
            if value:
                set_secret(profile.name, kind, value)
    if password is not None:
        set_secret(profile.name, "password", password)
    if passphrase is not None:
        set_secret(profile.name, "passphrase", passphrase)
    _forget_connection(old)
    _forget_connection(profile.name)
    return profile


def delete(name: str) -> None:
    """Remove a profile and its saved password / passphrase."""
    _write([p for p in profiles() if p.name != name])
    for kind in ("password", "passphrase"):
        set_secret(name, kind, "")
    _forget_connection(name)


def move(name: str, offset: int) -> None:
    items = profiles()
    i = next((i for i, p in enumerate(items) if p.name == name), None)
    if i is None:
        return
    j = max(0, min(len(items) - 1, i + offset))
    items.insert(j, items.pop(i))
    _write(items)


def add_bookmark(name: str, folder: str) -> None:
    p = get(name)
    if p is not None and folder not in p.bookmarks:
        _write([replace(q, bookmarks=q.bookmarks + [folder]) if q.name == name else q for q in profiles()])


def remove_bookmark(name: str, folder: str) -> None:
    _write([replace(q, bookmarks=[b for b in q.bookmarks if b != folder]) if q.name == name else q
            for q in profiles()])


# ----------------------------------------------------------------- secrets
def _secret_name(name: str, kind: str) -> str:
    if kind not in ("password", "passphrase"):
        raise ValueError(kind)
    return f"ssh-{kind}:{name}"


def secret(name: str, kind: str = "password") -> str | None:
    """The saved password or key passphrase of profile ``name`` (``None`` if none)."""
    return keychain.get_secret(_secret_name(name, kind))[0]


def secret_location(name: str, kind: str = "password") -> str | None:
    """'keychain', 'file' or None."""
    return keychain.get_secret(_secret_name(name, kind))[1]


def set_secret(name: str, kind: str, value: str | None) -> str | None:
    """Save (a non-empty string) or forget (``""`` / ``None``) a password or passphrase."""
    if value:
        return keychain.set_secret(_secret_name(name, kind), value)
    keychain.delete_secret(_secret_name(name, kind))
    return None


# ----------------------------------------------------------------- ~/.ssh/config
def from_ssh_config(alias: str) -> Profile:
    """A profile that names the ``~/.ssh/config`` alias (its settings stay in the config)."""
    from . import remote

    return Profile(name=re.sub(r"[^A-Za-z0-9_.-]", "-", alias).strip("-.") or "host", host=alias,
                   note=f"from {remote.ssh_config_path()}")


def unsaved_config_hosts() -> list[str]:
    """``~/.ssh/config`` aliases that no profile names or points to."""
    from . import remote

    taken = {p.name for p in profiles()} | {p.host for p in profiles() if p.host}
    return [h for h in remote.config_hosts() if h not in taken]


# ----------------------------------------------------------------- connecting
def _forget_connection(name: str) -> None:
    try:
        from . import remote
    except ImportError:
        return
    remote.disconnect(name)


def test(name_or_profile, password: str | None = None, passphrase: str | None = None) -> tuple[bool, str]:
    """Open a fresh connection; (ok, message) with the remote home or the error."""
    from . import remote

    prof = get(name_or_profile) if isinstance(name_or_profile, str) else name_or_profile
    if prof is None:
        return False, f"no profile named {name_or_profile!r}"
    try:
        validate(prof)
        remote._require_paramiko()
        conn = remote.Connection(prof.name, profile=prof, password=password, passphrase=passphrase)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        return True, f"Connected to {conn.user}@{conn.hostname}:{conn.port}, home {conn.home()}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


__all__ = ["Profile", "AUTH", "profiles", "names", "get", "save", "delete", "move", "validate",
           "add_bookmark", "remove_bookmark", "secret", "secret_location", "set_secret",
           "from_ssh_config", "unsaved_config_hosts", "test"]
