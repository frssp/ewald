"""In-process SSH/SFTP server (paramiko) for the remote-file tests; no sshd needed.

The server exposes ``root`` as its filesystem ("/data/x" -> root/data/x), accepts
one public key and/or a password, runs ``grep`` exec requests against the mapped
files when the machine has grep, and forwards direct-tcpip channels (ProxyJump).
"""
from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import threading

import paramiko
from paramiko import SFTPAttributes, SFTPHandle, SFTPServer, SFTPServerInterface


class _Handle(SFTPHandle):
    def stat(self):
        return SFTPAttributes.from_stat(os.fstat(self.readfile.fileno()))


class _SFTP(SFTPServerInterface):
    def __init__(self, iface, *args, **kwargs):
        self.server = iface.server  # the SFTPTestServer
        super().__init__(iface, *args, **kwargs)

    def _local(self, path):
        return self.server.local(path)

    def _err(self, exc):
        return SFTPServer.convert_errno(exc.errno)

    def list_folder(self, path):
        try:
            local = self._local(path)
            out = []
            for name in os.listdir(local):
                attr = SFTPAttributes.from_stat(os.stat(os.path.join(local, name)))
                attr.filename = name
                out.append(attr)
            return out
        except OSError as exc:
            return self._err(exc)

    def stat(self, path):
        try:
            return SFTPAttributes.from_stat(os.stat(self._local(path)))
        except OSError as exc:
            return self._err(exc)

    lstat = stat

    def open(self, path, flags, attr):
        try:
            fh = open(self._local(path), "rb")
        except OSError as exc:
            return self._err(exc)
        self.server.opens.append(path)
        h = _Handle(flags)
        h.readfile = fh
        h.filename = path
        return h

    def canonicalize(self, path):
        if not path.startswith("/"):
            path = "/home/alice/" + path
        return os.path.normpath(path).replace("\\", "/").replace("//", "/") or "/"


class _Interface(paramiko.ServerInterface):
    def __init__(self, server):
        self.server = server

    def get_allowed_auths(self, username):
        return ",".join(a for a, on in (("publickey", self.server.client_key), ("password", self.server.password),
                                        ("keyboard-interactive", self.server.password)) if on)

    def check_auth_publickey(self, username, key):
        ok = self.server.client_key is not None and key == self.server.client_key and username == self.server.user
        return paramiko.AUTH_SUCCESSFUL if ok else paramiko.AUTH_FAILED

    def check_auth_password(self, username, password):
        ok = self.server.password and password == self.server.password and username == self.server.user
        return paramiko.AUTH_SUCCESSFUL if ok else paramiko.AUTH_FAILED

    def check_auth_interactive(self, username, submethods):
        return paramiko.server.InteractiveQuery("", "", ("Password: ", False))

    def check_auth_interactive_response(self, responses):
        return paramiko.AUTH_SUCCESSFUL if list(responses) == [self.server.password] else paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_direct_tcpip_request(self, chanid, origin, destination):
        if not self.server.allow_forward:
            return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
        self.server.forwards.append(destination)
        self.server._pending_forward = destination
        return paramiko.OPEN_SUCCEEDED

    def check_channel_exec_request(self, channel, command):
        self.server.commands.append(command.decode() if isinstance(command, bytes) else command)
        if not self.server.allow_exec:
            return False
        threading.Thread(target=self.server._exec, args=(channel, command), daemon=True).start()
        return True


class SFTPTestServer:
    def __init__(self, root, client_key=None, password=None, user="alice", allow_exec=True, allow_forward=False):
        self.root = str(root)
        self.client_key = client_key
        self.password = password
        self.user = user
        self.allow_exec = allow_exec
        self.allow_forward = allow_forward
        self.host_key = paramiko.RSAKey.generate(2048)
        self.opens, self.commands, self.forwards = [], [], []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._transports = []
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def local(self, path):
        return os.path.join(self.root, *[p for p in path.split("/") if p])

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            t = paramiko.Transport(conn)
            t.add_server_key(self.host_key)
            t.set_subsystem_handler("sftp", SFTPServer, _SFTP)
            self._transports.append(t)
            iface = _Interface(self)
            try:
                t.start_server(server=iface)
            except Exception:
                continue
            threading.Thread(target=self._accept_channels, args=(t,), daemon=True).start()

    def _accept_channels(self, t):
        while t.is_active():
            chan = t.accept(1)
            if chan is None:
                continue
            dest, self._pending_forward = getattr(self, "_pending_forward", None), None
            if dest is not None:
                threading.Thread(target=self._pump, args=(chan, dest), daemon=True).start()

    def _pump(self, chan, dest):
        sock = socket.create_connection(dest)

        def a_to_b(src, dst, recv, send):
            try:
                while True:
                    data = recv(65536)
                    if not data:
                        break
                    send(data)
            except Exception:
                pass
            finally:
                for c in (src, dst):
                    try:
                        c.close()
                    except Exception:
                        pass

        threading.Thread(target=a_to_b, args=(chan, sock, chan.recv, sock.sendall), daemon=True).start()
        a_to_b(sock, chan, sock.recv, chan.sendall)

    def _exec(self, channel, command):
        try:
            self._run(channel, command.decode() if isinstance(command, bytes) else command)
        except Exception:
            channel.send_exit_status(255)
        finally:
            channel.close()

    def _run(self, channel, command):
        args = shlex.split(command)
        env = {}
        while args and "=" in args[0]:
            k, v = args.pop(0).split("=", 1)
            env[k] = v
        grep = shutil.which("grep")
        if not args or args[0] != "grep" or grep is None:
            channel.send_exit_status(127)
            return
        args = [grep] + args[1:-1] + [self.local(args[-1])]
        proc = subprocess.run(args, capture_output=True, env={**os.environ, **env})
        channel.sendall(proc.stdout)
        channel.send_exit_status(proc.returncode)

    def known_hosts_line(self, host="127.0.0.1"):
        return f"[{host}]:{self.port} {self.host_key.get_name()} {self.host_key.get_base64()}\n"

    def close(self):
        self._stop = True
        self._sock.close()
        for t in self._transports:
            t.close()
