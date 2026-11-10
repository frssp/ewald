"""Reading files over SSH/SFTP against an in-process paramiko server."""
import os
import shutil

import numpy as np
import paramiko
import pytest

from ewald import import_file  # noqa: E402
from ewald.io import remote  # noqa: E402
from ewald.io.remote import RemotePath, parse_remote  # noqa: E402

from helpers.sftp_server import SFTPTestServer  # noqa: E402


def test_parse_remote_forms(tmp_path, monkeypatch):
    assert parse_remote("hpc:/scratch/a/dump.lammpstrj") == RemotePath("hpc", "/scratch/a/dump.lammpstrj")
    assert parse_remote("me@login.edu:run/XDATCAR") == RemotePath("login.edu", "run/XDATCAR", "me")
    assert parse_remote("me@login.edu:~/run/XDATCAR") == RemotePath("login.edu", "run/XDATCAR", "me")
    assert parse_remote("ssh://me@login.edu:2222/scratch/x") == RemotePath("login.edu", "/scratch/x", "me", 2222)
    assert parse_remote("sftp://login.edu/~/x") == RemotePath("login.edu", "x")
    assert parse_remote("ssh://[::1]:22/x").host == "::1"
    for local in ("dump.lammpstrj", "/abs/path", r"C:\data\XDATCAR", "C:/data/XDATCAR", "./a:b"):
        assert parse_remote(local) is None, local
    odd = tmp_path / "ab:cd"
    if os.name != "nt":
        odd.write_text("x")
        monkeypatch.chdir(tmp_path)
        assert parse_remote("ab:cd") is None  # an existing local file wins
    rp = RemotePath("login.edu", "/x/y", "me", 2222)
    assert parse_remote(str(rp)) == rp
    assert parse_remote(str(RemotePath("h", "rel/x", port=2))) == RemotePath("h", "rel/x", port=2)


def test_lammps_dump_read_in_place(server, lammps_dir, tmp_path):
    local = import_file(lammps_dir / "dump.custom.lammpstrj")
    pipe = import_file("hpc:/data/dump.custom.lammpstrj")
    assert pipe.num_frames == local.num_frames > 1
    for i in range(pipe.num_frames):
        a, b = pipe.compute(i), local.compute(i)
        np.testing.assert_allclose(a.positions, b.positions)
        assert a.attributes["Timestep"] == b.attributes["Timestep"]
    assert any(c.startswith("LC_ALL=C grep -b") for c in server.commands) == (shutil.which("grep") is not None)
    assert not (tmp_path / "cache").exists()  # nothing downloaded


def test_lammps_dump_without_remote_grep_streams(server, lammps_dir):
    server.allow_exec = False
    pipe = import_file("ssh://hpc/data/dump.custom.lammpstrj")
    local = import_file(lammps_dir / "dump.custom.lammpstrj")
    assert pipe.num_frames == local.num_frames
    np.testing.assert_allclose(pipe.compute(-1).positions, local.compute(-1).positions)


def test_compressed_dump_xdatcar_and_data_file(server, lammps_dir, vasp_dir):
    for name, src in (("dump.atom.gz", lammps_dir), ("XDATCAR", vasp_dir), ("data.full.lmp", lammps_dir)):
        pipe, local = import_file(f"hpc:/data/{name}"), import_file(src / name)
        assert pipe.num_frames == local.num_frames
        np.testing.assert_allclose(pipe.compute(-1).positions, local.compute(-1).positions)


def test_other_formats_download_once_to_cache(server, vasp_dir, tmp_path):
    pipe = import_file("alice@hpc:/data/POSCAR")
    local = import_file(vasp_dir / "POSCAR")
    np.testing.assert_allclose(pipe.compute(0).positions, local.compute(0).positions)
    cached = list((tmp_path / "cache" / "remote").rglob("POSCAR"))
    assert len(cached) == 1
    stamp = cached[0].stat().st_mtime_ns
    import_file("hpc:/data/POSCAR")
    assert cached[0].stat().st_mtime_ns == stamp  # reused, not downloaded again

    remote_file = tmp_path / "remote_fs" / "data" / "POSCAR"
    lines = remote_file.read_text().splitlines()
    lines[0] = "changed"
    remote_file.write_text("\n".join(lines) + "\n\n")
    import_file("hpc:/data/POSCAR")  # size changed -> downloaded again
    assert cached[0].read_text().startswith("changed")


def test_remote_glob_and_missing_file(server, lammps_dir):
    data = server.local("/data")
    for i in range(3):
        shutil.copy(lammps_dir / "dump.atom", os.path.join(data, f"dump.{i}.atom"))
    pipe = import_file("hpc:/data/dump.*.atom", type_map={1: "Cu"})
    assert len(pipe.source.paths) == 3 and pipe.source.paths[0] == "hpc:/data/dump.0.atom"
    assert pipe.num_frames == 3 * import_file(lammps_dir / "dump.atom").num_frames
    with pytest.raises(FileNotFoundError):
        import_file("hpc:/data/nothing.lammpstrj")
    with pytest.raises(FileNotFoundError):
        import_file("hpc:/data/nothing.*")


def test_listdir_and_home(server):
    rows = remote.listdir("hpc:/data")
    names = [r[0] for r in rows]
    assert "dump.custom.lammpstrj" in names and "XDATCAR" in names
    assert remote.home("hpc:.") == "/home/alice"
    assert "hpc" in remote.config_hosts()


def test_password_and_keyboard_interactive(tmp_path, ssh_env, lammps_dir):
    root = tmp_path / "fs"
    root.mkdir()
    shutil.copy(lammps_dir / "dump.atom", root / "dump.atom")
    srv = SFTPTestServer(root, password="s3cret")
    try:
        known = ssh_env("pw", srv)
        known.write_text(srv.known_hosts_line())
        asked = []
        remote.set_password_prompt(lambda prompt, echo: asked.append(prompt) or None)
        with pytest.raises(OSError, match="cancelled"):
            import_file("pw:/dump.atom")
        remote.set_password_prompt(lambda prompt, echo: asked.append(prompt) or "s3cret")
        assert import_file("pw:/dump.atom").num_frames > 0
        assert asked and "alice@127.0.0.1" in asked[-1]
    finally:
        srv.close()


def test_unknown_host_key_rejected_then_accepted(tmp_path, ssh_env, client_key, lammps_dir):
    root = tmp_path / "fs"
    root.mkdir()
    shutil.copy(lammps_dir / "dump.atom", root / "dump.atom")
    srv = SFTPTestServer(root, client_key=client_key[0])
    try:
        known = ssh_env("new", srv, identity=client_key[1])
        with pytest.raises(OSError, match="not in known_hosts"):
            import_file("new:/dump.atom")
        seen = []
        remote.set_host_key_prompt(lambda host, kind, fp: seen.append((host, kind, fp)) or True)
        assert import_file("new:/dump.atom").num_frames > 0
        assert seen[0][0] == f"[127.0.0.1]:{srv.port}" and seen[0][2].startswith("SHA256:")
        assert srv.host_key.get_base64() in known.read_text()  # remembered, like ssh's "yes"

        known.write_text(f"[127.0.0.1]:{srv.port} ssh-rsa {paramiko.RSAKey.generate(2048).get_base64()}\n")
        remote.disconnect_all()
        with pytest.raises(OSError, match="does not match"):
            import_file("new:/dump.atom")
    finally:
        srv.close()


def test_proxy_jump(tmp_path, ssh_env, client_key, lammps_dir):
    root = tmp_path / "fs"
    root.mkdir()
    shutil.copy(lammps_dir / "dump.atom", root / "dump.atom")
    jump = SFTPTestServer(tmp_path, client_key=client_key[0], allow_forward=True)
    target = SFTPTestServer(root, client_key=client_key[0])
    try:
        ssh_env("login", jump, identity=client_key[1])
        known = ssh_env("compute", target, identity=client_key[1], ProxyJump="login")
        known.write_text(jump.known_hosts_line() + target.known_hosts_line())
        assert import_file("compute:/dump.atom").num_frames > 0
        assert jump.forwards == [("127.0.0.1", target.port)]
    finally:
        jump.close()
        target.close()


def test_remote_file_block_cache(server, lammps_dir):
    raw = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    rp = parse_remote("hpc:/data/dump.custom.lammpstrj")
    f = remote.RemoteFile(remote.connect("hpc"), rp.path, block_size=97, cache_blocks=2)
    try:
        assert f.size == len(raw)
        assert f.read(10) == raw[:10]
        f.seek(len(raw) - 300)
        assert f.read() == raw[-300:]
        f.seek(5)
        buf = bytearray(1000)
        assert f.readinto(buf) == 1000 and bytes(buf) == raw[5:1005]
        assert len(f._cache) <= max(2, 1000 // 97 + 4)
    finally:
        f.close()


def test_volumetric_chgcar_via_cache(server, tmp_path):
    from test_volumetric import gaussians, write_vasp

    write_vasp(server.local("/data/CHGCAR"), [gaussians()])
    remote_grid = import_file("hpc:/data/CHGCAR").compute().grids["charge-density"]
    local_grid = import_file(server.local("/data/CHGCAR")).compute().grids["charge-density"]
    np.testing.assert_allclose(remote_grid["Charge Density"], local_grid["Charge Density"])
    assert list((tmp_path / "cache" / "remote").rglob("CHGCAR"))


def test_plugin_reader_gets_a_local_copy(server, monkeypatch):
    from pathlib import Path

    from ewald import DataCollection, plugins

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    seen = []

    def read_xyzlist(path):
        seen.append(path)
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        return DataCollection(np.array(rows, dtype=float), ["Cu"] * len(rows))

    plugins.register_reader("xyzlist", read_xyzlist, patterns=("*.xyzlist",))
    with open(server.local("/data/atoms.xyzlist"), "w") as fh:
        fh.write("0 0 0\n1 1 1\n")
    data = import_file("hpc:/data/atoms.xyzlist").compute(0)
    assert data.count == 2 and os.path.isfile(seen[0])  # the plugin read a local cached file


def test_known_host_with_another_key_type_is_refused(tmp_path, ssh_env, client_key, lammps_dir):
    import paramiko

    root = tmp_path / "fs"
    root.mkdir()
    shutil.copy(lammps_dir / "dump.atom", root / "dump.atom")
    srv = SFTPTestServer(root, client_key=client_key[0])  # offers only its RSA key
    try:
        known = ssh_env("swapped", srv, identity=client_key[1])
        other = paramiko.ECDSAKey.generate()
        known.write_text(f"[127.0.0.1]:{srv.port} {other.get_name()} {other.get_base64()}\n")
        asked = []
        remote.set_host_key_prompt(lambda *a: asked.append(a) or True)
        with pytest.raises(OSError, match="intercepting"):
            import_file("swapped:/dump.atom")
        assert not asked and srv.host_key.get_base64() not in known.read_text()  # neither asked nor added
    finally:
        srv.close()


def test_neb_folders_over_ssh(server, vasp_dir, lammps_dir, tmp_path):
    data = tmp_path / "remote_fs" / "data"
    shutil.copytree(vasp_dir / "neb", data / "vneb")
    shutil.copytree(lammps_dir / "neb", data / "lneb")
    for name, local in (("vneb", vasp_dir / "neb"), ("lneb", lammps_dir / "neb")):
        pipe, ref = import_file(f"hpc:/data/{name}"), import_file(local)
        assert pipe.num_frames == ref.num_frames >= 5
        for i in (0, 2, pipe.num_frames - 1):
            a, b = pipe.compute(i), ref.compute(i)
            np.testing.assert_allclose(a.positions, b.positions)
            assert a.attributes.get("Energy") == b.attributes.get("Energy")
        assert a.attributes["NEB.Barrier"] == pytest.approx(b.attributes["NEB.Barrier"])
    pipe = import_file("hpc:/data/lneb/dump.neb.*", format="neb")  # wildcard, log.lammps found next to it
    assert pipe.num_frames == 7 and pipe.compute(3).attributes["Energy"] == ref.compute(3).attributes["Energy"]


def test_threads_share_one_sftp_session(server, lammps_dir):
    """The compute worker and the GUI read through one SFTP session at the same time; every
    request holds the connection lock, so replies never interleave ("Garbage packet received")."""
    import threading

    raw = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    errors, done = [], []

    def reader():
        try:
            for _ in range(15):
                f = remote.RemoteFile(remote.connect("hpc"), "/data/dump.custom.lammpstrj", block_size=211)
                try:
                    assert f.read() == raw
                finally:
                    f.close()
            done.append(1)
        except Exception as e:  # pragma: no cover - the failure being tested
            errors.append(e)

    def browser():
        try:
            for _ in range(40):
                remote.listdir("hpc:/data")
                remote.stat("hpc:/data/POSCAR")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=browser) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert len(done) == 3


def test_broken_sftp_session_is_reopened(server, lammps_dir):
    """A broken SFTP session (garbage packet, closed channel) is reopened on the same SSH
    transport, and open files carry on with a fresh handle."""
    raw = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    conn = remote.connect("hpc")
    f = remote.RemoteFile(conn, "/data/dump.custom.lammpstrj", block_size=97, cache_blocks=2)
    try:
        assert f.read(10) == raw[:10]
        transport = conn.transport
        conn.sftp.close()  # the session is gone, the SSH connection is not
        assert remote.stat("hpc:/data/POSCAR").st_size > 0
        assert conn.generation == 1 and conn.transport is transport
        f.seek(len(raw) - 300)
        assert f.read() == raw[-300:]  # reopened its handle on the new session

        calls = []
        real = conn.sftp.listdir_attr

        def garbage_once(path):
            calls.append(path)
            if len(calls) == 1:
                raise paramiko.SFTPError("Garbage packet received")
            return real(path)

        conn.sftp.listdir_attr = garbage_once
        assert "POSCAR" in {r[0] for r in remote.listdir("hpc:/data")}
        assert conn.generation == 2
        with pytest.raises(FileNotFoundError):  # a missing file is not a broken session
            remote.stat("hpc:/data/nothing")
        assert conn.generation == 2
    finally:
        f.close()


def test_open_file_reconnects_after_the_ssh_connection_dropped(server, lammps_dir):
    """A dropped SSH transport (laptop sleep, Wi-Fi change, a profile edited) used to leave every
    open file failing on each read until it was opened again; now the next read connects again."""
    raw = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    buffered = remote.open_remote("hpc:/data/dump.custom.lammpstrj")
    f = buffered.raw                                # the block reader, not its buffer
    try:
        assert f.read(10) == raw[:10]
        old = f._conn
        old.transport.close()                       # the connection is gone, not just the SFTP session
        assert not old.is_active()
        f._cache.clear()
        f.seek(len(raw) - 300)
        assert f.read() == raw[-300:]
        new = remote.connect("hpc")
        assert new is not old and new.is_active() and f._conn is new
        remote.disconnect("hpc")                    # a profile changed: the next read connects again too
        f._cache.clear()
        f.seek(0)
        assert f.read(10) == raw[:10] and f._conn is not new
    finally:
        buffered.close()


def test_a_host_out_of_reach_does_not_hold_up_other_hosts(server, monkeypatch):
    """The handshake runs under that host's own lock: connecting to a host that does not answer
    (up to ConnectTimeout, or a password prompt) used to block every other host meanwhile."""
    import threading

    gate, entered = threading.Event(), threading.Event()
    real = remote.Connection

    class Stuck:
        def is_active(self):
            return True

        def close(self):
            pass

    def make(host, user=None, port=None):
        if host == "stuck":
            entered.set()
            gate.wait(20)
            return Stuck()
        return real(host, user, port)

    monkeypatch.setattr(remote, "Connection", make)
    t = threading.Thread(target=lambda: remote.connect("stuck"))
    t.start()
    try:
        assert entered.wait(10)
        assert remote.connect("hpc").is_active()    # while "stuck" is still connecting
    finally:
        gate.set()
        t.join(20)
    assert isinstance(remote.connect("stuck"), Stuck)   # connected once, cached
