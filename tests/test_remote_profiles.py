"""Saved remote profiles (ewald.io.profiles): storage without secrets, keychain, login, CLI."""
import json
import shutil

import pytest

paramiko = pytest.importorskip("paramiko")

from ewald import import_file, keychain, settings  # noqa: E402
from ewald.io import profiles, remote  # noqa: E402
from ewald.io.profiles import Profile  # noqa: E402

from helpers.sftp_server import SFTPTestServer  # noqa: E402


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, value):
        self.store[(service, user)] = value

    def delete_password(self, service, user):
        self.store.pop((service, user))


@pytest.fixture
def kr(monkeypatch):
    k = FakeKeyring()
    monkeypatch.setattr(keychain, "_keyring", lambda: k)
    return k


@pytest.fixture
def known(tmp_path, ssh_env, monkeypatch):
    """known_hosts for hosts that are not in ~/.ssh/config (profiles name 127.0.0.1 directly)."""
    path = tmp_path / "known_hosts"
    cfg = tmp_path / "ssh_config"
    cfg.write_text(f"Host *\n  UserKnownHostsFile {path}\n")
    return path


@pytest.fixture
def fs(tmp_path, lammps_dir, vasp_dir):
    root = tmp_path / "fs"
    (root / "run").mkdir(parents=True)
    shutil.copy(lammps_dir / "dump.atom", root / "run" / "dump.atom")
    shutil.copy(vasp_dir / "POSCAR", root / "run" / "POSCAR")
    return root


def _no_prompt(prompt, echo):
    raise AssertionError(f"asked: {prompt}")


def test_saved_without_secrets_and_renamed(kr):
    p = profiles.save(Profile("nurion", host="nurion.ksc.re.kr", user="me", auth="password",
                              folder="/scratch/me", bookmarks=["/scratch/me/a", "/scratch/me/a"]),
                      password="hunter2")
    assert p.bookmarks == ["/scratch/me/a"]
    text = (settings.config_dir() / "remote_profiles.json").read_text()
    assert "hunter2" not in text and "nurion.ksc.re.kr" in text
    assert kr.store[("ewald", "ssh-password:nurion")] == "hunter2"
    assert not (settings.config_dir() / "credentials.json").exists()
    assert profiles.get("nurion").target == "me@nurion.ksc.re.kr"
    assert profiles.get("nurion").url("/x/dump") == "nurion:/x/dump"

    profiles.save(Profile("kisti", host="nurion.ksc.re.kr", user="me", auth="password"), old_name="nurion")
    assert profiles.names() == ["kisti"]
    assert profiles.secret("kisti") == "hunter2" and profiles.secret("nurion") is None
    profiles.save(Profile("other", host="h"))
    with pytest.raises(ValueError, match="already exists"):
        profiles.save(Profile("other", host="h"), old_name="kisti")
    profiles.move("other", -1)
    assert profiles.names() == ["other", "kisti"]
    profiles.add_bookmark("kisti", "/runs")
    assert profiles.get("kisti").bookmarks == ["/runs"]
    profiles.remove_bookmark("kisti", "/runs")
    assert profiles.get("kisti").bookmarks == []
    profiles.delete("kisti")
    assert profiles.names() == ["other"] and kr.store == {}


def test_validation():
    for bad in ("x", "a:b", "me@h", "a b", ""):
        with pytest.raises(ValueError):
            profiles.validate(Profile(bad))
    with pytest.raises(ValueError, match="key file"):
        profiles.validate(Profile("ok", auth="key"))
    with pytest.raises(ValueError, match="User"):
        profiles.validate(Profile("ok", host="me@host"))


def test_file_fallback_keeps_api_key(monkeypatch):
    monkeypatch.setattr(keychain, "_keyring", lambda: None)
    from ewald.llm import credentials

    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    credentials.save_api_key("sk-ant-file-key-000000000")
    profiles.save(Profile("pw", host="h", auth="password"), password="pw1")
    path = settings.config_dir() / "credentials.json"
    assert json.loads(path.read_text()) == {"anthropic-api-key": "sk-ant-file-key-000000000",
                                            "ssh-password:pw": "pw1"}
    assert profiles.secret_location("pw") == "file"
    credentials.delete_api_key()  # must not drop the SSH password
    assert profiles.secret("pw") == "pw1"
    profiles.delete("pw")
    assert not path.exists()


def test_password_profile_logs_in_with_saved_password(kr, known, fs):
    srv = SFTPTestServer(fs, password="s3cret")
    try:
        known.write_text(srv.known_hosts_line())
        profiles.save(Profile("lab", host="127.0.0.1", port=srv.port, user="alice", auth="password"),
                      password="s3cret")
        remote.set_password_prompt(_no_prompt)
        assert import_file("lab:/run/dump.atom").num_frames > 0
        assert profiles.test("lab")[0]

        profiles.save(profiles.get("lab"), password="stale")  # wrong saved password: asked instead
        asked = []
        remote.set_password_prompt(lambda prompt, echo: asked.append(prompt) or "s3cret")
        assert import_file("lab:/run/POSCAR").num_frames == 1
        assert len(asked) == 1
    finally:
        srv.close()


def test_key_profile_with_saved_passphrase(kr, known, fs, tmp_path):
    rsa = paramiko.RSAKey.generate(2048)
    key_file = tmp_path / "id_locked"
    rsa.write_private_key_file(str(key_file), password="phrase")
    srv = SFTPTestServer(fs, client_key=rsa)
    try:
        known.write_text(srv.known_hosts_line())
        profiles.save(Profile("hpc2", host="127.0.0.1", port=srv.port, user="alice", auth="key",
                              key_file=str(key_file)), passphrase="phrase")
        remote.set_password_prompt(_no_prompt)
        rows = remote.listdir("hpc2:/run")
        assert {r[0] for r in rows} == {"dump.atom", "POSCAR"}
        ok, msg = profiles.test("hpc2")
        assert ok and "alice@127.0.0.1" in msg
        remote.set_password_prompt(lambda prompt, echo: None)  # cancel
        assert not profiles.test(Profile("hpc2", host="127.0.0.1", port=srv.port, user="alice", auth="password"))[0]
    finally:
        srv.close()


def test_profile_jump_through_profile(kr, known, fs, client_key, tmp_path):
    jump = SFTPTestServer(tmp_path, client_key=client_key[0], allow_forward=True)
    target = SFTPTestServer(fs, password="pw2")
    try:
        known.write_text(jump.known_hosts_line() + target.known_hosts_line())
        profiles.save(Profile("login", host="127.0.0.1", port=jump.port, user="alice", auth="key",
                              key_file=str(client_key[1])))
        profiles.save(Profile("compute", host="127.0.0.1", port=target.port, user="alice", auth="password",
                              jump="login"), password="pw2")
        remote.set_password_prompt(_no_prompt)
        assert import_file("compute:/run/dump.atom").num_frames > 0
        assert jump.forwards == [("127.0.0.1", target.port)]
        profiles.save(Profile("loop", host="127.0.0.1", jump="loop"))
        ok, msg = profiles.test("loop")
        assert not ok and "too long" in msg
    finally:
        jump.close()
        target.close()


def test_profile_naming_a_config_alias(kr, server):
    profiles.save(Profile("work", host="hpc", folder="/data"))
    remote.set_password_prompt(_no_prompt)
    assert import_file("work:/data/POSCAR").num_frames == 1
    assert profiles.unsaved_config_hosts() == []
    profiles.delete("work")
    assert profiles.unsaved_config_hosts() == ["hpc"]
    assert profiles.from_ssh_config("hpc").host == "hpc"


def test_session_keeps_remote_path(server, tmp_path):
    from ewald.session import source_from_dict, source_to_dict

    pipe = import_file("hpc:/data/POSCAR")
    entry = source_to_dict(pipe.source, str(tmp_path))
    assert entry["path"] == "hpc:/data/POSCAR" and entry["relative_path"] is None
    assert source_from_dict(entry, str(tmp_path)).path == "hpc:/data/POSCAR"


def test_cli(kr, server, tmp_path, capsys, monkeypatch):
    from ewald.__main__ import main

    monkeypatch.setattr("getpass.getpass", lambda prompt="": "pw-from-tty")
    main(["remote", "add", "cl", "--host", "hpc", "--folder", "/data", "--password"])
    assert profiles.secret("cl") == "pw-from-tty"
    assert "pw-from-tty" not in (settings.config_dir() / "remote_profiles.json").read_text()
    main(["remote", "add", "cl", "--folder", "/data"])  # update keeps host and auth
    assert profiles.get("cl").host == "hpc" and profiles.get("cl").auth == "password"
    profiles.save(Profile("cl", host="hpc", folder="/data"))
    main(["remote", "list"])
    out = capsys.readouterr().out
    assert "cl" in out and "password saved" in out
    main(["remote", "ls", "cl"])
    assert "POSCAR" in capsys.readouterr().out
    assert main(["remote", "test", "cl"]) == 0
    main(["remote", "get", "cl:/data/POSCAR", str(tmp_path)])
    assert (tmp_path / "POSCAR").is_file()
    main(["remote", "remove", "cl"])
    assert profiles.get("cl") is None and profiles.secret("cl") is None
