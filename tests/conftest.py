import shutil

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection


def pytest_addoption(parser):
    parser.addoption("--update-references", action="store_true",
                     help="rewrite the reference images in tests/data/visual/ instead of comparing (test_visual.py)")


def pytest_configure(config):
    config.addinivalue_line("markers", "visual: renders compared with reference images (tests/test_visual.py)")


@pytest.fixture
def cu_fcc():
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((4, 4, 4))
    return DataCollection.from_ase(atoms)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


DATA = __import__("pathlib").Path(__file__).parent / "data"


@pytest.fixture
def vasp_dir():
    return DATA / "vasp"


@pytest.fixture
def lammps_dir():
    return DATA / "lammps"


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Never read or write the developer's real ~/.config/ewald during tests."""
    monkeypatch.setenv("EWALD_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("EWALD_PLUGIN_INDEX", "")    # no network: tests name their own indexes


# --- SSH/SFTP: an in-process server (tests/helpers/sftp_server.py) and an isolated ~/.ssh/config
@pytest.fixture
def client_key(tmp_path):
    paramiko = pytest.importorskip("paramiko")
    rsa = paramiko.RSAKey.generate(2048)
    path = tmp_path / "id_test"
    rsa.write_private_key_file(str(path))
    return rsa, path


@pytest.fixture
def ssh_env(tmp_path, monkeypatch):
    """Isolated ~/.ssh/config, known_hosts and cache; returns a function writing config entries."""
    pytest.importorskip("paramiko")
    from ewald.io import remote

    monkeypatch.setenv("EWALD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SSH_AUTH_SOCK", "")
    config = tmp_path / "ssh_config"
    known = tmp_path / "known_hosts"
    known.write_text("")
    monkeypatch.setenv("EWALD_SSH_CONFIG", str(config))

    def reset():  # also drops GUI prompts a MainWindow test may have installed
        remote.disconnect_all()
        remote.set_password_prompt(None)
        remote.set_host_key_prompt(None)
        remote.set_progress_callback(None)

    reset()
    entries = []

    def add(alias, server, identity=None, **extra):
        lines = [f"Host {alias}", "  HostName 127.0.0.1", f"  Port {server.port}", f"  User {server.user}",
                 f"  UserKnownHostsFile {known}", "  IdentitiesOnly yes"]
        if identity:
            lines.append(f"  IdentityFile {identity}")
        lines += [f"  {k} {v}" for k, v in extra.items()]
        entries.append("\n".join(lines))
        config.write_text("\n\n".join(entries) + "\n")
        return known

    yield add
    reset()


@pytest.fixture
def server(tmp_path, client_key, ssh_env, lammps_dir, vasp_dir):
    root = tmp_path / "remote_fs"
    data = root / "data"
    data.mkdir(parents=True)
    (root / "home" / "alice").mkdir(parents=True)
    for f in ("dump.custom.lammpstrj", "dump.atom.gz", "data.full.lmp"):
        shutil.copy(lammps_dir / f, data / f)
    for f in ("XDATCAR", "POSCAR"):
        shutil.copy(vasp_dir / f, data / f)
    from helpers.sftp_server import SFTPTestServer

    srv = SFTPTestServer(root, client_key=client_key[0])
    known = ssh_env("hpc", srv, identity=client_key[1])
    known.write_text(srv.known_hosts_line())
    yield srv
    srv.close()
