import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection


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
