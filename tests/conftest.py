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
