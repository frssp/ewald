import numpy as np
import pytest
from ase.neighborlist import primitive_neighbor_list

from ewald.data import Cell
from ewald.neighbors import _brute_force, find_pairs
from ewald.backend import CPU


def _pairset(i, j):
    return set(zip(np.asarray(i).tolist(), np.asarray(j).tolist()))


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False), (False, False, False)])
def test_cell_list_matches_ase_triclinic(rng, pbc):
    matrix = np.array([[14.0, 0, 0], [4.0, 13.0, 0], [-3.0, 2.5, 15.0]])
    cell = Cell(matrix, np.zeros(3), pbc)
    pos = rng.random((900, 3)) @ matrix
    i, j, d, dist = find_pairs(pos, cell, 3.1)
    ai, aj, ad = primitive_neighbor_list("ijd", np.array(pbc), matrix, pos, 3.1)
    assert _pairset(i, j) == _pairset(ai, aj)
    # delta and distance are consistent
    np.testing.assert_allclose(np.linalg.norm(np.asarray(d), axis=1), np.asarray(dist))


def test_brute_force_equals_cell_list(rng):
    matrix = np.diag([20.0, 18.0, 16.0])
    cell = Cell(matrix)
    pos = rng.random((700, 3)) @ matrix
    i, j, _, _ = find_pairs(pos, cell, 3.5)
    frac = cell.to_fractional(pos)
    bi, bj, _, _ = _brute_force(CPU, frac, matrix, cell.pbc, 3.5, chunk_bytes=10_000)
    assert _pairset(i, j) == _pairset(bi, bj)


def test_cutoff_too_large_raises(cu_fcc):
    with pytest.raises(ValueError, match="half the periodic cell"):
        find_pairs(cu_fcc.positions, cu_fcc.cell, 8.0)
