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


def test_cutoff_beyond_half_the_cell_uses_periodic_images(cu_fcc, monkeypatch):
    """A cutoff over half the cell width (rdf(pipe, cutoff=6) on a small box; models asked for it
    in every real-model eval run) lists a pair once per image of j, as ASE's neighbor list does;
    an atom of a one-atom cell is its own neighbor."""
    from ase.build import bulk
    from ase.neighborlist import neighbor_list

    from ewald import neighbors

    for atoms, cutoff in [(bulk("Cu", "fcc", a=3.615), 6.0), (bulk("Si", "diamond", a=5.431, cubic=True), 4.0),
                          (bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 1)), 5.0)]:
        cell = Cell(np.array(atoms.cell), np.zeros(3), np.array([True, True, True]))
        ai, aj, ad, a_vec = neighbor_list("ijdD", atoms, cutoff)
        for method in ("auto", "brute", "xp"):
            i, j, delta, dist = find_pairs(atoms.positions, cell, cutoff, method=method)
            assert len(i) == len(ai)
            assert np.array_equal(np.bincount(i, minlength=len(atoms)), np.bincount(ai, minlength=len(atoms)))
            assert np.allclose(np.sort(dist), np.sort(ad)) and np.allclose(np.linalg.norm(delta, axis=1), dist)
            got = sorted(zip(i, j, np.round(delta, 6).tolist()))
            assert got == sorted(zip(ai, aj, np.round(a_vec, 6).tolist()))
            hi, hj, _, hdist = find_pairs(atoms.positions, cell, cutoff, half=True, method=method)
            assert 2 * len(hi) == len(i) and np.allclose(np.sort(np.r_[hdist, hdist]), np.sort(dist))
    prim = bulk("Cu", "fcc", a=3.615)
    i, j, _, _ = find_pairs(prim.positions, Cell(np.array(prim.cell)), 3.0)
    assert len(i) == 12 and not i.any() and not j.any()      # the primitive cell's one atom: its 12 images
    monkeypatch.setattr(neighbors, "MAX_IMAGE_ATOMS", 1000)
    with pytest.raises(ValueError, match="half the periodic cell"):
        find_pairs(cu_fcc.positions, cu_fcc.cell, 8.0)
