"""CPU (numba) vs array-code (GPU path) consistency.

There is no GPU in CI, so the CuPy path is exercised through ``fake_gpu``: a
backend whose array module is a numpy proxy that is *not* ``numpy`` itself.
That makes every dispatch take the branch it takes for CuPy (``method="xp"``,
``xp.asnumpy`` on the way out) while computing with numpy.
"""
import numpy as np
import pytest
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list

import ewald.neighbors as nbmod
from ewald import DataCollection, Pipeline, StaticSource
from ewald.backend import CPU, Backend
from ewald.data import Cell
from ewald.kernels import msd_fft, rdf, trajectory_msd, unwrap
from ewald.modifiers.analysis import CalculateDisplacements, CoordinationAnalysis
from ewald.neighbors import find_pairs


class _NumpyAsGPU:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture
def fake_gpu():
    return Backend("fake-cuda", _NumpyAsGPU())


def _pairs(out):
    i, j, d, dist = (np.asarray(a) for a in out)
    return i, j, d, dist


TRICLINIC = np.array([[14.0, 0, 0], [4.0, 13.0, 0], [-3.0, 2.5, 15.0]])
needs_numba = pytest.mark.skipif(not nbmod.HAVE_NUMBA, reason="numba not installed")


@needs_numba
@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (False, False, False)])
@pytest.mark.parametrize("half", [False, True])
def test_methods_agree(rng, pbc, half):
    cell = Cell(TRICLINIC, np.zeros(3), pbc)
    pos = rng.random((1200, 3)) @ TRICLINIC
    ref = _pairs(find_pairs(pos, cell, 3.1, method="brute", half=half))
    order = np.lexsort((ref[1], ref[0]))
    ref = tuple(a[order] for a in ref)
    for method in ("numba", "xp"):
        out = _pairs(find_pairs(pos, cell, 3.1, method=method, half=half))
        np.testing.assert_array_equal(out[0], ref[0])
        np.testing.assert_array_equal(out[1], ref[1])
        np.testing.assert_allclose(out[2], ref[2], atol=1e-12)
        np.testing.assert_allclose(out[3], ref[3], atol=1e-12)
    if half:
        assert np.all(ref[0] < ref[1])
    ai, aj = primitive_neighbor_list("ij", np.array(pbc), TRICLINIC, pos, 3.1)
    assert len(ai) == (len(ref[0]) * (2 if half else 1))


@needs_numba
def test_sparse_cluster_in_large_box(rng):
    # 300 atoms in a 200 A box: the bin grid must be capped, results unchanged.
    matrix = np.diag([200.0, 200.0, 200.0])
    cell = Cell(matrix, np.zeros(3), (True, True, True))
    pos = 100 + rng.random((300, 3)) * 12
    a = _pairs(find_pairs(pos, cell, 3.0, method="numba"))
    b = _pairs(find_pairs(pos, cell, 3.0, method="xp"))
    c = _pairs(find_pairs(pos, cell, 3.0, method="brute"))
    assert set(zip(a[0], a[1])) == set(zip(c[0], c[1])) == set(zip(b[0], b[1]))


def test_auto_dispatch(fake_gpu, cu_fcc, monkeypatch):
    calls = []
    for name in ("_cell_list_numba", "_cell_list_xp"):
        if hasattr(nbmod, name):
            orig = getattr(nbmod, name)
            monkeypatch.setattr(nbmod, name,
                                lambda *a, _o=orig, _n=name, **k: calls.append(_n) or _o(*a, **k))
    big = cu_fcc.copy()
    find_pairs(big.positions, big.cell, 3.0, backend=fake_gpu)
    assert calls[-1] == "_cell_list_xp"
    if nbmod.HAVE_NUMBA:
        find_pairs(big.positions, big.cell, 3.0)
        assert calls[-1] == "_cell_list_numba"
        with pytest.raises(ValueError, match="CPU backend"):
            find_pairs(big.positions, big.cell, 3.0, backend=fake_gpu, method="numba")


def _rdf_reference(pos, matrix, cutoff, nbins, types):
    """The step-1 formulas: full ordered pair list + np.histogram."""
    i, j, dist = primitive_neighbor_list("ijd", np.array([True] * 3), matrix, pos, cutoff)
    n, volume = len(pos), abs(np.linalg.det(matrix))
    edges = np.linspace(0, cutoff, nbins + 1)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    out = {"g(r)": np.histogram(dist, edges)[0] * volume / (n * (n - 1) * shell)}
    names = sorted(set(types))
    for a in range(len(names)):
        for b in range(a, len(names)):
            na, nb = (types == names[a]).sum(), (types == names[b]).sum()
            m = (types[i] == names[a]) & (types[j] == names[b])
            pairs = na * (nb - 1 if a == b else nb)
            out[f"g(r) {names[a]}-{names[b]}"] = np.histogram(dist[m], edges)[0] * volume / (pairs * shell)
    return out


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_rdf_matches_reference(rng, device, fake_gpu):
    matrix = np.diag([16.0, 17.0, 18.0])
    pos = rng.random((800, 3)) @ matrix
    types = np.where(rng.random(800) < 0.3, "O", "Si")
    be = fake_gpu if device == "fake" else CPU
    table = rdf(pos, Cell(matrix), 6.0, nbins=60, types=types, backend=be)
    ref = _rdf_reference(pos, matrix, 6.0, 60, types)
    assert set(ref) <= set(table)
    for key, val in ref.items():
        np.testing.assert_allclose(table[key], val, rtol=1e-12, atol=1e-12, err_msg=key)


def test_coordination_modifier_same_on_both_backends(fake_gpu):
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((6, 6, 6))
    atoms.rattle(0.05, seed=3)
    atoms.symbols[::5] = "Ni"
    pipe = Pipeline(StaticSource(atoms), [CoordinationAnalysis(cutoff=3.0, number_of_bins=50,
                                                               partial=True)])
    cpu = pipe.compute(0)
    gpu = pipe.compute(0, device=fake_gpu)
    assert np.all(cpu["Coordination"] == 12)
    np.testing.assert_array_equal(cpu["Coordination"], gpu["Coordination"])
    for key, val in cpu.tables["coordination-rdf"].items():
        np.testing.assert_allclose(gpu.tables["coordination-rdf"][key], val, err_msg=key)
    assert {"g(r) Cu-Cu", "g(r) Cu-Ni", "g(r) Ni-Ni"} <= set(cpu.tables["coordination-rdf"])


def _random_walk(rng, T=60, n=40, box=10.0, step=0.4):
    true = np.cumsum(rng.normal(scale=step, size=(T, n, 3)), axis=0) + rng.random((n, 3)) * box
    return true, np.diag([box] * 3)


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_unwrap_recovers_random_walk(rng, device, fake_gpu):
    true, matrix = _random_walk(rng)
    wrapped = true % 10.0
    be = fake_gpu if device == "fake" else CPU
    out = np.asarray(unwrap(wrapped, matrix, (True, True, True), be))
    # frame 0 is kept as stored (wrapped); displacements from it are exact
    np.testing.assert_allclose(out[0], wrapped[0])
    np.testing.assert_allclose(out - out[0], true - true[0], atol=1e-9)


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_msd_fft_matches_direct(rng, device, fake_gpu):
    true, _ = _random_walk(rng, T=50, n=30)
    direct = np.array([np.mean(np.sum((true[m:] - true[:len(true) - m]) ** 2, axis=-1))
                       for m in range(len(true))])
    be = fake_gpu if device == "fake" else CPU
    np.testing.assert_allclose(msd_fft(true, be, chunk_bytes=4000), direct, rtol=1e-10, atol=1e-10)


def test_trajectory_msd_from_pipeline(rng, fake_gpu):
    true, matrix = _random_walk(rng, T=30, n=20)
    frames = []
    perm = np.arange(20)
    for x in true:
        perm = rng.permutation(20)          # shuffle storage order every frame
        d = DataCollection((x % 10.0)[perm], np.array(["A"] * 10 + ["B"] * 10)[perm], Cell(matrix))
        d.particles["Particle Identifier"] = perm + 1
        frames.append(d)
    pipe = Pipeline(StaticSource(frames))
    table = trajectory_msd(pipe)
    np.testing.assert_allclose(table["MSD"], msd_fft(true), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(table["MSD A"], msd_fft(true[:, :10]), rtol=1e-10, atol=1e-10)
    gpu = trajectory_msd(pipe, device=fake_gpu, step=2)
    np.testing.assert_allclose(gpu["MSD"], msd_fft(true[::2]), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(gpu["lag"], np.arange(15) * 2)
    # the per-frame reference MSD of CalculateDisplacements is lag t from frame 0
    pipe.modifiers.append(CalculateDisplacements(reference_frame=0))
    last = pipe.compute(1, device=fake_gpu)
    np.testing.assert_allclose(last.attributes["MSD"], np.mean(np.sum((true[1] - true[0]) ** 2, -1)))
