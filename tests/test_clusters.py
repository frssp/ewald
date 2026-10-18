from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc

from ewald import DataCollection, Pipeline, StaticSource
from ewald.backend import CPU, Backend
from ewald.data import Cell
from ewald.kernels import cluster_properties, connected_components
from ewald.modifiers import ClusterAnalysis


class _NumpyAsGPU:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture
def fake_gpu():
    return Backend("fake-cuda", _NumpyAsGPU())


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("seed", range(4))
def test_components_xp_matches_scipy(seed, fake_gpu):
    rng = np.random.default_rng(seed)
    n = 500
    e = rng.integers(0, n, size=(rng.integers(50, 600), 2))
    e = e[e[:, 0] != e[:, 1]]
    ref_count, ref = scipy_cc(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)),
                              directed=False)
    for be, method in ((CPU, "scipy"), (fake_gpu, "xp"), (CPU, "xp")):
        lab, count = connected_components(n, e[:, 0], e[:, 1], be, method=method)
        lab = np.asarray(lab)
        assert count == ref_count
        assert _same_partition(lab, ref)
        # numbering: by lowest member index
        firsts = [np.nonzero(lab == c)[0][0] for c in range(count)]
        assert firsts == sorted(firsts)


def test_components_long_chain(fake_gpu):
    n = 2000
    perm = np.random.default_rng(1).permutation(n)        # a path visiting nodes in random order
    lab, count = connected_components(n, perm[:-1], perm[1:], fake_gpu)
    assert count == 1 and np.all(np.asarray(lab) == 0)


def _blobs():
    """Three blobs, one straddling the periodic boundary, plus isolated atoms."""
    centers = np.array([[5.0, 5.0, 5.0], [20.0, 20.0, 20.0], [39.5, 10.0, 0.5]])
    sizes = [30, 60, 45]
    pts, truth = [], []
    for k, (c, s) in enumerate(zip(centers, sizes)):
        g = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)][:s]) * 1.0
        pts.append(g - g.mean(0) + c)
        truth.append(np.full(s, k))
    loners = np.array([[30.0, 3.0, 33.0], [12.0, 33.0, 15.0]])
    pos = np.concatenate(pts + [loners]) % 40.0
    return pos, np.concatenate(truth + [np.array([3, 4])]), centers


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_cluster_modifier_wrapped_blob(device, fake_gpu):
    pos, truth, centers = _blobs()
    data = DataCollection(pos, np.array(["Ar"] * len(pos)), Cell(np.diag([40.0] * 3)))
    mod = ClusterAnalysis(cutoff=1.1, sort_by_size=True, compute_com=True, compute_gyration=True,
                          unwrap_particles=True)
    out = Pipeline(StaticSource(data), [mod]).compute(0, device=fake_gpu if device == "fake" else None)
    assert out.attributes["ClusterAnalysis.cluster_count"] == 5
    assert out.attributes["ClusterAnalysis.largest_size"] == 60
    assert _same_partition(out["Cluster"], truth)
    t = out.tables["clusters"]
    np.testing.assert_array_equal(t["Cluster Size"], [60, 45, 30, 1, 1])
    # the 45-atom blob wraps across x and z; unwrapped from its lowest-index atom
    blob = out.positions[truth == 2]
    assert np.ptp(blob, axis=0).max() < 4
    np.testing.assert_allclose(t["Center of Mass"][1], blob.mean(0), atol=1e-9)
    np.testing.assert_allclose(t["Center of Mass"][1] % 40.0, centers[2] % 40.0, atol=1e-9)
    g = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)][:45]) * 1.0
    d = g - g.mean(0)
    np.testing.assert_allclose(t["Radius of Gyration"][1], np.sqrt((d ** 2).sum(1).mean()), atol=1e-9)
    np.testing.assert_allclose(t["Gyration Tensor"][1, 3], (d[:, 0] * d[:, 1]).mean(), atol=1e-9)
    assert t["Radius of Gyration"][3] == 0.0


def test_cluster_defaults_follow_ovito():
    pos, truth, _ = _blobs()
    data = DataCollection(pos, np.array(["Ar"] * len(pos)), Cell(np.diag([40.0] * 3)))
    data.particles["Selection"] = (truth != 1).astype(np.int8)  # drop the 60-atom blob
    out = Pipeline(StaticSource(data), [ClusterAnalysis(1.1, only_selected=True)]).compute(0)
    assert np.all(out["Cluster"][truth == 1] == 0)
    assert out.attributes["ClusterAnalysis.cluster_count"] == 4
    assert "ClusterAnalysis.largest_size" not in out.attributes      # only set when sorting
    np.testing.assert_array_equal(out.tables["clusters"]["Cluster Size"], [30, 45, 1, 1])
    np.testing.assert_array_equal(out.positions, pos)                 # not unwrapped by default


def test_cluster_properties_mass_weighted(fake_gpu):
    pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [10.0, 0, 0]])
    lab = np.array([0, 0, 1])
    for be in (CPU, fake_gpu):
        size, com, gyr, rg = cluster_properties(pos, lab, 2, masses=[1.0, 2.0, 5.0], backend=be)
        np.testing.assert_array_equal(size, [2, 1])
        np.testing.assert_allclose(com[0], [2.0, 0, 0])
        np.testing.assert_allclose(rg[0], np.sqrt((1 * 4 + 2 * 1) / 3))
        np.testing.assert_allclose(gyr[0, 0], rg[0] ** 2)


# ---- against OVITO 3.16 (reference generated with the ovito Python module)

REF = np.load(Path(__file__).parent / "data" / "ovito_cluster_reference.npz")


def _ovito_system():
    d = DataCollection(REF["cluster/pos"], None, Cell(REF["cluster/cell"]))
    d.particles["Mass"] = REF["cluster/mass"]
    return d


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_cluster_ids_match_ovito(device, fake_gpu):
    d = Pipeline(StaticSource(_ovito_system()), [ClusterAnalysis(cutoff=2.0)]).compute(
        0, device=fake_gpu if device == "fake" else None)
    np.testing.assert_array_equal(d["Cluster"], REF["cluster/plain/Cluster"])
    np.testing.assert_array_equal(d.tables["clusters"]["Cluster Size"], REF["cluster/plain/table/Cluster Size"])
    assert d.attributes["ClusterAnalysis.cluster_count"] == int(REF["cluster/plain/count"])


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_cluster_sorted_com_gyration_match_ovito(device, fake_gpu):
    mod = ClusterAnalysis(cutoff=2.0, sort_by_size=True, compute_com=True, compute_gyration=True,
                          unwrap_particles=True)
    d = Pipeline(StaticSource(_ovito_system()), [mod]).compute(
        0, device=fake_gpu if device == "fake" else None)
    ours, theirs = d["Cluster"], REF["cluster/sorted/Cluster"]
    # Equal-size clusters may be ordered differently (OVITO uses an unstable sort): match via members.
    mapping = {}
    for a, b in zip(ours, theirs):
        assert mapping.setdefault(a, b) == b
    assert len(set(mapping.values())) == len(mapping)
    rows_ours = np.array(sorted(mapping)) - 1
    rows_theirs = np.array([mapping[k] for k in sorted(mapping)]) - 1
    t = d.tables["clusters"]
    for key in ("Cluster Size", "Center of Mass", "Radius of Gyration", "Gyration Tensor"):
        np.testing.assert_allclose(t[key][rows_ours], REF[f"cluster/sorted/table/{key}"][rows_theirs],
                                   atol=1e-9, err_msg=key)
    np.testing.assert_allclose(d.positions, REF["cluster/sorted/Position"], atol=1e-9)
    assert d.attributes["ClusterAnalysis.largest_size"] == t["Cluster Size"][0]
