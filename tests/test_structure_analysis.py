"""CNA, PTM, Voronoi and Wigner-Seitz results compared against OVITO 3.16.

Reference data come from tests/data/make_ovito_reference.py (needs `ovito`).
"""
import os

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, Pipeline, StaticSource
from ewald.backend import CPU, Backend
from ewald.data import Cell
from ewald.modifiers import (AcklandJones, AtomicStrain, CentroSymmetry, CommonNeighborAnalysis, IdentifyDiamond,
                             PolyhedralTemplateMatching, VoronoiAnalysis,
                             WignerSeitzAnalysis)
from ewald.modifiers._knn import nearest_neighbors
from ewald.pipeline import ComputeContext

REF = np.load(os.path.join(os.path.dirname(__file__), "data", "ovito_reference.npz"))
SYSTEMS = sorted({k.split("/")[0] for k in REF if k.split("/")[0] not in ("ws", "cluster", "strain")})
CTX = ComputeContext(0, None, CPU)

try:
    import ewald._native  # noqa: F401
    HAVE_NATIVE = True
except ImportError:
    HAVE_NATIVE = False
needs_native = pytest.mark.skipif(not HAVE_NATIVE, reason="ewald._native extension not built")


def system(name):
    cell = Cell(REF[f"{name}/cell"], np.zeros(3), REF[f"{name}/pbc"])
    return DataCollection(REF[f"{name}/pos"], REF[f"{name}/types"].astype(str), cell)


def run(mod, data):
    data = data.copy()
    out = mod.modify(data, CTX)
    return data if out is None else out


@pytest.mark.parametrize("name", SYSTEMS)
def test_cna_adaptive_matches_ovito(name):
    d = run(CommonNeighborAnalysis(), system(name))
    np.testing.assert_array_equal(d["Structure Type"], REF[f"{name}/cna/Structure Type"])
    assert d.attributes["CommonNeighborAnalysis.counts.FCC"] == np.count_nonzero(d["Structure Type"] == 1)


@pytest.mark.parametrize("name", SYSTEMS)
def test_cna_fixed_matches_ovito(name):
    d = run(CommonNeighborAnalysis(mode="fixed", cutoff=3.05), system(name))
    np.testing.assert_array_equal(d["Structure Type"], REF[f"{name}/cnafix/Structure Type"])


@needs_native
@pytest.mark.parametrize("name", SYSTEMS)
def test_cna_interval_matches_ovito(name):
    d = run(CommonNeighborAnalysis(mode="interval"), system(name))
    np.testing.assert_array_equal(d["Structure Type"], REF[f"{name}/cnaint/Structure Type"])


@pytest.mark.parametrize("name", SYSTEMS)
@pytest.mark.parametrize("mode,tag,kw", [("adaptive", "cna", {}), ("fixed", "cnafix", {"cutoff": 3.05})])
def test_cna_array_kernel_matches_ovito(name, mode, tag, kw):
    # The vectorized kernel used with device="cuda", run here on numpy arrays.
    gpu_like = ComputeContext(0, None, Backend("cuda", np))
    data = system(name)
    CommonNeighborAnalysis(mode=mode, chunk_size=1000, **kw).modify(data, gpu_like)
    np.testing.assert_array_equal(data["Structure Type"], REF[f"{name}/{tag}/Structure Type"])


@needs_native
@pytest.mark.parametrize("name", SYSTEMS)
def test_ptm_matches_ovito(name):
    mod = PolyhedralTemplateMatching(types=range(1, 9), output_orientation=True,
                                     output_interatomic_distance=True, output_deformation_gradient=True,
                                     output_ordering=True)
    d = run(mod, system(name))
    r = lambda p: REF[f"{name}/ptm/{p}"]  # noqa: E731
    np.testing.assert_array_equal(d["Structure Type"], r("Structure Type"))
    np.testing.assert_array_equal(d["Ordering Type"], r("Ordering Type"))
    np.testing.assert_array_equal(d["Correspondences"], r("Correspondences"))
    np.testing.assert_allclose(d["Interatomic Distance"], r("Interatomic Distance"), atol=1e-9)
    np.testing.assert_allclose(d["Orientation"], r("Orientation"), atol=1e-6)   # OVITO stores float32
    np.testing.assert_allclose(d["Elastic Deformation Gradient"].reshape(-1, 9),
                               r("Elastic Deformation Gradient").reshape(-1, 9), atol=1e-9)


@needs_native
@pytest.mark.parametrize("name", SYSTEMS)
def test_voronoi_matches_ovito(name):
    d = run(VoronoiAnalysis(compute_indices=True), system(name))
    r = lambda p: REF[f"{name}/vor/{p}"]  # noqa: E731
    np.testing.assert_allclose(d["Atomic Volume"], r("Atomic Volume"), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(d["Cavity Radius"], r("Cavity Radius"), atol=1e-9)
    np.testing.assert_array_equal(d["Coordination"], r("Coordination"))
    np.testing.assert_array_equal(d["Max Face Order"], r("Max Face Order"))
    np.testing.assert_array_equal(d["Voronoi Index"], r("Voronoi Index"))
    if d.cell.pbc.all():
        assert d.attributes["Voronoi.volume_sum"] == pytest.approx(d.cell.volume)


@pytest.mark.parametrize("tag,kw", [("site", {}), ("cur", {"output_current_config": True}),
                                    ("pertype", {"per_type_occupancies": True})])
def test_wigner_seitz_matches_ovito(tag, kw):
    cell = Cell(REF["ws/cell"])
    types = REF["ws/types"].astype(str)
    frames = [DataCollection(REF["ws/ref"], types, cell), DataCollection(REF["ws/cur"], types, cell)]
    d = Pipeline(StaticSource(frames), [WignerSeitzAnalysis(**kw)]).compute(1)
    np.testing.assert_array_equal(d["Occupancy"], REF[f"ws/{tag}/Occupancy"])
    assert [d.attributes["WignerSeitz.vacancy_count"], d.attributes["WignerSeitz.interstitial_count"]] == \
        list(REF[f"ws/{tag}/counts"])
    if tag == "cur":
        np.testing.assert_array_equal(d["Site Index"], REF["ws/cur/Site Index"])


def test_knn_small_periodic_cell():
    # 4-atom FCC cell: all 12 nearest neighbors are periodic images.
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True))
    idx, delta, dist = nearest_neighbors(d.positions, d.cell, 18)
    np.testing.assert_allclose(dist[:, :12], 3.6 / np.sqrt(2))
    np.testing.assert_allclose(dist[:, 12:18], 3.6)
    np.testing.assert_allclose(np.linalg.norm(delta, axis=-1), dist)


def test_cna_small_cell_and_selection():
    d = DataCollection.from_ase(bulk("Fe", "bcc", a=2.87, cubic=True))
    out = run(CommonNeighborAnalysis(), d)
    assert list(out["Structure Type"]) == [3, 3]
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3)))
    sel = np.zeros(d.count, bool)
    sel[:10] = True
    d.particles["Selection"] = sel
    out = run(CommonNeighborAnalysis(only_selected=True), d)
    assert np.all(out["Structure Type"][10:] == 0)


@needs_native
def test_ptm_and_voronoi_small_cell():
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True))
    out = run(PolyhedralTemplateMatching(), d)
    assert list(out["Structure Type"]) == [1] * 4
    out = run(VoronoiAnalysis(compute_indices=True), d)
    np.testing.assert_allclose(out["Atomic Volume"], 3.6 ** 3 / 4)
    assert np.all(out["Voronoi Index"][:, 3] == 12)    # rhombic dodecahedron: 12 four-edged faces


@needs_native
def test_voronoi_radical_and_faces():
    d = DataCollection.from_ase(bulk("CsCl", "cesiumchloride", a=4.1).repeat((3, 3, 3)))
    d.particles["Radius"] = np.where(d.types == "Cs", 1.8, 1.0)
    plain = run(VoronoiAnalysis(), d)
    radical = run(VoronoiAnalysis(use_radii=True, compute_bonds=True), d)
    big = d.types == "Cs"
    assert radical["Atomic Volume"][big].mean() > plain["Atomic Volume"][big].mean()
    assert radical.attributes["Voronoi.volume_sum"] == pytest.approx(d.cell.volume)
    faces = radical.tables["voronoi-faces"]
    assert len(faces["i"]) == radical["Coordination"].sum()


def test_cna_gpu_path_uses_backend_arrays():
    # The signature kernel is written against xp; running it with numpy covers the CuPy path.
    from ewald.modifiers.structure import _bond_matrix, _cna_signatures

    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)))
    _, delta, dist = nearest_neighbors(d.positions, d.cell, 12)
    rc = dist.mean(-1) * (1 + np.sqrt(2)) / 2
    ncommon, nbonds, chain = _cna_signatures(np, _bond_matrix(np, delta, rc * rc))
    assert np.all(ncommon == 4) and np.all(nbonds == 2) and np.all(chain == 1)



@pytest.mark.parametrize("name", SYSTEMS)
def test_ackland_jones_matches_ovito(name):
    d = run(AcklandJones(), system(name))
    np.testing.assert_array_equal(d["Structure Type"], REF[f"{name}/aj/Structure Type"])


@pytest.mark.parametrize("name", SYSTEMS)
def test_centrosymmetry_matches_ovito(name):
    d = run(CentroSymmetry(), system(name))
    np.testing.assert_allclose(d["Centrosymmetry"], REF[f"{name}/csp/Centrosymmetry"], rtol=1e-9, atol=1e-9)


@needs_native
@pytest.mark.parametrize("name", SYSTEMS)
def test_centrosymmetry_matching_matches_ovito(name):
    d = run(CentroSymmetry(num_neighbors=8, mode="matching"), system(name))
    np.testing.assert_allclose(d["Centrosymmetry"], REF[f"{name}/cspmwm/Centrosymmetry"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("tag,kw", [("plain", {}), ("toref", {"affine_mapping": "to_reference"}),
                                    ("small", {"cutoff": 2.2})])
def test_atomic_strain_matches_ovito(tag, kw):
    ref = system("fcc_hot")
    cur = DataCollection(REF["strain/cur"], ref.types, Cell(REF["strain/cell"]))
    mod = AtomicStrain(cutoff=kw.pop("cutoff", 3.0), output_deformation_gradients=True, output_strain_tensors=True,
                       output_nonaffine_squared_displacements=True, output_rotations=True,
                       output_stretch_tensors=True, **kw)
    d = Pipeline(StaticSource([ref, cur]), [mod]).compute(1)
    r = lambda p: REF[f"strain/{tag}/{p}"]  # noqa: E731
    for p in ("Shear Strain", "Volumetric Strain", "Strain Tensor", "Nonaffine Squared Displacement",
              "Stretch Tensor"):
        np.testing.assert_allclose(d[p], r(p), atol=1e-9, err_msg=p)
    # OVITO stores 3x3 matrices column by column.
    np.testing.assert_allclose(d["Deformation Gradient"].transpose(0, 2, 1).reshape(-1, 9),
                               r("Deformation Gradient"), atol=1e-9)
    q, qr = d["Rotation"], r("Rotation")
    sign = np.where((q * qr).sum(1) < 0, -1.0, 1.0)[:, None]
    np.testing.assert_allclose(q * sign, qr, atol=1e-7)
    np.testing.assert_array_equal(d["Selection"], r("Selection").astype(bool))
    assert d.attributes["AtomicStrain.invalid_particle_count"] == int(r("invalid"))


@pytest.mark.parametrize("name", SYSTEMS)
def test_identify_diamond_matches_ovito(name):
    d = run(IdentifyDiamond(), system(name))
    np.testing.assert_array_equal(d["Structure Type"], REF[f"{name}/diamondid/Structure Type"])
    assert d.attributes["IdentifyDiamond.counts.CUBIC_DIAMOND"] == np.count_nonzero(d["Structure Type"] == 1)
