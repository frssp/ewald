"""Elastic strain, grain segmentation and fcc planar faults against OVITO 3.16 results.

The reference arrays come from tests/data/make_crystal_reference.py (needs `pip install ovito`).
"""
import os

import numpy as np
import pytest

pytest.importorskip("ewald._native")

from ewald import DataCollection  # noqa: E402
from ewald.backend import CPU  # noqa: E402
from ewald.data import Cell  # noqa: E402
from ewald.modifiers import (ElasticStrain, GrainSegmentation, IdentifyFCCPlanarFaults,  # noqa: E402
                             PolyhedralTemplateMatching)
from ewald.pipeline import ComputeContext  # noqa: E402

REF = np.load(os.path.join(os.path.dirname(__file__), "data", "crystal_reference.npz"))
CTX = ComputeContext(0, None, CPU)


def system(name):
    pos = REF[f"{name}/pos"]
    cell = Cell(REF[f"{name}/cell"], np.zeros(3), np.array([True] * 3))
    return DataCollection(pos, np.array(["Cu"] * len(pos)), cell)


def apply(data, *mods):
    for m in mods:
        out = m.modify(data, CTX)
        data = data if out is None else out
    return data


def ptm():
    return PolyhedralTemplateMatching(types=["FCC", "HCP", "BCC", "ICO", "CUBIC_DIAMOND", "HEX_DIAMOND"],
                                      output_orientation=True, output_interatomic_distance=True)


LATTICES = {"fcc": ("FCC", 3.615, 1.0), "faults": ("FCC", 2.556 * np.sqrt(2), 1.0), "bcc": ("BCC", 2.87, 1.0),
            "hcp": ("HCP", 3.21, 5.21 / 3.21), "diamond": ("CUBIC_DIAMOND", 5.43, 1.0),
            "hexdiamond": ("HEX_DIAMOND", 3.84, 6.27 / 3.84), "poly_fcc": ("FCC", 3.615, 1.0)}


@pytest.mark.parametrize("name", list(LATTICES))
@pytest.mark.parametrize("tag", ["es", "esmat"])
def test_elastic_strain_matches_ovito(name, tag):
    if f"{name}/{tag}/Structure Type" not in REF:
        pytest.skip("no reference for this combination")
    lattice, a0, ca = LATTICES[name]
    d = apply(system(name), ElasticStrain(lattice, a0, ca, calculate_deformation_gradients=True,
                                          push_strain_tensors_forward=tag == "es"))
    p = f"{name}/{tag}/"
    np.testing.assert_array_equal(d["Structure Type"], REF[p + "Structure Type"])
    np.testing.assert_array_equal(d["Cluster"], REF[p + "Cluster"])
    np.testing.assert_allclose(d["Volumetric Strain"], REF[p + "Volumetric Strain"], atol=1e-9)
    if p + "Elastic Strain" in REF:
        np.testing.assert_allclose(d["Elastic Strain"], REF[p + "Elastic Strain"], atol=1e-9)
    if p + "Elastic Deformation Gradient" in REF:
        # OVITO stores the 3x3 components column by column (XX, YX, ZX, XY, ...).
        F = REF[p + "Elastic Deformation Gradient"].reshape(-1, 3, 3).transpose(0, 2, 1)
        np.testing.assert_allclose(d["Elastic Deformation Gradient"], F, atol=1e-9)


def test_elastic_strain_recovers_applied_strain():
    d = apply(system("fcc"), ElasticStrain("FCC", 3.615, push_strain_tensors_forward=False,
                                           calculate_deformation_gradients=True))
    crystal = d["Structure Type"] == 1
    assert crystal.mean() > 0.9
    F = np.array([[1.012, 0.004, -0.002], [0.0, 0.993, 0.003], [0.001, -0.002, 1.006]])
    # The cluster orientation is arbitrary up to a cubic symmetry, but the stretch is the
    # same: F_elastic^T F_elastic = F^T F up to that rotation, and det F matches.
    Fe = d["Elastic Deformation Gradient"][crystal].mean(0)
    assert abs(np.linalg.det(Fe) - np.linalg.det(F)) < 2e-3
    np.testing.assert_allclose(Fe.T @ Fe, F.T @ F, atol=3e-3)


def test_elastic_strain_options():
    d = apply(system("bcc"), ElasticStrain("BCC", 2.87, calculate_strain_tensors=False))
    assert "Elastic Strain" not in d and "Elastic Deformation Gradient" not in d
    assert d.attributes["ElasticStrain.counts.BCC"] == np.sum(REF["bcc/es/Structure Type"] == 3)
    with pytest.raises(ValueError):
        ElasticStrain("ICO")


VARIANTS = {"gs": {}, "gsmst": dict(algorithm="mst", merging_threshold=3.0),
            "gsman": dict(algorithm="manual", merging_threshold=8.5, min_grain_size=50, orphan_adoption=False),
            "gsnosf": dict(handle_stacking_faults=False)}
GS_SYSTEMS = ("poly_fcc", "poly_bcc", "poly_diamond", "faults", "hcp", "sphere_fcc")


@pytest.fixture(scope="module")
def ptm_systems():
    return {name: apply(system(name), ptm()) for name in GS_SYSTEMS}


@pytest.mark.parametrize("name", GS_SYSTEMS)
@pytest.mark.parametrize("tag", list(VARIANTS))
def test_grain_segmentation_matches_ovito(ptm_systems, name, tag):
    d = apply(ptm_systems[name].copy(), GrainSegmentation(**VARIANTS[tag]))
    p = f"{name}/{tag}/"
    np.testing.assert_array_equal(d["Grain"], REF[p + "Grain"])
    assert d.attributes["GrainSegmentation.grain_count"] == int(REF[p + "count"])
    if p + "threshold" in REF:
        assert d.attributes["GrainSegmentation.auto_merge_threshold"] == pytest.approx(float(REF[p + "threshold"]),
                                                                                      abs=1e-9)
    grains = d.tables["grains"]
    np.testing.assert_array_equal(grains["Grain Size"], REF[p + "grains/Grain Size"])
    if int(REF[p + "count"]):
        np.testing.assert_array_equal(grains["Grain Identifier"], REF[p + "grains/Grain Identifier"])
        np.testing.assert_array_equal(grains["Structure Type"], REF[p + "grains/Structure Type"])
        np.testing.assert_allclose(grains["Color"], REF[p + "grains/Color"], atol=1e-6)
        np.testing.assert_allclose(grains["Orientation"], REF[p + "grains/Orientation"], atol=1e-6)
        # The center of a grain that percolates through the periodic boundaries depends on the
        # order the bonds are visited in (in OVITO too), so compare compact grains only.
        compact = grains["Radius"] < 0.4 * np.min(d.cell.perpendicular_widths())
        if name == "sphere_fcc" or tag == "gsman":
            for col in ("Center of Mass", "Radius", "Radius of Gyration"):
                np.testing.assert_allclose(grains[col][compact], REF[p + "grains/" + col][compact], atol=1e-6)
    merge = REF.get(p + "grains-merge/Log merge distance", REF.get(p + "grains-merge/Misorientation (degrees)"))
    if merge is not None:
        col = "Misorientation (degrees)" if tag == "gsmst" else "Log merge distance"
        np.testing.assert_allclose(d.tables["grains-merge"][col], merge, atol=1e-9)
    if p + "grains-log/Log merge distance" in REF:
        np.testing.assert_allclose(d.tables["grains-log"]["Log merge distance"],
                                   REF[p + "grains-log/Log merge distance"], atol=1e-9)
        np.testing.assert_allclose(d.tables["grains-log"]["Log geometric merge size"],
                                   REF[p + "grains-log/Log geometric merge size"], atol=1e-9)


def test_grain_segmentation_colors_and_bonds(ptm_systems):
    d = apply(ptm_systems["sphere_fcc"].copy(), GrainSegmentation(output_bonds=True))
    colors = d.tables["grains"]["Color"]
    grain = d["Grain"]
    np.testing.assert_allclose(d["Color"][grain > 0], colors[grain[grain > 0] - 1])
    assert d.bonds is not None and len(d.bonds) > d.count
    dis = d.bonds["Disorientation"]
    assert np.all(np.isfinite(dis) | np.isinf(dis)) and np.nanmin(dis) >= 0
    # Bonds inside one grain are nearly aligned; some across the sphere surface are not.
    topo = d.bonds.topology
    same = (grain[topo[:, 0]] == grain[topo[:, 1]]) & (grain[topo[:, 0]] > 0)
    assert np.median(dis[same]) < 5


def test_grain_segmentation_needs_ptm_orientations():
    d = apply(system("poly_fcc"), PolyhedralTemplateMatching())
    with pytest.raises(ValueError, match="orientation"):
        GrainSegmentation().modify(d, CTX)
    with pytest.raises(ValueError, match="PTM"):
        GrainSegmentation().modify(system("poly_fcc"), CTX)
    with pytest.raises(ValueError):
        GrainSegmentation(algorithm="kmeans")


def test_planar_faults_match_ovito(ptm_systems):
    d = apply(ptm_systems["faults"].copy(), IdentifyFCCPlanarFaults())
    np.testing.assert_array_equal(d["Planar Fault Type"], REF["faults/pf/Planar Fault Type"])
    table = d.tables["planar_faults"]
    for key in ("Atom Count", "Atom Fraction", "Estimated Area"):
        np.testing.assert_allclose(table[key], REF["faults/pf/table/" + key], atol=1e-6)
    kind = d["Planar Fault Type"]
    np.testing.assert_allclose(d["Color"][kind == 2], np.tile([0.9, 0.7, 0.2], ((kind == 2).sum(), 1)))
    assert d.attributes["IdentifyFCCPlanarFaults.counts.intrinsic_stacking_fault"] == table["Atom Count"][2]


def test_planar_faults_without_hcp():
    d = apply(system("fcc"), ptm(), IdentifyFCCPlanarFaults())
    assert not d["Planar Fault Type"].any()
    assert d.tables["planar_faults"]["Atom Count"].sum() == 0
