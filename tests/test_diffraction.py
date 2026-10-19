"""Symmetry (spglib), powder diffraction (vs pymatgen) and S(q) (vs freud).

References in tests/data/diffraction_reference.npz come from make_diffraction_reference.py."""
import os

import numpy as np
import pytest
from ase.build import bulk

from ewald import diffraction as dif
from ewald import modifiers as mods
from ewald.backend import CPU, Backend
from ewald.data import Cell, DataCollection
from ewald.pipeline import ComputeContext

REF = np.load(os.path.join(os.path.dirname(__file__), "data", "diffraction_reference.npz"))
CASES = {"cu_xrd": ("cu", "xray", "CuKa", None), "cu_dw": ("cu", "xray", "MoKa", {"Cu": 0.55}),
         "nacl_xrd": ("nacl", "xray", "CuKa", None), "nacl_nd": ("nacl", "neutron", 1.54, None),
         "zno_xrd": ("zno", "xray", "CuKa", None), "zno_nd": ("zno", "neutron", 1.8, None),
         "tri_xrd": ("tri", "xray", "CuKa", None)}


def crystal(name):
    return DataCollection(REF[f"{name}.positions"], REF[f"{name}.types"], Cell(REF[f"{name}.cell"]))


CTX = ComputeContext(0, None, CPU)
FAKE_GPU = Backend("cuda", np)   # the CuPy code path, run with numpy


def run(mod, data, ctx=CTX):
    data = data.copy()
    out = mod.modify(data, ctx)
    return data if out is None else out


@pytest.mark.parametrize("case", sorted(CASES))
def test_powder_peaks_match_pymatgen(case):
    name, radiation, lam, dw = CASES[case]
    d = crystal(name)
    p = dif.powder_peaks(d.positions, d.types, d.cell, lam, radiation, (5, 120), dw)
    np.testing.assert_allclose(p["2theta"], REF[f"{case}.2theta"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(p["Intensity"], REF[f"{case}.intensity"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(p["d"], REF[f"{case}.d"], rtol=1e-12)


def test_powder_peaks_hkl_and_multiplicity():
    d = crystal("cu")
    p = dif.powder_peaks(d.positions, d.types, d.cell, "CuKa")
    first = (p["h"][0], p["k"][0], p["l"][0])
    assert first == (1, 1, 1) and p["Multiplicity"][0] == 8
    assert (p["h"][1], p["k"][1], p["l"][1]) == (2, 0, 0) and p["Multiplicity"][1] == 6
    assert p["2theta"][0] == pytest.approx(43.3, abs=0.1)
    # a 2x2x2 supercell has the same pattern (extra reflections are systematically absent)
    sc = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2))
    q = dif.powder_peaks(sc.positions, sc.types, sc.cell, "CuKa")
    np.testing.assert_allclose(q["2theta"], p["2theta"], atol=1e-9)
    np.testing.assert_allclose(q["Intensity"], p["Intensity"], atol=1e-6)


def test_powder_diffraction_modifier_and_experiment(tmp_path):
    d = crystal("nacl")
    exp = dif.powder_profile(dif.powder_peaks(d.positions, d.types, d.cell, "CuKa"),
                             np.arange(10, 80, 0.01), fwhm=0.15)
    path = tmp_path / "nacl.xy"
    path.write_text("# 2theta intensity\n" + "\n".join(f"{x:.3f} {3 * y + 0.01:.5f}" for x, y in
                                                         zip(exp["2theta"], exp["Intensity"])))
    out = run(mods.PowderDiffraction(two_theta_max=80, fwhm=0.15, experimental=str(path)), d)
    t = out.tables["diffraction"]
    assert set(t) == {"2theta", "Intensity", "Experimental"} and t["Intensity"].max() == pytest.approx(100)
    assert out.attributes["PowderDiffraction.Rwp"] < 0.05
    cu = run(mods.PowderDiffraction(two_theta_max=80, fwhm=0.15, experimental=str(path)), crystal("cu"))
    assert cu.attributes["PowderDiffraction.Rwp"] > 0.5   # the wrong phase
    assert np.isnan(t["Experimental"][t["2theta"] < 10]).all()
    peaks = out.tables["diffraction-peaks"]
    assert out.attributes["PowderDiffraction.num_peaks"] == len(peaks["2theta"])
    strongest = np.argmax(peaks["Intensity"])
    assert peaks["d"][strongest] == pytest.approx(5.64 / 2)   # (200); hkl are in the primitive cell's basis
    # neutron and numbered types with an element map
    num = DataCollection(d.positions, np.where(d.types == "Na", "1", "2"), d.cell)
    nd = run(mods.PowderDiffraction(radiation="neutron", wavelength="1.54", elements="1=Na, 2=Cl"), num)
    ref = dif.powder_peaks(d.positions, d.types, d.cell, 1.54, "neutron", (5, 90))
    np.testing.assert_allclose(nd.tables["diffraction-peaks"]["Intensity"], ref["Intensity"])
    with pytest.raises(ValueError, match="not an element"):
        run(mods.PowderDiffraction(), num)


def test_wavelengths_and_profile():
    assert dif.wavelength_of("CuKa1") == 1.54056 and dif.wavelength_of("mo-kalpha") == 0.71073
    assert dif.wavelength_of("1.2") == 1.2
    with pytest.raises(ValueError):
        dif.wavelength_of("XYZ")
    prof = dif.powder_profile({"2theta": np.array([30.0]), "Intensity": np.array([1.0])},
                              np.arange(20, 40, 0.001), fwhm=0.2, eta=0.0, scaled=False)
    assert np.trapezoid(prof["Intensity"], prof["2theta"]) == pytest.approx(1.0, rel=1e-6)
    half = prof["2theta"][prof["Intensity"] >= prof["Intensity"].max() / 2]
    assert half[-1] - half[0] == pytest.approx(0.2, abs=2e-3)


def test_structure_factor_direct_matches_freud_definition():
    pos, L = REF["liquid.positions"], REF["liquid.cell"]
    # freud samples a subset of the reciprocal lattice: its S(k) is the binned mean of
    # |sum exp(ik.r)|^2 / N over those points, ours the same over all of them
    kp = REF["direct.k_points"]
    k = np.linalg.norm(kp, axis=1)
    s = np.abs(np.exp(1j * pos @ kp.T).sum(0)) ** 2 / len(pos)
    edges = np.linspace(0.5, 8.0, 61)
    b = np.searchsorted(edges, k, side="right") - 1
    ok = (b >= 0) & (b < 60)
    freud_like = np.bincount(b[ok], s[ok], 60) / np.maximum(np.bincount(b[ok], minlength=60), 1)
    np.testing.assert_allclose(freud_like, REF["direct.S"], atol=1e-4)
    # all lattice points, brute force
    ours = dif.structure_factor(pos, Cell(L), q_max=8.0, q_min=0.5, number_of_bins=60, method="direct")
    np.testing.assert_allclose(ours["q"], REF["direct.k"], atol=1e-6)
    n = np.arange(-3, 4)
    g = np.stack(np.meshgrid(n, n, n, indexing="ij"), -1).reshape(-1, 3) * 2 * np.pi / L[0, 0]
    gk = np.linalg.norm(g, axis=1)
    sel = (gk > 0) & (gk <= 8.0) & (gk >= 0.5)
    sb = np.abs(np.exp(1j * pos @ g[sel].T).sum(0)) ** 2 / len(pos)
    bb = np.searchsorted(edges, gk[sel], side="right") - 1
    brute = np.bincount(bb, sb, 60) / np.maximum(np.bincount(bb, minlength=60), 1)
    low = ours["q"] < 3 * 2 * np.pi / L[0, 0] - 0.2   # bins fully inside the brute-force box
    np.testing.assert_allclose(ours["S(q)"][low & ~np.isnan(ours["S(q)"])],
                               brute[low & ~np.isnan(ours["S(q)"])], rtol=1e-10)
    assert np.isnan(ours["S(q)"][0])   # below 2 pi / L: no lattice vector


def test_structure_factor_debye_matches_freud():
    k = REF["debye.k"]
    h = k[1] - k[0]
    s = dif.structure_factor(REF["cluster.positions"], Cell(np.zeros((3, 3)), pbc=False), q_max=k[-1] + h / 2,
                             q_min=k[0] - h / 2, number_of_bins=len(k))   # auto: debye without a cell
    np.testing.assert_allclose(s["q"], k, atol=1e-6)
    np.testing.assert_allclose(s["S(q)"], REF["debye.S"], atol=5e-4)


def test_structure_factor_methods_agree_and_weighting():
    rng = np.random.default_rng(3)
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat(5)
    atoms.positions += rng.normal(0, 0.12, atoms.positions.shape)
    d = DataCollection.from_ase(atoms)
    direct = dif.structure_factor(d.positions, d.cell, 6.0, 30, q_min=2.0, method="direct")
    debye_pbc = dif.structure_factor(d.positions, d.cell, 6.0, 30, q_min=2.0, method="debye")
    rdf = dif.structure_factor(d.positions, d.cell, 6.0, 30, q_min=2.0, method="rdf", lorch=False)
    big = np.argmax(direct["S(q)"])   # the (111) peak at q = 2 pi sqrt(3) / a
    assert direct["q"][big] == pytest.approx(2 * np.pi * np.sqrt(3) / 3.615, abs=0.1)
    assert abs(np.argmax(rdf["S(q)"]) - big) <= 1   # g(r) cut at a third of the cell: broader
    assert np.isfinite(debye_pbc["S(q)"]).all()
    # one element: X-ray / neutron Faber-Ziman weighting changes nothing
    for w in ("xray", "neutron"):
        np.testing.assert_allclose(dif.structure_factor(d.positions, d.cell, 6.0, 30, 2.0, "direct", d.types, w)["S(q)"],
                                   direct["S(q)"], rtol=1e-9)
    # two elements, neutron weighting: direct and rdf both tend to 1
    nacl = DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(3))
    for method in ("direct", "rdf"):
        s = dif.structure_factor(nacl.positions, nacl.cell, 30.0, 40, 25.0, method, nacl.types, "neutron")
        assert np.nanmean(s["S(q)"]) == pytest.approx(1.0, abs=0.15)
    out = run(mods.StructureFactor(q_max=8.0, weighting="xray"), nacl)
    assert set(out.tables["structure-factor"]) == {"q", "S(q)"}
    with pytest.raises(ValueError, match="needs particle types"):
        dif.structure_factor(nacl.positions, nacl.cell, weighting="xray")


def test_cuda_code_path_with_fake_backend():
    d = crystal("tri")
    p = dif.powder_peaks(d.positions, d.types, d.cell, "CuKa", two_theta_range=(5, 120), backend=FAKE_GPU)
    np.testing.assert_allclose(p["Intensity"], REF["tri_xrd.intensity"], rtol=1e-9, atol=1e-9)
    pos, L = REF["liquid.positions"], Cell(REF["liquid.cell"])
    for method in ("direct", "rdf", "debye"):
        s1 = dif.structure_factor(pos, L, 8, 20, 1, method, backend=FAKE_GPU)
        s2 = dif.structure_factor(pos, L, 8, 20, 1, method)
        np.testing.assert_allclose(s1["S(q)"], s2["S(q)"], rtol=1e-10)


def test_symmetry_analysis():
    pytest.importorskip("spglib")
    out = run(mods.SymmetryAnalysis(select_unique=True),
              DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True)))
    a = out.attributes
    assert a["SymmetryAnalysis.space_group"] == "Fm-3m" and a["SymmetryAnalysis.space_group_number"] == 225
    assert a["SymmetryAnalysis.crystal_system"] == "cubic" and a["SymmetryAnalysis.point_group"] == "m-3m"
    assert a["SymmetryAnalysis.num_operations"] == 192 and a["SymmetryAnalysis.num_unique_sites"] == 2
    assert sorted(set(out["Wyckoff Position"])) == ["4a", "4b"]
    assert a["SymmetryAnalysis.wyckoff_sites"] in ("Na 4a, Cl 4b", "Na 4b, Cl 4a")
    assert set(out["Site Symmetry"]) == {"m-3m"} and out["Selection"].sum() == 2
    # primitive cell: same Wyckoff labels (multiplicity of the conventional cell)
    prim = run(mods.SymmetryAnalysis(), DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    assert sorted(prim["Wyckoff Position"]) == sorted(set(out["Wyckoff Position"]))
    zno = run(mods.SymmetryAnalysis(), crystal("zno"))
    assert zno.attributes["SymmetryAnalysis.space_group"] == "P6_3mc"
    assert set(zno["Wyckoff Position"]) == {"2b"} and zno.attributes["SymmetryAnalysis.crystal_system"] == "hexagonal"
    # a 2x2x2 supercell keeps the space group; noise below symprec too, above it not
    rng = np.random.default_rng(0)
    big = bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2)
    big.positions += rng.normal(0, 1e-3, big.positions.shape)
    d = DataCollection.from_ase(big)
    assert run(mods.SymmetryAnalysis(), d).attributes["SymmetryAnalysis.space_group"] == "Fm-3m"
    assert run(mods.SymmetryAnalysis(symprec=1e-5), d).attributes["SymmetryAnalysis.space_group_number"] < 225
    tri = run(mods.SymmetryAnalysis(), crystal("tri"))
    assert tri.attributes["SymmetryAnalysis.space_group"] == "P1"
    assert list(tri["Equivalent Atom"]) == list(range(tri.count))
