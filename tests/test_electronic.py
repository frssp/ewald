"""DOS / PDOS / band structure from vasprun.xml and vaspout.h5 (ewald.electronic).

The fixtures in tests/data/vasp/electronic come from make_electronic.py: the same made-up
spin-polarized GaAs run written as vasprun.xml (+ KPOINTS) and as vaspout.h5. When pymatgen
or py4vasp is installed the files are read with them too and compared.
"""
import gzip
import os
import shutil

import numpy as np
import pytest

import ewald
from ewald.electronic import ElectronicStructure, line_ticks, orbital_mask, path_distances, pretty_label

DIR = os.path.join(os.path.dirname(__file__), "data", "vasp", "electronic")
XML = os.path.join(DIR, "vasprun.xml")
H5 = os.path.join(DIR, "vaspout.h5")


def electronic(path):
    return ewald.import_file(path).compute(0).attributes["Electronic Structure"]


def test_vasprun_dos_bands_and_tables():
    data = ewald.import_file(XML).compute(0)
    es = data.attributes["Electronic Structure"]
    assert es.num_spins == 2 and es.efermi == pytest.approx(0.5)
    assert es.dos.shape == (2, 41) and es.pdos.shape == (2, 2, 9, 41)
    assert es.eigenvalues.shape == (2, 18, 8) and es.projections.shape == (2, 18, 8, 2, 9)
    assert es.orbitals[:4] == ["s", "py", "pz", "px"]
    # G-X-W | K-G: 6 points per line, the jump from W to K adds no distance
    assert [lab for _, lab in es.ticks] == ["Γ", "X", "W|K", "Γ"]
    assert es.breaks == [12]
    assert es.distances[12] == es.distances[11]
    assert data.attributes["Fermi Energy"] == pytest.approx(0.5)
    gap = es.band_gap()
    assert data.attributes["Band Gap"] == pytest.approx(0.9)            # up VBM 0, down VBM 0.1, CBM 1.0
    assert gap["vbm"] == pytest.approx(0.1) and gap["cbm"] == pytest.approx(1.0) and not gap["direct"]
    dos = data.tables["DOS"]
    assert list(dos)[:5] == ["Energy", "Total (up)", "Total (down)", "Ga (up)", "Ga (down)"]
    assert "As p (down)" in dos and "Ga d (up)" in dos
    np.testing.assert_allclose(dos["Energy"], es.dos_energies - 0.5)
    np.testing.assert_allclose(dos["Total (down)"], -es.dos[1])          # down channel below the axis
    np.testing.assert_allclose(dos["Ga (up)"], es.pdos[0, 0].sum(axis=0))
    bands = data.tables["Bands"]
    assert list(bands)[:3] == ["Distance", "Band 1 (up)", "Band 1 (down)"] and len(bands) == 17
    np.testing.assert_allclose(bands["Band 4 (up)"], es.eigenvalues[0, :, 3] - 0.5)


def test_every_frame_and_plain_runs():
    pipe = ewald.import_file(XML)
    assert all("DOS" in pipe.compute(i).tables for i in range(pipe.num_frames))
    plain = ewald.import_file(os.path.join(DIR, "..", "vasprun.xml")).compute(0)  # no <dos>
    assert "Electronic Structure" not in plain.attributes and "DOS" not in plain.tables


def test_projections_and_orbitals():
    es = electronic(XML)
    np.testing.assert_allclose(es.pdos_of(), es.pdos.sum(axis=(0, 2)))
    np.testing.assert_allclose(es.pdos_of([1], "p"), es.pdos[1][:, 1:4].sum(axis=1))
    np.testing.assert_allclose(es.pdos_of(np.array([True, False]), ["s", "dxy"]), es.pdos[0][:, [0, 4]].sum(axis=1))
    np.testing.assert_allclose(es.band_weights(), 1 / 1.2)              # the fixture's projections sum to 1/1.2
    np.testing.assert_allclose(es.band_weights([0], "d"), es.projections[:, :, :, 0, 4:].sum(axis=-1))
    with pytest.raises(KeyError, match="no orbital 'g'"):
        es.pdos_of(None, "g")
    assert orbital_mask(["s", "p", "d"], "p").tolist() == [False, True, False]  # LORBIT = 10


def test_vaspout_matches_vasprun():
    a, b = electronic(XML), electronic(H5)
    for name in ("dos_energies", "dos", "pdos", "eigenvalues", "occupations", "distances", "kpoints", "projections"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), atol=1e-7, err_msg=name)  # XML: 8 decimals
    assert a.orbitals == b.orbitals and a.breaks == b.breaks and a.efermi == b.efermi
    assert [lab for _, lab in a.ticks] == [lab for _, lab in b.ticks]
    np.testing.assert_allclose([d for d, _ in a.ticks], [d for d, _ in b.ticks])


def test_vasprun_matches_pymatgen():
    pytest.importorskip("pymatgen.io.vasp")
    from pymatgen.electronic_structure.core import OrbitalType, Spin
    from pymatgen.io.vasp import Vasprun

    with pytest.warns(UserWarning):  # no POTCAR next to the fixture
        v = Vasprun(XML, parse_projected_eigen=True)
    es = electronic(XML)
    assert v.efermi == pytest.approx(es.efermi)
    gap, cbm, vbm, direct = v.eigenvalue_band_properties
    assert es.band_gap() == {"gap": pytest.approx(gap), "vbm": pytest.approx(vbm), "cbm": pytest.approx(cbm),
                             "direct": direct}
    cd = v.complete_dos
    np.testing.assert_allclose(cd.energies, es.dos_energies)
    for s, spin in enumerate((Spin.up, Spin.down)):
        np.testing.assert_allclose(cd.densities[spin], es.dos[s])
        np.testing.assert_allclose(cd.get_element_spd_dos("As")[OrbitalType.p].densities[spin], es.pdos_of([1], "p")[s])
        np.testing.assert_allclose(v.projected_eigenvalues[spin], es.projections[s])
    bs = v.get_band_structure(kpoints_filename=os.path.join(DIR, "KPOINTS"), line_mode=True)
    np.testing.assert_allclose(np.asarray(bs.distance) / (2 * np.pi), es.distances, atol=1e-12)  # 2 pi in pymatgen
    assert bs.get_band_gap()["energy"] == pytest.approx(es.band_gap()["gap"])


def test_vaspout_matches_py4vasp():
    py4vasp = pytest.importorskip("py4vasp")
    calc = py4vasp.Calculation.from_path(DIR)
    es = electronic(H5)
    dos = calc.dos.read("Ga(d)")
    np.testing.assert_allclose(dos["energies"], es.dos_energies - es.efermi)
    np.testing.assert_allclose(dos["down"], es.dos[1])
    np.testing.assert_allclose(dos["Ga_d_up"], es.pdos_of([0], "d")[0])
    band = calc.band.read("As(p)")
    np.testing.assert_allclose(band["kpoint_distances"], es.distances)
    np.testing.assert_allclose(band["bands_down"], es.eigenvalues[1] - es.efermi)
    np.testing.assert_allclose(band["As_p_up"], es.band_weights([1], "p")[0])


def test_vasprun_gz_without_kpoints_file(tmp_path):
    with open(XML, "rb") as src, gzip.open(tmp_path / "vasprun.xml.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    es = electronic(tmp_path / "vasprun.xml.gz")
    np.testing.assert_allclose(es.dos, electronic(XML).dos)
    # no KPOINTS: the line ends are labelled by their reduced coordinates
    assert [lab for _, lab in es.ticks] == ["Γ", "(1/2 0 1/2)", "(1/2 1/4 3/4)|(3/8 3/8 3/4)", "Γ"]


def test_hybrid_band_run_uses_zero_weight_kpoints(tmp_path):
    text = open(XML).read()
    head, rest = text.split('<varray name="weights" >', 1)
    body, tail = rest.split("</varray>", 1)
    rows = [r for r in body.split("\n") if "<v>" in r]
    weights = ["   <v>       0.25000000 </v>"] * 4 + ["   <v>       0.00000000 </v>"] * (len(rows) - 4)
    (tmp_path / "vasprun.xml").write_text(head + '<varray name="weights" >\n' + "\n".join(weights) + "\n  </varray>" + tail)
    es = electronic(tmp_path / "vasprun.xml")
    assert es.eigenvalues.shape == (2, 14, 8) and es.projections.shape == (2, 14, 8, 2, 9)
    assert (es.weights == 0).all() and es.ticks == [] and es.dos is not None


def test_path_helpers():
    lattice = np.eye(3) * 2.0
    k = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0.5, 0.5]])
    d, breaks = path_distances(k, lattice, segment=2)
    np.testing.assert_allclose(d, [0, 0.25, 0.25, 0.5])
    assert breaks == [2]
    assert line_ticks(d, 2, ["G", "X", "Y", "T"]) == [(0.0, "Γ"), (0.25, "X|Y"), (0.5, "T")]
    assert pretty_label("\\Gamma") == "Γ" and pretty_label("GAMMA") == "Γ" and pretty_label("S_1") == "S₁"
    assert pretty_label("$\\mathrm{X}$") == "X"


def test_band_gap_metal_and_missing():
    es = ElectronicStructure(types=np.array(["Cu"]), efermi=0.0, eigenvalues=np.array([[[-1.0, 0.2], [-0.2, 1.0]]]),
                             occupations=np.array([[[1.0, 0.0], [1.0, 1.0]]]), kpoints=np.zeros((2, 3)))
    assert es.band_gap()["gap"] == 0.0
    assert ElectronicStructure(types=np.array(["Cu"])).band_gap() is None
