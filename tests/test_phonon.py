"""Phonon modes: phonopy yaml and VASP OUTCAR readers, and the PhononMode animation, checked
against phonopy's own modulations (fixtures from tests/data/phonon/make_phonon_reference.py)."""
from pathlib import Path

import numpy as np
import pytest

from ewald import import_file
from ewald.modifiers import PhononMode
from ewald.phonon import read_modes, read_outcar_modes, read_phonopy_yaml

DATA = Path(__file__).parent / "data" / "phonon"


@pytest.fixture(scope="module")
def band():
    return read_phonopy_yaml(DATA / "band.yaml")


def _match(a, b, cell):
    """Index into ``b`` of the atom at each position of ``a`` (minimum image)."""
    inv = np.linalg.inv(cell)
    d = (a[:, None] - b[None]) @ inv
    d -= np.round(d)
    return np.argmin(np.linalg.norm(d @ cell, axis=2), axis=1)


def test_band_yaml_cell_path_and_eigenvectors(band):
    assert band.num_qpoints == 20 and band.num_bands == 12
    assert list(band.types) == ["Au", "Cu", "Cu", "Cu"]
    np.testing.assert_allclose(band.cell.matrix, np.eye(3) * 3.75)
    assert [lab for _, lab in band.ticks] == ["$\\Gamma$", "X", "M", "$\\Gamma$", "R"]
    np.testing.assert_allclose(band.qpoints[4], [0.5, 0, 0])
    e = band.eigenvectors.reshape(20, 12, 12)
    np.testing.assert_allclose(np.einsum("qbi,qci->qbc", e.conj(), e), np.broadcast_to(np.eye(12), (20, 12, 12)),
                               atol=1e-10)  # orthonormal modes at every q
    assert np.all(np.diff(band.frequencies, axis=1) > -1e-6)  # bands sorted by frequency
    t = band.table()
    assert list(t)[:2] == ["Distance", "Band 1"] and len(t) == 13


@pytest.mark.parametrize("k", range(5))
def test_displacements_match_phonopy_modulation(band, k):
    ref = np.load(DATA / "modulation_reference.npz")
    q, b, dim = ref[f"q{k}"], int(ref[f"band{k}"]), tuple(ref[f"dim{k}"])
    qi = int(np.argmin(np.linalg.norm(band.qpoints - q, axis=1)))
    np.testing.assert_allclose(band.qpoints[qi], q, atol=1e-6)
    data, u = band.displacements(qi, b, dim, amplitude=0.25)
    assert data.count == len(ref[f"positions{k}"])
    idx = _match(ref[f"positions{k}"], data.positions, data.cell.matrix)
    assert len(set(idx)) == data.count and list(data.types[idx]) == list(ref[f"symbols{k}"])
    ours, theirs = u.real[idx], ref[f"u{k}"].real
    assert np.linalg.norm(np.abs(u), axis=1).max() == pytest.approx(0.25)
    ours, theirs = ours / np.abs(ours).max(), theirs / np.abs(theirs).max()
    # the global sign follows the largest component; equal ones (by symmetry) may pick either sign
    sign = np.sign(np.sum(ours * theirs))
    np.testing.assert_allclose(sign * ours, theirs, atol=1e-6)


def test_commensurate_supercell(band):
    assert band.commensurate_supercell(0) == (1, 1, 1)
    assert band.commensurate_supercell(4) == (2, 1, 1)          # X
    assert band.commensurate_supercell(1) == (8, 1, 1)          # q = (1/8, 0, 0)
    assert band.commensurate_supercell(1, max_size=4) == (4, 1, 1)


def test_outcar_modes_match_phonopy_gamma():
    vasp = read_outcar_modes(DATA / "OUTCAR.phonon")
    ph = read_phonopy_yaml(DATA / "qpoints.yaml", structure=DATA / "OUTCAR.phonon")
    assert vasp.num_qpoints == 1 and vasp.num_bands == 12 and vasp.ticks == []
    np.testing.assert_allclose(vasp.masses, [196.97, 63.55, 63.55, 63.55])  # POMASS, not ASE's table
    np.testing.assert_allclose(vasp.frequencies[0], ph.frequencies[0], atol=2e-6)
    # degenerate modes may come in any basis: compare the projector onto each subspace
    f = np.round(ph.frequencies[0], 4)
    for value in np.unique(f):
        sel = f == value
        a = vasp.eigenvectors[0, sel].reshape(sel.sum(), 12).real
        b = ph.eigenvectors[0, sel].reshape(sel.sum(), 12).real
        np.testing.assert_allclose(a.T @ a, b.T @ b, atol=2e-5)


def test_imaginary_modes_are_negative(tmp_path):
    text = (DATA / "OUTCAR.phonon").read_text()
    head, rest = text.split("Eigenvectors after division", 1)
    lines = head.splitlines()
    k = next(i for i, ln in enumerate(lines) if "THz" in ln and ln.split()[0] == "12")
    lines[k] = "  12 f/i=    1.500000 THz     9.424778 2PiTHz    50.034615 cm-1     6.203502 meV"
    (tmp_path / "OUTCAR").write_text("\n".join(lines) + "\n")
    m = read_modes(tmp_path / "OUTCAR")
    assert m.frequencies[0, 0] == pytest.approx(-1.5) and np.all(m.frequencies[0, 1:] > -1e-5)


def test_reader_sniffing_and_errors(tmp_path):
    assert type(import_file(DATA / "band.yaml").source.readers[0]).__name__ == "PhonopyYamlReader"
    assert type(import_file(DATA / "OUTCAR.phonon").source.readers[0]).__name__ == "VaspPhononReader"
    assert type(import_file(DATA / "OUTCAR.relax").source.readers[0]).__name__ == "OutcarReader"
    with pytest.raises(ValueError, match="structure="):
        import_file(DATA / "qpoints.yaml")
    stripped = "".join(ln for ln in (DATA / "band.yaml").read_text().splitlines(True)
                       if "eigenvector" not in ln and not ln.lstrip().startswith(("- [", "- #")) or "q-position" in ln
                       or ln.startswith("- ["))
    (tmp_path / "band.yaml").write_text(stripped)
    with pytest.raises(ValueError, match="--eigvecs"):
        import_file(tmp_path / "band.yaml")


def test_phonon_mode_animation_loops_one_period():
    pipe = import_file(DATA / "band.yaml", frames=8)
    assert pipe.num_frames == 8
    src = pipe.compute(0)
    assert "Phonon Bands" in src.tables and src.count == 4
    pipe.modifiers.append(PhononMode(qpoint=4, band=5, amplitude=0.2))   # X point: 2x1x1 supercell
    d0, d2, d4 = pipe.compute(0), pipe.compute(2), pipe.compute(4)
    assert d0.count == 8 and d0.attributes["Phonon.Supercell"] == (2, 1, 1)
    np.testing.assert_allclose(d0.cell.matrix, np.diag([7.5, 3.75, 3.75]))
    eq = d0.positions - d0["Phonon Mode"]
    np.testing.assert_allclose(d4.positions, eq - d0["Phonon Mode"], atol=1e-12)  # half a period later
    assert np.linalg.norm(d0["Phonon Mode"], axis=1).max() <= 0.2 + 1e-12
    assert not np.allclose(d2.positions, d0.positions)
    a = d0.attributes
    assert a["Phonon.QPoint"] == 4 and a["Phonon.Band"] == 5 and a["Phonon.q"] == (0.5, 0.0, 0.0)
    assert a["Phonon.Wavenumber"] == pytest.approx(a["Phonon.Frequency"] * 33.35641)
    assert "Phonon Mode" in d0.vectors and d0.vectors["Phonon Mode"].scaling == pytest.approx(2.5 / 0.2)
    pipe.modifiers[0].arrows = False
    assert "Phonon Mode" not in pipe.compute(1).vectors
    assert "Phonon Mode" not in pipe.source_frame(0)  # source frames stay untouched


def test_default_band_is_first_optical_and_modifier_needs_modes(cu_fcc):
    from ewald import Pipeline, StaticSource

    pipe = import_file(DATA / "OUTCAR.phonon")
    pipe.modifiers.append(PhononMode())
    d = pipe.compute(0)
    assert d.attributes["Phonon.Band"] == 3 and d.attributes["Phonon.Frequency"] == pytest.approx(3.565804)
    bad = Pipeline(StaticSource(cu_fcc))
    bad.modifiers.append(PhononMode())
    with pytest.raises(ValueError, match="phonon modes"):
        bad.compute(0)


def test_render_phonon_arrows(tmp_path):
    from ewald.vector import render_vector

    pipe = import_file(DATA / "band.yaml", frames=4)
    pipe.modifiers.append(PhononMode(qpoint=4, band=5))
    data = pipe.compute(1)
    cam = {"position": (20, -30, 15), "focal_point": (3.75, 1.9, 1.9), "up": (0, 0, 1), "parallel_projection": True,
           "parallel_scale": 6.0, "view_angle": 30.0}
    bare = data.copy()
    bare.vectors = {}
    plain = tmp_path / "plain.svg"
    render_vector(bare, plain, camera=cam, size=(120, 120))
    out = tmp_path / "mode.svg"
    render_vector(data, out, camera=cam, size=(120, 120))
    assert out.read_text().count("<path") > plain.read_text().count("<path")  # the mode arrows


def test_phonons_are_a_built_in_group_that_can_be_switched_off(tmp_path, monkeypatch):
    from ewald import plugins

    monkeypatch.setenv("EWALD_CONFIG_DIR", str(tmp_path))
    plugins.load_plugins(force=True)                  # whatever registry an earlier test left
    assert plugins.installed()["phonons"].kind == "built-in"
    assert plugins.find_reader(DATA / "band.yaml").name == "phonopy-yaml"
    assert plugins.find_reader(DATA / "band.yaml", format="phonopy").name == "phonopy-yaml"     # the old plugin's format name
    plugins.set_enabled("phonons", False)
    try:
        assert plugins.find_reader(DATA / "OUTCAR.phonon").name == "outcar"   # ionic steps only
    finally:
        plugins.set_enabled("phonons", True)
