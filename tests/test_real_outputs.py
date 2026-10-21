"""Readers on real VASP and LAMMPS output (tests/data/real, sources in its README).

The expected energies are ASE's (``ase.io.read(..., format="vasp-xml")``) for the same files.
"""
import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

from ewald import import_file
from ewald.io import read_lammps_log

REAL = Path(__file__).parent / "data" / "real"


def test_vasprun_vasp5_energy_comes_from_last_scstep():
    # VASP 5.2 writes e_0_energy = 0 after the last <scstep>
    pipe = import_file(REAL / "vasprun.static_silicon.xml.gz")
    d = pipe.compute(0)
    assert d.attributes["Energy"] == pytest.approx(-10.64527774)
    assert d.attributes["Free Energy"] == pytest.approx(-10.64527774)
    # with a correction outside the SCF (etest): E0 = F + (E0 - F) of the last <scstep>, as ASE
    thermo = import_file(REAL / "vasprun.etest3.xml.gz").source.readers[0].thermo
    np.testing.assert_allclose(thermo["Energy"][:3], [-14.17903605, 31.57893841, -14.5577799])
    np.testing.assert_allclose(thermo["Free Energy"][:3], [-14.20652677, 31.55143079, -14.58528579])


def test_vasprun_projections_with_fewer_columns_than_the_pdos():
    # VASP 5.2 (LORBIT unset): PDOS s..dx2, projections only s..dxy
    es = import_file(REAL / "vasprun.static_silicon.xml.gz").source.readers[0].electronic
    assert es.orbitals == ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
    p = es.projections
    assert p.shape == (2, 10, 9, 2, 9)
    np.testing.assert_allclose(p[0, 0, 0, 0], [0.2089, 0.0008, 0.0008, 0.0008, 0, 0, 0, 0, 0])
    assert es.band_weights(atoms=[0], orbitals="d").shape == (2, 10, 9)


def test_vasprun_overflowed_forces_read_as_nan(tmp_path):
    text = gzip.open(REAL / "vasprun.static_silicon.xml.gz", "rt").read()
    i = text.index('<varray name="forces" >')
    j = text.index("<v>", i) + 3
    k = text.index(" ", text.index(".", j))  # end of the first force component
    path = tmp_path / "vasprun.xml"
    path.write_text(text[:j] + "   **********" + text[k:])
    f = import_file(path).compute(0)["Force"]
    assert np.isnan(f[0, 0]) and not np.isnan(f[0, 1:]).any() and not np.isnan(f[1:]).any()


def test_vasprun_ignores_kpoints_and_eigenvalues_of_electron_velocities(tmp_path):
    # LVEL = .TRUE. writes <electronvelocities> with its own (full-zone) <kpoints> and <eigenvalues>
    text = gzip.open(REAL / "vasprun.static_silicon.xml.gz", "rt").read()
    block = ("  <electronvelocities>\n   <kpoints>\n    <varray name=\"kpointlist\" >\n"
             "     <v>  0.0 0.0 0.0 </v>\n    </varray>\n    <varray name=\"weights\" >\n     <v> 1.0 </v>\n"
             "    </varray>\n   </kpoints>\n   <eigenvalues>\n    <array>\n     <set>\n"
             "      <set comment=\"spin 1\">\n       <set comment=\"kpoint 1\">\n"
             "        <r>  -1.0  0.5  0.5  0.5 </r>\n       </set>\n      </set>\n     </set>\n"
             "    </array>\n   </eigenvalues>\n  </electronvelocities>\n")
    i = text.index("  </eigenvalues>\n") + len("  </eigenvalues>\n")
    path = tmp_path / "vasprun.xml"
    path.write_text(text[:i] + block + text[i:])
    ref = import_file(REAL / "vasprun.static_silicon.xml.gz").source.readers[0].electronic
    es = import_file(path).source.readers[0].electronic
    assert es.eigenvalues.shape == (2, 10, 9) and es.kpoints.shape == (10, 3)
    np.testing.assert_array_equal(es.eigenvalues, ref.eigenvalues)
    np.testing.assert_array_equal(es.occupations, ref.occupations)


def test_vasprun_without_ionic_steps_gives_the_structure():
    # GW0 run: <calculation> has no <structure>; the frame is <structure name="finalpos">
    pipe = import_file(REAL / "vasprun.GW0.xml.gz")
    assert pipe.num_frames == 1
    d = pipe.compute(0)
    assert list(d.types) == ["Si", "Si"]
    np.testing.assert_allclose(d.positions[1], [1.3575] * 3)
    np.testing.assert_allclose(d.cell.matrix[0], [2.715, 2.715, 0])
    es = pipe.source.readers[0].electronic
    assert es.eigenvalues.shape == (1, 16, 128) and es.efermi == pytest.approx(5.3844888)


def test_xdatcar_with_columns_run_together():
    pipe = import_file(REAL / "XDATCAR.bad_fmt.gz")  # '-0.63265286-0.11227753 -0.15402785'
    assert pipe.num_frames == 3
    d = pipe.compute(1)
    np.testing.assert_allclose(d.cell.to_fractional(d.positions)[0], [-0.63265286, -0.11227753, -0.15402785])
    assert [pipe.compute(i).attributes["Timestep"] for i in range(3)] == [1, 2, 3]


def test_vaspout_hybrid_band_path_and_orbitals():
    pytest.importorskip("h5py")
    # band path as zero-weight k-points after the SCF mesh, labels at both ends of every line;
    # lchar lists s..f but the PDOS has s..dx2 only (projections removed to keep the file small)
    pipe = import_file(REAL / "vaspout.line_mode_band_structure.h5")
    es = pipe.source.readers[0].electronic
    assert es.eigenvalues.shape == (1, 147, 20) and len(es.kpoints) == 147 and (es.weights == 0).all()
    assert len(es.orbitals) == 9 and es.pdos.shape == (2, 1, 9, 301)
    assert [t[1] for t in es.ticks] == ["Γ", "X", "W", "K", "Γ", "L", "U", "W", "L", "K|U", "X"]
    assert es.breaks == [138]  # K|U: no distance between the two lines
    assert es.distances[138] == es.distances[137]
    gap = es.band_gap()
    assert gap["gap"] == pytest.approx(0.78098165, abs=1e-6) and gap["direct"]
    d = pipe.compute(0)
    assert "DOS" in d.tables and d.attributes["Energy"] == pytest.approx(-8.953035077)


def test_lammps_log_thermo_style_multi():
    t = read_lammps_log(REAL / "log.gauss-diel")
    np.testing.assert_array_equal(t["Step"], [0, 500, 1000, 1500, 2000])
    assert t["TotEng"][0] == 9.9401 and t["Press"][1] == 0.0065 and t["CPU"][1] == 1.0851
    assert {"KinEng", "Temp", "PotEng", "E_bond", "E_long"} <= set(t)


def test_neb_endpoint_structure_from_poscar(vasp_dir, tmp_path):
    # an endpoint OUTCAR from a separate relaxation can list the atoms in another order;
    # VASP's NEB uses the endpoint POSCAR, so the frame has its atoms and the OUTCAR's energy
    folder = tmp_path / "neb"
    shutil.copytree(vasp_dir / "neb", folder)
    lines = (folder / "00" / "POSCAR").read_text().splitlines()
    lines[8], lines[9] = lines[9], lines[8]  # swap two atoms
    (folder / "00" / "POSCAR").write_text("\n".join(lines) + "\n")
    pipe = import_file(folder)
    d = pipe.source.compute(0)
    ref = import_file(vasp_dir / "neb" / "00" / "OUTCAR").compute(0)
    np.testing.assert_allclose(d.positions[[1, 0]], ref.positions[:2], atol=1e-4)
    assert d.attributes["Energy"] == ref.attributes["Energy"] and "Force" not in d.particles
    unchanged = import_file(vasp_dir / "neb").source.compute(0)  # same atoms: the OUTCAR's forces stay
    np.testing.assert_allclose(unchanged["Force"], ref["Force"])


def test_neb_endpoint_poscar_rule_with_windows_paths(vasp_dir, monkeypatch):
    # on Windows the image files are C:\...\00\OUTCAR; the rule above once split them on "/"
    # only, missed the OUTCAR and took a reordered endpoint OUTCAR's atoms (a 13 A last step
    # in the Fe vacancy example)
    import ntpath

    from ewald.io import neb

    image_file = neb._image_file
    monkeypatch.setattr(neb, "_image_file", lambda d, *a: (p := image_file(d, *a)) and p.replace("/", "\\"))
    with monkeypatch.context() as m:
        m.setattr(neb.os.path, "basename", ntpath.basename)
        reader = neb.NebReader(vasp_dir / "neb")
    assert reader.files[0].endswith("00\\OUTCAR") and 0 in reader._endpoint_poscars
