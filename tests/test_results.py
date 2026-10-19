"""VASP/LAMMPS results (OUTCAR, vasprun.xml, OSZICAR, log.lammps) and their alignment with frames.

tests/data/lammps/thermo was written by LAMMPS 2025 (generate.in there); tests/data/vasp/md
is hand-written in VASP 6.4 format (MD, ISPIN=2). The native VASP readers are checked
against ASE's parsers on the same files.
"""
import gzip
import shutil

import numpy as np
import pytest

from ewald import import_file
from ewald.io import OutcarReader, VasprunReader, open_reader, read_lammps_log, read_oszicar
from ewald.io.base import grep_lines


@pytest.fixture
def md_dir(vasp_dir):
    return vasp_dir / "md"


@pytest.fixture
def thermo_dir(lammps_dir):
    return lammps_dir / "thermo"


@pytest.mark.parametrize("name", ["OUTCAR", "vasprun.xml", "md/OUTCAR"])
def test_native_vasp_readers_match_ase(vasp_dir, name):
    ase_io = pytest.importorskip("ase.io")
    path = vasp_dir / name
    pipe = import_file(path)
    assert isinstance(pipe.source.readers[0], (OutcarReader, VasprunReader))
    images = ase_io.read(path, index=":")
    assert pipe.num_frames == len(images)
    for i, atoms in enumerate(images):
        d = pipe.compute(i)
        assert list(d.types) == atoms.get_chemical_symbols()
        np.testing.assert_allclose(d.positions, atoms.positions, atol=1e-8)
        np.testing.assert_allclose(d.cell.matrix, atoms.cell[:], atol=1e-8)
        np.testing.assert_allclose(d["Force"], atoms.get_forces(), atol=1e-8)
        np.testing.assert_allclose(d.attributes["Stress"], atoms.get_stress(), atol=1e-10)
        assert d.attributes["Energy"] == pytest.approx(atoms.get_potential_energy())
        assert d.attributes["Free Energy"] == pytest.approx(atoms.get_potential_energy(force_consistent=True))
        assert d.attributes["Timestep"] == i + 1
        if "magmoms" in atoms.calc.results:
            np.testing.assert_allclose(d["Magnetic Moment"], atoms.get_magnetic_moments())


def test_outcar_md_spin_attributes_and_table(md_dir):
    pipe = import_file(md_dir / "OUTCAR")
    d = pipe.compute(1)
    np.testing.assert_allclose(d["Magnetic Moment"], [2.231, 2.198])
    a = d.attributes
    assert (a["Temperature"], a["Kinetic Energy"], a["Total Energy"]) == (233.29, 0.06031, -16.21969)
    assert a["Magnetization"] == 4.429  # converged SCF step, not an earlier one
    assert a["Pressure"] == 11.25  # external pressure (Pulay stress removed)
    table = d.tables["Thermo"]
    np.testing.assert_array_equal(table["Step"], [1, 2])
    np.testing.assert_allclose(table["Temperature"], [300.0, 233.29])


def test_xdatcar_picks_up_oszicar(md_dir):
    pipe = import_file(md_dir / "XDATCAR")
    assert pipe.source.thermo.sources[0].endswith("OSZICAR")
    d = pipe.compute(0)
    assert d.attributes["Temperature"] == 300.0
    assert d.attributes["Energy"] == pytest.approx(-16.295)
    assert d.attributes["Free Energy"] == pytest.approx(-16.30)
    assert d.attributes["Magnetization"] == pytest.approx(4.42)
    # the OSZICAR and the OUTCAR of the same run agree
    outcar = import_file(md_dir / "OUTCAR").compute(0).attributes
    for k in ("Energy", "Free Energy", "Total Energy", "Kinetic Energy", "Magnetization"):
        assert d.attributes[k] == pytest.approx(outcar[k]), k
    assert "Thermo" in d.tables
    assert import_file(md_dir / "XDATCAR", thermo=None).source.thermo is None


def test_oszicar_relaxation_lines(tmp_path):
    (tmp_path / "OSZICAR").write_text(
        "       N       E                     dE             d eps       ncg     rms          rms(c)\n"
        "DAV:   1    -0.204500000000E+02   -0.20450E+02   -0.12345E+03    48   0.123E+02\n"
        "   1 F= -.20500000E+02 E0= -.20450000E+02  d E =-.205000E+02  mag=     2.0000\n"
        "   2 F= -.20600000E+02 E0= -.20550000E+02  d E =-.100000E+00  mag=     0.1000     0.2000     0.2000\n")
    t = read_oszicar(tmp_path / "OSZICAR")
    np.testing.assert_array_equal(t["Step"], [1, 2])
    np.testing.assert_allclose(t["Energy"], [-20.45, -20.55])
    np.testing.assert_allclose(t["Magnetization"], [2.0, 0.3])  # non-collinear: |m|
    assert set(t) == {"Step", "Free Energy", "Energy", "Magnetization"}


def test_lammps_log_two_runs(thermo_dir):
    t = read_lammps_log(thermo_dir / "log.lammps")
    np.testing.assert_array_equal(t["Step"], np.arange(0, 45, 5))
    assert list(t) == ["Step", "Temp", "PotEng", "KinEng", "TotEng", "Press", "Volume", "c_thermo_press[1]"]
    assert np.isnan(t["Volume"][0]) and np.isnan(t["KinEng"][-1])
    i20 = 4  # last step of run 1 = first of run 2: both rows' columns
    assert t["KinEng"][i20] == pytest.approx(1.0955282) and t["Volume"][i20] == pytest.approx(377.93307)


def test_dump_aligned_with_log(thermo_dir):
    pipe = import_file(thermo_dir / "dump.lammpstrj")
    assert pipe.num_frames == 5
    steps = [pipe.compute(i).attributes["Timestep"] for i in range(5)]
    assert steps == [0, 10, 20, 30, 40]
    d = pipe.compute(1)
    assert d.attributes["Temp"] == pytest.approx(144.84182)
    assert "Volume" not in d.attributes  # run 1 does not print it
    assert pipe.compute(4).attributes["Volume"] == pytest.approx(377.93307)
    assert len(d.tables["Thermo"]["Step"]) == 9


def test_log_with_warnings_truncated_line_and_unmatched_frames(thermo_dir, tmp_path):
    text = (thermo_dir / "log.lammps").read_text().splitlines(keepends=True)
    i = next(k for k, line in enumerate(text) if line.split()[:1] == ["5"])
    text.insert(i, "WARNING: Pair style restartinfo set but has no restart support (src/pair.cpp:1234)\n")
    text = text[: next(k for k, line in enumerate(text) if line.split()[:1] == ["35"])] + ["        35   301.85"]
    log = tmp_path / "md.log"
    log.write_text("".join(text))
    t = read_lammps_log(log)
    np.testing.assert_array_equal(t["Step"], [0, 5, 10, 15, 20, 25, 30])
    shutil.copy(thermo_dir / "dump.lammpstrj", tmp_path / "dump.lammpstrj")
    pipe = import_file(tmp_path / "dump.lammpstrj", thermo=log)
    assert pipe.compute(3).attributes["Temp"] == pytest.approx(290.71894)
    assert "Temp" not in pipe.compute(4).attributes  # step 40 is not in the log
    with pytest.raises(ValueError):
        import_file(tmp_path / "dump.lammpstrj", thermo=thermo_dir / "generate.in")


def test_truncated_outcar_and_vasprun(md_dir, vasp_dir, tmp_path):
    text = (md_dir / "OUTCAR").read_text()
    cut = text.rindex("POSITION")
    (tmp_path / "OUTCAR").write_text(text[: cut + 300])  # job killed while writing step 2
    with pytest.warns(UserWarning, match="incomplete"):
        pipe = import_file(tmp_path / "OUTCAR")
    assert pipe.num_frames == 1
    xml = (vasp_dir / "vasprun.xml").read_text()
    cut = xml.index("<varray name=\"positions\"", xml.rindex("<calculation>"))
    (tmp_path / "vasprun.xml").write_text(xml[: cut + 80])  # running job
    with pytest.warns(UserWarning, match="incomplete"):
        pipe = import_file(tmp_path / "vasprun.xml")
    assert pipe.num_frames == 1


def test_noncollinear_magnetic_moments(md_dir, tmp_path):
    text = (md_dir / "OUTCAR").read_text()
    blocks = text.split(" magnetization (x)\n")
    new = blocks[0]
    for b in blocks[1:]:  # repeat each (x) block as (y) and (z)
        body = b[: b.index("\n\n") + 2]
        new += "".join(f" magnetization ({a})\n" + body for a in "xyz") + b[len(body):]
    (tmp_path / "OUTCAR").write_text(new)
    m = import_file(tmp_path / "OUTCAR").compute(0)["Magnetic Moment"]
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m[:, 2], [2.216, 2.204])


def test_detected_by_content_and_gzip(md_dir, vasp_dir, tmp_path):
    shutil.copy(md_dir / "OUTCAR", tmp_path / "run1.out")
    shutil.copy(vasp_dir / "vasprun.xml", tmp_path / "run1.xml")
    assert isinstance(open_reader(tmp_path / "run1.out"), OutcarReader)
    assert isinstance(open_reader(tmp_path / "run1.xml"), VasprunReader)
    gz = tmp_path / "OUTCAR.gz"
    gz.write_bytes(gzip.compress((md_dir / "OUTCAR").read_bytes()))
    np.testing.assert_allclose(import_file(gz).compute(1)["Force"], import_file(md_dir / "OUTCAR").compute(1)["Force"])
    assert import_file(md_dir / "OUTCAR", format="vasp-out").num_frames == 2  # ASE's reader by its own name


def test_grep_lines_across_chunks(md_dir):
    path = md_dir / "OUTCAR"
    raw = path.read_bytes()
    whole = grep_lines(path, "TOTAL-FORCE|^  in kB")
    assert grep_lines(path, "TOTAL-FORCE|^  in kB", chunk_size=97) == whole
    assert [raw[off:off + len(text)] for off, text in whole] == [text for _, text in whole]
    assert len(whole) == 4


def test_remote_results_read_in_place(server, md_dir, thermo_dir, tmp_path):
    data = tmp_path / "remote_fs" / "data"
    for src in (md_dir / "OUTCAR", md_dir / "OSZICAR", thermo_dir / "dump.lammpstrj", thermo_dir / "log.lammps"):
        shutil.copy(src, data / src.name)
    shutil.copy(md_dir / "XDATCAR", data / "XDATCAR")  # replaces the fixture's Si XDATCAR
    pipe = import_file("hpc:/data/OUTCAR")
    np.testing.assert_allclose(pipe.compute(1)["Magnetic Moment"], [2.231, 2.198])
    assert pipe.compute(1).attributes["Temperature"] == 233.29
    xd = import_file("hpc:/data/XDATCAR")
    assert xd.compute(1).attributes["Temperature"] == 233.0  # OSZICAR found next to it on the host
    dump = import_file("hpc:/data/dump.lammpstrj")
    assert dump.compute(2).attributes["Volume"] == pytest.approx(377.93307)
    assert not (tmp_path / "cache").exists()  # nothing downloaded
