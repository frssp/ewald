"""Files of calculations that have not finished: VASP still running or killed, LAMMPS mid-dump.

Every reader reads what is complete, leaves the part being written out, says so
(``IncompleteFileWarning``, ``reader.incomplete``, ``FileSource.incomplete``) and raises
``IncompleteFileError`` only when nothing is complete yet.
"""
import gzip
import os
import shutil
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from ewald import import_file
from ewald.io import IncompleteFileError, IncompleteFileWarning
from ewald.io.base import FrameReader


def _open(path, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = import_file(path, **kwargs)
    return pipe, [w for w in caught if issubclass(w.category, IncompleteFileWarning)]


# ------------------------------------------------------------------ every cut of every fixture
def _fixtures(vasp_dir, lammps_dir, tmp_path):
    real = vasp_dir.parent / "real" / "vasprun.static_silicon.xml.gz"
    static = tmp_path / "static" / "vasprun.xml"
    static.parent.mkdir()
    static.write_bytes(gzip.open(real).read())
    return [vasp_dir / "vasprun.xml", vasp_dir / "electronic" / "vasprun.xml", static, vasp_dir / "OUTCAR",
            vasp_dir / "md" / "OUTCAR", vasp_dir / "XDATCAR", vasp_dir / "XDATCAR.npt",
            lammps_dir / "dump.custom.lammpstrj", lammps_dir / "dump.triclinic.lammpstrj"]


def test_every_truncation_reads_or_says_why(vasp_dir, lammps_dir, tmp_path):
    """Cut each fixture at 25 places (the last one inside its last frame): either the complete frames are read (each one computes)
    or IncompleteFileError says nothing is complete yet; never another exception."""
    for src in _fixtures(vasp_dir, lammps_dir, tmp_path):
        blob = src.read_bytes()
        work = tmp_path / "cut"
        for cut in np.linspace(0.05 * len(blob), 0.95 * len(blob), 25).astype(int):  # always inside the last frame
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
            path = work / src.name
            path.write_bytes(blob[:cut])
            try:
                pipe, caught = _open(path, thermo=None)
            except IncompleteFileError:
                continue
            for i in range(pipe.num_frames):
                pipe.compute(i)
            assert pipe.source.incomplete, (src, cut)  # a cut file always says so
            assert caught, (src, cut)


def test_complete_files_are_not_flagged(vasp_dir, lammps_dir):
    for path in (vasp_dir / "vasprun.xml", vasp_dir / "electronic" / "vasprun.xml", vasp_dir / "OUTCAR",
                 vasp_dir / "md" / "OUTCAR", vasp_dir / "XDATCAR", lammps_dir / "dump.custom.lammpstrj"):
        pipe, caught = _open(path)
        assert not caught and pipe.source.incomplete == [], path


# ------------------------------------------------------------------ vasprun.xml
def test_running_vasprun(vasp_dir, tmp_path):
    xml = (vasp_dir / "vasprun.xml").read_text()
    cut = xml.index("<varray name=\"positions\"", xml.rindex("<calculation>"))
    (tmp_path / "vasprun.xml").write_text(xml[: cut + 80])
    pipe, caught = _open(tmp_path / "vasprun.xml")
    assert pipe.num_frames == 1 and len(caught) == 1
    (note,) = pipe.source.incomplete
    assert note.startswith("vasprun.xml: 1 ionic step(s) read, the partial last one skipped")
    assert "DOS and bands not written yet" in note and "</modeling>" in note
    assert "Energy" in pipe.compute(0).attributes


def test_vasprun_before_the_first_step_shows_the_initial_structure(vasp_dir, tmp_path):
    xml = (vasp_dir / "vasprun.xml").read_text()
    (tmp_path / "vasprun.xml").write_text(xml[: xml.index("<calculation>") + 40])
    pipe, _ = _open(tmp_path / "vasprun.xml")
    assert pipe.num_frames == 1
    assert "initial structure" in pipe.source.incomplete[0]
    full = import_file(vasp_dir / "vasprun.xml").compute(0)
    assert pipe.compute(0).count == full.count
    assert "Energy" not in pipe.compute(0).attributes


def test_vasprun_that_has_just_started(vasp_dir, tmp_path):
    xml = (vasp_dir / "vasprun.xml").read_text()
    for end in (xml.index("<atominfo>") - 5, xml.index("<atominfo>") + 30):
        (tmp_path / "vasprun.xml").write_text(xml[:end])
        with pytest.raises(IncompleteFileError, match="atominfo"):
            import_file(tmp_path / "vasprun.xml")


def test_vasprun_cut_inside_the_dos(vasp_dir, tmp_path):
    xml = (vasp_dir / "electronic" / "vasprun.xml").read_text()
    (tmp_path / "vasprun.xml").write_text(xml[: xml.index("<partial>")])
    pipe, caught = _open(tmp_path / "vasprun.xml")
    assert pipe.num_frames == 1 and len(caught) == 1  # one warning, not a separate DOS error
    assert "DOS and bands cut short" in pipe.source.incomplete[0]
    assert pipe.source.readers[0].electronic is None
    pipe.compute(0)


def test_vasprun_followed_until_the_run_ends(vasp_dir, tmp_path):
    """Live reload: a vasprun.xml written step by step; at the end the DOS appears."""
    blob = (vasp_dir / "electronic" / "vasprun.xml").read_bytes()
    path = tmp_path / "vasprun.xml"
    path.write_bytes(blob[: blob.index(b"<dos>")])
    pipe, _ = _open(path)
    assert pipe.source.readers[0].electronic is not None  # the eigenvalues come first
    assert not pipe.source.readers[0].electronic.has_dos
    assert pipe.source.incomplete
    pipe.compute(0)
    path.write_bytes(blob)
    assert pipe.source.refresh() == 0
    assert pipe.source.incomplete == []
    assert pipe.source.readers[0].electronic.has_dos


def test_vasprun_grows_step_by_step(vasp_dir, tmp_path):
    from test_live_reload import _grow_and_compare

    blob = (vasp_dir / "vasprun.xml").read_bytes()
    pipe = _grow_and_compare(tmp_path / "vasprun.xml", blob, 397)
    assert pipe.num_frames == import_file(vasp_dir / "vasprun.xml").num_frames
    assert pipe.source.incomplete == []


# ------------------------------------------------------------------ OUTCAR
def _outcar_with_initial_positions(md_outcar):
    text = md_outcar.read_text()
    block = ("\n position of ions in cartesian coordinates  (Angst):\n"
             "      0.00000000  0.00000000  0.00000000\n      1.43300000  1.43300000  1.43300000\n\n")
    first = text.index("------------------------------------- Iteration")
    first = text.rindex("\n", 0, first) + 1
    return text[:first] + block + text[first:]


def test_outcar_before_the_first_ionic_step(vasp_dir, tmp_path):
    text = _outcar_with_initial_positions(vasp_dir / "md" / "OUTCAR")
    (tmp_path / "OUTCAR").write_text(text[: text.index("POSITION")])  # electronic steps of step 1 only
    pipe, caught = _open(tmp_path / "OUTCAR")
    assert caught and "initial structure" in pipe.source.incomplete[0]
    data = pipe.compute(0)
    np.testing.assert_allclose(data.positions[1], [1.433] * 3)
    assert data.attributes["Timestep"] == 0 and "Force" not in data.particles
    # without the initial positions there is nothing to show yet
    plain = (vasp_dir / "md" / "OUTCAR").read_text()
    (tmp_path / "OUTCAR").write_text(plain[: plain.index("POSITION")])
    with pytest.raises(IncompleteFileError, match="no ionic step"):
        import_file(tmp_path / "OUTCAR")


def test_outcar_followed_from_the_initial_structure(vasp_dir, tmp_path):
    text = _outcar_with_initial_positions(vasp_dir / "md" / "OUTCAR").encode()
    path = tmp_path / "OUTCAR"
    path.write_bytes(text[: text.index(b"POSITION")])
    pipe, _ = _open(path)
    assert pipe.compute(0).attributes["Timestep"] == 0
    path.write_bytes(text)
    pipe.refresh()
    assert pipe.num_frames == import_file(vasp_dir / "md" / "OUTCAR").num_frames
    assert pipe.compute(0).attributes["Timestep"] == 1
    assert pipe.source.incomplete == []


def test_outcar_without_the_header_yet(vasp_dir, tmp_path):
    text = (vasp_dir / "md" / "OUTCAR").read_text()
    (tmp_path / "OUTCAR").write_text(text[: text.index("Dimension of arrays")])
    with pytest.raises(IncompleteFileError, match="ions per type"):
        import_file(tmp_path / "OUTCAR")


# ------------------------------------------------------------------ OSZICAR, XDATCAR, dump, extxyz, CHGCAR
def test_oszicar_half_written_line(vasp_dir, tmp_path):
    from ewald.io import read_oszicar

    text = (vasp_dir / "md" / "OSZICAR").read_text()
    full = read_oszicar(vasp_dir / "md" / "OSZICAR")
    last = text.rstrip("\n").rindex("\n") + 1
    (tmp_path / "OSZICAR").write_text(text[: last + 12])  # "   3 T=   3" ...
    table = read_oszicar(tmp_path / "OSZICAR")
    assert len(table["Step"]) == len(full["Step"]) - 1


def test_nothing_complete_yet_is_an_incomplete_file_error(vasp_dir, lammps_dir, tmp_path):
    xdatcar = (vasp_dir / "XDATCAR").read_text()
    (tmp_path / "XDATCAR").write_text(xdatcar[: xdatcar.index("configuration=     2") - 40])
    dump = (lammps_dir / "dump.custom.lammpstrj").read_text()
    (tmp_path / "dump.lammpstrj").write_text(dump[: dump.index("ITEM: ATOMS") + 60])
    (tmp_path / "md.xyz").write_text("3\nLattice=\"5 0 0 0 5 0 0 0 5\"\nCu 0 0 0\nCu 1 1 1\n")
    for name in ("XDATCAR", "dump.lammpstrj", "md.xyz"):
        with pytest.raises(IncompleteFileError):
            import_file(tmp_path / name, thermo=None)
    assert issubclass(IncompleteFileError, ValueError)  # code catching ValueError keeps working


def test_incomplete_note_clears_when_the_frame_is_finished(lammps_dir, tmp_path):
    blob = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    path = tmp_path / "dump.lammpstrj"
    second = blob.index(b"ITEM: TIMESTEP", blob.index(b"ITEM: TIMESTEP") + 1)
    path.write_bytes(blob[: second + 200])
    pipe, caught = _open(path)
    assert pipe.num_frames == 1 and caught
    assert pipe.source.incomplete == ["dump.lammpstrj: 1 frame(s) read, the partial last one skipped"]
    path.write_bytes(blob)
    pipe.refresh()
    assert pipe.source.incomplete == []


def test_chgcar_killed_while_writing_the_grid(tmp_path):
    from ewald.examples import path as example

    text = open(example("CHGCAR.Si")).read()
    lines = text.splitlines(keepends=True)
    grid = next(i for i, line in enumerate(lines) if len(line.split()) == 3 and line.split()[0].isdigit()
                and i > 8)
    (tmp_path / "CHGCAR").write_text("".join(lines[: grid + 5]))
    natoms = import_file(example("CHGCAR.Si")).compute(0).count
    pipe = import_file(tmp_path / "CHGCAR")
    with pytest.warns(IncompleteFileWarning, match="grid block 1"):
        data = pipe.compute(0)
    assert data.count == natoms and not data.grids  # the atoms alone
    assert "grid block 1" in pipe.source.incomplete[0]


# ------------------------------------------------------------------ HDF5: vaspout.h5, H5MD
h5py = pytest.importorskip("h5py")


def test_vaspout_open_while_vasp_holds_the_write_lock(vasp_dir, tmp_path):
    shutil.copy(vasp_dir / "h5" / "md" / "vaspout.h5", tmp_path / "vaspout.h5")
    script = textwrap.dedent("""
        import sys, time, h5py
        f = h5py.File(sys.argv[1], "r+")  # like a running VASP: the file stays open for writing
        print("open", flush=True)
        sys.stdin.readline()
    """)
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "vaspout.h5")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "open"
        with pytest.raises(OSError):  # plain h5py: "unable to lock file"
            h5py.File(tmp_path / "vaspout.h5", "r")
        pipe = import_file(tmp_path / "vaspout.h5")
        assert pipe.num_frames == 3
        pipe.compute(2)
    finally:
        proc.communicate("\n")


def test_unreadable_vaspout_falls_back_to_vasprun(vasp_dir, tmp_path):
    blob = (vasp_dir / "h5" / "md" / "vaspout.h5").read_bytes()
    (tmp_path / "vaspout.h5").write_bytes(blob[: len(blob) // 2])  # killed before HDF5 closed the file
    with pytest.raises(IncompleteFileError, match="HDF5"):
        import_file(tmp_path / "vaspout.h5")
    xml = (vasp_dir / "vasprun.xml").read_text()
    (tmp_path / "vasprun.xml").write_text(xml[: xml.rindex("<calculation>") + 20])
    pipe, caught = _open(tmp_path / "vaspout.h5")
    assert type(pipe.source.readers[0]).__name__ == "VasprunReader"
    (note,) = pipe.source.incomplete
    assert note.startswith("vaspout.h5: not readable yet, read from vasprun.xml instead: 1 ionic step(s) read")
    assert len(caught) == 1


def test_vaspout_with_unwritten_steps(tmp_path, vasp_dir):
    path = tmp_path / "vaspout.h5"
    shutil.copy(vasp_dir / "h5" / "md" / "vaspout.h5", path)
    with h5py.File(path, "r+") as f:  # room for 5 steps, 3 written (as a run stopped by the queue)
        d = f["intermediate/ion_dynamics"]
        for key in ("position_ions", "lattice_vectors", "forces"):
            old = d[key][()]
            del d[key]
            d[key] = np.concatenate([old, np.zeros((2,) + old.shape[1:])])
    pipe, caught = _open(path)
    assert pipe.num_frames == 3 and caught
    assert "2 empty step(s) at the end skipped" in pipe.source.incomplete[0]


def test_vaspout_without_results_yet(tmp_path):
    with h5py.File(tmp_path / "vaspout.h5", "w") as f:
        f["input/incar/SYSTEM"] = np.bytes_("running")
    with pytest.raises(IncompleteFileError, match="ion_types"):
        import_file(tmp_path / "vaspout.h5")


def test_h5md_grows(tmp_path):
    path = tmp_path / "md.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = [1, 1]
        g = f.create_group("h5md")
        g.attrs["version"] = np.array([1, 1])
        pos = f.create_group("particles/all/position")
        pos.create_dataset("value", data=rng.random((2, 4, 3)), maxshape=(None, 4, 3))
        pos.create_dataset("step", data=np.array([0, 10]), maxshape=(None,))
        pos.create_dataset("time", data=np.array([0.0, 1.0]), maxshape=(None,))
        box = f.create_group("particles/all/box")
        box.attrs["dimension"] = 3
        box.attrs["boundary"] = np.array([b"periodic"] * 3)
        box.create_dataset("edges", data=np.array([5.0, 5.0, 5.0]))
    pipe = import_file(path)
    assert pipe.num_frames == 2
    with h5py.File(path, "r+") as f:
        v = f["particles/all/position/value"]
        v.resize((3, 4, 3))  # LAMMPS grew the positions, the step comes next
        v[2] = rng.random((4, 3))
    assert pipe.refresh() == 0 and pipe.num_frames == 2
    with h5py.File(path, "r+") as f:
        for key, value in (("step", 20), ("time", 2.0)):
            ds = f["particles/all/position"][key]
            ds.resize((3,))
            ds[2] = value
    assert pipe.refresh() == 1 and pipe.num_frames == 3
    pipe.compute(2)


# ------------------------------------------------------------------ NEB
def test_neb_image_without_a_finished_step(vasp_dir, tmp_path):
    neb = tmp_path / "neb"
    shutil.copytree(vasp_dir / "neb", neb)
    full = (neb / "02" / "OUTCAR").read_text()
    (neb / "02" / "OUTCAR").write_text(full[: full.index("POSITION")])
    pipe, _ = _open(neb)
    data = pipe.source.compute(2)
    assert data.attributes["NEB.Source"].endswith("OUTCAR")
    assert pipe.source.incomplete == ["neb: image 02: OUTCAR has no finished ionic step yet, CONTCAR shown"]
    assert "Force" not in data.particles
    (neb / "02" / "OUTCAR").write_text(full)
    assert pipe.source.refresh() == 2
    assert "Force" in pipe.source.compute(2).particles
    assert pipe.source.incomplete == []


def test_reader_default_is_complete():
    assert FrameReader.incomplete is None
