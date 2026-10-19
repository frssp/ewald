"""Built-in format readers and writers (extxyz, XSF, H5MD, XTC/TRR/DCD, ASE db/traj, POSCAR,
LAMMPS data) and the writer registry."""
import textwrap

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, import_file, plugins
from ewald.data import VoxelGrid
from ewald.io import export_file


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


def _trajectory(n=4, element="Cu"):
    frames = []
    for k in range(n):
        atoms = bulk(element, "fcc", a=3.6) * (2, 2, 2)   # triclinic primitive cell
        atoms.rattle(0.05, seed=k)
        d = DataCollection.from_ase(atoms)
        d.particles["Particle Identifier"] = np.arange(1, len(atoms) + 1)
        d.particles["Force"] = np.random.default_rng(k).normal(size=(len(atoms), 3))
        d.attributes["Energy"] = -1.5 * k
        d.attributes["Timestep"] = 10 * k
        frames.append(d)
    return frames


def _nacl():
    d = DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64) * (2, 1, 1))
    d.particles["Particle Identifier"] = np.arange(1, d.count + 1)
    return d


def _same_structure(a, b, atol=1e-5):
    """Same atoms in the same cell up to a rigid rotation (formats that store a standard box)."""
    assert a.count == b.count
    assert list(map(str, a.types)) == list(map(str, b.types))
    np.testing.assert_allclose(np.linalg.norm(a.cell.matrix, axis=1), np.linalg.norm(b.cell.matrix, axis=1), atol=atol)
    fa, fb = a.cell.to_fractional(a.positions), b.cell.to_fractional(b.positions)
    np.testing.assert_allclose((fa - fb + 0.5) % 1 - 0.5, 0, atol=atol)


@pytest.mark.parametrize("name, reader", [("traj.extxyz", "extxyz"), ("traj.xyz", "extxyz"),
                                          ("traj.axsf", "xsf"), ("traj.traj", "ase-traj"),
                                          ("traj.db", "ase-db")])
def test_trajectory_round_trip(tmp_path, name, reader):
    frames = _trajectory()
    path = tmp_path / name
    export_file(frames, path)
    assert plugins.find_reader(path).name == reader
    pipe = import_file(path)
    assert pipe.num_frames == 4
    for k in (0, 3, 1):   # random access
        got = pipe.compute(k)
        _same_structure(got, frames[k])
        if reader in ("extxyz", "xsf", "ase-traj", "ase-db"):
            np.testing.assert_allclose(got["Force"], frames[k]["Force"], atol=1e-6)
        if reader in ("extxyz", "ase-traj", "ase-db"):
            assert got.attributes["Energy"] == pytest.approx(frames[k].attributes["Energy"])


def test_extxyz_reader_is_lazy_and_skips_a_truncated_frame(tmp_path):
    path = tmp_path / "md.extxyz"
    export_file(_trajectory(3), path)
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-5]) + "\n")   # killed while writing the last frame
    with pytest.warns(UserWarning, match="incomplete"):
        reader = plugins.find_reader(path).open(path)
    assert len(reader) == 2 and reader.read(-1).count == 8


def test_extxyz_keeps_info_and_custom_properties(tmp_path):
    d = _trajectory(1)[0]
    d.particles["Charge"] = np.linspace(-1, 1, d.count)
    d.attributes["config_type"] = "bulk"
    export_file(d, tmp_path / "a.extxyz")
    got = import_file(tmp_path / "a.extxyz").compute(0)
    np.testing.assert_allclose(got["Charge"], d["Charge"])
    assert got.attributes["config_type"] == "bulk"


_MIXED_EXTXYZ = textwrap.dedent("""\
    3
    Lattice="4.0 0.0 0.0 0.5 4.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:forces:R:3:id:I:1:move_mask:L:1:tag:S:1:magmoms:R:1 energy=-12.5 stress="1 2 3 2 4 5 3 5 6" pbc="T T F" config_type=bulk note="two words"
    cu 0.1 0.2 0.3 1e-3 -2.5E+01 0.30000000000000004 1 T alpha 0.5
    Au 1.0 1.5 2.0 nan 0 -0 2 F beta-long-tag -1
    CU -0.25 3.75 4.99 1.0 2.0 3.0 3 True alpha 0

    2

    H 0 0 0
    H 0 0 0.74
    2
    Properties=species:S:1:pos:R:3:move_mask:L:3 cutoff=5
    O 0 0 0 T F T
    O 0 0 1.2 F F F
    1
    Properties=Z:I:1:species:S:1:pos:R:3
    26 Ni 0 0 0
    """)


def _same(a, b):
    assert list(a.particles) == list(b.particles)
    for name in a.particles:
        x, y = np.asarray(a.particles[name]), np.asarray(b.particles[name])
        assert x.dtype == y.dtype and x.shape == y.shape, name
        if x.dtype.kind == "f":
            np.testing.assert_array_equal(x, y, err_msg=name)   # bit-identical, NaN == NaN
        else:
            assert (x == y).all(), name
    assert a.attributes.keys() == b.attributes.keys()
    for key in a.attributes:
        np.testing.assert_array_equal(a.attributes[key], b.attributes[key], err_msg=key)
    np.testing.assert_array_equal(a.cell.matrix, b.cell.matrix)
    np.testing.assert_array_equal(a.cell.pbc, b.cell.pbc)


def test_extxyz_fast_parser_matches_ase(tmp_path):
    import io

    from ase.io.extxyz import read_xyz

    from ewald.io.ase_reader import _ovito_names
    from ewald.io.extxyz import ExtxyzReader, _read_frame

    path = tmp_path / "mixed.extxyz"
    path.write_text(_MIXED_EXTXYZ)
    reader = ExtxyzReader(path)
    assert len(reader) == 4
    for k, (offset, size) in enumerate(reader._frames):
        text = path.read_bytes()[offset:offset + size]
        fast = _read_frame(text)   # must not fall back to ASE for any of these frames
        ref = next(read_xyz(io.StringIO(text.decode()), 0))
        a, b = DataCollection.from_ase(fast), DataCollection.from_ase(ref)
        _ovito_names(a, fast), _ovito_names(b, ref)
        _same(a, b)
        assert reader.read(k).count == len(ref)
    assert list(reader.read(0)["Particle Type"]) == ["Cu", "Au", "Cu"]
    assert reader.read(0)["tag"].tolist() == ["alpha", "beta-long-tag", "alpha"]
    assert list(reader.read(3)["Particle Type"]) == ["Fe"]   # a Z column wins over species


def test_extxyz_frame_index_is_the_same_for_any_chunk_size(tmp_path):
    from ewald.io.extxyz import _scan_frames

    path = tmp_path / "md.extxyz"
    export_file(_trajectory(4), path)
    path.write_bytes(b"\n" + path.read_bytes().replace(b"\n8\n", b"\n\n\n8\n").rstrip(b"\n"))
    frames, complete = _scan_frames(path)
    assert complete and len(frames) == 4
    for chunk in (1, 7, 64, 1000):
        assert _scan_frames(path, chunk_size=chunk) == (frames, True)
    data = path.read_bytes()
    assert all(data[o:o + 2] == b"8\n" for o, _ in frames)


def test_xsf_grid_round_trip_matches_ase(tmp_path):
    from ase.io.xsf import read_xsf

    d = _nacl()
    grid = VoxelGrid(d.cell.copy())
    grid["Charge Density"] = np.random.default_rng(0).random((6, 5, 4))
    d.grids["charge"] = grid
    export_file(d, tmp_path / "rho.xsf")
    got = import_file(tmp_path / "rho.xsf").compute(0)
    np.testing.assert_allclose(got.grids["xsf"]["Charge Density"], grid["Charge Density"])
    np.testing.assert_allclose(got.cell.matrix, d.cell.matrix, atol=1e-8)
    with open(tmp_path / "rho.xsf") as fh:   # ASE sees XSF's general grid: end points repeated
        values, *_ = read_xsf(fh, read_data=True)
    assert values.shape == (7, 6, 5)
    np.testing.assert_allclose(values[:-1, :-1, :-1], grid["Charge Density"], atol=1e-7)


def test_xsf_molecule_with_forces(tmp_path):
    path = tmp_path / "h2o.xsf"
    path.write_text(textwrap.dedent("""\
        # Quantum ESPRESSO style molecule
        ATOMS
         8  0.0 0.0 0.119  0.1 0.0 0.0
         H  0.0 0.763 -0.477  0.0 0.0 0.0
         h  0.0 -0.763 -0.477  0.0 0.0 0.0
        # trailing comment
        """))
    got = import_file(path).compute(0)
    assert list(got.types) == ["O", "H", "H"] and not got.cell.is_valid
    assert got["Force"][0, 0] == pytest.approx(0.1)


def test_poscar_writer_groups_species(tmp_path):
    from ewald.io.ase_reader import AseReader

    d = _nacl()   # Na Cl Na Cl
    export_file(d, tmp_path / "POSCAR")
    lines = (tmp_path / "POSCAR").read_text().splitlines()
    assert lines[5].split() == ["Na", "Cl"] and lines[6].split() == ["2", "2"]
    assert lines[7].strip().lower().startswith("direct")
    got = AseReader(tmp_path / "POSCAR").read(0)
    assert sorted(got.types) == sorted(d.types)


def test_one_frame_formats_need_one_frame(tmp_path):
    pipe = import_file(_write(tmp_path / "md.extxyz", _trajectory(3)))
    with pytest.raises(ValueError, match="one frame"):
        export_file(pipe, tmp_path / "POSCAR")
    export_file(pipe, tmp_path / "CONTCAR", frames=[-1])
    assert (tmp_path / "CONTCAR").exists()


def _write(path, frames):
    export_file(frames, path)
    return path


def test_lammps_data_writer_round_trip(tmp_path):
    d = _trajectory(1)[0]
    d.particles["Velocity"] = np.ones((d.count, 3))
    export_file(d, tmp_path / "cu.data")
    text = (tmp_path / "cu.data").read_text()
    assert "xy xz yz" in text and "# Cu" in text and "Velocities" in text
    got = import_file(tmp_path / "cu.data").compute(0)
    _same_structure(got, d)
    np.testing.assert_allclose(np.linalg.norm(got["Velocity"], axis=1), np.sqrt(3))


def test_lammps_data_writer_charge_style(tmp_path):
    d = _nacl()
    d.particles["Charge"] = np.where(d.types == "Na", 1.0, -1.0)
    export_file(d, tmp_path / "nacl.data")
    assert "Atoms # charge" in (tmp_path / "nacl.data").read_text()
    got = import_file(tmp_path / "nacl.data").compute(0)
    np.testing.assert_allclose(got["Charge"], d["Charge"])


def test_ase_db_selection(tmp_path):
    frames = _trajectory(3)
    for k, f in enumerate(frames):
        f.attributes["relaxed"] = k % 2
    export_file(frames, tmp_path / "set.db")
    assert import_file(tmp_path / "set.db", selection="relaxed=1").num_frames == 1
    got = import_file(tmp_path / "set.db").compute(2)
    assert got.attributes["relaxed"] == 0 and got.attributes["Row Id"] == 3


def test_cif_is_listed_and_read(tmp_path):
    export_file(_nacl(), tmp_path / "nacl.cif")
    assert plugins.find_reader(tmp_path / "nacl.cif").name == "cif"
    assert import_file(tmp_path / "nacl.cif").compute(0).count == 4


def test_h5md_round_trip_and_mdanalysis_interop(tmp_path):
    pytest.importorskip("h5py")
    frames = _trajectory()
    for f in frames:
        f.particles["Velocity"] = np.full((8, 3), 0.5)
    path = tmp_path / "md.h5md"
    export_file(frames, path)
    assert plugins.find_reader(path).name == "h5md"
    pipe = import_file(path)
    assert pipe.num_frames == 4
    got = pipe.compute(2)
    np.testing.assert_allclose(got.positions, frames[2].positions)
    np.testing.assert_allclose(got.cell.matrix, frames[2].cell.matrix)
    np.testing.assert_allclose(got["Force"], frames[2]["Force"])
    assert list(got.types) == list(frames[2].types)
    assert got.attributes["Timestep"] == 20 and got.attributes["Energy"] == pytest.approx(-3.0)
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe.empty(8, trajectory=True)
    u.load_new(str(path), format="H5MD", convert_units=False)
    assert u.trajectory.n_frames == 4
    np.testing.assert_allclose(u.trajectory[3].positions, frames[3].positions, atol=1e-5)


def test_h5md_type_map_and_nm_units(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "lammps.h5"
    with h5py.File(path, "w") as f:   # what LAMMPS dump h5md writes, but in nm
        f.create_group("h5md")
        g = f.create_group("particles/all")
        box = g.create_group("box")
        box.attrs["boundary"] = np.array([b"periodic"] * 3)
        edges = box.create_group("edges")
        edges["value"] = np.full((2, 3), 1.0)
        edges["value"].attrs["unit"] = "nm"
        edges["step"] = np.array([0, 100])
        pos = g.create_group("position")
        pos["value"] = np.random.default_rng(0).random((2, 3, 3))
        pos["value"].attrs["unit"] = "nm"
        pos["step"] = np.array([0, 100])
        g["species"] = np.array([1, 2, 2])
    got = import_file(path, type_map={1: "O", 2: "H"}).compute(1)
    assert list(got.types) == ["O", "H", "H"]
    np.testing.assert_allclose(np.diag(got.cell.matrix), 10.0)
    assert got.attributes["Timestep"] == 100
    assert import_file(path).compute(0).types[0] == "1"


@pytest.mark.parametrize("ext", ["xtc", "trr", "dcd"])
def test_md_trajectories_via_mdanalysis(tmp_path, ext):
    pytest.importorskip("MDAnalysis")
    frames = _trajectory(3)
    path = tmp_path / f"run.{ext}"
    export_file(frames, path)
    pipe = import_file(path)
    assert pipe.num_frames == 3 and set(pipe.compute(0).types) == {"X"}   # no topology
    export_file(frames[0], tmp_path / "run.pdb", format="proteindatabank")
    pipe = import_file(path)   # same-stem topology found
    got = pipe.compute(2)
    assert set(got.types) == {"Cu"}
    _same_structure(got, frames[2], atol=2e-3)   # XTC stores 1e-3 nm precision


def test_gro_topology_keeps_two_letter_elements(tmp_path):
    # MDAnalysis guesses GRO types from names and cuts 'La'/'Zr' to 'L'/'Z' (kinisi's LLZO run)
    pytest.importorskip("MDAnalysis")
    gro = tmp_path / "llzo.gro"
    gro.write_text("LLZO\n4\n"
                   "    1UNK     La    1   0.100   0.100   0.100\n"
                   "    1UNK    Li1    2   0.300   0.100   0.100\n"
                   "    1UNK     Zr    3   0.100   0.300   0.100\n"
                   "    1UNK      O    4   0.100   0.100   0.300\n"
                   "   1.00000   1.00000   1.00000\n")
    assert list(import_file(gro).compute(0).types) == ["La", "Li", "Zr", "O"]


def test_writer_plugin_from_the_plugin_folder(tmp_path):
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "csv_writer.py").write_text(textwrap.dedent("""\
        from ewald.plugins import writer

        @writer("positions-csv", patterns=("*.pcsv",), description="positions as CSV")
        def write_csv(path, frames, precision=3):
            with open(path, "w") as fh:
                for k, data in enumerate(frames):
                    for t, p in zip(data.types, data.positions):
                        fh.write(f"{k},{t}," + ",".join(f"{x:.{precision}f}" for x in p) + "\\n")
        """))
    plugins.load_plugins(force=True)
    st = {w["name"]: w for w in plugins.status()["writers"]}
    assert st["positions-csv"]["origin"].endswith("csv_writer.py")
    export_file(_trajectory(2), tmp_path / "out.pcsv", precision=1)
    rows = (tmp_path / "out.pcsv").read_text().splitlines()
    assert len(rows) == 16 and rows[-1].startswith("1,Cu,")
    assert plugins.find_writer(tmp_path / "x.xtc").name == "md-trajectory"
    assert plugins.find_writer(tmp_path / "x.foo", "espresso-in").name == "ase"


def test_pipeline_export_streams_frames(tmp_path):
    src = _write(tmp_path / "md.extxyz", _trajectory(5))
    pipe = import_file(src)
    seen = []
    original = pipe.compute

    def compute(i, *a, **k):
        seen.append(i)
        return original(i, *a, **k)

    pipe.compute = compute

    def writer(path, frames):
        for k, _ in enumerate(frames):
            assert len(seen) == k + 1   # one frame computed per frame written
    plugins.register_writer("probe", writer, patterns=("*.probe",))
    export_file(pipe, tmp_path / "x.probe")
    assert seen == [0, 1, 2, 3, 4]


def test_gui_export_filter_lists_writers():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from ewald.gui.app import save_filters

    plugins.register_writer("probe", lambda path, frames: None, patterns=("*.probe",), description="probe")
    filters = dict(save_filters())
    assert filters["probe (*.probe)"] == "probe"
    assert any(name == "poscar" and "POSCAR*" in text for text, name in filters.items())
    assert list(filters.values())[-1] is None   # "All files": format from the name


def test_compressed_output_is_compressed(tmp_path):
    import gzip

    export_file(_trajectory(2), tmp_path / "md.extxyz.gz")
    with gzip.open(tmp_path / "md.extxyz.gz", "rt") as fh:
        assert fh.readline().strip() == "8"
    assert import_file(tmp_path / "md.extxyz.gz").num_frames == 2
    assert [p.name for p in tmp_path.iterdir()] == ["md.extxyz.gz"]   # no temporary file left


def test_claude_export_tool_refuses_to_overwrite(tmp_path):
    from ewald.llm.tools import Toolset

    src = _write(tmp_path / "md.extxyz", _trajectory(2))
    ts = Toolset(workdir=str(tmp_path / "work"))
    ts.call("load_structure", {"path": str(src)})
    assert not ts.call("export_file", {"path": "POSCAR"}).is_error
    assert ts.call("export_file", {"path": "POSCAR"}).is_error
    assert ts.call("export_file", {"path": str(src)}).is_error            # absolute path, existing file
    assert not ts.call("export_file", {"path": "POSCAR", "overwrite": True}).is_error
