"""LAMMPS dump/data readers checked against real LAMMPS output (tests/data/lammps/generate.in).

The fixtures cross-check each other: each data file was written at the same
timestep as the last frame of a dump, so both readers must agree.
"""
import gzip

import numpy as np
import pytest

from ewald import import_file
from ewald.io import LammpsDataReader, LammpsDumpReader, open_reader


def _wrapped_equal(a, b, cell, atol=1e-5):
    """Positions equal up to lattice translations."""
    df = cell.to_fractional(a) - cell.to_fractional(b)
    np.testing.assert_allclose(df - np.round(df), 0, atol=atol)


def test_dump_custom_units_time_element_vectors(lammps_dir):
    pipe = import_file(lammps_dir / "dump.custom.lammpstrj")
    assert pipe.num_frames == 3
    d = pipe.compute(2)
    assert d.attributes["Timestep"] == 20
    assert d.attributes["Units"] == "metal"
    assert d.attributes["Time"] == pytest.approx(0.04)
    assert set(d.types) == {"Cu", "Zr"}
    np.testing.assert_array_equal(d["Particle Identifier"], np.arange(1, 33))  # sorted by id
    assert d["Velocity"].shape == (32, 3) and d["Force"].shape == (32, 3)
    assert d["c_st"].shape == (32, 3)  # c_st[1] c_st[2] c_st[3] -> one vector property
    assert d["Potential Energy"].dtype.kind == "f"
    np.testing.assert_allclose(np.unique(d["Charge"]), [-0.3, 0.1])


def test_dump_agrees_with_data_file(lammps_dir):
    dump = import_file(lammps_dir / "dump.custom.lammpstrj").compute(2)
    data = import_file(lammps_dir / "data.full.lmp").compute(0)
    _wrapped_equal(dump.positions, data.positions, data.cell)
    np.testing.assert_allclose(dump["Velocity"], data["Velocity"], rtol=1e-6)
    assert list(dump.types) == list(data.types)


def test_dump_atom_scaled_with_images_gzip(lammps_dir):
    plain = import_file(lammps_dir / "dump.atom")
    gz = import_file(lammps_dir / "dump.atom.gz")
    assert plain.num_frames == gz.num_frames == 3
    a, b = plain.compute(2), gz.compute(2)
    np.testing.assert_allclose(a.positions, b.positions)
    assert a["Periodic Image"].dtype.kind == "i"
    ref = import_file(lammps_dir / "data.full.lmp").compute(0)
    _wrapped_equal(a.positions, ref.positions, ref.cell)


def test_dump_restricted_triclinic_scaled_and_unwrapped(lammps_dir):
    pipe = import_file(lammps_dir / "dump.triclinic.lammpstrj")
    d = pipe.compute(-1)
    ref = import_file(lammps_dir / "data.triclinic.lmp").compute(0)
    assert d.attributes["Timestep"] == 40
    np.testing.assert_allclose(d.cell.matrix, ref.cell.matrix, atol=1e-6)
    np.testing.assert_allclose(d.cell.origin, ref.cell.origin, atol=1e-6)
    # 'xs' wins over 'xu' for Position; both describe the same atoms modulo the cell
    _wrapped_equal(d.positions, ref.positions, ref.cell, atol=1e-4)
    assert "Position.unwrapped" not in d and "Position.reduced" not in d


def test_dump_general_triclinic(lammps_dir):
    d = import_file(lammps_dir / "dump.general.lammpstrj").compute(0)
    ref = import_file(lammps_dir / "data.general.lmp").compute(0)
    np.testing.assert_allclose(d.cell.matrix, [[4, 0, 0], [1, 4, 0], [0, 1, 4]])
    np.testing.assert_allclose(d.cell.origin, [-1, -1, -1])
    np.testing.assert_allclose(d.positions, ref.positions)
    np.testing.assert_allclose(ref.cell.matrix, d.cell.matrix)


def test_dump_truncated_last_frame(lammps_dir, tmp_path):
    lines = (lammps_dir / "dump.custom.lammpstrj").read_bytes().splitlines(keepends=True)
    path = tmp_path / "killed.lammpstrj"
    path.write_bytes(b"".join(lines[:-5]))
    with pytest.warns(UserWarning, match="incomplete"):
        pipe = import_file(path)
    assert pipe.num_frames == 2
    # a long compressed run cut short: only the tail is lost, earlier frames stay readable
    gz = tmp_path / "killed.lammpstrj.gz"
    gz.write_bytes(gzip.compress((lammps_dir / "dump.custom.lammpstrj").read_bytes() * 50)[:-200])
    with pytest.warns(UserWarning, match="incomplete"):
        pipe = import_file(gz)
    assert 140 <= pipe.num_frames < 150
    assert pipe.compute(pipe.num_frames - 1).count == 32


def test_dump_scan_independent_of_chunk_boundaries(lammps_dir, tmp_path):
    from ewald.io.lammps_dump import _scan_frames

    for name in ("dump.custom.lammpstrj", "dump.atom", "dump.triclinic.lammpstrj"):
        path = lammps_dir / name
        ref = _scan_frames(path)
        assert ref[1] and len(ref[0]) >= 2
        for chunk in (1, 5, 7, 64, 1000):  # boundaries inside ITEM lines and the atom count
            assert _scan_frames(path, chunk_size=chunk) == ref
    # a missing final newline still counts the last atom line, a cut-off one does not
    raw = (lammps_dir / "dump.atom").read_bytes()
    (tmp_path / "a.dump").write_bytes(raw.rstrip(b"\n"))
    assert _scan_frames(tmp_path / "a.dump", chunk_size=7)[1]
    (tmp_path / "b.dump").write_bytes(raw.rstrip(b"\n")[:-12])
    assert not _scan_frames(tmp_path / "b.dump", chunk_size=7)[1]


def test_dump_type_labels_are_compact(tmp_path):
    path = tmp_path / "t.dump"
    path.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n"
                    "0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n"
                    "4 12 0 0 0\n3 2 0 0 0\n2 1 0 0 0\n1 2 0 0 0\n")
    d = import_file(path).compute(0)
    assert d.types.dtype == np.dtype("<U2") and list(d.types) == ["2", "1", "2", "12"]
    np.testing.assert_array_equal(d["Type Index"], [2, 1, 2, 12])
    d = import_file(path, type_map={2: "Cu", 12: "Zr"}).compute(0)
    assert list(d.types) == ["Cu", "1", "Cu", "Zr"]


def test_dump_empty_frame(tmp_path):
    path = tmp_path / "empty.dump"
    path.write_text("ITEM: TIMESTEP\n5\nITEM: NUMBER OF ATOMS\n0\nITEM: BOX BOUNDS pp pp pp\n"
                    "0 1\n0 1\n0 1\nITEM: ATOMS id type x y z\n")
    d = import_file(path).compute(0)
    assert d.count == 0 and d.attributes["Timestep"] == 5


def test_data_full_labels_velocities_images(lammps_dir):
    reader = open_reader(lammps_dir / "data.full.lmp")
    assert isinstance(reader, LammpsDataReader)
    d = reader.read(0)
    assert d.attributes["AtomStyle"] == "full"
    assert d.count == 32 and set(d.types) == {"Cu", "Zr"}
    np.testing.assert_allclose(np.unique(d["Mass"]), [63.546, 91.224])
    assert d["Molecule Identifier"].dtype.kind == "i"
    assert d["Periodic Image"].shape == (32, 3)
    assert np.abs(d["Velocity"]).max() > 0


def test_data_without_labels_uses_numbers_or_type_map(lammps_dir):
    assert set(import_file(lammps_dir / "data.nolabel.lmp").compute(0).types) == {"1", "2"}
    d = import_file(lammps_dir / "data.nolabel.lmp", type_map={1: "Cu", 2: "Zr"}).compute(0)
    ref = import_file(lammps_dir / "data.full.lmp").compute(0)
    assert list(d.types) == list(ref.types)


def test_data_style_inference_mass_comments_bonds(lammps_dir):
    d = import_file(lammps_dir / "data.charge.nocomment.lmp").compute(0)
    assert d.attributes["AtomStyle"] == "charge"
    assert list(d.types) == ["O", "H", "H"]
    np.testing.assert_allclose(d["Charge"], [-0.8476, 0.4238, 0.4238])
    np.testing.assert_allclose(d.cell.origin, [-5, -5, -5])
    np.testing.assert_array_equal(d.tables["Bonds"]["Topology"], [[0, 1], [0, 2]])


def test_data_sorting_remaps_bonds(tmp_path):
    path = tmp_path / "water.data"
    path.write_text("x\n\n3 atoms\n2 bonds\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
                    "Atoms # molecular\n\n3 1 2 1 0 0\n1 1 1 0 0 0\n2 1 2 0 1 0\n\nBonds\n\n1 1 1 2\n2 1 1 3\n")
    d = import_file(path).compute(0)
    np.testing.assert_array_equal(d["Particle Identifier"], [1, 2, 3])
    ids = d["Particle Identifier"][d.tables["Bonds"]["Topology"]]
    np.testing.assert_array_equal(ids, [[1, 2], [1, 3]])


def test_data_explicit_style_and_errors(tmp_path):
    path = tmp_path / "a.lmp"
    path.write_text("x\n\n1 atoms\n1 atom types\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
                    "Atoms\n\n1 1 2.0 1.0 0.5 0.5 0.5\n")
    d = import_file(path, atom_style="sphere").compute(0)
    np.testing.assert_allclose(d.positions, [[0.5, 0.5, 0.5]])
    assert d["Diameter"][0] == 2.0
    bad = tmp_path / "b.lmp"
    bad.write_text("x\n\n2 atoms\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\nAtoms # atomic\n\n1 1 0 0 0\n")
    with pytest.raises(ValueError, match="header says 2 atoms"):
        import_file(bad).compute(0)


def test_dump_reader_is_selected_for_units_header(lammps_dir):
    assert isinstance(open_reader(lammps_dir / "dump.custom.lammpstrj"), LammpsDumpReader)


def test_fast_table_parser_matches_loadtxt():
    import io

    from ewald.io._textparse import parse_table

    rng = np.random.default_rng(3)
    vals = rng.standard_normal((700, 4)) * 10.0 ** rng.integers(-30, 30, (700, 4))
    fmts = ["%.6f", "%g", "%.15g", "%.17g", "%.8e", "%.3E", "%.1f"]
    lines = [" ".join(fmts[i % len(fmts)] % v for v in row) for i, row in enumerate(vals)]
    lines[:4] = ["1 2 inf -nan", "  +1.5e+3\t-0.0 00012 .5", "1. -2 3e-5 4E+2", "7 8 9 10\r"]
    block = ("\n".join(lines) + "\n\n").encode()
    ref = np.loadtxt(io.BytesIO(block), ndmin=2)
    out = parse_table(block, 700, 4, [0, 1, 2, 3])
    np.testing.assert_array_equal(out, ref)  # bit-identical, NaN and -0.0 included
    np.testing.assert_array_equal(np.signbit(out), np.signbit(ref))
    np.testing.assert_array_equal(parse_table(block, 700, 4, [3, 1]), ref[:, [3, 1]])
    with pytest.raises(ValueError):
        parse_table(block, 701, 4, [0])
    with pytest.raises(ValueError):
        parse_table(b"1 2\n3\n", 2, 2, [0, 1])
