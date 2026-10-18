import numpy as np
from ase.build import bulk
import ase.io

from ewald import DataCollection, export_file, import_file
from ewald.data import Cell
from ewald.io.lammps_dump import write_lammps_dump


def _frames(n=3, triclinic=False):
    rng = np.random.default_rng(1)
    m = np.array([[10.0, 0, 0], [2.0 if triclinic else 0, 9.0, 0], [1.0 if triclinic else 0, -1.5 if triclinic else 0, 8.0]])
    out = []
    for k in range(n):
        d = DataCollection(rng.random((5, 3)) @ m, ["Cu", "Cu", "Zr", "Cu", "Zr"], Cell(m, [1.0, -2.0, 0.5]))
        d.particles["Particle Identifier"] = np.array([5, 3, 1, 2, 4])
        d.attributes["Timestep"] = 100 * k
        out.append(d)
    return out


def test_lammps_dump_roundtrip_triclinic(tmp_path):
    frames = _frames(triclinic=True)
    path = tmp_path / "traj.lammpstrj"
    write_lammps_dump(path, frames, columns=("id", "type", "element", "x", "y", "z"))
    pipe = import_file(path)
    assert pipe.num_frames == 3
    d = pipe.compute(2)
    assert d.attributes["Timestep"] == 200
    np.testing.assert_allclose(d.cell.matrix, frames[2].cell.matrix)
    np.testing.assert_allclose(d.cell.origin, frames[2].cell.origin)
    order = np.argsort(frames[2]["Particle Identifier"])
    np.testing.assert_allclose(d.positions, frames[2].positions[order])
    assert list(d.types) == list(frames[2].types[order])


def test_lammps_dump_type_map_and_scaled(tmp_path):
    path = tmp_path / "dump.atom"
    path.write_text("ITEM: TIMESTEP\n7\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp ff\n"
                    "0 10\n0 10\n0 20\nITEM: ATOMS id type xs ys zs c_pe\n"
                    "2 2 0.5 0.5 0.5 -3.5\n1 1 0.1 0.2 0.3 -3.25\n")
    d = import_file(path, type_map={1: "Cu", 2: "Zr"}).compute(0)
    assert list(d.types) == ["Cu", "Zr"]
    np.testing.assert_allclose(d.positions[0], [1, 2, 6])
    np.testing.assert_allclose(d["Potential Energy"], [-3.25, -3.5])
    assert list(d.cell.pbc) == [True, True, False]


def test_xdatcar_variable_cell(tmp_path):
    def header(a):
        return f"test\n1.0\n{a} 0 0\n0 {a} 0\n0 0 {a}\nSi O\n1 2\n"
    text = header(5.0) + "Direct configuration=     1\n0 0 0\n0.5 0.5 0.5\n0.25 0.25 0.25\n"
    text += header(6.0) + "Direct configuration=     2\n0.1 0 0\n0.5 0.5 0.5\n0.25 0.25 0.25\n"
    path = tmp_path / "XDATCAR"
    path.write_text(text)
    pipe = import_file(path)
    assert pipe.num_frames == 2
    d = pipe.compute(1)
    np.testing.assert_allclose(d.cell.matrix, np.eye(3) * 6.0)
    np.testing.assert_allclose(d.positions[0], [0.6, 0, 0])
    assert list(d.types) == ["Si", "O", "O"]
    assert d.attributes["Timestep"] == 2


def test_xdatcar_fixed_cell_from_ase(tmp_path):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    images = []
    for k in range(4):
        a = atoms.copy()
        a.positions[0] += 0.01 * k
        images.append(a)
    path = tmp_path / "XDATCAR"
    ase.io.write(path, images, format="vasp-xdatcar")
    pipe = import_file(path)
    assert pipe.num_frames == 4
    np.testing.assert_allclose(pipe.compute(3).positions, images[3].positions, atol=1e-6)


def test_poscar_via_ase_and_export(tmp_path):
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    ase.io.write(tmp_path / "POSCAR", atoms, format="vasp")
    pipe = import_file(tmp_path / "POSCAR")
    d = pipe.compute(0)
    assert sorted(set(d.types)) == ["Cl", "Na"]
    export_file(pipe, tmp_path / "out.extxyz", format="extxyz")
    back = ase.io.read(tmp_path / "out.extxyz")
    np.testing.assert_allclose(back.positions, atoms.positions)


def test_wildcard_import(tmp_path):
    for k, d in enumerate(_frames(2)):
        write_lammps_dump(tmp_path / f"dump.{k}.lammpstrj", [d], columns=("id", "element", "x", "y", "z"))
    assert import_file(tmp_path / "dump.*.lammpstrj").num_frames == 2
