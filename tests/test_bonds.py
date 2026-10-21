import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from ewald import DataCollection, Pipeline, StaticSource
from ewald.bonds import Bonds, _half_pairs_small_cell, find_bonds
from ewald.data import Cell
from ewald.modifiers import (CoordinationPolyhedra, CreateBonds, DeleteSelected, EditBonds, ExpressionSelection,
                             Replicate, SelectType, WrapAtoms)


@pytest.fixture
def srtio3():
    a = 3.905
    atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                    cellpar=[a, a, a, 90, 90, 90])
    return DataCollection.from_ase(atoms)


def run(data, *modifiers):
    return Pipeline(StaticSource(data), list(modifiers)).compute(0)


def test_uniform_bonds_fcc(cu_fcc):
    d = run(cu_fcc, CreateBonds(cutoff=3.0))
    assert d.bonds.count == 6 * cu_fcc.count                 # 12 neighbors, each bond once
    assert d.attributes["CreateBonds.num_bonds"] == d.bonds.count
    np.testing.assert_allclose(d.bonds.lengths(d), 3.615 / np.sqrt(2))
    assert (d.bonds.neighbor_counts(d.count) == 12).all()
    assert "bonds: " in d.summary()
    assert cu_fcc.bonds is None                              # source untouched


def test_small_cell_uses_periodic_images():
    si = DataCollection.from_ase(bulk("Si"))                 # 2-atom primitive cell, thinner than 2 x cutoff
    b = find_bonds(si, cutoff=2.5)
    assert b.count == 4
    np.testing.assert_allclose(b.lengths(si), 5.43 * np.sqrt(3) / 4)
    sup = DataCollection.from_ase(bulk("Si") * (4, 4, 4))
    i, j, img, dist = _half_pairs_small_cell(sup.positions, sup.cell, 2.5)
    assert len(i) == find_bonds(sup, cutoff=2.5).count == 2 * sup.count


def test_replicate_wrap_delete_keep_bonds_consistent():
    si = DataCollection.from_ase(bulk("Si"))
    d = run(si, CreateBonds(cutoff=2.5), Replicate((3, 3, 3)))
    assert d.bonds.count == 4 * 27
    np.testing.assert_allclose(d.bonds.lengths(d), 2.3513, atol=1e-4)
    assert (d.bonds.neighbor_counts(d.count) == 4).all()
    shifted = d.copy()
    shifted.positions = d.positions + 7.3
    wrapped = run(shifted, WrapAtoms())
    np.testing.assert_allclose(wrapped.bonds.lengths(wrapped), 2.3513, atol=1e-4)
    cut = run(d, ExpressionSelection("Position.Z < 3"), DeleteSelected())
    t = cut.bonds.topology
    assert t.max() < cut.count and cut.bonds.count < d.bonds.count
    np.testing.assert_allclose(cut.bonds.lengths(cut), 2.3513, atol=1e-4)


def test_pairwise_and_covalent_modes(srtio3):
    d = run(srtio3, CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]))
    assert d.bonds.count == 6
    assert set(d.types[d.bonds.topology].ravel()) == {"Ti", "O"}
    m = CreateBonds()
    m.set_pairwise_cutoff("O", "Ti", 2.2)
    assert m.mode == "pairwise" and run(srtio3, m).bonds.count == 6
    wild = run(srtio3, CreateBonds(mode="pairwise", pairs=[("*", "O", 0.0, 2.2)]))
    assert wild.bonds.count == 6
    cov = run(srtio3, CreateBonds(mode="covalent"))
    pairs = {tuple(sorted(p)) for p in cov.types[cov.bonds.topology]}
    assert ("O", "Ti") in pairs and ("O", "O") not in pairs
    assert not pairs & {("Sr", "Ti"), ("Sr", "Sr"), ("Ti", "Ti")}  # no metal-metal bonds in a compound
    alloy = run(srtio3, ExpressionSelection("ParticleType == 'O'"), DeleteSelected(), CreateBonds(mode="covalent"))
    assert {tuple(sorted(p)) for p in alloy.types[alloy.bonds.topology]} >= {("Sr", "Ti")}  # an alloy keeps them
    with pytest.raises(ValueError):
        CreateBonds(mode="pairwise").modify(srtio3.copy(), None)


def test_edit_bonds_add_remove_toggle(cu_fcc):
    data = cu_fcc.copy()
    data["Particle Identifier"] = np.arange(1, data.count + 1)
    base = run(data, CreateBonds(cutoff=3.0))
    i, j = base.bonds.topology[0]
    a, b = int(i) + 1, int(j) + 1
    far = (1, 1 + int(np.argmax(np.linalg.norm(data.positions - data.positions[0], axis=1))))
    edit = EditBonds(remove=[(b, a)], add=[far])
    d = run(data, CreateBonds(cutoff=3.0), edit)
    assert d.bonds.count == base.bonds.count
    ids = d["Particle Identifier"][d.bonds.topology]
    assert not any(set(p) == {a, b} for p in ids.tolist()) and any(set(p) == set(far) for p in ids.tolist())
    assert d.bonds.lengths(d)[-1] < 0.9 * np.linalg.norm(data.cell.matrix.sum(0))   # shortest image
    edit.toggle(a, b, bonded=False)                          # re-adding undoes the deletion
    assert edit.remove == [] and edit.add == [far]
    assert run(data, CreateBonds(cutoff=3.0), edit).bonds.count == base.bonds.count + 1
    edit.toggle(*far, bonded=True)
    assert edit.add == edit.remove == []
    assert run(data, CreateBonds(cutoff=3.0), EditBonds(add=[(a, b)])).bonds.count == base.bonds.count  # no duplicate


def _volume(mesh):
    v, f = mesh.vertices, mesh.faces
    return np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6


def test_octahedra(srtio3):
    d = run(srtio3, CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]), CoordinationPolyhedra(["Ti"]))
    mesh = d.surfaces["coordination-polyhedra"]
    assert d.attributes["CoordinationPolyhedra.count"] == 1
    assert mesh.face_count == 8 and mesh.vertex_count == 6 and len(mesh.edges) == 12
    r = 3.905 / 2
    assert _volume(mesh) == pytest.approx(4 / 3 * r ** 3)     # outward-oriented faces
    assert set(mesh.face_properties["Region"]) == {1}
    # selection as centers, no edges
    d2 = run(srtio3, CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]), SelectType("Ti"),
             CoordinationPolyhedra(edges=False))
    assert d2.surfaces["coordination-polyhedra"].edges is None
    # no bonds upstream (the GUI's bonds are display only): covalent bonds, Ti-O but not Ti-Sr
    alone = srtio3.copy()
    CoordinationPolyhedra(["Ti"]).modify(alone, None)
    assert alone.bonds is None and alone.surfaces["coordination-polyhedra"].vertex_count == 6
    with pytest.raises(ValueError, match="center"):
        CoordinationPolyhedra().modify(d.copy(), None)


def test_square_planar_and_cube():
    sq = DataCollection(np.array([[5, 5, 5], [7, 5, 5], [3, 5, 5], [5, 7, 5], [5, 3, 5.0]]),
                        np.array(["Cu", "O", "O", "O", "O"]), Cell(np.diag([10.0] * 3)))
    mesh = run(sq, CreateBonds(cutoff=2.1), CoordinationPolyhedra(["Cu"])).surfaces["coordination-polyhedra"]
    assert mesh.face_count == 2 and len(mesh.edges) == 4
    corners = np.array([[x, y, z] for x in (4, 6) for y in (4, 6) for z in (4, 6)], float)
    cube = DataCollection(np.vstack([[5, 5, 5], corners]), np.array(["Cs"] + ["Cl"] * 8), Cell(np.diag([10.0] * 3)))
    mesh = run(cube, CreateBonds(cutoff=1.8), CoordinationPolyhedra(["Cs"])).surfaces["coordination-polyhedra"]
    assert mesh.face_count == 12 and len(mesh.edges) == 12   # square-face diagonals are not outlined
    assert _volume(mesh) == pytest.approx(8.0)


def test_polyhedra_across_periodic_boundary(srtio3):
    shifted = srtio3.copy()
    shifted.positions = srtio3.positions + np.array([1.9, 1.9, 1.9])   # Ti near the cell corner
    d = run(shifted, WrapAtoms(), CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]), CoordinationPolyhedra(["Ti"]))
    mesh = d.surfaces["coordination-polyhedra"]
    assert _volume(mesh) == pytest.approx(4 / 3 * (3.905 / 2) ** 3)


def test_bonds_container():
    b = Bonds([[0, 1], [1, 2]], bond_type=["a", "b"])
    assert list(b.properties) == ["Topology", "Periodic Image", "Bond Type"]
    c = b.extend(Bonds([[2, 3]]))
    assert c.count == 3 and c["Bond Type"][-1] == ""
    r = c.remap_particles(np.array([True, False, True, True]), 4)
    np.testing.assert_array_equal(r.topology, [[1, 2]])
    with pytest.raises(ValueError):
        b["Width"] = [1.0, 2.0, 3.0]


def test_lammps_data_bonds(tmp_path):
    from ewald import import_file

    path = tmp_path / "water.data"
    path.write_text("x\n\n3 atoms\n2 bonds\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
                    "Atoms # molecular\n\n3 1 2 1 0 0\n1 1 1 0.5 0 0\n2 1 2 9.5 1 0\n\nBonds\n\n1 1 1 2\n2 1 1 3\n")
    d = import_file(path).compute(0)
    ids = d["Particle Identifier"][d.bonds.topology]
    np.testing.assert_array_equal(ids, [[1, 2], [1, 3]])
    np.testing.assert_allclose(d.bonds.lengths(d), [np.hypot(1, 1), 0.5])   # bond 1-2 crosses x boundary


def test_tube_mesh_is_an_open_cylinder_per_segment():
    """Bond sticks are built in numpy (vtkTubeFilter was most of a frame's build time during playback)."""
    from ewald.render import TUBE_SIDES, tube_mesh

    rng = np.random.default_rng(1)
    start = rng.normal(size=(50, 3))
    end = start + rng.normal(size=(50, 3))
    end[0] = start[0] + [0, 0, 2.0]       # along an axis
    end[1] = start[1]                     # zero length: still a valid (degenerate) ring
    verts, faces, normals = tube_mesh(start, end, 0.2)
    k, n = len(start), TUBE_SIDES
    assert verts.shape == normals.shape == (2 * n * k, 3) and faces.shape == (2 * n * k, 3)
    assert faces.min() == 0 and faces.max() == 2 * n * k - 1 and np.isfinite(verts).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    seg = np.repeat(np.arange(k), 2 * n)
    axis = end - start
    t = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    ok = np.linalg.norm(axis, axis=1)[seg] > 0
    assert np.allclose(np.einsum("ij,ij->i", normals, t[seg])[ok], 0, atol=1e-12)   # normals point outward
    ring = verts.reshape(k, 2, n, 3)
    assert np.allclose(ring[:, 0] - 0.2 * normals.reshape(k, 2, n, 3)[:, 0], start[:, None])
    assert np.allclose(ring[:, 1] - 0.2 * normals.reshape(k, 2, n, 3)[:, 1], end[:, None])
    assert (faces // (2 * n) == np.repeat(np.arange(k), 2 * n)[:, None]).all()   # no face spans two bonds
