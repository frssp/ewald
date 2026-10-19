"""VESTA-style boundary display (ewald.boundary): ghost atoms, completed bonds and polyhedra."""
import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource
from ewald import scene as ir
from ewald.boundary import expand, normalize_boundary
from ewald.data import Cell
from ewald.modifiers import CoordinationPolyhedra, CreateBonds

A = 3.905
TIO = [("Ti", "O", 0.0, 2.2)]


def perovskite(pbc=(True, True, True)):
    """SrTiO3 with Ti on the corner: in 0..1 there are 8 Ti, 12 O (edge centers) and 1 Sr."""
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    return DataCollection(frac * A, ["Ti", "Sr", "O", "O", "O"], Cell(np.eye(3) * A, pbc=np.array(pbc)))


def counts(scene):
    return {lay.name: lay.primitive.count if lay.kind != "mesh" else lay.primitive.face_count for lay in scene.layers}


def test_normalize_boundary_forms():
    assert normalize_boundary(None) is None and normalize_boundary(False) is None
    assert np.allclose(normalize_boundary(True), [[0, 1]] * 3)
    assert np.allclose(normalize_boundary(0.1), [[-0.1, 1.1]] * 3)
    assert np.allclose(normalize_boundary((0, 2)), [[0, 2]] * 3)
    assert np.allclose(normalize_boundary([(0, 1), (0, 2), (0, 1)])[1], [0, 2])
    with pytest.raises(ValueError):
        normalize_boundary((1, 0))
    with pytest.raises(ValueError):
        normalize_boundary([1, 2, 3])


def test_ghosts_on_faces_edges_corners():
    d = perovskite()
    disp, src = expand(d, (0, 1))
    assert disp.count == 21 and d.count == 5          # the real data is untouched
    assert np.array_equal(src[:5], np.arange(5))      # real atoms first, unchanged
    assert np.array_equal(np.bincount(src), [8, 1, 4, 4, 4])
    assert list(np.unique(disp.types[src == 0])) == ["Ti"]
    f = disp.cell.to_fractional(disp.positions)
    assert f.min() > -1e-9 and f.max() < 1 + 1e-9
    # positions are lattice translations of the real atom
    shift = (disp.positions - d.positions[src]) @ np.linalg.inv(d.cell.matrix)
    assert np.allclose(shift, np.rint(shift))
    # a larger range gets more images; a margin form works too
    assert expand(d, (-0.6, 1.6))[0].count > 21
    assert expand(d, 0.0)[0].count == 21


def test_non_periodic_axes_get_no_images():
    disp, src = expand(perovskite((True, True, False)), (0, 1))
    assert np.bincount(src)[0] == 4                   # Ti only on the four in-plane corners


@pytest.mark.parametrize("mode, atoms, cylinders", [("half", 21, 72), ("complete", 45, 96), ("none", 21, 48)])
def test_bonds_across_the_cell(mode, atoms, cylinders):
    sc = ir.build(perovskite(), bond_pairs=TIO, boundary=(0, 1), boundary_bonds=mode)
    c = counts(sc)
    # 24 whole Ti-O bonds on the 12 edges; each corner Ti has 3 more bonds leaving the cell
    assert c["atoms"] == atoms and c["bonds"] == cylinders
    lengths = np.linalg.norm(sc["bonds"].primitive.end - sc["bonds"].primitive.start, axis=1)
    assert np.allclose(lengths, A / 4)                # every drawn piece is a half bond of a real bond


def test_without_boundary_nothing_changes():
    d = perovskite()
    assert counts(ir.build(d, bond_pairs=TIO)) == counts(ir.build(d, bond_pairs=TIO, boundary=None))
    assert counts(ir.build(d, bond_pairs=TIO))["atoms"] == 5


def test_ghosts_pick_as_their_real_atom():
    sc = ir.build(perovskite(), bond_pairs=TIO, boundary=(0, 1))
    spheres = sc["atoms"].primitive
    assert spheres.count == 21 and set(spheres.index) == set(range(5))
    assert np.array_equal(np.bincount(spheres.index), [8, 1, 4, 4, 4])
    assert sc["bonds"].primitive.owner.max() < 5


def test_polyhedra_completed_and_analysis_untouched():
    pipe = Pipeline(StaticSource(perovskite()), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]),
                                                  CoordinationPolyhedra(["Ti"])])
    data = pipe.compute(0)
    assert data.count == 5 and data.attributes["CoordinationPolyhedra.count"] == 1
    sc = ir.build(data, boundary=(0, 1))
    c = counts(sc)
    assert c["surface:coordination-polyhedra"] == 8 * 8   # an octahedron on every corner Ti
    assert c["atoms"] == 45                               # with all their O corners, outside the cell too
    mesh = sc["surface:coordination-polyhedra"].primitive
    assert mesh.edges is not None and len(mesh.edges) == 8 * 12
    assert np.array_equal(np.unique(mesh.face_properties["Region"]), [0])
    # the pipeline result is unchanged
    assert pipe.compute(0).count == 5 and data.surfaces["coordination-polyhedra"].face_count == 8


def test_transparency_and_coarse_mask_follow_the_real_atom():
    d = perovskite()
    d["Transparency"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])   # hide Sr
    sc = ir.build(d, boundary=(0, 1))
    assert 1 not in sc["atoms"].primitive.index
    ctx = ir.SceneContext(only_visible=np.array([True, False, False, False, False]))
    sc = ir.build(perovskite(), ctx=ctx, boundary=(0, 1))
    assert sc["atoms"].primitive.count == 8


def test_gui_style_round_trip():
    from ewald.gui.style import ViewStyle

    style = ViewStyle()
    assert "boundary" not in style.scene_kwargs()
    style.boundary, style.boundary_bonds = [-0.1, 1.1], "complete"
    kw = style.scene_kwargs()
    assert kw["boundary"] == [-0.1, 1.1] and kw["boundary_bonds"] == "complete"
    assert counts(ir.build(perovskite(), bond_pairs=TIO, **{k: kw[k] for k in ("boundary", "boundary_bonds")}))
