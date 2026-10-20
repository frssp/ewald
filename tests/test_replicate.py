"""Replicated cell display (ewald.replicate, ``replicate=`` of scene.build/render/view): display only."""
import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource, view
from ewald import scene as ir
from ewald.data import Cell
from ewald.modifiers import CoordinationPolyhedra, CreateBonds, Replicate
from ewald.replicate import cell_edges, normalize_replicate, tile

A = 3.905
TIO = [("Ti", "O", 0.0, 2.2)]


def perovskite():
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    return DataCollection(frac * A, ["Ti", "Sr", "O", "O", "O"], Cell(np.eye(3) * A))


def counts(scene):
    return {lay.name: lay.primitive.count if lay.kind != "mesh" else lay.primitive.face_count for lay in scene.layers}


def test_normalize_replicate():
    assert normalize_replicate(None) is None and normalize_replicate(1) is None
    assert normalize_replicate((1, 1, 1)) is None and normalize_replicate(False) is None
    assert normalize_replicate(2) == (2, 2, 2) and normalize_replicate([3, 3, 1]) == (3, 3, 1)
    for bad in (0, (2, 2), (1.5, 1, 1), (-1, 2, 2)):
        with pytest.raises(ValueError):
            normalize_replicate(bad)


def test_tile_matches_replicate_modifier_and_keeps_data():
    d = perovskite()
    real = Pipeline(StaticSource(d), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]),
                                      Replicate((2, 3, 1))]).compute(0)
    bonded = Pipeline(StaticSource(d), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)])]).compute(0)
    disp, src = tile(bonded, (2, 3, 1))
    assert bonded.count == 5 and disp.count == 30            # the data itself is untouched
    assert np.allclose(disp.positions, real.positions)
    assert np.array_equal(src, np.tile(np.arange(5), 6))
    assert np.array_equal(disp.bonds.topology, real.bonds.topology)
    assert np.array_equal(disp.bonds.pbc_images, real.bonds.pbc_images)
    assert np.allclose(disp.cell.matrix, real.cell.matrix)
    assert tile(bonded, 1) == (bonded, None)


def test_scene_draws_copies_that_pick_as_real_atoms():
    d = perovskite()
    one, rep = ir.build(d, bond_pairs=TIO), ir.build(d, bond_pairs=TIO, replicate=(2, 2, 2))
    assert counts(rep)["atoms"] == 8 * counts(one)["atoms"]
    assert counts(rep)["bonds"] == 8 * counts(one)["bonds"]
    atoms = rep["atoms"].primitive
    assert atoms.index.max() == 4 and np.array_equal(np.bincount(atoms.index), [8] * 5)
    assert rep["bonds"].primitive.owner.max() == 4
    assert d.count == 5 and d.bonds is None


def test_polyhedra_repeat_on_every_copy():
    d = Pipeline(StaticSource(perovskite()), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]),
                                             CoordinationPolyhedra(center_types=["Ti"])]).compute(0)
    one = counts(ir.build(d))["surface:coordination-polyhedra"]
    rep = ir.build(d, replicate=(3, 1, 2))
    mesh = rep["surface:coordination-polyhedra"].primitive
    assert mesh.face_count == 6 * one


@pytest.mark.parametrize("mode, lines", [("super", 12), ("unit", 12), ("all", 12 * 8)])
def test_cell_lines(mode, lines):
    sc = ir.build(perovskite(), replicate=2, replicate_cell=mode)
    cell = sc["cell"].primitive
    assert len(cell.strips) == lines
    pts, edges = cell_edges(Cell(np.eye(3) * A), (2, 2, 2), mode)
    assert np.ptp(pts, 0).max() == pytest.approx(A if mode == "unit" else 2 * A)
    with pytest.raises(ValueError):
        ir.build(perovskite(), replicate=2, replicate_cell="bogus")


def test_boundary_applies_to_the_replicated_block():
    sc = ir.build(perovskite(), replicate=(2, 1, 1), boundary=(0, 1))
    # Ti on every corner of a 2x1x1 block: 3 x 2 x 2 = 12 images of atom 0
    assert np.bincount(sc["atoms"].primitive.index)[0] == 12


def test_view_setting_round_trip():
    pipe = Pipeline(StaticSource(perovskite()))
    view(pipe, replicate=3, replicate_cell="all")
    assert pipe.view.replicate == [3, 3, 3]
    kw = pipe.view.render_kwargs()
    assert kw["replicate"] == [3, 3, 3] and kw["replicate_cell"] == "all"
    view(pipe, replicate=(1, 1, 1))
    assert pipe.view.replicate is None
    with pytest.raises(ValueError):
        view(pipe, replicate=0)
    assert pipe.compute(0).count == 5
