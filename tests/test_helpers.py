"""Shared helpers (minimum image, element lookup, colors) and small modifiers without other tests."""
import numpy as np
import pytest
from ase.build import bulk
from ase.geometry import find_mic

from ewald import DataCollection, Pipeline, StaticSource
from ewald import modifiers as mods
from ewald.bonds import covalent_pairs
from ewald.colormaps import to_rgb
from ewald.data import Cell
from ewald.radii import atomic_number, element_masses


def test_minimum_image(rng):
    v = rng.uniform(-12, 12, (50, 3))
    box = np.diag([5.0, 4.0, 6.0])  # exact in an orthogonal cell
    for pbc in ([True] * 3, [True, False, True]):
        np.testing.assert_allclose(Cell(box, pbc=np.array(pbc)).minimum_image(v), find_mic(v, box, pbc)[0], atol=1e-12)
    m = np.array([[5.0, 0, 0], [1.5, 4.0, 0], [0.3, 0.7, 6.0]])  # skewed: nearest image in fractional terms
    f = Cell(m).to_fractional(Cell(m).minimum_image(v) + Cell(m).origin)
    assert np.all(np.abs(f) <= 0.5 + 1e-12)
    np.testing.assert_allclose(np.round(Cell(m).to_fractional(v) - f), Cell(m).to_fractional(v) - f, atol=1e-9)
    assert Cell(m, pbc=np.zeros(3, bool)).minimum_image(v) is not None
    np.testing.assert_array_equal(Cell(m, pbc=np.zeros(3, bool)).minimum_image(v), v)
    one = Cell(m).minimum_image(v[0])  # a single vector keeps its shape
    assert one.shape == (3,)


def test_element_lookup_understands_ion_names():
    assert atomic_number("Fe3+") == 26 and atomic_number("cu") == 29 and atomic_number("1") == 0
    m = element_masses(["Fe3+", "O2-", "Fe3+", "X1"])
    assert m[0] == pytest.approx(55.845) and m[1] == pytest.approx(15.999) and m[3] == 1.0
    assert np.isnan(element_masses(["Q"], default=None)[0]) and element_masses([]).shape == (0,)
    rows = {(a, b): d for a, b, _, d in covalent_pairs(["Fe3+", "O2-"])}
    assert rows[("Fe3+", "O2-")] == pytest.approx(1.15 * (1.32 + 0.66), abs=0.01)  # not the 0.8 fallback


def test_colors_in_0_255_are_scaled():
    np.testing.assert_allclose(to_rgb((255, 0, 0)), [1, 0, 0])
    np.testing.assert_allclose(to_rgb("#00ff00"), [0, 1, 0])
    np.testing.assert_allclose(to_rgb((0.2, 0.4, 0.6, 1.0)), [0.2, 0.4, 0.6])
    d = DataCollection.from_ase(bulk("Cu"))
    out = Pipeline(StaticSource([d]), [mods.AssignColor((255, 128, 0))]).compute(0)
    np.testing.assert_allclose(out["Color"][0], [1, 128 / 255, 0])


def test_invert_and_clear_selection():
    d = DataCollection.from_ase(bulk("Cu", cubic=True))
    pipe = Pipeline(StaticSource([d]), [mods.ExpressionSelection("Position.X < 1"), mods.InvertSelection()])
    first = Pipeline(StaticSource([d]), [mods.ExpressionSelection("Position.X < 1")]).compute(0).selection
    np.testing.assert_array_equal(pipe.compute(0).selection, ~first)
    pipe.modifiers.append(mods.ClearSelection())
    assert "Selection" not in pipe.compute(0).particles


def test_affine_transformation_moves_atoms_and_cell():
    d = DataCollection.from_ase(bulk("Cu", cubic=True))
    shear = [[1, 0.2, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0]]
    out = Pipeline(StaticSource([d]), [mods.AffineTransformation(shear)]).compute(0)
    lin = np.asarray(shear)[:, :3]
    np.testing.assert_allclose(out.positions, d.positions @ lin.T + [1, 0, 0])
    np.testing.assert_allclose(out.cell.matrix, d.cell.matrix @ lin.T)
    np.testing.assert_allclose(out.cell.to_fractional(out.positions), d.cell.to_fractional(d.positions), atol=1e-12)
    kept = Pipeline(StaticSource([d]), [mods.AffineTransformation(np.eye(3) * 2, transform_cell=False)]).compute(0)
    np.testing.assert_allclose(kept.cell.matrix, d.cell.matrix)
