# Testing a plugin

A plugin test needs nothing but pytest and ewald. Build a small structure whose answer you
know, run the plugin through a pipeline exactly as a user would, and check the result.

## Modifiers: synthetic structures with known answers

ASE builds perfect crystals in one line, and a perfect crystal has exact answers (coordination
12 in FCC, a CNA class for every atom, zero displacement). `StaticSource` turns them into a
pipeline:

```python
# tests/test_local_composition.py
import numpy as np
import pytest
from ase.build import bulk

from ewald import Pipeline, StaticSource
from local_composition import LocalComposition


@pytest.fixture
def l12():
    """Cu3Au L1_2: every Au has 12 Cu neighbors, every Cu has 8 Cu and 4 Au."""
    atoms = bulk("Cu", "fcc", a=3.75, cubic=True).repeat(3)
    symbols = np.array(atoms.get_chemical_symbols())
    corner = np.all(np.isclose(atoms.get_scaled_positions() * 3 % 1, 0), axis=1)
    symbols[corner] = "Au"
    atoms.set_chemical_symbols(symbols)
    return atoms


def test_fractions_in_l12(l12):
    pipe = Pipeline(StaticSource(l12))
    pipe.modifiers.append(LocalComposition(cutoff=3.0, element="Cu"))
    data = pipe.compute(0)
    frac = data["Local Cu Fraction"]
    assert np.allclose(frac[data.types == "Au"], 1.0)
    assert np.allclose(frac[data.types == "Cu"], 8 / 12)


def test_does_not_touch_the_input(l12):
    pipe = Pipeline(StaticSource(l12))
    before = pipe.source_frame(0).positions.copy()
    pipe.modifiers.append(LocalComposition())
    pipe.compute(0)
    assert np.array_equal(pipe.source_frame(0).positions, before)


def test_parameters_round_trip():
    mod = LocalComposition(cutoff=2.5, element="Au")
    assert mod.parameters() == {"cutoff": 2.5, "element": "Au"}
    assert repr(mod) == "LocalComposition(cutoff=2.5, element='Au')"
```

What to cover, in order of how often each catches a bug:

1. **A known answer** on a perfect crystal, and one on a disturbed structure (a vacancy, a
   displaced atom, random noise with a fixed seed) checked against a brute-force reference.
2. **The input is unchanged** after `compute`, which catches arrays written in place.
3. **Edge cases**: no particles, a single particle, a cell without periodic boundaries, a
   triclinic cell, a cutoff close to half the cell width.
4. **Parameters**: `parameters()` returns every constructor argument, so the GUI form and
   generated scripts work.
5. **Comparison with OVITO** if you port an OVITO analysis: compute the reference with the
   `ovito` pip package once, store it as `.npz`, and compare against the file.
   `tests/data/make_ovito_reference.py` in ewald does this for the built-in analyses.

## The GPU path without a GPU

Code written against `ctx.xp` is tested on NumPy. To check that it does not rely on NumPy-only
behavior, run it with an explicit backend object, which is how ewald's own tests exercise
the CUDA path:

```python
from ewald.backend import Backend
import numpy as np

def test_backend_is_respected(l12):
    pipe = Pipeline(StaticSource(l12))
    pipe.modifiers.append(LocalComposition())
    data = pipe.compute(0, device=Backend("test", np))
    assert "Local Cu Fraction" in data
```

On a machine with CuPy, `pytest.importorskip("cupy")` and compare `device="cuda"` against
`device="cpu"`.

## Readers: small real files

Keep a few small files written by the real program in `tests/data/`, with the input that
produced them, and check frame count, particle count, cell, types and one or two values
against what the program reported. Test a compressed copy, a truncated last frame and the
file-name patterns. A reader that worked on a file you wrote by hand often fails on the
program's own output.

Isolate tests from your own configuration and plugin folder by pointing ewald at a temporary
config folder:

```python
@pytest.fixture(autouse=True)
def clean_config(tmp_path, monkeypatch):
    monkeypatch.setenv("EWALD_CONFIG_DIR", str(tmp_path))
```

A reader-plugin test can copy the plugin file into
`tmp_path / "plugins"` and call `ewald.plugins.load_plugins(force=True)`, then check that
`import_file` picks the plugin without `format=` and that `ewald.plugins.status()["errors"]`
is empty. ewald's `tests/test_plugins.py` does this.

## Visual elements: check the scene, not the pixels

`ewald.scene.build` returns the primitives an element produces without rendering anything, so
geometry is checked with plain asserts and no display:

```python
from ewald.scene import build

scene = build(data, visuals=[CellAxes(length=3.0)])
arrows = scene["visual:cell-axes"].primitive
assert arrows.count == 3
```

When pixels matter, render with `renderer="raytrace"` or `"vector"` and a fixed `camera=`;
both run headless. See [Visual element plugins](visuals.md).

## Checking the GUI integration

The GUI tests need PySide6 and a display; `pytest.importorskip("PySide6")` skips them
elsewhere, and `xvfb-run -a pytest` provides a virtual display on Linux. Checking that the
parameter form builds is usually enough:

```python
def test_form_builds(qtbot):            # needs pytest-qt, or create a QApplication yourself
    from ewald.gui.params import ParameterForm
    form = ParameterForm(LocalComposition())
    assert set(form.editors) == {"cutoff", "element"}
```
