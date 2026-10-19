# Testing and CI

```bash
pip install -e ".[dev]"            # builds the C++ extension
pip install -e ".[dev,gui]"        # adds the GUI tests' dependencies
xvfb-run -a python -m pytest -q    # full suite
python -m pytest -q --ignore=tests/test_render.py --ignore=tests/test_gui.py --ignore=tests/test_visual.py   # no display
```

Rendering, visual and GUI tests need an X display with OpenGL. Without one, VTK aborts the whole
pytest process (a fatal error, not a failed test), so use `xvfb-run` or the `--ignore` form.

## Visual regression tests

`tests/test_visual.py` renders a fixed set of scenes (Cu with color coding, bonds and legend;
SrTiO3 octahedra; an isosurface with a grid slice; NaCl with text, type legend and scale bar; the
ionic and vdW radius presets; impostor sprites; the draft, standard and publication quality
presets) with the OpenGL viewport renderer, the ray tracer and the vector renderer, and compares
each image with `tests/data/visual/<name>.png`. Vector cases also compare the SVG's element counts
(`<name>.json`) and check that the PDF has no embedded raster image.

```bash
xvfb-run -a python -m pytest tests/test_visual.py                        # compare (about 15 s)
xvfb-run -a python -m pytest tests/test_visual.py --update-references    # rewrite references
xvfb-run -a python -m pytest tests/test_visual.py -k polyhedra --update-references   # just some
```

A failed comparison writes `<name>_expected.png`, `_actual.png`, `_diff.png` and `_compare.png`
(all three side by side) to `visual-diffs/` in the repository (`EWALD_VISUAL_DIFFS` changes the
folder). In the diff, red pixels are beyond the per-pixel tolerance and orange ones changed by
less. When a change to the picture is intended, rerun with `--update-references`, look at the new
PNGs in `git diff` (GitHub shows image diffs) and commit them with the change.

Everything is fixed: structures, cameras, image size (200x150), sample counts and the ray
tracer's per-pixel random seeds, so the ray tracer and matplotlib reproduce images bit for bit and
are compared almost exactly (`visual.EXACT`). OpenGL images depend on the driver; the references
come from Mesa llvmpipe under `xvfb-run`, those cases are skipped on other drivers
(`EWALD_VISUAL_OPENGL=1` forces them) and use a looser `Tolerance` for Mesa version differences.
Keep new scenes small: the whole reference folder is about 0.5 MB.

## Layout

One test module per area (`test_pipeline.py`, `test_io.py`, `test_lammps_readers.py`,
`test_vasp_readers.py`, `test_neighbors.py`, `test_structure_analysis.py`, `test_trajectory.py`,
`test_render.py`, `test_visual.py`, `test_gui.py`, `test_llm.py`, `test_settings.py`, `test_docs.py`, ...).
`conftest.py` provides `cu_fcc`, a seeded `rng`, and the `vasp_dir`/`lammps_dir` fixture folders.
Reader fixtures in `tests/data/` come from real program output; `tests/data/lammps/generate.in`
regenerates the LAMMPS ones.

## Conventions

- Optional dependencies are skipped with `pytest.importorskip("cupy")`, `("PySide6")`,
  `("ovito")`, `("anthropic")`.
- Tests that touch settings or plugins set `EWALD_CONFIG_DIR` to a temporary folder.
- Analyses ported from OVITO compare against the stored OVITO reference; see
  [Native code](native.md).
- GPU code is tested through the NumPy backend and the g++ shim; see
  [Compute backends](backends.md).
- `test_llm.py::test_llms_txt_lists_every_modifier` and `test_docs.py` fail when a modifier is
  missing from `llms.txt` or from the user manual.

## CI

`.github/workflows/ci.yml` runs `pytest -ra` on Linux with Python 3.10, 3.12 and 3.13, on macOS
and on Windows (Python 3.12), with a headless OpenGL set up by
`pyvista/setup-headless-display-action`. When a test fails, `visual-diffs/` is uploaded as an
artifact. GUI extras are installed on Linux only. A separate
`docs` job builds the manual with `mkdocs build --strict` and uploads the site as an artifact.
