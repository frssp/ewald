# Testing and CI

```bash
pip install -e ".[dev]"            # builds the C++ extension
pip install -e ".[dev,gui]"        # adds the GUI tests' dependencies
xvfb-run -a python -m pytest -q    # full suite
python -m pytest -q --ignore=tests/test_render.py --ignore=tests/test_gui.py   # no display
```

Rendering and GUI tests need an X display with OpenGL. Without one, VTK aborts the whole pytest
process (a fatal error, not a failed test), so use `xvfb-run` or the `--ignore` form.

## Layout

One test module per area (`test_pipeline.py`, `test_io.py`, `test_lammps_readers.py`,
`test_vasp_readers.py`, `test_neighbors.py`, `test_structure_analysis.py`, `test_trajectory.py`,
`test_render.py`, `test_gui.py`, `test_llm.py`, `test_settings.py`, `test_docs.py`, ...).
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
`pyvista/setup-headless-display-action`. GUI extras are installed on Linux only. A separate
`docs` job builds the manual with `mkdocs build --strict` and uploads the site as an artifact.
