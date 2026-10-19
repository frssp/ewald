# CLAUDE.md

ewald is an OVITO-style atomistic structure/trajectory viewer and analysis library:
`import_file` -> `pipe.modifiers` -> `pipe.compute(frame, device=...)` -> `pipe.render(...)`,
plus a PySide6 desktop GUI and Claude tools. User-facing docs: the manual in docs/ (MkDocs,
`mkdocs.yml`; user/, plugins/, developer/, reference/) and README.md; what exists:
docs/features.md (generated); open work: docs/todo.md; milestones: docs/ROADMAP.md; GUI design
specs live outside the repo (project files `design/`).

## Commands

```bash
pip install -e ".[dev]"          # builds the C++ extension ewald._native (needs a C++17 compiler)
pip install -e ".[dev,gui]"      # + PySide6/pyvistaqt/qtconsole for tests/test_gui.py
xvfb-run -a python -m pytest -q  # full suite; rendering and GUI tests need an X display with OpenGL
python -m pytest -q --ignore=tests/test_render.py --ignore=tests/test_gui.py --ignore=tests/test_visual.py   # no display
xvfb-run -a python -m pytest tests/test_visual.py [--update-references]   # visual regression (reference images)
python -m ewald api > llms.txt   # regenerate the LLM API reference (see below)
xvfb-run -a python -m ewald inventory -o docs/features.md --gui   # regenerate the feature inventory
python benchmarks/neighbors.py   # neighbor-list timings
python benchmarks/large_files.py # open/frame-switch/memory on generated multi-million-atom files
python examples/demo.py          # sample trajectories + figures into demo_out/
mkdocs build --strict            # the manual (pip install -r docs/requirements.txt); CI runs it
```

Without a display, VTK rendering aborts the whole pytest process (fatal error, not a test failure),
so use `xvfb-run` or the `--ignore` form. Re-run `pip install -e .` after editing anything in
`src/ewald/_csrc/`; the extension is not rebuilt automatically. CI (`.github/workflows/ci.yml`)
runs `pytest -ra` on Linux py3.10/3.12/3.13, macOS and Windows; GUI extras only on Linux.

Visual regression: `tests/test_visual.py` compares fixed scenes (OpenGL, ray tracer, vector; atoms,
bonds, polyhedra, isosurface, overlays, radius and quality presets) with `tests/data/visual/*.png`.
Run it before merging any change to `render.py`, `raytrace.py`, `vector.py`, `overlays.py`,
`bonds.py`, `mesh.py` or `radii.py`, since CI may not run. A failure writes expected/actual/diff
images to `visual-diffs/`; look at them. If the change is intended, rerun with
`--update-references` (optionally `-k <scene>`), view the new PNGs and commit them with the change;
never edit references by hand. OpenGL references are made with Mesa llvmpipe (`xvfb-run`) and skip
on other drivers. Keep scenes at 200x150 so the references stay small.

## Layout (`src/ewald/`)

- `data.py` (`DataCollection`, `Cell`), `pipeline.py` (`Pipeline`, `Modifier`, `PythonModifier`,
  `ComputeContext`, frame cache), `__init__.py` (`import_file`), `__main__.py` (CLI `ewald ...`).
- `io/`: native lazy readers (`lammps_dump.py`, `xdatcar.py`, `extxyz.py`, `xsf.py`, `h5md.py`), MDAnalysis
  for XTC/TRR/DCD (`mdanalysis_reader.py`), everything else via `ase_reader.py`; each module also has its
  writer. `io.builtin_readers()`/`builtin_writers()` register them in `plugins.py` (reader and writer registries).
  `remote.py` opens `host:/path` / `ssh://` paths (paramiko): `base.open_binary` streams them to the
  native readers, `FileSource` downloads them to a local cache for every other reader. Tests use the
  in-process server in `tests/sftp_server.py` (fixtures `server`, `ssh_env` in conftest), no sshd.
- `bonds.py` (`Bonds`: Topology + Periodic Image, bond search), `mesh.py` (`SurfaceMesh` for
  `data.surfaces`: polyhedra now, surface meshes later); both drawn by `render.py` and `raytrace.py`.
- `backend.py` (numpy/CuPy `xp` backends), `neighbors.py` (numba CPU cell list, `find_pairs`),
  `cuda_kernels.py` (CUDA C via CuPy `RawKernel`), `kernels.py` (RDF, unwrap, MSD).
- `modifiers/`: one module per family; `__init__.py` `__all__` is the public list and also resolves
  OVITO names (`XxxModifier` -> `Xxx`). `ovito_bridge.py` wraps the optional `ovito` package (DXA).
- `_csrc/`: C++ ports of OVITO CNA/PTM/Voronoi/CSP plus vendored ptm, voro++, mwm_csp; origins and
  licenses in `_csrc/README.md`. Built by `setup.py` (pybind11).
- `radii.py`: radius presets (covalent, Slater, Clementi, vdW, Shannon ionic with oxidation-state and CN guessing);
  tables in `tables/radii.json`, regenerated from pymatgen with `tables/make_radii.py`, never by hand.
- `render.py` (VTK/PyVista offscreen; impostors above 150k atoms), `raytrace.py` (numba ray tracer,
  the publication renderer; pip VTK has no OSPRay), `vector.py` (SVG/PDF/EPS via matplotlib),
  `overlays.py` (text, color/type legends, scale bar for every renderer), `movie.py` (GIF/PNG via
  Pillow, MP4/AVI/MOV via ffmpeg), `session.py` (`.ewald` JSON: source, modifiers, style, camera).
- `plugins.py`: reader and modifier registry. Built-in formats (`io.builtin_readers()`, with
  priorities) plus plugins from `ewald.readers` / `ewald.modifiers` entry points and
  `<config dir>/plugins/*.py`; `io.open_reader`, the GUI open filter and menu, the Claude tools and
  `ewald.modifiers.__getattr__` read it. Plugin modifiers stay out of `__all__` and `llms.txt`.
  Examples in `examples/plugins/`.
- `settings.py`: user preferences as a registry of `Option(...)`, stored as JSON with only
  non-default values; `EWALD_CONFIG_DIR` overrides the location (tests set it per test).
- `inventory.py`: `ewald inventory`, docs/features.md from the registries, CLI parser and GUI menus.
- `manual.py`: Help-menu links into docs/ (GitHub by default, or `general.docs_url`).
- `gui/`: `app.py` main window, `viewport.py`, `pipeline_panel.py`, `params.py`, `render_page.py`,
  `preferences.py`, `remote_dialog.py` (SSH browser), `console.py`, `inspector.py`, `plots.py`,
  `timeline.py`, `palette.py`, `theme.py`.
- `llm/`: Claude agent (`agent.py`), tools (`tools.py`), Modifier Forge (`forge.py`: Claude writes a
  plugin + tests, verified in a subprocess, saved to the plugin folder), descriptors (`describe.py`, `narrate.py`),
  MCP server (`mcp_server.py`), and `reference.py`, the source of `llms.txt`.

## Conventions

- Never mutate arrays of a `DataCollection` in place; they are shared with cached source frames.
  Build a new array and assign it. Modifiers return `None` (keep data) or a new collection.
- Compute code is written once against `ctx.xp` / a `Backend`, so the numpy run in CI exercises the
  CUDA path. No GPU in CI or in the cloud container; `tests/test_cuda_kernel.py` compiles the CUDA C
  source with g++ behind a shim. Keep backends pluggable (Apple Metal may come later).
- Follow OVITO for property names ("Position", "Particle Type", "Structure Type", ...), structure
  type ids, colors and attribute names. Ported analyses are tested against OVITO results stored in
  `tests/data/ovito_reference.npz` and `ovito_cluster_reference.npz`; regenerate them with
  `tests/data/make_ovito_reference.py` (needs `pip install ovito`), never by hand. Statistics,
  bond-network and Steinhardt references (OVITO + freud) are in `analysis_reference.npz`, from
  `tests/data/make_analysis_reference.py` (needs `pip install ovito freud-analysis`).
- `Modifier.parameters()` comes from the constructor signature, and the GUI parameter form
  (`gui/params.py`) is generated from it, so store every constructor argument as an attribute of the
  same name. New modifiers that construct with no arguments appear in the GUI menu automatically;
  otherwise add an entry with a sensible default to `MODIFIER_MENU` in `gui/pipeline_panel.py`.
- Optional dependencies (`gui`, `gpu`, `ovito`, `llm`, `movie`, `formats`) are imported lazily; tests use
  `pytest.importorskip` for them. Headless `render()`/`compute()` never read `settings` implicitly.
- Python first; numba or C++ only for measured hot spots. Ported OVITO code is used under its MIT
  option and keeps the notice. No LICENSE file (private project).

## When you change the API

- Every PR that adds or changes a feature (modifier, reader/writer, renderer option, CLI command,
  Claude tool, preference, GUI menu entry) regenerates docs/features.md with
  `xvfb-run -a python -m ewald inventory -o docs/features.md --gui` and updates docs/todo.md: drop
  finished items, add what is left open, keep "In progress" current. features.md is generated,
  never hand-edited; `tests/test_docs.py` and `tests/test_gui.py` fail when it is stale. todo.md
  is the only TODO list; docs/ROADMAP.md keeps milestones and decisions.
- `llms.txt` is a snapshot of `ewald api` (`llm/reference.py`: the `GUIDE` text plus every modifier
  signature and docstring from `ewald.modifiers.__all__`). After adding, renaming or changing the
  signature/docstring of a modifier, or editing `GUIDE`, run `python -m ewald api > llms.txt` and
  commit it. `tests/test_llm.py::test_llms_txt_lists_every_modifier` fails when a modifier is missing.
- Add new modifiers to `__all__` in `modifiers/__init__.py`, to the modifier list in README.md and
  to docs/user/modifiers.md (`tests/test_docs.py` checks); update docs/ROADMAP.md when a roadmap
  item lands.
- User-visible changes update the matching manual page (docs/developer/docs.md says which); new
  pages go into `nav` in mkdocs.yml. Pages are plain Markdown that also reads on GitHub.
