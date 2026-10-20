# CLAUDE.md

EWALD (Extensible Workbench for Atomic Lattices & Dynamics; package and import name `ewald`) is an
atomistic structure/trajectory viewer and analysis library at OVITO Pro level, for VASP and LAMMPS
first: `load`/`import_file` -> `pipe.add(...)` -> `pipe.compute(frame, device=...)` -> `show`/`pipe.render(...)`.
Its identity is not "OVITO clone" but a Jupyter-like notebook (scripts plus Python visualization)
with Claude and plugins built in: the GUI is notebooks as top tabs, step and view cells, a VS Code
style plugin side bar and an AI chat side bar. User-facing docs: the manual in docs/ (MkDocs,
`mkdocs.yml`; user/ with tutorials/, plugins/, developer/, reference/) and README.md, both
notebook-first; tutorial notebooks in examples/notebooks/ (run by tests/test_tutorials.py); what exists:
docs/features.md (generated); open work: docs/todo.md; milestones: docs/ROADMAP.md. Current GUI
design specs live outside the repo (project files `design/`); `docs/design/` is an older snapshot
kept for history.

## How we work (decisions from the project, 2026-10-18/19)

- The owner (Sunghyun) writes Korean with English technical terms; answer the same way. Ask the
  few real decisions up front, then work autonomously. Code, commits, PRs and docs stay in English.
- Merging: a thread merges its own PR once it has merged the latest `main` and the relevant local
  tests pass (full suite under `xvfb-run` when GUI or rendering is touched, `tests/test_visual.py`
  for renderer files). No need to wait for an explicit "merge". When PRs must land in order, the
  thread that merges tells the next one.
- GitHub Actions minutes are exhausted, so every workflow is `workflow_dispatch` only (PR #62) and
  CI does not run on PRs; local runs are the gate. Do not watch, re-run or re-enable Actions, and do
  not push tags (v0.1.0 is prepared but untagged; docs/developer/release.md, testing.md say how to
  restore triggers when minutes return).
- Quality vs speed (owner's rule): file and render output put quality first (render presets
  Draft/Standard/Publication, default Publication); the interactive viewport puts speed first (LOD,
  target FPS, cache budget under `viewport.*`/`performance.*`). Both stay adjustable in settings.
- Core vs plugins: VASP and LAMMPS (and the other core formats) are built in, registered as
  "built-in" plugin groups that can be disabled. Domain extras live as first-party packages in
  frssp/ewald-plugins: `ewald-phonopy` (phonons), `ewald-plumed` (all PLUMED), `ewald-aiida`; the
  core keeps only stubs and `modifiers.MOVED` hints. The plugin index pins commits, so merge plugin
  PRs with a merge commit, not squash. Plugins declare themselves in `ewald-plugin.toml` (no import
  needed to list them) and run in-process.
- Deferred until the owner asks: structure editing (supercell, slab, defects); the remaining
  folder/renaming steps 2-4 of the layout review (gui renames, internal renames, splitting the
  package root); a separate DXA implementation (use the optional `ovito` package).
- Reuse first: OVITO's public source is GPLv3-or-MIT, so ports are taken under MIT with the notice
  kept (`tests/test_licenses.py`); freud, VIAMD/mdlib and tomviz are fine references, VMD code is not.
- Security: a session, notebook, `.ewd` or text inside a structure file (prompt injection to Claude)
  must not run code unless trusted (`allow_code`, `Notebook.trust()`). Code that evaluates strings,
  imports by name, or writes files for Claude needs the same checks and a case in
  `tests/test_security.py` (every exec/eval/import-by-name site is listed in its `EXEC_SITES`;
  `ewald/trust.py` describes the gates). AI API keys (one per provider) live in the keychain (`ewald/keychain.py`,
  `llm/credentials.py`); never put them in `os.environ` or settings.json.
- GUI: PySide6 only (`QT_API=pyside6`). Over `ssh -X` the legacy OpenGL 1.x viewport is chosen
  automatically (`viewport.backend` / `EWALD_VIEWPORT`); this is still unverified on a real ssh -X
  host, as are macOS, real HPC SSH and remote AiiDA. Restyle through QSS object names in
  `gui/theme.py`, not inline `setStyleSheet`. New window menu, file or pick code goes into the
  matching `window_*.py` mixin, not `app.py`.

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
python benchmarks/readers.py     # native readers vs ASE/OVITO/MDAnalysis/chemfiles (docs/developer/readers.md)
python examples/demo.py          # sample trajectories + figures into demo_out/
mkdocs build --strict            # the manual (pip install -r docs/requirements.txt); CI runs it
xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_screenshots.py   # manual screenshots from examples/notebooks
```

Without a display, VTK rendering aborts the whole pytest process (fatal error, not a test failure),
so use `xvfb-run` or the `--ignore` form. Re-run `pip install -e .` after editing anything in
`src/ewald/_csrc/`; the extension is not rebuilt automatically. CI (`.github/workflows/ci.yml`)
runs `pytest -ra` on Linux py3.10/3.12/3.13, macOS and Windows (GUI extras only on Linux), but only
when dispatched by hand for now (see "How we work").

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
- `io/`: native lazy readers (`lammps_dump.py`, `lammps_data.py`, `xdatcar.py`, `vasp_output.py` for
  OUTCAR/vasprun.xml/OSZICAR, `vasp_h5.py`, `extxyz.py`, `xsf.py`, `h5md.py`, `mdtraj_h5.py`), MDAnalysis
  for XTC/TRR/DCD (`mdanalysis_reader.py`), everything else via `ase_reader.py`; each module also has its
  writer. `io.builtin_readers()`/`builtin_writers()` register them in `plugins.py` (reader and writer registries).
  `remote.py` opens `host:/path` / `ssh://` paths (paramiko): `base.open_binary` streams them to the
  native readers, `FileSource` downloads them to a local cache for every other reader. Tests use the
  in-process server in `tests/helpers/sftp_server.py` (fixtures `server`, `ssh_env` in conftest), no sshd.
  `profiles.py` keeps saved SSH computers (no secrets; passwords in `keychain.py`; GUI
  `gui/remotes_panel.py`, CLI `ewald remote`). `_textparse.py` is the numba atom-table parser shared by the dump and extxyz readers. Write a native
  reader only where an index plus that parser beats ASE by a wide margin (big text trajectories);
  delegate everything else (docs/developer/readers.md has the measurements and the rule).
- `bonds.py` (`Bonds`: Topology + Periodic Image, bond search), `mesh.py` (`SurfaceMesh` for
  `data.surfaces`: polyhedra now, surface meshes later); both drawn by `render.py` and `raytrace.py`.
- `backend.py` (numpy/CuPy `xp` backends), `neighbors.py` (numba CPU cell list, `find_pairs`),
  `cuda_kernels.py` (CUDA C via CuPy `RawKernel`), `kernels.py` (RDF, unwrap, MSD).
- `modifiers/`: one module per family; `__init__.py` `__all__` is the public list and also resolves
  OVITO names (`XxxModifier` -> `Xxx`) and moved ones (`MOVED`: an ImportError with the pip hint for
  the plugin). `ovito_bridge.py` wraps the optional `ovito` package (DXA).
- `_csrc/`: C++ ports of OVITO CNA/PTM/Voronoi/CSP plus vendored ptm, voro++, mwm_csp; origins and
  licenses in `_csrc/README.md`. Built by `setup.py` (pybind11).
- `radii.py`: radius presets (covalent, Slater, Clementi, vdW, Shannon ionic with oxidation-state and CN guessing);
  tables in `tables/radii.json`, regenerated from pymatgen with `tables/make_radii.py`, never by hand.
- `scene.py`: renderer-neutral Scene IR (Spheres, Cylinders, Lines, SurfaceMesh, Arrows) and
  `VisualElement`s (built-in Particles, Bonds, Cell, ... and `@visual` plugins); every renderer
  draws a `Scene`, so new geometry goes into an element, not into one renderer.
  `boundary.py`: VESTA boundary ghosts (display copy with periodic images, bonds/polyhedra completed);
  `scene.build(boundary=...)` applies it and maps primitives back to real atoms. Display only.
- `render.py` (VTK/PyVista offscreen; impostors above 150k atoms), `raytrace.py` (numba ray tracer,
  the publication renderer; pip VTK has no OSPRay), `vector.py` (SVG/PDF/EPS via matplotlib),
  `overlays.py` (text, color/type legends, scale bar and `OverlayElement`s returning 2D items, for every
  renderer), `colormaps.py` (`get_colormap`: matplotlib names plus `@colormap` plugins; use it instead of
  `matplotlib.colormaps[...]`), `themes.py` (element colors, background, cell color, radius preset;
  VESTA colors in `tables/colors.json`, regenerated with `tables/make_colors.py`, never by hand), `movie.py` (GIF/PNG via
  Pillow, MP4/AVI/MOV via ffmpeg), `session.py` (`.ewald` JSON: source, modifiers, style, camera).
- `plugins.py`: reader, writer, modifier, visual element, overlay, color map and theme registry. Built-in formats (`io.builtin_readers()`, with
  priorities) plus plugins from `ewald.readers` / `ewald.modifiers` entry points and
  `<config dir>/plugins/*.py`; `io.open_reader`, the GUI open filter and menu, the Claude tools and
  `ewald.modifiers.__getattr__` read it. Plugin modifiers stay out of `__all__` and `llms.txt`.
  Examples in `examples/plugins/`. The plugin dataclasses and built-in entries are in
  `plugin_kinds.py`, manifests and version checks in `plugin_manifest.py` (re-exported by
  `plugins.py`, which keeps the registry state tests monkeypatch). `plugin_api.py` is the
  supported surface plugins import from, versioned by `PLUGIN_API`: changing a name there
  incompatibly means raising it.
- `assets/icon/`: app icon. `ewald.svg` (and `ewald-small.svg` for 16-32 px) are the sources; the PNG
  sizes, `ewald.ico` and `ewald.icns` are rendered by `make_icons.py` (cairosvg), never by hand.
  `gui.icons.app_icon()` loads them.
- `view_style.py` (`ViewStyle`, the Qt-free view style sessions and the GUI share) and `ui_tokens.py`
  (theme palettes, `derive`); `gui/style.py` and `gui/theme.py` re-export them. The core never imports
  `ewald.gui` (`tests/test_architecture.py`).
- `settings.py`: user preferences as a registry of `Option(...)`, stored as JSON with only
  non-default values; `EWALD_CONFIG_DIR` overrides the location (tests set it per test).
- `paint.py` (`paint`/`hide`/`isolate`/`fade`/`resize` for Python steps), `views.py` (`show`, the
  notebook view cells), `verbs.py` (`rdf`/`msd`/`sq`/... add a step and return its `Table`), `display.py` (`_repr_png_` of Pipeline/DataCollection/Table for Jupyter, `options`), `plugin_catalog.py` (installed plugins, catalog indexes, install/remove;
  GUI side `gui/plugins_panel.py`, the left Plugins side bar).
- `inventory.py`: `ewald inventory`, docs/features.md from the registries, CLI parser and GUI menus.
- `plumed.py`, `phonon.py`: stubs that alias the `ewald-plumed` / `ewald-phonopy` plugin modules
  (frssp/ewald-plugins); the code and its tests live there.
- Domain modules: `electronic.py` (VASP DOS/bands), `diffraction.py` (XRD, S(q)), `arrows.py`
  (per-atom vectors), `measure.py` (distances/angles/dihedrals of the view's `measurements`),
  `linking.py` (linked selection between plots and the 3D view), `keychain.py`
  (Claude key, SSH secrets); readers `io/volumetric.py` (CHGCAR/cube grids), `io/neb.py`, `io/thermo.py`.
- `manual.py`: Help-menu links into docs/ (GitHub by default, or `general.docs_url`).
- `notebook.py`: notebooks as percent-format `.py` files (`# %%` cells, one namespace, `pipe` is the
  pipeline; step cells = one `pipe.add(...)`, view cells = one `show(pipe, ...)` from
  `views.py`). The GUI is built on it; the OVITO-style classic layout (pipeline list + properties
  panel) was removed on 2026-10-20 and must not come back.
  `notebook_io.py` reads and writes `.ipynb` and `.ewd` (zip: notebook.ipynb, manifest.json,
  figures/, data/, plugins/); an `.ewd` opens restricted until `Notebook.trust()`. `kernel.py`
  (`Kernel` via jupyter_client, `KernelSource`) runs a notebook's cells in another Python;
  `kernel_bridge.py` is its kernel side (cells run by a Notebook there, frames back as `.npz`).
- `gui/`: `app.py` main window (frames, recompute, styling; built from mixins: `window_menus.py` toolbar,
  menus, activity bar and minimal mode, `window_files.py` open/export/sessions/notebooks, `viewport_menu.py`
  right-click menu, `window_links.py` picks, plot links (Plots tab and view cells), Ask AI and console),
  `notebook_panel.py` (`NotebookView`/`NotebookPanel`: running, undo, tabs; `cell_widget.py` one cell with its
  step form, outputs and view cells with live `PlotsPanel`s; `code_editor.py` the editor and highlighter), `notebook_tabs.py` (one tab per
  open notebook), `plugins_panel.py` (Plugins side bar), `viewport.py` (VTK; `legacy_viewport.py` + Qt-free
  `legacy_gl.py` are the OpenGL 1.x viewport for `ssh -X`, chosen by `gui.viewport_backend`), `pipeline_panel.py`
  (step menus and `DisplayPanel`, the visual elements side bar), `params.py`, `render_page.py`,
  `preferences.py` (dialog; `pref_widgets.py`, `element_table.py`, `prefs_apply.py` apply settings to the window),
  `remote_dialog.py` (SSH browser), `workers.py` (`detach` for QThreads that may outlive their widget, `relay` for pool threads), `ask_panel.py` (AI chat side bar), `console.py`,
  `inspector.py`, `plots.py`, `timeline.py`, `palette.py`, `theme.py`, `collapsible.py` (fold buttons in dock
  title bars), `header.py` (one panel header: height, inset, icon buttons; no glyph labels), `macos.py` (menu bar name; Finder file-open
  events go through `app.EwaldApplication`). Frozen apps: `packaging/windows/` and `packaging/macos/` (EWALD.app,
  docs/developer/macos-release.md); the macOS workflow is manual only (10x minutes) and never needs to run for a PR.
- `llm/`: AI agent (`agent.py`) over any provider: `providers.py` (Claude by default, OpenAI, Gemini,
  OpenAI-compatible servers such as Ollama, an API gateway reading sim-agent's `GATEWAY_*` variables;
  `@ai_provider` plugins; `llm.*` settings, `from_settings()` and saved connections `llm.profiles`) and `chat.py` (`Chat`: one conversation in a provider's format, `AnthropicChat`/`OpenAIChat`; the agent
  loop, forge and narrator use only it), tools (`tools.py`), `credentials.py` (API keys in the OS keychain via
  `keyring`, one entry per provider, 0600 file fallback, never in settings.json or os.environ; every client comes
  from `credentials.client(provider=...)`), Modifier Forge (`forge.py`: the model writes a
  plugin + tests, verified in a subprocess, saved to the plugin folder), descriptors (`describe.py`, `narrate.py`),
  `selection.py` (descriptor + close-up scene for Ask about selection; GUI side `gui/ask_panel.py`),
  MCP server (`mcp_server.py`), `reference.py`, the source of `llms.txt`, and the AI eval suite
  (`evals.py` harness and graders, `eval_tasks.py` tasks; `ewald eval`, never in the default
  pytest run since it costs tokens; `tests/test_evals.py` plays each task's reference solution).

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
- Extras are only for heavy or platform-specific packages (`gui`, `gpu`, `ovito`, `formats` = MDAnalysis;
  `all` = gui+formats+ovito); small pure wheels (spglib, h5py, imageio-ffmpeg, anthropic/openai/mcp,
  jupyter_client/ipykernel) are core dependencies, and old extra names stay as empty aliases. Heavy or
  optional packages are still imported lazily; tests use `pytest.importorskip` for extras. Headless `render()`/`compute()` never read `settings` implicitly.
- Python first; numba or C++ only for measured hot spots. Ported OVITO code is used under its MIT
  option and keeps the notice. EWALD itself is MIT (`LICENSE`).

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
