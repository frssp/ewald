# Architecture

ewald is a Python package (`src/ewald/`) with one C++ extension. Everything a user does,
in a script, the GUI or through Claude, goes through the same objects:

```text
file ──reader──▶ source ──▶ Pipeline ──modifiers──▶ DataCollection ──▶ render / GUI / tools
                  (lazy)    (frame cache)  (per frame, on a Backend)
```

`import_file` picks a reader and wraps it in a `FileSource`. A `Pipeline` holds the source and
a list of modifiers; `pipe.compute(frame, device=...)` reads (or takes from the cache) the
source frame, runs each enabled modifier on a `ComputeContext` that carries the array backend,
and returns a `DataCollection`. Renderers, the GUI and the AI tools only consume that
result, which is why a figure made in the GUI can be printed as a script and an analysis asked
for in words ends up as an ordinary pipeline.

## Layout

| Path | Contents |
|---|---|
| `data.py` | `DataCollection` (particles, attributes, tables, cell) and `Cell` |
| `pipeline.py` | `Pipeline`, `Modifier`, `PythonModifier`, `ComputeContext`, the frame cache |
| `__init__.py`, `__main__.py` | `import_file` and the public names; the `ewald` CLI |
| `io/` | native lazy readers (`lammps_dump.py`, `lammps_data.py`, `xdatcar.py`, `vasp_output.py`, `vasp_h5.py`, `extxyz.py`, `xsf.py`, `h5md.py`, `mdtraj_h5.py`; `_textparse.py` is their shared numba atom-table parser), MDAnalysis for XTC/TRR/DCD, everything else through `ase_reader.py`; grids, NEB, phonon and thermo files in `volumetric.py`, `neb.py`, `phonon.py`, `thermo.py`; `remote.py` and `profiles.py` for SSH paths; `base.py` has `FrameReader` and file helpers |
| `backend.py` | `Backend` and `get_backend`: NumPy or CuPy as `xp` |
| `neighbors.py` | `find_pairs`: numba CPU cell list, CUDA cell list, vectorised fallback |
| `cuda_kernels.py` | CUDA C sources launched through CuPy `RawKernel` |
| `kernels.py` | RDF, unwrapping, MSD/VACF FFTs, clustering, trajectory helpers |
| `modifiers/` | one module per family; `__init__.py`'s `__all__` is the public list and resolves OVITO names |
| `modifiers/ovito_bridge.py` | DXA and `OvitoModifier` through the optional `ovito` package |
| `_csrc/` | C++ ports of OVITO's CNA, PTM, Voronoi and CSP plus vendored ptm, voro++ and mwm_csp |
| `scene.py` | the renderer-neutral scene: `VisualElement`s (built-in Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation cell, and plugins) turn a `DataCollection` into primitives (`Spheres`, `Cylinders`, `Lines`, `SurfaceMesh`, `Arrows`) |
| `boundary.py` | VESTA's boundary range for drawing: periodic images inside a fractional range, bonds and polyhedra completed across the cell; `scene.build(boundary=...)` runs the elements on that display copy and maps every primitive back to its real atom |
| `overlays.py` | image-space overlays for every renderer: text label, legends, scale bar, and `OverlayElement`s (Coordinate tripod, Attributes, plugins) returning 2D items |
| `colormaps.py`, `themes.py` | color maps by name (matplotlib plus `@colormap` plugins, with a grayscale check) and themes (element colors, background, cell color, radius preset; VESTA colors in `tables/colors.json`) |
| `render.py`, `raytrace.py`, `vector.py` | draw a scene: VTK/PyVista (impostors above 150k atoms), the numba ray tracer, SVG/PDF via matplotlib; each draws every primitive kind |
| `notebook.py`, `notebook_io.py`, `shell.py`, `kernel.py` | the notebook every GUI tab is: percent-format `.py` cells in one namespace (`pipe` is the pipeline), `.ipynb`/`.ewd` files, cells run through IPython, or in another Python through jupyter_client (`kernel_bridge.py` is its kernel side); `progress.py` draws tqdm under a cell |
| `views.py`, `view3d.py`, `paint.py`, `verbs.py`, `display.py` | what notebook cells call: `show` and the view cells, `pipe.view` (every display setting of the 3D view as code), `paint`/`hide`/`isolate`, analysis verbs (`rdf`, `msd`, ...) that add a step and return its `Table`, and the Jupyter pictures |
| `color_rule.py`, `colorbars.py`, `plotstyle.py`, `view_style.py`, `ui_tokens.py`, `framing.py`, `replicate.py` | what colors the atoms (for legends and the AI), color bars as overlay items, the matplotlib look of every plot, the Qt-free view style and theme palettes, the Render frame region, and replicated-cell display |
| `session.py`, `movie.py` | `.ewald` sessions; GIF/PNG and ffmpeg movies |
| `runinfo.py`, `runs.py` | what a run is (from INCAR, `in.lammps`, logs) and `check_run`; several runs as one notebook (`load_runs`, `arrhenius`) |
| `workflow.py`, `workflow_folder.py`, `workflow_table.py`, `potential.py` | workflows (calculations in steps) from plugin sources or plain folders, their numbers as tables, and the electrostatic potential analyses (planar and macroscopic averages, lineup, work function) |
| `plugins.py`, `plugin_kinds.py`, `plugin_manifest.py`, `plugin_api.py`, `plugin_catalog.py` | the registry of every plugin kind (built-in formats included), `ewald-plugin.toml` manifests, the versioned surface plugins import from, and installing from catalog indexes |
| `trust.py`, `keychain.py`, `fileops.py` | the gates that keep untrusted notebooks, sessions and files from running code; secrets in the OS keychain; replacing files safely on every platform |
| `settings.py` | user preferences as a registry of `Option`s, stored as JSON with only non-default values |
| `gui/` | PySide6 main window (`app.py` plus the `window_*.py` mixins), notebook tabs and cells, viewport (and the OpenGL 1.x one for `ssh -X`), View tab (`DisplayPanel`), generated parameter forms, render page, preferences, side bars (AI chat, Plugins, Workflows, Remotes), console, inspector, plots |
| `llm/` | AI agent and providers (`providers.py`, `chat.py`), tools (`tools.py`; `notebook_tools.py` and `notebook_prompt.py` for the notebook the GUI chat works in; `access.py` for which files and SSH hosts the tools may read), descriptors, narration, Modifier Forge, MCP server, the eval suite (`evals.py`, `eval_tasks.py`), and `reference.py`, the source of `llms.txt` |

### One feature, three layers

A feature that has science code, a file format and a pipeline step keeps the same module name in
each layer (the phonon support: `phonon.py`, `io/phonon.py` and `modifiers/phonon.py`):

| Layer | Holds | Examples |
|---|---|---|
| `ewald/<name>.py` | the domain library: plain functions and types with no pipeline or GUI | `diffraction.py`, `electronic.py`, `bonds.py` |
| `ewald/io/<name>.py` | readers and writers for that domain's files | `io/neb.py`, `io/volumetric.py` |
| `ewald/modifiers/<name>.py` | `Modifier` classes that call the library inside a pipeline | `modifiers/neb.py`, `modifiers/diffraction.py` |

Dependencies point one way: `modifiers/` imports the root module and `io/`; the root module may use
`io.base` helpers to parse files but never imports `modifiers/`, `gui/` or `llm/`. New features follow the same split; a feature
that is only a pipeline step needs only the `modifiers/` module. Support tied to one outside
code (phonopy, PLUMED) is a first-party plugin in
[frssp/ewald-plugins](https://github.com/frssp/ewald-plugins) (`packages/ewald-<name>`) rather
than a core module.

## Principles

**One API.** The GUI, the CLI and the AI tools are clients of the Python API; none of them
has analysis code of its own. A feature lands in the API first.

**Follow OVITO.** Property names, attribute names, structure type ids, colors and modifier names
match OVITO, so OVITO users and OVITO-trained LLMs are productive at once, and ported analyses
can be tested against OVITO's numbers.

**Compute once, on any device.** Compute code is written against an array module, so NumPy in
CI exercises the CUDA path. See [Compute backends](backends.md).

**Python first.** numba or C++ only for a measured hot spot, with the timing in
`benchmarks/`.

**Optional dependencies stay optional.** `gui`, `gpu`, `ovito`, `llm` and `movie` are imported
lazily, inside the functions that need them, and tests skip with `pytest.importorskip`.
Headless `compute()` and `render()` never read user settings implicitly.

**One place decides about running code.** `trust.py` describes every gate (restricted `.ewd`
notebooks, `allow_code` for sessions and AI tools, plugin `sha256`), and
`tests/test_security.py::EXEC_SITES` lists every `exec`/`eval`/import-by-name with its gate, so a
new one fails until it is reviewed and listed.

**The core never imports the GUI.** `ewald.gui` sits on top; data both sides need lives in the
core (`view_style.py` for the view style, `ui_tokens.py` for the theme palettes) and `ewald.gui`
re-exports it. `tests/test_architecture.py` fails when a core module imports `ewald.gui` or
PySide6, when headless compute or rendering imports `ewald.settings`, or when importing the core
loads Qt.

## Where to start reading

`pipeline.py` and `data.py` are short and define the contracts everything else follows.
`modifiers/coloring.py` is a small complete modifier, `modifiers/trajectory.py` shows modifiers
that read other frames, and `io/xdatcar.py` a small lazy reader. For the notebook side, `notebook.py` (cells and how a step or view cell is recognised) and
`views.py` come first; the GUI's `gui/notebook_panel.py` and `gui/cell_widget.py` build on them.
