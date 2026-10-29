# Changelog

All notable changes to ewald. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (before 1.0, a minor version may
change the API). Pull request numbers refer to <https://github.com/frssp/ewald/pulls>.
Pull request numbers up to #242 refer to the pre-publication repository, which is kept private; the public repository's numbering starts again at #1 (decision D32).
How to cut a release is in [docs/developer/release.md](docs/developer/release.md).

## [Unreleased]

### Added

- `data.particles.count` and `data.particles.positions` work as in OVITO (read-only shortcuts for
  `data.count` and `data.positions`): the AI wrote them in 11 of 69 eval tasks. (D84, #116)
- NumPy `.npz` trajectories are read and written built in (`positions`, `types`/`numbers`, `cell`,
  per-atom and per-frame arrays such as `forces` and `energies`; group "npz"). (D78, #100)
- Windows online installer `ewald-X.Y.Z-win64-online-setup.exe`: a private Python into which setup
  pip-installs ewald and its dependencies, so plugins and packages install as in any Python; an
  NVIDIA GPU checkbox adds CuPy. The offline installers stay. (D79, #101)
- GPU compute installs from the app: Preferences › Compute › Optional components › Install (or
  `ewald gpu --install`) pip-installs CuPy with the CUDA 12 libraries, also into the Windows
  standalone app's add-on folder; only the NVIDIA driver is needed. (D80, #102)
- `MeanSquaredDisplacement` writes the per-atom property `MSD`, and `ExpressionSelection(expression,
  top=n, by=...)` keeps the n particles with the largest value, so "the paths of the 5 Li with the
  largest MSD" is three steps; Ask AI gets the recipe and a new eval task (`li-top-msd-paths`).
  Expression names drop characters a name cannot hold (`SqDispAverage`). (#105)
- `MeanSquaredDisplacement` and `IonicConductivity` take `fit_time=(t0, t1)`, so runs of different
  length fit the same lag times; the MSD writes the fitted window as `.fit_start`/`.fit_end`. (D84, #110)
- Ask AI's `search_api` tool finds the step, function or recipe for a job and returns its whole
  docstring; the system prompt asks for steps before hand-written code and for how much a number
  moves with each analysis choice. Plugins offer functions to cells and to the AI with
  `@api_function`. (D84, #111, #112)
- Ask AI's questions carry the steps and functions of ewald that match their words (signature and
  first sentence, paths and file names left out), so the model starts from them rather than from
  its own code; Preferences › Notebook › Show the AI related steps (`llm.related_steps`). (D84, #114)
- With permission to run cells, Ask AI works as a coding agent: it is offered only the cell tools,
  `search_api`, `python_api`, `render_image`, `get_view`, `describe_trajectory`,
  `describe_selection`, `check_run` and plugin tools, and computes everything in cells it writes
  and runs; without that permission it keeps every tool. (#115)
- `sensitivity(pipe, step, parameter, values, attributes)` tries each value of a step's parameter
  (a fit window, a cutoff) and returns the step's results per value with their relative spread;
  Ask AI is told to use it when it reports how a number depends on a choice. (D84, #113)

### Changed

- `FindRings` leaves out chains that close only through the periodic boundary (counted in
  `FindRings.periodic_rings`; `periodic_rings=True` keeps them, as OVITO does). (D84, #110)
- Step cells are edited in Python only: the *Form* button and its generated parameter form are
  gone; a new step and View › Edit step open the cell's code. (D83, #107)
- The Windows standalone app is one build, `ewald-X.Y.Z-win64-setup.exe` / `-portable.zip`; the
  separate GPU build (about 1.8 GB per file) is gone, since GPU compute now installs from the app.
  The app bundles the whole standard library, so packages pip installs later find what they
  import. (D80, #102)
- `pymatgen` is a required dependency (also bundled in the standalone apps): `to_pymatgen()`,
  `from_pymatgen()` and bond-valence oxidation states for ionic radii always work. (D79, #101)

- OPTIMADE structure databases are built in (group "optimade"): `load("optimade:mp/mp-149")`,
  `ewald.optimade.search`, the OPTIMADE side bar, the read-only AI tool `optimade_search`,
  Preferences › OPTIMADE (settings kept) and the new `ewald optimade search|databases`. They were the
  `ewald-optimade` plugin, which an older install no longer loads; `ewald-optimade` 0.2 only
  re-exports `ewald.optimade`. (D78, #100)

## [0.2.0] - 2026-10-28

The first public release. 0.1.0 below was prepared in the private repository and never tagged,
so this release also carries everything listed there.

### Added

- The standalone Windows and macOS apps install pip plugins from the Plugins side bar: they bundle
  pip, `ewald.exe` runs `-m pip` like python, and plugins go into the app's data folder with the
  bundled packages kept as they are. Without git, GitHub plugins are fetched as zip archives. (D76)

- The notebook shows which cell runs and which wait: a tinted card with an accent border, a stop
  button with a turning ring and the elapsed time on the running cell, a dashed border and `*` on
  waiting ones, a chip beside Interrupt ("Cell 3 · 2.8 s · 2 waiting") that jumps to it, a play
  icon on the notebook's tab, and `python.follow_running` to scroll along (D72). Before, a cell that ran longer than 0.1 s lost its `*`.
  `Notebook.on_cell(cell, running)` reports each cell's start and end. (#84)

- Laptops with two GPUs: Preferences › 3D view › GPU for the 3D view (`viewport.gpu`), `EWALD_GPU`
  and `ewald gpu [--prefer high-performance]` move the 3D view to the NVIDIA/AMD GPU (the Windows
  Graphics settings choice for this Python, PRIME offload on Linux); About lists the GPUs and says
  when the view draws on the integrated one; OpenCL prefers a discrete GPU.
- `ewald eval --category study`: whole analyses from one sentence (activation energy from a series
  of MD runs with a trap in it, the Li diffusion path, a paragraph for a paper, a follow-up
  question, wrong premises), on synthetic VASP MD series (`ewald.llm.eval_data`) and the LLZO data.
- `MeanSquaredDisplacement(by_axis=True)`: the MSD along x, y and z and 1D diffusion coefficients;
  the AI's `compute_msd` reports how the MSD splits over the axes.
- New notebooks import `rdf, msd, vacf, sq, xrd, load_runs, check_run, run_info, arrhenius`.
- `Arrhenius.plot()` and `.savefig(path)`, `arrhenius({T: D})`; `runs["500"]` finds the run
  labelled "T500". (#73)
- Every color argument (view settings, steps, overlays, step forms) takes the same forms through
  `ewald.colors`: names in any case, hex with or without `#`, `rgb()`/`rgba()`, grey levels and
  0..1 or 0..255 numbers; a bad color fails when set, and the GUI writes hex into the view cell.
  Step forms get a color field with a swatch. (#74)
- Ask AI › Test connection shows the server's reason for an API error (a 422's missing header,
  for example), not only the status code.
- Ask AI, OpenAI-compatible provider: *API key header* (send the key as `api-key` for Azure
  OpenAI or a gateway's own header), `{uuid}` in *Extra headers* for a new id per request, and
  *Test connection* falls back to a chat request on servers without a models endpoint.
- MIT license (`LICENSE`); the `Private :: Do Not Upload` classifier is gone.
- `CITATION.cff` (GitHub's *Cite this repository*), `CONTRIBUTING.md`, `SECURITY.md`, issue
  templates and `.github/FUNDING.yml`, in preparation for a public repository.
- A pull request template and `CODE_OF_CONDUCT.md`; `CONTRIBUTING.md` now covers the
  Windows/macOS/Linux rules and how design decisions are recorded.
- `ewald.plugin_api`, the supported import surface for plugins, versioned by `PLUGIN_API` (1). A
  plugin declares `plugin-api` in its manifest and an ewald that does not support that version
  refuses it before importing it. The example plugins, the plugin docs and Modifier Forge use it.
- Plugin catalogs can give a `.py` entry a `sha256`; *Install* refuses a download that does not
  match it. `ewald.trust` collects the code-execution checks (`UntrustedCode`, a
  `PermissionError`, is raised for an untrusted `.ewd` notebook).

### Changed

- Playing a large trajectory: the bond search of each frame runs on the compute worker instead of
  the GUI thread, the next frame is read while the current one is drawn, and the neighbor-list
  kernels let the window paint while they run; the color legend no longer loops over every atom in
  Python (0.3 s per frame at 1M atoms). `benchmarks/viewport.py` measures the 3D view (D73). (#90, #93)
- Bonds in the 3D view: the half-bonds and their GPU sticks are built by parallel numba kernels
  (1M Cu atoms: about 10 s down to 2-3 s), and the two halves of a bond in one color are one stick
  (half the GPU memory, same picture). Above Preferences › 3D view › Performance › *Bond
  cylinders at most* (`viewport.max_bond_sticks`, 8M) the view leaves bonds out and says so
  instead of running out of GPU memory. `Cylinders.partner` pairs the halves of a bond. (#95)

- New app icon: an ammonia molecule with a glass lone-pair lobe over an orange chat bubble
  (replaces the Ewald sphere).
- The separate `gateway` AI provider and its *User id* / *System name* settings are gone; a
  gateway is set up as an OpenAI-compatible server with an API key header and extra headers.
- Fewer extras: spglib, imageio-ffmpeg, h5py, anthropic, openai, mcp, jupyter_client and ipykernel
  are core dependencies, so symmetry, MP4 export, HDF5 formats, Ask AI with every provider and
  kernels work in a plain install. Extras are `gui`, `gpu`, `formats` (MDAnalysis), `ovito` and
  the new `all` (gui, formats and ovito); `symmetry`, `movie`, `llm`, `kernel` and `remote` remain
  as empty aliases so old commands work.
- A step that fails because a package is missing reports it once instead of on every frame: the
  pipeline stops retrying it (with `errors="skip"`) until its settings change or
  `Pipeline.invalidate()`, and the GUI logs a step's error once until it changes.

- Notebook trust works as in Jupyter: the app opens any `.py`, `.ipynb` or `.ewd` notebook you did
  not save or trust yourself restricted (before, only `.ewd`), and asks before it runs. Saved and
  trusted files are remembered by content hash (`ewald.trust.is_trusted_file`/`remember_file`), so
  your own notebooks open and run without the question, and a saved `.ewd` reopens trusted in a
  script too. *Run notebooks when opened* = *always* now means the same as *ask*.
- Notebook cells in the app's own Python run on a worker thread: a long cell no longer freezes the
  window, and **Interrupt** (or `Notebook.interrupt()`) stops it with `CellInterrupted`, as it
  already did in another kernel. pyplot in cells draws off screen (Agg).
- Frames the pipeline caches are read-only: a modifier, plugin or script that writes into an array
  in place (`data.positions[0] += 1`) now raises `ValueError: assignment destination is read-only`
  instead of silently changing the cached frame and every later result. Replace the array instead
  (`data.positions = data.positions + d`). `DataCollection.freeze()` does this for any collection.

### Fixed

- The Windows app no longer crashes on exit while the Plugins side bar is still reading the
  catalogs, nor when started on a Qt platform without OpenGL (`QT_QPA_PLATFORM=offscreen`); the
  frozen app's release build passes its smoke tests again.
- Applying a new UI theme in Preferences restyles the open Preferences dialog too (its pages,
  icons, element table and switches), not only the main window; windows that draw theme colors
  themselves define `retheme()`. (#83)
- The window no longer freezes while the Plugins side bar installs, updates or removes a pip
  plugin: the package metadata read around pip (seconds, longer on Windows) moved off the GUI
  thread, Cancel also works before pip starts, and catalogs older than ten minutes are read
  again when the side bar is shown, so new versions get their *Update* button. (#80)
- No more small windows flashing up when the GUI starts or opens a notebook (dock title bars,
  tab groups and notebook bars were shown before they had a parent). (#81)
- Bond ends look the same in every renderer (D50): bonds cut by the periodic cell end in flat
  disks instead of hollow tubes in the viewport (and instead of round ends in the ray tracer),
  and bond ends at hidden, small or see-through atoms are round, so stick-only drawings have
  smooth joints in the viewport, the OpenGL 1.x viewport, the ray tracer and vector files.
- Every renderer looks alike in four more ways (D51): OpenGL pictures draw the cell light grey on
  dark backgrounds (it vanished in black before); lines are as wide in every renderer, scaling with
  the picture height (`cell_width` pixels at 1600 high, at least 1), and ray-traced one-pixel lines
  are solid instead of dotted (sub-pixel samples follow the R2 sequence); the flat (ASE) style is
  unlit with dark outlines in the ray tracer and the OpenGL 1.x viewport too; with no `light` set,
  every renderer uses the ray tracer's key light (the VTK viewport's light kit and the impostors'
  and OpenGL 1.x viewport's own head lights are gone), and polyhedra in the OpenGL 1.x viewport
  are grey instead of washed out.
- Plugins side bar *From file…* also takes a downloaded `.zip`, `.whl` or `.tar.gz` package and installs it with pip.
- Plugin index and example downloads trust the OS certificate store (`truststore`), so they work behind a
  company proxy that re-signs HTTPS ("certificate verify failed"). In a uv environment without pip, the
  Plugins panel installs with `uv pip ...`, asking uv for the OS store through
  `UV_NATIVE_TLS`/`UV_SYSTEM_CERTS` instead of a flag, so it works with every uv version
  (`error: unexpected argument '--system-certs'` on older uv).

### Changed

- Phonon support (phonopy and VASP IBRION=5-8 readers, `PhononMode`) and PLUMED support (`Plumed`,
  `LoadColvar`, `FreeEnergySurface`, `ewald.plumed`, the PLUMED Claude tools) moved to the
  first-party plugins `ewald-phonopy` and `ewald-plumed` in frssp/ewald-plugins. Old imports and
  sessions work once the plugin is installed; without it they raise an `ImportError` with the
  install command. The `plumed` extra is gone (#75).

### Fixed

- DOS, band structure and phonon band plots looked different when a notebook ran in another
  Python kernel: frames from a kernel now carry the `Electronic Structure` (and `Phonon Modes`)
  attribute, so the Plots tab and view cells draw them exactly as in-process.

### Added

- Plain matplotlib plots in notebook cells, notebook kernels and the GUI console use the
  publication style by default (`ewald.plotstyle.Session`: per-notebook rcParams swapped in around
  each cell, so rcParams a cell sets are kept and ewald's renderers are untouched). Preference
  `python.plot_style` (Publication, Publication in UI theme colors, Matplotlib default);
  `plotstyle.use("default")` opts a notebook out.
- Publication style for every plot (`ewald.plotstyle`): journal column sizes, 8 pt sans-serif
  type, inward ticks with minor ticks, axis labels with units, a colorblind-safe color cycle in
  Paper and Graphite variants, and editable text in PDF/SVG. *Save plot…* now redraws the plot as
  a print figure (white background, chosen width, PNG at the image-quality DPI) instead of
  saving the panel; new preferences `plots.size` and `plots.font_size`.
- UI theme plugins (`@ui_theme`, `ewald.ui_themes` entry points): color themes for the desktop app
  built on a built-in theme, chosen under Preferences › General › UI theme; example
  `examples/plugins/ui_themes.py` (Nord, Sepia).
- Built-in UI themes Midnight, Solarized light, Solarized dark and High contrast; Graphite (dark)
  is now the default. The Python console, open plots, notebook code colors and the activity bar
  follow a theme switch at once, and the plot color cycle starts with the theme's accent.
  The 3D view background follows the theme too (`viewport.follow_theme`, on by default).
- Claude tool plugins (`@claude_tool`, `ewald.tools` entry points); built-in formats listed as
  plugins that `ewald plugins --disable` can switch off (#75).

## [0.1.0] - 2026-10-19 (not released)

First release: EWALD (Extensible Workbench for Atomic Lattices & Dynamics), an atomistic
structure and trajectory viewer and analysis library with an OVITO-style pipeline API
(`import_file` → `pipe.modifiers` → `pipe.compute(frame, device=...)` → `pipe.render(...)`),
a PySide6 desktop GUI built around Python notebooks, and Claude tools.

### Core and compute

- Data model (`DataCollection`, `Cell`), pipeline with a frame cache, `PythonModifier`, headless
  Python API and the `ewald` command line (`render`, `movie`, `info`, `describe`, `examples`,
  `gui`, `ask`, `narrate`, `plugins`, `forge`, `mcp`, `api`, `inventory`, `--version`) (#3).
- Array backends: NumPy and CuPy behind `ctx.xp`, `device="auto"` picks CUDA when CuPy finds a GPU;
  numba CPU cell-list neighbor search and a CUDA `RawKernel` neighbor list (#3, #4, #66).
- C++ extension `ewald._native` with ports of OVITO CNA, PTM, Voronoi and CSP; ewald still installs
  without a C++ compiler, minus those analyses (#7, #49).
- Bonds in the data model, bond editing and coordination polyhedra (#16); per-particle vectors as
  arrows (#29); volumetric data with isosurfaces and slice planes (#19).
- Large systems: interactive level of detail, a memory cap on the frame cache, a parallel dump
  parser and faster dump indexing for multi-million-atom files (#17, #25).

### Readers and writers

- Native lazy readers for LAMMPS dump, XDATCAR, extxyz, XSF and H5MD; MDAnalysis for XTC/TRR/DCD;
  everything else through ASE, with a writer registry (#6, #23, #56).
- VASP and LAMMPS results (OUTCAR, vasprun.xml, OSZICAR, log.lammps) plotted against the timeline
  (#31); CHGCAR/LOCPOT/ELFCAR/cube (#19); VASP 6 `vaspout.h5`/`vaspwave.h5` and MDTraj/OpenMM HDF5
  (#39); DOS, PDOS and band structures (#47); phonopy `band.yaml` and OUTCAR phonon modes (#37);
  NEB runs as trajectories with the energy barrier synced to frames (#35).
- Files over SSH: `host:/path` and `ssh://` paths, and a remote file browser in the GUI (#21).
- Live reload of trajectories a running job is still writing (#59).
- Reader fixes found on real VASP and LAMMPS outputs (#68).

### Analysis modifiers

- Ported from OVITO: Common Neighbor Analysis, Polyhedral Template Matching, Voronoi, Wigner-Seitz,
  Centrosymmetry, Ackland-Jones, Identify Diamond, Chill+, atomic strain, cluster analysis with a
  GPU connected-components kernel, and a DXA bridge through the optional `ovito` package (#7, #9).
- OVITO Pro analyses: histogram, spatial binning, reduce, bond angles and lengths, rings,
  Steinhardt order parameters (#30); elastic strain, grain segmentation, fcc planar faults (#60);
  FreezeProperty, UnwrapTrajectories, ExpandSelection, LoadTrajectory, CombineDatasets (#55).
- Trajectory analysis: MSD and VACF diffusion, time averaging, time series, trajectory lines (#13);
  ion transport: conductivity and Haven ratio, van Hove, F_s(q,t), non-Gaussian parameter, site
  hopping (#58).
- Symmetry (spglib), powder X-ray and neutron diffraction, S(q) (#38).
- PLUMED collective variables: `Plumed`, `LoadColvar`, `FreeEnergySurface` (#32).
- Published analyses reproduced on paper data, with examples (#68, #69).

### Rendering and publication output

- Offscreen VTK rendering with GPU impostor spheres, FXAA and a software OpenGL warning (#42); a
  numba ray tracer as the publication renderer (#5).
- Vector SVG/PDF/EPS, color and type legends, scale bar, movies (GIF, MP4/AVI/MOV), `.ewald`
  sessions and Draft/Standard/Publication quality presets (#18).
- Radius presets: covalent, atomic, van der Waals and Shannon ionic (#24); VESTA element colors and
  VESTA-style boundary ghosts with bonds and polyhedra completed across the cell (#57); a Simple
  (ASE) style and Ball & stick with automatic bonds by default (#48).
- A renderer-neutral Scene IR with visual element, color map, theme and overlay plugins (#33, #36).
- Visual regression tests against reference images (#27).

### Desktop GUI

- PySide6 main window with an embedded IPython console, timeline, pipeline panel, Preferences
  backed by `ewald.settings` (#5, #10).
- Notebook layout: notebooks as percent-format `.py` files with Python cells, view cells, paint
  helpers and notebook tabs; a VS Code-style Plugins side bar; minimal mode (#63). Notebook files
  as `.ewd` (notebook, results, figures, data and plugins in one file, opened restricted until
  trusted) and `.ipynb` (#70).
- The UI style sheet is set only when it changes (#72).
- Linked plots: drag on a plot to select atoms or frames (#34); atom info on hover (#51);
  context-dependent right-click menu and toolbar icons (#45); example files to open first (#41).
- Phonon mode animation (click a band to play) (#37).
- Remote displays: OpenGL probe with a Mesa fallback and a legacy OpenGL 1.x viewport for `ssh -X`
  (#44, #49); readable Qt startup errors (#40).
- The EWALD app icon, an Ewald sphere (#54).

### Plugins and Claude

- Plugins for readers, writers, modifiers, visual elements, overlays, color maps and themes, from
  entry points or the user plugin folder (#14, #22, #33, #36); `ewald-plugin.toml` manifests with
  a `requires-ewald` check, enable/disable and `plugins.installed()` (#65); plugin catalogs with
  an Update button, compatibility check, a default index and GitHub tokens (#71).
- `ewald.llm`: structure descriptors, a Claude tool layer, an agent and an MCP server (#8); Modifier
  Forge (`ewald forge`) writes and verifies plugins (#22); Ask about selection (#26); API key in
  the OS keychain, model and effort in Preferences › Claude (#52).

### Packaging and docs

- Windows release build: wheels for Python 3.10 to 3.13, a standalone app (CPU and GPU) with an
  Inno Setup installer and portable zip (#15), now with the EWALD icon.
- macOS support: clang build, Finder file-open, `EWALD.app` in a `.dmg` for Apple Silicon and
  Intel, CPU fallback on Apple GPUs (#73).
- Release tooling: one version source (`ewald.__version__`), `ewald --version`, sdist and Linux
  wheels (`release.yml`), release notes from this file, and
  `packaging/release_check.py` to build and smoke-test a release locally before tagging.
- The manual (user guide, plugin author guide, developer guide, API reference) (#20), a feature
  inventory generated from the code and one TODO list (#28), `llms.txt` for LLM use.
- Security and license fixes from the code health review (#46).

[Unreleased]: https://github.com/frssp/ewald/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/frssp/ewald/releases/tag/v0.2.0
[0.1.0]: https://github.com/frssp/ewald/releases/tag/v0.1.0
