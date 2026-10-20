# Changelog

All notable changes to ewald. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/) (before 1.0, a minor version may
change the API). Pull request numbers refer to <https://github.com/frssp/ewald/pulls>.
How to cut a release is in [docs/developer/release.md](docs/developer/release.md).

## [Unreleased]

### Added

- Plugin catalogs can give a `.py` entry a `sha256`; *Install* refuses a download that does not
  match it. `ewald.trust` collects the code-execution checks (`UntrustedCode`, a
  `PermissionError`, is raised for an untrusted `.ewd` notebook).

### Changed

- Notebook cells in the app's own Python run on a worker thread: a long cell no longer freezes the
  window, and **Interrupt** (or `Notebook.interrupt()`) stops it with `CellInterrupted`, as it
  already did in another kernel. pyplot in cells draws off screen (Agg).
- Frames the pipeline caches are read-only: a modifier, plugin or script that writes into an array
  in place (`data.positions[0] += 1`) now raises `ValueError: assignment destination is read-only`
  instead of silently changing the cached frame and every later result. Replace the array instead
  (`data.positions = data.positions + d`). `DataCollection.freeze()` does this for any collection.

### Fixed

- Plugin index and example downloads trust the OS certificate store (`truststore`), so they work behind a
  company proxy that re-signs HTTPS ("certificate verify failed"). In a uv environment without pip, the
  Plugins panel installs with `uv pip ... --system-certs`.

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

## [0.1.0] - 2026-10-19

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

[Unreleased]: https://github.com/frssp/ewald/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/frssp/ewald/releases/tag/v0.1.0
