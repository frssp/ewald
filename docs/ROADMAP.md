# Roadmap

Goal: an OVITO-Pro-level structure and trajectory viewer for VASP and LAMMPS
output, with publication-quality figures, OVITO-style analysis, full Python
scripting (headless and inside the GUI) and NVIDIA GPU acceleration.

This page holds the milestones and decisions. Open tasks are in the [TODO list](todo.md) and
what exists is in the [feature inventory](features.md).

| Step | Scope | State |
|---|---|---|
| 1. Core | DataCollection, Pipeline (source → modifiers), lazy LAMMPS dump / XDATCAR readers, ASE fallback, first modifiers, offscreen PNG rendering, CLI | done |
| 1b. Remote files | `host:/path` / `ssh://` paths in `import_file` and a GUI SSH browser; `~/.ssh/config` incl. ProxyJump, agent/keys, password/OTP prompt (never stored); dumps and XDATCARs read in place (remote `grep -b` index), other formats cached locally | done |
| 2. GUI | OVITO-style layout with VESTA crystal controls: view along a/b/c, pick/measure, per-type styles, bond table, command palette, console, inspector, plots, Render workspace with code generator | done (PR #5) |
| 3. Analysis | CNA, PTM, Voronoi, cluster, Wigner–Seitz (ported from OVITO's MIT-licensed source), RDF/MSD kernels, numba CPU and CuPy GPU neighbor lists, optional DXA via the `ovito` pip package; trajectory analysis: MSD/VACF with diffusion coefficients, time averaging (g(r)), time series, trajectory lines and path metrics; bonds in the data model (`data.bonds`, CreateBonds/EditBonds, GUI bond tool) and coordination polyhedra in `data.surfaces`; Histogram, SpatialBinning, ReduceProperty, bond angle/length distributions (OVITO-exact), FindRings (shortest-path rings), Steinhardt q_l/w_l | in progress |
| 3b. Volumetric data | `VoxelGrid` in `data.grids`; readers for VASP CHGCAR/CHG/PARCHG/AECCAR (spin and non-collinear), LOCPOT, ELFCAR and Gaussian cube, writers for cube/CHGCAR; `CreateIsosurface` (signed, VESTA colours), `GridSlice` (hkl plane, linear/log colour map), `GridDifference` (charge-density difference); drawn by the OpenGL viewport/renderer and the ray tracer (triangles with translucency) | done |
| 4. Figures | Built-in numba ray tracer (AO, shadows, anti-aliasing), Render workspace with journal presets, vector renderer (SVG/PDF/EPS), overlays on every renderer (text, color legend, type legend, scale bar), movies (MP4/GIF/AVI/MOV/PNG sequence), `.ewald` session save/load (GUI and CLI), Draft/Standard/Publication quality presets (Publication default for file output), per-particle vector arrows (`VectorVis`: forces, displacements, velocities, magnetic moments) in the viewport, ray tracer and SVG/PDF done | in progress |
| Plugins | One reader registry for built-in formats and plugins (`ewald.readers` entry points, `<config dir>/plugins/*.py`) used by `import_file`, GUI open and Claude tools; modifier plugins (`ewald.modifiers` entry points, `@modifier`) in the GUI menu, Claude tools and `ewald.modifiers`; `ewald plugins`; Modifier Forge (`ewald forge`: Claude writes a modifier and its tests, ewald verifies them in a subprocess and saves them); Ask about selection (GUI Ask Claude tab on the picked atoms, `ewald ask --select`, `describe_selection` tool) | readers, writers (`@writer`, `ewald.writers`), modifiers, Modifier Forge (`ewald forge`), Ask about selection and visual elements (Scene IR in `ewald.scene`, `@visual`, `ewald.visuals`, drawn by every renderer) done |
| Results | Native lazy OUTCAR and vasprun.xml readers (forces, magnetic moments, energies, stress, MD temperature), OSZICAR and log.lammps (multi-run) thermo tables matched to frames by step and found next to the trajectory (also over SSH), GUI Plots time series with a current-frame marker | done; next: DOS/band plots, `thermo_style multi`/YAML logs, click a plot to jump to its frame |
| Formats | Native lazy extxyz, XSF/AXSF (with grids), H5MD; XTC/TRR/DCD/NetCDF via MDAnalysis; ASE db and `.traj`; writers for POSCAR, LAMMPS data, extxyz, XSF, H5MD, XTC/TRR/DCD, ASE db/traj, CIF; `formats` extra (h5py, MDAnalysis) | done |
| Scale | Multi-million-atom LAMMPS dumps and long XDATCARs: `benchmarks/large_files.py` measures open, frame switch and memory; dump indexing reads in chunks (2M atoms x 4 frames: open 13 s → 0.8 s, frame switch 3.0 s → 1.5 s); frame-cache memory budget (`cache_memory`, Preferences › Compute); interactive level of detail in the viewport (coarse while moving or scrubbing, refined when idle; detail mode, target fps, atom budget, refine delay in Preferences › Viewports); per-type color/radius tables (2M-atom scene build 0.9 s → 0.16 s); parallel numba atom-table parser, bit-identical to `np.loadtxt` (2M-atom frame switch 1.5 s → 0.66 s) | in progress |
| Docs | Manual in `docs/` (user manual, plugin author guide, developer guide, API reference), MkDocs build in CI, GUI Help menu | in progress |

Decisions

- NVIDIA CUDA is the GPU target (CuPy); every kernel also runs on NumPy, so tests cover the GPU code path without a GPU.
- Private, personal project without a LICENSE file for now; reuse freely, treating MIT compatibility as the bar when it matters.
- OVITO's public source is "GPLv3 or MIT"; ported code keeps the MIT notice. DXA is not in the public source.
- Python first, numba/C++ only for measured hot spots.
- Windows distribution: wheels plus a PyInstaller app with an Inno Setup installer, in CPU and CUDA variants, built by `windows-release.yml` on `v*` tags (docs/developer/windows-release.md).
