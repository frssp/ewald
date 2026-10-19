# Roadmap

Goal: an OVITO-Pro-level structure and trajectory viewer for VASP and LAMMPS
output, with publication-quality figures, OVITO-style analysis, full Python
scripting (headless and inside the GUI) and NVIDIA GPU acceleration.

| Step | Scope | State |
|---|---|---|
| 1. Core | DataCollection, Pipeline (source → modifiers), lazy LAMMPS dump / XDATCAR readers, ASE fallback, first modifiers, offscreen PNG rendering, CLI | done |
| 2. GUI | OVITO-style layout with VESTA crystal controls: view along a/b/c, pick/measure, per-type styles, bond table, command palette, console, inspector, plots, Render workspace with code generator | done (PR #5); next: quad views, undo/redo, background compute thread |
| 3. Analysis | CNA, PTM, Voronoi, cluster, Wigner–Seitz (ported from OVITO's MIT-licensed source), RDF/MSD kernels, numba CPU and CuPy GPU neighbor lists, optional DXA via the `ovito` pip package | in progress |
| 4. Figures | Built-in numba ray tracer (AO, shadows, anti-aliasing) and Render workspace with journal presets done; next: GPU (CUDA) tracer, session save/load, PDF/vector overlays | in progress |
| Plugins | Reader plugins (`ewald.readers` entry points, `<config dir>/plugins/*.py`) used by `import_file`, GUI open and Claude tools; `ewald plugins` | readers done; next: modifier plugins, LLM Modifier Forge, Ask about selection |

Decisions

- NVIDIA CUDA is the GPU target (CuPy); every kernel also runs on NumPy, so tests cover the GPU code path without a GPU.
- Private, personal project without a LICENSE file for now; reuse freely, treating MIT compatibility as the bar when it matters.
- OVITO's public source is "GPLv3 or MIT"; ported code keeps the MIT notice. DXA is not in the public source.
- Python first, numba/C++ only for measured hot spots.
