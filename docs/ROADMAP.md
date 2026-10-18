# Roadmap

Goal: an OVITO-Pro-level structure and trajectory viewer for VASP and LAMMPS
output, with publication-quality figures, OVITO-style analysis, full Python
scripting (headless and inside the GUI) and NVIDIA GPU acceleration.

| Step | Scope | State |
|---|---|---|
| 1. Core | DataCollection, Pipeline (source → modifiers), lazy LAMMPS dump / XDATCAR readers, ASE fallback, first modifiers, offscreen PNG rendering, CLI | done |
| 2. GUI | PySide6 + VTK viewport, timeline, pipeline panel with parameter forms and Python script editor, render settings, data inspector, IPython console | first version done; UX redesign (OVITO/VESTA reference) in progress |
| 3. Analysis | CNA, PTM, Voronoi, cluster, Wigner–Seitz (ported from OVITO's MIT-licensed source), RDF/MSD kernels, numba CPU and CuPy GPU neighbor lists, optional DXA via the `ovito` pip package | in progress |
| 4. Figures | Ray-traced rendering (OSPRay / OptiX), ambient occlusion, legends, animation export, session save/load | open |

Decisions

- NVIDIA CUDA is the GPU target (CuPy); every kernel also runs on NumPy, so tests cover the GPU code path without a GPU.
- Private, personal project without a LICENSE file for now; reuse freely, treating MIT compatibility as the bar when it matters.
- OVITO's public source is "GPLv3 or MIT"; ported code keeps the MIT notice. DXA is not in the public source.
- Python first, numba/C++ only for measured hot spots.
