# ewald

Atomistic structure and trajectory visualization and analysis with an OVITO-style
pipeline API in Python. Goal: an open alternative to OVITO Pro (GUI + Python
scripting + GPU), built on open libraries (ASE, VTK/PyVista, NumPy/CuPy).

Status: **step 1 of 4** (core data model, readers, pipeline, first modifiers,
offscreen rendering). The desktop GUI, the full analysis set and the
publication renderer follow.

## Install

```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # + CuPy for device="cuda" (NVIDIA)
```

Headless servers need an offscreen OpenGL (e.g. `apt install libosmesa6` or EGL).

## Quick start

```python
from ewald import import_file
from ewald.modifiers import CoordinationAnalysis, ColorCoding, Slice

pipe = import_file("dump.lammpstrj", type_map={1: "Cu", 2: "Zr"})   # or "XDATCAR", "OUTCAR", "POSCAR", ...
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.1))
pipe.modifiers.append(ColorCoding("Coordination", cmap="coolwarm_r"))
pipe.modifiers.append(Slice(normal=(0, 1, 0), distance=20.0, inverse=True))

def my_modifier(frame, data):            # OVITO-compatible Python modifier
    data["Undercoordinated"] = data["Coordination"] < 12
pipe.modifiers.append(my_modifier)

data = pipe.compute(10, device="cuda")   # analysis on the GPU (falls back with a clear error if no CuPy)
pipe.render("frame10.png", frame=10, size=(2000, 1500), direction="iso", bonds=None)
```

Command line:

```bash
ewald info XDATCAR
ewald render dump.lammpstrj out.png --frame 5 --type-map 1=Cu,2=Zr --color-by "Position.Z"
```

`python examples/demo.py` builds sample LAMMPS and VASP trajectories and renders
example figures.

## Design

* `DataCollection`: per-particle arrays with OVITO property names
  (`Position`, `Particle Type`, `Particle Identifier`, `Color`, `Radius`, ...),
  plus `cell`, `attributes`, `tables`.
* `Pipeline`: source + modifier list, evaluated lazily per frame; source frames are
  cached and never mutated by modifiers.
* Readers: native lazy readers for LAMMPS text dumps (orthogonal/triclinic,
  scaled/unwrapped coordinates, wildcards) and VASP XDATCAR (fixed and
  variable cell); everything else through ASE.
* Compute backends: kernels are written against an array module, so the same
  code runs on NumPy (CPU) and CuPy (CUDA). The neighbor list has a compiled,
  multithreaded numba kernel on CPU (256k-atom fcc, 3 Å cutoff: ~0.3–0.6 s vs ~5 s
  pure NumPy) and the same algorithm as a CUDA C kernel on GPU (CuPy `RawKernel`,
  with the vectorized array code as fallback, `method="xp"`). Timings:
  `python benchmarks/neighbors.py`; `ewald.kernels` adds total and
  partial RDF, trajectory unwrapping and FFT time-averaged MSD
  (`trajectory_msd(pipe, device="cuda")`).
* Rendering: VTK via PyVista, offscreen; sphere meshes for figures and GPU
  impostors above 150k atoms.

## Modifiers available now

`ExpressionSelection`, `SelectType`, `InvertSelection`, `ClearSelection`,
`DeleteSelected`, `Slice`, `AssignColor`, `ColorCoding`, `ComputeProperty`,
`CoordinationAnalysis` (+ total/partial RDF), `CalculateDisplacements` (+ MSD),
`Replicate`, `WrapAtoms`, `AffineTransformation`, and any Python function
`f(frame, data)`.
