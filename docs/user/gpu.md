# GPU compute

Analysis runs on an NVIDIA GPU through CuPy when you pass `device="cuda"`:

```python
data = pipe.compute(10, device="cuda")
```

Install CuPy with the `gpu` extra (CUDA 12) or the CuPy wheel that matches your CUDA. The
device can be:

| `device` | Meaning |
|---|---|
| `None` or `"cpu"` | NumPy on the CPU (default) |
| `"cuda"` or `"gpu"` | CuPy; raises a clear error when CuPy or a GPU is missing |
| `"auto"` | the GPU when one is available, else the CPU |

Set a default for a pipeline with `pipe.device = "cuda"`; a `device=` passed to `compute`
wins. Scripts default to the CPU, and so does the GUI. Preferences › Compute › Device switches
the GUI to `cuda` or `auto` (the GPU when CuPy finds one, else the CPU); the status bar shows
which, for example `AUTO (CUDA)`. The notebook's `pipe` follows that setting, so a cell's
`pipe.compute()` runs on the same device as the window, unless the cell sets `pipe.device`
itself.

## What runs on the GPU

Compute code in ewald is written once against an array module (`numpy` or `cupy`), so every
modifier runs on either device and gives the same results. The neighbor list, which most
analyses start from, has a CUDA C cell-list kernel on the GPU and a multithreaded numba kernel
on the CPU. CNA signatures, RDF, MSD/VACF FFTs, clustering and the trajectory kernels run on
the GPU. The OVITO-ported PTM, Voronoi and matching-mode CSP kernels are C++ and run on the
CPU threads even when `device="cuda"`.

Results come back as NumPy arrays in the `DataCollection`, so code after `compute` does not
change with the device.

The ray tracer runs there too: `render(data, "fig.png", renderer="raytrace", device="cuda")`,
or `pipe.render(...)` with `pipe.device` set, and the GUI's Render page and movie export when
Preferences › Compute › Device is `cuda` or `auto`. It traces in single precision
([Rendering](rendering.md)).

On other GPUs (Intel Arc and integrated graphics, AMD, Apple) the ray tracer runs through
OpenCL: `pip install 'ewald[opencl]'`, then `device="opencl"`, or OpenCL in Preferences ›
Compute › Device. The GPU's driver brings the OpenCL runtime (on Windows the Intel, AMD and
NVIDIA graphics drivers do). Only the ray tracer uses OpenCL; the analysis runs on the CPU
with this device. `device="auto"` picks CUDA first, then an OpenCL GPU for ray tracing (on a Mac, Metal; see
below). To see
what the GPU gains, run `python benchmarks/analysis.py --skip-modifiers --device opencl` and
compare with `--device cpu`.

## CPU threads

The numba and C++ kernels use all cores by default. Preferences › Compute › CPU threads, or
`NUMBA_NUM_THREADS` in the environment, limits them.

## macOS

On a Mac the ray tracer runs on the GPU through Metal: `pip install 'ewald[metal]'` (PyObjC's
Metal bindings), then `device="metal"` (or `"mps"`, or `"auto"`), or Apple GPU (Metal) in
Preferences › Compute › Device. It is the same kernel source as CUDA and OpenCL, in single
precision throughout, since Metal has no double. Apple's OpenCL works too (`device="opencl"`),
but on an M2 MacBook Air it traced a 400x300 Publication picture in 2.4 s against 0.5 s on the
CPU, so `auto` never picks it. On that Mac Metal traced atoms about twice as fast as the CPU
(108 000 fcc Cu atoms at 800x600, Publication: 0.85 s against 1.64 s) and see-through
polyhedra and fine isosurfaces about as fast; the first picture of a session waits a few
seconds while Metal builds the kernel.

The analysis has no Apple GPU backend yet: with `metal`, `mps` or `auto` it runs on the CPU.
The viewport and the OpenGL renderer still draw on the GPU. Apple Silicon CPUs are fast for
this work: the numba neighbor list and the C++ kernels use every performance and efficiency
core (Preferences › Compute › CPU threads).

A Metal backend would go into `backend.py` next to CuPy. The candidates, MLX and PyTorch's MPS
device, do not cover the NumPy subset the kernels use (boolean masks and other data-dependent
shapes, `bincount`, scatter-add), so they would need their own versions of the neighbor list
and the other kernels. The dense ones (powder diffraction and S(q), atoms x reflections) would
gain the most and are the likely first step.

## Timing

```bash
python benchmarks/neighbors.py      # neighbor-list timings on this machine, CPU and GPU
```
