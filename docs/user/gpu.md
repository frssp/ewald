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
wins. Scripts default to the CPU. The GUI defaults to `auto` (Preferences › Compute ›
Default device): it computes on the GPU when CuPy finds one and on the CPU otherwise, and the
status bar shows which, for example `AUTO (CUDA)`. The Compute switch in the toolbar changes the
device for the session only.

## What runs on the GPU

Compute code in ewald is written once against an array module (`numpy` or `cupy`), so every
modifier runs on either device and gives the same results. The neighbor list, which most
analyses start from, has a CUDA C cell-list kernel on the GPU and a multithreaded numba kernel
on the CPU. CNA signatures, RDF, MSD/VACF FFTs, clustering and the trajectory kernels run on
the GPU. The OVITO-ported PTM, Voronoi and matching-mode CSP kernels are C++ and run on the
CPU threads even when `device="cuda"`.

Results come back as NumPy arrays in the `DataCollection`, so code after `compute` does not
change with the device.

## CPU threads

The numba and C++ kernels use all cores by default. Preferences › Compute › CPU threads, or
`NUMBA_NUM_THREADS` in the environment, limits them.

## macOS

Apple GPUs have no compute backend yet, so on a Mac analysis runs on the CPU: `device="auto"`
(the GUI default) picks the CPU, and `device="metal"` or `"mps"` stops with an error that says
so. The viewport and the OpenGL renderer still draw on the GPU. Apple Silicon CPUs are fast for
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
