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
below). On an Intel Arc GPU (Windows) the SrTiO3 example with TiO6 polyhedra took 1.13 s
at 1600x1200 Publication against 4.80 s on the CPU, and 108 000 fcc Cu atoms traced at 800x600 in
0.20 s. To see what the GPU gains, run `python benchmarks/analysis.py --skip-modifiers --device opencl` and
compare with `--device cpu`.

## Laptops with two GPUs

Many laptops pair an integrated GPU (Intel UHD/Iris/Arc, AMD Radeon Graphics) with an NVIDIA
GeForce RTX or AMD Radeon RX. Windows starts every program on the power-saving one unless the
program asks for the other, and EWALD runs inside `python.exe`, which does not ask. So the 3D
view and the OpenGL renderer draw on the integrated GPU; Help › About EWALD shows it as
*Viewport OpenGL*, lists the GPUs in the computer and says when another one is there. Games ask
by exporting `NvOptimusEnablement` or `AmdPowerXpressRequestHighPerformance` from their .exe;
a Python program cannot add those exports, so EWALD uses the per-program choice Windows offers
every app instead:

- Preferences › 3D view › Performance › *GPU for the 3D view*: **High performance**, then start
  EWALD again. On Windows this writes the same value as Settings › System › Display › Graphics
  for the Python that runs EWALD (`python.exe`, `pythonw.exe`, and for a venv the base
  interpreter it starts). **Auto** removes it again and leaves the choice to Windows.
- From a terminal: `ewald gpu --prefer high-performance` does the same, and `ewald gpu` prints
  the GPUs, the choice for each `python.exe`, and the CUDA and OpenCL devices.
- For one run: `EWALD_GPU=high-performance` (`set EWALD_GPU=high-performance` in cmd,
  `$env:EWALD_GPU="high-performance"` in PowerShell) overrides the setting.
- By hand: Windows Settings › System › Display › Graphics › Add desktop app, pick the
  `python.exe` (`python -c "import sys; print(sys.executable)"` prints it), Options › High
  performance. The NVIDIA Control Panel's per-program setting works too.

On Linux with the NVIDIA driver, High performance sets PRIME render offload
(`__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`) before the 3D view starts,
or `DRI_PRIME=1` on Mesa; variables you set yourself win. macOS switches GPUs on its own.

None of this affects CUDA: CuPy talks to the NVIDIA GPU directly, so `device="cuda"` runs on
the RTX even while the 3D view draws on the integrated GPU. With OpenCL a discrete GPU comes
before the integrated one.

To check a laptop like this (Windows, in the environment EWALD is installed in):

```bash
nvidia-smi                                   # the driver sees the RTX ("CUDA Version" 12 or newer)
pip install "cupy-cuda12x[ctk]"              # CuPy for CUDA 12 with the CUDA runtime and NVRTC from pip
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
ewald gpu                                    # GPUs, Windows choice, CUDA and OpenCL devices
python benchmarks/neighbors.py               # neighbor list, CPU vs CUDA
python benchmarks/analysis.py --device cuda  # modifiers and the ray tracer on the RTX
python benchmarks/analysis.py --skip-modifiers --device opencl   # compare the OpenCL tracer
ewald gpu --prefer high-performance          # then the 3D view on the RTX from the next start
python benchmarks/resize.py                  # prints "OpenGL: <renderer>"; compare the first maximize
```

Run the commands one at a time. EWALD's CUDA kernels are compiled when first used, with NVRTC, so
CuPy needs the CUDA runtime libraries: the `[ctk]` extra brings them as pip wheels, or install the
CUDA Toolkit 12.x from NVIDIA, which also sets `CUDA_PATH`. CuPy's warning "CUDA path could not be
detected" means neither is there.

`benchmarks/resize.py` uses a temporary settings folder, so compare the two GPUs with
`EWALD_GPU=power-saving` and `EWALD_GPU=high-performance` around a run each, and check that the
*OpenGL:* line names the GPU you expect (the Windows choice counts from the start after it was
written, so run once to write it and again to measure).

## CPU threads

The numba and C++ kernels use all cores by default. Preferences › Compute › CPU threads, or
`NUMBA_NUM_THREADS` in the environment, limits them.

## macOS

On a Mac the ray tracer runs on the GPU through Metal: `pip install 'ewald[metal]'` (PyObjC's
Metal bindings), then `device="auto"`, `"metal"` (or `"mps"`), or Apple GPU (Metal) in
Preferences › Compute › Device. It is the same kernel source as CUDA and OpenCL, in single
precision throughout, since Metal has no double. On an M2 MacBook Air (plugged in) Metal traced
108 000 fcc Cu atoms at 1600x1200 Publication in 2.6 s against 9.0 s on the CPU, 500 000 atoms
in 2.5 s against 11.0 s, and small pictures, see-through polyhedra and fine isosurfaces as fast
as the CPU. Apple's OpenCL works too (`device="opencl"`), but it traced a 400x300 Publication
picture in 2.4 s against 0.5 s on the CPU, so `auto` on a Mac is Metal when the `metal` extra
is installed and the CPU otherwise. On battery a Mac GPU may clock down and Metal times vary
more. The first Metal picture of a session waits a few seconds while Metal builds the kernel.

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
