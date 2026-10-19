# Compute backends

`ewald.backend.Backend` bundles an array module `xp` (NumPy or CuPy), `to_numpy`, `asarray`,
`synchronize` and a `raw_kernels` flag that says whether `xp` can compile CUDA C.
`get_backend(device)` maps `None`/`"cpu"`, `"cuda"`/`"gpu"` and `"auto"` to a backend, and
passes a `Backend` instance through unchanged.

## Rules for compute code

1. Write array code against `ctx.xp` (in modifiers) or `backend.xp` (in kernels), never against
   `numpy` directly, for anything that scales with the particle count.
2. Accept `backend=None` in kernel functions and call `get_backend(backend)` at the top.
3. Return results to `DataCollection` as NumPy arrays (`backend.to_numpy`). Everything after
   `compute` is device-independent.
4. Avoid NumPy-only functions and behaviors (object arrays, `np.add.at` on large inputs,
   in-place fancy-index updates with repeats). `kernels._scatter_min` shows a portable scatter.
5. Keep backends pluggable. NVIDIA CUDA through CuPy is the target now; an Apple Metal backend
   may follow, so code must not assume CUDA beyond the `raw_kernels` branch.

## CUDA C kernels

Where vectorised array code is too slow on the GPU, `cuda_kernels.py` holds CUDA C source that
CuPy compiles with `RawKernel`. The neighbor list is the main one: a two-pass cell list (count,
prefix sum, fill and sort per row), the same algorithm as the numba CPU kernel in
`neighbors.py`. Keep kernel sources plain C++ apart from `__global__` and the thread index, so
they can be tested on the CPU.

## Testing without a GPU

CI and the cloud containers have no GPU. Two things keep the CUDA path tested:

- The array code runs with NumPy as `xp`, which exercises the same lines a CuPy run does.
  Tests pass `Backend("name", np)` objects to force the backend code paths.
- `tests/test_cuda_kernel.py` compiles the CUDA C source with g++ behind a small shim
  (`__global__` removed, a loop standing in for the thread grid) and replaces `RawKernel` with
  a ctypes wrapper, so dispatch, prefix sums and output layout run as on a GPU.

## CPU kernels

The CPU neighbor list is numba (`@njit(parallel=True)`), multithreaded over atoms. The
OVITO-ported structure analyses are C++ (see [Native code](native.md)) and use their own
threads. `python benchmarks/neighbors.py` prints neighbor-list timings for every method.
