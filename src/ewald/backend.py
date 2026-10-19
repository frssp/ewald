"""Array backends: numpy on CPU, CuPy on NVIDIA GPUs.

Apple GPUs (Metal) have no compute backend yet: ``"auto"`` runs on the CPU on a Mac and
``"metal"``/``"mps"`` raise an error that says so (docs/user/gpu.md, "Apple Silicon").

Compute kernels are written once against an array module ``xp`` and run on
either device, so the GPU path is exercised by the same tests as the CPU path.
"""
from __future__ import annotations

import numpy as np


class Backend:
    def __init__(self, name: str, xp, raw_kernels: bool = False):
        self.name = name
        self.xp = xp
        # True when ``xp`` can compile CUDA C (CuPy RawKernel), see cuda_kernels.py
        self.raw_kernels = raw_kernels

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a):
        if self.xp is np:
            return np.asarray(a)
        return self.xp.asnumpy(a)

    def synchronize(self):
        """Wait for queued device work (for timing); no-op on CPU."""
        if self.raw_kernels:
            self.xp.cuda.Device().synchronize()

    def __repr__(self):
        return f"<Backend {self.name}>"


CPU = Backend("cpu", np)


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


APPLE_DEVICES = ("metal", "mps", "apple")
APPLE_GPU_MESSAGE = ("Apple GPUs (Metal) are not supported for compute yet; use device='cpu' "
                     "(or 'auto', which picks the CPU on a Mac). The numba and C++ kernels use every CPU core, "
                     "and the viewport and renderer still draw on the GPU.")


def get_backend(device: str | Backend | None = None) -> Backend:
    """``None``/``"cpu"`` -> numpy; ``"cuda"``/``"gpu"`` -> CuPy; ``"auto"`` -> GPU if present."""
    if isinstance(device, Backend):
        return device
    device = (device or "cpu").lower()
    if device == "cpu":
        return CPU
    if device == "auto":
        return get_backend("cuda") if gpu_available() else CPU
    if device in ("cuda", "gpu"):
        try:
            import cupy
        except ImportError as exc:
            raise RuntimeError("device='cuda' needs CuPy: pip install 'ewald[gpu]' "
                               "(or cupy-cuda12x matching your CUDA)") from exc
        if not gpu_available():
            raise RuntimeError("CuPy is installed but no CUDA device was found")
        return Backend("cuda", cupy, raw_kernels=True)
    if device in APPLE_DEVICES:
        raise RuntimeError(APPLE_GPU_MESSAGE)
    raise ValueError(f"unknown device {device!r}; use 'cpu', 'cuda' or 'auto'")
