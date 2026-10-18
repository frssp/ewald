"""Array backends: numpy on CPU, CuPy on NVIDIA GPUs.

Compute kernels are written once against an array module ``xp`` and run on
either device, so the GPU path is exercised by the same tests as the CPU path.
"""
from __future__ import annotations

import numpy as np


class Backend:
    def __init__(self, name: str, xp):
        self.name = name
        self.xp = xp

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a):
        if self.xp is np:
            return np.asarray(a)
        return self.xp.asnumpy(a)

    def __repr__(self):
        return f"<Backend {self.name}>"


CPU = Backend("cpu", np)


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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
        return Backend("cuda", cupy)
    raise ValueError(f"unknown device {device!r}; use 'cpu', 'cuda' or 'auto'")
