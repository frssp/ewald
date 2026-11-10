"""Array backends: numpy on CPU, CuPy on NVIDIA GPUs.

Apple GPUs (Metal) run only the ray tracer (:mod:`ewald.raytrace_gpu`): with ``"metal"``/``"mps"``
the analysis stays on the CPU, and ``"auto"`` is the CPU on a Mac (docs/user/gpu.md, "Apple Silicon").

Compute kernels are written once against an array module ``xp`` and run on
either device, so the GPU path is exercised by the same tests as the CPU path.
"""
from __future__ import annotations

import difflib
import os
import posixpath
import sys

import numpy as np

#: Bytes one analysis step may allocate. ``None`` asks the machine (or the GPU) what is free now;
#: set a number where that reading is wrong, and in tests.
MEMORY_BUDGET: int | None = None
#: Assumed free when the platform does not say (no ``/proc``, no Windows API, no ``os.sysconf``).
FALLBACK_MEMORY = 8 * 2**30
#: Requests below this are never refused: not worth asking the system, and nothing useful to say.
SMALL_ALLOCATION = 64 * 2**20


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

    def short_of_memory(self, need: float) -> tuple[int, str] | None:
        """``None`` when ``need`` bytes can be allocated on this device now, else ``(bytes, words)``:
        what there is and how to say it in the error (``"14.2 GiB are free"``). Analyses ask before
        an allocation that grows with a parameter (a cutoff, a number of bins), so that a wrong
        value is refused with advice instead of failing inside numpy or being killed by the OS."""
        if MEMORY_BUDGET is not None:
            budget, words = int(MEMORY_BUDGET), "are allowed (ewald.backend.MEMORY_BUDGET)"
        elif not need >= SMALL_ALLOCATION:     # "not >=": a nan estimate is no reason to refuse either
            return None
        else:
            budget = _device_memory(self.xp) if self.xp is not np else None
            words = "are free on the GPU"
            if budget is None:
                budget, words = free_memory(), "are free"
            if budget is None:
                budget, words = FALLBACK_MEMORY, ("are assumed free (this system does not report its free memory; "
                                                  "set ewald.backend.MEMORY_BUDGET to the bytes it has)")
        return (budget, f"{format_bytes(budget)} {words}") if need > budget else None

    def __repr__(self):
        return f"<Backend {self.name}>"


CPU = Backend("cpu", np)


def format_bytes(nbytes: float) -> str:
    """``1.5 GiB``-style size for messages."""
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.0f} {unit}" if unit == "bytes" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, ValueError):
        return None


def _cgroup_limit(root: str = "/sys/fs/cgroup", membership: str = "/proc/self/cgroup") -> int | None:
    """The memory limit of this process's Linux control group and its parents, if any: a container
    or a batch job (SLURM) is killed at that limit however much the machine has free."""
    limit = None
    for line in (_read_text(membership) or "").splitlines():
        _, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if controllers == "":                                   # cgroup v2: "0::/path"
            base, name = root, "memory.max"
        elif "memory" in controllers.split(","):                # v1: "9:memory:/path"
            base, name = posixpath.join(root, "memory"), "memory.limit_in_bytes"
        else:
            continue
        while True:
            text = (_read_text(posixpath.join(base, path.lstrip("/"), name)) or "").strip()
            if text.isdigit():                                  # v2 says "max" when unlimited
                limit = int(text) if limit is None else min(limit, int(text))
            if path in ("", "/"):
                break
            path = posixpath.dirname(path)
    return limit


def _linux_memory(meminfo: str = "/proc/meminfo") -> int | None:
    fields = {}
    for line in (_read_text(meminfo) or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemAvailable", "SwapFree") and value.split()[:1]:
            fields[key] = int(value.split()[0]) * 1024          # the file counts in kB
    if "MemAvailable" not in fields:
        return None
    free = fields["MemAvailable"] + fields.get("SwapFree", 0)
    limit = _cgroup_limit()
    return free if limit is None else min(free, limit)


def _windows_memory() -> int | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)      # what this process may still commit: RAM plus page file


def free_memory() -> int | None:
    """Bytes of host memory a new allocation can still get, or ``None`` when the platform does
    not say. Standard library only, and never a fork or a subprocess: ``/proc/meminfo`` capped
    by the control-group limit on Linux, ``GlobalMemoryStatusEx`` on Windows, and elsewhere
    (macOS, BSD) the installed RAM, the only figure ``os.sysconf`` has. Swap counts as free:
    this is what can be allocated at all, so that nothing that would have run is refused."""
    try:
        if sys.platform == "win32":
            return _windows_memory()
        if sys.platform.startswith("linux"):
            free = _linux_memory()
            if free is not None:
                return free
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:      # a probe that fails must never be what stops an analysis
        pass
    return None


def _device_memory(xp) -> int | None:
    """Free bytes on the current CUDA device, counting what CuPy's pool holds for reuse; ``None``
    for an array module without that interface."""
    try:
        return int(xp.cuda.Device().mem_info[0]) + int(xp.get_default_memory_pool().free_bytes())
    except Exception:
        return None


_DLL_DIRS: set = set()


def cuda_dll_dirs(paths=None) -> list[str]:
    """Windows: put the ``nvidia/<lib>/bin`` folders of pip's CUDA wheels (``cupy-cuda12x[ctk]``,
    also when installed into the standalone app's add-on folder) on the DLL search path before
    CuPy loads, so only an NVIDIA driver is needed. Returns the folders added; elsewhere none."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []
    added = []
    for entry in sys.path if paths is None else paths:
        nvidia = os.path.join(entry or ".", "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for name in sorted(os.listdir(nvidia)):
            for b in (os.path.join(nvidia, name, "bin"), os.path.join(nvidia, name, "bin", "x64")):
                if os.path.isdir(b) and b not in _DLL_DIRS:
                    os.add_dll_directory(b)
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                    _DLL_DIRS.add(b)
                    added.append(b)
    return added


def gpu_available() -> bool:
    """True when CuPy is installed and sees an NVIDIA GPU, so ``device="cuda"`` will run there."""
    try:
        cuda_dll_dirs()
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


APPLE_DEVICES = ("metal", "mps", "apple")
#: Every name :func:`get_backend` takes ("gpu" is another word for "cuda").
DEVICES = ("cpu", "cuda", "gpu", "auto", "opencl") + APPLE_DEVICES


def get_backend(device: str | Backend | None = None) -> Backend:
    """``None``/``"cpu"`` -> numpy; ``"cuda"``/``"gpu"`` -> CuPy; ``"auto"`` -> GPU if present.
    ``"opencl"`` and ``"metal"``/``"mps"`` -> numpy: only the ray tracer runs on OpenCL or Metal
    (:mod:`ewald.raytrace_gpu`), the analysis stays on the CPU."""
    if isinstance(device, Backend):
        return device
    device = device or "cpu"
    if not isinstance(device, str):
        raise TypeError(f"device must be a name or a Backend, not {device!r}. Devices: {', '.join(DEVICES)}")
    device = device.lower()
    if device in ("cpu", "opencl") or device in APPLE_DEVICES:
        return CPU
    if device == "auto":
        return get_backend("cuda") if gpu_available() else CPU
    if device in ("cuda", "gpu"):
        try:
            cuda_dll_dirs()
            import cupy
        except ImportError as exc:
            raise RuntimeError("device='cuda' needs CuPy: Preferences › Compute › Optional components › "
                               "Install, or pip install 'ewald[gpu]'") from exc
        if not gpu_available():
            raise RuntimeError("CuPy is installed but no CUDA device was found")
        return Backend("cuda", cupy, raw_kernels=True)
    close = difflib.get_close_matches(device, DEVICES, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise ValueError(f"unknown device {device!r}.{hint} Devices: {', '.join(DEVICES)}")
