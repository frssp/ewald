"""Smoke test for GPU compute installed into the frozen app: ``ewald run smoke_gpu.py``, after
``ewald.exe gpu --install`` (what Preferences › Compute › Optional components › Install runs).

The CI runner has no NVIDIA GPU, so this checks what can be checked without one: a new ewald
process imports CuPy from the add-on folder (every standard library module it needs is bundled,
the CUDA wheels' DLL folders are found) and falls back to the CPU for ``device="auto"``.
"""
import subprocess

from ewald import plugin_catalog as pc

CHECK = """
import cupy, cupyx, ewald
from ewald import backend
print("cupy", cupy.__version__, "from", cupy.__file__)
print("CUDA DLL folders:", len(backend.cuda_dll_dirs()) + len(backend._DLL_DIRS))
assert backend._DLL_DIRS, "no nvidia/*/bin folder from cupy-cuda12x[ctk]"
print("GPU found:", backend.gpu_available())
print("auto ->", backend.get_backend("auto").name)
"""

print(">", pc.app_python(), "-c CHECK", flush=True)
subprocess.run([pc.app_python(), "-c", CHECK], check=True)
print("GPU SMOKE OK")
