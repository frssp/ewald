"""Which GPU draws the 3D view on a laptop with two of them (integrated + NVIDIA/AMD).

Windows starts every program on the power-saving GPU unless the program asks for the other
one. Games ask by exporting ``NvOptimusEnablement`` / ``AmdPowerXpressRequestHighPerformance``
from their .exe; EWALD runs inside ``python.exe``, which exports neither, and a DLL loaded later
cannot add them. What Windows offers every program is the per-app choice of Settings › System ›
Display › Graphics, stored as ``GpuPreference=N;`` under ``HKCU\\Software\\Microsoft\\DirectX\\
UserGpuPreferences`` with the program's full path as the value name. :func:`set_windows_preference`
writes that value for the Python that runs EWALD (and its ``pythonw.exe``/venv base interpreter),
which is what Preferences › 3D view › GPU for the 3D view and ``ewald gpu --prefer`` do.

On Linux with NVIDIA PRIME render offload the same choice is environment variables set before the
first OpenGL context (:func:`linux_offload_env`). macOS switches GPUs itself.

CUDA compute (CuPy) does not depend on any of this: it talks to the NVIDIA GPU directly.
Qt-free, so the CLI and the GUI share it.
"""
from __future__ import annotations

import os
import subprocess
import sys

CHOICES = ("auto", "high-performance", "power-saving")
WINDOWS_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"
_WINDOWS_VALUE = {"auto": 0, "power-saving": 1, "high-performance": 2}
_INTEGRATED = ("intel", "uhd", "iris", "arc", "radeon(tm) graphics", "radeon graphics", "llvmpipe",
               "microsoft basic", "gdi generic")
_DISCRETE = ("nvidia", "geforce", "quadro", "rtx", "radeon rx", "radeon pro")


def env_choice(choice: str | None) -> str:
    """``EWALD_GPU`` (auto/high-performance/power-saving) overrides the setting."""
    value = (os.environ.get("EWALD_GPU") or choice or "auto").strip().lower()
    return value if value in CHOICES else "auto"


# --- Windows: the per-app GPU preference -------------------------------------------------------------
class _WinReg:  # pragma: no cover - Windows only; tests pass a dict-backed registry
    """The UserGpuPreferences key of the current user."""

    def get(self, name: str) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WINDOWS_KEY) as key:
                return str(winreg.QueryValueEx(key, name)[0])
        except OSError:
            return None

    def set(self, name: str, value: str):
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, WINDOWS_KEY) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete(self, name: str):
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WINDOWS_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except OSError:
            pass


def windows_executables(executable: str | None = None, base: str | None = None) -> list[str]:
    """The programs that may own EWALD's OpenGL context: ``sys.executable``, the base interpreter a
    venv's ``python.exe`` starts, and the ``pythonw.exe``/``python.exe`` next to each. A frozen app
    is its own .exe. Paths are Windows paths whatever OS runs the tests."""
    import ntpath

    exes = [executable or sys.executable]
    if not getattr(sys, "frozen", False) and executable is None:
        exes.append(base or getattr(sys, "_base_executable", "") or "")
    elif base:
        exes.append(base)
    out = []
    for exe in filter(None, exes):
        folder, name = ntpath.split(exe)
        names = [name]
        if name.lower() in ("python.exe", "pythonw.exe"):
            names = ["python.exe", "pythonw.exe"]
        for n in names:
            path = ntpath.join(folder, n)
            if path.lower() not in (p.lower() for p in out):
                out.append(path)
    return out


def _fields(value: str | None) -> dict:
    """``"GpuPreference=2;SwapEffectUpgradeEnable=1;"`` -> {"GpuPreference": "2", ...}."""
    fields = {}
    for part in (value or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def windows_preference(exe: str, reg=None) -> str:
    """The Windows choice for ``exe``: auto (Let Windows decide), power-saving or high-performance."""
    value = _fields((reg or _WinReg()).get(exe)).get("GpuPreference")
    return {"1": "power-saving", "2": "high-performance"}.get(value or "", "auto")


def set_windows_preference(choice: str, exes: list[str] | None = None, reg=None) -> list[str]:
    """Write ``GpuPreference`` for each program (keeping the other fields Windows stores there);
    "auto" removes it. Returns the paths whose value changed. Applies from the next start."""
    reg = reg or _WinReg()
    changed = []
    for exe in exes or windows_executables():
        old = reg.get(exe)
        fields = _fields(old)
        if choice == "auto":
            fields.pop("GpuPreference", None)
        else:
            fields["GpuPreference"] = str(_WINDOWS_VALUE[choice])
        new = "".join(f"{k}={v};" for k, v in fields.items())
        if new == (old or ""):
            continue
        if new:
            reg.set(exe, new)
        else:
            reg.delete(exe)
        changed.append(exe)
    return changed


# --- Linux: PRIME render offload ---------------------------------------------------------------------
def linux_offload_env(choice: str, nvidia: bool | None = None) -> dict:
    """Environment that sends OpenGL to the NVIDIA GPU (PRIME render offload), or {}."""
    if choice != "high-performance":
        return {}
    if nvidia is None:
        nvidia = os.path.isdir(os.path.join(os.sep, "proc", "driver", "nvidia", "gpus"))
    if not nvidia:
        return {"DRI_PRIME": "1"}   # Mesa: the other GPU (AMD/Intel discrete)
    return {"__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__VK_LAYER_NV_optimus": "NVIDIA_only"}


def apply_at_startup(choice: str | None, platform: str | None = None, environ=None, reg=None) -> str | None:
    """Before the first OpenGL context: carry the choice to the OS. On Windows the value only
    counts from the next start of Python, so a note is returned when it was just written."""
    choice = env_choice(choice)
    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    if choice == "auto":
        return None
    if platform == "win32":
        try:
            changed = set_windows_preference(choice, reg=reg)
        except Exception as exc:  # a locked-down registry must not stop the GUI
            return f"ewald-gui: could not set the Windows GPU preference ({exc})"
        if changed:
            return (f"ewald-gui: asked Windows to run Python on the {choice} GPU; "
                    "it takes effect when EWALD starts again.")
        return None
    if platform.startswith("linux"):
        for k, v in linux_offload_env(choice).items():
            environ.setdefault(k, v)
    return None


# --- what the machine has ----------------------------------------------------------------------------
_gpus_cache: list = []


def _run(args, timeout=10) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                             errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def list_gpus(platform: str | None = None, run=_run) -> list[str]:
    """Names of the GPUs in this computer (best effort, cached; [] when the OS does not say)."""
    if platform is None and _gpus_cache:
        return _gpus_cache[0]
    platform = platform or sys.platform
    names = []
    if platform == "win32":
        out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                   "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                   "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        names = [line.strip() for line in out.splitlines() if line.strip()]
    elif platform.startswith("linux"):
        out = run(["lspci"])
        for line in out.splitlines():
            if any(k in line for k in ("VGA compatible controller", "3D controller", "Display controller")):
                names.append(line.split(": ", 1)[-1].strip())
    if not any("nvidia" in n.lower() for n in names):
        names += [line.split(" (UUID")[0].split(": ", 1)[-1].strip()
                  for line in run(["nvidia-smi", "-L"]).splitlines() if line.startswith("GPU")]
    if platform == sys.platform:
        _gpus_cache[:] = [names]
    return names


def is_integrated(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in _INTEGRATED) and not any(k in low for k in _DISCRETE)


def other_gpu_hint(renderer: str, gpus: list[str], platform: str | None = None) -> str | None:
    """When the view draws on an integrated GPU and a discrete one is in the machine: what to do."""
    if not renderer or not is_integrated(renderer):
        return None
    discrete = [g for g in gpus if not is_integrated(g) and any(k in g.lower() for k in _DISCRETE)]
    if not discrete:
        return None
    platform = platform or sys.platform
    how = ("Preferences › 3D view › GPU for the 3D view: High performance (or `ewald gpu --prefer "
           "high-performance`), then start EWALD again")
    if platform == "darwin":
        how = "macOS picks the GPU itself; turn off Automatic graphics switching in Battery settings"
    return f"This computer also has {', '.join(discrete)}. To draw the 3D view there: {how}."


# --- battery or mains -------------------------------------------------------------------------------
def _windows_power() -> str | None:  # pragma: no cover - Windows only
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = _Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    if status.BatteryFlag == 128:      # no system battery: a desktop
        return None
    return {0: "battery", 1: "ac"}.get(status.ACLineStatus)


def _linux_power(root: str) -> str | None:
    """From ``/sys/class/power_supply``: "ac" when a mains/USB supply is online, "battery" when
    only batteries are there, None without any (a desktop or a container)."""
    def read(*parts):
        try:
            with open(os.path.join(root, *parts), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    kinds = {name: read(name, "type").lower() for name in names}
    if any(kind in ("mains", "usb", "usb_c", "usb_pd") and read(name, "online") == "1" for name, kind in kinds.items()):
        return "ac"
    if any(kind == "battery" and read(name, "scope").lower() != "device" for name, kind in kinds.items()):
        return "battery"
    return None


def _mac_power(run=None) -> str | None:
    first = ((run or _run)(["pmset", "-g", "batt"]).splitlines() or [""])[0]
    if "Battery Power" in first:
        return "battery"
    if "AC Power" in first:
        return "ac"
    return None


def power_source(platform: str | None = None, sys_root: str | None = None, run=None) -> str | None:
    """"battery", "ac" (plugged in) or None (no battery, or the OS does not say). Laptops lower
    the GPU's clocks on battery, and some switch the discrete GPU off, so the 3D view may draw on
    the integrated GPU even with High performance chosen."""
    platform = platform or sys.platform
    try:
        if platform == "win32":
            return _windows_power()
        if platform.startswith("linux"):
            return _linux_power(sys_root or os.path.join(os.sep, "sys", "class", "power_supply"))
        if platform == "darwin":
            return _mac_power(run)
    except Exception:  # never let a power probe break the GUI or the CLI
        return None
    return None


def wanted_gpu_missing(renderer: str, choice: str | None) -> str | None:
    """High performance was asked for but the 3D view got an integrated GPU (the discrete one is
    off, e.g. on battery or in a laptop's Eco/Hybrid mode, or Windows ignored the choice): a
    warning for the status bar, else None. Needs no GPU list, so it costs nothing at startup."""
    if env_choice(choice) != "high-performance" or not renderer or not is_integrated(renderer):
        return None
    power = power_source()
    why = ("the computer runs on battery, and laptops often switch the discrete GPU off then"
           if power == "battery" else "the discrete GPU may be switched off (a laptop's Eco or "
           "Hybrid mode) or Windows did not apply the choice yet")
    return (f"The 3D view draws on {renderer}, not the high-performance GPU: {why}. Maximizing or "
            "resizing the window can then take several seconds; plug in and start EWALD again, or "
            "see Help › About EWALD.")


def summary(renderer: str | None = None, platform: str | None = None) -> str:
    """Plain-text report for ``ewald gpu``: GPUs, the viewport preference, CUDA and OpenCL."""
    from .backend import gpu_available

    platform = platform or sys.platform
    lines = ["GPUs: " + (", ".join(list_gpus()) or "unknown")]
    power = power_source(platform)
    lines.append("Power: " + {"battery": "on battery (GPUs may run slower or switch off)",
                              "ac": "plugged in"}.get(power or "", "no battery or unknown"))
    if renderer:
        lines.append(f"Viewport OpenGL: {renderer}")
    if platform == "win32":
        for exe in windows_executables():
            lines.append(f"Windows graphics preference for {exe}: {windows_preference(exe)}")
    if os.environ.get("EWALD_GPU"):
        lines.append(f"EWALD_GPU={os.environ['EWALD_GPU']}")
    if gpu_available():
        import cupy

        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        lines.append("CUDA (CuPy): " + (name.decode() if isinstance(name, bytes) else name))
    else:
        lines.append("CUDA (CuPy): not available (pip install cupy-cuda12x, and an NVIDIA driver)")
    from .raytrace_gpu import opencl_devices

    devices = opencl_devices(gpu_only=True)
    lines.append("OpenCL GPUs: " + (", ".join(d.name.strip() for d in devices) or "none (or no pyopencl)"))
    return "\n".join(lines)
