"""GPU choice on laptops with two GPUs (Windows Graphics settings, Linux PRIME), simulated here."""
import os
import sys

import pytest

from ewald import gpu_select


class FakeReg:
    """Dict-backed stand-in for HKCU\\...\\UserGpuPreferences (value names are full paths)."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


VENV = r"C:\Users\me\proj\.venv\Scripts\python.exe"
BASE = r"C:\Users\me\AppData\Local\Programs\Python\Python312\python.exe"


def test_windows_executables_cover_pythonw_and_venv_base():
    exes = gpu_select.windows_executables(VENV, BASE)
    assert exes == [VENV, VENV.replace("python.exe", "pythonw.exe"), BASE, BASE.replace("python.exe", "pythonw.exe")]
    assert gpu_select.windows_executables(r"C:\Program Files\EWALD\EWALD.exe") == [r"C:\Program Files\EWALD\EWALD.exe"]
    # the same interpreter twice (no venv) is listed once, ignoring case
    assert len(gpu_select.windows_executables(BASE, BASE.upper())) == 2


def test_set_windows_preference_keeps_other_fields_and_auto_removes():
    reg = FakeReg({BASE: "SwapEffectUpgradeEnable=1;"})
    exes = [BASE, VENV]
    assert gpu_select.set_windows_preference("high-performance", exes, reg) == exes
    assert reg.values[BASE] == "SwapEffectUpgradeEnable=1;GpuPreference=2;"
    assert reg.values[VENV] == "GpuPreference=2;"
    assert gpu_select.windows_preference(BASE, reg) == "high-performance"
    assert gpu_select.set_windows_preference("high-performance", exes, reg) == []   # nothing to change
    gpu_select.set_windows_preference("power-saving", exes, reg)
    assert gpu_select.windows_preference(VENV, reg) == "power-saving"
    gpu_select.set_windows_preference("auto", exes, reg)
    assert reg.values == {BASE: "SwapEffectUpgradeEnable=1;"}
    assert gpu_select.windows_preference(VENV, reg) == "auto"


def test_apply_at_startup_windows_and_linux(monkeypatch):
    monkeypatch.delenv("EWALD_GPU", raising=False)
    monkeypatch.setattr(gpu_select, "windows_executables", lambda: [BASE])
    reg = FakeReg()
    assert gpu_select.apply_at_startup("auto", "win32", {}, reg) is None and reg.values == {}
    note = gpu_select.apply_at_startup("high-performance", "win32", {}, reg)
    assert "next" in note or "again" in note
    assert reg.values == {BASE: "GpuPreference=2;"}
    assert gpu_select.apply_at_startup("high-performance", "win32", {}, reg) is None   # already set

    env = {"__GLX_VENDOR_LIBRARY_NAME": "mesa"}   # the user's own setting wins
    monkeypatch.setattr(gpu_select, "linux_offload_env",
                        lambda c: {"__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia"})
    gpu_select.apply_at_startup("high-performance", "linux", env)
    assert env == {"__GLX_VENDOR_LIBRARY_NAME": "mesa", "__NV_PRIME_RENDER_OFFLOAD": "1"}

    monkeypatch.setenv("EWALD_GPU", "power-saving")
    gpu_select.apply_at_startup("high-performance", "win32", {}, reg)
    assert reg.values == {BASE: "GpuPreference=1;"}
    monkeypatch.setenv("EWALD_GPU", "nonsense")
    assert gpu_select.env_choice("high-performance") == "auto"


def test_apply_at_startup_survives_a_locked_registry(monkeypatch):
    class Locked(FakeReg):
        def set(self, name, value):
            raise PermissionError("access denied")

    monkeypatch.delenv("EWALD_GPU", raising=False)
    monkeypatch.setattr(gpu_select, "windows_executables", lambda: [BASE])
    assert "could not" in gpu_select.apply_at_startup("high-performance", "win32", {}, Locked())


def test_linux_offload_env():
    assert gpu_select.linux_offload_env("auto", nvidia=True) == {}
    assert gpu_select.linux_offload_env("high-performance", nvidia=True)["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert gpu_select.linux_offload_env("high-performance", nvidia=False) == {"DRI_PRIME": "1"}


def test_list_gpus_parses_each_os():
    win = "Intel(R) Arc(TM) Graphics\r\nNVIDIA GeForce RTX 4060 Laptop GPU\r\n"   # CRLF from PowerShell

    def run(args, timeout=10):
        return {"powershell": win, "nvidia-smi": "GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU (UUID: GPU-x)\n",
                "lspci": "00:02.0 VGA compatible controller: Intel Corporation Meteor Lake-P [Intel Arc Graphics]\n"
                         "01:00.0 3D controller: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q / Mobile]\n"}[args[0]]

    assert gpu_select.list_gpus("win32", run) == ["Intel(R) Arc(TM) Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"]
    assert gpu_select.list_gpus("linux", run)[1].startswith("NVIDIA Corporation AD107M")
    assert gpu_select.list_gpus("darwin", run) == ["NVIDIA GeForce RTX 4060 Laptop GPU"]   # nvidia-smi only
    assert gpu_select.list_gpus("darwin", lambda a, timeout=10: "") == []


@pytest.mark.parametrize("renderer, hint", [
    ("Intel(R) Arc(TM) Graphics", True),
    ("Intel(R) UHD Graphics 620", True),
    ("NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2", False),
    ("", False),
])
def test_other_gpu_hint(renderer, hint):
    gpus = ["Intel(R) Arc(TM) Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"]
    text = gpu_select.other_gpu_hint(renderer, gpus, "win32")
    assert bool(text) is hint
    if hint:
        assert "RTX 4060" in text and "High performance" in text
    assert gpu_select.other_gpu_hint(renderer, gpus[:1], "win32") is None   # nothing else to switch to


def test_setting_and_cli(capsys, monkeypatch):
    from ewald import settings
    from ewald.__main__ import build_parser

    assert settings.load()["viewport.gpu"] == "auto"
    assert tuple(next(o for o in settings.OPTIONS if o.key == "viewport.gpu").choices) == gpu_select.CHOICES
    args = build_parser().parse_args(["gpu", "--prefer", "high-performance"])
    assert args.prefer == "high-performance"
    monkeypatch.setattr(gpu_select, "list_gpus", lambda: ["Test GPU"])
    from ewald.__main__ import main

    main(["gpu"])
    assert "GPUs: Test GPU" in capsys.readouterr().out


def test_gpu_install_runs_pip_for_cupy_with_cuda_wheels(capsys, monkeypatch):
    """`ewald gpu --install` (D80): the same pip as the Plugins side bar, with cupy-cuda12x[ctk]."""
    import subprocess

    from ewald import plugin_catalog
    from ewald.__main__ import main

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, env=None: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(gpu_select, "list_gpus", lambda: ["Test GPU"])
    main(["gpu", "--install"])
    assert calls == [plugin_catalog.pip_command("install", ["cupy-cuda12x[ctk]"])]
    assert "Installed" in capsys.readouterr().out
    monkeypatch.setattr(sys, "platform", "darwin")   # no CUDA on a Mac: nothing to install
    monkeypatch.setattr(gpu_select, "summary", lambda: "")
    calls.clear()
    assert main(["gpu", "--install"]) == 1 and not calls


def test_cuda_dll_dirs_finds_the_nvidia_wheels(tmp_path, monkeypatch):
    """Windows: the bin folders of pip's nvidia-* wheels go on the DLL path before CuPy loads."""
    from ewald import backend

    for lib in ("cublas", "cuda_runtime"):
        (tmp_path / "nvidia" / lib / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cu12").mkdir()               # a folder without bin/ is skipped
    added_dirs = []
    monkeypatch.setattr(backend, "_DLL_DIRS", set())
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backend.os, "add_dll_directory", added_dirs.append, raising=False)
    monkeypatch.setenv("PATH", "")
    found = backend.cuda_dll_dirs([str(tmp_path), str(tmp_path / "missing")])
    assert [os.path.basename(os.path.dirname(d)) for d in found] == ["cublas", "cuda_runtime"]
    assert added_dirs == found and found[0] in os.environ["PATH"]
    assert backend.cuda_dll_dirs([str(tmp_path)]) == []   # once per folder
    monkeypatch.setattr(sys, "platform", "linux")
    assert backend.cuda_dll_dirs([str(tmp_path)]) == []
