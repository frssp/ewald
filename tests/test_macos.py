"""macOS support that can be checked on any OS: GPU fallback, packaging metadata, GUI file-open events."""
import os
import plistlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("device", ["metal", "mps", "MPS"])
def test_apple_gpu_device_runs_the_analysis_on_the_cpu(device):
    """Only the ray tracer runs on Metal (tests/test_gpu_raytrace.py); the analysis stays on the CPU."""
    from ewald import backend

    assert backend.get_backend(device) is backend.CPU


def test_auto_without_cuda_is_cpu(monkeypatch):
    from ewald import backend

    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    assert backend.get_backend("auto") is backend.CPU


def test_gpu_extra_skips_cupy_on_macos():
    text = (ROOT / "pyproject.toml").read_text()
    gpu = re.search(r"^gpu = \[(.*)\]", text, re.M).group(1)
    assert "cupy" in gpu and "sys_platform != 'darwin'" in gpu


def test_strip_psn():
    from ewald.gui.macos import strip_psn

    assert strip_psn(["-psn_0_123456", "POSCAR", "--device", "cpu"]) == ["POSCAR", "--device", "cpu"]


@pytest.mark.skipif(sys.platform == "darwin", reason="checks the no-op off macOS")
def test_set_app_name_noop_off_macos():
    from ewald.gui.macos import set_app_name

    assert set_app_name() is False


def test_macos_spec_info_plist():
    """The bundle's Info.plist settings serialize and name the right executable and icon."""
    spec = (ROOT / "packaging" / "macos" / "ewald.spec").read_text()
    ns = {}
    start, end = spec.index("EXTENSIONS = "), spec.index("if sys.platform == \"darwin\":")
    exec(compile(spec[start:end], "ewald.spec", "exec"), {"__version__": "0.0.0"}, ns)
    info = ns["INFO"]
    assert info["CFBundleExecutable"] == "ewald-gui" and info["NSHighResolutionCapable"] is True
    assert "lammpstrj" in info["CFBundleDocumentTypes"][0]["CFBundleTypeExtensions"]
    plistlib.dumps(info)  # every value is plist-serializable
    assert 'name="ewald-gui"' in spec and 'name="ewald"' in spec
    assert (ROOT / "src" / "ewald" / "assets" / "icon" / "ewald.icns").stat().st_size > 1000
    for f in ("launch_gui.py", "launch_cli.py", "smoke.py"):  # entry points shared with Windows
        assert (ROOT / "packaging" / "windows" / f).exists()
    assert os.access(ROOT / "packaging" / "macos" / "make_dmg.sh", os.X_OK)


def test_macos_workflow_is_manual_only():
    yaml = pytest.importorskip("yaml")

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "macos-release.yml").read_text())
    triggers = wf.get("on", wf.get(True))  # YAML 1.1 reads the key `on` as True
    assert list(triggers) == ["workflow_dispatch"]


def test_rthook_sets_numba_cache_and_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    path = sys.path[:]
    try:
        exec(compile((ROOT / "packaging" / "macos" / "rthook_macos.py").read_text(), "rthook", "exec"), {})
    finally:
        sys.path[:] = path
    assert os.environ["NUMBA_CACHE_DIR"].startswith(str(tmp_path / "Library" / "Caches"))
    assert os.environ["PATH"].startswith("/usr/bin:/bin")


def test_file_open_event_reaches_main_window():
    """Finder's QFileOpenEvents wait for the first window, then go straight to the newest one."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui

    from ewald.gui.app import EwaldApplication

    class App:  # the Python side of EwaldApplication, without a second QApplication
        event = EwaldApplication.event
        set_file_open_target = EwaldApplication.set_file_open_target

        def __init__(self):
            self._pending_files, self._file_open_target = [], None

    app, opened = App(), []
    assert app.event(QtGui.QFileOpenEvent("/data/POSCAR")) is True
    assert opened == [] and app._pending_files == ["/data/POSCAR"]  # no window yet: queued
    app.set_file_open_target(opened.append)
    assert opened == ["/data/POSCAR"]
    app.event(QtGui.QFileOpenEvent(QtCore.QUrl.fromLocalFile("/data/run.lammpstrj")))
    assert opened == ["/data/POSCAR", "/data/run.lammpstrj"]

    def closed(path):
        raise RuntimeError("Internal C++ object (MainWindow) already deleted.")

    app.set_file_open_target(closed)
    app.event(QtGui.QFileOpenEvent("/data/XDATCAR"))
    assert app._file_open_target is None and app._pending_files == ["/data/XDATCAR"]  # kept for the next window


def test_vtk_surface_format_matches_pyvistaqt():
    """The window's OpenGL default must be the format pyvistaqt's QtInteractor asks for: macOS cannot
    share a legacy 2.1 context with a 4.1 core one, and the VTK view composites black (the 3D view
    stayed black on a Mac while everything else worked)."""
    pytest.importorskip("PySide6")
    rwi = pytest.importorskip("pyvistaqt.rwi")
    if not hasattr(rwi, "_default_format"):
        pytest.skip("pyvistaqt without the QOpenGLWidget interactor")
    from PySide6 import QtGui

    from ewald.gui.viewport import vtk_surface_format

    ours, theirs = vtk_surface_format(), rwi._default_format()
    assert ours == theirs
    assert ours.profile() == QtGui.QSurfaceFormat.CoreProfile and ours.version() == (3, 2)
