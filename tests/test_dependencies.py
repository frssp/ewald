"""Declared dependencies match the imports, and Qt startup failures explain themselves (no Qt needed)."""
import ast
import pathlib
import sys

import pytest

import ewald.gui as gui

ROOT = pathlib.Path(__file__).resolve().parents[1]
# import name -> distribution name, where they differ
DIST = {"PIL": "pillow", "cupyx": "cupy-cuda12x", "cupy": "cupy-cuda12x", "imageio_ffmpeg": "imageio-ffmpeg",
        "MDAnalysis": "mdanalysis", "PySide6": "pyside6"}
# used opportunistically when installed, or arriving with a declared package
UNDECLARED_OK = {"pymatgen", "anyio"}


def _requirement_names(reqs):
    import re

    return {re.split(r"[<>=\[ ;]", r, maxsplit=1)[0].lower() for r in reqs}


def test_every_third_party_import_is_declared():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = _requirement_names(project["dependencies"])
    for reqs in project["optional-dependencies"].values():
        declared |= _requirement_names(reqs)
    stdlib = set(sys.stdlib_module_names)
    missing = {}
    for path in (ROOT / "src" / "ewald").rglob("*.py"):
        if path.parent.name == "tables" and path.name.startswith("make_"):
            continue  # table generators run by developers (pymatgen, PyYAML), not at runtime
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or top == "ewald" or top in UNDECLARED_OK:
                    continue
                if DIST.get(top, top).lower() not in declared:
                    missing.setdefault(top, path.name)
    assert not missing, f"imported but not in pyproject.toml dependencies or extras: {missing}"


def test_gui_requires_pyside6_not_pyqt():
    tomllib = pytest.importorskip("tomllib")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    names = _requirement_names(extras["gui"])
    assert "pyside6" in names and not names & {"pyqt5", "pyqt6", "pyside2"}


def test_other_qt_binding_already_imported_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyQt6", object())
    with pytest.raises(gui.QtStartupError, match="PyQt6 is already imported"):
        gui.check_qt_binding()


def test_mixed_qt_versions_name_ld_library_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib:/lib/x86_64-linux-gnu:/usr/lib")
    msg = gui.qt_load_hint("/lib/x86_64-linux-gnu/libQt6DBus.so.6: undefined symbol: "
                           "_ZN14QObjectPrivateC2Ei, version Qt_6_PRIVATE_API")
    assert "LD_LIBRARY_PATH" in msg and "/lib/x86_64-linux-gnu, /usr/lib" in msg and "/usr/local/lib" not in msg
    assert "libegl1" in gui.qt_load_hint("libEGL.so.1: cannot open shared object file: No such file or directory")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="xcb is Linux only")
def test_missing_xcb_libraries_are_named(monkeypatch):
    import ctypes

    real = ctypes.CDLL

    def fake(name, *a, **k):
        if name == "libxcb-cursor.so.0":
            raise OSError(name)
        return real("libc.so.6")

    monkeypatch.setattr(ctypes, "CDLL", fake)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":99")
    with pytest.raises(gui.QtStartupError, match="sudo apt install libxcb-cursor0$|libxcb-cursor0 "):
        gui.check_qt_platform()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    gui.check_qt_platform()  # explicit non-X11 platform: nothing to check
    monkeypatch.delenv("QT_QPA_PLATFORM")
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(gui.QtStartupError, match="no display"):
        gui.check_qt_platform()


def test_cli_names_the_extra_for_a_missing_package(monkeypatch, capsys):
    from ewald import __main__ as cli

    def boom(args):
        raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")

    monkeypatch.setattr(cli, "_run", boom)
    with pytest.raises(SystemExit, match=r"pip install 'ewald\[llm\]'"):
        cli.main(["mcp"])


def test_ssh_forwarded_display_detection(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.2 5000 10.0.0.1 22")
    for display, forwarded in (("localhost:11.0", True), ("localhost:10", True), (":0", False),
                               (":1.0", False), ("localhost:2.0", False)):
        monkeypatch.setenv("DISPLAY", display)
        assert gui.ssh_forwarded_display() is forwarded, display
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.setenv("DISPLAY", "localhost:11.0")
    assert not gui.ssh_forwarded_display()


def test_forwarded_display_falls_back_to_software_opengl(monkeypatch, capsys):
    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")
    monkeypatch.setenv("DISPLAY", "localhost:11.0")
    for key in ("LIBGL_ALWAYS_SOFTWARE", "QT_XCB_GL_INTEGRATION", "EWALD_SKIP_GL_PROBE"):
        monkeypatch.delenv(key, raising=False)
    tried = []

    def probe(extra):
        tried.append(extra)
        return bool(extra), "XIO: fatal IO error 17"

    gui.check_remote_display(probe)
    assert tried == [{}, {"LIBGL_ALWAYS_SOFTWARE": "1"}]
    import os

    assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert "software rendering" in capsys.readouterr().err

    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE")
    with pytest.raises(gui.QtStartupError, match="(?s)forwarded over SSH.*XIO: fatal IO error 17.*vglrun"):
        gui.check_remote_display(lambda extra: (False, "XIO: fatal IO error 17"))
    monkeypatch.setenv("EWALD_SKIP_GL_PROBE", "1")
    gui.check_remote_display(lambda extra: pytest.fail("probe should be skipped"))
