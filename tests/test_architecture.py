"""Layering contracts: the headless core never depends on the desktop app.

``ewald.gui`` sits on top of the core (pipeline, readers, renderers, notebook, plugins, llm); the
core must import, compute and render without Qt. These tests fail when a core module starts
importing ``ewald.gui`` or when headless compute and rendering start reading user settings.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ewald"

# Core modules allowed to reach into the GUI package, and why.
GUI_IMPORT_ALLOWED = {
    "__main__.py",   # `ewald gui` starts the desktop app
    "inventory.py",  # `ewald inventory --gui` lists the GUI menus for docs/features.md
    "workflow.py",   # workflow_panel(): a plugin's side bar factory builds the Qt panel when it opens
    "optimade/panel.py",   # the OPTIMADE side bar's factory builds the Qt panel when it opens
}

# Headless compute and rendering take every option as an argument (CLAUDE.md, Conventions).
NO_SETTINGS = ["pipeline.py", "render.py", "raytrace.py", "vector.py", "scene.py", "kernels.py",
               "neighbors.py", "backend.py", "data.py"]


def _imports(path: Path):
    """(module, level) of every import in ``path``, nested ones included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.level
            if node.level and not node.module:   # from . import gui
                for alias in node.names:
                    yield alias.name, node.level


def _absolute(path: Path, module: str, level: int) -> str:
    if not level:
        return module
    package = list(path.relative_to(SRC.parent).with_suffix("").parts[:-1])
    base = package[:len(package) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _core_files():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[0] in ("gui", "examples"):
            continue
        yield rel, path


def test_core_does_not_import_the_gui():
    bad = []
    for rel, path in _core_files():
        if rel.as_posix() in GUI_IMPORT_ALLOWED:
            continue
        for module, level in _imports(path):
            name = _absolute(path, module, level)
            if name == "ewald.gui" or name.startswith("ewald.gui.") or name.split(".")[0] in ("PySide6", "pyvistaqt"):
                bad.append(f"{rel}: {name}")
    assert not bad, "core modules importing the GUI (move the Qt-free part into the core):\n" + "\n".join(bad)


@pytest.mark.parametrize("name", NO_SETTINGS)
def test_headless_compute_and_render_do_not_read_settings(name):
    path = SRC / name
    names = {_absolute(path, m, lvl) for m, lvl in _imports(path)}
    assert "ewald.settings" not in names, f"{name} imports ewald.settings; pass the option as an argument"


def test_core_imports_without_qt():
    """Importing the core and using the pieces the GUI shares with it loads no Qt."""
    code = """
import sys
import ewald, ewald.pipeline, ewald.session, ewald.notebook, ewald.view_style, ewald.ui_tokens
from ewald import plugins
plugins.ui_themes()
plugins.ui_themes()["graphite"].palette()
ewald.session.Session(style={"radius_scale": 0.5}).render_kwargs()
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("PySide6", "pyvistaqt") or m.startswith("ewald.gui"))
print(",".join(loaded))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "EWALD_NO_PLUGINS": "1"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"the core loaded GUI modules: {out.stdout.strip()}"
