"""The "first start after installing or updating" notice of ``ewald-gui``.

The first start after an install or update is much slower than later ones: Python compiles the
``.pyc`` files, numba compiles the analysis kernels into its cache, and macOS checks every new
native library (Qt, VTK, numba) before it loads it. This module tells whether this start is
such a one, by comparing a stamp of the installed packages with the one saved by the last
start that got its window up (``startup.json`` in the config dir), and shows a small notice
while the main window is being imported and built. The stamp is plain data and Qt-free; the
notice is a frameless window styled through ``#firstRunNotice`` in ``gui/theme.py``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

# packages whose reinstall or update makes the next start slow (new native libraries,
# new bytecode, numba kernels to compile again)
PACKAGES = ("PySide6", "vtkmodules", "pyvista", "numba", "matplotlib", "numpy")
# ewald's own sources: an editable install changes these, not its version
SOURCE_DIRS = ("", "gui", "modifiers", "io")
FILE = "startup.json"

TITLE = "Preparing EWALD"
MESSAGE = ("This is the first start since EWALD or one of its libraries was installed or updated. "
           "Python, numba and the system are preparing them, so it takes longer than usual. "
           "Later starts are faster.")

_notice = None


def _mtime(path) -> float:
    try:
        return round(os.stat(path).st_mtime, 3)
    except OSError:
        return 0.0


def _executable(path: str, osp=os.path) -> str:
    """The interpreter, compared case-insensitively on Windows (``osp`` = ``ntpath`` in tests)."""
    return osp.normcase(osp.abspath(path))


def stamp() -> dict:
    """What the installed packages look like now, without importing them (``find_spec`` of a
    top-level name only locates it)."""
    import ewald

    root = Path(ewald.__file__).parent
    sources = max((_mtime(p) for d in SOURCE_DIRS for p in (root / d).glob("*.py")), default=0.0)
    packages = {}
    for name in PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        packages[name] = _mtime(spec.origin) if spec is not None and spec.origin else None
    return {"ewald": ewald.__version__, "python": sys.version.split()[0],
            "executable": _executable(sys.executable),
            "sources": sources, "packages": packages}


def _path() -> Path:
    from .. import settings

    return settings.config_dir() / FILE


def is_first_run(current: dict | None = None) -> bool:
    """True unless a start with these same packages already got its window up."""
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return saved != (stamp() if current is None else current)


def mark_done(current: dict | None = None) -> None:
    """Save the stamp: the next start with these packages is not a first run."""
    from .. import fileops

    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fileops.write_text(path, json.dumps(stamp() if current is None else current, indent=1) + "\n")
    except OSError:
        pass  # a read-only config dir only means the notice shows again next time


def _make_notice(app):
    from PySide6 import QtCore, QtWidgets

    w = QtWidgets.QFrame(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint)
    w.setObjectName("firstRunNotice")
    w.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(20, 18, 24, 18)
    lay.setSpacing(16)
    icon = QtWidgets.QLabel()
    icon.setPixmap(app.windowIcon().pixmap(56, 56))
    icon.setAlignment(QtCore.Qt.AlignTop)
    lay.addWidget(icon)
    text = QtWidgets.QVBoxLayout()
    text.setSpacing(6)
    title = QtWidgets.QLabel(TITLE)
    title.setObjectName("firstRunTitle")
    body = QtWidgets.QLabel(MESSAGE)
    body.setObjectName("firstRunBody")
    body.setWordWrap(True)
    text.addWidget(title)
    text.addWidget(body)
    text.addStretch(1)
    lay.addLayout(text, 1)
    w.setFixedWidth(460)
    w.adjustSize()
    screen = app.primaryScreen()
    if screen is not None:
        w.move(screen.availableGeometry().center() - w.rect().center())
    return w


def show_notice(app):
    """Show the notice and paint it now: the imports that follow block the event loop."""
    global _notice
    _notice = _make_notice(app)
    _notice.show()
    _notice.raise_()
    for _ in range(3):  # macOS needs a few rounds to put a new window on screen
        app.processEvents()
    return _notice


def start(argv=None):
    """Before ``gui.app`` is imported: on a first run, make the QApplication and show the notice."""
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in args):
        return None
    try:
        current = stamp()
        if not is_first_run(current):
            return None
        from .application import _app

        notice = show_notice(_app())
        notice.setProperty("stamp", json.dumps(current))
        return notice
    except Exception:  # never let the notice stop the GUI from starting
        return None


def finish(window=None) -> None:
    """The main window is up: close the notice and save the stamp."""
    global _notice
    notice, _notice = _notice, None
    if notice is None:
        return
    try:
        current = json.loads(notice.property("stamp") or "null")
        notice.close()
    except RuntimeError:  # already deleted
        current = None
    mark_done(current)
