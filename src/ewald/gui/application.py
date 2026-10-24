"""The QApplication, made without the main window's heavy imports (VTK, pyvista, matplotlib).

``ewald-gui`` creates it first so that the first-run notice (``gui.first_run``) is on screen
while ``gui.app`` is still importing; ``gui.app`` re-exports these names.
"""
from __future__ import annotations

import sys

from . import check_qt_binding

check_qt_binding()  # before PySide6: a clear error when another Qt binding is loaded

from PySide6 import QtCore, QtGui, QtWidgets

from .icons import app_icon
from .macos import set_app_name, strip_psn


def vtk_surface_format() -> QtGui.QSurfaceFormat:
    """The OpenGL format of the VTK view: desktop OpenGL 3.2 core, as pyvistaqt's ``QtInteractor``
    requests (``pyvistaqt.rwi._default_format``; MSAA happens in VTK's own framebuffers)."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 2)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
    for size in ("Red", "Green", "Blue", "Alpha"):
        getattr(fmt, f"set{size}BufferSize")(8)
    fmt.setDepthBufferSize(8)
    fmt.setStencilBufferSize(0)
    fmt.setSamples(0)
    return fmt


def use_vtk_surface_format() -> None:
    """Make the VTK view's format the process default, before any window has an OpenGL context.

    Every QOpenGLWidget of a window shares OpenGL objects with the window's compositing context,
    which Qt creates from the default format when the first one (the placeholder) appears.
    pyvistaqt installs its 3.2 core default only when the VTK widget is built, after the window
    is on screen. Linux and Windows drivers share between the two profiles anyway; macOS cannot
    share a legacy 2.1 context with a 4.1 core one, so the VTK view composites black there.

    Only on macOS: on Windows this default (alpha and an 8-bit depth buffer for the window's own
    compositing surface) left the whole window black after a resize or maximize, until restart.
    """
    if sys.platform != "darwin":
        return
    want = vtk_surface_format()
    if QtGui.QSurfaceFormat.defaultFormat() != want:
        QtGui.QSurfaceFormat.setDefaultFormat(want)


def _set_windows_app_id():
    """Own taskbar entry and icon on Windows instead of python.exe's."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ewald.gui")
        except (AttributeError, OSError):
            pass


class EwaldApplication(QtWidgets.QApplication):
    """QApplication that hands macOS file-open events to the newest main window.

    Finder sends a QFileOpenEvent, not argv, when a file is opened with the .app (double-click,
    Open With, a drop on the Dock icon). It can arrive before the main window exists, so paths
    wait until a window registers itself with :meth:`set_file_open_target`.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._pending_files: list[str] = []
        self._file_open_target = None

    def set_file_open_target(self, open_path):
        self._file_open_target = open_path
        while self._pending_files:
            try:
                open_path(self._pending_files[0])
            except RuntimeError:  # that window was closed and deleted: wait for the next one
                self._file_open_target = None
                return
            self._pending_files.pop(0)

    def event(self, event):
        if event.type() == QtCore.QEvent.FileOpen:
            path = event.file() or event.url().toLocalFile()
            if path:
                self._pending_files.append(path)
                if self._file_open_target is not None:
                    self.set_file_open_target(self._file_open_target)
            return True
        return super().event(event)


def _app():
    from . import theme

    app = QtWidgets.QApplication.instance()
    if app is None:
        from . import check_qt_platform, check_remote_display, use_software_gl_if_needed, viewport_backend
        from .. import settings as settings_mod

        check_qt_platform()
        backend = viewport_backend(settings_mod.load()["viewport.backend"])
        use_software_gl_if_needed(backend)
        if backend == "vtk":
            check_remote_display()
            use_vtk_surface_format()  # Qt wants it before the QApplication on macOS
        _set_windows_app_id()
        set_app_name()
        app = EwaldApplication(strip_psn(sys.argv))
        app.setApplicationName("ewald")
        app.setDesktopFileName("ewald")
        app.setWindowIcon(app_icon())
        from .fonts import load_fonts

        load_fonts()   # before the theme, whose QSS names IBM Plex Sans and JetBrains Mono
        try:
            name = settings_mod.load()["general.ui_theme"]
        except Exception:  # a broken settings file must not stop the GUI
            name = None
        theme.apply(app, name)
    return app
