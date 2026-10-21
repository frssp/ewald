"""Desktop GUI (PySide6 + VTK). Install with ``pip install 'ewald[gui]'``.

The GUI is written against PySide6 only. pyvistaqt and qtconsole reach Qt through QtPy, and
matplotlib and IPython's ``%gui qt`` pick a binding of their own, so importing this package sets
``QT_API=pyside6`` (unless already set) to make them all agree. The main window and viewport call
:func:`check_qt_binding` first: when another binding (PyQt5, PyQt6, PySide2) is already loaded,
QtPy hands that one to pyvistaqt, and PyQt6 next to PySide6 fails with undefined Qt symbols.
``gui.style`` stays importable without Qt (sessions use it headless).
"""
import os
import sys

os.environ.setdefault("QT_API", "pyside6")

OTHER_BINDINGS = ("PyQt5", "PyQt6", "PySide2")
INSTALL_HINT = "the desktop GUI needs PySide6: pip install 'ewald[gui]'"


class QtStartupError(ImportError):
    """Qt can't be loaded or started; the message says why and what to install or change."""


def check_qt_binding():
    """Raise ImportError when PySide6 is missing or another Qt binding is already imported."""
    loaded = [b for b in OTHER_BINDINGS if b in sys.modules]
    if loaded:
        raise QtStartupError(
            f"ewald's GUI uses PySide6, but {', '.join(loaded)} is already imported in this process "
            "(by matplotlib's Qt backend, IPython's %gui qt or another package). Import ewald.gui "
            "first, or set QT_API=pyside6 before anything loads Qt; in Jupyter run %gui qt only "
            "after import ewald.gui.")
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401  (loads the Qt libraries)
    except ModuleNotFoundError as exc:
        raise QtStartupError(INSTALL_HINT) from exc
    except ImportError as exc:
        raise QtStartupError(qt_load_hint(str(exc))) from exc


SYSTEM_LIB_DIRS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/lib/x86_64-linux-gnu",
                   "/usr/lib/x86_64-linux-gnu", "/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu")
# shared library -> Debian/Ubuntu package (for "cannot open shared object file")
LINUX_PACKAGES = {"libEGL.so.1": "libegl1", "libGL.so.1": "libgl1", "libOpenGL.so.0": "libopengl0",
                  "libxkbcommon.so.0": "libxkbcommon0", "libfontconfig.so.1": "libfontconfig1",
                  "libdbus-1.so.3": "libdbus-1-3", "libglib-2.0.so.0": "libglib2.0-0"}


def qt_load_hint(message: str) -> str:
    """Explain a failed ``import PySide6.QtCore`` (wrong Qt picked up, or a missing system library)."""
    lines = [f"PySide6 is installed but its Qt libraries failed to load: {message}"]
    shadowing = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                 if d.rstrip("/") in SYSTEM_LIB_DIRS]
    if "undefined symbol" in message or "version `Qt_" in message or "Qt_6" in message:
        lines.append("Qt libraries from two different Qt versions were mixed: usually a system Qt "
                     "(apt/dnf) is found before the one bundled with the PySide6 wheel.")
        if shadowing:
            lines.append(f"LD_LIBRARY_PATH lists system directories ({', '.join(shadowing)}), which take "
                         "precedence over the wheel's own Qt; remove them from LD_LIBRARY_PATH (they are "
                         "searched by default anyway) and start again.")
        else:
            lines.append("Check LD_LIBRARY_PATH and LD_PRELOAD for directories with another Qt 6.")
    elif "cannot open shared object file" in message:
        lib = message.split(":")[0].strip()
        pkg = LINUX_PACKAGES.get(lib)
        lines.append(f"A system library is missing{f'; on Debian/Ubuntu: sudo apt install {pkg}' if pkg else ''}. "
                     "See the Linux section of docs/user/install.md.")
    return "\n".join(lines)


# Qt's xcb platform plugin (X11) aborts the process when one of these is missing, so they are
# checked before QApplication is created. soname -> Debian/Ubuntu package
XCB_LIBS = {"libxcb-cursor.so.0": "libxcb-cursor0", "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
            "libxcb-icccm.so.4": "libxcb-icccm4", "libxcb-keysyms.so.1": "libxcb-keysyms1",
            "libxcb-shape.so.0": "libxcb-shape0", "libxcb-render-util.so.0": "libxcb-render-util0",
            "libxcb-image.so.0": "libxcb-image0", "libxcb-randr.so.0": "libxcb-randr0",
            "libxcb-xkb.so.1": "libxcb-xkb1"}


def check_qt_platform():
    """On Linux, raise QtStartupError (instead of Qt aborting) when the X11 platform plugin can't start."""
    if not sys.platform.startswith("linux"):
        return
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform and not platform.startswith("xcb"):
        return  # offscreen, wayland, ... chosen explicitly
    if not platform and os.environ.get("WAYLAND_DISPLAY"):
        return  # Qt picks wayland
    if not os.environ.get("DISPLAY"):
        raise QtStartupError("no display: DISPLAY is not set. Run on a desktop session, through ssh -X, "
                           "or headless with xvfb-run -a ewald-gui (QT_QPA_PLATFORM=offscreen for tests).")
    import ctypes

    missing = []
    for soname, pkg in XCB_LIBS.items():
        try:
            ctypes.CDLL(soname)
        except OSError:
            missing.append(pkg)
    if missing:
        raise QtStartupError(
            "Qt's X11 (xcb) platform plugin needs system libraries that are not installed. On "
            f"Debian/Ubuntu: sudo apt install {' '.join(missing)} (Fedora: xcb-util-cursor, "
            "xcb-util-wm, xcb-util-keysyms, xcb-util-image, xcb-util-renderutil, libxkbcommon-x11).")


# Run in a child process: an OpenGL failure over a forwarded X display can corrupt the X
# connection and abort the process (malloc_consolidate, XIO fatal error) instead of raising.
GL_PROBE = """
import sys
from PySide6 import QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
app = QtWidgets.QApplication(sys.argv[:1])
w = QtInteractor(None, off_screen=False)
w.add_mesh(pv.Sphere())
w.show(); app.processEvents(); w.render(); app.processEvents()
print(w.ren_win.ReportCapabilities().splitlines()[1:3])
w.close()
"""
# tried in order when the display is forwarded over SSH; the first that renders is kept
GL_FALLBACKS = ({}, {"LIBGL_ALWAYS_SOFTWARE": "1"},
                {"LIBGL_ALWAYS_SOFTWARE": "1", "QT_XCB_GL_INTEGRATION": "none"})


def ssh_forwarded_display() -> bool:
    """True when DISPLAY is an SSH X11 forwarding (``ssh -X``/``-Y``: ``localhost:10.0`` and up)."""
    display = os.environ.get("DISPLAY", "")
    if not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")) or ":" not in display:
        return False
    host, _, number = display.rpartition(":")
    try:
        return bool(host) and int(number.split(".")[0]) >= 10
    except ValueError:
        return False


VIEWPORT_BACKENDS = ("auto", "vtk", "legacy-gl")


def viewport_backend(choice: str | None = None) -> str:
    """The 3D view to use: ``"vtk"`` or ``"legacy-gl"`` (OpenGL 1.x display lists, VESTA-style).

    ``EWALD_VIEWPORT`` wins over ``choice`` (the ``viewport.backend`` preference); ``"auto"`` picks
    legacy-gl on an SSH-forwarded display, where GLX is indirect and VTK can't draw.
    """
    choice = (os.environ.get("EWALD_VIEWPORT") or choice or "auto").strip().lower()
    if choice in ("legacy", "gl1", "opengl1"):
        choice = "legacy-gl"
    if choice not in VIEWPORT_BACKENDS:
        raise QtStartupError(f"unknown viewport {choice!r}; use one of {', '.join(VIEWPORT_BACKENDS)}")
    if choice == "auto":
        return "legacy-gl" if ssh_forwarded_display() else "vtk"
    return choice


def probe_opengl(extra_env: dict, timeout: float = 60) -> tuple[bool, str]:
    """Open a small VTK window in a child process with ``extra_env``; (rendered, output)."""
    import subprocess

    try:
        out = subprocess.run([sys.executable, "-c", GL_PROBE], env={**os.environ, **extra_env},
                             capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "timed out"
    return out.returncode == 0, (out.stdout + out.stderr).strip()


def check_remote_display(probe=probe_opengl):
    """Over SSH X11 forwarding, make sure VTK can draw, falling back to Mesa software OpenGL.

    Forwarded GLX is indirect rendering (OpenGL 1.x), which VTK can't use. If the first probe
    fails, the fallbacks in ``GL_FALLBACKS`` are tried and the one that works is set in
    ``os.environ`` for this process; if none works, :class:`QtStartupError` lists the options.
    ``EWALD_SKIP_GL_PROBE=1`` skips the check.
    """
    if os.environ.get("EWALD_SKIP_GL_PROBE") or not ssh_forwarded_display():
        return
    last = ""
    for extra in GL_FALLBACKS:
        if any(os.environ.get(k) not in (None, v) for k, v in extra.items()):
            continue  # the user chose a different value
        ok, last = probe(extra)
        if ok:
            if extra:
                os.environ.update(extra)
                print("ewald-gui: OpenGL over SSH X11 forwarding failed; using "
                      + " ".join(f"{k}={v}" for k, v in extra.items()) + " (software rendering, slower)",
                      file=sys.stderr)
            return
    raise QtStartupError(
        f"the display {os.environ.get('DISPLAY')} is forwarded over SSH, and VTK could not draw on it "
        "(forwarded GLX only offers OpenGL 1.x), not even with Mesa software rendering "
        f"(LIBGL_ALWAYS_SOFTWARE=1). Last error: {last[-400:] or 'none'}\n"
        "Options: run ewald on your own computer and open the cluster's files over SSH "
        "(ewald gui host:/path/to/file, or File > Load remote file (SSH)); use a remote desktop with GPU "
        "OpenGL (TurboVNC + VirtualGL: vglrun ewald-gui, X2Go, NoMachine, Open OnDemand); install "
        "Mesa's software OpenGL on the cluster (RHEL: dnf install mesa-dri-drivers mesa-libGL) and "
        "retry; or render without a window: ewald render FILE out.png. "
        "See 'Troubleshooting the GUI' in docs/user/install.md.")

def show(pipeline=None, device=None, block: bool = True):
    from .app import show as _show

    return _show(pipeline, device, block)


def main(argv=None):
    """``ewald-gui``: a failure to load or start Qt is one explanatory message, not a traceback."""
    try:
        from .app import main as _main

        return _main(argv)
    except QtStartupError as exc:
        sys.exit(f"ewald-gui: {exc}")
