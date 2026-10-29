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


def skip_pyside_feature_scan():
    """Stop PySide6 from reading the source of every module imported after it.

    Once PySide6 is loaded, its import hook (``shibokensupport.feature``) looks into each newly
    imported module to see whether it uses PySide's ``__feature__`` options (snake_case,
    true_property). That made importing numba, scipy, pyvista and matplotlib about 40 % slower,
    1.3 s of the owner's Windows start-up. EWALD and the libraries it loads never use
    ``__feature__``, so the check is answered "no" for every module. Private PySide API: when it
    is not there, nothing changes."""
    feature = sys.modules.get("shibokensupport.feature")
    if feature is not None and hasattr(feature, "_mod_uses_pyside"):
        feature._mod_uses_pyside = lambda module: False


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


X11_SOCKET_DIR = "/tmp/.X11-unix"   # X servers of this machine listen on X0, X1, ... here


def _sockets(directory: str, prefix: str) -> list[str]:
    """Names in ``directory`` that start with ``prefix``, sorted ([] when it can't be read)."""
    try:
        return sorted(n for n in os.listdir(directory) if n.startswith(prefix))
    except OSError:
        return []


def no_display_hint(env=None, x11_dir: str = X11_SOCKET_DIR) -> str:
    """Why there is no ``DISPLAY`` here and the command that gives one: an SSH login without X11
    forwarding, tmux/screen or sudo dropping the desktop's variables, or a Wayland session."""
    env = os.environ if env is None else env
    lines = ["no display: DISPLAY is not set, so there is no X server to open the window on."]
    local = [":" + n[1:] for n in _sockets(x11_dir, "X") if n[1:].isdigit()]
    runtime = env.get("XDG_RUNTIME_DIR", "")
    wayland = [n for n in _sockets(runtime, "wayland-") if not n.endswith(".lock")] if runtime else []
    if env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"):
        lines.append(
            "You are logged in over SSH without X11 forwarding. Reconnect with ssh -X user@host "
            "(or ssh -Y); on a Mac start XQuartz first, on Windows use MobaXterm or an X server such "
            "as VcXsrv. After logging in, echo $DISPLAY should print localhost:10.0 or similar; if it "
            "is still empty, the server needs 'X11Forwarding yes' in /etc/ssh/sshd_config and the "
            "xauth package. Faster: run ewald on your own computer and open the files over SSH "
            "(ewald gui user@host:/path/to/file).")
    if env.get("SUDO_USER"):
        lines.append("sudo drops DISPLAY: start ewald gui as your own user, without sudo.")
    if env.get("TMUX") or env.get("STY"):
        lines.append("This shell runs inside tmux/screen, which keeps the variables of the session "
                     "that started it, so it does not see the desktop's DISPLAY.")
    if local:
        lines.append(f"An X server is running on this machine ({', '.join(local)}): "
                     f"export DISPLAY={local[0]} and run ewald gui again (from the desktop's own "
                     "terminal it is set for you).")
    elif wayland:
        lines.append(f"A Wayland session is running ({wayland[0]}): export WAYLAND_DISPLAY={wayland[0]} "
                     "(or DISPLAY=:0 for its XWayland) and run ewald gui again.")
    lines.append("Without any display: xvfb-run -a ewald gui (screenshots, tests) or ewald render FILE "
                 "out.png. See 'Troubleshooting the GUI' in docs/user/install.md.")
    return "\n".join(lines)


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
        raise QtStartupError(no_display_hint())
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
    legacy-gl on an SSH-forwarded display, where GLX is indirect and VTK can't draw, and on
    Windows without OpenGL 3.2 (:func:`windows_needs_software_gl`).
    """
    choice = (os.environ.get("EWALD_VIEWPORT") or choice or "auto").strip().lower()
    if choice in ("legacy", "gl1", "opengl1"):
        choice = "legacy-gl"
    if choice not in VIEWPORT_BACKENDS:
        raise QtStartupError(f"unknown viewport {choice!r}; use one of {', '.join(VIEWPORT_BACKENDS)}")
    if choice == "auto":
        return "legacy-gl" if ssh_forwarded_display() or windows_needs_software_gl() else "vtk"
    return choice


# ----------------------------------------------------------------------- OpenGL on Windows
#: The OpenGL version VTK's viewport needs.
VTK_MIN_GL = (3, 2)


def _windows() -> bool:
    return sys.platform == "win32"


def parse_gl_version(text: str) -> tuple[int, int] | None:
    """``"4.6.0 NVIDIA 551.23"`` -> (4, 6); ``"1.1.0"`` -> (1, 1); None when unreadable."""
    import re

    m = re.match(r"\s*(?:OpenGL ES\s+)?(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


_windows_gl_cache: list = []


def windows_opengl() -> tuple[tuple[int, int] | None, str]:
    """(version, renderer) of the OpenGL the Windows display driver gives a plain window (WGL),
    asked once per process; (None, reason) when it gives none. ``EWALD_SKIP_GL_PROBE=1`` skips
    the check (reported as OpenGL 4.6)."""
    if not _windows_gl_cache:
        if os.environ.get("EWALD_SKIP_GL_PROBE"):
            _windows_gl_cache.append(((4, 6), "not checked (EWALD_SKIP_GL_PROBE)"))
        else:
            try:
                _windows_gl_cache.append(_wgl_probe())
            except Exception as exc:   # no opengl32.dll, a locked-down session, ...
                _windows_gl_cache.append((None, f"{type(exc).__name__}: {exc}"))
    return _windows_gl_cache[0]


def windows_needs_software_gl() -> bool:
    """True on Windows when the driver offers less than OpenGL 3.2: a virtual machine, Remote
    Desktop without GPU acceleration, or Windows' own OpenGL 1.1 ("GDI Generic"). Then the
    viewport is the OpenGL 1.x one, drawn by Qt's software OpenGL (``opengl32sw.dll``, Mesa
    llvmpipe, shipped with PySide6) when the PySide6 wheel has it."""
    if not _windows():
        return False
    version, _ = windows_opengl()
    return version is None or version < VTK_MIN_GL


def use_software_gl_if_needed(backend: str) -> str | None:
    """Before the QApplication exists: on Windows without OpenGL 3.2, make Qt draw the legacy
    viewport with its software OpenGL and return a note for the user (else None)."""
    if backend != "legacy-gl" or not windows_needs_software_gl():
        return None
    from PySide6 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    version, renderer = windows_opengl()
    found = f"OpenGL {version[0]}.{version[1]} ({renderer})" if version else f"no OpenGL ({renderer})"
    note = (f"ewald-gui: the display driver offers {found}, and the 3D view needs OpenGL "
            f"{VTK_MIN_GL[0]}.{VTK_MIN_GL[1]}; using the OpenGL 1.x viewport with software rendering. "
            "A GPU driver from the vendor (or Remote Desktop with GPU acceleration) gives the full view.")
    print(note, file=sys.stderr)
    return note


def _wgl_probe() -> tuple[tuple[int, int] | None, str]:  # pragma: no cover - Windows only
    """Create a hidden window with a legacy WGL context and read GL_VERSION / GL_RENDERER."""
    import ctypes
    from ctypes import wintypes

    user32, gdi32, gl = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32"), ctypes.WinDLL("opengl32")
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                       wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.GetDC.restype, user32.GetDC.argtypes = wintypes.HDC, [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.DestroyWindow.argtypes = [wintypes.HWND]

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [("nSize", wintypes.WORD), ("nVersion", wintypes.WORD), ("dwFlags", wintypes.DWORD)]
        _fields_ += [(n, ctypes.c_ubyte) for n in (
            "iPixelType", "cColorBits", "cRedBits", "cRedShift", "cGreenBits", "cGreenShift", "cBlueBits",
            "cBlueShift", "cAlphaBits", "cAlphaShift", "cAccumBits", "cAccumRedBits", "cAccumGreenBits",
            "cAccumBlueBits", "cAccumAlphaBits", "cDepthBits", "cStencilBits", "cAuxBuffers", "iLayerType",
            "bReserved")]
        _fields_ += [("dwLayerMask", wintypes.DWORD), ("dwVisibleMask", wintypes.DWORD),
                     ("dwDamageMask", wintypes.DWORD)]

    pfd_p = ctypes.POINTER(PIXELFORMATDESCRIPTOR)
    gdi32.ChoosePixelFormat.argtypes = [wintypes.HDC, pfd_p]
    gdi32.SetPixelFormat.argtypes = [wintypes.HDC, ctypes.c_int, pfd_p]
    gl.wglCreateContext.restype, gl.wglCreateContext.argtypes = ctypes.c_void_p, [wintypes.HDC]
    gl.wglMakeCurrent.argtypes = [wintypes.HDC, ctypes.c_void_p]
    gl.wglDeleteContext.argtypes = [ctypes.c_void_p]
    gl.glGetString.restype, gl.glGetString.argtypes = ctypes.c_char_p, [ctypes.c_uint]

    hwnd = user32.CreateWindowExW(0, "STATIC", "ewald OpenGL check", 0, 0, 0, 16, 16, None, None, None, None)
    if not hwnd:
        return None, "could not create a window"
    hdc = user32.GetDC(hwnd)
    ctx = None
    try:
        pfd = PIXELFORMATDESCRIPTOR()
        pfd.nSize, pfd.nVersion = ctypes.sizeof(pfd), 1
        pfd.dwFlags = 0x4 | 0x20 | 0x1          # PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        pfd.cColorBits, pfd.cDepthBits = 32, 24
        fmt = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
        if not fmt or not gdi32.SetPixelFormat(hdc, fmt, ctypes.byref(pfd)):
            return None, "no OpenGL pixel format"
        ctx = gl.wglCreateContext(hdc)
        if not ctx or not gl.wglMakeCurrent(hdc, ctx):
            return None, "no OpenGL context"
        version = (gl.glGetString(0x1F02) or b"").decode("ascii", "replace")    # GL_VERSION
        renderer = (gl.glGetString(0x1F01) or b"").decode("ascii", "replace")   # GL_RENDERER
        return parse_gl_version(version), renderer or "unknown renderer"
    finally:
        if ctx:
            gl.wglMakeCurrent(None, None)
            gl.wglDeleteContext(ctx)
        user32.ReleaseDC(hwnd, hdc)
        user32.DestroyWindow(hwnd)


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
        "(ewald gui host:/path/to/file, or File > Open from > Remote file (SSH)); use a remote desktop with GPU "
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
        from . import first_run

        first_run.start(argv)  # on screen while gui.app imports, the slow part of a first start
        from .app import main as _main

        return _main(argv)
    except QtStartupError as exc:
        sys.exit(f"ewald-gui: {exc}")
