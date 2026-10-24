"""Embedded Python console sharing objects with the viewer."""
from __future__ import annotations

import code
import io
import contextlib

from PySide6 import QtCore, QtGui, QtWidgets


def make_console(namespace: dict, on_execute, parent=None) -> QtWidgets.QWidget:
    """IPython (qtconsole, in-process kernel) if available, else a minimal console."""
    try:
        return _ipython_console(namespace, on_execute, parent)
    except Exception:
        return SimpleConsole(namespace, on_execute, parent)


class LazyConsole(QtWidgets.QWidget):
    """Stands in for the console until it is first shown (or :meth:`widget` is called): the
    in-process IPython kernel and qtconsole cost about half a second, so the window builds
    them after its first paint, or never if the console tab stays closed. ``namespace`` is a
    function giving the objects to share; ``on_built`` runs once the console exists (the
    window pushes its current state)."""

    def __init__(self, namespace, on_execute, parent=None, on_built=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._namespace, self._on_execute, self._on_built = namespace, on_execute, on_built
        self._console = None
        self._scheduled = False

    @property
    def built(self) -> bool:
        return self._console is not None

    def widget(self) -> QtWidgets.QWidget:
        """The real console, built now if it was not yet."""
        if self._console is None:
            self._console = make_console(self._namespace(), self._on_execute, self)
            self.layout().addWidget(self._console)
            self.setFocusProxy(self._console)
            if self._on_built is not None:
                self._on_built()
        return self._console

    def paintEvent(self, event):
        # after the window has painted (a zero timer from showEvent can run before the first
        # paint): it appears first, then the console
        super().paintEvent(event)
        if self._console is None and not self._scheduled:
            self._scheduled = True
            QtCore.QTimer.singleShot(0, self.widget)

    def __getattr__(self, name):  # shell, kernel_manager, execute, ...: the real console's
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.widget(), name)


def _ipython_console(namespace, on_execute, parent):
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget

    from .. import shell as ipython

    ipython.release_host()   # notebook cells made a private global shell: this kernel's shell takes its place
    km = QtInProcessKernelManager()
    km.start_kernel(show_banner=False)
    km.kernel.gui = "qt"
    shell = km.kernel.shell
    shell.push(namespace)
    from . import workers

    def after(*args):   # the in-process shell outlives windows: one that is gone stops listening
        if parent is None or workers.alive(parent):
            return on_execute(*args)
        shell.events.unregister("post_execute", after)
    shell.events.register("post_execute", after)
    _style_shell(shell)
    kc = km.client()
    kc.start_channels()
    w = RichJupyterWidget(parent)
    restyle(w)
    w.kernel_manager = km
    w.kernel_client = kc
    w.banner = ("ewald console. Objects: pipeline, data (current frame), viewer, np, mods.\n"
                "Changes to the pipeline redraw the view after each command.\n")
    w.shell = shell
    w.on_execute = after  # the in-process shell is shared by every window: unregistered on close
    return w


def pygments_style(tokens: dict):
    """A Pygments style in the colors of UI theme ``tokens``: keywords and builtins in the accent,
    strings and numbers in the data color, comments muted, on the theme background."""
    from pygments.style import Style
    from pygments.token import (Comment, Error, Generic, Keyword, Name, Number, Operator, Punctuation,
                                String, Text, Token)

    t = tokens
    styles = {
        Token: t["text"], Text: t["text"], Punctuation: t["secondary"], Operator: t["secondary"],
        Keyword: f"bold {t['accent']}", Keyword.Constant: t["accent"], Name.Builtin: t["accent"],
        Name.Function: t["text"], Name.Class: f"bold {t['text']}", Name.Decorator: t["selection"],
        Name.Exception: t["error"], String: t["data"], String.Doc: f"italic {t['muted']}",
        Number: t["data"], Comment: f"italic {t['muted']}", Error: t["error"],
        Generic.Prompt: t["accent"], Generic.Traceback: t["error"], Generic.Error: t["error"],
    }
    return type("EwaldConsoleStyle", (Style,), {"background_color": t["bg"], "highlight_color": t["accent_soft"],
                                                "styles": styles})


def restyle(w, tokens: dict | None = None):
    """Give an IPython console widget the colors of the UI theme (background, text, selection,
    syntax and ANSI colors); called again when the theme changes."""
    from .theme import TOKENS

    t = tokens or TOKENS
    if isinstance(w, LazyConsole):
        if not w.built:
            return  # it takes the theme's colors when it is built
        w = w.widget()
    if not hasattr(w, "syntax_style"):   # SimpleConsole follows the application style sheet
        return
    w.style_sheet = f"""
        QPlainTextEdit, QTextEdit {{ background-color: {t['bg']}; color: {t['text']};
                                     selection-background-color: {t['accent']}; selection-color: {t['on_accent']};
                                     border: none; }}
        .error {{ color: {t['error']}; }}
        .in-prompt, .in-prompt-number {{ color: {t['accent']}; }}
        .out-prompt, .out-prompt-number {{ color: {t['data']}; }}
        .in-prompt-number, .out-prompt-number {{ font-weight: bold; }}
    """
    hl = getattr(w, "_highlighter", None)
    if hl is None:
        return
    hl.set_style(pygments_style(t))      # syntax_style only takes Pygments style names
    ap = w._ansi_processor               # ANSI colors (tracebacks) as qtconsole picks them per background
    ap.default_color_map = ap.darkbg_color_map.copy()
    if not dark_color(t["bg"]):
        for i in range(8):
            ap.default_color_map[i + 8] = ap.default_color_map[i]
        ap.default_color_map[7] = ap.default_color_map[15] = t["text"]
    ap.color_map.update(ap.default_color_map)
    hl.rehighlight()


def dark_color(color: str) -> bool:
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5


PLOT_STYLE = ["publication", None, None]   # (style, tokens, user rc) of the console's plots (python.plot_style)


def _plot_session():
    from ..plotstyle import Session

    return Session(*PLOT_STYLE)


def _style_shell(shell):
    """Draw the console's matplotlib figures in :data:`PLOT_STYLE`: its rcParams are swapped in
    around each command (an ewald.plotstyle.Session), so ewald's own renderers in this process
    keep matplotlib's defaults. The in-process shell is shared by every window: hooked once."""
    if getattr(shell, "_ewald_plots", None) is not None:
        return
    try:
        shell._ewald_plots = _plot_session()
    except ImportError:
        return
    active = []

    def pre(*_):
        cm = shell._ewald_plots.active()
        cm.__enter__()
        active.append(cm)

    def post(*_):
        while active:
            active.pop().__exit__(None, None, None)

    shell.events.register("pre_run_cell", pre)
    shell.events.register("post_run_cell", post)


def set_plot_style(console, style: str = "publication", tokens: dict | None = None, user: dict | None = None):
    """Change the style of the console's plots (rcParams a command set stay)."""
    PLOT_STYLE[:] = [style, tokens, user]
    if not getattr(console, "built", True):   # a LazyConsole not built yet reads PLOT_STYLE then
        return
    w = console.widget() if hasattr(console, "widget") else console
    plots = getattr(getattr(w, "shell", None), "_ewald_plots", None) or getattr(w, "plots", None)
    if plots is not None:
        plots.configure(style, tokens, user)


class SimpleConsole(QtWidgets.QWidget):
    def __init__(self, namespace, on_execute, parent=None):
        super().__init__(parent)
        self.shell = code.InteractiveConsole(namespace)
        self.on_execute = on_execute
        try:
            self.plots = _plot_session()
        except ImportError:
            self.plots = None
        self.output = QtWidgets.QPlainTextEdit(readOnly=True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.input = QtWidgets.QLineEdit()
        self.input.setFont(self.output.font())
        self.input.returnPressed.connect(self._run)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.output)
        lay.addWidget(self.input)
        self.output.appendPlainText("ewald console (install qtconsole for IPython). Objects: pipeline, data, viewer, np, mods")

    def _run(self):
        line = self.input.text()
        self.input.clear()
        self.output.appendPlainText(">>> " + line)
        buf = io.StringIO()
        style = self.plots.active() if self.plots is not None else contextlib.nullcontext()
        with style, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.shell.push(line)
        if buf.getvalue():
            self.output.appendPlainText(buf.getvalue().rstrip())
        QtCore.QTimer.singleShot(0, self.on_execute)
