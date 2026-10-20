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

    def showEvent(self, event):
        if self._console is None:  # after the window has painted: it appears first, then the console
            QtCore.QTimer.singleShot(0, self.widget)
        super().showEvent(event)

    def __getattr__(self, name):  # shell, kernel_manager, execute, ...: the real console's
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.widget(), name)


def _ipython_console(namespace, on_execute, parent):
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget

    km = QtInProcessKernelManager()
    km.start_kernel(show_banner=False)
    km.kernel.gui = "qt"
    shell = km.kernel.shell
    shell.push(namespace)
    shell.events.register("post_execute", on_execute)
    kc = km.client()
    kc.start_channels()
    w = RichJupyterWidget(parent)
    restyle(w)
    w.kernel_manager = km
    w.kernel_client = kc
    w.banner = ("ewald console. Objects: pipeline, data (current frame), viewer, np, mods.\n"
                "Changes to the pipeline redraw the view after each command.\n")
    w.shell = shell
    w.on_execute = on_execute  # the in-process shell is shared by every window: unregistered on close
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


class SimpleConsole(QtWidgets.QWidget):
    def __init__(self, namespace, on_execute, parent=None):
        super().__init__(parent)
        self.shell = code.InteractiveConsole(namespace)
        self.on_execute = on_execute
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
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self.shell.push(line)
        if buf.getvalue():
            self.output.appendPlainText(buf.getvalue().rstrip())
        QtCore.QTimer.singleShot(0, self.on_execute)
