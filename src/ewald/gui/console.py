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
    from .theme import TOKENS
    w.syntax_style = "monokai"
    w.style_sheet = f"""
        QPlainTextEdit, QTextEdit {{ background-color: {TOKENS['bg']}; color: {TOKENS['text']};
                                     selection-background-color: {TOKENS['raised']}; border: none; }}
        .error {{ color: {TOKENS['error']}; }}
        .in-prompt, .in-prompt-number {{ color: {TOKENS['accent']}; }}
        .out-prompt, .out-prompt-number {{ color: {TOKENS['selection']}; }}
        .in-prompt-number, .out-prompt-number {{ font-weight: bold; }}
    """
    w.kernel_manager = km
    w.kernel_client = kc
    w.banner = ("ewald console. Objects: pipeline, data (current frame), viewer, np, mods.\n"
                "Changes to the pipeline redraw the view after each command.\n")
    w.shell = shell
    w.on_execute = on_execute  # the in-process shell is shared by every window: unregistered on close
    return w


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
