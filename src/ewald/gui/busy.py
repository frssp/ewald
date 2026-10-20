"""Busy indicators: a small spinning arc for buttons and rows whose action runs in the
background (plugin install, remote connect, ...), :class:`BusyLabel`, a spinner with a text, and
:func:`wait_for`, which runs a blocking call (opening a file, an SSH request) off the GUI thread."""
from __future__ import annotations

import contextlib

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS


@contextlib.contextmanager
def busy(say=None, text: str = ""):
    """Around a call that blocks the GUI thread: a wait cursor, and ``say(text)`` (a status line)
    painted before the call starts, so the window shows what it is waiting for."""
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        if say is not None and text:
            say(text)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


class Cancelled(Exception):
    """The user pressed Cancel while :func:`wait_for` waited."""


DIALOG_DELAY_MS = 400     # wait_for shows its progress dialog only for calls slower than this
_orphans: set = set()     # cancelled calls still finishing in the background


class _Call(QtCore.QThread):
    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.result = self.error = None

    def run(self):
        try:
            self.result = self.fn()
        except BaseException as exc:          # handed back to the GUI thread
            self.error = exc


def wait_for(fn, text: str, parent=None, say=None, delay_ms: int | None = None):
    """Run ``fn()`` (a file read, an SSH call) on a worker thread and return its result, raising what
    it raised. The window keeps painting; ``say(text)`` goes to the status line, and a call slower
    than ``delay_ms`` gets a small window-modal dialog with ``text``, a busy bar and Cancel. Cancel
    raises :class:`Cancelled` at once; the call itself finishes in the background and is dropped
    (a connect timeout cannot be interrupted)."""
    delay_ms = DIALOG_DELAY_MS if delay_ms is None else delay_ms
    if say is not None and text:
        say(text)
    call = _Call(fn)
    cancelled = False
    dialog = None
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        call.start()
        # polled with processEvents rather than a nested QEventLoop, which would also run the
        # deferred deletes of the caller's widgets. First a short wait that ignores clicks and
        # keys, so a quick call shows no dialog at all
        clock = QtCore.QElapsedTimer()
        clock.start()
        while not call.wait(10) and clock.elapsed() < delay_ms:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        if call.isRunning():
            dialog = QtWidgets.QProgressDialog(text, "Cancel", 0, 0, parent)
            dialog.setObjectName("waitDialog")
            dialog.setWindowTitle("EWALD")
            dialog.setWindowModality(QtCore.Qt.WindowModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            stop = []
            dialog.canceled.connect(lambda: stop.append(True))
            dialog.rejected.connect(lambda: stop.append(True))      # Esc
            dialog.show()
            QtWidgets.QApplication.restoreOverrideCursor()   # the dialog has a Cancel button to point at
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
            while not call.wait(15) and not stop:
                QtWidgets.QApplication.processEvents()       # the modal dialog keeps clicks off the window
            cancelled = bool(stop) and call.isRunning()
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
    if cancelled:
        _orphans.add(call)
        call.finished.connect(lambda: _orphans.discard(call))
        raise Cancelled(text)
    call.wait()
    if call.error is not None:
        raise call.error
    return call.result


def finish_orphans(ms: int = 15000) -> None:
    """Wait for cancelled :func:`wait_for` calls still running (before the window closes: Qt aborts
    on a thread destroyed while running)."""
    for call in list(_orphans):
        call.wait(ms)
    _orphans.clear()


class Spinner(QtWidgets.QWidget):
    """An indeterminate progress circle drawn in the theme accent; spins only while visible."""

    def __init__(self, size: int = 16, parent=None):
        super().__init__(parent)
        self.setObjectName("spinner")
        self.setFixedSize(size, size)
        self.angle = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._step)
        self.setAccessibleName("Busy")

    def _step(self):
        self.angle = (self.angle + 12) % 360
        self.update()

    def showEvent(self, event):
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    @property
    def spinning(self) -> bool:
        return self._timer.isActive()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        width = max(2.0, self.width() / 8)
        rect = QtCore.QRectF(self.rect()).adjusted(width, width, -width, -width)
        track = QtGui.QColor(TOKENS.get("border", "#888888"))
        p.setPen(QtGui.QPen(track, width))
        p.drawEllipse(rect)
        pen = QtGui.QPen(QtGui.QColor(TOKENS.get("accent", "#1A66CC")), width)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(-self.angle * 16), 100 * 16)   # Qt angles are 1/16 degree, counter-clockwise
        p.end()


class BusyLabel(QtWidgets.QWidget):
    """A :class:`Spinner` followed by a short text such as "Installing…"."""

    def __init__(self, text: str = "", size: int = 16, parent=None):
        super().__init__(parent)
        self.setObjectName("busyLabel")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.spinner = Spinner(size, self)
        self.label = QtWidgets.QLabel(text, objectName="muted")
        lay.addWidget(self.spinner)
        lay.addWidget(self.label)

    def text(self) -> str:
        return self.label.text()

    def setText(self, text: str):
        self.label.setText(text)
