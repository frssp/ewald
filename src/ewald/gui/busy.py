"""Busy indicators: a small spinning arc for buttons and rows whose action runs in the
background (plugin install, remote connect, ...), and :class:`BusyLabel`, a spinner with a text."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS


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
