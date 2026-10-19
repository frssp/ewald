"""Animation timeline: slider, spinbox, play/pause, and the frames selected on a plot."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import TOKENS


class _MarkedSlider(QtWidgets.QSlider):
    """Slider with a tick under every selected frame."""

    def __init__(self, *args):
        super().__init__(*args)
        self.marks: list[int] = []

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.marks or self.maximum() <= self.minimum():
            return
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self)
        handle = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        x0 = groove.x() + handle.width() // 2
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor(TOKENS["pick"]), 1))
        top, bottom = self.rect().top() + 1, self.rect().bottom() - 1
        for f in self.marks:
            x = x0 + QtWidgets.QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), f, span)
            p.drawLine(x, top, x, top + 3)
            p.drawLine(x, bottom - 3, x, bottom)
        p.end()


class Timeline(QtWidgets.QWidget):
    frameChanged = QtCore.Signal(int)
    selectionChanged = QtCore.Signal(object)  # selected frames (empty: none), from the clear button

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self.first = QtWidgets.QToolButton(text="⏮", toolTip="First frame (Home)")
        self.prev = QtWidgets.QToolButton(text="◀", toolTip="Previous frame (Alt+←)")
        self.play = QtWidgets.QToolButton(text="▶", toolTip="Play / pause (Space)", checkable=True)
        self.next = QtWidgets.QToolButton(text="▶|", toolTip="Next frame (Alt+→)")
        self.last = QtWidgets.QToolButton(text="⏭", toolTip="Last frame (End)")
        self.slider = _MarkedSlider(QtCore.Qt.Horizontal)
        self.sel = QtWidgets.QToolButton(text="", toolTip="Frames selected on a plot; playback steps through "
                                         "them. Click to clear.")
        self.sel.setStyleSheet(f"color: {TOKENS['pick']};")
        self.sel.clicked.connect(lambda: self.set_selection([], emit=True))
        self.spin = QtWidgets.QSpinBox()
        self.count = QtWidgets.QLabel("/ 0")
        self.fps = QtWidgets.QSpinBox(suffix=" fps", minimum=1, maximum=120, value=10)
        self.stride = QtWidgets.QSpinBox(prefix="every ", minimum=1, maximum=10000, value=1)
        self.loop = QtWidgets.QCheckBox("Loop", checked=True)
        for w in (self.first, self.prev, self.play, self.next, self.last, self.slider, self.spin, self.count,
                  self.sel, self.fps, self.stride, self.loop):
            lay.addWidget(w)
        lay.setStretch(5, 1)
        self.prev.clicked.connect(lambda: self._set(max(0, self.frame - self.stride.value())))
        self.next.clicked.connect(lambda: self._set(min(self.slider.maximum(), self.frame + self.stride.value())))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.slider.valueChanged.connect(self._set)
        self.spin.valueChanged.connect(self._set)
        self.first.clicked.connect(lambda: self._set(0))
        self.last.clicked.connect(lambda: self._set(self.slider.maximum()))
        self.play.toggled.connect(self._toggle)
        self.fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self.set_num_frames(1)

    @property
    def frame(self) -> int:
        return self.slider.value()

    def set_num_frames(self, n: int):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(0, max(0, n - 1))
            w.blockSignals(False)
        self.count.setText(f"/ {max(0, n - 1)}")
        self.setEnabled(n > 1)
        self.set_selection([])

    @property
    def selection(self) -> list[int]:
        """Selected frames (sorted), or [] for none."""
        return list(self.slider.marks)

    def set_selection(self, frames, emit: bool = False):
        """Mark ``frames`` on the slider; playback then steps through them only."""
        n = self.slider.maximum() + 1
        self.slider.marks = sorted({int(f) for f in frames or () if 0 <= int(f) < n})
        k = len(self.slider.marks)
        self.sel.setText(f"{k} selected ✕" if k else "")
        self.sel.setVisible(bool(k))
        self.slider.update()
        if emit:
            self.selectionChanged.emit(self.selection)

    def _set(self, value: int):
        for w in (self.slider, self.spin):
            if w.value() != value:
                w.blockSignals(True)
                w.setValue(value)
                w.blockSignals(False)
        self.frameChanged.emit(value)

    def _toggle(self, on):
        self.play.setText("⏸" if on else "▶")
        if on:
            self.timer.start(int(1000 / self.fps.value()))
        else:
            self.timer.stop()

    def _advance(self):
        marks = self.slider.marks
        if marks:
            later = [f for f in marks if f > self.frame]
            if not later and not self.loop.isChecked():
                self.play.setChecked(False)
                return
            self._set(later[0] if later else marks[0])
            return
        nxt = self.frame + self.stride.value()
        if nxt > self.slider.maximum():
            if not self.loop.isChecked():
                self.play.setChecked(False)
                return
            nxt = 0
        self._set(nxt)
