"""Animation timeline: slider, spinbox, play/pause."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class Timeline(QtWidgets.QWidget):
    frameChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self.first = QtWidgets.QToolButton(text="⏮", toolTip="First frame")
        self.play = QtWidgets.QToolButton(text="▶", toolTip="Play / pause (Space)", checkable=True)
        self.last = QtWidgets.QToolButton(text="⏭", toolTip="Last frame")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spin = QtWidgets.QSpinBox()
        self.count = QtWidgets.QLabel("/ 0")
        self.fps = QtWidgets.QSpinBox(suffix=" fps", minimum=1, maximum=120, value=10)
        self.loop = QtWidgets.QCheckBox("loop", checked=True)
        for w in (self.first, self.play, self.last, self.slider, self.spin, self.count, self.fps, self.loop):
            lay.addWidget(w)
        lay.setStretch(3, 1)
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
        nxt = self.frame + 1
        if nxt > self.slider.maximum():
            if not self.loop.isChecked():
                self.play.setChecked(False)
                return
            nxt = 0
        self._set(nxt)
