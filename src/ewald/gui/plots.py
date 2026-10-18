"""Plots tab: draw any data table (g(r), histograms, user tables) with matplotlib."""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from .theme import TOKENS


class PlotsPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QComboBox()
        self.table.currentTextChanged.connect(self.redraw)
        self.hist = QtWidgets.QComboBox()
        self.hist.currentTextChanged.connect(self.redraw)
        save = QtWidgets.QPushButton("Save plot…")
        save.clicked.connect(self._save)
        bar.addWidget(QtWidgets.QLabel("Table", objectName="muted"))
        bar.addWidget(self.table, 1)
        bar.addWidget(QtWidgets.QLabel("or histogram of", objectName="muted"))
        bar.addWidget(self.hist, 1)
        bar.addWidget(save)
        lay.addLayout(bar)
        self.figure = Figure(figsize=(5, 3), facecolor=TOKENS["bg"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay.addWidget(self.canvas, 1)

    def set_data(self, data):
        self.data = data
        cur_t, cur_h = self.table.currentText(), self.hist.currentText()
        for combo, items, cur in ((self.table, [""] + list(data.tables) if data else [""], cur_t),
                                  (self.hist, [""] + [k for k, v in data.particles.items()
                                                      if v.ndim == 1 and v.dtype.kind in "biuf"] if data else [""], cur_h)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(cur if cur in items else "")
            combo.blockSignals(False)
        if not self.table.currentText() and not self.hist.currentText() and data is not None and data.tables:
            self.table.setCurrentText(next(iter(data.tables)))
        self.redraw()

    def redraw(self, *_):
        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111, facecolor=TOKENS["bg"])
        for spine in ax.spines.values():
            spine.set_color(TOKENS["border"])
        ax.tick_params(colors=TOKENS["secondary"])
        ax.xaxis.label.set_color(TOKENS["secondary"]); ax.yaxis.label.set_color(TOKENS["secondary"])
        d = self.data
        if d is not None and self.hist.currentText():
            name = self.hist.currentText()
            v = np.asarray(d[name])
            if v.dtype.kind in "iu" and np.ptp(v) < 60:
                vals, counts = np.unique(v, return_counts=True)
                ax.bar(vals, counts, color=TOKENS["accent"])
            else:
                ax.hist(v, bins=50, color=TOKENS["accent"])
            ax.set_xlabel(name); ax.set_ylabel("count")
        elif d is not None and self.table.currentText() in d.tables:
            table = d.tables[self.table.currentText()]
            keys = list(table)
            x = np.asarray(table[keys[0]])
            for k in keys[1:]:
                ax.plot(x, np.asarray(table[k]), lw=1.4, label=k)
            ax.set_xlabel(keys[0])
            if len(keys) > 2:
                leg = ax.legend(frameon=False)
                for t in leg.get_texts():
                    t.set_color(TOKENS["secondary"])
            elif len(keys) == 2:
                ax.set_ylabel(keys[1])
        fig.tight_layout()
        self.canvas.draw_idle()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.pdf", "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=300)
