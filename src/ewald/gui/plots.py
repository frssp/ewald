"""Plots tab: draw any data table (g(r), histograms, user tables) with matplotlib.

Tables whose first column is ``Step``/``Timestep`` (the ``Thermo`` table from log.lammps,
OSZICAR, OUTCAR, vasprun.xml) are time series: a column is picked under "y" and a marker
follows the current frame. Changing frames only moves that marker. Tables named ``*-peaks``
(``PowderDiffraction``) draw as sticks, and an ``Experimental`` column as points under the rest.
"""
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
        self.column = QtWidgets.QComboBox(toolTip="Column to plot (all columns share one axis)")
        self.column.currentTextChanged.connect(self.redraw)
        self.hist = QtWidgets.QComboBox()
        self.hist.currentTextChanged.connect(self.redraw)
        save = QtWidgets.QPushButton("Save plot…")
        save.clicked.connect(self._save)
        bar.addWidget(QtWidgets.QLabel("Table", objectName="muted"))
        bar.addWidget(self.table, 1)
        bar.addWidget(QtWidgets.QLabel("y", objectName="muted"))
        bar.addWidget(self.column, 1)
        bar.addWidget(QtWidgets.QLabel("or histogram of", objectName="muted"))
        bar.addWidget(self.hist, 1)
        bar.addWidget(save)
        lay.addLayout(bar)
        self.figure = Figure(figsize=(5, 3), facecolor=TOKENS["bg"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay.addWidget(self.canvas, 1)
        self._drawn = None   # (table x column, y choice) currently on the canvas
        self._marker = None  # current-frame line on a time-series plot

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
        self._fill_columns()
        self.redraw()

    def _table(self):
        d = self.data
        return d.tables.get(self.table.currentText()) if d is not None and not self.hist.currentText() else None

    def _fill_columns(self):
        table = self._table()
        keys = list(table)[1:] if table else []
        cur = self.column.currentText()
        items = keys if _is_series(table) else ["(all)"] + keys if len(keys) > 1 else keys
        self.column.blockSignals(True)
        self.column.clear()
        self.column.addItems(items)
        self.column.setCurrentText(cur if cur in items else items[0] if items else "")
        self.column.setEnabled(len(items) > 1)
        self.column.blockSignals(False)

    def redraw(self, *_):
        self._fill_columns()
        table = self._table()
        key = (id(next(iter(table.values()))), self.column.currentText()) if table else None
        if key is not None and key == self._drawn and self._marker is not None:
            self._place_marker()  # same series, new frame: move the marker only
            self.canvas.draw_idle()
            return
        self._drawn, self._marker = key, None
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
        elif d is not None and self.table.currentText() in d.tables and _grid(d.tables[self.table.currentText()]):
            keys, (x, y, z) = _grid(d.tables[self.table.currentText()])
            cs = ax.contourf(x, y, z, levels=20, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=keys[2])
            ax.set_xlabel(keys[0]); ax.set_ylabel(keys[1])
        elif d is not None and self.table.currentText() in d.tables:
            table = d.tables[self.table.currentText()]
            keys = list(table)
            x = np.asarray(table[keys[0]])
            col = self.column.currentText()
            if col in table:
                keys = [keys[0], col]
            sticks = self.table.currentText().endswith("-peaks") and "Intensity" in table
            if sticks:   # reflections (PowderDiffraction)
                keys = [keys[0], "Intensity"]
                ax.vlines(x, 0, np.asarray(table["Intensity"], float), color=TOKENS["accent"], lw=1.2)
                ax.set_ylim(bottom=0)
            for k in [] if sticks else keys[1:]:
                y = np.asarray(table[k], dtype=float)
                ok = ~np.isnan(y)  # thermo columns a run did not print
                if k == "Experimental":   # measured pattern under the calculated one
                    ax.plot(x[ok], y[ok], lw=0, marker=".", ms=2.5, color=TOKENS["secondary"], label=k, zorder=1)
                else:
                    ax.plot(x[ok], y[ok], lw=1.4, label=k, zorder=2)
            ax.set_xlabel(keys[0])
            if _is_series(table):
                self._marker = ax.axvline(x[0], color=TOKENS["secondary"], lw=1, ls="--")
                self._place_marker()
            if len(keys) > 2:
                leg = ax.legend(frameon=False)
                for t in leg.get_texts():
                    t.set_color(TOKENS["secondary"])
            elif len(keys) == 2:
                ax.set_ylabel(keys[1])
        fig.tight_layout()
        self.canvas.draw_idle()

    def _place_marker(self):
        ts = self.data.attributes.get("Timestep") if self.data is not None else None
        visible = isinstance(ts, (int, float, np.integer, np.floating))
        self._marker.set_visible(visible)
        if visible:
            self._marker.set_xdata([ts, ts])

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.pdf", "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=300)


def _is_series(table) -> bool:
    return bool(table) and next(iter(table)) in ("Step", "Timestep") and len(table) > 1


def _grid(table):
    """``(names, (x, y, z))`` when ``table`` is a map on a regular 2D grid (three columns, the
    first two covering every combination of their values, e.g. a 2D free energy surface)."""
    keys = list(table)
    if len(keys) != 3:
        return None
    x, y, z = (np.asarray(table[k], float) for k in keys)
    ux, uy = np.unique(x), np.unique(y)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(z):
        return None
    ix, iy = np.searchsorted(ux, x), np.searchsorted(uy, y)
    grid = np.full((len(uy), len(ux)), np.nan)
    grid[iy, ix] = z
    if np.isnan(grid).any():
        return None
    return keys, (ux, uy, grid)
