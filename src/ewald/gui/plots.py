"""Plots tab: draw any data table (g(r), histograms, user tables) with matplotlib.

Tables whose first column is ``Step``/``Timestep`` (the ``Thermo`` table from log.lammps,
OSZICAR, OUTCAR, vasprun.xml) are time series: a column is picked under "y" and a marker
follows the current frame. Changing frames only moves that marker.

Plots are linked to the 3D view (:mod:`ewald.linking`): dragging an x range on a histogram,
g(r), bond length or bond angle plot selects those particles (``atomsSelected``), dragging on a
per-frame plot selects frames (``framesSelected``; Shift+drag a box to select by value too) and
a click there jumps to a frame (``frameRequested``). The other way, :meth:`set_highlight` draws
the viewport's selected particles on the plot and :meth:`set_frame_selection` the selected frames.

The "Phonon Bands" table of a phonon file (band.yaml, OUTCAR of IBRION=5-8) is drawn as a band
structure with its high-symmetry labels; clicking a band picks that mode (``phononModePicked``,
which the main window plays through the PhononMode modifier) and a circle marks the mode shown.

The ``NEB`` table (``NebEnergyProfile``) is drawn as an energy profile: the spline, the
images as points, the current image highlighted, the barriers in the corner; clicking an
image goes to its frame (its ``Frame`` column links it like a per-frame plot).
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import linking
from ..linking import grid_map
from .theme import TOKENS


class PlotsPanel(QtWidgets.QWidget):
    atomsSelected = QtCore.Signal(object)       # particle indices picked by a dragged range
    framesSelected = QtCore.Signal(object)      # frame numbers picked on a per-frame plot
    frameRequested = QtCore.Signal(int)         # click on a time series: go to this frame
    expressionRequested = QtCore.Signal(str)    # keep the range as an ExpressionSelection
    phononModePicked = QtCore.Signal(int, int)  # click on a phonon band: (q-point, band), 0-based

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
        sel = QtWidgets.QHBoxLayout()
        self.link_label = QtWidgets.QLabel(objectName="muted")
        self.keep = QtWidgets.QPushButton("Keep as selection", toolTip="Add an Expression selection "
                                          "modifier for this range, re-evaluated on every frame")
        self.keep.clicked.connect(self._keep)
        self.clear = QtWidgets.QPushButton("Clear", toolTip="Clear the plot selection")
        self.clear.clicked.connect(self.clear_selection)
        sel.addWidget(self.link_label, 1)
        sel.addWidget(self.keep)
        sel.addWidget(self.clear)
        lay.addLayout(sel)
        self.figure = Figure(figsize=(5, 3), facecolor=TOKENS["bg"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("button_press_event", self._clicked)
        self._drawn = None   # (table x column, y choice) currently on the canvas
        self._marker = None  # current-frame line on a time-series plot, current mode on a band plot
        self._phonon = None  # (modes, x) while a phonon band structure is on the canvas
        self.pipeline = None         # for mapping table rows to frames
        self.link = None             # linking.Link of the plot on the canvas
        self.highlight = []          # particles selected in the viewport, drawn on the plot
        self.frame_selection = []    # frames selected (here or in the timeline)
        self._range = None           # (lo, hi) of the dragged particle range
        self._box = None             # (x1, x2, y1, y2) of the dragged frame box
        self._ax = self._ax2 = None
        self._bars = False
        self._edges = None           # histogram bin edges, for the highlight overlay
        self._press = None
        self._rubber = None
        self._overlays = []
        self._update_controls()

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
        if self._range is not None and self.link is not None and self.link.selects_atoms:
            self._emit_atoms()  # the range stays, its particles follow the frame

    def _table(self):
        d = self.data
        return d.tables.get(self.table.currentText()) if d is not None and not self.hist.currentText() else None

    def _fill_columns(self):
        table = self._table()
        keys = list(table)[1:] if table else []
        cur = self.column.currentText()
        items = (["Energy"] if _is_neb(table) else keys if _is_series(table)
                 else ["(all)"] + keys if len(keys) > 1 else keys)
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
        link = None if self._phonon_modes() is not None else self._make_link()
        if link != self.link:
            self._range = self._box = None
            self.link = link
        if key is not None and key == self._drawn and self._marker is not None:
            self._place_marker()  # same series, new frame: move the marker only
            self._draw_overlays()
            self._update_controls()
            return
        self._drawn, self._marker, self._phonon = key, None, None
        fig = self.figure
        fig.clear()
        ax = self._ax = fig.add_subplot(111, facecolor=TOKENS["bg"])
        self._ax2, self._edges, self._overlays, self._rubber = None, None, [], None
        self._bars = False  # integer histogram drawn as bars
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
                self._edges = np.arange(v.min(), v.max() + 2) - 0.5
                self._bars = True
            else:
                _, self._edges, _ = ax.hist(v, bins=50, color=TOKENS["accent"])
            ax.set_xlabel(name); ax.set_ylabel("count")
        elif d is not None and self._phonon_modes() is not None:
            self._draw_phonon(ax, self._phonon_modes())
        elif d is not None and self.table.currentText() in d.tables and grid_map(d.tables[self.table.currentText()]):
            keys, (x, y, z) = grid_map(d.tables[self.table.currentText()])
            cs = ax.contourf(x, y, z, levels=20, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=keys[2])
            ax.set_xlabel(keys[0]); ax.set_ylabel(keys[1])
        elif d is not None and _is_neb(d.tables.get(self.table.currentText())):
            self._draw_neb(ax, d.tables[self.table.currentText()], d.tables.get("NEB spline"))
        elif d is not None and self.table.currentText() in d.tables:
            table = d.tables[self.table.currentText()]
            keys = list(table)
            x = np.asarray(table[keys[0]])
            col = self.column.currentText()
            if col in table:
                keys = [keys[0], col]
            for k in keys[1:]:
                y = np.asarray(table[k], dtype=float)
                ok = ~np.isnan(y)  # thermo columns a run did not print
                ax.plot(x[ok], y[ok], lw=1.4, label=k)
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
            if len(x) > 1:
                mid = 0.5 * (x[1:] + x[:-1])
                self._edges = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
        fig.tight_layout()
        self._draw_overlays()
        self._update_controls()

    # ------------------------------------------------------------ linked selection
    def _make_link(self):
        d = self.data
        if d is None:
            return None
        nf = self.pipeline.num_frames if self.pipeline is not None else None
        col = self.column.currentText()
        return linking.link_for(d, self.table.currentText() or None, None if col == "(all)" else col,
                                hist=self.hist.currentText() or None, num_frames=nf)

    def _plotted_columns(self):
        table = self._table()
        if not table:
            return None
        col = self.column.currentText()
        return [col] if col in table and col != next(iter(table)) else list(table)[1:]

    def select(self, x1, x2, y1=None, y2=None):
        """Select what the plot's range ``x1``..``x2`` (and ``y1``..``y2`` on a per-frame plot or a
        free energy surface; None for the whole y range) stands for and emit it."""
        link, d = self.link, self.data
        if link is None or d is None:
            return
        if link.selects_atoms:
            self._range = tuple(sorted((float(x1), float(x2))))
            self._emit_atoms()
        else:
            lo, hi = (-np.inf, np.inf) if y1 is None or y2 is None else (y1, y2)
            self._box = (x1, x2, lo, hi)
            frames = linking.frames_in_box(self.pipeline, d, link, x1, x2, lo, hi,
                                           columns=None if link.kind == "frames2d" else self._plotted_columns())
            self.frame_selection = list(frames)
            self.framesSelected.emit(list(frames))
        self._draw_overlays()
        self._update_controls()

    def _emit_atoms(self):
        try:
            idx = linking.atoms_in_range(self.data, self.link, *self._range)
        except Exception as exc:  # e.g. bonds missing on this frame
            self.link_label.setText(f"Selection failed: {exc}")
            return
        self._selected_count = len(idx)
        self.atomsSelected.emit(idx)
        self._update_controls()

    def clear_selection(self):
        """Drop the dragged range/box; emits an empty selection of the same kind."""
        atoms = self._range is not None
        frames = self._box is not None or bool(self.frame_selection)
        self._range = self._box = None
        if atoms:
            self.atomsSelected.emit(np.zeros(0, dtype=int))
        if frames:
            self.frame_selection = []
            self.framesSelected.emit([])
        self._draw_overlays()
        self._update_controls()

    def forget_selection(self, frames: bool = True):
        """Drop the dragged particle range (and the frame selection) without emitting."""
        self._range = None
        if frames:
            self._box, self.frame_selection = None, []
        self._draw_overlays()
        self._update_controls()

    def set_highlight(self, indices):
        """Draw the particles selected in the viewport on the plot (no signal)."""
        self.highlight = list(np.asarray(indices, dtype=int)) if indices is not None else []
        self._draw_overlays()

    def set_frame_selection(self, frames):
        """Mark the selected frames on a per-frame plot (no signal)."""
        self.frame_selection = list(frames or [])
        if not self.frame_selection:
            self._box = None
        self._draw_overlays()
        self._update_controls()

    def _update_controls(self):
        link = self.link
        text = ""
        if link is None:
            text = ""
        elif self._range is not None:
            lo, hi = self._range
            text = f"{getattr(self, '_selected_count', 0)} particles with {self._xlabel()} in {lo:.4g} – {hi:.4g}"
        elif self.frame_selection and link.kind in ("frames", "frames2d"):
            f = self.frame_selection
            text = f"{len(f)} frame{'s' if len(f) != 1 else ''} selected" + (
                f" ({f[0]}–{f[-1]})" if len(f) > 1 else f" ({f[0]})")
        elif link.selects_atoms:
            text = "Drag a range to select particles"
        else:
            text = "Drag to select frames (Shift+drag: by value too), click to go to a frame" \
                if link.kind == "frames" else "Drag a box to select the frames in it"
        self.link_label.setText(text)
        self.keep.setVisible(link is not None and link.kind == "property")
        self.keep.setEnabled(self._range is not None)
        self.clear.setEnabled(self._range is not None or self._box is not None or bool(self.frame_selection))

    def _highlight_values(self):
        """values_of() for the highlight, cached per frame and plot (a g(r) highlight runs a
        neighbor search); skipped while the tab is hidden and redone when it is shown."""
        if not self.isVisible():
            self._overlays_stale = True
            return np.zeros(0)
        xmax = float(self._edges[-1]) if self._edges is not None else None
        h = np.asarray(self.highlight, dtype=np.int64)
        key = (id(self.data), self.link, xmax, len(h), hash(h.tobytes()))
        if getattr(self, "_hl_cache", (None,))[0] != key:
            try:
                vals = linking.values_of(self.data, self.link, h, xmax=xmax)
            except Exception:  # e.g. no bonds on this frame
                vals = np.zeros(0)
            self._hl_cache = (key, vals)
        return self._hl_cache[1]

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_overlays_stale", False):
            self._overlays_stale = False
            self._draw_overlays()

    def _counts_axis(self) -> bool:
        """Whether the plot shows plain counts, so a selection's counts share its y axis."""
        if self.hist.currentText():
            return True
        cols = self._plotted_columns() or []
        return bool(cols) and all(c.startswith("Count") for c in cols)

    def _xlabel(self):
        return self._ax.get_xlabel() if self._ax is not None else ""

    def _keep(self):
        expr = linking.selection_expression(self.link, *self._range) if self._range else None
        if expr:
            self.expressionRequested.emit(expr)

    def _draw_overlays(self):
        for a in self._overlays:
            try:
                a.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        self._overlays = []
        ax, link, d = self._ax, self.link, self.data
        if ax is None or link is None or d is None or ax.figure is not self.figure:
            self.canvas.draw_idle()
            return
        pick = TOKENS["pick"]
        if self._range is not None:
            self._overlays.append(ax.axvspan(*self._range, color=pick, alpha=0.15, lw=0))
        if link.selects_atoms and self.highlight:
            vals = self._highlight_values()
            if len(vals) and self._edges is not None and self._counts_axis():
                # counts of a subset: drawn over the full histogram, same axis
                if self._bars:
                    u, c = np.unique(vals, return_counts=True)
                    patches = ax.bar(u, c, color=pick, alpha=0.85)
                else:
                    _, _, patches = ax.hist(vals, bins=self._edges, histtype="stepfilled", color=pick, alpha=0.75)
                self._overlays.extend(patches)
                if self._ax2 is not None:
                    self._ax2.set_visible(False)
            elif len(vals) and self._edges is not None:
                # g(r), frequencies, densities: the selection's counts on their own axis
                if self._ax2 is None:
                    self._ax2 = ax.twinx()
                    self._ax2.tick_params(colors=pick)
                    for spine in self._ax2.spines.values():
                        spine.set_color(TOKENS["border"])
                    self._ax2.set_ylabel("selected", color=pick)
                    self.figure.tight_layout()
                self._ax2.set_visible(True)
                _, _, patches = self._ax2.hist(vals, bins=self._edges, histtype="step", color=pick, lw=1.4)
                self._overlays.extend(patches)
                self._ax2.relim()
                self._ax2.autoscale_view()
            elif self._ax2 is not None:
                self._ax2.set_visible(False)
        elif self._ax2 is not None:
            self._ax2.set_visible(False)
        if link.kind in ("frames", "frames2d"):
            if self._box is not None:
                x1, x2, y1, y2 = self._box
                if np.isfinite(y1):
                    from matplotlib.patches import Rectangle

                    r = Rectangle((min(x1, x2), min(y1, y2)), abs(x2 - x1), abs(y2 - y1), fill=False,
                                  ec=pick, lw=1, ls="--")
                    ax.add_patch(r)
                    self._overlays.append(r)
                else:
                    self._overlays.append(ax.axvspan(x1, x2, color=pick, alpha=0.12, lw=0))
            if self.frame_selection:
                cols = [None] if link.kind == "frames2d" else (self._plotted_columns() or [None])
                for c in cols:
                    try:
                        px, py = linking.frame_points(self.pipeline, d, link, self.frame_selection, column=c)
                    except Exception:
                        continue
                    if len(px):
                        self._overlays.extend(ax.plot(px, py, "o", ms=3.5, color=pick, mec="none", zorder=5))
        self.canvas.draw_idle()

    def _draw_neb(self, ax, table, spline):
        s, e = np.asarray(table["Reaction Coordinate"], float), np.asarray(table["Energy"], float)
        if spline is not None:
            ax.plot(spline["Reaction Coordinate"], spline["Energy"], lw=1.4, color=TOKENS["accent"])
        ok = ~np.isnan(e)
        ax.plot(s[ok], e[ok], "o", ms=6, color=TOKENS["accent"], mfc=TOKENS["bg"])
        ax.set_xlabel("Reaction coordinate (Å)")
        ax.set_ylabel("Energy (eV)")
        a = self.data.attributes
        lines = [f"{label} {a[k]:.3f} eV" for label, k in (("Ea →", "NEB.Barrier"), ("Ea ←", "NEB.Reverse Barrier"),
                                                         ("ΔE", "NEB.Reaction Energy")) if k in a]
        if lines:
            ax.text(0.02, 0.96, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
                    color=TOKENS["secondary"])
        self._marker = ax.plot([], [], "o", ms=9, color=TOKENS["accent"])[0]
        self._place_marker()

    # mouse: drag selects, click on a per-frame plot goes to that frame
    def _on_press(self, event):
        if event.button != 1 or self._ax is None or self.link is None or event.inaxes is None \
                or event.inaxes not in (self._ax, self._ax2):
            self._press = None
            return
        x, y = self._ax.transData.inverted().transform((event.x, event.y))
        shift = "shift" in (event.key or "")
        self._press = (x, y, event.x, event.y, shift)

    def _on_motion(self, event):
        if self._press is None or event.x is None:
            return
        x0, y0, *_, shift = self._press
        x, y = self._ax.transData.inverted().transform((event.x, event.y))
        if self._rubber is not None:
            try:
                self._rubber.remove()
            except (ValueError, NotImplementedError):
                pass
        box = self.link.kind == "frames2d" or (shift and self.link.kind == "frames")
        if box:
            from matplotlib.patches import Rectangle

            self._rubber = self._ax.add_patch(Rectangle((min(x0, x), min(y0, y)), abs(x - x0), abs(y - y0),
                                                        fill=False, ec=TOKENS["pick"], lw=1))
        else:
            self._rubber = self._ax.axvspan(min(x0, x), max(x0, x), color=TOKENS["pick"], alpha=0.25, lw=0)
        self.canvas.draw_idle()

    def _on_release(self, event):
        press, self._press = self._press, None
        if self._rubber is not None:
            try:
                self._rubber.remove()
            except (ValueError, NotImplementedError):
                pass
            self._rubber = None
        if press is None or event.x is None:
            return
        x0, y0, px, py, shift = press
        x, y = self._ax.transData.inverted().transform((event.x, event.y))
        if abs(event.x - px) + abs(event.y - py) < 4:  # a click
            if self.link.kind == "frames":
                f = linking.frame_at(self.pipeline, self.data, self.link, x)
                if f is not None:
                    self.frameRequested.emit(f)
            self.canvas.draw_idle()
            return
        box = self.link.kind == "frames2d" or (shift and self.link.kind == "frames")
        self.select(x0, x, *((y0, y) if box else (None, None)))

    # phonon band structure: click a band to play that mode
    def _phonon_modes(self):
        """The PhononModes behind the selected table when it is a phonon band table shown whole."""
        d = self.data
        if d is None or self.hist.currentText() or self.table.currentText() != "Phonon Bands":
            return None
        modes = d.attributes.get("Phonon Modes")
        if modes is None or not hasattr(modes, "frequencies"):
            return None
        return modes if self.column.currentText() in ("(all)", "Frequency") else None

    def _draw_phonon(self, ax, modes):
        f = modes.frequencies
        if modes.num_qpoints == 1:  # Gamma only (VASP): one tick per mode
            x = np.arange(1, modes.num_bands + 1, dtype=float)
            ax.scatter(x, f[0], marker="_", s=160, color=TOKENS["accent"], linewidths=2)
            ax.set_xlabel("Mode")
        else:
            x = modes.distances if modes.is_path else np.arange(modes.num_qpoints, dtype=float)
            for b in range(modes.num_bands):
                ax.plot(x, f[:, b], lw=1.2, color=TOKENS["accent"])
            ax.set_xlim(x[0], x[-1])
            if modes.ticks:
                for d, _ in modes.ticks:
                    ax.axvline(d, color=TOKENS["border"], lw=0.8)
                ax.set_xticks([d for d, _ in modes.ticks])
                ax.set_xticklabels([lab for _, lab in modes.ticks])
            else:
                ax.set_xlabel("Q-point" if not modes.is_path else "Distance")
        ax.axhline(0, color=TOKENS["border"], lw=0.8)
        ax.set_ylabel("Frequency (THz)")
        ax.set_title("Click a band to play the mode", color=TOKENS["secondary"], fontsize=9)
        self._phonon = (modes, x)
        self._marker, = ax.plot([], [], "o", ms=8, mfc="none", mew=2, color=TOKENS["pick"])
        self._place_marker()

    def _clicked(self, event):
        if self._phonon is None or event.inaxes is None or event.button != 1:
            return
        modes, x = self._phonon
        f = modes.frequencies
        if modes.num_qpoints == 1:
            q, xs, ys = np.zeros(modes.num_bands, int), x, f[0]
            bands = np.arange(modes.num_bands)
        else:
            q = np.repeat(np.arange(modes.num_qpoints), modes.num_bands)
            bands = np.tile(np.arange(modes.num_bands), modes.num_qpoints)
            xs, ys = x[q], f.ravel()
        pts = event.inaxes.transData.transform(np.column_stack([xs, ys]))
        k = int(np.argmin(np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)))
        self.phononModePicked.emit(int(q[k]), int(bands[k]))

    def _place_marker(self):
        if _is_neb(self._table()):
            a = self.data.attributes
            here = "NEB.Relative Energy" in a
            self._marker.set_visible(here)
            if here:
                self._marker.set_data([a["NEB.Reaction Coordinate"]], [a["NEB.Relative Energy"]])
            return
        if self._phonon is not None:
            modes, x = self._phonon
            a = self.data.attributes if self.data is not None else {}
            if "Phonon.QPoint" in a and "Phonon.Band" in a:
                q, b = int(a["Phonon.QPoint"]), int(a["Phonon.Band"])
                xm = x[b] if modes.num_qpoints == 1 else x[q]
                self._marker.set_data([xm], [modes.frequencies[q, b]])
            else:
                self._marker.set_data([], [])
            return
        ts = self.data.attributes.get("Timestep") if self.data is not None else None
        visible = isinstance(ts, (int, float, np.integer, np.floating))
        self._marker.set_visible(visible)
        if visible:
            self._marker.set_xdata([ts, ts])

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.pdf", "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=300)


def _is_neb(table) -> bool:
    return bool(table) and next(iter(table)) == "Reaction Coordinate" and "Frame" in table and "Energy" in table


def _is_series(table) -> bool:
    return bool(table) and next(iter(table)) in ("Step", "Timestep") and len(table) > 1


def _grid(table):
    return grid_map(table)
