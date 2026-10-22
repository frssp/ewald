"""Plots tab: draw any data table (g(r), histograms, user tables) with matplotlib.

Tables whose first column is ``Step``/``Timestep`` (the ``Steps`` table from log.lammps,
OSZICAR, OUTCAR, vasprun.xml) are time series: a column is picked under "y" and a marker
follows the current frame. Changing frames only moves that marker. Tables named ``*-peaks``
(``PowderDiffraction``) draw as sticks, and an ``Experimental`` column as points under the rest.

Plots are linked to the 3D view (:mod:`ewald.linking`): dragging an x range on a histogram,
g(r), bond length or bond angle plot selects those particles (``atomsSelected``), dragging on a
per-frame plot selects frames (``framesSelected``; Shift+drag a box to select by value too) and
a click there jumps to a frame (``frameRequested``). The other way, :meth:`set_highlight` draws
the viewport's selected particles on the plot and :meth:`set_frame_selection` the selected frames.

The "Phonon Bands" table of a phonon file (band.yaml, OUTCAR of IBRION=5-8) is drawn as a band
structure with its high-symmetry labels; clicking a band picks that mode (``phononModePicked``,
which the main window plays through the PhononMode modifier) and a circle marks the mode shown.

The "DOS" and "Bands" tables of a VASP run with electronic output (vasprun.xml, vaspout.h5;
:mod:`ewald.electronic`) are drawn as a density of states and a band structure, relative to the
Fermi energy. The particles selected in the viewport show up as their projected DOS (restricted
to the orbital of the chosen column, e.g. "Ga d") or, with band projections (LORBIT >= 10), as
fat bands whose markers grow with the selected atoms' weight in each state.

The ``NEB`` table (``NebEnergyProfile``) is drawn as an energy profile: the spline, the
images as points, the current image highlighted, the barriers in the corner; clicking an
image goes to its frame (its ``Frame`` column links it like a per-frame plot).

Every plot can be navigated the same way: the wheel zooms x around the cursor (Ctrl+wheel: y;
the energy axis on a band structure), a right or middle drag pans, a left drag on a plot that
selects nothing (DOS, bands, phonons, maps) zooms to the dragged range and a double-click goes
back to the full range (:meth:`set_view`, :meth:`reset_view`; ``viewChanged`` reports it). The
range stays while frames change and *Save plot…* writes what is shown. DOS and band structure
share one energy window, also typed in the E boxes. Hovering shows the values under the cursor
and, on a per-frame plot, the frame (the NEB image) a click goes to; a click on a two-CV plot
goes to the nearest frame.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import linking, plotstyle
from ..linking import grid_map
from .theme import TOKENS


ENERGY_WINDOW = (-6.0, 6.0)  # eV around the Fermi energy shown for DOS and bands
ZOOM_STEP = 1.25             # range factor per wheel step
CLICK_RADIUS = 12.0          # px: a click on a two-CV plot goes to a frame this close
NAV_HINT = "Wheel: zoom (Ctrl: other axis) · right or middle drag: pan · double-click: full range"
SCREEN_FONT = 9.0            # pt on the canvas (12 px at 100 dpi, the UI font)


def _styled(method):
    """Run a drawing method in the plot style, in the colors of the active UI theme."""
    def wrapper(self, *args, **kwargs):
        with plotstyle.context(TOKENS, self._font_size):
            return method(self, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


class PlotsPanel(QtWidgets.QWidget):
    atomsSelected = QtCore.Signal(object)       # particle indices picked by a dragged range
    framesSelected = QtCore.Signal(object)      # frame numbers picked on a per-frame plot
    frameRequested = QtCore.Signal(int)         # click on a time series: go to this frame
    expressionRequested = QtCore.Signal(str)    # keep the range as an ExpressionSelection
    phononModePicked = QtCore.Signal(int, int)  # click on a phonon band: (q-point, band), 0-based
    viewChanged = QtCore.Signal(object, object)  # zoomed / panned range (xlim, ylim), None = full

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.save_button = QtWidgets.QPushButton("Save plot…")
        self.save_button.clicked.connect(self._save)
        self.table_label = QtWidgets.QLabel("Table", objectName="muted")
        self.hist_label = QtWidgets.QLabel("or histogram of", objectName="muted")
        bar.addWidget(self.table_label)
        bar.addWidget(self.table, 1)
        bar.addWidget(QtWidgets.QLabel("y", objectName="muted"))
        bar.addWidget(self.column, 1)
        bar.addWidget(self.hist_label)
        bar.addWidget(self.hist, 1)
        bar.addWidget(self.save_button)
        lay.addLayout(bar)
        sel = QtWidgets.QHBoxLayout()
        self.energy_label = QtWidgets.QLabel("E", objectName="muted")
        self.e_min, self.e_max = (QtWidgets.QDoubleSpinBox(minimum=-1000, maximum=1000, singleStep=0.5, decimals=2,
                                                           suffix=" eV", value=v, keyboardTracking=False)
                                  for v in ENERGY_WINDOW)
        self.e_min.setToolTip("Lowest energy shown, relative to the Fermi energy")
        self.e_max.setToolTip("Highest energy shown, relative to the Fermi energy")
        sel.addWidget(self.energy_label)
        for box in (self.e_min, self.e_max):
            box.valueChanged.connect(self._energy_typed)
            sel.addWidget(box)
        self.link_label = QtWidgets.QLabel(objectName="muted")
        self.keep = QtWidgets.QPushButton("Keep as selection", toolTip="Add an Expression selection "
                                          "modifier for this range, re-evaluated on every frame")
        self.keep.clicked.connect(self._keep)
        self.clear = QtWidgets.QPushButton("Clear", toolTip="Clear the plot selection")
        self.clear.clicked.connect(self.clear_selection)
        sel.addWidget(self.link_label, 1)
        self.readout = QtWidgets.QLabel(objectName="muted", toolTip=NAV_HINT)
        sel.addWidget(self.readout)
        sel.addWidget(self.keep)
        sel.addWidget(self.clear)
        lay.addLayout(sel)
        # matplotlib (0.3 s to import) and the canvas are made on first use of ``figure``/``canvas``
        self._figure = self._canvas = None
        self._canvas_slot = QtWidgets.QWidget()
        lay.addWidget(self._canvas_slot, 1)
        self._drawn = None   # (table x column, y choice) currently on the canvas
        self._live = None    # artists a new frame can update in place (see _update_in_place)
        self._marker = None  # current-frame line on a time-series plot, current mode on a band plot
        self._phonon = None  # (modes, x) while a phonon band structure is on the canvas
        self._elec = None    # (ElectronicStructure, "DOS" | "Bands", column) while one is on the canvas
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
        self._font_size = SCREEN_FONT
        self._exporting = False      # drawing a file: no UI hints, no current-frame marker
        self._xname = ""             # x column of the plot, for the selection text
        self._view = {}              # ranges the user zoomed to: {"x": (lo, hi), "y": (lo, hi)}
        self._view_key = None        # (table, column, histogram) the zoom belongs to
        self._energy_window = tuple(ENERGY_WINDOW)  # shared by the DOS and the band structure
        self._pan = None
        self.extra_menu = None       # () -> [(label, callable)]: a cell's entries (where the plot is shown)
        self._update_controls()

    @property
    def figure(self):
        if self._figure is None:
            self._build_canvas()
        return self._figure

    @figure.setter
    def figure(self, figure):  # export() draws into a print figure for a moment
        self._figure = figure

    @property
    def canvas(self):
        if self._canvas is None:
            self._build_canvas()
        return self._canvas

    @property
    def canvas_built(self) -> bool:
        return self._canvas is not None

    def _build_canvas(self):
        import matplotlib as mpl
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # outside the plot style a redraw may be running in: the screen figure keeps the rc
        # settings matplotlib started with (its DPI above all)
        with mpl.rc_context(mpl.rcParamsOrig):
            self._figure = Figure(figsize=(5, 3), facecolor=TOKENS["bg"])
            self._canvas = canvas = FigureCanvasQTAgg(self._figure)
        self.layout().replaceWidget(self._canvas_slot, canvas)
        self._canvas_slot.deleteLater()
        self._canvas_slot = None
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("scroll_event", self._on_scroll)

    def compact(self):
        """Narrow layout for a notebook view cell: shorter labels, Save in the cell's menu."""
        self.table_label.hide()
        self.hist_label.setText("or")
        self.save_button.hide()
        for combo in (self.table, self.column, self.hist):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
        self.link_label.setWordWrap(True)
        self.link_label.setMinimumWidth(40)

    def save_plot(self):
        self._save()

    def set_data(self, data):
        self.data = data
        cur_t, cur_h = self.table.currentText(), self.hist.currentText()
        for combo, items, cur in ((self.table, [""] + list(data.tables) if data else [""], cur_t),
                                  (self.hist, [""] + [k for k, v in data.particles.items()
                                                      if v.ndim == 1 and v.dtype.kind in "biuf"] if data else [""], cur_h)):
            if [combo.itemText(i) for i in range(combo.count())] == items:
                continue   # the same tables and properties as the last frame
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
        if [self.column.itemText(i) for i in range(self.column.count())] == items and \
                (cur in items or not items):
            self.column.setEnabled(len(items) > 1)
            return
        self.column.blockSignals(True)
        self.column.clear()
        self.column.addItems(items)
        self.column.setCurrentText(cur if cur in items else items[0] if items else "")
        self.column.setEnabled(len(items) > 1)
        self.column.blockSignals(False)

    def restyle(self):
        """Redraw in the colors of the UI theme now active (after a theme switch)."""
        self.figure.set_facecolor(TOKENS["bg"])
        self._drawn = None   # a full redraw, not just a moved marker
        self.redraw()

    @_styled
    def redraw(self, *_):
        self._fill_columns()
        key = self._content_key()
        link = None if self._phonon_modes() is not None or self._electronic() is not None else self._make_link()
        if link != self.link:
            self._range = self._box = None
            self.link = link
        vkey = (self.table.currentText(), self.column.currentText(), self.hist.currentText())
        if vkey != self._view_key:
            self._view_key, self._view = vkey, {}
        if key is not None and _same_content(key, self._drawn):
            # same data as on the canvas (a frame-independent or trajectory-wide result during
            # playback): move the current-frame marker only, no fig.clear()
            if self._marker is not None:
                self._place_marker()
            self._draw_overlays()
            self._update_controls()
            return
        if key is not None and self._update_in_place(key):
            # the same plot with new numbers (a per-frame histogram or g(r) during playback):
            # new bar heights or line data on the axes already there, no fig.clear()
            self._drawn = key
            self._draw_overlays()
            self._update_controls()
            return
        self._drawn, self._marker, self._phonon, self._elec = key, None, None, None
        self._live = None
        fig = self.figure
        fig.clear()
        fig.set_facecolor(TOKENS["bg"])
        ax = self._ax = fig.add_subplot(111, facecolor=TOKENS["bg"])
        self._ax2, self._edges, self._overlays, self._rubber = None, None, [], None
        self._bars = False  # integer histogram drawn as bars
        self._xname = ""
        d = self.data
        if d is not None and self.hist.currentText():
            name = self.hist.currentText()
            v = np.asarray(d[name])
            if v.dtype.kind in "iu" and np.ptp(v) < 60:
                vals, counts = np.unique(v, return_counts=True)
                bars = ax.bar(vals, counts, color=TOKENS["accent"], lw=0)
                self._live = ("bars", list(bars), vals)
                self._edges = np.arange(v.min(), v.max() + 2) - 0.5
                self._bars = True
                ax.xaxis.set_major_locator(_integer_locator())
                ax.set_xlim(vals.min() - 1, vals.max() + 1)
                plotstyle.no_minor_x(ax)
            else:
                _, self._edges, patches = ax.hist(v, bins=50, color=TOKENS["accent"], lw=0)
                self._live = ("hist", list(patches))
            ax.set_ylim(bottom=0)
            self._xname = name
            ax.set_xlabel(plotstyle.axis_label(name)); ax.set_ylabel("Count")
        elif d is not None and self._phonon_modes() is not None:
            self._draw_phonon(ax, self._phonon_modes())
        elif d is not None and self._electronic() is not None:
            self._draw_electronic(ax, self._electronic())
        elif d is not None and self.table.currentText() in d.tables and grid_map(d.tables[self.table.currentText()]):
            keys, (x, y, z) = grid_map(d.tables[self.table.currentText()])
            cs = ax.contourf(x, y, z, levels=20, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=plotstyle.axis_label(keys[2]))
            ax.set_xlabel(plotstyle.axis_label(keys[0])); ax.set_ylabel(plotstyle.axis_label(keys[1]))
        elif d is not None and _is_neb(d.tables.get(self.table.currentText())):
            self._draw_neb(ax, d.tables[self.table.currentText()], d.tables.get("NEB spline"))
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
                ax.vlines(x, 0, np.asarray(table["Intensity"], float), color=TOKENS["accent"], lw=1.0)
                ax.set_ylim(bottom=0)
            lines = []
            for k, label in [] if sticks else zip(keys[1:], plotstyle.legend_labels(keys[1:])):
                y = np.asarray(table[k], dtype=float)
                ok = ~np.isnan(y)  # thermo columns a run did not print
                if k == "Experimental":   # measured pattern under the calculated one
                    line, = ax.plot(x[ok], y[ok], lw=0, marker=".", ms=2.5, color=TOKENS["muted"], label=label,
                                    zorder=1)
                elif k.endswith(" fit"):
                    line, = ax.plot(x[ok], y[ok], ls="--", lw=0.9, label=label, zorder=3)
                else:
                    line, = ax.plot(x[ok], y[ok], label=label, zorder=2)
                lines.append((k, line))
            if lines:
                self._live = ("lines", lines, tuple(table))
            self._xname = keys[0]
            ax.set_xlabel(plotstyle.axis_label(keys[0]))
            if not _is_series(table):
                plotstyle.start_at_zero(ax, x)
            if sticks or all(k.startswith("Count") for k in keys[1:]) or \
                    self.table.currentText() in ("diffraction", "structure-factor"):
                ax.set_ylim(bottom=0)
            if _is_series(table):
                self._marker = ax.axvline(x[0], color=TOKENS["secondary"], lw=0.8, ls="--")
                self._place_marker()
            units = {plotstyle.axis_label(k) for k in keys[1:]}
            if len(keys) > 2:
                ax.legend()
            if len(units) == 1:
                ax.set_ylabel(units.pop())
            if len(x) > 1:
                mid = 0.5 * (x[1:] + x[:-1])
                self._edges = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
        if self._exporting and self._marker is not None:   # current frame / image / mode: UI state
            self._marker.set_visible(False)
        energy = self._energy_axis()
        for axis, lim in self._view.items():
            if axis != energy:
                (ax.set_xlim if axis == "x" else ax.set_ylim)(lim)
        fig.tight_layout(pad=0.4)
        self._draw_overlays()
        self._update_controls()

    def _update_in_place(self, key) -> bool:
        """Put new numbers into the artists of the plot on the canvas when only the data
        changed (same table, column and histogram choice, same columns, same number of bins
        or lines). Rebuilding the figure costs about twice as much, so this keeps per-frame
        plots responsive during playback. False when a full redraw is needed."""
        live, ax, d = getattr(self, "_live", None), self._ax, self.data
        if live is None or self._drawn is None or key[0] != self._drawn[0] or ax is None \
                or ax.figure is not self.figure or d is None:
            return False
        kind = live[0]
        if kind in ("hist", "bars"):
            name = self.hist.currentText()
            if not name or name not in d:
                return False
            v = np.asarray(d[name])
            if kind == "hist":
                if v.dtype.kind in "iu" and np.ptp(v) < 60:
                    return False  # now drawn as integer bars
                counts, edges = np.histogram(v, bins=len(live[1]))
                for rect, c, lo, hi in zip(live[1], counts, edges[:-1], edges[1:]):
                    rect.set_x(lo)
                    rect.set_width(hi - lo)
                    rect.set_height(c)
                self._edges = edges
            else:
                if v.dtype.kind not in "iu":
                    return False
                vals, counts = np.unique(v, return_counts=True)
                if not np.array_equal(vals, live[2]):
                    return False  # other values: other bars
                for rect, c in zip(live[1], counts):
                    rect.set_height(c)
            ax.set_autoscale_on(True)  # set_xlim/set_ylim of the last draw turned it off
            ax.relim()
            ax.autoscale_view()
            if kind == "bars":
                ax.set_xlim(vals.min() - 1, vals.max() + 1)
            ax.set_ylim(bottom=0)
        elif kind == "lines":
            table = self._table()
            if not table or tuple(table) != live[2]:
                return False
            keys = list(table)
            x = np.asarray(table[keys[0]])
            for k, line in live[1]:
                y = np.asarray(table[k], dtype=float)
                ok = ~np.isnan(y)
                line.set_data(x[ok], y[ok])
            ax.set_autoscale_on(True)  # set_xlim/set_ylim of the last draw turned it off
            ax.relim()
            ax.autoscale_view()
            names = [k for k, _ in live[1]]
            if not _is_series(table):
                plotstyle.start_at_zero(ax, x)
            if all(k.startswith("Count") for k in names) or \
                    self.table.currentText() in ("diffraction", "structure-factor"):
                ax.set_ylim(bottom=0)
            if self._marker is not None:
                self._place_marker()
            if len(x) > 1:
                mid = 0.5 * (x[1:] + x[:-1])
                self._edges = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
        else:
            return False
        energy = self._energy_axis()
        for axis, lim in self._view.items():
            if axis != energy:
                (ax.set_xlim if axis == "x" else ax.set_ylim)(lim)
        return True

    def _content_key(self):
        """What the canvas would show: the choices and the arrays behind them (compared by
        identity, or by value for a recomputed copy), None when it cannot be told."""
        d = self.data
        if d is None:
            return None
        labels = (self.table.currentText(), self.column.currentText(), self.hist.currentText(),
                  tuple(self._energy_window), self._exporting)
        name = self.hist.currentText()
        if name:
            return labels, (d.particles.get(name),)
        modes = self._phonon_modes()
        if modes is not None:
            return labels, (modes,)
        elec = self._electronic()
        if elec is not None:
            return labels, (elec[0],)
        table = self._table()
        if not table:
            return None
        objs = tuple(table.values())
        if _is_neb(table):
            objs += tuple((d.tables.get("NEB spline") or {}).values())
        return labels, objs

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
        if self._elec is not None:
            self._update_controls()

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
        if self._elec is not None:
            text = self._electronic_status()
        elif link is None:
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
        energy = self._elec is not None
        for w in (self.energy_label, self.e_min, self.e_max):
            w.setVisible(energy)
        if energy:
            lo, hi = (self._ax.get_xlim() if self._elec[1] == "DOS" else self._ax.get_ylim()) \
                if self._ax is not None else self._energy_window
            for box, v in ((self.e_min, lo), (self.e_max, hi)):
                box.blockSignals(True)
                box.setValue(v)
                box.blockSignals(False)
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
        return self._xname

    def _keep(self):
        expr = linking.selection_expression(self.link, *self._range) if self._range else None
        if expr:
            self.expressionRequested.emit(expr)

    @_styled
    def _draw_overlays(self):
        for a in self._overlays:
            try:
                a.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        self._overlays = []
        ax, link, d = self._ax, self.link, self.data
        if self._elec is not None and ax is not None and ax.figure is self.figure:
            self._draw_electronic_selection(ax)
            self.canvas.draw_idle()
            return
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
                    ax.tick_params(which="both", right=False)
                    self._ax2 = ax.twinx()
                    self._ax2.tick_params(which="both", colors=pick, labelcolor=pick)
                    self._ax2.set_ylabel("Selected", color=pick)
                    self.figure.tight_layout(pad=0.4)
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
            ax.plot(spline["Reaction Coordinate"], spline["Energy"], color=TOKENS["accent"])
        ok = ~np.isnan(e)
        ax.plot(s[ok], e[ok], "o", ms=5, mew=1.0, color=TOKENS["accent"], mfc=TOKENS["bg"], zorder=3)
        ax.margins(x=0.04, y=0.08)
        ax.set_xlabel("Reaction coordinate (Å)")
        ax.set_ylabel("Energy (eV)")
        a = self.data.attributes
        lines = [f"{label} {a[k]:.3f} eV" for label, k in (("Ea →", "NEB.Barrier"), ("Ea ←", "NEB.Reverse Barrier"),
                                                         ("ΔE", "NEB.Reaction Energy")) if k in a]
        if lines:
            ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize="small",
                    color=TOKENS["secondary"])
        self._marker = ax.plot([], [], "o", ms=9, color=TOKENS["accent"])[0]
        self._place_marker()

    # navigation: zoom, pan and reset, the same on every plot
    def _energy_axis(self):
        """"x" on a DOS, "y" on a band structure (the axis of the shared energy window), else None."""
        if self._elec is not None:
            return "x" if self._elec[1] == "DOS" else "y"
        which = self._electronic()
        return None if which is None else "x" if which[1] == "DOS" else "y"

    def set_view(self, xlim=None, ylim=None, emit: bool = True):
        """Show the range ``xlim`` / ``ylim`` (None keeps that axis as it is). The range stays
        while frames change, until :meth:`reset_view` or another table; on a DOS or band plot
        the energy axis sets the energy window that both share."""
        energy = self._energy_axis()
        redraw = False
        for axis, lim in (("x", xlim), ("y", ylim)):
            if lim is None:
                continue
            lo, hi = sorted(float(v) for v in lim)
            if not hi > lo:
                continue
            if axis == energy:
                redraw |= (lo, hi) != self._energy_window
                self._energy_window = (lo, hi)
            else:
                self._view[axis] = (lo, hi)
        if redraw or self._ax is None:
            self._drawn = None
            self.redraw()
        else:
            for axis, lim in self._view.items():
                (self._ax.set_xlim if axis == "x" else self._ax.set_ylim)(lim)
            self.canvas.draw_idle()
            self._update_controls()
        if emit:
            self.viewChanged.emit(*self.view_limits())

    def reset_view(self, emit: bool = True):
        """The full range again (and the default energy window on DOS and bands)."""
        self._view = {}
        self._energy_window = tuple(ENERGY_WINDOW)
        self._drawn = None
        self.redraw()
        if emit:
            self.viewChanged.emit(None, None)

    def view_limits(self):
        """``(xlim, ylim)`` the user chose (the energy window counts when it is not the
        default), None for an axis left at its full range."""
        lims = dict(self._view)
        energy = self._energy_axis()
        if energy is not None and self._energy_window != tuple(ENERGY_WINDOW):
            lims[energy] = self._energy_window
        return lims.get("x"), lims.get("y")

    def _energy_typed(self, *_):
        lo, hi = self.e_min.value(), self.e_max.value()
        if hi > lo and self._energy_axis() is not None:
            energy = self._energy_axis()
            self.set_view(**{f"{energy}lim": (lo, hi)})

    def _on_scroll(self, event):
        ax = self._ax
        if ax is None or event.inaxes is None or event.x is None:
            return
        key = event.key or ""
        axis = "y" if "control" in key or "ctrl" in key else "x"
        if self._energy_axis() == "y":   # band structure: the wheel zooms the energy
            axis = "x" if axis == "y" else "y"
        x, y = ax.transData.inverted().transform((event.x, event.y))
        c = x if axis == "x" else y
        lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
        f = ZOOM_STEP ** -float(event.step)   # wheel up zooms in
        self.set_view(**{f"{axis}lim": (c - (c - lo) * f, c + (hi - c) * f)})

    def _pan_to(self, event):
        ax = self._ax
        px, py, (x0, x1), (y0, y1) = self._pan
        bb = ax.bbox
        dx = (event.x - px) * (x1 - x0) / max(bb.width, 1.0)
        dy = (event.y - py) * (y1 - y0) / max(bb.height, 1.0)
        if ax.get_xscale() == "linear":
            ax.set_xlim(x0 - dx, x1 - dx)
        if ax.get_yscale() == "linear":
            ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()

    # hover: the values under the cursor and the frame a click goes to
    def _readout(self, event):
        ax = self._ax
        if ax is None or event.inaxes is None or event.x is None or self.data is None:
            self.readout.setText("")
            return
        x, y = ax.transData.inverted().transform((event.x, event.y))
        text = f"{_num(x)}, {_num(y)}"
        energy = self._energy_axis()
        if energy is not None:
            text = f"E − E_F = {_num(x if energy == 'x' else y)} eV"
        link = self.link
        try:
            if link is not None and link.kind == "frames":
                f = linking.frame_at(self.pipeline, self.data, link, x)
                if f is not None:
                    text += f"   {'image' if _is_neb(self._table()) else 'frame'} {f}" + self._frame_value(f)
            elif link is not None and link.kind == "frames2d":
                f = self._frame_near(event)
                if f is not None:
                    text += f"   frame {f}"
        except Exception:   # a table that does not map to frames on this pipeline
            pass
        self.readout.setText(text)

    def _frame_value(self, frame):
        """``": <column> <value>"`` of the one plotted column at ``frame``, or ""."""
        cols = self._plotted_columns() or []
        if len(cols) != 1:
            return ""
        rows = linking.frame_rows(self.pipeline, self.data, self.link.name)
        if not 0 <= frame < len(rows) or rows[frame] < 0:
            return ""
        v = float(np.asarray(self._table()[cols[0]], float)[rows[frame]])
        return "" if np.isnan(v) else f": {_num(v)}"

    def _frame_near(self, event, pixels: float = CLICK_RADIUS):
        """The frame whose point on a two-CV plot is within ``pixels`` of the cursor, or None."""
        ax = self._ax
        frames, xs, ys = linking.frame_xy(self.pipeline, self.data, self.link)
        if not len(frames):
            return None
        pts = ax.transData.transform(np.column_stack([xs, ys]))
        d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        k = int(np.argmin(d))
        return int(frames[k]) if d[k] <= pixels else None

    # mouse: left drag selects (or zooms on a plot that selects nothing), click goes to a frame
    def _on_press(self, event):
        self._press = self._pan = None
        if self._ax is None or event.inaxes is None or event.inaxes not in (self._ax, self._ax2):
            return
        if event.button in (2, 3):
            self._pan = (event.x, event.y, self._ax.get_xlim(), self._ax.get_ylim())
            return
        if event.button != 1:
            return
        if event.dblclick:
            self.reset_view()
            return
        x, y = self._ax.transData.inverted().transform((event.x, event.y))
        shift = "shift" in (event.key or "")
        self._press = (x, y, event.x, event.y, shift)

    def _zoom_axis(self):
        """The axis a left drag zooms on a plot that selects nothing."""
        return "y" if self._energy_axis() == "y" else "x"

    def _on_motion(self, event):
        if self._pan is not None and event.x is not None:
            self._pan_to(event)
            return
        if self._press is None or event.x is None:
            self._readout(event)
            return
        x0, y0, *_, shift = self._press
        x, y = self._ax.transData.inverted().transform((event.x, event.y))
        if self._rubber is not None:
            try:
                self._rubber.remove()
            except (ValueError, NotImplementedError):
                pass
        link = self.link
        box = link is not None and (link.kind == "frames2d" or (shift and link.kind == "frames"))
        if box:
            from matplotlib.patches import Rectangle

            self._rubber = self._ax.add_patch(Rectangle((min(x0, x), min(y0, y)), abs(x - x0), abs(y - y0),
                                                        fill=False, ec=TOKENS["pick"], lw=1))
        elif link is None and self._zoom_axis() == "y":
            self._rubber = self._ax.axhspan(min(y0, y), max(y0, y), color=TOKENS["muted"], alpha=0.2, lw=0)
        else:
            color = TOKENS["pick"] if link is not None else TOKENS["muted"]
            self._rubber = self._ax.axvspan(min(x0, x), max(x0, x), color=color, alpha=0.25, lw=0)
        self.canvas.draw_idle()

    def context_menu(self) -> QtWidgets.QMenu:
        """Right click on the plot: Save plot…, Reset view and the "Plot" entries of
        ``@menu_item`` plugins (``context.target`` is :meth:`target`)."""
        menu = QtWidgets.QMenu(self)
        menu.addAction("Save plot…", self.save_plot)
        menu.addAction("Reset view", self.reset_view)
        for label, fn in (self.extra_menu() if self.extra_menu else []):
            menu.addAction(label, fn)
        menu.addSeparator()
        from .window_plugins import plugin_host

        host = plugin_host(self)
        if host is not None and host.add_plugin_items(menu, "Plot", target=self.target()):
            menu.removeAction(menu.actions()[-1])    # no separator at the end
        return menu

    def target(self):
        """This plot for plugins (:class:`ewald.plugins.PlotTarget`)."""
        from ..plugins import PlotTarget

        table = self._table()
        ax = self._ax
        return PlotTarget(table=self.hist.currentText() or self.table.currentText(), values=table,
                          columns=list(self._plotted_columns() or []),
                          xlim=tuple(ax.get_xlim()) if ax is not None else None,
                          ylim=tuple(ax.get_ylim()) if ax is not None else None, data=self.data)

    def _on_release(self, event):
        press, self._press = self._press, None
        pan = self._pan
        if event.button == 3 and (pan is None or event.x is None or abs(event.x - pan[0]) + abs(event.y - pan[1]) < 4):
            self._pan = None      # a right click, not a pan
            menu = self.context_menu()
            menu.exec(QtGui.QCursor.pos())
            menu.deleteLater()
            return
        if self._pan is not None:
            pan, self._pan = self._pan, None
            if event.x is not None and abs(event.x - pan[0]) + abs(event.y - pan[1]) >= 4:
                self.set_view(self._ax.get_xlim(), self._ax.get_ylim())
            return
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
        link = self.link
        if abs(event.x - px) + abs(event.y - py) < 4:  # a click
            self._clicked(event)   # a phonon band: play that mode
            f = None
            if link is not None and link.kind == "frames":
                f = linking.frame_at(self.pipeline, self.data, link, x)
            elif link is not None and link.kind == "frames2d":
                f = self._frame_near(event)
            if f is not None:
                self.frameRequested.emit(f)
            self.canvas.draw_idle()
            return
        if link is None:   # nothing to select: zoom to the dragged range
            if self._zoom_axis() == "y":
                self.set_view(ylim=(y0, y))
            else:
                self.set_view(xlim=(x0, x))
            return
        box = link.kind == "frames2d" or (shift and link.kind == "frames")
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
            ax.scatter(x, f[0], marker="_", s=120, color=TOKENS["accent"], linewidths=1.5)
            ax.set_xlabel("Mode")
            ax.margins(x=0.03)
            ax.xaxis.set_major_locator(_integer_locator())
            plotstyle.no_minor_x(ax)
        else:
            x = modes.distances if modes.is_path else np.arange(modes.num_qpoints, dtype=float)
            for b in range(modes.num_bands):
                ax.plot(x, f[:, b], lw=1.0, color=TOKENS["accent"])
            ax.set_xlim(x[0], x[-1])
            if modes.ticks:
                for d, _ in modes.ticks:
                    ax.axvline(d, color=TOKENS["border"], lw=0.6)
                ax.set_xticks([d for d, _ in modes.ticks])
                ax.set_xticklabels([_kpoint_label(lab) for _, lab in modes.ticks])
                plotstyle.no_minor_x(ax)
            else:
                ax.set_xlabel("Q-point" if not modes.is_path else "Distance")
        ax.axhline(0, color=TOKENS["border"], lw=0.6)
        ax.set_ylabel("Frequency (THz)")
        if not self._exporting:
            ax.set_title("Click a band to play the mode", color=TOKENS["muted"], fontsize="small")
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

    # electronic structure: DOS and bands, the viewport selection as PDOS / fat bands
    def _electronic(self):
        """``(ElectronicStructure, table name)`` when the selected table is its DOS or band table."""
        d = self.data
        name = self.table.currentText()
        if d is None or self.hist.currentText() or name not in ("DOS", "Bands") or name not in d.tables:
            return None
        es = d.attributes.get("Electronic Structure")
        return (es, name) if es is not None and hasattr(es, "dos_table") else None

    def _draw_electronic(self, ax, which):
        es, name = which
        col = self.column.currentText()
        table = self.data.tables[name]
        lo, hi = self._energy_window
        clamp = self._energy_window == tuple(ENERGY_WINDOW)  # the default window stops at the data
        if name == "DOS":
            e = np.asarray(table["Energy"], float)
            totals = [k for k in table if k.startswith("Total")]
            for k in totals:
                ax.fill_between(e, np.asarray(table[k], float), color=TOKENS["secondary"], alpha=0.18, lw=0,
                                label="Total" if k == totals[0] else None)
            if col == "(all)":
                shown = [k for k in table if k not in totals and k != "Energy" and " " not in _spinless(k)]
            else:
                shown = [k for k in table if _spinless(k) == _spinless(col) and k not in totals]
            colors = {}
            for k in shown:
                base = _spinless(k)
                cycle = plotstyle.colors(TOKENS)   # hex now: "C0" would resolve outside the style
                c = colors.setdefault(base, cycle[len(colors) % len(cycle)])
                ax.plot(e, np.asarray(table[k], float), lw=1.1, color=c, label=base if base not in
                        [t.get_label() for t in ax.lines] else None)
            ax.axvline(0, color=TOKENS["secondary"], lw=0.6, ls="--")
            if es.num_spins == 2:
                ax.axhline(0, color=TOKENS["border"], lw=0.6)
            ax.set_xlim((max(lo, e.min()), min(hi, e.max())) if clamp else (lo, hi))
            ax.set_xlabel(E_FERMI)
            ax.set_ylabel("DOS (states/eV)")
            inside = (e >= lo) & (e <= hi)
            vals = np.concatenate([np.abs(np.asarray(table[k], float)[inside]) for k in totals + shown] or [[1.0]])
            top = 1.08 * (vals.max() if vals.size and vals.max() > 0 else 1.0)
            ax.set_ylim(-top if es.num_spins == 2 else 0, top)
        else:
            x = np.asarray(table[next(iter(table))], float)
            cut = np.asarray(es.breaks, int)
            for k in list(table)[1:]:
                y = np.insert(np.asarray(table[k], float), cut, np.nan)
                xs = np.insert(x, cut, np.nan)
                pick = col not in ("(all)", k)
                ax.plot(xs, y, lw=0.7 if pick else 1.0, ls="--" if k.endswith("(down)") else "-",
                        color=TOKENS["border"] if pick else TOKENS["accent"], zorder=2 if pick else 3)
            ax.set_xlim(x[0], x[-1])
            for d, _ in es.ticks:
                ax.axvline(d, color=TOKENS["border"], lw=0.6)
            if es.ticks:
                ax.set_xticks([d for d, _ in es.ticks])
                ax.set_xticklabels([_kpoint_label(lab) for _, lab in es.ticks])
                plotstyle.no_minor_x(ax)
            else:
                ax.set_xlabel(next(iter(table)))
            ax.axhline(0, color=TOKENS["secondary"], lw=0.6, ls="--")
            e = es.eigenvalues - es.efermi
            ax.set_ylim((max(lo, e.min() - 0.2), min(hi, e.max() + 0.2)) if clamp else (lo, hi))
            ax.set_ylabel(E_FERMI)
        gap = self.data.attributes.get("Band Gap")
        if gap is not None:
            ax.text(0.97, 0.95, f"$E_\\mathrm{{g}}$ = {gap:.3f} eV", transform=ax.transAxes, ha="right", va="top",
                    fontsize="small", color=TOKENS["secondary"])
        if name == "DOS":
            ax.legend(loc="upper left")
        self._elec = (es, name, col)

    def _selected_atoms(self):
        es = self._elec[0]
        h = np.asarray(self.highlight, dtype=int)
        return h[(h >= 0) & (h < len(es.types))]

    def _column_orbital(self):
        """The l of a column like "Ga d (up)", or None."""
        parts = _spinless(self._elec[2]).split()
        return parts[1] if len(parts) == 2 and parts[1] in "spdf" else None

    def _draw_electronic_selection(self, ax):
        es, name, _ = self._elec
        atoms = self._selected_atoms()
        if not len(atoms):
            return
        pick, orb = TOKENS["pick"], self._column_orbital()
        try:
            if name == "DOS" and es.pdos is not None:
                e = es.dos_energies - es.efermi
                p = es.pdos_of(atoms, orb)
                for s in range(es.num_spins):
                    y = -p[s] if s == 1 else p[s]
                    self._overlays.append(ax.fill_between(e, y, color=pick, alpha=0.35, lw=0))
                    self._overlays.extend(ax.plot(e, y, color=pick, lw=1.1))
            elif name == "Bands" and es.has_projections:
                w = es.band_weights(atoms, orb)
                x = es.distances if es.distances is not None else np.arange(w.shape[1], dtype=float)
                e = es.eigenvalues - es.efermi
                lo, hi = ax.get_ylim()
                for s in range(es.num_spins):
                    xs = np.repeat(x, w.shape[2])
                    ys, ws = e[s].ravel(), w[s].ravel()
                    ok = (ys >= lo) & (ys <= hi) & (ws > 1e-3)
                    self._overlays.append(ax.scatter(xs[ok], ys[ok], s=60 * ws[ok], color=pick, alpha=0.6,
                                                     lw=0, zorder=4, marker="o" if s == 0 else "s"))
        except (ValueError, KeyError, IndexError) as exc:
            self.link_label.setText(f"Projection failed: {exc}")

    def _electronic_status(self):
        es, name, _ = self._elec
        n = len(self._selected_atoms())
        orb = self._column_orbital()
        what = f"{orb} " if orb else ""
        if name == "DOS":
            if es.pdos is None:
                return "No projected DOS in this run (LORBIT >= 10 writes it)"
            return f"{what}PDOS of {n} selected atom{'s' if n != 1 else ''}" if n else \
                "Select atoms in the view to see their projected DOS"
        if not es.has_projections:
            return "No band projections in this run (LORBIT >= 10 writes them)"
        return f"Fat bands: {what}weight of {n} selected atom{'s' if n != 1 else ''}" if n else \
            "Select atoms in the view to see their weight in each band"

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
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.pdf",
                                                             "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if path:
            from .file_dialogs import suffixed

            path = suffixed(path, chosen)
            from .. import settings as settings_mod

            s = settings_mod.load()
            self.export(path, s["plots.size"], s["plots.font_size"], s["render.quality"])

    def export(self, path, size: str | tuple | None = None, font_size: float | None = None,
               quality: str = "Publication"):
        """Write the plot on the canvas to ``path`` (PDF/SVG/EPS/PNG) as a publication figure:
        redrawn at a journal column width (:data:`ewald.plotstyle.SIZES`) in print colors
        (white background, whatever the UI theme), without the UI hints, with editable text in
        vector files and the DPI of ``quality`` ('Draft', 'Standard', 'Publication') for PNG."""
        from matplotlib.figure import Figure

        w, h = plotstyle.figsize(size)
        fs = font_size or plotstyle.FONT_SIZE * max(1.0, (w / 3.5) ** 0.5)
        screen, theme = self.figure, dict(TOKENS)
        TOKENS.update(plotstyle.PRINT)
        self.figure, self._font_size, self._exporting = Figure(figsize=(w, h)), fs, True
        self._drawn = None
        try:
            self.redraw()
            fig = self.figure
            fig._ewald_style = (plotstyle.PRINT, fs)
            q = quality.lower() if quality.lower() in plotstyle.DPI else "publication"
            plotstyle.save(fig, path, quality=q)
        finally:
            TOKENS.clear()
            TOKENS.update(theme)
            self.figure, self._font_size, self._exporting = screen, SCREEN_FONT, False
            self._drawn = self._ax2 = None
            self.redraw()


E_FERMI = "$E - E_\\mathrm{F}$ (eV)"


def _num(v: float) -> str:
    """A value for the hover readout: 4 significant digits, no exponent for ordinary sizes."""
    return f"{v:.4g}" if v == 0 or 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"


def _kpoint_label(label: str) -> str:
    """High-symmetry point names for tick labels: G/Gamma/\\Gamma as Γ."""
    return "Γ" if label.strip().lstrip("\\").upper() in ("G", "GAMMA", "Γ") else label


def _integer_locator():
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)


def _spinless(column: str) -> str:
    return column.replace(" (up)", "").replace(" (down)", "")


def _is_neb(table) -> bool:
    return bool(table) and next(iter(table)) == "Reaction Coordinate" and "Frame" in table and "Energy" in table


def _same_content(a, b) -> bool:
    """Whether two :meth:`PlotsPanel._content_key` values show the same plot."""
    if a is None or b is None or a[0] != b[0] or len(a[1]) != len(b[1]):
        return False
    for x, y in zip(a[1], b[1]):
        if x is y:
            continue
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) or x.shape != y.shape \
                or x.dtype != y.dtype or x.size > 200_000:
            return False
        if not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
            return False
    return True


def _is_series(table) -> bool:
    return bool(table) and next(iter(table)) in ("Step", "Timestep") and len(table) > 1


def _grid(table):
    return grid_map(table)
