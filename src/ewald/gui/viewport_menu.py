"""The viewport's right-click menu and the actions it offers (picks, frame overlay, isosurface, plots)."""
from __future__ import annotations

import numpy as np

from PySide6 import QtGui, QtWidgets

from .. import examples
from .. import modifiers as mods
from .icons import icon
from .plots import _is_neb


class ViewportMenuMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    # ---------------------------------------------------------------- viewport context menu
    def _overlay(self):
        from ..modifiers import OverlayFrames

        mods = self.pipeline.modifiers if self.pipeline is not None else []
        return next((m for m in mods if isinstance(m, OverlayFrames)), None)

    def set_overlay(self, on: bool = True, **params):
        """Show the other frames as see-through copies (``OverlayFrames``, added last when
        missing); ``params`` set its fields, e.g. ``transparency=0.5``."""
        if self.pipeline is None:
            return
        from ..modifiers import OverlayFrames

        mod = self._overlay()
        if mod is None:
            if not on:
                return
            mod = OverlayFrames()
            self.pipeline.modifiers.append(mod)
        mod.enabled = on
        for k, v in params.items():
            setattr(mod, k, v)
        self.panel.refresh(select=("mod", mod))
        self.recompute()

    def viewport_menu(self, everything: bool = False) -> QtWidgets.QMenu:
        """Right click in the viewport. The entries follow what is loaded and picked: picked
        atoms, the kind of data (phonon modes, NEB band, volumetric grid), trajectory
        playback and frame overlay, display toggles and views. ``everything`` lists every
        section regardless (for the feature inventory)."""
        menu = QtWidgets.QMenu(self)
        d, pipe = self.data, self.pipeline
        if d is None and not everything:
            menu.addAction(icon("open"), "Load file…", self.open_dialog)
            menu.addAction(icon("remote"), "Load remote file (SSH)…", self.open_remote_dialog)
            ex = menu.addMenu(icon("example"), "Open example")
            for example in examples.EXAMPLES:
                ex.addAction(example.file, lambda e=example: self.open_example(e.file))
            return menu

        picks = self.viewport.picks
        if picks or everything:
            count = f"{len(picks)} atom{'s' if len(picks) != 1 else ''}" if picks else "N atoms"
            head = menu.addAction(f"{count} picked")
            head.setEnabled(False)
            menu.addAction(icon("center"), "Center view on picked", self.center_on_picks)
            menu.addAction(icon("select"), "Select picked atoms", lambda: self.select_picked())
            menu.addAction(icon("delete"), "Delete picked atoms", lambda: self.select_picked(delete=True))
            menu.addAction(icon("clear"), "Clear picks", self.viewport.clear_picks)
            menu.addSeparator()
        vp = self.viewport
        if everything or len(picks) == {"distance": 2, "angle": 3}.get(vp.tool):
            menu.addAction(icon("distance"), "Keep measurement in the view", vp.keep_measurement)
        if everything or self.style.measurements:
            menu.addAction(icon("clear"), "Clear measurements", self.clear_measurements)

        tables = d.tables if d is not None else {}
        neb = next((name for name, t in tables.items() if _is_neb(t)), None)
        phonon = d is not None and "Phonon Modes" in d.attributes
        grids = list(d.grids) if d is not None else []
        if phonon or everything:
            menu.addAction(icon("chart"), "Phonon band structure", lambda: self.show_plot())
        if neb or everything:
            menu.addAction(icon("chart"), "NEB energy profile", lambda: self.show_plot(neb))
        if grids or everything:
            iso = [m for m in pipe.modifiers if isinstance(m, mods.CreateIsosurface)] if pipe else []
            if iso:
                menu.addAction(icon("isosurface"), "Edit isosurface", lambda: self.panel.refresh(select=("mod", iso[0])))
            else:
                menu.addAction(icon("isosurface"), "Add isosurface", self.add_isosurface)
        if phonon or neb or grids or everything:
            menu.addSeparator()

        if (pipe is not None and pipe.num_frames > 1) or everything:
            playing = self.timeline.play.isChecked()
            play = menu.addAction(icon("pause" if playing else "play"), "Pause" if playing else "Play")
            play.setShortcut("Space")
            play.triggered.connect(self.timeline.play.toggle)
            self._overlay_entries(menu)
            menu.addSeparator()

        show = menu.addMenu(icon("eye"), "Show")
        for label, attr in (("Bonds", "bonds_enabled"), ("Simulation cell", "show_cell")):
            act = show.addAction(label)
            act.setCheckable(True)
            act.setChecked(bool(getattr(self.style, attr)))
            act.toggled.connect(lambda on, attr=attr: self._toggle_style(attr, on))
        if getattr(self, "render_frame", None) is not None:
            act = show.addAction("Render frame")
            act.setToolTip("The part of the view the rendered picture shows")
            act.setCheckable(True)
            act.setChecked(bool(self.settings["viewport.render_frame"]))
            act.toggled.connect(self._toggle_render_frame)
        if (d is not None and d.cell.is_valid) or everything:
            self.fill_replicate_menu(menu.addMenu(icon("layers"), "Replicate cell"))
            views = menu.addMenu("View along")
            for ax in ("a", "b", "c", "a*", "b*", "c*"):
                views.addAction(ax, lambda ax=ax: self.viewport.view_along(ax))
        ortho = menu.addAction("Orthographic")
        ortho.setCheckable(True)
        ortho.setChecked(self.viewport.ortho)
        ortho.toggled.connect(self.viewport.ortho_button.setChecked)
        menu.addAction(icon("fit"), "Zoom to fit", self.viewport.zoom_fit)
        menu.addSeparator()
        menu.addAction(icon("claude"), "Ask AI about selection…" if picks or everything else "Ask AI about this structure…",
                       self.ask_claude)
        return menu

    def _show_viewport_menu(self):
        menu = self.viewport_menu()
        menu.exec(QtGui.QCursor.pos())
        menu.deleteLater()  # a new menu is built for every right click

    def _overlay_entries(self, menu):
        mod = self._overlay()
        on = mod is not None and mod.enabled
        act = menu.addAction(icon("layers"), "Overlay other frames (see-through)")
        act.setCheckable(True)
        act.setChecked(on)
        act.toggled.connect(lambda checked: self.set_overlay(checked))
        sub = menu.addMenu("Overlay transparency")
        sub.setEnabled(on)
        for pct in (50, 70, 85, 95):
            a = sub.addAction(f"{pct} %")
            a.setCheckable(True)
            a.setChecked(on and abs(mod.transparency - pct / 100) < 1e-6)
            a.triggered.connect(lambda _=False, t=pct / 100: self.set_overlay(True, transparency=t))
        moving = menu.addAction("Overlay only atoms that move")
        moving.setCheckable(True)
        moving.setEnabled(on)
        moving.setChecked(on and mod.min_displacement > 0)
        moving.toggled.connect(lambda checked: self.set_overlay(True, min_displacement=0.1 if checked else 0.0))

    # ---------------------------------------------------------------- replicated cell (display only)
    REPLICATE_CHOICES = ((1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 2, 1), (3, 3, 1), (4, 4, 1))

    def set_replicate(self, n=(2, 2, 2), cell: str | None = None):
        """Draw ``n = (na, nb, nc)`` periodic copies of the cell (``view(pipe, replicate=n)``; (1, 1, 1)
        turns it off) and frame them. ``cell`` picks the cell lines: 'super', 'unit' or 'all'.
        Display only: the data, analysis and export keep the original cell."""
        from ..replicate import CELL_MODES, normalize_replicate

        n = normalize_replicate(n)
        self.style.replicate = list(n) if n else []
        if cell is not None:
            if cell not in CELL_MODES:
                raise ValueError(f"cell must be one of {', '.join(CELL_MODES)}")
            self.style.replicate_cell = cell
        self._style_changed()
        self.viewport.zoom_fit()

    def fill_replicate_menu(self, menu: QtWidgets.QMenu):
        """Replicate cell: off, common n x n x m blocks, a custom one, and which cell lines to draw."""
        from ..replicate import CELL_MODES

        menu.clear()
        now = tuple(self.style.replicate or (1, 1, 1))
        group = QtGui.QActionGroup(menu)
        for n in self.REPLICATE_CHOICES:
            act = menu.addAction("Off (1 × 1 × 1)" if n == (1, 1, 1) else " × ".join(map(str, n)))
            act.setCheckable(True)
            act.setChecked(now == n)
            group.addAction(act)
            act.triggered.connect(lambda _=False, n=n: self.set_replicate(n))
        custom = menu.addAction("Custom…", self._custom_replicate)
        custom.setCheckable(True)
        custom.setChecked(now not in self.REPLICATE_CHOICES)
        group.addAction(custom)
        menu.addSeparator()
        lines = menu.addMenu("Cell lines")
        lines.setEnabled(now != (1, 1, 1))
        labels = {"super": "Outline of the block", "unit": "Original cell", "all": "Every cell"}
        cells = QtGui.QActionGroup(lines)
        for mode, tip in CELL_MODES.items():
            act = lines.addAction(labels[mode])
            act.setToolTip(tip)
            act.setCheckable(True)
            act.setChecked(self.style.replicate_cell == mode)
            cells.addAction(act)
            act.triggered.connect(lambda _=False, m=mode: self.set_replicate(now, cell=m))
        menu.setToolTipsVisible(True)
        return menu

    def _custom_replicate(self):
        now = " ".join(map(str, self.style.replicate or (2, 2, 2)))
        text, ok = QtWidgets.QInputDialog.getText(self, "Replicate cell", "Copies along a, b, c (e.g. 3 3 2):", text=now)
        if not ok:
            return
        try:
            n = [int(x) for x in text.replace(",", " ").replace("x", " ").replace("×", " ").split()]
            self.set_replicate(n[0] if len(n) == 1 else n)
        except (ValueError, IndexError) as exc:
            QtWidgets.QMessageBox.warning(self, "Replicate cell", str(exc))

    def _toggle_style(self, attr, on):
        setattr(self.style, attr, on)
        if attr == "bonds_enabled" and on and self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
        self._style_changed()

    def _toggle_render_frame(self, on):
        self.settings["viewport.render_frame"] = bool(on)
        self.settings.save()
        self._render_frame_visibility()

    def clear_measurements(self):
        """Drop the view's ``measurements`` (and the view cell's)."""
        self.style.measurements = []
        self._style_changed()

    def center_on_picks(self):
        """Rotate and zoom about the picked atoms (their mean position)."""
        picks = self.viewport.picks
        if picks and self.data is not None:
            self.viewport.plotter.set_focus(self.data.positions[picks].mean(axis=0))
            self.viewport.render()

    def select_picked(self, delete: bool = False):
        """Turn the viewport picks into an ``ExpressionSelection`` (by particle identifier when
        the data has one, so the same atoms stay selected on every frame), followed by
        ``DeleteSelected`` when ``delete``."""
        picks, d = self.viewport.picks, self.data
        if self.pipeline is None or d is None or not picks:
            return
        if "Particle Identifier" in d.particles:
            ids = sorted(int(i) for i in np.asarray(d["Particle Identifier"])[picks])
            expr = f"np.isin(ParticleIdentifier, {ids})"
        else:
            expr = f"np.isin(ParticleIndex, {sorted(picks)})"
        self.viewport.clear_picks()
        self.pipeline.modifiers.append(mods.ExpressionSelection(expr))
        if delete:
            self.pipeline.modifiers.append(mods.DeleteSelected())
        self.panel.refresh(select=("mod", self.pipeline.modifiers[-1]))
        self.recompute()

    def add_isosurface(self):
        """Add ``CreateIsosurface`` for the first grid property of the current frame."""
        if self.pipeline is None or self.data is None or not self.data.grids:
            return
        name, grid = next(iter(self.data.grids.items()))
        mod = mods.CreateIsosurface(property=next(iter(grid.properties)), grid=name)
        self.pipeline.modifiers.append(mod)
        self.panel.refresh(select=("mod", mod))
        self.recompute()

    def show_plot(self, table: str | None = None):
        """Open the Plots tab, on ``table`` when given (phonon bands show by themselves)."""
        self.workbench.show(self.plots)
        if table is not None:
            self.plots.hist.setCurrentText("")
            self.plots.table.setCurrentText(table)
