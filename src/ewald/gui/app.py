"""ewald main window: Analyze and Render workspaces (layout per docs/design)."""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import manual
from .. import modifiers as mods
from ..io import import_file
from ..render import camera_state, render
from .ask_panel import AskPanel
from .console import make_console
from .inspector import DataInspector
from .palette import CommandPalette, menu_entries
from .pipeline_panel import MODIFIER_MENU, PipelinePanel, fill_modifier_menu
from .plots import PlotsPanel
from . import preferences as prefs
from .render_page import RenderPage
from .style import ViewStyle
from .theme import TOKENS
from .timeline import Timeline
from .viewport import Viewport

def file_filter() -> str:
    """Open-dialog filter from the reader registry (built-in formats and plugins)."""
    from .. import plugins

    return (f"Simulation files ({' '.join(plugins.file_patterns('structure'))});;"
            f"Volumetric data ({' '.join(plugins.file_patterns('volumetric'))});;"
            "Sessions (*.ewald);;All files (*)")


SESSION_FILTER = "ewald session (*.ewald)"


def save_filters() -> list[tuple[str, str | None]]:
    """Save-dialog filters from the writer registry: (filter text, writer name or None)."""
    from .. import plugins

    out = []
    for plugin in sorted(plugins.writers().values(), key=lambda p: (p.priority is None, p.priority or 0)):
        if plugin.patterns:
            out.append((f"{plugin.description} ({' '.join(plugin.patterns)})", plugin.name))
    return out + [("All files, format from the name (*)", None)]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, pipeline=None, device=None, settings=None):
        super().__init__()
        from .. import settings as settings_mod

        self.settings = settings if settings is not None else settings_mod.load()
        self.setWindowTitle("ewald")
        self.resize(1600, 1000)
        self.pipeline = None
        self.data = None
        self.device = device
        self.frame = 0
        self._frame_pending = False
        self.style = ViewStyle()
        self.session_path = None

        # --- central: workspaces (Analyze = viewport + timeline, Render = render page)
        self.viewport = Viewport(self.style)
        self.viewport.picked.connect(self._on_pick)
        self.viewport.bondEdit.connect(self.toggle_bond)
        self.viewport.styleChanged.connect(self._style_changed)
        self.timeline = Timeline()
        self.timeline.frameChanged.connect(self.set_frame)
        analyze = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(analyze)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(0)
        al.addWidget(self.viewport, 1)
        al.addWidget(self.timeline)
        self.render_page = RenderPage(self)
        self.render_page.renderRequested.connect(self.render_from_page)
        self.workspaces = QtWidgets.QStackedWidget()
        self.workspaces.addWidget(analyze)
        self.workspaces.addWidget(self.render_page)
        self.setCentralWidget(self.workspaces)

        # --- right: command panel
        self.panel = PipelinePanel(self.style)
        self.panel.changed.connect(self.recompute)
        self.panel.styleChanged.connect(self._style_changed)
        pipeline_dock = self._dock(self.panel, "Pipeline", QtCore.Qt.RightDockWidgetArea, 400)
        self.ask_panel = AskPanel(self._ask_context)
        self.ask_panel.pipelineChanged.connect(self._after_claude)
        self.ask_dock = self._dock(self.ask_panel, "Ask Claude", QtCore.Qt.RightDockWidgetArea, None)
        self.tabifyDockWidget(pipeline_dock, self.ask_dock)
        pipeline_dock.raise_()

        # --- bottom: console / inspector / plots / log
        self.inspector = DataInspector()
        self.console = make_console(self._console_namespace(), self._after_console, self)
        self.plots = PlotsPanel()
        self.plots.frameRequested.connect(lambda f: self.timeline._set(f))
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(self.console, "Python console")
        self.bottom.addTab(self.inspector, "Data inspector")
        self.bottom.addTab(self.plots, "Plots")
        self.bottom.addTab(self.log, "Log")
        self.bottom_dock = self._dock(self.bottom, "Data", QtCore.Qt.BottomDockWidgetArea, None)
        self.resizeDocks([self.bottom_dock], [280], QtCore.Qt.Vertical)

        self._build_menus()
        self._build_toolbar()
        self.status = QtWidgets.QLabel()
        self.pick_label = QtWidgets.QLabel()
        self.pick_label.setStyleSheet(f"color: {TOKENS['pick']};")
        self.statusBar().addWidget(self.pick_label, 1)
        self.statusBar().addPermanentWidget(self.status)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self.viewport, activated=self.timeline.play.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), self, activated=self.command_palette)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Right"), self, activated=self.timeline.next.click)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Left"), self, activated=self.timeline.prev.click)
        prefs.apply_settings(self, self.settings, startup=True)
        prefs.restore_layout(self, self.settings)
        if self.settings["general.startup_workspace"] == "Render":
            self.set_workspace(1)
        prefs.run_startup_script(self, self.settings)
        from .remote_dialog import install_prompts

        install_prompts(self)  # SSH password / host-key dialogs for remote files
        if pipeline is not None:
            self.set_pipeline(pipeline)

    # ---------------------------------------------------------------- chrome
    def _dock(self, widget, title, area, width):
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        if width:
            self.resizeDocks([d], [width], QtCore.Qt.Horizontal)
        return d

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("main-toolbar")
        tb.setMovable(False)
        tb.addAction("Open", self.open_dialog)
        tb.addAction("Open remote", self.open_remote_dialog)
        tb.addSeparator()
        self.ws_buttons = []
        for i, name in enumerate(("Analyze", "Render")):
            b = QtWidgets.QToolButton(text=name, checkable=True)
            b.clicked.connect(lambda _=False, i=i: self.set_workspace(i))
            tb.addWidget(b)
            self.ws_buttons.append(b)
        self.ws_buttons[0].setChecked(True)
        tb.addSeparator()
        search = QtWidgets.QPushButton("  Search commands, modifiers…      Ctrl K")
        search.setStyleSheet(f"text-align: left; color: {TOKENS['muted']}; background: {TOKENS['panel']}; min-width: 320px;")
        search.clicked.connect(self.command_palette)
        tb.addWidget(search)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(QtWidgets.QLabel("Compute ", objectName="muted"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["CPU", "CUDA", "auto"])
        self.device_combo.setCurrentText({"cuda": "CUDA", "auto": "auto"}.get((self.device or "cpu").lower(), "CPU"))
        self.device_combo.currentTextChanged.connect(self._set_device)
        tb.addWidget(self.device_combo)
        render_btn = QtWidgets.QPushButton("Render")
        render_btn.setObjectName("primary")
        render_btn.clicked.connect(lambda: self.set_workspace(1))
        tb.addWidget(render_btn)

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction("&Load file…", self.open_dialog, QtGui.QKeySequence.Open)
        f.addAction("Load &remote file (SSH)…", self.open_remote_dialog, "Ctrl+Alt+O")
        f.addAction("Load &NEB folder…", self.open_neb_dialog)
        f.addAction("Load &session…", self.load_session_dialog, "Ctrl+Shift+O")
        f.addAction("&Save session", self.save_session, QtGui.QKeySequence.Save)
        f.addAction("Save session &as…", lambda: self.save_session(ask=True), QtGui.QKeySequence.SaveAs)
        f.addSeparator()
        f.addAction("Render &image…", self.render_image, "Ctrl+Shift+R")
        f.addAction("E&xport data…", self.export_dialog, "Ctrl+E")
        f.addAction("Export &movie…", self.export_movie)
        f.addSeparator()
        f.addAction("&Quit", self.close, QtGui.QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        act = e.addAction("&Preferences…", self.preferences, QtGui.QKeySequence.Preferences)
        act.setMenuRole(QtGui.QAction.PreferencesRole)  # app menu on macOS, like OVITO/VESTA
        if act.shortcut().isEmpty():
            act.setShortcut("Ctrl+,")
        v = mb.addMenu("&View")
        for ax in ("a", "b", "c", "a*", "b*", "c*"):
            v.addAction(f"View along {ax}", lambda ax=ax: self.viewport.view_along(ax))
        v.addSeparator()
        for label, d in (("Top (+z)", "z"), ("Front (-y)", "front"), ("Right (+x)", "x"), ("Isometric", "iso")):
            v.addAction(label, lambda d=d: self.viewport.view_direction(d))
        v.addAction("Zoom to fit", self.viewport.zoom_fit, "Ctrl+0")
        v.addSeparator()
        for bg in ("Dark background", "White background"):
            v.addAction(bg, lambda bg=bg: self._set_background(TOKENS["bg"] if bg.startswith("Dark") else "white"))
        m = mb.addMenu("&Pipeline")
        m.aboutToShow.connect(lambda: fill_modifier_menu(m, self.panel.add_modifier))
        fill_modifier_menu(m, self.panel.add_modifier)
        t = mb.addMenu("&Tools")
        for key, _icon, tip in Viewport.TOOLS:
            t.addAction(tip.split(":")[0], lambda k=key: self.viewport.set_tool(k))
        t.addSeparator()
        t.addAction("Ask Claude about selection…", self.ask_claude, "Ctrl+J")
        w = mb.addMenu("&Window")
        w.addAction("Analyze workspace", lambda: self.set_workspace(0), "Ctrl+1")
        w.addAction("Render workspace", lambda: self.set_workspace(1), "Ctrl+2")
        w.addAction("Python console", lambda: self.bottom.setCurrentWidget(self.console))
        w.addAction("Data inspector", lambda: self.bottom.setCurrentWidget(self.inspector))
        w.addAction("Plots", lambda: self.bottom.setCurrentWidget(self.plots))
        w.addAction("Ask Claude", self.ask_claude)
        w.addAction("Command palette", self.command_palette)
        h = mb.addMenu("&Help")
        for label, page in manual.PAGES.items():
            h.addAction(label, lambda page=page: self.open_help(page))
        h.addSeparator()
        h.addAction("Open plugin folder", self.open_plugin_folder)
        about = h.addAction("About ewald", self.about)
        about.setMenuRole(QtGui.QAction.AboutRole)

    def help_url(self, page="index.md"):
        return manual.url(page, self.settings["general.docs_url"] or None)

    def open_help(self, page="index.md"):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.help_url(page)))

    def open_plugin_folder(self):
        from ..settings import config_dir

        folder = config_dir() / "plugins"
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
        self.message(f"Plugin folder: {folder}")

    def about(self):
        from .. import __version__

        QtWidgets.QMessageBox.about(
            self, "About ewald",
            f"<b>ewald {__version__}</b><p>Atomistic structure and trajectory visualization and analysis "
            f"with an OVITO-style Python pipeline.</p><p><a href='{self.help_url()}'>Manual</a></p>")

    def set_workspace(self, index):
        self.workspaces.setCurrentIndex(index)
        for i, b in enumerate(self.ws_buttons):
            b.setChecked(i == index)
        if index == 1:
            self.render_page.update_code()

    def preferences(self):
        dlg = prefs.PreferencesDialog(self.settings, self)
        dlg.applied.connect(lambda s: prefs.apply_settings(self, s))
        dlg.exec()
        return dlg

    def command_palette(self):
        entries = [(label, f"Add modification › {group}", lambda f=factory: self.panel.add_modifier(f()))
                   for group, items in MODIFIER_MENU.items() for label, factory in items.items()]
        entries += menu_entries(self.menuBar())
        pal = CommandPalette(entries, self)
        geo = self.geometry()
        pal.move(geo.x() + (geo.width() - 560) // 2, geo.y() + 90)
        pal.edit.setFocus()
        pal.exec()

    def _set_device(self, text):
        dev = {"CPU": "cpu", "CUDA": "cuda", "auto": "auto"}[text]
        if dev == "cuda":
            from ..backend import get_backend

            try:
                get_backend("cuda")
            except RuntimeError as exc:
                self.message(str(exc), error=True)
                self.device_combo.blockSignals(True)
                self.device_combo.setCurrentText("CPU")
                self.device_combo.blockSignals(False)
                return
        self.device = dev
        self.recompute()

    def _set_background(self, color):
        self.style.background = color
        self._style_changed()

    def message(self, text, error=False):
        self.statusBar().showMessage(text, 10000)
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + ("ERROR " if error else "") + text)

    # ---------------------------------------------------------------- pipeline / data
    def open_file(self, path, **kwargs):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pipe = import_file(path, **kwargs)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.message(f"Load failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Load failed", f"{path}\n\n{exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()
        self.set_pipeline(pipe)
        self.setWindowTitle(f"ewald — {os.path.basename(str(path))}")
        self.message(f"Loaded {path}: {pipe.num_frames} frame(s)")
        return pipe

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load file", "", file_filter())
        if not path:
            return
        from ..session import is_session_file

        if is_session_file(path):
            self.load_session(path)
            return
        self._open_asking_types(path)

    def open_neb_dialog(self):
        """A VASP NEB folder (00 .. NN) or a LAMMPS NEB run folder, one frame per image."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Load NEB folder")
        if path:
            self.open_file(path)
            self.bottom.setCurrentWidget(self.plots)

    def open_remote_dialog(self):
        from .remote_dialog import RemoteFileDialog

        dlg = RemoteFileDialog(self)
        if dlg.exec() and dlg.url:
            self._open_asking_types(dlg.url)
        return dlg

    def _open_asking_types(self, path):
        kwargs = {}
        from ..io.lammps_dump import is_lammps_dump

        if is_lammps_dump(path):
            default_map = self.settings["general.lammps_type_map"]
            if self.settings["general.lammps_type_prompt"] == "ask":
                text, ok = QtWidgets.QInputDialog.getText(
                    self, "LAMMPS types", "Optional type → element map (e.g. 1=Cu,2=Zr):", text=default_map)
            else:
                text, ok = default_map, True
            if ok and text.strip():
                kwargs["type_map"] = dict(p.split("=") for p in text.replace(" ", "").split(","))
        self.open_file(path, **kwargs)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        pipeline.cache_memory = self.settings["performance.cache_memory"] * 2**20
        self.viewport.reset_view()
        self.viewport.clear_picks()
        self.panel.set_pipeline(pipeline)
        self.timeline.set_num_frames(pipeline.num_frames)
        self._push_console()
        self.frame = 0
        self.timeline._set(0) if self.timeline.frame != 0 else self.recompute()

    def set_frame(self, frame):
        self.frame = frame
        tl = self.timeline
        if tl.slider.isSliderDown() or tl.play.isChecked():
            # Scrubbing/playback: frames arriving while one is computed collapse into the
            # latest, which is drawn coarse and refined when the timeline comes to rest.
            if not self._frame_pending:
                self._frame_pending = True
                QtCore.QTimer.singleShot(0, self._apply_pending_frame)
        else:
            self.recompute()

    def _apply_pending_frame(self):
        self._frame_pending = False
        self.recompute(interactive=True)

    def recompute(self, interactive: bool = False):
        if self.pipeline is None:
            return
        t = time.perf_counter()
        try:
            self.data = self.pipeline.compute(self.frame, device=self.device, errors="skip")
        except Exception as exc:  # source failed (bad file/frame)
            self.message(f"Error: {exc}", error=True)
            traceback.print_exc()
            return
        dt = time.perf_counter() - t
        for m in self.pipeline.modifiers:
            if m.enabled and m.status:
                self.message(f"{m.name}: {m.status}", error=True)
        if self.style.bonds_enabled:
            self.style.ensure_bond_pairs(self.data.types)
        self._apply_element_defaults()
        self.panel.set_data(self.data)
        ts = self.data.attributes.get("Timestep", self.frame)
        self.viewport.overlay_text = f"Frame {self.frame}   timestep {ts}"
        self.viewport.set_data(self.data, interactive=interactive)
        self.inspector.set_data(self.data)
        self.plots.set_data(self.data)
        self._push_console()
        dev = (self.device or "cpu").upper()
        self.status.setText(f"N = {self.data.count}   frame {self.frame}/{self.pipeline.num_frames - 1}   "
                            f"pipeline {dt * 1000:.0f} ms   {dev}")
        if self.workspaces.currentIndex() == 1:
            self.render_page.update_code()

    def _apply_element_defaults(self):
        table = prefs.apply_element_defaults(self.style, self.data, self.settings)
        if table is not None and table.fallback and table.summary() != getattr(self, "_radius_note", None):
            self._radius_note = table.summary()  # say once which types fell back, not on every frame
            self.message(f"{self._radius_note}. Set oxidation states under Particles.")

    def _style_changed(self):
        if self.data is not None:
            self._apply_element_defaults()  # a new radius preset or oxidation states refill the type radii
        self.viewport.style_combo.blockSignals(True)
        self.viewport.style_combo.setCurrentText(self.style.preset)
        self.viewport.style_combo.blockSignals(False)
        self.viewport.redraw()
        if self.panel._current() and self.panel._current()[0] == "visual":
            pass  # editor already reflects the change
        else:
            self.panel.refresh()

    def toggle_bond(self, i, j):
        """Bond tool: delete the i-j bond if it exists, else add it, via an EditBonds modifier on top."""
        from ..modifiers import EditBonds

        if self.pipeline is None or self.data is None:
            return
        d = self.data
        ids = d["Particle Identifier"] if "Particle Identifier" in d else np.arange(d.count)
        bonded = d.bonds is not None and bool(np.any(np.all(np.sort(d.bonds.topology, axis=1) == sorted((i, j)), axis=1)))
        mods = self.pipeline.modifiers
        if not (mods and isinstance(mods[-1], EditBonds)):
            mods.append(EditBonds())
        mods[-1].toggle(int(ids[i]), int(ids[j]), bonded)
        self.panel.refresh(select=("mod", mods[-1]))
        self.recompute()
        self.message(f"{'Deleted' if bonded else 'Added'} bond {d.types[i]}{int(ids[i])}-{d.types[j]}{int(ids[j])}")

    def _on_pick(self, text):
        self.pick_label.setText(text)
        self.ask_panel.update_selection()
        if text:
            self.log.appendPlainText(time.strftime("%H:%M:%S ") + text)

    # ---------------------------------------------------------------- Claude
    def _ask_context(self):
        """(pipeline, frame, indices) for the Ask Claude panel: the viewport picks, else the
        'Selection' property of the current frame."""
        indices = self.viewport.picks
        if not indices and self.data is not None and "Selection" in self.data:
            indices = np.nonzero(self.data.selection)[0].tolist()
        return self.pipeline, self.frame, indices

    def ask_claude(self):
        self.ask_dock.show()
        self.ask_dock.raise_()
        self.ask_panel.update_selection()
        self.ask_panel.input.setFocus()

    def _after_claude(self):
        self.panel.refresh()
        self.recompute()

    # ---------------------------------------------------------------- output
    def render_image(self, filename=None, return_image=False, **overrides):
        if self.data is None:
            return None
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Render image", "render.png", "Images (*.png *.jpg *.tif);;Vector (*.svg *.pdf *.eps)")
            if not filename:
                return None
        kw = dict(self.style.scene_kwargs())
        kw.update(self.render_page.render_kwargs())
        kw.update(overrides)
        text = self.render_page.label_text(self.frame, self.data)
        try:
            img = render(self.data, filename, camera=camera_state(self.viewport.plotter), text=text or None, **kw)
        except Exception as exc:
            self.message(f"Render failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Render failed", str(exc))
            return None
        self.message(f"Rendered {filename}")
        return img if return_image else filename

    def render_from_page(self):
        page = self.render_page
        out = page.output.text() or "render.png"
        if page.is_movie:
            self.export_movie(out)
            return
        img = self.render_image(out, return_image=True)
        if img is not None:
            self.show_preview(img)
        page.update_code()

    def show_preview(self, img):
        page = self.render_page
        img = np.ascontiguousarray(img)
        fmt = QtGui.QImage.Format_RGBA8888 if img.shape[-1] == 4 else QtGui.QImage.Format_RGB888
        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt).copy()
        page.preview.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            page.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def export_dialog(self):
        if self.pipeline is None:
            return
        from .. import plugins
        from ..io import export_file

        filters = save_filters()
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data", "export.extxyz", ";;".join(f for f, _ in filters))
        if not path:
            return
        fmt = dict(filters).get(chosen)
        writer = plugins.find_writer(path, fmt)
        all_frames = False
        n = self.pipeline.num_frames
        if writer.multi_frame and n > 1:
            answer = QtWidgets.QMessageBox.question(
                self, "Export trajectory", f"{writer.description}\n\nExport all {n} frames? "
                "(No exports only the current frame.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                return
            all_frames = answer == QtWidgets.QMessageBox.Yes
        try:
            if all_frames:
                export_file(self.pipeline, path, format=fmt)
            else:
                export_file(self.data, path, format=fmt)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{path}\n\n{exc}")
            return
        self.message(f"Exported {path} ({writer.name})")

    def export_movie(self, path=None):
        if self.pipeline is None:
            return
        if not path:
            from .render_page import MOVIE_FORMATS

            ext = MOVIE_FORMATS.get(self.render_page.movie_format.currentText(), ".mp4")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export movie", "movie" + ext, "Movies (*.mp4 *.gif *.avi *.mov);;PNG sequence (*.png)")
            if not path:
                return
        from ..render import render_animation

        page = self.render_page
        kw = dict(self.style.scene_kwargs())
        kw.update(page.render_kwargs(movie=True))
        w, h = kw.pop("size")
        frames = range(0, self.pipeline.num_frames, self.timeline.stride.value())
        dlg = QtWidgets.QProgressDialog(f"Rendering {os.path.basename(path)}…", "Cancel", 0, len(frames), self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i, n):
            dlg.setValue(i)
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            n = render_animation(self.pipeline, path, frames=frames, fps=self.timeline.fps.value(),
                                 device=self.device, size=(w // 2 * 2, h // 2 * 2),
                                 camera=camera_state(self.viewport.plotter), text=page.text.text() or None,
                                 time_per_frame=page.time_per_frame.value(), progress=progress, **kw)
        except Exception as exc:
            self.message(f"Movie export failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Movie export failed", str(exc))
            return None
        finally:
            dlg.close()
        self.message(f"Wrote {path} ({n} frames)")
        return path

    # ---------------------------------------------------------------- sessions
    def save_session(self, path=None, ask=False):
        """Save pipeline, style, camera and Render settings to an .ewald file."""
        if self.pipeline is None:
            return None
        path = path or (None if ask else self.session_path)
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", "session.ewald", SESSION_FILTER)
            if not path:
                return None
        from ..session import save_session

        try:
            save_session(path, self.pipeline, frame=self.frame, camera=camera_state(self.viewport.plotter),
                         style=self.style, render=self.render_page.state(), device=self.device)
        except (TypeError, OSError) as exc:
            self.message(f"Save failed: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Save session failed", str(exc))
            return None
        self.session_path = path
        self.setWindowTitle(f"ewald — {os.path.basename(path)}")
        self.message(f"Saved session {path}")
        return path

    def load_session_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", "", SESSION_FILTER + ";;All files (*)")
        if path:
            self.load_session(path)

    def load_session(self, path, allow_code=None):
        """Restore a session. Python modifiers run only after the user agrees (or ``allow_code=True``)."""
        import json
        import warnings

        from ..session import CODE_MODIFIERS, load_session

        if allow_code is None:
            try:
                with open(path, encoding="utf-8") as fh:
                    has_code = any(m.get("type") in CODE_MODIFIERS for m in json.load(fh).get("modifiers", []))
            except (OSError, ValueError):
                has_code = False
            allow_code = has_code and QtWidgets.QMessageBox.question(
                self, "Python code in session",
                "This session contains Python modifiers. Run their code?\n\n"
                "Only answer Yes for sessions you trust; otherwise they are loaded disabled.") \
                == QtWidgets.QMessageBox.Yes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = load_session(path, allow_code=bool(allow_code))
        except Exception as exc:
            self.message(f"Session load failed: {path}: {exc}", error=True)
            QtWidgets.QMessageBox.critical(self, "Load session failed", f"{path}\n\n{exc}")
            return None
        for w in s.warnings:
            self.message(w, error=True)
        if s.style:
            self.style._pref_filled = {}
            for k, v in s.style.items():  # in place: viewport, panel and console share this object
                if k in self.style.__dataclass_fields__:
                    setattr(self.style, k, list(map(list, v)) if k == "bond_pairs" else v)
        if s.device:
            self.device = s.device
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentText({"cuda": "CUDA", "auto": "auto"}.get(s.device, "CPU"))
            self.device_combo.blockSignals(False)
        self.set_pipeline(s.pipeline)
        if s.frame:
            self.timeline._set(min(s.frame, s.pipeline.num_frames - 1))
        if s.camera:
            from ..render import apply_camera_state

            apply_camera_state(self.viewport.plotter, s.camera)
            self.viewport.plotter.render()
        self.render_page.set_state(s.render)
        self._style_changed()
        self.session_path = path
        self.setWindowTitle(f"ewald — {os.path.basename(path)}")
        self.message(f"Loaded session {path}")
        return s

    def closeEvent(self, event):
        prefs.save_layout(self, self.settings)
        self.timeline.timer.stop()
        self.viewport.plotter.close()  # release the VTK/OpenGL context before Qt tears down
        km = getattr(self.console, "kernel_manager", None)
        if km is not None:
            self.console.kernel_client.stop_channels()
            km.shutdown_kernel()
        super().closeEvent(event)

    # ---------------------------------------------------------------- console
    def _console_namespace(self):
        return {"np": np, "mods": mods, "viewer": self, "vp": self.viewport, "pipeline": self.pipeline,
                "pipe": self.pipeline, "data": self.data, "import_file": import_file, "style": self.style}

    def _push_console(self):
        shell = getattr(self.console, "shell", None)
        ns = {"pipeline": self.pipeline, "pipe": self.pipeline, "data": self.data}
        if hasattr(shell, "push") and hasattr(shell, "user_ns"):
            shell.push(ns)
        elif shell is not None and hasattr(shell, "locals"):
            shell.locals.update(ns)

    def _after_console(self, *args):
        shell = getattr(self.console, "shell", None)
        ns = getattr(shell, "user_ns", None) or getattr(shell, "locals", {})
        for key in ("pipe", "pipeline"):
            pipe = ns.get(key)
            if pipe is not None and pipe is not self.pipeline:
                self.set_pipeline(pipe)  # user did pipe = import_file(...)
                return
        if self.pipeline is not None:
            self.panel.refresh()
            self.timeline.set_num_frames(self.pipeline.num_frames)
            self.recompute()


def _app():
    from . import theme

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("ewald")
        theme.apply(app)
    return app


def show(pipeline=None, device=None, block: bool = True):
    """Open the viewer on an existing Pipeline (from a script or Jupyter with %gui qt)."""
    app = _app()
    win = MainWindow(pipeline, device=device)
    win.show()
    if block and not _in_ipython_qt_loop():
        app.exec()
    return win


def _in_ipython_qt_loop():
    try:
        ip = get_ipython()  # noqa: F821
        return getattr(ip, "active_eventloop", None) in ("qt", "qt5", "qt6")
    except NameError:
        return False


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="ewald-gui")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--type-map", default=None, help="LAMMPS types, e.g. 1=Cu,2=Zr")
    ap.add_argument("--device", default=None, help="cpu, cuda or auto")
    args = ap.parse_args(argv)
    app = _app()
    win = MainWindow(device=args.device)
    if args.file:
        from ..session import is_session_file

        kw = {"type_map": dict(p.split("=") for p in args.type_map.split(","))} if args.type_map else {}
        win.load_session(args.file) if is_session_file(args.file) else win.open_file(args.file, **kw)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
