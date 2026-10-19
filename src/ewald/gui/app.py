"""ewald main window: Analyze and Render workspaces (layout per docs/design)."""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from ..io import import_file
from ..render import camera_state, render
from .console import make_console
from .inspector import DataInspector
from .palette import CommandPalette, menu_entries
from .pipeline_panel import MODIFIER_MENU, PipelinePanel
from .plots import PlotsPanel
from . import preferences as prefs
from .render_page import RenderPage
from .style import ViewStyle
from .theme import TOKENS
from .timeline import Timeline
from .viewport import Viewport

FILE_FILTER = ("Simulation files (*.lammpstrj *.dump *.data XDATCAR* POSCAR* CONTCAR* OUTCAR* "
               "vasprun*.xml *.xyz *.extxyz *.cif *.vasp);;"
               "Volumetric data (CHGCAR* CHG* PARCHG* AECCAR* LOCPOT* ELFCAR* *.cube *.cub);;All files (*)")


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
        self.style = ViewStyle()

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
        self._dock(self.panel, "Pipeline", QtCore.Qt.RightDockWidgetArea, 400)

        # --- bottom: console / inspector / plots / log
        self.inspector = DataInspector()
        self.console = make_console(self._console_namespace(), self._after_console, self)
        self.plots = PlotsPanel()
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
        f.addAction("Load &remote file (SSH)…", self.open_remote_dialog, "Ctrl+Shift+O")
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
        for group, items in MODIFIER_MENU.items():
            sub = m.addMenu(group)
            for label, factory in items.items():
                sub.addAction(label, lambda fct=factory: self.panel.add_modifier(fct()))
        t = mb.addMenu("&Tools")
        for key, _icon, tip in Viewport.TOOLS:
            t.addAction(tip.split(":")[0], lambda k=key: self.viewport.set_tool(k))
        w = mb.addMenu("&Window")
        w.addAction("Analyze workspace", lambda: self.set_workspace(0), "Ctrl+1")
        w.addAction("Render workspace", lambda: self.set_workspace(1), "Ctrl+2")
        w.addAction("Python console", lambda: self.bottom.setCurrentWidget(self.console))
        w.addAction("Data inspector", lambda: self.bottom.setCurrentWidget(self.inspector))
        w.addAction("Plots", lambda: self.bottom.setCurrentWidget(self.plots))
        w.addAction("Command palette", self.command_palette)

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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load file", "", FILE_FILTER)
        if path:
            self._open_asking_types(path)

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
        self.viewport.reset_view()
        self.viewport.clear_picks()
        self.panel.set_pipeline(pipeline)
        self.timeline.set_num_frames(pipeline.num_frames)
        self._push_console()
        self.frame = 0
        self.timeline._set(0) if self.timeline.frame != 0 else self.recompute()

    def set_frame(self, frame):
        self.frame = frame
        self.recompute()

    def recompute(self):
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
        prefs.apply_element_defaults(self.style, self.data.types, self.settings)
        self.panel.set_data(self.data)
        ts = self.data.attributes.get("Timestep", self.frame)
        self.viewport.overlay_text = f"Frame {self.frame}   timestep {ts}"
        self.viewport.set_data(self.data)
        self.inspector.set_data(self.data)
        self.plots.set_data(self.data)
        self._push_console()
        dev = (self.device or "cpu").upper()
        self.status.setText(f"N = {self.data.count}   frame {self.frame}/{self.pipeline.num_frames - 1}   "
                            f"pipeline {dt * 1000:.0f} ms   {dev}")
        if self.workspaces.currentIndex() == 1:
            self.render_page.update_code()

    def _style_changed(self):
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
        if text:
            self.log.appendPlainText(time.strftime("%H:%M:%S ") + text)

    # ---------------------------------------------------------------- output
    def render_image(self, filename=None, **overrides):
        if self.data is None:
            return None
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Render image", "render.png",
                                                                "Images (*.png *.jpg *.tif)")
            if not filename:
                return None
        kw = dict(self.style.scene_kwargs())
        kw.update(self.render_page.render_kwargs())
        kw.update(overrides)
        text = self.render_page.label_text(self.frame, self.data)
        render(self.data, filename, camera=camera_state(self.viewport.plotter), text=text or None, **kw)
        self.message(f"Rendered {filename}")
        return filename

    def render_from_page(self):
        page = self.render_page
        out = page.output.text() or "render.png"
        if page.animation.isChecked() or out.endswith((".mp4", ".gif")):
            self.export_movie(out)
            return
        self.render_image(out)
        pix = QtGui.QPixmap(out)
        if not pix.isNull():
            page.preview.setPixmap(pix.scaled(page.preview.size(), QtCore.Qt.KeepAspectRatio,
                                              QtCore.Qt.SmoothTransformation))
        page.update_code()

    def export_dialog(self):
        if self.pipeline is None:
            return
        from ..io import export_file

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data", "export.extxyz",
            "Extended XYZ (*.extxyz);;LAMMPS dump (*.lammpstrj);;VASP POSCAR (POSCAR*);;All (*)")
        if path:
            fmt = "vasp" if os.path.basename(path).startswith("POSCAR") else None
            export_file(self.data, path, format=fmt)
            self.message(f"Exported {path}")

    def export_movie(self, path=None):
        if self.pipeline is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export movie", "movie.mp4", "Movies (*.mp4 *.gif)")
            if not path:
                return
        from ..render import render_animation

        kw = dict(self.style.scene_kwargs())
        kw.update(self.render_page.render_kwargs())
        w, h = kw.pop("size")
        frames = range(0, self.pipeline.num_frames, self.timeline.stride.value())
        render_animation(self.pipeline, path, frames=frames, fps=self.timeline.fps.value(), device=self.device,
                         size=(w // 2 * 2, h // 2 * 2), camera=camera_state(self.viewport.plotter), **kw)
        self.message(f"Wrote {path}")

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
        kw = {"type_map": dict(p.split("=") for p in args.type_map.split(","))} if args.type_map else {}
        win.open_file(args.file, **kw)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
