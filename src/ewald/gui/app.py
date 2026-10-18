"""ewald main window."""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from ..io import import_file
from ..render import build_scene, camera_state, render, setup_camera, _add_colorbar
from .console import make_console
from .inspector import DataInspector
from .pipeline_panel import PipelinePanel
from .timeline import Timeline

FILE_FILTER = ("Simulation files (*.lammpstrj *.dump *.data XDATCAR* POSCAR* CONTCAR* OUTCAR* "
               "vasprun*.xml *.xyz *.extxyz *.cif *.vasp);;All files (*)")


class RenderSettings(QtWidgets.QWidget):
    """Viewport + image render settings (OVITO's 'Render' tab, simplified)."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtWidgets.QFormLayout(self)
        self.radius_scale = QtWidgets.QDoubleSpinBox(value=1.0, minimum=0.05, maximum=5, singleStep=0.1)
        self.bonds = QtWidgets.QCheckBox("draw bonds")
        self.bond_cutoff = QtWidgets.QDoubleSpinBox(value=2.8, minimum=0.1, maximum=20, singleStep=0.1, suffix=" Å")
        self.bond_radius = QtWidgets.QDoubleSpinBox(value=0.15, minimum=0.01, maximum=2, singleStep=0.05, suffix=" Å")
        self.show_cell = QtWidgets.QCheckBox("show cell", checked=True)
        self.atom_style = QtWidgets.QComboBox()
        self.atom_style.addItems(["auto", "mesh", "impostor"])
        self.background = QtWidgets.QComboBox()
        self.background.addItems(["white", "black", "#2b2b2b"])
        for label, w in (("Atom radius ×", self.radius_scale), ("", self.bonds), ("Bond cutoff", self.bond_cutoff),
                         ("Bond radius", self.bond_radius), ("", self.show_cell), ("Atom style", self.atom_style),
                         ("Background", self.background)):
            f.addRow(label, w)
            sig = getattr(w, "valueChanged", None) or getattr(w, "toggled", None) or w.currentTextChanged
            sig.connect(self.changed)
        f.addRow(QtWidgets.QLabel("<b>Image output</b>"))
        self.width = QtWidgets.QSpinBox(minimum=64, maximum=16384, value=2000)
        self.height = QtWidgets.QSpinBox(minimum=64, maximum=16384, value=1500)
        size = QtWidgets.QHBoxLayout(); size.addWidget(self.width); size.addWidget(QtWidgets.QLabel("×")); size.addWidget(self.height)
        f.addRow("Size (px)", size)
        self.ao = QtWidgets.QCheckBox("ambient occlusion")
        self.transparent = QtWidgets.QCheckBox("transparent background")
        f.addRow("", self.ao)
        f.addRow("", self.transparent)
        self.render_button = QtWidgets.QPushButton("Render active viewport…")
        f.addRow(self.render_button)

    def scene_kwargs(self):
        return {"radius_scale": self.radius_scale.value(), "show_cell": self.show_cell.isChecked(),
                "bonds": self.bond_cutoff.value() if self.bonds.isChecked() else None,
                "bond_radius": self.bond_radius.value(), "atom_style": self.atom_style.currentText()}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, pipeline=None, device=None):
        super().__init__()
        from pyvistaqt import QtInteractor

        self.setWindowTitle("ewald")
        self.resize(1500, 950)
        self.pipeline = None
        self.data = None
        self.device = device
        self.frame = 0
        self._first_view = True

        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.viewport = QtInteractor(central)
        cl.addWidget(self.viewport.interactor, 1)
        self.setCentralWidget(central)
        self.viewport.set_background("white")
        self.viewport.enable_parallel_projection()

        self.panel = PipelinePanel()
        self.panel.changed.connect(self.recompute)
        self.render_settings = RenderSettings()
        self.render_settings.changed.connect(self.redraw)
        self.render_settings.render_button.clicked.connect(self.render_image)
        right = QtWidgets.QTabWidget()
        right.addTab(self.panel, "Pipeline")
        right.addTab(self.render_settings, "Render")
        self._dock(right, "Pipeline", QtCore.Qt.RightDockWidgetArea, 380)

        self.timeline = Timeline()
        self.timeline.frameChanged.connect(self.set_frame)
        cl.addWidget(self.timeline)  # directly under the viewport, as in OVITO

        self.inspector = DataInspector()
        self.console = make_console(self._console_namespace(), self._after_console, self)
        bottom = QtWidgets.QTabWidget()
        bottom.addTab(self.inspector, "Data inspector")
        bottom.addTab(self.console, "Python console")
        self.bottom_dock = self._dock(bottom, "Data", QtCore.Qt.BottomDockWidgetArea, None)

        self._build_menus()
        self.status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.timeline.play.toggle)
        if pipeline is not None:
            self.set_pipeline(pipeline)

    # ---------------------------------------------------------------- layout helpers
    def _dock(self, widget, title, area, width):
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        if width:
            self.resizeDocks([d], [width], QtCore.Qt.Horizontal)
        return d

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction("&Load file…", self.open_dialog, QtGui.QKeySequence.Open)
        f.addAction("Export &image…", self.render_image, "Ctrl+Shift+R")
        f.addAction("E&xport data…", self.export_dialog, "Ctrl+E")
        f.addAction("Export &movie…", self.export_movie)
        f.addSeparator()
        f.addAction("&Quit", self.close, QtGui.QKeySequence.Quit)
        v = mb.addMenu("&View")
        for label, d in (("Top (+z)", "z"), ("Bottom (-z)", "-z"), ("Front (-y)", "front"), ("Back (+y)", "y"),
                         ("Left (-x)", "-x"), ("Right (+x)", "x"), ("Perspective (iso)", "iso")):
            v.addAction(label, lambda d=d: self.view(d))
        v.addSeparator()
        self.ortho_action = v.addAction("Orthographic", self._toggle_ortho)
        self.ortho_action.setCheckable(True)
        self.ortho_action.setChecked(True)
        v.addAction("Zoom all", lambda: (self.viewport.reset_camera(), self.viewport.render()), "Ctrl+0")
        m = mb.addMenu("&Modifiers")
        from .pipeline_panel import MODIFIER_MENU

        for group, items in MODIFIER_MENU.items():
            sub = m.addMenu(group)
            for label, factory in items.items():
                sub.addAction(label, lambda fct=factory: self.panel.add_modifier(fct()))

    # ---------------------------------------------------------------- pipeline / data
    def open_file(self, path, **kwargs):
        try:
            pipe = import_file(path, **kwargs)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"{path}\n\n{exc}")
            return None
        self.set_pipeline(pipe)
        self.setWindowTitle(f"ewald — {os.path.basename(str(path))}")
        return pipe

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load file", "", FILE_FILTER)
        if not path:
            return
        kwargs = {}
        from ..io.lammps_dump import is_lammps_dump

        if is_lammps_dump(path):
            text, ok = QtWidgets.QInputDialog.getText(
                self, "LAMMPS types", "Optional type → element map (e.g. 1=Cu,2=Zr):")
            if ok and text.strip():
                kwargs["type_map"] = dict(p.split("=") for p in text.replace(" ", "").split(","))
        self.open_file(path, **kwargs)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self._first_view = True
        self.panel.set_pipeline(pipeline)
        self.timeline.set_num_frames(pipeline.num_frames)
        self._push_console()
        self.set_frame(0)

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
            self.statusBar().showMessage(f"Error: {exc}", 10000)
            traceback.print_exc()
            return
        dt = time.perf_counter() - t
        self.panel.update_status()
        errors = [m for m in self.pipeline.modifiers if m.enabled and m.status]
        if errors:
            self.statusBar().showMessage(f"{errors[0].name}: {errors[0].status}", 10000)
        self.redraw()
        self.inspector.set_data(self.data)
        self._push_console()
        ts = self.data.attributes.get("Timestep", self.frame)
        self.status.setText(f"frame {self.frame}  timestep {ts}  N={self.data.count}  pipeline {dt * 1000:.0f} ms")

    def redraw(self):
        if self.data is None:
            return
        p = self.viewport
        for name in ("atoms", "bonds", "cell", "_legend"):
            p.remove_actor(name, render=False)
        p.remove_scalar_bar() if p.scalar_bars else None
        bg = self.render_settings.background.currentText()
        p.set_background(bg)
        kw = self.render_settings.scene_kwargs()
        kw["cell_color"] = "black" if bg == "white" else "white"
        kw["sphere_resolution"] = 10  # interactive; image export uses full quality
        build_scene(p, self.data, **kw)
        if "ColorCoding.range" in self.data.attributes:
            _add_colorbar(p, self.data)
        if self._first_view:
            setup_camera(p, self.data, "iso", self.ortho_action.isChecked())
            self._first_view = False
        p.render()

    # ---------------------------------------------------------------- view
    def view(self, direction):
        if self.data is not None:
            setup_camera(self.viewport, self.data, direction, self.ortho_action.isChecked())
            self.viewport.render()

    def _toggle_ortho(self):
        if self.ortho_action.isChecked():
            self.viewport.enable_parallel_projection()
        else:
            self.viewport.disable_parallel_projection()
        self.viewport.render()

    # ---------------------------------------------------------------- output
    def render_image(self, filename=None):
        if self.data is None:
            return None
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Render image", "render.png",
                                                                "Images (*.png *.jpg *.tif)")
            if not filename:
                return None
        rs = self.render_settings
        render(self.data, filename, size=(rs.width.value(), rs.height.value()), camera=camera_state(self.viewport),
               background=rs.background.currentText(), transparent_background=rs.transparent.isChecked(),
               ambient_occlusion=rs.ao.isChecked(), **rs.scene_kwargs())
        self.statusBar().showMessage(f"Rendered {filename}", 5000)
        return filename

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

    def export_movie(self):
        if self.pipeline is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export movie", "movie.mp4", "Movies (*.mp4 *.gif)")
        if not path:
            return
        from ..render import render_animation

        rs = self.render_settings
        render_animation(self.pipeline, path, fps=self.timeline.fps.value(), device=self.device,
                         size=(rs.width.value() // 2 * 2, rs.height.value() // 2 * 2),
                         camera=camera_state(self.viewport), background=rs.background.currentText(),
                         **rs.scene_kwargs())
        self.statusBar().showMessage(f"Wrote {path}", 5000)

    def closeEvent(self, event):
        self.timeline.timer.stop()
        self.viewport.close()  # release the VTK/OpenGL context before Qt tears down
        km = getattr(self.console, "kernel_manager", None)
        if km is not None:
            self.console.kernel_client.stop_channels()
            km.shutdown_kernel()
        super().closeEvent(event)

    # ---------------------------------------------------------------- console
    def _console_namespace(self):
        return {"np": np, "mods": mods, "viewer": self, "pipeline": self.pipeline, "data": self.data,
                "import_file": import_file}

    def _push_console(self):
        shell = getattr(self.console, "shell", None)
        ns = {"pipeline": self.pipeline, "data": self.data}
        if hasattr(shell, "push") and hasattr(shell, "user_ns"):
            shell.push(ns)
        elif shell is not None and hasattr(shell, "locals"):
            shell.locals.update(ns)

    def _after_console(self, *args):
        shell = getattr(self.console, "shell", None)
        ns = getattr(shell, "user_ns", None) or getattr(shell, "locals", {})
        pipe = ns.get("pipeline")
        if pipe is not None and pipe is not self.pipeline:
            self.set_pipeline(pipe)  # user did pipeline = import_file(...)
        elif self.pipeline is not None:
            self.panel.refresh()
            self.timeline.set_num_frames(self.pipeline.num_frames)
            self.recompute()


def show(pipeline=None, device=None, block: bool = True):
    """Open the viewer on an existing Pipeline (from a script or Jupyter with %gui qt)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ewald")
    win = MainWindow(device=args.device)
    if args.file:
        kw = {"type_map": dict(p.split("=") for p in args.type_map.split(","))} if args.type_map else {}
        win.open_file(args.file, **kw)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
