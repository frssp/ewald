"""ewald main window: Analyze and Render workspaces (layout per docs/design)."""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback

import numpy as np

from . import check_qt_binding

check_qt_binding()  # before PySide6: a clear error when another Qt binding is loaded

from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from ..render import camera_state, render
from .ask_panel import AskPanel, agent_from_settings
from .console import make_console
from .icons import app_icon
from .inspector import DataInspector
from .macos import set_app_name, strip_psn
from .notebook_panel import NotebookPanel
from .pipeline_panel import DisplayPanel, PipelinePanel
from .plots import PlotsPanel
from . import preferences as prefs
from .remotes_panel import install as install_remotes
from .render_page import RenderPage
from .style import ViewStyle
from .theme import TOKENS
from .live_reload import LiveReload
from .timeline import Timeline
from .viewport import Viewport
from .viewport_menu import ViewportMenuMixin
from .window_files import SESSION_FILTER, FilesMixin, file_filter, save_filters  # noqa: F401 (public)
from .window_links import LinksMixin
from .window_menus import MenusMixin

class MainWindow(MenusMixin, FilesMixin, ViewportMenuMixin, LinksMixin, QtWidgets.QMainWindow):
    def __init__(self, pipeline=None, device=None, settings=None):
        super().__init__()
        from .. import settings as settings_mod

        self.settings = settings if settings is not None else settings_mod.load()
        self.setWindowTitle("EWALD")
        self.setWindowIcon(app_icon())
        self.resize(1600, 1000)
        self.pipeline = None
        self.data = None
        self.device = device
        self.frame = 0
        self._frame_pending = False
        self.style = ViewStyle()
        self.session_path = None

        # --- central: workspaces (Analyze = viewport + timeline, Render = render page)
        self.viewport = make_viewport(self.style, self.settings["viewport.backend"])
        self.viewport.picked.connect(self._on_pick)
        self.viewport.hovered.connect(lambda text: self.hover_label.setText(text))
        self.viewport.bondEdit.connect(self.toggle_bond)
        self.viewport.styleChanged.connect(self._style_changed)
        self.viewport.contextMenuRequested.connect(self._show_viewport_menu)
        self.timeline = Timeline()
        self.timeline.frameChanged.connect(self.set_frame)
        self.live_reload = LiveReload.attach(self)
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

        self.layout_mode = self.settings["general.layout"]
        self.ask_panel = AskPanel(self._ask_context, agent_from_settings(lambda: self.settings))
        self.ask_panel.pipelineChanged.connect(self._after_claude)
        if self.layout_mode == "notebook":
            # --- left: notebook tabs (the pipeline as Python cells); right: display settings, Ask Claude
            self.panel = NotebookPanel()
            self.panel.changed.connect(self.recompute)
            self.panel.pipelineReplaced.connect(self.set_pipeline)
            self.panel.askRequested.connect(self.ask_claude)
            self.panel.plotCreated.connect(self._connect_plot)
            self.notebook_dock = self._dock(self.panel, "Notebook", QtCore.Qt.LeftDockWidgetArea, 560)
            if not prefs.layout_path(self.settings).exists():  # first start: a wide notebook
                QtCore.QTimer.singleShot(0, lambda: self.resizeDocks([self.notebook_dock], [640], QtCore.Qt.Horizontal))
            # VS Code-like activity bar: the Plugins side bar opens beside the notebook
            from .plugins_panel import PluginsPanel

            self.plugins_panel = PluginsPanel(self.settings)
            self.plugins_panel.stepRequested.connect(lambda m: self.panel.add_modifier(m))
            self.plugins_panel.changed.connect(lambda: self.message("Plugins changed"))
            self.plugins_panel.createRequested.connect(self._create_plugin)
            self.plugins_dock = self._dock(self.plugins_panel, "Plugins", QtCore.Qt.LeftDockWidgetArea, 300)
            self.splitDockWidget(self.plugins_dock, self.notebook_dock, QtCore.Qt.Horizontal)
            self.plugins_dock.hide()
            self.remotes_dock = install_remotes(self)
            self._build_activity_bar()
            self.display = DisplayPanel(self.style)
            self.display.styleChanged.connect(self._style_changed)
            self.display.modifierRequested.connect(lambda m: self.panel.add_modifier(m))
            side_dock = self._dock(self.display, "Display", QtCore.Qt.RightDockWidgetArea, 320)
            self.display.refresh()
        else:
            # --- right: command panel (pipeline list + properties)
            self.panel = self.display = PipelinePanel(self.style)
            self.panel.changed.connect(self.recompute)
            self.panel.styleChanged.connect(self._style_changed)
            side_dock = self._dock(self.panel, "Pipeline", QtCore.Qt.RightDockWidgetArea, 400)
            self.remotes_dock = install_remotes(self)
        self.ask_dock = self._dock(self.ask_panel, "Ask Claude", QtCore.Qt.RightDockWidgetArea, None)
        self.tabifyDockWidget(side_dock, self.ask_dock)
        side_dock.raise_()

        # --- bottom: console / inspector / plots / log
        self.inspector = DataInspector()
        self.console = make_console(self._console_namespace(), self._after_console, self)
        self.plots = PlotsPanel()
        self._connect_plot(self.plots)
        self._plot_picking = False
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(self.console, "Python console")
        self.bottom.addTab(self.inspector, "Data inspector")
        self.bottom.addTab(self.plots, "Plots")
        self.bottom.addTab(self.log, "Log")
        if importlib.util.find_spec("ewald._native") is None:
            self.log.appendPlainText("ewald._native (C++ extension) is not installed: PTM, Voronoi, CSP matching "
                                     "and interval CNA are unavailable. See Preferences › Compute.")
        self.bottom.currentChanged.connect(self._update_bottom_tab)
        self._stale_tabs = set()  # bottom tabs not yet showing the current frame
        self.bottom_dock = self._dock(self.bottom, "Data", QtCore.Qt.BottomDockWidgetArea, None)
        if self.layout_mode == "notebook":  # notebook and side panels run the full height
            self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
            self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.resizeDocks([self.bottom_dock], [280], QtCore.Qt.Vertical)

        self._build_menus()
        self._build_toolbar()
        self.status = QtWidgets.QLabel()
        self.pick_label = QtWidgets.QLabel()
        self.pick_label.setStyleSheet(f"color: {TOKENS['pick']};")
        self.hover_label = QtWidgets.QLabel(objectName="muted")
        self.statusBar().addWidget(self.pick_label, 1)
        self.statusBar().addPermanentWidget(self.hover_label)
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
        app = QtWidgets.QApplication.instance()
        if isinstance(app, EwaldApplication):  # files from Finder (Open With, Dock icon, double-click)
            app.set_file_open_target(self.open_any)

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

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_gl_checked", False):  # the GL context exists once the window shows
            self._gl_checked = True
            QtCore.QTimer.singleShot(0, self.check_opengl)

    def check_opengl(self):
        """Warn once in the status bar when the viewport runs on a software OpenGL driver."""
        gl = self.viewport.opengl_info()
        if gl["software"]:
            self.message(f"Viewport uses software OpenGL ({gl['renderer']}): no GPU acceleration, "
                         "so large scenes are slow. Check the graphics driver (Help › About EWALD).")
        return gl

    def set_workspace(self, index):
        self.workspaces.setCurrentIndex(index)
        for i, b in enumerate(self.ws_buttons):
            b.setChecked(i == index)
        if index == 1:
            self.render_page.update_code()

    def _toggle_live_reload(self, on):
        """File › Live reload: on or off for this session (the default is in Preferences › General)."""
        self.live_reload.configure(on, self.settings["general.live_reload_interval"])
        self.message(f"Live reload {'on' if on else 'off'}")

    def preferences(self):
        dlg = prefs.PreferencesDialog(self.settings, self)
        dlg.applied.connect(lambda s: prefs.apply_settings(self, s))
        dlg.exec()
        return dlg

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
        self.style.cell_color = None
        self._style_changed()

    def apply_theme(self, name: str):
        """Apply a theme (element colors, background, cell color, radius preset) to the view."""
        self.style.apply_theme(name, self.data.types if self.data is not None else ())
        self.message(f"Theme: {name}")
        self._style_changed()

    def message(self, text, error=False):
        self.statusBar().showMessage(text, 10000)
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + ("ERROR " if error else "") + text)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        pipeline.cache_memory = self.settings["performance.cache_memory"] * 2**20
        self.viewport.reset_view()
        self.viewport.clear_picks()
        self.plots.pipeline = pipeline
        self.plots.forget_selection()
        self.panel.set_pipeline(pipeline)
        if self.display is not self.panel:
            self.display.set_pipeline(pipeline)
        self.timeline.set_num_frames(pipeline.num_frames)
        self.live_reload.set_pipeline(pipeline)
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
        n = self.pipeline.num_frames  # LoadTrajectory changes the frame count
        if self.timeline.slider.maximum() != max(0, n - 1):
            self.timeline.set_num_frames(n)
        for m in self.pipeline.modifiers:
            if m.enabled and m.status:
                self.message(f"{m.name}: {m.status}", error=True)
        if getattr(self, "_check_bond_limit", False):
            self._check_bond_limit = False
            limit = self.settings["bonds.auto_limit"]
            if getattr(self, "_bonds_auto_off", False) and self.data.count <= limit:
                self.style.bonds_enabled = True  # the previous file was too big; this one gets its bonds back
            self._bonds_auto_off = False
            if self.style.bonds_enabled and self.data.count > limit:
                # bond search and sticks would make a big system slow
                self.style.bonds_enabled, self._bonds_auto_off = False, True
                self.message(f"Bonds off: {self.data.count:,} atoms is above the automatic-bonds limit "
                             f"({limit:,}, Preferences › Elements). Turn them on under Bonds.")
        if self.style.bonds_enabled:
            self.style.ensure_bond_pairs(self.data.types)
        self._apply_element_defaults()
        self.panel.set_data(self.data)
        if self.display is not self.panel:
            self.display.set_data(self.data)
        ts = self.data.attributes.get("Timestep", self.frame)
        self.viewport.overlay_text = f"Frame {self.frame}   timestep {ts}"
        self.viewport.set_data(self.data, interactive=interactive)
        self._stale_tabs = {self.inspector, self.plots}
        self._update_bottom_tab()
        for p in self.plot_panels()[1:]:  # view cells: their data comes with the notebook's
            p.set_highlight(self._selected_indices())
        self._push_console()
        dev = (self.device or "cpu").upper()
        if dev == "AUTO":  # show what auto resolved to
            dev = "AUTO (CUDA)" if self._gpu_available() else "AUTO (CPU)"
        self.status.setText(f"N = {self.data.count}   frame {self.frame}/{self.pipeline.num_frames - 1}   "
                            f"pipeline {dt * 1000:.0f} ms   {dev}")
        if self.workspaces.currentIndex() == 1:
            self.render_page.update_code()

    def _update_bottom_tab(self, *_):
        """Fill the inspector or plots tab with the current frame, only while it is the open
        tab (a plot range linked to the viewport follows the frame either way): a hidden
        histogram or particle table cost a redraw on every frame of playback."""
        if self.data is None:
            return
        for tab in list(self._stale_tabs):
            linked = tab is self.plots and self.plots._range is not None
            if tab is not self.bottom.currentWidget() and not linked:
                continue
            self._stale_tabs.discard(tab)
            tab.set_data(self.data)
            if tab is self.plots:
                self.plots.set_highlight(self._selected_indices())

    def play_phonon_mode(self, qpoint: int, band: int):
        """Show phonon mode (qpoint, band), e.g. clicked in the band plot, and start playback."""
        if self.pipeline is None:
            return
        mod = next((m for m in self.pipeline.modifiers if isinstance(m, mods.PhononMode)), None)
        if mod is None:
            mod = mods.PhononMode()
            self.pipeline.modifiers.append(mod)
        mod.enabled = True
        mod.qpoint, mod.band = int(qpoint), int(band)
        before = self.data.attributes.get("Phonon.Supercell") if self.data is not None else None
        self.panel.refresh(select=("mod", mod))
        if self.pipeline.compute(self.frame, errors="skip").attributes.get("Phonon.Supercell") != before:
            self.viewport.reset_view()  # a new q-point may need a bigger supercell: refit the camera
        self.recompute()
        if not self.timeline.play.isChecked() and self.pipeline.num_frames > 1:
            self.timeline.play.setChecked(True)
        a = self.data.attributes if self.data is not None else {}
        if "Phonon.Frequency" in a:
            q = ", ".join(f"{x:g}" for x in a["Phonon.q"])
            self.message(f"Phonon mode q = ({q}), band {a['Phonon.Band']}: {a['Phonon.Frequency']:.4g} THz "
                         f"({a['Phonon.Wavenumber']:.4g} cm-1)")

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
        if self.display._current() and self.display._current()[0] == "visual":
            pass  # editor already reflects the change
        else:
            self.display.refresh()

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

    def closeEvent(self, event):
        if self.layout_mode == "notebook" and self.isVisible() and not self.panel.confirm_discard(title="Quit"):
            event.ignore()
            return
        prefs.save_layout(self, self.settings)
        self.timeline.timer.stop()
        self.live_reload.timer.stop()
        self.viewport.plotter.close()  # release the VTK/OpenGL context before Qt tears down
        km = getattr(self.console, "kernel_manager", None)
        if km is not None:
            self.console.kernel_client.stop_channels()
            km.shutdown_kernel()
        super().closeEvent(event)

def make_viewport(style, choice: str | None = None) -> Viewport:
    """The VTK viewport, or the OpenGL 1.x one for SSH X11 forwarding (see ``viewport_backend``)."""
    from . import viewport_backend

    if viewport_backend(choice) == "legacy-gl":
        from .legacy_viewport import LegacyViewport

        return LegacyViewport(style)
    return Viewport(style)


def _set_windows_app_id():
    """Own taskbar entry and icon on Windows instead of python.exe's."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ewald.gui")
        except (AttributeError, OSError):
            pass


class EwaldApplication(QtWidgets.QApplication):
    """QApplication that hands macOS file-open events to the newest main window.

    Finder sends a QFileOpenEvent, not argv, when a file is opened with the .app (double-click,
    Open With, a drop on the Dock icon). It can arrive before the main window exists, so paths
    wait until a window registers itself with :meth:`set_file_open_target`.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._pending_files: list[str] = []
        self._file_open_target = None

    def set_file_open_target(self, open_path):
        self._file_open_target = open_path
        while self._pending_files:
            try:
                open_path(self._pending_files[0])
            except RuntimeError:  # that window was closed and deleted: wait for the next one
                self._file_open_target = None
                return
            self._pending_files.pop(0)

    def event(self, event):
        if event.type() == QtCore.QEvent.FileOpen:
            path = event.file() or event.url().toLocalFile()
            if path:
                self._pending_files.append(path)
                if self._file_open_target is not None:
                    self.set_file_open_target(self._file_open_target)
            return True
        return super().event(event)


def _app():
    from . import theme

    app = QtWidgets.QApplication.instance()
    if app is None:
        from . import check_qt_platform, check_remote_display, viewport_backend
        from .. import settings as settings_mod

        check_qt_platform()
        if viewport_backend(settings_mod.load()["viewport.backend"]) == "vtk":
            check_remote_display()
        _set_windows_app_id()
        set_app_name()
        app = EwaldApplication(strip_psn(sys.argv))
        app.setApplicationName("ewald")
        app.setDesktopFileName("ewald")
        app.setWindowIcon(app_icon())
        try:
            name = settings_mod.load()["general.ui_theme"]
        except Exception:  # a broken settings file must not stop the GUI
            name = None
        theme.apply(app, name)
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

    ap = argparse.ArgumentParser(prog="ewald-gui", description="EWALD desktop GUI (Extensible Workbench for Atomic Lattices & Dynamics)")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--type-map", default=None, help="LAMMPS types, e.g. 1=Cu,2=Zr")
    ap.add_argument("--device", default=None, help="cpu, cuda or auto")
    args = ap.parse_args(strip_psn(sys.argv[1:] if argv is None else argv))
    app = _app()
    win = MainWindow(device=args.device)
    if args.file:
        kw = {"type_map": dict(p.split("=") for p in args.type_map.split(","))} if args.type_map else {}
        win.open_any(args.file, ask_types=False, **kw)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
