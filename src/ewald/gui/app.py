"""ewald main window: Analyze and Render workspaces (layout per docs/design)."""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
import traceback

import numpy as np

from . import check_qt_binding

check_qt_binding()  # before PySide6: a clear error when another Qt binding is loaded

from PySide6 import QtCore, QtGui, QtWidgets

from .. import examples, manual
from .. import modifiers as mods
from ..io import import_file
from ..render import camera_state, render
from .ask_panel import AskPanel, agent_from_settings
from .console import make_console
from .icons import icon
from .inspector import DataInspector
from .palette import CommandPalette, menu_entries
from .pipeline_panel import MODIFIER_MENU, PipelinePanel, fill_modifier_menu
from .plots import PlotsPanel, _is_neb
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
        self.setWindowTitle("EWALD")
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
        self.ask_panel = AskPanel(self._ask_context, agent_from_settings(lambda: self.settings))
        self.ask_panel.pipelineChanged.connect(self._after_claude)
        self.ask_dock = self._dock(self.ask_panel, "Ask Claude", QtCore.Qt.RightDockWidgetArea, None)
        self.tabifyDockWidget(pipeline_dock, self.ask_dock)
        pipeline_dock.raise_()

        # --- bottom: console / inspector / plots / log
        self.inspector = DataInspector()
        self.console = make_console(self._console_namespace(), self._after_console, self)
        self.plots = PlotsPanel()
        self.plots.phononModePicked.connect(self.play_phonon_mode)
        self.plots.atomsSelected.connect(self._plot_selected_atoms)
        self.plots.framesSelected.connect(self._plot_selected_frames)
        self.plots.frameRequested.connect(self.timeline._set)
        self.plots.expressionRequested.connect(self._keep_plot_selection)
        self.timeline.selectionChanged.connect(self.plots.set_frame_selection)
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
        tb.setIconSize(QtCore.QSize(18, 18))
        self.styled_buttons = []  # (button, text in the default style); see icons.button_style

        def button(text, icon_name, slot, tip, text_in_hybrid=True, checkable=False):
            b = QtWidgets.QToolButton(text=text, checkable=checkable, toolTip=tip)
            b.setProperty("ewald_icon", icon_name)
            b.clicked.connect(slot)
            tb.addWidget(b)
            self.styled_buttons.append((b, text_in_hybrid))
            return b

        button("Open", "open", self.open_dialog, "Load file (Ctrl+O)")
        button("Open remote", "remote", self.open_remote_dialog, "Load remote file over SSH (Ctrl+Alt+O)", False)
        button("Save", "save", lambda: self.save_session(), "Save session (Ctrl+S)", False)
        tb.addSeparator()
        self.ws_buttons = []
        for i, (name, icon_name) in enumerate((("Analyze", "analyze"), ("Render", "render"))):
            b = button(name, icon_name, lambda _=False, i=i: self.set_workspace(i), f"{name} workspace (Ctrl+{i + 1})",
                       checkable=True)
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
        render_btn = QtWidgets.QPushButton("Render", toolTip="Render workspace: image, movie and figure code")
        render_btn.setObjectName("primary")
        render_btn.setProperty("ewald_icon", "render")
        render_btn.setProperty("ewald_icon_color", "#0B1220")
        render_btn.clicked.connect(lambda: self.set_workspace(1))
        tb.addWidget(render_btn)
        self.styled_buttons.append((render_btn, True))

    def set_button_style(self, style: str):
        """Toolbar and tool rail buttons as icons with text, icons or text
        (Preferences › General › Toolbar buttons); menus show icons unless 'Text'."""
        from .icons import button_style

        for b, text_in_hybrid in self.styled_buttons + self.viewport.styled_buttons:
            button_style(b, style, text_in_hybrid)
        self.viewport.set_button_style(style)
        QtWidgets.QApplication.instance().setAttribute(QtCore.Qt.AA_DontShowIconsInMenus, style == "Text")

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction(icon("open"), "&Load file…", QtGui.QKeySequence.Open, self.open_dialog)
        f.addAction(icon("remote"), "Load &remote file (SSH)…", QtGui.QKeySequence("Ctrl+Alt+O"), self.open_remote_dialog)
        f.addAction("Load &NEB folder…", self.open_neb_dialog)
        ex = f.addMenu(icon("example"), "Open e&xample")
        for example in examples.EXAMPLES:
            act = ex.addAction(f"{example.file}  ({example.format})", lambda e=example: self.open_example(e.file))
            act.setToolTip(example.description)
            act.setStatusTip(example.description)
        ex.setToolTipsVisible(True)
        f.addAction("Load &session…", self.load_session_dialog, "Ctrl+Shift+O")
        f.addAction(icon("save"), "&Save session", QtGui.QKeySequence.Save, self.save_session)
        f.addAction("Save session &as…", lambda: self.save_session(ask=True), QtGui.QKeySequence.SaveAs)
        f.addSeparator()
        f.addAction(icon("image"), "Render &image…", QtGui.QKeySequence("Ctrl+Shift+R"), self.render_image)
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
        self.theme_menu = v.addMenu("Theme")
        self.theme_menu.aboutToShow.connect(self._fill_theme_menu)
        self._fill_theme_menu()
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
        about = h.addAction("About EWALD", self.about)
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
        from .. import __title__, __version__

        QtWidgets.QMessageBox.about(
            self, "About EWALD",
            f"<b>EWALD {__version__}</b><br>{__title__.split(': ', 1)[1].replace('&', '&amp;')}<p>Atomistic structure and trajectory visualization and analysis "
            f"with an OVITO-style Python pipeline.</p><p>{self.hardware_summary()}</p>"
            f"<p><a href='{self.help_url()}'>Manual</a></p>")

    def hardware_summary(self) -> str:
        """OpenGL driver of the viewport and the CUDA device for compute, as HTML for About."""
        from ..backend import gpu_available

        gl = self.viewport.opengl_info()
        text = f"Viewport OpenGL: {gl['renderer'] or 'unknown'} ({gl['vendor']}, {gl['version']})"
        if gl["software"]:
            text += "<br><b>Software rendering (no GPU acceleration)</b>"
        if gpu_available():
            import cupy

            name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else name
            text += f"<br>Compute: CUDA available ({name}); current device {self.device or 'cpu'}"
        else:
            text += f"<br>Compute: CPU (no CUDA device or CuPy); current device {self.device or 'cpu'}"
        return text

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
        self.style.cell_color = None
        self._style_changed()

    def _fill_theme_menu(self):
        from .. import plugins

        self.theme_menu.clear()
        group = QtGui.QActionGroup(self.theme_menu)
        for name, p in plugins.themes().items():
            act = self.theme_menu.addAction(name, lambda n=name: self.apply_theme(n))
            act.setCheckable(True)
            act.setChecked(name == self.style.theme)
            act.setToolTip(p.description)
            group.addAction(act)

    def apply_theme(self, name: str):
        """Apply a theme (element colors, background, cell color, radius preset) to the view."""
        self.style.apply_theme(name, self.data.types if self.data is not None else ())
        self.message(f"Theme: {name}")
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
        if _has_phonon_modes(pipe) and not any(isinstance(m, mods.PhononMode) for m in pipe.modifiers):
            pipe.modifiers.append(mods.PhononMode())  # a phonon file opens playing a mode
        self._check_bond_limit = True
        self.style.bond_pairs.clear()  # a new structure gets its own VESTA bond table
        self.set_pipeline(pipe)
        self.setWindowTitle(f"EWALD — {os.path.basename(str(path))}")
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

    def open_example(self, name):
        """Open one of the shipped example files (File › Open example)."""
        return self.open_file(str(examples.path(name)))

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
        self.plots.pipeline = pipeline
        self.plots.forget_selection()
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
        ts = self.data.attributes.get("Timestep", self.frame)
        self.viewport.overlay_text = f"Frame {self.frame}   timestep {ts}"
        self.viewport.set_data(self.data, interactive=interactive)
        self._stale_tabs = {self.inspector, self.plots}
        self._update_bottom_tab()
        self._push_console()
        dev = (self.device or "cpu").upper()
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
        if not self._plot_picking:
            self.plots.forget_selection(frames=False)  # a click in the view replaces a plot range
        self.plots.set_highlight(self._selected_indices())
        self.ask_panel.update_selection()
        if text:
            self.log.appendPlainText(time.strftime("%H:%M:%S ") + text)

    # ---------------------------------------------------------------- linked plots
    def _selected_indices(self):
        """The viewport picks, else the 'Selection' property of the current frame."""
        indices = self.viewport.picks
        if not indices and self.data is not None and "Selection" in self.data:
            indices = np.nonzero(self.data.selection)[0].tolist()
        return indices

    def _plot_selected_atoms(self, indices):
        self._plot_picking = True
        try:
            self.viewport.set_picks(indices)
        finally:
            self._plot_picking = False

    def _plot_selected_frames(self, frames):
        self.timeline.set_selection(frames)
        if frames and self.frame not in frames:
            self.timeline._set(frames[0])

    def _keep_plot_selection(self, expression):
        from ..modifiers import ExpressionSelection

        if self.pipeline is None:
            return
        m = ExpressionSelection(expression)
        self.pipeline.modifiers.append(m)
        self.plots.forget_selection(frames=False)
        self.viewport.clear_picks()
        self.panel.refresh(select=("mod", m))
        self.recompute()
        self.message(f"Added Expression selection: {expression}")

    # ---------------------------------------------------------------- Claude
    def _ask_context(self):
        """(pipeline, frame, indices) for the Ask Claude panel: the viewport picks, else the
        'Selection' property of the current frame."""
        return self.pipeline, self.frame, self._selected_indices()

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
        if (d is not None and d.cell.is_valid) or everything:
            views = menu.addMenu("View along")
            for ax in ("a", "b", "c", "a*", "b*", "c*"):
                views.addAction(ax, lambda ax=ax: self.viewport.view_along(ax))
        ortho = menu.addAction("Orthographic")
        ortho.setCheckable(True)
        ortho.setChecked(self.viewport.ortho)
        ortho.toggled.connect(self.viewport.ortho_button.setChecked)
        menu.addAction(icon("fit"), "Zoom to fit", self.viewport.zoom_fit)
        menu.addSeparator()
        menu.addAction(icon("claude"), "Ask Claude about selection…" if picks or everything else "Ask Claude about this structure…",
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

    def _toggle_style(self, attr, on):
        setattr(self.style, attr, on)
        if attr == "bonds_enabled" and on and self.data is not None:
            self.style.ensure_bond_pairs(self.data.types)
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
        self.bottom_dock.show()
        self.bottom.setCurrentWidget(self.plots)
        if table is not None:
            self.plots.hist.setCurrentText("")
            self.plots.table.setCurrentText(table)

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
        self.setWindowTitle(f"EWALD — {os.path.basename(path)}")
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
            self.viewport.render()
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


def _has_phonon_modes(pipe) -> bool:
    try:
        return "Phonon Modes" in pipe.source_frame(0).attributes
    except Exception:
        return False


def make_viewport(style, choice: str | None = None) -> Viewport:
    """The VTK viewport, or the OpenGL 1.x one for SSH X11 forwarding (see ``viewport_backend``)."""
    from . import viewport_backend

    if viewport_backend(choice) == "legacy-gl":
        from .legacy_viewport import LegacyViewport

        return LegacyViewport(style)
    return Viewport(style)


def _app():
    from . import theme

    app = QtWidgets.QApplication.instance()
    if app is None:
        from . import check_qt_platform, check_remote_display, viewport_backend
        from .. import settings as settings_mod

        check_qt_platform()
        if viewport_backend(settings_mod.load()["viewport.backend"]) == "vtk":
            check_remote_display()
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

    ap = argparse.ArgumentParser(prog="ewald-gui", description="EWALD desktop GUI (Extensible Window for Atomic Lattices & Dynamics)")
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
