"""Toolbar, menu bar, Help menu and command palette of the main window."""
from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .. import examples, manual
from .icons import icon
from .palette import CommandPalette, menu_entries
from .pipeline_panel import fill_modifier_menu
from .cell_widget import fill_view_menu
from .theme import PALETTES
from .viewport import Viewport


class MenusMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    def _build_activity_bar(self):
        """Icons on the far left that show or hide the side bars (Notebook, Plugins, Remotes, AI chat)."""
        bar = self.activity_bar = QtWidgets.QToolBar("Activity bar")
        bar.setObjectName("activity-bar")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(22, 22))
        self.addToolBar(QtCore.Qt.LeftToolBarArea, bar)
        self.activity_buttons = {}
        for name, icon_name, dock in (("Notebook", "notebook", self.notebook_dock),
                                      ("Plugins", "plugins", self.plugins_dock),
                                      ("Remotes", "remote", self.remotes_dock),
                                      ("AI chat", "claude", self.ask_dock)):
            # not dock.toggleViewAction(): Qt disables it for docks without DockWidgetClosable
            act = QtGui.QAction(icon(icon_name), name, bar, checkable=True)
            act.triggered.connect(lambda on, d=dock: self._show_side_bar(d) if on else d.hide())
            dock.visibilityChanged.connect(lambda _shown, a=act, d=dock: a.setChecked(not d.isHidden()))
            act.setProperty("ewald_icon", icon_name)
            act.setToolTip("Chat with AI (Ctrl+J)" if dock is self.ask_dock
                           else f"Show or hide the {name} side bar")
            bar.addAction(act)
            self.activity_buttons[name] = act
        self.plugins_dock.visibilityChanged.connect(
            lambda shown: shown and self.plugins_panel.search.setFocus())
        self.plugins_panel.updatesChanged.connect(self._plugin_updates)
        self._plugin_updates(len(self.plugins_panel.updates()))
        self.ask_dock.visibilityChanged.connect(lambda shown: shown and self.ask_panel.input.setFocus())

    def refresh_activity_icons(self):
        """Redraw the activity bar icons in the colors of the UI theme now active."""
        for name, act in self.activity_buttons.items():
            if name == "Plugins":
                self._plugin_updates(act.property("updates") or 0)
            else:
                act.setIcon(icon(act.property("ewald_icon")))

    def _plugin_updates(self, count):
        """A count badge on the Plugins icon while installed plugins have newer catalog versions."""
        from .icons import badged

        act = self.activity_buttons["Plugins"]
        act.setIcon(badged("plugins", count))
        act.setToolTip(f"Plugins: {count} update{'s' if count != 1 else ''} available" if count
                       else "Show or hide the Plugins side bar")
        act.setProperty("updates", count)

    def _show_side_bar(self, dock):
        """Show an activity-bar side bar (Plugins, Remotes, AI chat) in a column of its own left of
        the notebook (Qt would otherwise stack a re-shown dock above it)."""
        placed = dock.isHidden() and not dock.isFloating() and self.notebook_dock.isVisible()
        if placed:
            self.splitDockWidget(dock, self.notebook_dock, QtCore.Qt.Horizontal)
        dock.show()
        dock.raise_()
        if placed:
            self.resizeDocks([dock], [380 if dock is self.ask_dock else 320], QtCore.Qt.Horizontal)

    def set_minimal(self, on: bool | None = None):
        """Hide the menu bar, toolbars and status bar, leaving the notebook, the viewport and the
        command palette (Ctrl+K); menu shortcuts keep working (View › Minimal mode, Ctrl+Shift+M).
        Ways back stay visible: a button in the bottom left corner, a tap on Alt shows the menu bar
        for one pick, and a note over the window says so when the mode starts."""
        on = not getattr(self, "minimal", False) if on is None else on
        if on:
            for menu in self.menuBar().findChildren(QtWidgets.QMenu):
                for act in menu.actions():
                    if not act.shortcut().isEmpty() and act not in self.actions():
                        self.addAction(act)  # shortcuts of a hidden menu bar do not fire otherwise
        self.minimal = on
        self.menuBar().setVisible(not on)
        self.statusBar().setVisible(not on)
        for tb in self.findChildren(QtWidgets.QToolBar):
            if tb.objectName() in ("main-toolbar", "activity-bar"):
                tb.setVisible(not on)
        if not hasattr(self, "minimal_exit"):
            self._build_minimal_exit()
        self.minimal_exit.setVisible(on)
        if on:
            self._place_minimal_exit()
            self.minimal_exit.raise_()
            self.toast("Minimal mode: Ctrl+Shift+M or the Menus button at the bottom left brings the menus back; "
                       "tap Alt for the menu bar, Ctrl+K for commands")
        else:
            self.minimal_note.hide()

    def _build_minimal_exit(self):
        b = self.minimal_exit = QtWidgets.QToolButton(self, objectName="minimalExit", autoRaise=True,
                                                      toolTip="Leave minimal mode (Ctrl+Shift+M)")
        b.setProperty("ewald_icon", "menu")
        b.setIcon(icon("menu"))
        b.setIconSize(QtCore.QSize(16, 16))
        b.setText("Menus")
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        b.clicked.connect(lambda: self.set_minimal(False))
        b.hide()
        self.minimal_note = QtWidgets.QLabel(self, objectName="toast", wordWrap=True)
        self.minimal_note.hide()
        self._note_timer = QtCore.QTimer(self, singleShot=True, timeout=self.minimal_note.hide)
        self.menuBar().triggered.connect(lambda _a: self.minimal and self.menuBar().hide())
        QtWidgets.QApplication.instance().installEventFilter(self._alt_filter())

    def _alt_filter(self):
        window = self

        class AltTap(QtCore.QObject):
            """A tap on Alt alone shows (or hides) the menu bar while minimal mode is on."""
            armed = False

            def eventFilter(self, obj, e):
                t = e.type()
                if t == QtCore.QEvent.KeyPress:
                    self.armed = e.key() == QtCore.Qt.Key_Alt and not e.isAutoRepeat()
                elif t == QtCore.QEvent.KeyRelease and e.key() == QtCore.Qt.Key_Alt and self.armed:
                    self.armed = False
                    if getattr(window, "minimal", False) and window.isActiveWindow():
                        bar = window.menuBar()
                        bar.setVisible(not bar.isVisible())
                        window._place_minimal_exit()
                elif t in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.Wheel):
                    self.armed = False
                return False

        self._alt_tap = AltTap(self)
        return self._alt_tap

    def _place_minimal_exit(self):
        b = getattr(self, "minimal_exit", None)
        if b is None:
            return
        b.adjustSize()
        top = self.menuBar().height() if self.menuBar().isVisible() else 0
        b.move(10, self.height() - b.height() - 10)  # the status bar's place, hidden in minimal mode
        note = self.minimal_note
        if note.isVisible():
            note.setFixedWidth(min(560, self.width() - 40))
            note.adjustSize()
            note.move((self.width() - note.width()) // 2, top + 40)

    def toast(self, text: str, ms: int = 6000):
        """A note over the top of the window (the status bar may be hidden), gone after ``ms``."""
        if not hasattr(self, "minimal_note"):
            self._build_minimal_exit()
        self.minimal_note.setText(text)
        self.minimal_note.show()
        self.minimal_note.raise_()
        self._place_minimal_exit()
        self._note_timer.start(ms)
        self.message(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_minimal_exit()

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

        button("Open", "open", self.open_dialog, "Open a data file, notebook (.py) or session (Ctrl+O)")
        button("Save", "save", lambda: self.save_notebook(), "Save the notebook (Ctrl+S)", False)
        # Run all, Restart and Interrupt live in the notebook's own bar (and Notebook › Run all, Ctrl+Shift+Enter)
        tb.addSeparator()
        self.ws_buttons = []
        for i, (name, icon_name) in enumerate((("Notebook", "analyze"), ("Render", "render"))):
            b = button(name, icon_name, lambda _=False, i=i: self.set_workspace(i), f"{name} workspace (Ctrl+{i + 1})",
                       checkable=True)
            self.ws_buttons.append(b)
        self.ws_buttons[0].setChecked(True)
        tb.addSeparator()
        search = QtWidgets.QPushButton("  Search commands, steps, cells…      Ctrl K")
        search.setObjectName("searchBox")
        search.clicked.connect(self.command_palette)
        tb.addWidget(search)
        spacer = QtWidgets.QWidget(objectName="toolbarSpacer")
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)

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
        f.addAction("&New notebook", lambda: self.panel.new_notebook(), "Ctrl+T")
        f.addAction(icon("open"), "&Open…", QtGui.QKeySequence.Open, self.open_dialog)
        f.addAction(icon("remote"), "Load &remote file (SSH)…", QtGui.QKeySequence("Ctrl+Alt+O"), self.open_remote_dialog)
        f.addAction("Remote &computers (SSH profiles)…", QtGui.QKeySequence("Ctrl+Alt+R"), self.show_remotes)
        f.addAction("Load &NEB folder…", self.open_neb_dialog)
        ex = f.addMenu(icon("example"), "Open e&xample")
        for example in examples.EXAMPLES:
            act = ex.addAction(f"{example.file}  ({example.format})", lambda e=example: self.open_example(e.file))
            act.setToolTip(example.description)
            act.setStatusTip(example.description)
        ex.setToolTipsVisible(True)
        f.addSeparator()
        f.addAction(icon("save"), "&Save notebook", QtGui.QKeySequence.Save, self.save_notebook)
        f.addAction("Save notebook &as…", lambda: self.save_notebook(ask=True), QtGui.QKeySequence.SaveAs)
        f.addAction("&Close notebook", lambda: self.panel.close_tab(self.panel.tabs.currentIndex()), "Ctrl+W")
        f.addSeparator()
        ses = f.addMenu("Sess&ion (.ewald)")  # the older single-pipeline file; notebooks and .ewd replace it
        ses.addAction("&Load session…", self.load_session_dialog, "Ctrl+Shift+O")
        ses.addAction("&Save session as…", lambda: self.save_session(ask=True))
        f.addSeparator()
        f.addAction(icon("image"), "Render &image…", QtGui.QKeySequence("Ctrl+Shift+R"), self.render_image)
        f.addAction("E&xport data…", self.export_dialog, "Ctrl+E")
        f.addAction("Export &movie…", self.export_movie)
        f.addSeparator()
        self.live_reload_action = f.addAction("&Live reload")
        self.live_reload_action.setCheckable(True)
        self.live_reload_action.triggered.connect(self._toggle_live_reload)
        self.live_reload_action.setStatusTip("Follow a trajectory that is still being written: new frames "
                                             "extend the timeline (Preferences › General).")
        f.addSeparator()
        f.addAction("&Quit", self.close, QtGui.QKeySequence.Quit).setMenuRole(QtGui.QAction.QuitRole)
        e = mb.addMenu("&Edit")
        # A focused cell editor takes Ctrl+Z itself (its typing); elsewhere these undo
        # notebook edits: cells added, deleted, moved or run, step forms, pipeline changes.
        self.undo_action = e.addAction("&Undo", self.undo, QtGui.QKeySequence.Undo)
        self.redo_action = e.addAction("&Redo", self.redo, QtGui.QKeySequence.Redo)
        e.aboutToShow.connect(self._update_undo_actions)
        e.addSeparator()
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
        rep = v.addMenu("Replicate cell")  # display only: n x n x m copies (view(pipe, replicate=...))
        rep.aboutToShow.connect(lambda: self.fill_replicate_menu(rep))
        self.fill_replicate_menu(rep)
        v.addSeparator()
        self.theme_menu = v.addMenu("Theme")
        self.theme_menu.aboutToShow.connect(self._fill_theme_menu)
        self._fill_theme_menu()
        lay = v.addMenu("Layout")  # tabs drag between groups; these are the keyboard and menu way
        for act in self.workbench.actions():
            lay.addAction(act)
        lay.setToolTipsVisible(True)
        lay.aboutToShow.connect(self.workbench.update_actions)
        lay.aboutToHide.connect(self.workbench.enable_actions)
        v.addSeparator()
        v.addAction("Focus on the viewport", self.toggle_focus, "Ctrl+\\")
        v.addAction("Minimal mode (menus hidden)", lambda: self.set_minimal(not getattr(self, "minimal", False)), "Ctrl+Shift+M")
        n = mb.addMenu("&Notebook")
        n.addAction("Run &all", lambda: self.panel.view.run_all(), "Ctrl+Shift+Return").setToolTip(
            "Restart and run every cell from the top")
        n.addAction("Run current cell and &below", lambda: self._cell_action("below"))
        n.addAction("&Restart", lambda: self.panel.view.restart())
        n.setToolTipsVisible(True)
        self.kernel_menu = n.addMenu("&Kernel")
        self.kernel_menu.aboutToShow.connect(lambda: self.panel.view.fill_kernel_menu(self.kernel_menu))
        self.trust_action = n.addAction("&Trust notebook…", lambda: self.panel.view.trust())
        self.trust_action.setToolTip("Allow a notebook received as an .ewd file to run its code and plugins")
        n.aboutToShow.connect(lambda: self.trust_action.setEnabled(
            self.panel.view is not None and self.panel.view.notebook.restricted))
        n.addSeparator()
        n.addAction("Move cell &up", lambda: self._cell_action("up"), "Ctrl+Shift+Up")
        n.addAction("Move cell &down", lambda: self._cell_action("down"), "Ctrl+Shift+Down")
        n.addAction("D&elete cell", lambda: self._cell_action("delete"))
        n.addSeparator()
        n.addAction("&Clear cell output", lambda: self._cell_action("clear"), "Ctrl+Shift+Backspace")
        n.addAction("Clear &all outputs", lambda: self.panel.view.clear_outputs(),
                    "Ctrl+Alt+Shift+Backspace").setToolTip("Remove every cell's output; the next save leaves them out")
        n.addSeparator()
        n.addAction("Show as Python &file", lambda: self.panel.view.py_button.toggle())
        ins = mb.addMenu("&Insert")
        ins.addAction("&Code cell", lambda: self.panel.view.add_cell("code"), "Ctrl+Shift+N")
        ins.addAction("&Text cell", lambda: self.panel.view.add_cell("markdown"))
        ins.addSeparator()
        m = ins.addMenu("&Step")
        m.aboutToShow.connect(lambda: fill_modifier_menu(m, self.panel.add_modifier, self._step_context))
        fill_modifier_menu(m, self.panel.add_modifier)
        vm = ins.addMenu("&View")
        vm.aboutToShow.connect(lambda: fill_view_menu(vm, self.data, lambda **kw: self.panel.view.add_view(**kw)))
        ins.addAction("3D &snapshot of the view", self._insert_snapshot).setToolTip(
            "A cell with a picture of the viewport as it is now (frame, camera, display style)")
        ins.addAction("View &settings with the camera", self.write_camera).setToolTip(
            "Write the display style, the camera and the frame into the notebook's view(pipe, ...) cell")
        am = ins.addMenu("&Analysis")
        from ..verbs import VERB_CELLS

        for verb, label, _code in VERB_CELLS:
            act = am.addAction(label, lambda v=verb: self._insert_verb(v))
            act.setToolTip(f"A {verb}(pipe) cell, run at once")
            act.setData(verb)
        am.setToolTipsVisible(True)
        am.aboutToShow.connect(self._update_analysis_menu)
        self.analysis_menu = am
        plug = ins.addMenu("&Plugin step")
        plug.aboutToShow.connect(lambda: self._fill_plugin_menu(plug))
        self.step_menu = m
        t = mb.addMenu("&Tools")
        for key, _icon, tip in Viewport.TOOLS:
            t.addAction(tip.split(":")[0], lambda k=key: self.viewport.set_tool(k))
        w = mb.addMenu("&Window")
        w.addAction("Notebook workspace", lambda: self.set_workspace(0), "Ctrl+1")
        w.addAction("Render workspace", lambda: self.set_workspace(1), "Ctrl+2")
        w.addAction("Notebook", lambda: (self.notebook_dock.show(), self.notebook_dock.raise_()))
        w.addAction("Plugins", lambda: self._show_side_bar(self.plugins_dock), "Ctrl+Shift+X")
        w.addAction("Structure", lambda: self.workbench.show(self.viewport))
        w.addAction("View settings", lambda: self.workbench.show(self.display))
        w.addAction("Python console", lambda: self.workbench.show(self.console))
        w.addAction("Data inspector", lambda: self.workbench.show(self.inspector))
        w.addAction("Plots", lambda: self.workbench.show(self.plots))
        w.addAction("New plot view", lambda: self.new_plot_view())
        w.addAction("Log", lambda: self.workbench.show(self.log))
        w.addAction("AI chat", self.ask_claude, "Ctrl+J").setToolTip(
            "Chat with AI about the notebook, the structure or the picked atoms")
        w.addAction("Command palette", self.command_palette, "Ctrl+K")
        h = mb.addMenu("&Help")
        for label, page in manual.PAGES.items():
            h.addAction(label, lambda page=page: self.open_help(page))
        h.addSeparator()
        about = h.addAction("About EWALD", self.about)
        about.setMenuRole(QtGui.QAction.AboutRole)

    def _fill_plugin_menu(self, menu):
        view = self.panel.view
        view._fill_plugin_menu()
        menu.clear()
        for act in view.plugin_menu.actions():
            menu.addAction(act)

    def snapshot_code(self) -> str:
        """The ``snapshot(pipe, ...)`` cell for the viewport as it is now."""
        from ..render import camera_state

        def short(v):
            return tuple(round(x, 4) for x in v) if isinstance(v, tuple) else round(v, 4) if isinstance(v, float) else v

        kw = {"camera": {k: short(v) for k, v in camera_state(self.viewport.plotter).items()},
              "background": self.style.background}
        kw.update(self.style.scene_kwargs())
        name = self.panel.view.notebook.pipeline_name
        body = ",\n         ".join(f"{k}={v!r}" for k, v in kw.items())
        return f"from ewald.views import snapshot\nsnapshot({name}, frame={int(self.frame)},\n         {body})"

    def _insert_snapshot(self):
        """Insert › 3D snapshot: a cell that draws the current view, run at once."""
        view = self.panel.view
        if view is None or view.notebook.pipeline is None:
            QtWidgets.QMessageBox.information(self, "No pipeline yet",
                                              "Open a file first, or run a cell that sets pipe = import_file(...).")
            return
        cell = view.add_cell("code", self.snapshot_code())
        view.run(cell)
        self.notebook_dock.show()

    def _update_analysis_menu(self):
        """Grey out the trajectory analyses (MSD, VACF) while the data has a single frame."""
        from ..verbs import TRAJECTORY_VERBS

        frames = self.pipeline.num_frames if self.pipeline is not None else 0
        for act in self.analysis_menu.actions():
            if act.data() in TRAJECTORY_VERBS:
                act.setEnabled(frames > 1)
                act.setToolTip(f"A {act.data()}(pipe) cell, run at once" if frames > 1
                               else "Needs a trajectory: the data has a single frame")

    def _insert_verb(self, verb):
        """Insert › Analysis: a ``verb(pipe)`` cell after the current one, run at once (its table shows below)."""
        from ..verbs import verb_cell

        view = self.panel.view
        if view is None:
            return
        if view.notebook.pipeline is None:
            QtWidgets.QMessageBox.information(self, "No pipeline yet",
                                              "Open a file first, or run a cell that sets pipe = import_file(...).")
            return
        cell = view.add_cell("code", verb_cell(verb, self.data))
        view.run(cell)
        self.notebook_dock.show()

    def _step_context(self):
        """(current frame, pipeline) for greying out steps in Insert › Step and the Pipeline menu."""
        return self.data, self.pipeline

    def _cell_action(self, action):
        view = self.panel.view
        if view is not None and view.current is not None:
            view._cell_action(view.current, action)

    def toggle_focus(self):
        """Hide or show the notebook and side panels, leaving the viewport (Ctrl+\\)."""
        docks = [self.notebook_dock, self.plugins_dock, self.ask_dock, self.remotes_dock]
        hide = self.notebook_dock.isVisible()
        if hide:
            self._focus_restore = [d for d in docks if d.isVisible()]
            self._focus_panel = not self.workbench.panel_host.isHidden()
            for d in docks:
                d.hide()
            self.workbench.set_panel_visible(False)
        else:
            for d in getattr(self, "_focus_restore", None) or [self.notebook_dock]:
                d.show()
            self.workbench.set_panel_visible(getattr(self, "_focus_panel", True))

    def help_url(self, page="index.md"):
        return manual.url(page, self.settings["general.docs_url"] or None)

    def open_help(self, page="index.md"):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.help_url(page)))

    def about(self):
        from .. import __title__, __version__

        QtWidgets.QMessageBox.about(
            self, "About EWALD",
            f"<b>EWALD {__version__}</b><br>{__title__.split(': ', 1)[1].replace('&', '&amp;')}<p>Atomistic structure and trajectory visualization and analysis "
            f"in Python notebooks, with AI and plugins built in.</p><p>{self.hardware_summary()}</p>"
            f"<p><a href='{self.help_url()}'>Manual</a></p>")

    def _gpu_available(self) -> bool:
        if getattr(self, "_gpu", None) is None:
            from ..backend import gpu_available

            self._gpu = gpu_available()
        return self._gpu

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
        elif sys.platform == "darwin":
            text += f"<br>Compute: CPU (Apple GPUs are not used for compute yet); current device {self.device or 'cpu'}"
        else:
            text += f"<br>Compute: CPU (no CUDA device or CuPy); current device {self.device or 'cpu'}"
        return text

    def command_palette(self):
        for menu in self.menuBar().findChildren(QtWidgets.QMenu):
            menu.aboutToShow.emit()  # fill the menus that fill themselves when opened (Kernel, Insert › View, ...)
        entries = menu_entries(self.menuBar())  # Insert › Step lists the steps
        for menu in self.menuBar().findChildren(QtWidgets.QMenu):
            menu.aboutToHide.emit()  # View › Layout enables its moves again for their shortcuts
        pal = CommandPalette(entries, self)
        geo = self.geometry()
        pal.move(geo.x() + (geo.width() - 560) // 2, geo.y() + 90)
        pal.edit.setFocus()
        pal.exec()

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
        self.theme_menu.addSeparator()  # the background alone, keeping the element colors
        for label, color in (("Dark background", PALETTES["graphite"]["viewport"]), ("White background", "white")):
            act = self.theme_menu.addAction(label, lambda c=color: self._set_background(c))
            act.setToolTip("Change only the viewport background")
