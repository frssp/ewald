"""Toolbar, menu bar, Help menu and command palette of the main window."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import examples, manual
from .icons import icon
from .palette import CommandPalette, menu_entries
from .pipeline_panel import MODIFIER_MENU, fill_modifier_menu
from .theme import TOKENS
from .viewport import Viewport


class MenusMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

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
        else:
            text += f"<br>Compute: CPU (no CUDA device or CuPy); current device {self.device or 'cpu'}"
        return text

    def command_palette(self):
        entries = [(label, f"Add modification › {group}", lambda f=factory: self.panel.add_modifier(f()))
                   for group, items in MODIFIER_MENU.items() for label, factory in items.items()]
        entries += menu_entries(self.menuBar())
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
