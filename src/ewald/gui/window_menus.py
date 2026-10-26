"""Toolbar, menu bar, Help menu and command palette of the main window."""
from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .. import examples, manual
from .cell_widget import fill_view_menu
from .icons import icon
from .palette import CommandCenter, CommandPalette, menu_entries, native_text
from .pipeline_panel import fill_modifier_menu
from .theme import PALETTES
from .viewport import Viewport


class _ResizeWatch(QtCore.QObject):
    """Calls ``callback`` after ``target`` changes size (once the layout settled)."""

    def __init__(self, target, callback):
        super().__init__(target)
        self.callback = callback
        target.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self, self.callback)
        return False


class MenusMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    def _build_activity_bar(self):
        """Icons on the far left that show or hide the side bars. Work on top (Notebook, AI chat,
        plugin side bars, Remotes), management at the bottom (Plugins, Preferences), as in VS Code;
        the Window menu lists the side bars in the same order."""
        bar = self.activity_bar = QtWidgets.QToolBar("Activity bar")
        bar.setObjectName("activity-bar")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(22, 22))
        self.addToolBar(QtCore.Qt.LeftToolBarArea, bar)
        self.activity_buttons = {}
        for name, icon_name, dock, tip in (("Notebook", "notebook", self.notebook_dock, "Notebook"),
                                           ("AI chat", "ai", self.ask_dock, "AI chat (Ctrl+J)"),
                                           ("Remotes", "remote", self.remotes_dock,
                                            "Remotes: saved SSH computers (Ctrl+Alt+R)"),
                                           (None, None, None, None),
                                           ("Plugins", "plugins", self.plugins_dock, "Plugins (Ctrl+Shift+X)")):
            if name is None:   # "More side bars" when the top group overflows, then the bottom group
                more = self.activity_more = QtGui.QAction(icon("more"), "More side bars", bar)
                more.setProperty("ewald_icon", "more")
                more.setToolTip("More side bars")
                more.triggered.connect(self._activity_overflow_menu)
                bar.addAction(more)
                spacer = QtWidgets.QWidget(objectName="activitySpacer")
                spacer.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
                self.activity_spacer = bar.addWidget(spacer)
                continue
            # not dock.toggleViewAction(): Qt disables it for docks without DockWidgetClosable
            act = QtGui.QAction(icon(icon_name), name, bar, checkable=True)
            act.triggered.connect(lambda on, d=dock: self._show_side_bar(d) if on else d.hide())
            dock.visibilityChanged.connect(lambda _shown, a=act, d=dock: a.setChecked(not d.isHidden()))
            act.setProperty("ewald_icon", icon_name)
            act.setProperty("ewald_tip", native_text(tip))
            act.setToolTip(native_text(tip))
            act.toggled.connect(self.arrange_activity_bar)   # an open side bar keeps its button
            bar.addAction(act)
            self.activity_buttons[name] = act
        act = QtGui.QAction(icon("settings"), "Preferences", bar)
        act.triggered.connect(lambda: self.preferences())
        act.setProperty("ewald_icon", "settings")
        act.setToolTip(native_text("Preferences (Ctrl+,)"))
        bar.addAction(act)
        self.activity_buttons["Preferences"] = act
        self.plugins_dock.visibilityChanged.connect(
            lambda shown: shown and self.plugins_panel.search.setFocus())
        self.plugins_panel.updatesChanged.connect(self._plugin_updates)
        self._plugin_updates(len(self.plugins_panel.updates()))
        self.ask_dock.visibilityChanged.connect(lambda shown: shown and self.ask_panel.input.setFocus())
        self._activity_overflow = []
        bar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._activity_context_menu)
        _ResizeWatch(bar, self.arrange_activity_bar)

    def _activity_top(self) -> list:
        """The actions of the activity bar's top group (side bars you work in), in bar order."""
        acts = self.activity_bar.actions()
        return [a for a in acts[:acts.index(self.activity_spacer)] if a is not self.activity_more]

    def _activity_hidden(self) -> set:
        return {n.strip() for n in self.settings["general.activity_hidden"].split(",") if n.strip()}

    def arrange_activity_bar(self, *_):
        """Fit the activity bar's top group into its height: buttons hidden from its right-click menu
        stay off, and when the rest do not fit the last slot becomes "More side bars" listing the
        others, as in VS Code. An open side bar keeps its button; Plugins and Preferences at the
        bottom always stay. Every side bar stays in the Window menu."""
        bar = self.activity_bar
        if not hasattr(self, "activity_spacer") or "Preferences" not in self.activity_buttons:
            return   # still being built
        hidden = self._activity_hidden()
        top = self._activity_top()
        wanted = [a for a in top if a.text() not in hidden]
        button = bar.widgetForAction(self.activity_buttons["Preferences"])
        step = button.sizeHint().height() + max(bar.layout().spacing(), 0)
        margins = bar.contentsMargins().top() + bar.contentsMargins().bottom() + 2 * bar.layout().contentsMargins().top()
        fit = max(1, int((bar.height() - margins - 2 * step) // step))   # 2: Plugins, Preferences
        keep = wanted
        if len(wanted) > fit:
            keep = wanted[:max(fit - 1, 0)]                       # one slot for "More side bars"
            for a in [a for a in wanted[len(keep):] if a.isChecked()][:len(keep)]:
                keep = keep[:-1] if not keep[-1].isChecked() else keep
                keep.append(a)
            keep = [a for a in wanted if a in keep]
        self._activity_overflow = [a for a in wanted if a not in keep]
        for a in top:
            a.setVisible(a in keep)
        self.activity_more.setVisible(bool(self._activity_overflow))

    def _activity_overflow_menu(self):
        """The side bars that did not fit the activity bar, beside its "More side bars" button."""
        menu = QtWidgets.QMenu(self)
        for a in self._activity_overflow:
            menu.addAction(a)
        button = self.activity_bar.widgetForAction(self.activity_more)
        menu.exec(button.mapToGlobal(QtCore.QPoint(button.width(), 0)))

    def _activity_context_menu(self, pos):
        """Right-click on the activity bar: tick the side bars whose buttons it shows."""
        menu = QtWidgets.QMenu(self)
        hidden = self._activity_hidden()
        for a in self._activity_top():
            entry = menu.addAction(a.text())
            entry.setCheckable(True)
            entry.setChecked(a.text() not in hidden)
            entry.toggled.connect(lambda on, n=a.text(): self.set_activity_hidden(n, not on))
        menu.addSeparator()
        menu.addAction("Show all buttons", lambda: self.set_activity_hidden(None))
        menu.exec(self.activity_bar.mapToGlobal(pos))

    def set_activity_hidden(self, name: str | None, hide: bool = False):
        """Hide or show the activity-bar button of side bar ``name`` (``None``: show all) and save
        it (Preferences › General › Hidden activity bar buttons)."""
        hidden = set() if name is None else self._activity_hidden()
        if name is not None:
            hidden = hidden | {name} if hide else hidden - {name}
        self.settings["general.activity_hidden"] = ", ".join(sorted(hidden))
        self.settings.save()
        self.arrange_activity_bar()

    def refresh_activity_icons(self):
        """Redraw the activity bar icons in the colors of the UI theme now active."""
        for name, act in self.activity_buttons.items():
            if name.startswith("plugin:"):
                continue  # PluginPanelsMixin.refresh_plugin_icons
            if name == "Plugins":
                self._plugin_updates(act.property("updates") or 0)
            else:
                act.setIcon(icon(act.property("ewald_icon")))
        self.activity_more.setIcon(icon("more"))
        self.refresh_plugin_icons()

    def _plugin_updates(self, count):
        """A count badge on the Plugins icon while installed plugins have newer catalog versions."""
        from .icons import badged

        act = self.activity_buttons["Plugins"]
        act.setIcon(badged("plugins", count))
        act.setToolTip(f"Plugins: {count} update{'s' if count != 1 else ''} available" if count
                       else act.property("ewald_tip"))
        act.setProperty("updates", count)

    def _show_side_bar(self, dock):
        """Show an activity-bar side bar (Plugins, Remotes, AI chat, a plugin's panel) in a column of
        its own left of the notebook (Qt would otherwise stack a re-shown dock above it)."""
        placed = dock.isHidden() and not dock.isFloating() and self.notebook_dock.isVisible()
        if placed:
            self.splitDockWidget(dock, self.notebook_dock, QtCore.Qt.Horizontal)
        dock.show()
        dock.raise_()
        if placed:
            width = dock.property("ewald_width") or (380 if dock is self.ask_dock else 320)
            self.resizeDocks([dock], [int(width)], QtCore.Qt.Horizontal)

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
        self._place_minimal_exit()
        if on:
            self.minimal_exit.raise_()
            self.toast(native_text("Minimal mode: Ctrl+Shift+M or the Menus button at the bottom left brings the "
                                   "menus back; tap Alt for the menu bar, Ctrl+K for commands"))
        else:
            self.minimal_note.hide()

    def _build_minimal_exit(self):
        b = self.minimal_exit = QtWidgets.QToolButton(self, objectName="minimalExit", autoRaise=True,
                                                      toolTip=native_text("Leave minimal mode (Ctrl+Shift+M)"))
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
        # the status bar's place, hidden in minimal mode: kept free so the button covers nothing
        # (it hid the notebook's + Code)
        self.setContentsMargins(0, 0, 0, b.height() + 8 if not b.isHidden() else 0)
        b.move(10, self.height() - b.height() - 4)
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

        button("Open", "open", self.open_dialog, native_text("Open a data file, notebook (.py) or session (Ctrl+O)"))
        button("Save", "save", lambda: self.save_notebook(), native_text("Save the notebook (Ctrl+S)"), False)
        # Run all, Restart and Interrupt live in the notebook's own bar (and Notebook › Run all, Ctrl+Shift+Enter)
        tb.addSeparator()
        self.ws_buttons = []
        for i, (name, icon_name) in enumerate((("Notebook", "analyze"), ("Render", "render"))):
            b = button(name, icon_name, lambda _=False, i=i: self.set_workspace(i), f"{name} workspace (Ctrl+{i + 1})",
                       checkable=True)
            self.ws_buttons.append(b)
        self.ws_buttons[0].setChecked(True)
        tb.addSeparator()
        # the search box in the middle of the window, the palette opens right below it (D-entry in decisions.md)
        self.command_center = CommandCenter(self.command_palette)
        tb.addWidget(self.command_center)

    def set_button_style(self, style: str):
        """Toolbar and tool rail buttons as icons with text, icons or text
        (Preferences › General › Toolbar buttons); menus show icons unless 'Text'."""
        from .icons import button_style

        tabs = [t.viewport for t in getattr(self, "view_tabs", {}).values()]
        for b, text_in_hybrid in self.styled_buttons + [x for v in [self.viewport, *tabs] for x in v.styled_buttons]:
            button_style(b, style, text_in_hybrid)
        for v in [self.viewport, *tabs]:
            v.set_button_style(style)
        QtWidgets.QApplication.instance().setAttribute(QtCore.Qt.AA_DontShowIconsInMenus, style == "Text")

    def _build_menus(self):
        """The menu bar. Grouped by role (2026-10-23 menu review, docs/developer/decisions.md):
        File opens and exports, Edit edits cells, View is the camera and what is drawn, Insert adds
        cells, Notebook runs them, Tools are the 3D view tools, Window shows side bars and tabs."""
        mb = self.menuBar()
        f = mb.addMenu("&File")
        new = f.addAction("&New notebook", lambda: self.panel.new_notebook())
        new.setShortcuts([QtGui.QKeySequence(QtGui.QKeySequence.New), QtGui.QKeySequence("Ctrl+T")])
        f.addAction(icon("open"), "&Open…", QtGui.QKeySequence.Open, self.open_dialog)
        self.recent_menu = f.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        ex = f.addMenu(icon("example"), "Open e&xample")
        for example in examples.EXAMPLES:
            act = ex.addAction(f"{example.file}  ({example.format})", lambda e=example: self.open_example(e.file))
            act.setToolTip(example.description)
            act.setStatusTip(example.description)
        ex.setToolTipsVisible(True)
        src = f.addMenu("Open &from")
        src.addAction(icon("remote"), "&Remote file (SSH)…", QtGui.QKeySequence("Ctrl+Alt+O"), self.open_remote_dialog)
        src.addAction("&Location…", QtGui.QKeySequence("Ctrl+L"), self.open_location_dialog).setStatusTip(
            "Open a path or address a reader understands: host:/path over SSH, or a plugin's "
            "scheme such as aiida:1234 or simagent:1234")
        src.addAction("&NEB folder…", self.open_neb_dialog)
        src.addAction("&Workflow folder…", self.open_workflow_dialog).setStatusTip(
            "A folder of calculations (relax, NEB, MD restarts, phonopy) as one notebook, a load() per step")
        f.addSeparator()
        f.addAction(icon("save"), "&Save notebook", QtGui.QKeySequence.Save, self.save_notebook)
        f.addAction("Save notebook &as…", lambda: self.save_notebook(ask=True), QtGui.QKeySequence.SaveAs)
        f.addAction("&Close notebook", lambda: self.panel.close_tab(self.panel.tabs.currentIndex()), "Ctrl+W")
        f.addSeparator()
        exp = f.addMenu("&Export")
        exp.addAction(icon("image"), "&Image…", QtGui.QKeySequence("Ctrl+Shift+R"), self.render_image)
        exp.addAction("&Data…", self.export_dialog, "Ctrl+E")
        exp.addAction("&Movie…", self.export_movie)
        exp.addSeparator()   # the older single-pipeline file; notebooks and .ewd replace it
        exp.addAction("&Session (.ewald)…", lambda: self.save_session(ask=True))
        f.addAction("&Import session (.ewald)…", self.load_session_dialog)
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
        # cells, with Jupyter's command-mode keys (Esc first); Ctrl+X/C/V stay with the text editors
        for label, action, keys in (("Cu&t cells", "cut", "X"), ("&Copy cells", "copy", "C"),
                                    ("&Paste cells below", "paste:below", "V"),
                                    ("Paste cells abo&ve", "paste:above", "Shift+V"),
                                    ("&Delete cells", "delete", "D, D"), ("Undo cell de&letion", "undelete", "Z"),
                                    ("&Merge cells", "merge", "Shift+M")):
            e.addAction(label, lambda a=action: self._cell_action(a)).setToolTip(f"Esc, then {keys}")
        e.setToolTipsVisible(True)
        e.addSeparator()
        act = e.addAction("Pr&eferences…", self.preferences, QtGui.QKeySequence.Preferences)
        act.setMenuRole(QtGui.QAction.PreferencesRole)  # app menu on macOS, like OVITO/VESTA
        if act.shortcut().isEmpty():
            act.setShortcut("Ctrl+,")
        self.edit_menu = e

        v = mb.addMenu("&View")
        look = v.addMenu("&Look along")
        for ax in ("a", "b", "c", "a*", "b*", "c*"):
            look.addAction(ax, lambda ax=ax: self.viewport.view_along(ax))
        look.addSeparator()
        for label, d in (("&Top (+z)", "z"), ("&Front (-y)", "front"), ("&Right (+x)", "x"), ("&Isometric", "iso")):
            look.addAction(label, lambda d=d: self.viewport.view_direction(d))
        self.ortho_action = v.addAction("&Orthographic")   # also the view toolbar's Ortho and the right-click menu
        self.ortho_action.setCheckable(True)
        self.ortho_action.toggled.connect(lambda on: self.viewport.ortho_button.setChecked(on))
        v.aboutToShow.connect(lambda: self.ortho_action.setChecked(self.viewport.ortho))
        v.addAction("&Zoom to fit", self.viewport.zoom_fit, "Ctrl+0")
        show = v.addMenu("&Show")
        show.aboutToShow.connect(lambda: self.fill_show_menu(show))
        self.fill_show_menu(show)
        rep = v.addMenu("Re&plicate cell")  # display only: n x n x m copies (view(pipe, replicate=...))
        rep.aboutToShow.connect(lambda: self.fill_replicate_menu(rep))
        self.fill_replicate_menu(rep)
        self.theme_menu = v.addMenu("Structure t&heme")   # element colors, background, radii; not the UI theme
        self.theme_menu.aboutToShow.connect(self._fill_theme_menu)
        self._fill_theme_menu()
        v.addSeparator()
        lay = v.addMenu("L&ayout")  # tabs drag between groups; these are the keyboard and menu way
        for act in self.workbench.actions():
            lay.addAction(act)
        lay.addSeparator()
        lay.addAction("Fold all View cards", lambda: self.display.fold_all(True)).setToolTip(
            "Fold every card of the View tab (right-click a card for the same); kept for the next start")
        lay.addAction("Unfold all View cards", lambda: self.display.fold_all(False))
        lay.setToolTipsVisible(True)
        lay.aboutToShow.connect(self.workbench.update_actions)
        lay.aboutToHide.connect(self.workbench.enable_actions)
        v.addAction("Focus on the structure &view", self.toggle_focus, "Ctrl+\\")
        v.addAction("&Minimal mode (menus hidden)", lambda: self.set_minimal(not getattr(self, "minimal", False)),
                    "Ctrl+Shift+M")
        self.full_screen_action = v.addAction("F&ull screen", self.toggle_full_screen)
        self.full_screen_action.setCheckable(True)
        keys = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.FullScreen)
        self.full_screen_action.setShortcuts(keys or [QtGui.QKeySequence("F11")])   # F11; Ctrl+Cmd+F on macOS
        v.aboutToShow.connect(lambda: self.full_screen_action.setChecked(self.isFullScreen()))

        n = mb.addMenu("&Notebook")
        n.addAction("Run current &cell", lambda: self._cell_action("run")).setToolTip(
            native_text("Run the current cell (Ctrl+Enter; Shift+Enter also moves to the next cell)"))
        n.addAction("Run current cell and &below", lambda: self._cell_action("below"))
        n.addAction("Run &all", lambda: self._on_view("run_all"), "Ctrl+Shift+Return").setToolTip(
            "Restart and run every cell from the top")
        n.addAction("&Interrupt", lambda: self._on_view("interrupt")).setToolTip("Stop the running cell")
        n.addSeparator()
        n.addAction("&Restart", lambda: self._on_view("restart")).setToolTip(
            "Clear the namespace; cells keep their code and outputs")
        n.addAction("Restart && cl&ear outputs", lambda: self._on_view("restart", True))
        n.setToolTipsVisible(True)
        self.kernel_menu = n.addMenu("&Kernel")
        self.notebook_menu = n
        self.kernel_menu.aboutToShow.connect(lambda: self._on_view("fill_kernel_menu", self.kernel_menu))
        n.aboutToShow.connect(self._update_notebook_menu)
        n.addSeparator()
        n.addAction("Move cell &up", lambda: self._cell_action("up"), "Ctrl+Shift+Up")
        n.addAction("Move cell &down", lambda: self._cell_action("down"), "Ctrl+Shift+Down")
        n.addAction("Change cell to c&ode", lambda: self._cell_action("kind:code"), "Ctrl+M, Y").setToolTip(
            "Turn a text cell into a code cell (Esc, then Y, like Jupyter)")
        n.addAction("Change cell to &text", lambda: self._cell_action("kind:markdown"), "Ctrl+M, M").setToolTip(
            "Turn a code cell into a Markdown text cell (Esc, then M); its steps leave the pipeline")
        n.addSeparator()
        n.addAction("C&lear cell output", lambda: self._cell_action("clear"), "Ctrl+Shift+Backspace")
        n.addAction("Clear all output&s", lambda: self._on_view("clear_outputs"),
                    "Ctrl+Alt+Shift+Backspace").setToolTip("Remove every cell's output; the next save leaves them out")
        n.addSeparator()
        self.trust_action = n.addAction("Tru&st notebook…", lambda: self._on_view("trust"))
        self.trust_action.setToolTip("Allow a notebook received as an .ewd file to run its code and plugins")
        n.addAction("Show as Python &file", lambda: self.panel.view is not None and self.panel.view.py_button.toggle())

        ins = mb.addMenu("&Insert")
        ins.addAction("&Code cell", lambda: self._notebook_view().add_cell("code"), "Ctrl+Shift+N")
        ins.addAction("&Text cell", lambda: self._notebook_view().add_cell("markdown"))
        ins.addAction("Cell &above", lambda: self._cell_action("above")).setToolTip("Esc, then A")
        ins.addAction("Cell &below", lambda: self._cell_action("after")).setToolTip("Esc, then B")
        ins.addSeparator()
        m = ins.addMenu("&Step")
        m.aboutToShow.connect(lambda: self._fill_insert_step_menu(m))
        fill_modifier_menu(m, self.panel.add_modifier)
        self.plot_menu = vm = ins.addMenu("&Plot")   # view cells: show(pipe, table=...) under the cell
        vm.aboutToShow.connect(lambda: fill_view_menu(vm, self.data, lambda **kw: self._notebook_view().add_view(**kw)))
        am = ins.addMenu("&Quick analysis")
        from ..verbs import VERB_CELLS

        for verb, label, _code in VERB_CELLS:
            act = am.addAction(label, lambda v=verb: self._insert_verb(v))
            act.setToolTip(f"A {verb}(pipe) cell that shows its table, run at once")
            act.setData(verb)
        am.setToolTipsVisible(True)
        am.aboutToShow.connect(self._update_analysis_menu)
        self.analysis_menu = am
        ins.addSeparator()
        ins.addAction("S&napshot of the structure view", self._insert_snapshot).setToolTip(
            "A cell with a picture of the structure view as it is now (frame, camera, display style)")
        ins.addAction("&Write view settings to notebook", self.write_camera).setToolTip(
            "Write the display style, the camera and the frame into the notebook's view(pipe, ...) cell")
        ins.setToolTipsVisible(True)
        self.step_menu = m

        t = self.tools_menu = mb.addMenu("&Tools")
        tools = QtGui.QActionGroup(t)   # one viewport tool at a time, like the tool rail
        names = {"orbit": "&Orbit, pan and zoom", "pick": "&Pick atom", "distance": "Measure &distance",
                 "angle": "Measure &angle", "bond": "Edit &bonds"}
        for key, _icon, tip in Viewport.TOOLS:
            act = t.addAction(names.get(key, tip.split(":")[0]), lambda k=key: self.viewport.set_tool(k))
            act.setToolTip(tip)
            act.setCheckable(True)
            act.setData(key)
            tools.addAction(act)
        t.aboutToShow.connect(self._update_tools_menu)
        t.addAction("&Clear picks and measurements", self.viewport.clear_picks).setToolTip(
            "Also Esc or a double click on empty space in the structure view")
        t.addSeparator()
        t.addAction("Command &palette", self.command_palette, "Ctrl+K")
        t.setToolTipsVisible(True)

        w = self.window_menu = mb.addMenu("&Window")
        w.addAction("&Notebook workspace", lambda: self.set_workspace(0), "Ctrl+1")
        w.addAction("&Render workspace", lambda: self.set_workspace(1), "Ctrl+2")
        w.addSeparator()   # side bars in the activity bar's order, ticked while shown;
        # plugin side bars go in before Remotes
        self.side_bar_actions = []

        def side_bar(label, dock, shortcut=None, show=None):
            act = w.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda on: (show or (lambda: self._show_side_bar(dock)))() if on else dock.hide())
            if shortcut:
                act.setShortcut(shortcut)
            act.setData(dock)
            self.side_bar_actions.append(act)
            return act

        side_bar("Note&book", self.notebook_dock, show=lambda: (self.notebook_dock.show(), self.notebook_dock.raise_()))
        side_bar("&AI chat", self.ask_dock, "Ctrl+J", self.ask_claude).setToolTip(
            "Chat with AI about the notebook, the structure or the picked atoms")
        self.remotes_action = side_bar("Re&motes", self.remotes_dock, "Ctrl+Alt+R", self.show_remotes)
        side_bar("Pl&ugins", self.plugins_dock, "Ctrl+Shift+X")
        w.aboutToShow.connect(self._update_window_menu)
        w.addSeparator()   # tabs of the editor area and the panel
        w.addAction("&Structure", lambda: self.workbench.show(self.viewport))
        w.addAction("New st&ructure tab", lambda: self.new_view_tab()).setToolTip(
            "Another 3D view of the structure (its own camera and style), written as a view(pipe, tab=...) cell")
        w.addAction("&View settings", lambda: self.workbench.show(self.display))
        w.addAction("&Python console", lambda: self.workbench.show(self.console))
        w.addAction("&Data inspector", lambda: self.workbench.show(self.inspector))
        w.addAction("P&lots", lambda: self.workbench.show(self.plots))
        w.addAction("New plot &tab", lambda: self.new_plot_view())
        w.addAction("L&og", lambda: self.workbench.show(self.log))
        w.setToolTipsVisible(True)

        h = mb.addMenu("&Help")
        for label, page in manual.PAGES.items():
            h.addAction(label, lambda page=page: self.open_help(page))
        h.addSeparator()
        h.addAction("&Keyboard shortcuts", self.show_shortcuts)
        h.addSeparator()
        about = h.addAction("&About EWALD", self.about)
        about.setMenuRole(QtGui.QAction.AboutRole)

    def _fill_recent_menu(self):
        """File › Open recent: the files the welcome page lists, newest first."""
        from .welcome import clear_recent, load_recent

        menu = self.recent_menu
        menu.clear()
        recent = load_recent()
        for path in recent:
            act = menu.addAction(os.path.basename(path) or path, lambda p=path: self.open_any(p))
            act.setStatusTip(path)
            act.setToolTip(path)
        if not recent:
            menu.addAction("No recent files").setEnabled(False)
        else:
            menu.addSeparator()
            menu.addAction("Clear recent files", lambda: (clear_recent(), self.panel.welcome.refresh()))
        menu.setToolTipsVisible(True)

    def _fill_insert_step_menu(self, menu):
        """Insert › Step: the same entries as the notebook's + Step, plugin steps under Plugins."""
        fill_modifier_menu(menu, self.panel.add_modifier, self._step_context)
        view = self.panel.view
        if view is not None:
            view.fill_plugin_steps(menu)
        else:
            menu.addSeparator()
            menu.addMenu("Plugins").addAction("Start or open a notebook first").setEnabled(False)

    def _update_window_menu(self):
        """Tick the side bars that are shown."""
        for act in self.side_bar_actions:
            act.setChecked(not act.data().isHidden())

    def toggle_full_screen(self, on: bool | None = None):
        """View › Full screen (F11; Ctrl+Cmd+F on macOS)."""
        on = not self.isFullScreen() if on is None else on
        self.showFullScreen() if on else self.showNormal()

    def show_shortcuts(self):
        """Help › Keyboard shortcuts: every menu shortcut and the notebook's command-mode keys."""
        from .shortcuts import ShortcutsDialog

        dlg = ShortcutsDialog(self)
        dlg.show()
        return dlg

    def _update_tools_menu(self):
        """Tick the viewport tool in use (the tool rail may have changed it)."""
        for act in self.tools_menu.actions():
            act.setChecked(act.data() == self.viewport.tool)

    def _notebook_view(self):
        """The current notebook's view; a new notebook when none is open (the welcome page)."""
        return self.panel.view or self.panel.new_notebook()

    def _update_notebook_menu(self):
        """With no notebook open, Notebook › ... greys out (the welcome page offers New notebook)."""
        view = self.panel.view
        for act in self.notebook_menu.actions():
            act.setEnabled(view is not None)
        self.trust_action.setEnabled(view is not None and view.notebook.restricted)

    def _on_view(self, method, *args):
        """Call a method of the current notebook view; nothing when no notebook is open."""
        view = self.panel.view
        if view is not None:
            getattr(view, method)(*args)

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
        """Insert › Quick analysis: a ``verb(pipe)`` cell after the current one, run at once (its table shows below)."""
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
        from .. import gpu_select

        gpus = gpu_select.list_gpus()
        if len(gpus) > 1:
            text += f"<br>GPUs in this computer: {', '.join(gpus)}"
        hint = gpu_select.other_gpu_hint(gl["renderer"] or "", gpus)
        if hint:
            text += f"<br><b>{hint.replace('`', '')}</b>"
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
            menu.aboutToShow.emit()  # fill the menus that fill themselves when opened (Kernel, Insert › Plot, ...)
        entries = menu_entries(self.menuBar())  # Insert › Step lists the steps
        for menu in self.menuBar().findChildren(QtWidgets.QMenu):
            menu.aboutToHide.emit()  # View › Layout enables its moves again for their shortcuts
        pal = CommandPalette(entries, self)
        box = self.command_center.box
        if box.isVisible():   # below the toolbar's search box
            pal.place_under(box, width=max(CommandPalette.WIDTH, box.width() + 80))
        else:                 # minimal mode: the toolbar is hidden
            geo = self.geometry()
            pal.move(geo.x() + (geo.width() - CommandPalette.WIDTH) // 2, geo.y() + 40)
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

    def edit_plugin_settings(self, name: str, exec_: bool = True):
        from .plugin_settings import edit

        dlg = edit(name, self, exec_=exec_)
        if exec_ and dlg.message:
            self.message(f"{dlg.page.title}: {dlg.message}")
        return dlg
