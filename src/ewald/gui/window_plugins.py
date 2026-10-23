"""Plugin contributions to the main window: one activity-bar button and side bar (or one tab
under the 3D view) per ``@panel`` (``ewald.plugins.panel``), built the first time it opens, the
entries of ``@menu_item`` plugins in the menu bar and the 3D view's right-click menu, and the
context both get."""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .icons import PATHS, icon
from .theme import TOKENS

log = logging.getLogger(__name__)

EVENTS = ("data", "pipeline", "settings")


def plugin_icon(spec: str) -> QtGui.QIcon:
    """The icon a plugin names: a built-in icon, an SVG/PNG file or SVG markup. An SVG drawn
    with ``currentColor`` gets the theme's colors (secondary, text on hover, accent when on)."""
    if spec in PATHS:
        return icon(spec)
    if spec.lstrip().startswith("<"):
        text = spec
    else:
        path = Path(spec)
        if not path.is_file():
            return icon("plugins")
        if path.suffix.lower() != ".svg":
            return QtGui.QIcon(str(path))
        text = path.read_text(encoding="utf-8")
    from PySide6 import QtSvg

    ic = QtGui.QIcon()
    for size in (16, 24, 32, 48):
        for color, mode, state in ((TOKENS["secondary"], QtGui.QIcon.Normal, QtGui.QIcon.Off),
                                   (TOKENS["text"], QtGui.QIcon.Active, QtGui.QIcon.Off),
                                   (TOKENS["accent"], QtGui.QIcon.Normal, QtGui.QIcon.On),
                                   (TOKENS["accent"], QtGui.QIcon.Active, QtGui.QIcon.On),
                                   (TOKENS["muted"], QtGui.QIcon.Disabled, QtGui.QIcon.Off)):
            pix = QtGui.QPixmap(size, size)
            pix.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pix)
            QtSvg.QSvgRenderer(QtCore.QByteArray(text.replace("currentColor", color).encode())).render(painter)
            painter.end()
            ic.addPixmap(pix, mode, state)
    return ic


def plugin_host(widget):
    """The main window ``widget`` sits in (through floating docks too), or None outside one."""
    while widget is not None:
        if hasattr(widget, "add_plugin_items"):
            return widget
        widget = widget.parentWidget()
    return None


class PanelContext:
    """What a plugin panel's ``factory(context)`` gets: the window's state and a few actions.

    ``window`` is the ``MainWindow`` itself, for what this does not cover; it is internal and
    may change in any release, the rest follows ``ewald.plugin_api.PLUGIN_API``."""

    def __init__(self, window, name: str):
        self.window = window
        self.name = name
        #: What a menu entry was opened on: the ``ewald.notebook.Cell`` of a cell's menu, the
        #: ``ewald.plugins.PlotTarget`` of a plot's; None elsewhere and for panels.
        self.target = None
        self._callbacks = {e: [] for e in EVENTS}

    @property
    def settings(self):
        """ewald's preferences (``ewald.settings.Settings``)."""
        return self.window.settings

    def plugin_settings(self, page: str) -> dict:
        """The values of a plugin settings page (``ewald.plugins.plugin_settings``)."""
        from .. import plugins

        return plugins.plugin_settings(page)

    @property
    def pipeline(self):
        """The pipeline of the notebook on screen (``pipe`` in its cells), or None."""
        return self.window.pipeline

    @property
    def data(self):
        """The computed frame the 3D view shows (a ``DataCollection``), or None."""
        return self.window.data

    @property
    def frame(self) -> int:
        return self.window.frame

    @property
    def picks(self) -> list[int]:
        """Indices of the atoms picked in the 3D view (empty when none are)."""
        return list(self.window.viewport.picks)

    def message(self, text: str, error: bool = False):
        """A line in the status bar and the Log."""
        self.window.message(f"{self.name}: {text}", error=error)

    def show_panel(self, name: str, on: bool = True):
        """Show (or with ``on=False`` hide) a ``@panel`` by its name, e.g. from a toolbar button."""
        if name not in self.window.plugin_panels:
            raise KeyError(f"no plugin panel {name!r}")
        return self.window.show_plugin_panel(name, on)

    def open_preferences(self, page: str | None = None):
        """Open Preferences, at ``page`` (e.g. the plugin's own settings page)."""
        return self.window.preferences(page)

    def insert_cell(self, code: str):
        """Add a code cell to the notebook on screen, after the current one, without running it."""
        view = self.window.panel.view or self.window.panel.new_notebook()
        view.add_cell("code", code)
        self.window.notebook_dock.show()

    def add_step(self, modifier):
        """Add ``modifier`` as a step cell of the notebook on screen (and compute)."""
        self.window.panel.add_modifier(modifier)

    def open(self, path: str):
        """Open ``path`` (a file, ``host:/path`` or a plugin scheme) as File › Open from › Location does."""
        return self.window.open_location_dialog(str(path))

    def open_workflow(self, workflow):
        """Open a :class:`ewald.workflow.Workflow` (or a path a workflow source describes) as one
        notebook, a variable per step."""
        return self.window.open_workflow(workflow)

    def on(self, event: str, callback):
        """Call ``callback`` on "data" (a frame was computed: ``callback(data)``), "pipeline"
        (another pipeline: ``callback(pipeline)``) or "settings" (Preferences applied, plugin
        pages included: ``callback(settings)``). Errors are logged, not raised into the window."""
        if event not in self._callbacks:
            raise ValueError(f"unknown event {event!r}; use one of {', '.join(EVENTS)}")
        self._callbacks[event].append(callback)
        return callback

    def _emit(self, event, value):
        for cb in list(self._callbacks[event]):
            try:
                cb(value)
            except Exception as exc:
                self.window.message(f"{self.name}: {event} callback failed: {type(exc).__name__}: {exc}", error=True)


class PluginPanelsMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    def _build_plugin_panels(self):
        self.plugin_panels = {}      # name -> {"action", "dock", "widget", "context", "plugin"}
        self.plugin_menu_entries = []    # (menu the action is in, action) of @menu_item plugins
        self._menu_contexts = {}     # menu item name -> PanelContext
        self.plugin_panel_actions = []   # Window entries of plugin panels, with the built-in side bars
        self.window_menu.aboutToShow.connect(self._update_plugin_panel_actions)
        self.workbench.viewClosed.connect(self._plugin_tab_closed)
        self.viewport.picked.connect(self._update_plugin_menu_items)   # when="picks" on the toolbar
        self.refresh_plugin_panels()

    def use_ui_theme(self, name: str):
        """Make ``name`` the UI theme (the Plugins side bar's Use theme), saved like Preferences does."""
        from . import preferences as prefs

        self.settings["general.ui_theme"] = name
        self.settings.save()
        prefs.apply_settings(self, self.settings)
        self.message(f"UI theme: {name}")

    def restart_app(self) -> bool:
        """Close the window (asking to save notebooks, as Quit does) and start ewald again with
        the saved notebooks open (Plugins side bar › Restart now). False when the close was
        cancelled."""
        from .. import plugin_catalog as pc

        if not self.close():
            return False
        paths = [v.notebook.path for v in self.panel.views() if v.notebook.path]
        cmd = pc.restart_command(dict.fromkeys(paths))
        started = QtCore.QProcess.startDetached(cmd[0], cmd[1:], os.getcwd())
        if not (started[0] if isinstance(started, tuple) else started):   # (ok, pid) in some PySide6
            log.warning("could not start %s", cmd)
        QtWidgets.QApplication.quit()
        return True

    def refresh_plugin_panels(self):
        """Match the window to the installed plugins: activity-bar buttons and bottom tabs for new
        panels, those of plugins switched off or removed dropped (their side bars close), and the
        menu entries of ``@menu_item`` plugins."""
        from .. import plugins

        try:
            found = plugins.panels()
        except Exception as exc:  # broken plugin loading must not take the window down
            log.warning("plugin panels: %s", exc)
            found = {}
        for name in [n for n, p in self.plugin_panels.items() if found.get(n) is not p["plugin"]]:
            entry = self.plugin_panels.pop(name)
            if entry["action"] is not None:
                self.activity_bar.removeAction(entry["action"])
                self.activity_buttons.pop(f"plugin:{name}", None)
                entry["action"].deleteLater()
            if entry["dock"] is not None:
                self.removeDockWidget(entry["dock"])
                entry["dock"].deleteLater()
            if entry["plugin"].area == "bottom" and f"plugin:{name}" in self.workbench.specs:
                self.workbench.unregister(f"plugin:{name}")
        for name, plugin in found.items():
            if name in self.plugin_panels:
                continue
            act = None
            if plugin.area == "side":
                act = QtGui.QAction(plugin_icon(plugin.icon), plugin.title, self.activity_bar, checkable=True)
                act.setToolTip(plugin.tooltip or plugin.description or f"Show or hide the {plugin.title} side bar")
                act.setProperty("ewald_plugin_icon", plugin.icon)
                act.triggered.connect(lambda on, n=name: self.show_plugin_panel(n, on))
                self.activity_bar.addAction(act)
                self.activity_buttons[f"plugin:{name}"] = act
            self.plugin_panels[name] = {"action": act, "dock": None, "widget": None, "context": None,
                                        "plugin": plugin}
        menu = self.window_menu
        for a in self.plugin_panel_actions:
            menu.removeAction(a)
            a.deleteLater()
        self.plugin_panel_actions = []
        for name, entry in self.plugin_panels.items():   # side bars between Notebook and Remotes
            plugin = entry["plugin"]
            a = QtGui.QAction(plugin_icon(plugin.icon), plugin.title, menu)
            a.setData(name)
            if plugin.area == "side":
                a.setCheckable(True)
                a.triggered.connect(lambda on, n=name: self.show_plugin_panel(n, on))
            else:
                a.triggered.connect(lambda _=False, n=name: self.show_plugin_panel(n))
            if plugin.shortcut:
                a.setShortcut(QtGui.QKeySequence(plugin.shortcut))
            menu.insertAction(self.remotes_action, a)
            self.plugin_panel_actions.append(a)
        self.refresh_plugin_menu_items()

    def _update_plugin_panel_actions(self):
        """Tick the plugin side bars that are shown (Window menu)."""
        for a in self.plugin_panel_actions:
            dock = self.plugin_panels.get(a.data(), {}).get("dock")
            a.setChecked(a.isCheckable() and dock is not None and not dock.isHidden())

    def show_plugin_panel(self, name: str, on: bool = True):
        """Show (``on``) or hide a plugin's side bar or bottom tab; the first show builds its
        widget. Returns the dock (side bar) or the widget (bottom tab)."""
        entry = self.plugin_panels[name]
        if entry["plugin"].area == "bottom":
            view = f"plugin:{name}"
            if not on:
                if view in self.workbench.specs:
                    self.workbench.close_view(view)
                return None
            self._build_plugin_tab(name)
            self.workbench.show(view)
            return entry["widget"]
        dock = entry["dock"]
        if not on:
            if dock is not None:
                dock.hide()
            return dock
        if dock is None:
            plugin = entry["plugin"]
            dock = entry["dock"] = self._dock(self._plugin_widget(entry), plugin.title,
                                              QtCore.Qt.LeftDockWidgetArea, None)
            dock.setObjectName(f"plugin:{name}")
            dock.setProperty("ewald_width", plugin.width)
            act = entry["action"]
            # where the saved layout had it (place and width), if it was open when the app last closed;
            # only then: Qt reports a restore for docks it never saw, too
            restored = name in getattr(self, "_saved_plugin_docks", ()) and self.restoreDockWidget(dock)
            if not restored:
                self.splitDockWidget(dock, self.notebook_dock, QtCore.Qt.Horizontal)
                dock.hide()
            dock.visibilityChanged.connect(lambda _shown, a=act, d=dock: a.setChecked(not d.isHidden()))
            if restored:
                dock.show()
                dock.raise_()
                act.setChecked(True)
                return dock
        self._show_side_bar(dock)
        return dock

    def _plugin_widget(self, entry):
        """Build a panel's widget, or a read-only traceback in its place when the plugin fails."""
        plugin = entry["plugin"]
        ctx = entry["context"] = PanelContext(self, plugin.title)
        try:
            widget = plugin.factory(ctx)
            if not isinstance(widget, QtWidgets.QWidget):
                raise TypeError(f"returned {type(widget).__name__}, not a QWidget")
        except Exception as exc:
            self.message(f"Plugin panel {plugin.title} failed: {type(exc).__name__}: {exc}", error=True)
            widget = QtWidgets.QPlainTextEdit(readOnly=True, objectName="pluginPanelError")
            widget.setPlainText(f"The panel {plugin.title!r} ({plugin.origin}) could not be built:\n\n"
                                + traceback.format_exc())
        entry["widget"] = widget
        return widget

    def _build_plugin_tab(self, name: str) -> bool:
        """Register a bottom-tab panel with the workbench (built once; a closed tab is rebuilt)."""
        entry = self.plugin_panels.get(name)
        if entry is None or entry["plugin"].area != "bottom":
            return False
        view = f"plugin:{name}"
        if view not in self.workbench.specs:
            self.workbench.register(view, self._plugin_widget(entry), entry["plugin"].title, closable=True)
        return True

    def _plugin_tab_closed(self, view: str):
        entry = self.plugin_panels.get(view[len("plugin:"):]) if view.startswith("plugin:") else None
        if entry is not None:
            entry["widget"] = entry["context"] = None     # the workbench deleted the widget

    def open_plugin_panels(self) -> list[str]:
        """Names of the plugin side bars open now (saved with the window layout)."""
        return [n for n, e in getattr(self, "plugin_panels", {}).items()
                if e["dock"] is not None and not e["dock"].isHidden()]

    # ------------------------------------------------------------------ menu items
    def _menu_bar_menu(self, title: str):
        for act in self.menuBar().actions():
            if act.menu() is not None and act.text().replace("&", "") == title:
                return act.menu()
        return None

    def refresh_plugin_menu_items(self):
        """Put the entries of ``@menu_item`` plugins into the menu bar and the toolbar (right-click
        menus get theirs each time they open, :meth:`add_plugin_items`)."""
        from .. import plugins

        for holder, act in self.plugin_menu_entries:
            holder.removeAction(act)
            act.deleteLater()
        self.plugin_menu_entries = []
        try:
            found = plugins.menu_items()
        except Exception as exc:
            log.warning("plugin menu items: %s", exc)
            found = {}
        self._menu_contexts = {n: c for n, c in self._menu_contexts.items() if n in found}
        separated = set()
        toolbar = self.findChild(QtWidgets.QToolBar, "main-toolbar")
        tool_menus = {}
        for name, item in found.items():
            top, *subs = item.menu.split("/")
            if top == "Toolbar":
                if toolbar is None:
                    continue
                if not subs:
                    act = self._plugin_menu_action(name, item, toolbar)
                    toolbar.addAction(act)
                    self.plugin_menu_entries.append((toolbar, act))
                    continue
                menu = tool_menus.get(subs[0])
                if menu is None:       # a drop-down button named after the first part of the path
                    menu = tool_menus[subs[0]] = QtWidgets.QMenu(subs[0], toolbar)
                    button = QtWidgets.QToolButton(text=subs[0], toolTip=subs[0], objectName="pluginToolButton")
                    button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
                    button.setMenu(menu)
                    menu.aboutToShow.connect(self._update_plugin_menu_items)
                    self.plugin_menu_entries.append((toolbar, toolbar.addWidget(button)))
                subs = subs[1:]
            else:
                menu = self._menu_bar_menu(top)
                if menu is None:      # a right-click menu, or a menu this window does not have
                    continue
                if top not in separated:
                    separated.add(top)
                    self.plugin_menu_entries.append((menu, menu.addSeparator()))
                    menu.aboutToShow.connect(self._update_plugin_menu_items)
            for title in subs:
                found_sub = next((a.menu() for a in menu.actions()
                                  if a.menu() is not None and a.text() == title), None)
                if found_sub is None:
                    found_sub = QtWidgets.QMenu(title, menu)
                    self.plugin_menu_entries.append((menu, menu.addMenu(found_sub)))
                menu = found_sub
            act = self._plugin_menu_action(name, item, menu)
            menu.addAction(act)
            self.plugin_menu_entries.append((menu, act))
        self._update_plugin_menu_items()

    def _plugin_menu_action(self, name, item, parent, target=None):
        act = QtGui.QAction(item.title, parent)
        if item.icon:
            act.setIcon(plugin_icon(item.icon))
        tip = item.description or item.title
        act.setToolTip(tip)
        act.setStatusTip(item.description)
        if item.shortcut:
            act.setShortcut(QtGui.QKeySequence(item.shortcut))
        act.setProperty("ewald_menu_item", name)
        act.triggered.connect(lambda _=False, n=name, t=target: self.run_plugin_menu_item(n, t))
        return act

    def _plugin_item_ready(self, item) -> bool:
        if item.when == "data":
            return self.data is not None
        if item.when == "picks":
            return bool(self.viewport.picks)
        return True

    def _update_plugin_menu_items(self, *_):
        """Grey out the menu-bar and toolbar entries whose ``when`` does not hold now."""
        from .. import plugins

        items = plugins.menu_items() if self.plugin_menu_entries else {}
        for _holder, act in self.plugin_menu_entries:
            item = items.get(act.property("ewald_menu_item"))
            if item is not None:
                act.setEnabled(self._plugin_item_ready(item))

    def add_plugin_items(self, menu: QtWidgets.QMenu, where: str, target=None, everything: bool = False) -> list:
        """Append the ``@menu_item`` entries for the right-click menu ``where`` ("3D view",
        "Notebook cell", "Plot") to ``menu``, those whose ``when`` holds (all with ``everything``),
        then a separator; ``target`` becomes ``context.target``. Returns the actions added."""
        from .. import plugins

        try:
            items = [(n, i) for n, i in plugins.menu_items().items() if i.menu.split("/")[0] == where]
        except Exception:
            return []
        items = [(n, i) for n, i in items if everything or self._plugin_item_ready(i)]
        if not items:
            return []
        added, subs = [], {}
        for name, item in items:
            target_menu = menu
            for title in item.menu.split("/")[1:]:
                key = (id(target_menu), title)
                if key not in subs:
                    subs[key] = target_menu.addMenu(title)
                    if target_menu is menu:
                        added.append(subs[key].menuAction())
                target_menu = subs[key]
            act = self._plugin_menu_action(name, item, target_menu, target)
            target_menu.addAction(act)
            if target_menu is menu:
                added.append(act)
        added.append(menu.addSeparator())
        return added

    def add_plugin_viewport_items(self, menu: QtWidgets.QMenu, everything: bool = False):
        """The "3D view" entries of ``@menu_item`` plugins, for the viewport's right-click menu."""
        return self.add_plugin_items(menu, "3D view", everything=everything)

    def run_plugin_menu_item(self, name: str, target=None):
        """Call a ``@menu_item`` plugin's function (``context.target`` = ``target``); errors go to
        the status bar and the Log."""
        from .. import plugins

        item = plugins.menu_items().get(name)
        if item is None:
            return None
        ctx = self._menu_contexts.get(name)
        if ctx is None:
            ctx = self._menu_contexts[name] = PanelContext(self, item.title)
        ctx.target = target
        try:
            return item.callback(ctx)
        except Exception as exc:
            self.message(f"{item.title} failed: {type(exc).__name__}: {exc}", error=True)
            log.debug("menu item %s", name, exc_info=True)
            return None
        finally:
            ctx.target = None

    def _plugin_event(self, event: str, value):
        """Tell every built plugin panel (and menu item context) about ``event`` (see :meth:`PanelContext.on`)."""
        if event == "data" and getattr(self, "plugin_menu_entries", None):
            self._update_plugin_menu_items()
        contexts = [e["context"] for e in getattr(self, "plugin_panels", {}).values()]
        for ctx in contexts + list(getattr(self, "_menu_contexts", {}).values()):
            if ctx is not None:
                ctx._emit(event, value)

    def refresh_plugin_icons(self):
        for entry in getattr(self, "plugin_panels", {}).values():
            if entry["action"] is not None:
                entry["action"].setIcon(plugin_icon(entry["plugin"].icon))
        if getattr(self, "plugin_menu_entries", None):   # toolbar and menu entries take the new theme too
            from .. import plugins

            items = plugins.menu_items()
            for _holder, act in self.plugin_menu_entries:
                item = items.get(act.property("ewald_menu_item"))
                if item is not None and item.icon:
                    act.setIcon(plugin_icon(item.icon))
