"""Plugin panels in the main window: one activity-bar button and side bar per ``@panel``
(``ewald.plugins.panel``), built the first time it opens, and the context they get."""
from __future__ import annotations

import logging
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


class PanelContext:
    """What a plugin panel's ``factory(context)`` gets: the window's state and a few actions.

    ``window`` is the ``MainWindow`` itself, for what this does not cover; it is internal and
    may change in any release, the rest follows ``ewald.plugin_api.PLUGIN_API``."""

    def __init__(self, window, name: str):
        self.window = window
        self.name = name
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

    def message(self, text: str, error: bool = False):
        """A line in the status bar and the Log."""
        self.window.message(f"{self.name}: {text}", error=error)

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
        self.plugin_panels = {}      # name -> {"action", "dock", "context", "plugin"}
        self.plugin_panel_menu = self.window_menu.addMenu("Plugin panels")
        self.refresh_plugin_panels()

    def refresh_plugin_panels(self):
        """Match the activity bar to the installed panels: add buttons for new ones, drop those
        of plugins switched off or removed (their side bars close)."""
        from .. import plugins

        try:
            found = plugins.panels()
        except Exception as exc:  # broken plugin loading must not take the window down
            log.warning("plugin panels: %s", exc)
            found = {}
        for name in [n for n, p in self.plugin_panels.items() if found.get(n) is not p["plugin"]]:
            entry = self.plugin_panels.pop(name)
            self.activity_bar.removeAction(entry["action"])
            self.activity_buttons.pop(f"plugin:{name}", None)
            entry["action"].deleteLater()
            if entry["dock"] is not None:
                self.removeDockWidget(entry["dock"])
                entry["dock"].deleteLater()
        for name, plugin in found.items():
            if name in self.plugin_panels:
                continue
            act = QtGui.QAction(plugin_icon(plugin.icon), plugin.title, self.activity_bar, checkable=True)
            act.setToolTip(plugin.tooltip or plugin.description or f"Show or hide the {plugin.title} side bar")
            act.setProperty("ewald_plugin_icon", plugin.icon)
            act.triggered.connect(lambda on, n=name: self.show_plugin_panel(n, on))
            self.activity_bar.addAction(act)
            self.activity_buttons[f"plugin:{name}"] = act
            self.plugin_panels[name] = {"action": act, "dock": None, "context": None, "plugin": plugin}
        menu = self.plugin_panel_menu
        menu.clear()
        for name, entry in self.plugin_panels.items():
            plugin = entry["plugin"]
            a = menu.addAction(plugin.title, lambda n=name: self.show_plugin_panel(n))
            if plugin.shortcut:
                a.setShortcut(QtGui.QKeySequence(plugin.shortcut))
        menu.menuAction().setVisible(bool(self.plugin_panels))

    def show_plugin_panel(self, name: str, on: bool = True):
        """Show (``on``) or hide a plugin's side bar; the first show builds its widget."""
        entry = self.plugin_panels[name]
        dock = entry["dock"]
        if not on:
            if dock is not None:
                dock.hide()
            return dock
        if dock is None:
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
            dock = entry["dock"] = self._dock(widget, plugin.title, QtCore.Qt.LeftDockWidgetArea, None)
            dock.setObjectName(f"plugin:{name}")
            dock.setProperty("ewald_width", plugin.width)
            self.splitDockWidget(dock, self.notebook_dock, QtCore.Qt.Horizontal)
            dock.hide()
            act = entry["action"]
            dock.visibilityChanged.connect(lambda _shown, a=act, d=dock: a.setChecked(not d.isHidden()))
        self._show_side_bar(dock)
        return dock

    def _plugin_event(self, event: str, value):
        """Tell every built plugin panel about ``event`` (see :meth:`PanelContext.on`)."""
        for entry in getattr(self, "plugin_panels", {}).values():
            if entry["context"] is not None:
                entry["context"]._emit(event, value)

    def refresh_plugin_icons(self):
        for entry in getattr(self, "plugin_panels", {}).values():
            entry["action"].setIcon(plugin_icon(entry["plugin"].icon))
