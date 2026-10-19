"""Plugins side bar, like VS Code's Extensions view: search the installed plugins and the
catalogs (:mod:`ewald.plugin_catalog`), install, remove, and add a plugin step to the notebook."""
from __future__ import annotations

import html

from PySide6 import QtCore, QtGui, QtWidgets

from .. import plugin_catalog as pc
from .. import plugins
from .theme import TOKENS


class _CatalogLoader(QtCore.QThread):
    """Reads the catalogs off the GUI thread (indexes may be URLs)."""

    loaded = QtCore.Signal(list, dict)

    def __init__(self, urls, parent=None):
        super().__init__(parent)
        self.urls = urls

    def run(self):
        entries, errors = pc.catalog(self.urls)
        self.loaded.emit(entries, errors)


class PluginRow(QtWidgets.QFrame):
    """One plugin: name, summary, kinds and its Install or Remove button."""

    clicked = QtCore.Signal(object)
    action = QtCore.Signal(object, str)   # item, "install" | "remove"

    def __init__(self, item, installed: bool, parent=None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("pluginRow")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"<b>{html.escape(item.name)}</b>" +
                                 (f" <span style='color:{TOKENS['muted']}'>{html.escape(item.version)}</span>"
                                  if getattr(item, "version", "") else ""))
        head.addWidget(title, 1)
        if installed and isinstance(item, pc.Installed):
            if item.removable:
                b = QtWidgets.QPushButton("Remove", objectName="pluginButton")
                b.clicked.connect(lambda: self.action.emit(self.item, "remove"))
                head.addWidget(b)
        elif not installed:
            b = QtWidgets.QPushButton("Install", objectName="pluginInstall")
            b.clicked.connect(lambda: self.action.emit(self.item, "install"))
            head.addWidget(b)
        else:
            head.addWidget(QtWidgets.QLabel("installed", objectName="muted"))
        lay.addLayout(head)
        summary = getattr(item, "summary", "") or _provides_text(item)
        if summary:
            s = QtWidgets.QLabel(summary, objectName="pluginSummary", wordWrap=True)
            lay.addWidget(s)
        kinds = QtWidgets.QLabel("  ".join(item.kinds), objectName="pluginKinds")
        lay.addWidget(kinds)
        if getattr(item, "error", ""):
            lay.addWidget(QtWidgets.QLabel(item.error, objectName="cellError", wordWrap=True))

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)
        super().mousePressEvent(event)


def _provides_text(item) -> str:
    provides = getattr(item, "provides", {})
    return ", ".join(n for names in provides.values() for n in names)


class PluginsPanel(QtWidgets.QWidget):
    """Search, install and remove plugins."""

    changed = QtCore.Signal()               # plugins were loaded or removed
    stepRequested = QtCore.Signal(object)   # a modifier instance to add to the notebook

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.entries: list = []
        self.catalog_errors: dict = {}
        self.items: list = []
        self.current = None
        self._loader = None
        self._process = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.search = QtWidgets.QLineEdit(placeholderText="Search plugins")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill)
        lay.addWidget(self.search)
        self.scroll = QtWidgets.QScrollArea(widgetResizable=True, frameShape=QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_widget = QtWidgets.QWidget(objectName="pluginList")
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.scroll.setWidget(self.list_widget)
        lay.addWidget(self.scroll, 3)
        self.details = QtWidgets.QTextBrowser(openExternalLinks=True, objectName="pluginDetails")
        self.details.setVisible(False)
        lay.addWidget(self.details, 2)
        self.use_button = QtWidgets.QPushButton("Add as a step in the notebook", objectName="primary")
        self.use_button.clicked.connect(self._use)
        self.use_button.setVisible(False)
        lay.addWidget(self.use_button)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True, objectName="pluginLog")
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        lay.addWidget(self.log)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("From file…", self.install_from_file, "Install a plugin .py file"),
                                ("pip…", self.install_with_pip, "Install a plugin package with pip"),
                                ("Folder", self.open_folder, "Open the plugin folder"),
                                ("Refresh", self.refresh, "Reload the installed plugins and the catalogs")):
            b = QtWidgets.QToolButton(text=text, toolTip=tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.refresh()

    # ----------------------------------------------------------- data
    def catalog_urls(self) -> list[str]:
        raw = self.settings["python.plugin_catalogs"] if self.settings is not None else ""
        return [u.strip() for u in raw.replace("\n", ";").split(";") if u.strip()]

    def refresh(self):
        self.items = pc.installed()
        self._fill()
        if self._loader is None or not self._loader.isRunning():
            self._loader = _CatalogLoader(self.catalog_urls(), self)
            self._loader.loaded.connect(self._catalog_loaded)
            self._loader.start()

    def wait(self, ms: int = 10000):
        """Block until the catalogs are read (tests, scripts)."""
        if self._loader is not None:
            self._loader.wait(ms)
            QtWidgets.QApplication.processEvents()

    def _catalog_loaded(self, entries, errors):
        self.entries, self.catalog_errors = entries, errors
        self._fill()

    def _fill(self):
        while self.list_layout.count():
            w = self.list_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        text = self.search.text()
        installed = pc.search(self.items, text)
        available = [e for e in pc.search(self.entries, text) if not pc.is_installed(e, self.items)]
        self._section(f"INSTALLED  {len(installed)}")
        for item in installed:
            self._row(item, True)
        if not installed:
            self.list_layout.addWidget(QtWidgets.QLabel("No plugins installed yet.", objectName="muted"))
        self._section(f"AVAILABLE  {len(available)}")
        for e in available:
            self._row(e, False)
        for url, err in self.catalog_errors.items():
            self.list_layout.addWidget(QtWidgets.QLabel(f"{url}: {err}", objectName="cellError", wordWrap=True))
        if not available and not self.catalog_errors:
            self.list_layout.addWidget(QtWidgets.QLabel(
                "Nothing more to install here. Add plugin catalogs in Preferences › Python.",
                objectName="muted", wordWrap=True))
        self.list_layout.addStretch()

    def _section(self, title):
        self.list_layout.addWidget(QtWidgets.QLabel(title, objectName="section"))

    def _row(self, item, installed):
        row = PluginRow(item, installed)
        row.clicked.connect(self.show_details)
        row.action.connect(self._act)
        self.list_layout.addWidget(row)

    # ----------------------------------------------------------- details
    def show_details(self, item):
        self.current = item
        parts = [f"<h3>{html.escape(item.name)}</h3>"]
        if getattr(item, "summary", ""):
            parts.append(f"<p>{html.escape(item.summary)}</p>")
        provides = getattr(item, "provides", {})
        for kind, names in provides.items():
            parts.append(f"<p><b>{kind}</b>: {html.escape(', '.join(names))}</p>")
        if getattr(item, "description", ""):
            parts.append(f"<pre style='white-space:pre-wrap'>{html.escape(item.description)}</pre>")
        src = getattr(item, "source", "") or getattr(item, "path", "") or getattr(item, "origin", "")
        if src:
            parts.append(f"<p style='color:{TOKENS['muted']}'>{html.escape(src)}</p>")
        if getattr(item, "homepage", ""):
            parts.append(f"<p><a href='{html.escape(item.homepage)}'>{html.escape(item.homepage)}</a></p>")
        self.details.setHtml("".join(parts))
        self.details.setVisible(True)
        self.use_button.setVisible(bool(self._modifier_names(item)))

    def _modifier_names(self, item) -> list[str]:
        names = getattr(item, "provides", {}).get("modifier", [])
        return [n for n in names if n in plugins.modifiers()]

    def _use(self):
        for name in self._modifier_names(self.current)[:1]:
            cls = plugins.modifiers()[name].cls
            try:
                self.stepRequested.emit(cls())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Add step", f"{name}() needs arguments: {exc}")

    # ----------------------------------------------------------- install / remove
    def _act(self, item, action):
        if action == "install":
            self.install(item)
        else:
            self.remove(item)

    def install(self, entry):
        if entry.is_pip:
            self._run_pip("install", entry.source[4:])
            return
        try:
            names = pc.install_file(entry)
        except Exception as exc:
            self._say(f"Install failed: {exc}")
            QtWidgets.QMessageBox.warning(self, "Install plugin", str(exc))
            return
        self._say(f"Installed {entry.name}: {', '.join(names) or 'nothing registered'}")
        self.refresh()
        self.changed.emit()

    def remove(self, item):
        if item.package:
            self._run_pip("uninstall", item.package)
            return
        if QtWidgets.QMessageBox.question(
                self, "Remove plugin", f"Remove {item.name}? The file moves to the plugin folder's .removed "
                "folder, and its steps stay available until ewald restarts.") != QtWidgets.QMessageBox.Yes:
            return
        dest = pc.remove_file(item)
        self._say(f"Removed {item.name} (moved to {dest})")
        self.refresh()
        self.changed.emit()

    def install_from_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Install plugin file", "", "Python (*.py)")
        if path:
            self.install(pc.describe_file(path))

    def install_with_pip(self):
        req, ok = QtWidgets.QInputDialog.getText(self, "Install with pip", "Package (e.g. ewald-cp2k):")
        if ok and req.strip():
            self._run_pip("install", req.strip())

    def _run_pip(self, action, requirement):
        cmd = pc.pip_command(action, requirement)
        if QtWidgets.QMessageBox.question(
                self, "pip", "Run this in the Python ewald runs in?\n\n" + " ".join(cmd)) != QtWidgets.QMessageBox.Yes:
            return
        self.log.clear()
        self.log.setVisible(True)
        p = self._process = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(
            bytes(p.readAllStandardOutput()).decode(errors="replace").rstrip()))
        p.finished.connect(lambda code, _status: self._pip_done(action, requirement, code))
        p.start(cmd[0], cmd[1:])

    def _pip_done(self, action, requirement, code):
        self._say(f"pip {action} {requirement}: {'done' if code == 0 else f'failed (exit {code})'}")
        if code == 0:
            plugins.load_plugins(force=True)
            self.refresh()
            self.changed.emit()

    def open_folder(self):
        folder = plugins.plugin_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _say(self, text):
        self.log.setVisible(True)
        self.log.appendPlainText(text)
