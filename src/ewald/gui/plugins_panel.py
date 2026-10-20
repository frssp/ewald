"""Plugins side bar, like VS Code's Extensions view: search the installed plugins and the
catalogs (:mod:`ewald.plugin_catalog`), install, remove, and add a plugin step to the notebook."""
from __future__ import annotations

import html
from pathlib import Path

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
    """One plugin: name, summary, kinds and its Install, Update, Disable or Remove buttons."""

    clicked = QtCore.Signal(object)
    action = QtCore.Signal(object, str)   # item (an Entry for "install"/"update"), action name

    def __init__(self, item, installed: bool, parent=None, update=None):
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
            if update is not None:
                u = QtWidgets.QPushButton(f"Update to {update.version}", objectName="pluginInstall")
                u.setToolTip(f"From {update.catalog}")
                u.clicked.connect(lambda: self.action.emit(update, "update"))
                head.addWidget(u)
            if item.id:
                on = item.enabled
                t = QtWidgets.QPushButton("Disable" if on else "Enable", objectName="pluginButton")
                t.clicked.connect(lambda: self.action.emit(self.item, "disable" if on else "enable"))
                head.addWidget(t)
            if item.removable:
                b = QtWidgets.QPushButton("Remove", objectName="pluginButton")
                b.clicked.connect(lambda: self.action.emit(self.item, "remove"))
                head.addWidget(b)
        elif not installed:
            b = QtWidgets.QPushButton("Install", objectName="pluginInstall")
            b.clicked.connect(lambda: self.action.emit(self.item, "install"))
            if not item.compatible:
                b.setEnabled(False)
                b.setToolTip(f"Needs ewald {item.requires_ewald}; this is {plugins.ewald_version()}")
                self.setProperty("incompatible", True)
            head.addWidget(b)
        else:
            head.addWidget(QtWidgets.QLabel("installed", objectName="muted"))
        lay.addLayout(head)
        summary = getattr(item, "summary", "") or _provides_text(item)
        if summary:
            s = QtWidgets.QLabel(summary, objectName="pluginSummary", wordWrap=True)
            lay.addWidget(s)
        badges = item.kinds + [_origin_badge(item)]
        if not installed and not item.compatible:
            badges.append(f"· needs ewald {item.requires_ewald}")
        kinds = QtWidgets.QLabel("  ".join(badges).strip(), objectName="pluginKinds")
        lay.addWidget(kinds)
        if getattr(item, "error", ""):
            lay.addWidget(QtWidgets.QLabel(item.error, objectName="cellError", wordWrap=True))

    def mousePressEvent(self, event):
        self.clicked.emit(self.item)
        super().mousePressEvent(event)


def _origin_badge(item) -> str:
    """Where a plugin comes from: folder, pip package, this session, or its catalog."""
    if isinstance(item, pc.Installed):
        origin = "· pip" if item.package else "· folder" if item.path else "· session" if not item.id else "· file"
        return origin + ("" if item.state in ("loaded", "not loaded") else f"  · {item.state}")
    return "· pip" if item.is_pip else "· example" if item.catalog == "examples" else ""


def _doc_html(doc: str) -> str:
    """A docstring as HTML: paragraphs re-flowed, indented blocks kept as code."""
    out = []
    for para in doc.strip().split("\n\n"):
        lines = para.splitlines()
        if lines and all(ln.startswith("    ") for ln in lines if ln.strip()):
            out.append(f"<pre>{html.escape(chr(10).join(ln[4:] for ln in lines))}</pre>")
        else:
            text = html.escape(" ".join(ln.strip() for ln in lines)).replace("``", "")
            out.append(f"<p>{text.removesuffix('::') + (':' if text.endswith('::') else '')}</p>")
    return "".join(out)


def _provides_text(item) -> str:
    provides = getattr(item, "provides", {})
    return ", ".join(n for names in provides.values() for n in names)


class PluginsPanel(QtWidgets.QWidget):
    """Search, install and remove plugins."""

    changed = QtCore.Signal()               # plugins were loaded or removed
    stepRequested = QtCore.Signal(object)   # a modifier instance to add to the notebook
    createRequested = QtCore.Signal(str)    # a prompt for the AI to write a new plugin
    updatesChanged = QtCore.Signal(int)     # how many installed plugins a catalog has a newer version of

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.entries: list = []
        self.catalog_errors: dict = {}
        self.items: list = []
        self.current = None
        self._loader = None
        self._process = None
        self._updates = 0
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.search = QtWidgets.QLineEdit(placeholderText="Search plugins  (@installed, @updates, @available, @modifier…)")
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
                                ("Create…", self.create_with_claude, "Ask AI to write a new plugin"),
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
        return pc.catalog_urls(raw)

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

    def updates(self) -> list:
        """``[(installed item, newer catalog entry)]`` for every plugin with an update."""
        out = []
        for item in self.items:
            new = pc.update_for(item, self.entries) if item.id else None
            if new is not None:
                out.append((item, new))
        return out

    def _fill(self):
        n = len(self.updates())
        if n != self._updates:
            self._updates = n
            self.updatesChanged.emit(n)
        while self.list_layout.count():
            w = self.list_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        words = self.search.text().split()
        filters = {w[1:].lower() for w in words if w.startswith("@")}
        text = " ".join(w for w in words if not w.startswith("@"))
        kinds = filters & set(pc.KINDS)
        installed = [i for i in pc.search(self.items, text) if not kinds or kinds & set(i.kinds)]
        available = [e for e in pc.search(self.entries, text)
                     if not pc.is_installed(e, self.items) and (not kinds or kinds & set(e.kinds))]
        if "updates" in filters:
            ids = {id(i) for i, _ in self.updates()}
            installed, available = [i for i in installed if id(i) in ids], []
        if "available" in filters:
            installed = []
        if "installed" in filters:
            available = []
        self._section(f"INSTALLED  {len(installed)}")
        for item in installed:
            self._row(item, True, pc.update_for(item, self.entries) if item.id else None)
        if not installed:
            self.list_layout.addWidget(QtWidgets.QLabel("No plugins installed yet.", objectName="muted"))
        self._section(f"AVAILABLE  {len(available)}")
        for e in available:
            self._row(e, False)
        for url, err in self.catalog_errors.items():
            if pc.is_missing(err):      # e.g. the default index before it is published
                private = pc._wants_token(url) and not pc.github_token()
                self.list_layout.addWidget(QtWidgets.QLabel(
                    f"No plugin index at {url}" + (" (if it is private, set GITHUB_TOKEN or run gh auth login)."
                                                   if private else "."), objectName="muted", wordWrap=True))
            else:
                self.list_layout.addWidget(QtWidgets.QLabel(f"{url}: {err}", objectName="cellError", wordWrap=True))
        if not available and not any(not pc.is_missing(e) for e in self.catalog_errors.values()):
            self.list_layout.addWidget(QtWidgets.QLabel(
                "Nothing more to install here. Add plugin catalogs in Preferences › Python.",
                objectName="muted", wordWrap=True))
        self.list_layout.addStretch()

    def _section(self, title):
        self.list_layout.addWidget(QtWidgets.QLabel(title, objectName="section"))

    def _row(self, item, installed, update=None):
        row = PluginRow(item, installed, update=update)
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
            parts.append(_doc_html(item.description))
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
        if action in ("install", "update"):
            self.install(item, update=action == "update")
        elif action in ("enable", "disable"):
            pc.set_enabled(item, action == "enable")
            self._say(f"{item.name}: {action}d" + (" (restart ewald to unload its package)"
                                                    if action == "disable" and item.package else ""))
            self.refresh()
            self.changed.emit()
        else:
            self.remove(item)

    def install(self, entry, update: bool = False):
        """Install a catalog entry; ``update`` replaces the installed plugin it matches."""
        if entry.is_pip:
            self._run_pip("update" if update else "install", entry.source[4:])
            return
        old = pc.match(entry, self.items) if update else None
        try:
            if old is not None and old.path and Path(old.path).name != entry.file_name:
                pc.remove_file(old)             # the new version comes under another file name
            names = pc.install_file(entry)
        except Exception as exc:
            self._say(f"Install failed: {exc}")
            QtWidgets.QMessageBox.warning(self, "Install plugin", str(exc))
            return
        self._say(f"{'Updated' if update else 'Installed'} {entry.name}"
                  f"{' to ' + entry.version if update and entry.version else ''}: {', '.join(names) or 'nothing registered'}")
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

    def create_with_claude(self):
        what, ok = QtWidgets.QInputDialog.getText(self, "Create a plugin with AI",
                                                  "What should the new step compute?")
        if ok and what.strip():
            self.createRequested.emit(
                f"Write an ewald modifier plugin that {what.strip()}. Save it in the plugin folder, "
                "load it with ewald.plugins.load_plugin_file and add it as a step.")

    def _run_pip(self, action, requirement):
        cmd = pc.pip_command(action, requirement)
        if QtWidgets.QMessageBox.question(
                self, "pip", "Run this in the Python ewald runs in?\n\n" + " ".join(cmd)) != QtWidgets.QMessageBox.Yes:
            return
        self.log.clear()
        self.log.setVisible(True)
        p = self._process = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        extra = pc.pip_env(requirement)         # a GitHub token for git+https sources, never logged
        if extra:
            env = QtCore.QProcessEnvironment.systemEnvironment()
            for k, v in extra.items():
                env.insert(k, v)
            p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(
            bytes(p.readAllStandardOutput()).decode(errors="replace").rstrip()))
        p.finished.connect(lambda code, _status: self._pip_done(action, requirement, code))
        p.start(cmd[0], cmd[1:])

    def install_requirements(self, requirements: list[str]):
        """pip install the packages a notebook needs (one confirmation, one pip run)."""
        if requirements:
            self._run_pip("install", list(requirements))

    def _pip_done(self, action, requirement, code):
        if not isinstance(requirement, str):
            requirement = " ".join(requirement)
        self._say(f"pip {action} {requirement}: {'done' if code == 0 else f'failed (exit {code})'}")
        if code == 0:
            plugins.load_plugins(force=True)
            if action == "uninstall":
                self._say("Restart ewald to unload the removed package's classes.")
            self.refresh()
            self.changed.emit()

    def open_folder(self):
        folder = plugins.plugin_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _say(self, text):
        self.log.setVisible(True)
        self.log.appendPlainText(text)
