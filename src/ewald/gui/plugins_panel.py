"""Plugins side bar, like VS Code's Extensions view: search the installed plugins and the
catalogs (:mod:`ewald.plugin_catalog`), install, remove, and add a plugin step to the notebook."""
from __future__ import annotations

import html
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import plugin_catalog as pc
from .. import plugins
from . import workers
from .busy import BusyLabel
from .icons import icon
from .theme import TOKENS

CATALOG_MAX_AGE = 600    # seconds: showing the side bar again reads older catalogs again


class _CatalogLoader(QtCore.QThread):
    """Reads the catalogs off the GUI thread (indexes may be URLs)."""

    loaded = QtCore.Signal(list, dict, bool)   # entries, errors, whether a GitHub token exists

    def __init__(self, urls, parent=None):
        super().__init__(parent)
        self.urls = urls

    def run(self):
        entries, errors = pc.catalog(self.urls)
        # for the "private index? set GITHUB_TOKEN" hint; gh auth token is a subprocess
        token = any(pc.is_missing(e) and pc._wants_token(u) for u, e in errors.items()) and bool(pc.github_token())
        self.loaded.emit(entries, errors, token)


class _FetchWorker(QtCore.QThread):
    """Downloads or copies a plugin file into the plugin folder off the GUI thread
    (:func:`ewald.plugin_catalog.fetch_file`); the import happens back on the GUI thread."""

    fetched = QtCore.Signal(object, str)    # path (None on failure), error

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.entry = entry

    def run(self):
        try:
            self.fetched.emit(pc.fetch_file(self.entry), "")
        except Exception as exc:
            self.fetched.emit(None, str(exc) or type(exc).__name__)


class _Task(QtCore.QThread):
    """Runs ``fn()`` off the GUI thread: the slow bookkeeping around a pip run (reading every
    installed package's metadata, ``gh auth token``) froze the window for seconds."""

    done = QtCore.Signal(object, str)    # result (None on failure), error

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), "")
        except Exception as exc:
            self.done.emit(None, str(exc) or type(exc).__name__)


def _key(item) -> str:
    return pc._norm(item.name)


class PluginRow(QtWidgets.QFrame):
    """One plugin: name, summary, kinds and its Install, Update, Disable or Remove buttons."""

    clicked = QtCore.Signal(object)
    action = QtCore.Signal(object, str)   # item (an Entry for "install"/"update"), action name

    def __init__(self, item, installed: bool, parent=None, update=None, busy: str = ""):
        super().__init__(parent)
        self.item = item
        self.busy = None
        self.setObjectName("pluginRow")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"<b>{html.escape(item.name)}</b>" +
                                 (f" <span style='color:{TOKENS['muted']}'>{html.escape(item.version)}</span>"
                                  if getattr(item, "version", "") else ""), wordWrap=True)
        head.addWidget(title, 1)          # wraps, like every label here, so the row never widens the side bar
        if busy:                                    # installing, updating or removing: a spinner, no buttons
            self.busy = BusyLabel(busy)
            head.addWidget(self.busy)
        elif installed and isinstance(item, pc.Installed):
            if update is not None:
                u = QtWidgets.QPushButton(f"Update to {update.version}", objectName="pluginInstall")
                u.setToolTip(f"From {update.catalog}")
                u.clicked.connect(lambda: self.action.emit(update, "update"))
                head.addWidget(u)
            if item.provides.get("settings_page"):
                c = QtWidgets.QPushButton("Settings", objectName="pluginButton")
                c.setToolTip("Set up this plugin (" + ", ".join(item.provides["settings_page"]) + ")")
                c.clicked.connect(lambda: self.action.emit(self.item, "settings"))
                head.addWidget(c)
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
        kinds = QtWidgets.QLabel("  ".join(badges).strip(), objectName="pluginKinds", wordWrap=True)
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
    restartRequested = QtCore.Signal()      # "Restart now" on the restart bar
    themeRequested = QtCore.Signal(str)     # "Use theme": make this UI theme the app's

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.entries: list = []
        self.catalog_errors: dict = {}
        self.items: list = []
        self.current = None
        self._loader = None
        self._loading = False
        self._process = None
        self._pip_key = None
        self._pip_job = None                 # (action, requirement) from the confirmation until pip's result is loaded
        self._catalog_time = None            # time.monotonic() when the catalogs were last read
        self._has_token = False              # a GitHub token exists (read with the catalogs, off the GUI thread)
        self._workers: list = []
        self._busy: dict[str, str] = {}      # plugin key -> "Installing…" while its action runs
        self._updates = 0
        self._before = None                  # pc.loaded_versions() while pip runs
        self.restart_reasons: list[str] = []  # why ewald should restart, shown on the restart bar
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
        self.theme_button = QtWidgets.QPushButton(objectName="primary")
        self.theme_button.clicked.connect(lambda: self.themeRequested.emit(self.theme_button.property("theme")))
        self.theme_button.setVisible(False)
        lay.addWidget(self.theme_button)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True, objectName="pluginLog")
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        lay.addWidget(self.log)
        self.busy_bar = QtWidgets.QWidget(objectName="pluginBusy")
        bar = QtWidgets.QHBoxLayout(self.busy_bar)
        bar.setContentsMargins(2, 0, 0, 0)
        self.busy_label = BusyLabel()
        bar.addWidget(self.busy_label, 1)
        self.cancel_button = QtWidgets.QToolButton(text="Cancel", toolTip="Stop pip")
        self.cancel_button.clicked.connect(self.cancel)
        bar.addWidget(self.cancel_button)
        self.busy_bar.setVisible(False)
        lay.addWidget(self.busy_bar)
        # a change that only takes effect after a restart (pip replaced code that is loaded)
        self.restart_bar = QtWidgets.QFrame(objectName="restartBar")
        rb = QtWidgets.QGridLayout(self.restart_bar)
        rb.setContentsMargins(8, 6, 6, 6)
        self.restart_label = QtWidgets.QLabel(wordWrap=True)
        rb.addWidget(self.restart_label, 0, 0, 1, 3)
        later = QtWidgets.QToolButton(text="Later", toolTip="Keep working; the change takes effect at the next start")
        later.clicked.connect(lambda: self.restart_bar.setVisible(False))
        now = QtWidgets.QPushButton("Restart now", objectName="primary",
                                    toolTip="Close ewald (asking to save notebooks) and start it again with them open")
        now.clicked.connect(self.restartRequested.emit)
        rb.setColumnStretch(0, 1)
        rb.addWidget(later, 1, 1)
        rb.addWidget(now, 1, 2)
        self.restart_bar.setVisible(False)
        lay.addWidget(self.restart_bar)
        row = QtWidgets.QHBoxLayout()
        # full names in short menus rather than five clipped words in a narrow side bar
        install = QtWidgets.QToolButton(text="Install", toolTip="Install a plugin that is not in a catalog",
                                        popupMode=QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(install)
        menu.addAction("From file…", self.install_from_file).setToolTip(
            "A plugin .py file, or a downloaded .zip/.whl/.tar.gz package")
        menu.addAction("With pip…", self.install_with_pip).setToolTip("A plugin package from PyPI or a URL")
        menu.setToolTipsVisible(True)
        install.setMenu(menu)
        create = QtWidgets.QToolButton(text="Create with AI…", toolTip="Ask AI to write a new plugin")
        create.clicked.connect(self.create_with_claude)
        more = QtWidgets.QToolButton(toolTip="More", popupMode=QtWidgets.QToolButton.InstantPopup,
                                     autoRaise=True)
        more.setProperty("ewald_icon", "more")
        more.setIcon(icon("more"))
        menu = QtWidgets.QMenu(more)
        menu.addAction("Open plugin folder", self.open_folder)
        menu.addAction("Refresh", self.refresh).setToolTip("Reload the installed plugins and the catalogs")
        more.setMenu(menu)
        for b in (install, create):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(more)
        lay.addLayout(row)
        self.refresh()

    # ----------------------------------------------------------- data
    def catalog_urls(self) -> list[str]:
        raw = self.settings["python.plugin_catalogs"] if self.settings is not None else ""
        return pc.catalog_urls(raw)

    def refresh(self):
        self.items = pc.installed()
        if self._loader is None or not self._loader.isRunning():
            self._loading = True
            self._loader = workers.detach(_CatalogLoader(self.catalog_urls()))  # may outlive the panel
            self._loader.loaded.connect(self._catalog_loaded)
            self._loader.start()
        self._fill()

    def wait(self, ms: int = 10000):
        """Block until the catalogs are read and running installs are done (tests, scripts)."""
        deadline = time.monotonic() + ms / 1000
        while time.monotonic() < deadline:
            if self._loader is not None:
                self._loader.wait(10)
            for worker in list(self._workers):
                worker.wait(10)
            if self._process is not None and self._process.state() != QtCore.QProcess.NotRunning:
                self._process.waitForFinished(10)
            QtWidgets.QApplication.processEvents()
            if (self._pip_job is None and not self._workers
                    and (self._loader is None or not self._loader.isRunning())):
                break
        QtWidgets.QApplication.processEvents()

    def showEvent(self, event):
        super().showEvent(event)
        # a side bar left open for hours still learns about new versions when it is shown again
        if self._catalog_time is not None and time.monotonic() - self._catalog_time > CATALOG_MAX_AGE:
            self.refresh()

    @property
    def busy(self) -> dict:
        """``{plugin key: "Installing…"}`` for the actions still running."""
        return dict(self._busy)

    def _set_busy(self, key, text=None, status=""):
        if text:
            self._busy[key] = text
        else:
            self._busy.pop(key, None)
        running = self._pip_job is not None or bool(self._busy)
        if status:
            self.busy_label.setText(status)
        elif not running:
            self.busy_label.setText("")
        self.busy_bar.setVisible(running)
        self.cancel_button.setVisible(self._process is not None)
        self._fill()

    @property
    def loading(self) -> bool:
        """The catalogs are being read."""
        return self._loading

    def _catalog_loaded(self, entries, errors, has_token=False):
        self.entries, self.catalog_errors, self._has_token = entries, errors, has_token
        self._loading = False
        self._catalog_time = time.monotonic()
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
        if self.loading:                          # the catalogs are still being read (URLs)
            self.list_layout.addWidget(BusyLabel("Loading plugin catalogs…"))
        for e in available:
            self._row(e, False)
        for url, err in self.catalog_errors.items():
            if pc.is_missing(err):      # e.g. the default index before it is published
                private = pc._wants_token(url) and not self._has_token
                self.list_layout.addWidget(QtWidgets.QLabel(
                    f"No plugin index at {url}" + (" (if it is private, set GITHUB_TOKEN or run gh auth login)."
                                                   if private else "."), objectName="muted", wordWrap=True))
            else:
                self.list_layout.addWidget(QtWidgets.QLabel(f"{url}: {err}", objectName="cellError", wordWrap=True))
        if not available and not self.loading and not any(not pc.is_missing(e) for e in self.catalog_errors.values()):
            self.list_layout.addWidget(QtWidgets.QLabel(
                "Nothing more to install here. Add plugin catalogs in Preferences › Plugins.",
                objectName="muted", wordWrap=True))
        self.list_layout.addStretch()

    def _section(self, title):
        self.list_layout.addWidget(QtWidgets.QLabel(title, objectName="section"))

    def _row(self, item, installed, update=None):
        busy = self._busy.get(_key(item)) or (update is not None and self._busy.get(_key(update))) or ""
        row = PluginRow(item, installed, update=update, busy=busy)
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
        themes = self._theme_names(item)
        self.theme_button.setVisible(bool(themes))
        if themes:
            self.theme_button.setProperty("theme", themes[0])
            self.theme_button.setText(f"Use theme {themes[0]}")
            self.theme_button.setToolTip("Make it the UI theme (Preferences › General › UI theme)")

    def _theme_names(self, item) -> list[str]:
        names = getattr(item, "provides", {}).get("ui_theme", [])
        return [n for n in names if n in plugins.ui_themes()] if getattr(item, "state", "loaded") == "loaded" else []

    def recommend_restart(self, reason: str):
        """Show the restart bar: ``reason`` says why the change needs a restart to take effect."""
        if not reason:
            return
        if reason not in self.restart_reasons:
            self.restart_reasons.append(reason)
        self._say(f"Restart ewald to finish: {reason}")
        self.restart_label.setText("Restart ewald to finish: " + "; ".join(self.restart_reasons))
        self.restart_bar.setVisible(True)

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
        elif action == "settings":
            self.edit_settings(item)
        elif action in ("enable", "disable"):
            pc.set_enabled(item, action == "enable")
            self._say(f"{item.name}: {action}d")
            self.recommend_restart(pc.restart_reason(action, item))
            self.refresh()
            self.changed.emit()
        else:
            self.remove(item)

    def edit_settings(self, item, exec_: bool = True):
        """Open the settings page(s) of an installed plugin (:func:`ewald.plugins.settings_page`)."""
        from .plugin_settings import edit

        dlg = None
        for name in item.provides.get("settings_page", []):
            if name in plugins.settings_pages():
                dlg = edit(name, self, exec_=exec_)
                if exec_ and dlg.message:
                    self._say(f"{dlg.page.title}: {dlg.message}")
        return dlg

    def install(self, entry, update: bool = False):
        """Install a catalog entry; ``update`` replaces the installed plugin it matches. The
        download (or pip) runs in the background while the row shows a spinner."""
        key = _key(entry)
        if key in self._busy:
            return
        verb = "Updating" if update else "Installing"
        if entry.is_pip:
            self._run_pip("update" if update else "install", entry.source[4:], key=key, busy=f"{verb}…")
            return
        old = pc.match(entry, self.items) if update else None
        worker = workers.detach(_FetchWorker(entry))  # may outlive the panel
        worker.fetched.connect(lambda path, error: self._fetched(worker, entry, update, old, path, error))
        self._workers.append(worker)
        self._set_busy(key, f"{verb}…", f"{verb} {entry.name}…")
        worker.start()

    def _fetched(self, worker, entry, update, old, path, error):
        worker.wait()
        if worker in self._workers:
            self._workers.remove(worker)
        names = []
        if not error:
            try:
                if old is not None and old.path and Path(old.path).name != Path(path).name:
                    pc.remove_file(old)             # the new version came under another file name
                names = plugins.load_plugin_file(path)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
        self._set_busy(_key(entry))
        if error:
            self._say(f"Install failed: {error}")
            QtWidgets.QMessageBox.warning(self, "Install plugin", error)
            return
        self._say(f"{'Updated' if update else 'Installed'} {entry.name}"
                  f"{' to ' + entry.version if update and entry.version else ''}: {', '.join(names) or 'nothing registered'}")
        self.recommend_restart(pc.restart_reason("update" if update else "install", old))
        self.refresh()
        self.changed.emit()
        self.highlight(path)
        if self.current is not None and str(getattr(self.current, "path", "")) == str(path):
            self.show_details(self.current)
        else:
            for item in self.items:     # show the new plugin, with its Use theme button
                if str(getattr(item, "path", "") or "") == str(path):
                    self.show_details(item)
                    break

    def highlight(self, path):
        """Scroll to the installed plugin's row and mark it for a moment (it moved to INSTALLED)."""
        for row in self.list_widget.findChildren(PluginRow):
            if str(getattr(row.item, "path", "") or "") == str(path):
                row.setProperty("fresh", True)
                row.style().unpolish(row)
                row.style().polish(row)
                self.scroll.ensureWidgetVisible(row)
                QtCore.QTimer.singleShot(2500, row, lambda r=row: (r.setProperty("fresh", False),
                                                                  r.style().unpolish(r), r.style().polish(r)))
                return row
        return None

    def remove(self, item):
        if item.package:
            self._run_pip("uninstall", item.package, key=_key(item), busy="Removing…")
            return
        if QtWidgets.QMessageBox.question(
                self, "Remove plugin", f"Remove {item.name}? The file moves to the plugin folder's .removed "
                "folder (put it back to restore it).") != QtWidgets.QMessageBox.Yes:
            return
        dest = pc.remove_file(item)
        plugins.load_plugins(force=True)        # its steps, readers, themes... leave the registries now
        self._say(f"Removed {item.name} (moved to {dest})")
        self.refresh()
        self.changed.emit()

    def install_from_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Install plugin", "", "Plugin (*.py *.zip *.whl *.tar.gz);;Python (*.py);;Package archive (*.zip *.whl *.tar.gz)")
        if not path:
            return
        if path.lower().endswith(pc.ARCHIVE_SUFFIXES):     # a package: pip builds it (and asks first)
            self._run_pip("install", path)
        else:
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

    def _run_pip(self, action, requirement, key=None, busy=""):
        """Run pip in a separate process while the window keeps working: the package metadata
        it needs before and after (what is loaded, a GitHub token) is read in a thread, pip's
        output goes to the log, Cancel stops it at any point."""
        if self._pip_job is not None:
            self._say("pip is still running; wait for it to finish or press Cancel.")
            return
        cmd = pc.pip_command(action, requirement)
        if QtWidgets.QMessageBox.question(
                self, "pip", "Run this in the Python ewald runs in?\n\n" + " ".join(cmd)) != QtWidgets.QMessageBox.Yes:
            return
        self.log.clear()
        self.log.setVisible(True)
        self._pip_job, self._pip_key = (action, requirement), key
        p = self._process = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(
            bytes(p.readAllStandardOutput()).decode(errors="replace").rstrip()))
        p.finished.connect(lambda code, status: self._pip_done(
            action, requirement, -1 if status == QtCore.QProcess.CrashExit else code))
        p.errorOccurred.connect(lambda err: err == QtCore.QProcess.FailedToStart and self._pip_done(action, requirement, -1))
        reqs = requirement if isinstance(requirement, str) else " ".join(requirement)
        verb = {"install": "Installing", "update": "Updating", "uninstall": "Removing"}.get(action, "Running pip for")
        self._set_busy(key or "", busy or "", f"{verb} {reqs} with pip…")
        # a GitHub token for git+https sources (never logged), and what is loaded now: pip may
        # replace code that is already imported (pc.restart_reason compares afterwards)
        self._background(lambda: (pc.pip_env(requirement), pc.loaded_versions()),
                         lambda result, error: self._pip_start(p, cmd, result))

    def _background(self, fn, then):
        task = workers.detach(_Task(fn))        # may outlive the panel
        task.done.connect(lambda result, error: (task in self._workers and self._workers.remove(task),
                                                 then(result, error)))
        self._workers.append(task)
        task.start()
        return task

    def _pip_start(self, p, cmd, result):
        if p is not self._process:          # cancelled while getting ready
            return
        extra, self._before = result if result is not None else ({}, None)
        if extra:
            env = QtCore.QProcessEnvironment.systemEnvironment()
            for k, v in extra.items():
                env.insert(k, v)
            p.setProcessEnvironment(env)
        p.start(cmd[0], cmd[1:])

    def cancel(self):
        """Stop a running pip."""
        if self._process is not None:
            self._say("Cancelling pip…")
            if self._process.state() == QtCore.QProcess.NotRunning:   # not started yet
                self._pip_done(*self._pip_job, -1)
            else:
                self._process.kill()

    def install_requirements(self, requirements: list[str]):
        """pip install the packages a notebook needs (one confirmation, one pip run)."""
        if requirements:
            self._run_pip("install", list(requirements))

    def _pip_done(self, action, requirement, code):
        if self._process is None:            # FailedToStart and finished both report
            return
        self._process.deleteLater()
        self._process, key = None, self._pip_key
        self.cancel_button.setVisible(False)
        if not isinstance(requirement, str):
            requirement = " ".join(requirement)
        self._say(f"pip {action} {requirement}: {'done' if code == 0 else f'failed (exit {code})'}")
        before, self._before = self._before, None
        if code != 0:
            self._pip_finished(key, None)
            return
        self._set_busy(key or "", self._busy.get(key or "") or "Loading…", "Loading the installed plugins…")
        # which loaded packages pip changed: the metadata of every installed package is read
        self._background(lambda: pc.restart_reason(action, before=before),
                         lambda reason, error: self._pip_finished(key, reason or ""))

    def _pip_finished(self, key, reason):
        """pip is done; ``reason`` is None when it failed, else why a restart is needed ("" for none)."""
        self._pip_job = self._pip_key = None
        self._set_busy(key or "")
        if reason is None:
            return
        plugins.load_plugins(force=True)
        self.recommend_restart(reason)
        self.refresh()
        self.changed.emit()

    def open_folder(self):
        folder = plugins.plugin_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _say(self, text):
        self.log.setVisible(True)
        self.log.appendPlainText(text)
