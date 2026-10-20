"""Remotes side bar: saved SSH computers (:mod:`ewald.io.profiles`), like the session list of an
SSH terminal app, with a file browser that opens remote files in place or downloads a copy.

Passwords and key passphrases typed here go to the OS keychain (:mod:`ewald.keychain`); the
profile list itself (``remote_profiles.json``) and notebooks only ever hold ``name:/path``.
"""
from __future__ import annotations

import posixpath
import time

from PySide6 import QtCore, QtGui, QtWidgets

from .. import keychain
from ..io import profiles, remote
from . import header
from .remote_dialog import _size, add_recent, load_recent

ROLE = QtCore.Qt.UserRole


def _busy():
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


def _idle():
    QtWidgets.QApplication.restoreOverrideCursor()


def keychain_note() -> str:
    where = keychain.keyring_name()
    return (f"Saved in {where}." if where else
            "No system keychain here: saved to credentials.json, readable only by you.")


class ProfileDialog(QtWidgets.QDialog):
    """Create or edit a profile: host, user, port, authentication, jump host, start folder."""

    def __init__(self, profile: profiles.Profile | None = None, parent=None):
        super().__init__(parent)
        self.old_name = profile.name if profile else None
        p = profile or profiles.Profile("")
        self.setWindowTitle("Edit remote computer" if profile else "New remote computer")
        self.resize(480, 0)

        self.name = QtWidgets.QLineEdit(p.name, placeholderText="e.g. nurion  (used as nurion:/path)")
        self.host = QtWidgets.QComboBox(editable=True)
        self.host.addItems(remote.config_hosts())
        self.host.setEditText(p.host)
        self.host.lineEdit().setPlaceholderText("host name, IP or ~/.ssh/config alias (default: name)")
        self.user = QtWidgets.QLineEdit(p.user, placeholderText="from ~/.ssh/config, else your local user")
        self.port = QtWidgets.QSpinBox(minimum=0, maximum=65535, specialValueText="22 / ~/.ssh/config")
        self.port.setValue(p.port or 0)

        self.auth = {}
        auth_box = QtWidgets.QHBoxLayout()
        group = QtWidgets.QButtonGroup(self)
        for key, label in (("auto", "Agent / default keys"), ("key", "Key file"), ("password", "Password")):
            b = QtWidgets.QRadioButton(label)
            b.setToolTip(profiles.AUTH[key])
            group.addButton(b)
            auth_box.addWidget(b)
            self.auth[key] = b
        auth_box.addStretch(1)
        self.auth.get(p.auth, self.auth["auto"]).setChecked(True)

        self.key_file = QtWidgets.QLineEdit(p.key_file, placeholderText="~/.ssh/id_ed25519")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse_key)
        key_row = QtWidgets.QHBoxLayout()
        key_row.addWidget(self.key_file, 1)
        key_row.addWidget(browse)
        self.key_widget = QtWidgets.QWidget()
        self.key_widget.setLayout(key_row)
        key_row.setContentsMargins(0, 0, 0, 0)

        name = p.name or ""
        self.saved = {k: bool(name and profiles.secret(name, k)) for k in ("password", "passphrase")}
        self.password = QtWidgets.QLineEdit(echoMode=QtWidgets.QLineEdit.Password)
        self.passphrase = QtWidgets.QLineEdit(echoMode=QtWidgets.QLineEdit.Password)
        self.forget = {}
        for kind, edit in (("password", self.password), ("passphrase", self.passphrase)):
            edit.setPlaceholderText("saved; type to replace" if self.saved[kind] else
                                    "leave empty to be asked when connecting")
            self.forget[kind] = QtWidgets.QCheckBox("Forget saved", visible=self.saved[kind])
        self.jump = QtWidgets.QComboBox(editable=True)
        self.jump.addItems([""] + [n for n in profiles.names() if n != p.name] + remote.config_hosts())
        self.jump.setEditText(p.jump)
        self.jump.lineEdit().setPlaceholderText("none, or a profile / alias / user@host:port (ProxyJump)")
        self.folder = QtWidgets.QLineEdit(p.folder, placeholderText="/scratch/me  (empty: remote home)")
        self.note = QtWidgets.QLineEdit(p.note)
        self.bookmarks = list(p.bookmarks)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Host", self.host)
        form.addRow("User", self.user)
        form.addRow("Port", self.port)
        form.addRow("Login", auth_box)
        form.addRow("Key file", self.key_widget)
        self.pw_row = self._secret_row(self.password, self.forget["password"])
        self.pp_row = self._secret_row(self.passphrase, self.forget["passphrase"])
        form.addRow("Password", self.pw_row)
        form.addRow("Passphrase", self.pp_row)
        form.addRow("Jump host", self.jump)
        form.addRow("Start folder", self.folder)
        form.addRow("Note", self.note)
        self.form = form
        self.secret_note = QtWidgets.QLabel(keychain_note() + " One-time codes are always asked.",
                                            objectName="muted", wordWrap=True)
        self.status = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.test_button = buttons.addButton("Test connection", QtWidgets.QDialogButtonBox.ActionRole)
        self.test_button.clicked.connect(self.test)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.secret_note)
        lay.addWidget(self.status)
        lay.addWidget(buttons)
        for b in self.auth.values():
            b.toggled.connect(self._auth_changed)
        self._auth_changed()

    @staticmethod
    def _secret_row(edit, forget):
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        row.addWidget(forget)
        return w

    def auth_kind(self) -> str:
        return next(k for k, b in self.auth.items() if b.isChecked())

    def _auth_changed(self):
        kind = self.auth_kind()
        for widget, show in ((self.key_widget, kind == "key"), (self.pw_row, kind == "password"),
                             (self.pp_row, kind in ("key", "auto") and (kind == "key" or self.saved["passphrase"]))):
            self.form.setRowVisible(widget, show)
        self.adjustSize()

    def _browse_key(self):
        import os

        start = os.path.expanduser("~/.ssh")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Private key", start)
        if path:
            self.key_file.setText(path)

    def profile(self) -> profiles.Profile:
        return profiles.Profile(
            name=self.name.text().strip(), host=self.host.currentText().strip(), user=self.user.text().strip(),
            port=self.port.value() or None, auth=self.auth_kind(), key_file=self.key_file.text().strip(),
            jump=self.jump.currentText().strip(), folder=self.folder.text().strip(), note=self.note.text().strip(),
            bookmarks=self.bookmarks)

    def secrets(self) -> dict:
        """password / passphrase for :func:`profiles.save`: text, '' (forget) or None (keep)."""
        out = {}
        kind = self.auth_kind()
        for k, edit in (("password", self.password), ("passphrase", self.passphrase)):
            text = edit.text()
            if self.forget[k].isChecked():
                out[k] = ""
            elif text:
                relevant = (k == "password" and kind == "password") or (k == "passphrase" and kind != "password")
                out[k] = text if relevant else None
            else:
                out[k] = None
        return out

    def test(self) -> bool:
        prof = self.profile()
        if not prof.name:
            prof.name = "test-connection"
        s = self.secrets()
        self.status.setText("Connecting…")
        QtWidgets.QApplication.processEvents()
        _busy()
        try:
            ok, msg = profiles.test(prof, password=s["password"] or None, passphrase=s["passphrase"] or None)
        finally:
            _idle()
        self.status.setText(("✓ " if ok else "✗ ") + msg)
        return ok

    def _save(self):
        try:
            self.saved_profile = profiles.save(self.profile(), old_name=self.old_name, **self.secrets())
        except ValueError as exc:
            self.status.setText(f"✗ {exc}")
            return
        self.accept()


class RemotesPanel(QtWidgets.QWidget):
    """Saved remote computers on top, a file browser for the selected one below."""

    openRequested = QtCore.Signal(str)        # name:/path to open (streamed or cached)
    nebRequested = QtCore.Signal(str)         # name:/folder of a NEB run (VASP 00..NN or LAMMPS neb)
    downloadedOpen = QtCore.Signal(str)       # a local copy to open
    message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("remotesPanel")
        self.cwd = None
        self.host = None
        self.rows = []

        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>REMOTE COMPUTERS</b>"), 1)
        new = QtWidgets.QToolButton(text="+ New", toolTip="Save a new remote computer (SSH profile)")
        new.clicked.connect(lambda: self.edit(None))
        head.addWidget(new)

        self.list = QtWidgets.QListWidget(objectName="remotesList")
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._profile_menu)
        self.list.itemActivated.connect(lambda it: self.connect_to(it.data(ROLE)[1]))
        self.list.itemDoubleClicked.connect(lambda it: self.connect_to(it.data(ROLE)[1]))

        self.path = QtWidgets.QLineEdit(placeholderText="pick a computer above", objectName="remotePath")
        self.path.returnPressed.connect(lambda: self.browse(self.path.text().strip()))
        nav = QtWidgets.QHBoxLayout()
        for name, tip, slot in (("folder_up", "Parent folder", self.up), ("home", "Start folder", lambda: self.browse("")),
                                ("refresh", "Refresh", lambda: self.browse(self.cwd or "")),
                                ("star", "Bookmark this folder", self.bookmark)):
            b = header.button(name, tip)
            b.clicked.connect(slot)
            nav.addWidget(b)

        self.places = QtWidgets.QComboBox(objectName="remotePlaces")
        self.places.activated.connect(self._place)
        nav.addWidget(self.places, 1)
        self.filter = QtWidgets.QLineEdit(placeholderText="Filter files", clearButtonEnabled=True)
        self.filter.textChanged.connect(self._fill)

        self.files = QtWidgets.QTreeWidget(objectName="remoteFiles")
        self.files.setHeaderLabels(["Name", "Size", "Modified"])
        self.files.setRootIsDecorated(False)
        cols = self.files.header()
        cols.setStretchLastSection(False)
        cols.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in (1, 2):
            cols.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.files.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.files.customContextMenuRequested.connect(self._file_menu)
        self.files.itemActivated.connect(self._activated)

        self.status = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.list)
        lower = QtWidgets.QWidget()
        low = QtWidgets.QVBoxLayout(lower)
        low.setContentsMargins(0, 0, 0, 0)
        low.addWidget(self.path)
        low.addLayout(nav)
        low.addWidget(self.filter)
        low.addWidget(self.files, 1)
        split.addWidget(lower)
        split.setSizes([120, 480])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addLayout(head)
        lay.addWidget(split, 1)
        lay.addWidget(self.status)
        self.refresh()

    # ------------------------------------------------------------ profile list
    def refresh(self):
        """Reload the profile list (keeps the selection)."""
        current = self.list.currentItem().data(ROLE) if self.list.currentItem() else None
        self.list.clear()
        connected = {k[0] for k, c in list(remote._connections.items()) if c.is_active()}
        for p in profiles.profiles():
            dot = "●" if p.name in connected else "○"
            extra = f" via {p.jump}" if p.jump else ""
            item = QtWidgets.QListWidgetItem(f"{dot}  {p.name}\n      {p.target}{extra}")
            item.setToolTip(f"{p.name}: {p.target}\n{profiles.AUTH[p.auth]}" + (f"\n{p.note}" if p.note else "") +
                            f"\nOpen files as {p.name}:/path")
            item.setData(ROLE, ("profile", p.name))
            f = item.font()
            f.setBold(True)
            item.setFont(f)
            self.list.addItem(item)
        for h in profiles.unsaved_config_hosts():
            item = QtWidgets.QListWidgetItem(f"○  {h}\n      ~/.ssh/config")
            item.setData(ROLE, ("config", h))
            item.setForeground(QtGui.QBrush(self.palette().color(QtGui.QPalette.PlaceholderText)))
            item.setToolTip(f"Host {h} from {remote.ssh_config_path()}; right-click to save it as a profile")
            self.list.addItem(item)
        if not self.list.count():
            item = QtWidgets.QListWidgetItem("No remote computers yet: + New")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.list.addItem(item)
        for i in range(self.list.count()):
            if self.list.item(i).data(ROLE) == current:
                self.list.setCurrentRow(i)
        self._fill_places()

    def edit(self, name: str | None):
        dlg = ProfileDialog(profiles.get(name) if name else None, self)
        if dlg.exec():
            if self.host == name and name != dlg.saved_profile.name:
                self.host = dlg.saved_profile.name
            self.refresh()
            self.message.emit(f"Saved remote computer {dlg.saved_profile.name}")
        return dlg

    def _profile_menu(self, pos):
        item = self.list.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        if item and item.data(ROLE):
            kind, name = item.data(ROLE)
            menu.addAction("Browse files", lambda: self.connect_to(name))
            if kind == "profile":
                menu.addAction("Edit…", lambda: self.edit(name))
                menu.addAction("Test connection", lambda: self.test(name))
                menu.addAction("Duplicate", lambda: self.duplicate(name))
                menu.addAction("Move up", lambda: (profiles.move(name, -1), self.refresh()))
                menu.addAction("Move down", lambda: (profiles.move(name, 1), self.refresh()))
                menu.addAction("Disconnect", lambda: (remote.disconnect(name), self.refresh()))
                menu.addSeparator()
                menu.addAction("Delete", lambda: self.delete(name))
            else:
                menu.addAction("Save as profile…", lambda: self.save_config_host(name))
        else:
            menu.addAction("New remote computer…", lambda: self.edit(None))
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def duplicate(self, name):
        p = profiles.get(name)
        taken = set(profiles.names())
        k = 2
        while f"{name}-{k}" in taken:
            k += 1
        from dataclasses import replace

        profiles.save(replace(p, name=f"{name}-{k}"))
        self.refresh()

    def save_config_host(self, alias):
        dlg = ProfileDialog(profiles.from_ssh_config(alias), self)
        dlg.old_name = None
        if dlg.exec():
            self.refresh()

    def delete(self, name, confirm=True):
        if confirm and QtWidgets.QMessageBox.question(
                self, "Delete remote computer",
                f"Delete {name} and its saved password or passphrase? Notebooks that load {name}:/… "
                "will need it again.") != QtWidgets.QMessageBox.Yes:
            return
        profiles.delete(name)
        if self.host == name:
            self.host = self.cwd = None
            self.files.clear()
        self.refresh()

    def test(self, name):
        _busy()
        self.status.setText(f"Testing {name}…")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        try:
            ok, msg = profiles.test(name)
        finally:
            _idle()
        self.status.setText(("✓ " if ok else "✗ ") + msg)
        return ok

    # ------------------------------------------------------------ browsing
    def connect_to(self, host: str):
        self.host = host
        p = profiles.get(host)
        return self.browse(p.folder if p and p.folder else "")

    def _loc(self, path: str) -> str:
        return f"{self.host}:{path}"

    def browse(self, path: str) -> bool:
        if not self.host:
            self.status.setText("Pick a remote computer first.")
            return False
        _busy()
        self.status.setText(f"Connecting to {self.host}…")
        QtWidgets.QApplication.processEvents()
        try:
            if not path:
                p = profiles.get(self.host)
                path = p.folder if p and p.folder else remote.home(self._loc("."))
            if not path.startswith("/"):
                path = posixpath.join(self.cwd or remote.home(self._loc(".")), path)
            rows = remote.listdir(self._loc(path))
        except Exception as exc:
            self.status.setText(f"✗ {type(exc).__name__}: {exc}")
            return False
        finally:
            _idle()
        self.cwd = posixpath.normpath(path)
        self.rows = [r for r in rows if not r[0].startswith(".")]
        self.path.setText(self.cwd)
        self._fill()
        self._fill_places()
        self.refresh()
        return True

    def _fill(self):
        self.files.clear()
        needle = self.filter.text().strip().lower()
        style = self.style()
        shown = 0
        for name, is_dir, size, mtime in self.rows:
            if needle and needle not in name.lower() and not is_dir:
                continue
            item = QtWidgets.QTreeWidgetItem([name + ("/" if is_dir else ""), "" if is_dir else _size(size),
                                              time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))])
            item.setData(0, ROLE, (name, is_dir))
            item.setIcon(0, style.standardIcon(QtWidgets.QStyle.SP_DirIcon if is_dir else QtWidgets.QStyle.SP_FileIcon))
            self.files.addTopLevelItem(item)
            shown += 1
        if self.cwd:
            self.status.setText(f"{shown} entries in {self.host}:{self.cwd}")

    def up(self):
        if self.cwd:
            self.browse(posixpath.dirname(self.cwd.rstrip("/")) or "/")

    def _fill_places(self):
        self.places.clear()
        p = profiles.get(self.host) if self.host else None
        marks = p.bookmarks if p else []
        recent = [u for u in load_recent() if self.host and u.startswith(f"{self.host}:")]
        self.places.addItem("Bookmarks, recent files" if marks or recent else "No bookmarks yet")
        for b in marks:
            self.places.addItem(f"★ {b}", ("dir", b))
        for u in recent:
            self.places.addItem(f"⟲ {u.split(':', 1)[1]}", ("file", u))
        self.places.setEnabled(bool(marks or recent))

    def _place(self, index):
        data = self.places.itemData(index)
        if not data:
            return
        kind, value = data
        if kind == "dir":
            self.browse(value)
        else:
            self.open(value)

    def bookmark(self):
        if not (self.host and self.cwd):
            return
        if profiles.get(self.host) is None:
            self.status.setText(f"Save {self.host} as a profile to keep bookmarks (right-click it).")
            return
        profiles.add_bookmark(self.host, self.cwd)
        self._fill_places()
        self.message.emit(f"Bookmarked {self.host}:{self.cwd}")

    # ------------------------------------------------------------ files
    def selected_url(self) -> str | None:
        items = self.files.selectedItems()
        if not items or self.cwd is None:
            return None
        name, is_dir = items[0].data(0, ROLE)
        return None if is_dir else self._loc(posixpath.join(self.cwd, name))

    def _activated(self, item):
        name, is_dir = item.data(0, ROLE)
        if is_dir:
            self.browse(posixpath.join(self.cwd, name))
        else:
            self.open(self._loc(posixpath.join(self.cwd, name)))

    def open(self, url: str):
        add_recent(url)
        self._fill_places()
        self.openRequested.emit(url)

    def download(self, url: str, folder: str | None = None, then_open: bool = False) -> str | None:
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, f"Download {posixpath.basename(url)} to")
            if not folder:
                return None
        _busy()
        self.status.setText(f"Downloading {posixpath.basename(url)}…")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        try:
            local = remote.download(url, folder)
        except Exception as exc:
            self.status.setText(f"✗ {type(exc).__name__}: {exc}")
            return None
        finally:
            _idle()
        self.status.setText(f"Downloaded to {local}")
        self.message.emit(f"Downloaded {url} to {local}")
        if then_open:
            self.downloadedOpen.emit(local)
        return local

    def _file_menu(self, pos):
        item = self.files.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        if item:
            name, is_dir = item.data(0, ROLE)
            path = posixpath.join(self.cwd, name)
            url = self._loc(path)
            clip = QtWidgets.QApplication.clipboard()
            if is_dir:
                menu.addAction("Open folder", lambda: self.browse(path))
                menu.addAction("Open as NEB run", lambda: self.nebRequested.emit(url))
                menu.addAction("Bookmark folder",
                               lambda: (profiles.add_bookmark(self.host, path), self._fill_places()))
            else:
                menu.addAction("Open (read in place)", lambda: self.open(url))
                menu.addAction("Download and open local copy…", lambda: self.download(url, then_open=True))
                menu.addAction("Download to…", lambda: self.download(url))
            menu.addSeparator()
            menu.addAction("Copy path", lambda: clip.setText(url))
            menu.addAction("Copy Python: load(…)", lambda: clip.setText(f"pipe = load({url!r})"))
        menu.exec(self.files.viewport().mapToGlobal(pos))



def install(window) -> QtWidgets.QDockWidget:
    """Add the Remotes side bar to ``window`` (hidden; the activity bar or File › Remote computers shows it)."""
    panel = window.remotes_panel = RemotesPanel()
    panel.openRequested.connect(window._open_asking_types)
    panel.nebRequested.connect(window.open_neb)
    panel.downloadedOpen.connect(window.open_any)
    panel.message.connect(window.message)
    dock = window._dock(panel, "Remotes", QtCore.Qt.LeftDockWidgetArea, 320)
    beside = getattr(window, "notebook_dock", None)
    if beside is not None:
        window.splitDockWidget(dock, beside, QtCore.Qt.Horizontal)
    dock.hide()
    dock.visibilityChanged.connect(lambda shown: shown and panel.refresh())
    return dock
