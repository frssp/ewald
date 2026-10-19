"""Browse and open files on another machine over SSH (File › Load remote file…)."""
from __future__ import annotations

import json
import posixpath
import time

from PySide6 import QtCore, QtWidgets

from ..io import remote

MAX_RECENT = 12


def recent_path():
    from .. import settings

    return settings.config_dir() / "remote_recent.json"


def load_recent() -> list[str]:
    """Recently opened remote files (locations only; credentials are never stored)."""
    try:
        return [str(u) for u in json.loads(recent_path().read_text())][:MAX_RECENT]
    except (OSError, ValueError, TypeError):
        return []


def add_recent(url: str) -> None:
    items = [url] + [u for u in load_recent() if u != url]
    try:
        recent_path().parent.mkdir(parents=True, exist_ok=True)
        recent_path().write_text(json.dumps(items[:MAX_RECENT]))
    except OSError:
        pass


def install_prompts(parent) -> None:
    """Ask SSH passwords / one-time codes and unknown host keys with dialogs; show download progress."""

    def password(prompt, echo):
        mode = QtWidgets.QLineEdit.Normal if echo else QtWidgets.QLineEdit.Password
        QtWidgets.QApplication.restoreOverrideCursor()
        text, ok = QtWidgets.QInputDialog.getText(parent, "SSH login", prompt.strip(), mode)
        return text if ok else None

    def host_key(host, kind, fingerprint):
        QtWidgets.QApplication.restoreOverrideCursor()
        answer = QtWidgets.QMessageBox.question(
            parent, "Unknown SSH host",
            f"The authenticity of host {host} can't be established.\n\n{kind} key fingerprint:\n{fingerprint}\n\n"
            "Connect and remember this host in known_hosts?")
        return answer == QtWidgets.QMessageBox.Yes

    state = {"dlg": None}

    def progress(done, total, label):
        dlg = state["dlg"]
        if dlg is None or dlg.labelText() != label:
            dlg = state["dlg"] = QtWidgets.QProgressDialog(f"Downloading {label}", "Cancel", 0, 1000, parent)
            dlg.setWindowTitle("Remote file")
            dlg.setMinimumDuration(500)
        dlg.setValue(int(1000 * done / total) if total else 1000)
        QtWidgets.QApplication.processEvents()
        if dlg.wasCanceled():
            state["dlg"] = None
            raise remote.RemoteError(f"download of {label} cancelled")
        if total and done >= total:
            dlg.reset()
            state["dlg"] = None

    remote.set_password_prompt(password)
    remote.set_host_key_prompt(host_key)
    remote.set_progress_callback(progress)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


class RemoteFileDialog(QtWidgets.QDialog):
    """Pick a file on an SSH host: saved profiles and ~/.ssh/config hosts, a directory browser, recent files."""

    def __init__(self, parent=None, host: str = "", path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Load remote file (SSH)")
        self.resize(720, 520)
        self.url = None
        recent = load_recent()

        self.recent = QtWidgets.QComboBox()
        self.recent.addItem("Recent…" if recent else "No recent remote files")
        self.recent.addItems(recent)
        self.recent.setEnabled(bool(recent))
        self.recent.activated.connect(self._pick_recent)

        self.host = QtWidgets.QComboBox(editable=True)
        from ..io import profiles

        hosts = profiles.names() + [h for h in remote.config_hosts() if h not in profiles.names()]
        for u in recent:
            rp = remote.parse_remote(u)
            name = f"{rp.user}@{rp.host}" if rp and rp.user else rp.host if rp else None
            if name and name not in hosts:
                hosts.append(name)
        self.host.addItems(hosts)
        self.host.setEditText(host or (hosts[0] if hosts else ""))
        self.host.lineEdit().setPlaceholderText("saved remote computer, ~/.ssh/config alias, or user@host")
        connect = QtWidgets.QPushButton("Connect")
        connect.clicked.connect(lambda: self.browse(""))

        self.path = QtWidgets.QLineEdit(path)
        self.path.setPlaceholderText("/scratch/me/run1   (empty = home)")
        self.path.returnPressed.connect(lambda: self.browse(self.path.text()))
        up = QtWidgets.QPushButton("Up")
        up.clicked.connect(self._up)

        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderLabels(["Name", "Size", "Modified"])
        self.list.setRootIsDecorated(False)
        self.list.setColumnWidth(0, 380)
        self.list.itemActivated.connect(self._activated)
        self.list.itemSelectionChanged.connect(self._selection)

        self.status = QtWidgets.QLabel(objectName="muted")
        self.status.setWordWrap(True)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QtWidgets.QDialogButtonBox.Open).setEnabled(False)

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("Host"), 0, 0)
        form.addWidget(self.host, 0, 1)
        form.addWidget(connect, 0, 2)
        form.addWidget(QtWidgets.QLabel("Folder"), 1, 0)
        form.addWidget(self.path, 1, 1)
        form.addWidget(up, 1, 2)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.recent)
        lay.addLayout(form)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.status)
        lay.addWidget(self.buttons)
        self.cwd = None

    # --- browsing
    def _location(self, path: str) -> str:
        return f"{self.host.currentText().strip()}:{path}"

    def browse(self, path: str) -> bool:
        host = self.host.currentText().strip()
        if not host:
            self.status.setText("Enter a host.")
            return False
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if not path:
                path = remote.home(self._location("."))
            elif not path.startswith("/"):
                path = posixpath.join(self.cwd or remote.home(self._location(".")), path)
            rows = remote.listdir(self._location(path))
        except Exception as exc:
            self.status.setText(f"{type(exc).__name__}: {exc}")
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.cwd = posixpath.normpath(path)
        self.path.setText(self.cwd)
        self.list.clear()
        for name, is_dir, size, mtime in rows:
            if name.startswith("."):
                continue
            item = QtWidgets.QTreeWidgetItem([name + ("/" if is_dir else ""), "" if is_dir else _size(size),
                                              time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))])
            item.setData(0, QtCore.Qt.UserRole, (name, is_dir))
            item.setIcon(0, self.style().standardIcon(
                QtWidgets.QStyle.SP_DirIcon if is_dir else QtWidgets.QStyle.SP_FileIcon))
            self.list.addTopLevelItem(item)
        self.status.setText(f"{sum(1 for r in rows if not r[0].startswith('.'))} entries in {host}:{self.cwd}")
        return True

    def _up(self):
        if self.cwd:
            self.browse(posixpath.dirname(self.cwd.rstrip("/")) or "/")

    def _activated(self, item):
        name, is_dir = item.data(0, QtCore.Qt.UserRole)
        if is_dir:
            self.browse(posixpath.join(self.cwd, name))
        else:
            self._accept()

    def _selection(self):
        items = self.list.selectedItems()
        ok = bool(items) and not items[0].data(0, QtCore.Qt.UserRole)[1]
        self.buttons.button(QtWidgets.QDialogButtonBox.Open).setEnabled(ok)

    def _pick_recent(self, index):
        if index > 0:
            self.url = self.recent.itemText(index)
            add_recent(self.url)
            self.accept()

    def _accept(self):
        items = self.list.selectedItems()
        if not items or self.cwd is None:
            return
        name, is_dir = items[0].data(0, QtCore.Qt.UserRole)
        if is_dir:
            self.browse(posixpath.join(self.cwd, name))
            return
        self.url = self._location(posixpath.join(self.cwd, name))
        add_recent(self.url)
        self.accept()
