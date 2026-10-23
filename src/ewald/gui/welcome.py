"""What the notebook area shows with no notebook open (first start, or the last tab closed),
like VS Code's empty editor: start a notebook, open a file or an example, reopen a recent file."""
from __future__ import annotations

import json
import os

from PySide6 import QtCore, QtWidgets

from .palette import native_keys, native_text

MAX_RECENT = 10


def recent_path():
    from .. import settings

    return settings.config_dir() / "recent_files.json"


def load_recent() -> list[str]:
    """Files and notebooks opened lately, newest first (paths or ``host:/path`` locations)."""
    try:
        return [str(p) for p in json.loads(recent_path().read_text(encoding="utf-8"))][:MAX_RECENT]
    except (OSError, ValueError, TypeError):
        return []


def clear_recent() -> None:
    """Forget the recent files (File › Open recent › Clear recent files)."""
    try:
        recent_path().write_text("[]", encoding="utf-8", newline="\n")
    except OSError:
        pass


def add_recent(path) -> None:
    path = str(path)
    if "://" not in path and ":" not in path[2:]:     # a local file (not host:/path): absolute
        path = os.path.abspath(path)
    items = [path] + [p for p in load_recent() if p != path]
    try:
        recent_path().parent.mkdir(parents=True, exist_ok=True)
        recent_path().write_text(json.dumps(items[:MAX_RECENT]), encoding="utf-8", newline="\n")
    except OSError:
        pass


class WelcomePage(QtWidgets.QWidget):
    """The empty state of the notebook area."""

    newRequested = QtCore.Signal()
    openRequested = QtCore.Signal()
    exampleRequested = QtCore.Signal(str)
    recentRequested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("welcome")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        outer.addLayout(col)
        outer.addStretch(2)
        title = QtWidgets.QLabel("No notebook open", objectName="welcomeTitle")
        col.addWidget(title)
        col.addWidget(QtWidgets.QLabel("Start a notebook, or open a structure, trajectory or notebook file.",
                                       objectName="muted", wordWrap=True))
        col.addSpacing(6)
        row = QtWidgets.QHBoxLayout()
        self.new_button = QtWidgets.QPushButton("New notebook", objectName="primary",
                                                toolTip=native_text("An empty notebook (Ctrl+T)"))
        self.new_button.clicked.connect(self.newRequested.emit)
        self.open_button = QtWidgets.QPushButton("Open…", toolTip=native_text("A data file, notebook or session (Ctrl+O)"))
        self.open_button.clicked.connect(self.openRequested.emit)
        self.example_button = QtWidgets.QPushButton("Open example", toolTip="One of the example files shipped with EWALD")
        menu = QtWidgets.QMenu(self.example_button)
        menu.aboutToShow.connect(lambda: self._fill_examples(menu))
        self.example_button.setMenu(menu)
        for b in (self.new_button, self.open_button, self.example_button):
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)
        col.addSpacing(10)
        self.recent_title = QtWidgets.QLabel("Recent", objectName="section")
        col.addWidget(self.recent_title)
        self.recent_box = QtWidgets.QVBoxLayout()
        self.recent_box.setSpacing(0)
        col.addLayout(self.recent_box)
        col.addSpacing(10)
        col.addWidget(QtWidgets.QLabel(f"Drop files here to open them  ·  {native_keys('Ctrl+K')} for every command",
                                       objectName="muted", wordWrap=True))
        self.refresh()

    def _fill_examples(self, menu):
        from .. import examples

        menu.clear()
        for ex in examples.EXAMPLES:
            act = menu.addAction(f"{ex.file}  ({ex.format})")
            act.setToolTip(ex.description)
            act.triggered.connect(lambda _=False, f=ex.file: self.exampleRequested.emit(f))

    def refresh(self):
        """Re-read the recent files (shown each time the page appears)."""
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        recent = load_recent()
        self.recent_title.setVisible(bool(recent))
        home = os.path.expanduser("~")
        for path in recent:
            row = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            b = QtWidgets.QToolButton(text=os.path.basename(path.rstrip("/")) or path, objectName="welcomeRecent",
                                      toolTip=path, autoRaise=True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, p=path: self.recentRequested.emit(p))
            folder = os.path.dirname(path)
            folder = "~" + folder[len(home):] if folder.startswith(home) else folder
            where = QtWidgets.QLabel(folder, objectName="muted")
            where.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # long paths clip
            lay.addWidget(b)
            lay.addWidget(where, 1)
            self.recent_box.addWidget(row)

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)
