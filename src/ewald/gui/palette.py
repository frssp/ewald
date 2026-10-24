"""Command palette (Ctrl+K): fuzzy search over menu actions and modifiers."""
from __future__ import annotations

import re
import sys

from PySide6 import QtCore, QtGui, QtWidgets


def native_keys(keys: str) -> str:
    """``keys`` as this platform writes them (Ctrl+K on Windows and Linux, ⌘K on macOS)."""
    return QtGui.QKeySequence(keys).toString(QtGui.QKeySequence.NativeText)


_KEYS = re.compile(r"\b(?:Ctrl|Alt|Shift|Meta)\+[^\s,;)]+")


def native_text(text: str) -> str:
    """``text`` with its shortcuts ("Run all (Ctrl+Shift+Enter)") written the platform's way: on
    macOS ⇧⌘↩ style, elsewhere unchanged. For tooltips and labels that name a shortcut."""
    if sys.platform != "darwin":
        return text
    return _KEYS.sub(lambda m: native_keys(m.group(0)) or m.group(0), text)


def fuzzy_score(query: str, text: str):
    """Subsequence match score (higher is better), None if no match."""
    q, t = query.lower(), text.lower()
    if not q:
        return 0
    if q in t:
        return 100 - t.index(q)  # contiguous match wins, earlier is better
    initials = "".join(word[0] for word in re.findall(r"[a-z0-9]+", t))
    if len(q) > 1 and q in initials:
        return 90 - initials.index(q)  # "msd": Mean squared displacement
    pos, score, last = 0, 0, -2
    for ch in q:
        found = t.find(ch, pos)
        if found < 0:
            return None
        score += 5 if found == last + 1 else 1
        last, pos = found, found + 1
    return score


RECENT: list[str] = []   # labels run from the palette, newest first (kept for the session)
MAX_RECENT = 8


class CommandPalette(QtWidgets.QDialog):
    WIDTH = 560

    def __init__(self, entries, parent=None, placeholder="Search commands and steps…"):
        """``entries``: list of (label, kind, callable)."""
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.entries = entries
        self.setMinimumWidth(self.WIDTH)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.edit = QtWidgets.QLineEdit(placeholderText=placeholder)
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._run)
        self.list.itemActivated.connect(lambda _: self._run())
        self.edit.installEventFilter(self)
        self._filter("")

    def place_under(self, anchor: QtWidgets.QWidget, width: int | None = None, centered=True):
        """Open right below ``anchor`` (the toolbar's search box, the notebook's + Step), centered
        on it or left-aligned, at least as wide as it, and kept inside ``anchor``'s window."""
        width = max(width or self.WIDTH, anchor.width())
        self.setFixedWidth(width)
        bottom = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + 2))
        x = bottom.x() + (anchor.width() - width) // 2 if centered else bottom.x()
        win = anchor.window().geometry()
        x = max(win.left() + 4, min(x, win.right() - width - 4))
        self.move(x, bottom.y())

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
            row = self.list.currentRow() + (1 if event.key() == QtCore.Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        return super().eventFilter(obj, event)

    def _filter(self, text):
        scored = [(fuzzy_score(text, label), i) for i, (label, kind, fn) in enumerate(self.entries)]
        if text:
            scored = sorted((s, i) for s, i in scored if s is not None)[::-1]
        else:   # the commands used last first (this session), then the menus in order
            rank = {label: n for n, label in enumerate(RECENT)}
            order = sorted(range(len(self.entries)), key=lambda i: rank.get(self.entries[i][0], len(rank)))
            scored = [(0, i) for i in order]
        self.list.clear()
        for _, i in scored[:40]:
            label, kind, _fn = self.entries[i]
            it = QtWidgets.QListWidgetItem(f"{label}    ·  {kind}")
            it.setData(QtCore.Qt.UserRole, i)
            self.list.addItem(it)
        self.list.setCurrentRow(0)

    def _run(self):
        it = self.list.currentItem()
        if it is None:
            return
        self.accept()
        label, _kind, fn = self.entries[it.data(QtCore.Qt.UserRole)]
        if label in RECENT:
            RECENT.remove(label)
        RECENT.insert(0, label)
        del RECENT[MAX_RECENT:]
        fn()


class CommandCenter(QtWidgets.QWidget):
    """The rest of the main toolbar, with the search box (``box``, opens the palette) in the middle
    of the window as in VS Code; when the buttons left of it leave too little room, it moves right."""
    WIDTH = 480

    def __init__(self, slot, parent=None):
        super().__init__(parent, objectName="toolbarSpacer")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.box = QtWidgets.QPushButton(f"  Search commands and steps…      {native_keys('Ctrl+K')}", self,
                                         objectName="searchBox", toolTip=native_text("Command palette (Ctrl+K)"))
        self.box.clicked.connect(slot)
        self.setMinimumWidth(240)

    def sizeHint(self):
        return QtCore.QSize(self.WIDTH, self.box.sizeHint().height())

    def minimumSizeHint(self):
        return QtCore.QSize(240, self.box.sizeHint().height())

    def place(self):
        width = min(self.WIDTH, self.width() - 8)
        center = self.window().width() // 2 - self.mapTo(self.window(), QtCore.QPoint(0, 0)).x()
        x = max(0, min(center - width // 2, self.width() - width))
        self.box.setGeometry(x, (self.height() - self.box.sizeHint().height()) // 2, width, self.box.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.place()

    def moveEvent(self, event):
        super().moveEvent(event)
        self.place()


def _label(text: str) -> str:
    """Menu text without mnemonics or a tab-separated key hint; a literal "&&" stays "&"."""
    return text.split("\t")[0].replace("&&", "\0").replace("&", "").replace("\0", "&")


def menu_entries(menubar: QtWidgets.QMenuBar | QtWidgets.QMenu):
    """All enabled actions under a menu bar (or one menu: its submenus are the paths) as (label,
    'File › ...' and the shortcut, callable)."""
    out = []

    def walk(menu, path):
        for act in menu.actions():
            if act.menu():
                walk(act.menu(), path + [_label(act.text())])
            elif act.text() and not act.isSeparator() and act.isEnabled():
                where = " › ".join(path)
                if not act.shortcut().isEmpty():   # shown after the menu path, as the platform writes it
                    where += "   " + act.shortcut().toString(QtGui.QKeySequence.NativeText)
                out.append((_label(act.text()), where, act.trigger))

    if isinstance(menubar, QtWidgets.QMenu):
        walk(menubar, [])
        return out
    for act in menubar.actions():
        if act.menu():
            walk(act.menu(), [_label(act.text())])
    return out
