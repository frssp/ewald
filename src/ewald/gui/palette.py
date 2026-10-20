"""Command palette (Ctrl+K): fuzzy search over menu actions and modifiers."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def fuzzy_score(query: str, text: str):
    """Subsequence match score (higher is better), None if no match."""
    q, t = query.lower(), text.lower()
    if not q:
        return 0
    if q in t:
        return 100 - t.index(q)  # contiguous match wins, earlier is better
    pos, score, last = 0, 0, -2
    for ch in q:
        found = t.find(ch, pos)
        if found < 0:
            return None
        score += 5 if found == last + 1 else 1
        last, pos = found, found + 1
    return score


class CommandPalette(QtWidgets.QDialog):
    def __init__(self, entries, parent=None):
        """``entries``: list of (label, kind, callable)."""
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.entries = entries
        self.setMinimumWidth(560)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.edit = QtWidgets.QLineEdit(placeholderText="Search commands, modifiers, views…")
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._run)
        self.list.itemActivated.connect(lambda _: self._run())
        self.edit.installEventFilter(self)
        self._filter("")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
            row = self.list.currentRow() + (1 if event.key() == QtCore.Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        return super().eventFilter(obj, event)

    def _filter(self, text):
        scored = [(fuzzy_score(text, label), i) for i, (label, kind, fn) in enumerate(self.entries)]
        scored = sorted((s, i) for s, i in scored if s is not None)[::-1] if text else [(0, i) for i in range(len(self.entries))]
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
        self.entries[it.data(QtCore.Qt.UserRole)][2]()


def _label(text: str) -> str:
    """Menu text without mnemonics; a literal "&&" stays "&"."""
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&")


def menu_entries(menubar: QtWidgets.QMenuBar):
    """All enabled actions under a menu bar as (label, 'File › ...', callable)."""
    out = []

    def walk(menu, path):
        for act in menu.actions():
            if act.menu():
                walk(act.menu(), path + [_label(act.text())])
            elif act.text() and not act.isSeparator() and act.isEnabled():
                out.append((_label(act.text()), " › ".join(path), act.trigger))

    for act in menubar.actions():
        if act.menu():
            walk(act.menu(), [_label(act.text())])
    return out
