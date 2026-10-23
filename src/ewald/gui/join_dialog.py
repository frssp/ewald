"""The dialog that opens several data files as one trajectory: reading order and how to join."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

JOINS = {
    "timestep": ("Segments of one restarted run: join on Timestep",
                 "Where a later file starts at a step an earlier one already reached, the later file "
                 "wins and the earlier frames from that step on are left out (LAMMPS dumps; VASP runs "
                 "are appended). A dump a restarted run appended to is joined the same way."),
    "concat": ("Keep every frame of every file",
               "The files are read one after the other with nothing left out."),
}


def display_names(paths, pathmod=os.path) -> list[str]:
    """Short names for ``paths``: relative to their common folder, so ``run1/dump.lammpstrj`` and
    ``run2/dump.lammpstrj`` stay apart. Full paths when they share none (Windows drives)."""
    if len(paths) < 2:
        return [pathmod.basename(p) for p in paths]
    try:
        common = pathmod.commonpath([pathmod.dirname(p) for p in paths])
    except ValueError:  # C:\ and D:\, or relative and absolute
        return list(paths)
    return [pathmod.relpath(p, common) if common else p for p in paths]


class JoinFilesDialog(QtWidgets.QDialog):
    """Several files as one trajectory: drag (or Move up/down) them into reading order and pick
    ``join``; the result is ``load([...], join=...)``."""

    def __init__(self, paths, join: str = "timestep", parent=None):
        from ..io.base import natural_key

        super().__init__(parent)
        self.setWindowTitle("Open as one trajectory")
        self.setObjectName("joinFilesDialog")
        self._natural_key = natural_key
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(f"Read these {len(paths)} files one after the other, top first:"))
        row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for name, path in zip(display_names(list(paths)), paths):
            item = QtWidgets.QListWidgetItem(name)
            item.setToolTip(path)
            item.setData(QtCore.Qt.UserRole, path)
            self.list.addItem(item)
        self.list.setCurrentRow(0)
        row.addWidget(self.list, 1)
        buttons = QtWidgets.QVBoxLayout()
        for text, slot, tip in (("Move up", lambda: self.move(-1), "Read the selected file earlier (Alt+Up)"),
                                ("Move down", lambda: self.move(1), "Read the selected file later (Alt+Down)"),
                                ("Natural order", self.sort_natural, "dump.200 before dump.1000, run2 before run10"),
                                ("Reverse", self.reverse, "Reverse the order")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        row.addLayout(buttons)
        lay.addLayout(row)
        for key, delta in (("Alt+Up", -1), ("Alt+Down", 1)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self.list, activated=lambda d=delta: self.move(d))

        self.join_buttons: dict[str, QtWidgets.QRadioButton] = {}
        group = QtWidgets.QButtonGroup(self)
        for key, (label, tip) in JOINS.items():
            b = QtWidgets.QRadioButton(label)
            b.setToolTip(tip)
            group.addButton(b)
            self.join_buttons[key] = b
            lay.addWidget(b)
        self.join_buttons.get(join, self.join_buttons["timestep"]).setChecked(True)
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        group.buttonToggled.connect(lambda *_: self._show_hint())
        self._show_hint()

        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.resize(560, 360)

    def _show_hint(self):
        self.hint.setText(JOINS[self.join()][1])

    def paths(self) -> list[str]:
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())]

    def join(self) -> str:
        return next(k for k, b in self.join_buttons.items() if b.isChecked())

    def move(self, delta: int):
        row = self.list.currentRow()
        to = row + delta
        if row < 0 or not 0 <= to < self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(to, item)
        self.list.setCurrentRow(to)

    def _set_order(self, paths):
        items = {self.list.item(i).data(QtCore.Qt.UserRole): self.list.item(i) for i in range(self.list.count())}
        while self.list.count():
            self.list.takeItem(0)
        for p in paths:
            self.list.addItem(items[p])

    def sort_natural(self):
        self._set_order(sorted(self.paths(), key=self._natural_key))

    def reverse(self):
        self._set_order(self.paths()[::-1])


def ask_join(parent, paths, join: str = "timestep") -> tuple[list[str], str] | None:
    """(files in reading order, join) from the dialog, or None when it was cancelled."""
    dialog = JoinFilesDialog(paths, join, parent)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.paths(), dialog.join()
