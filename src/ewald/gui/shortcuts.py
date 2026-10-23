"""Help › Keyboard shortcuts: every menu shortcut, the notebook's command-mode keys and the
structure view's keys, written the platform's way (Ctrl on Windows and Linux, ⌘ on macOS)."""
from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from .palette import _label, native_keys

# keys that belong to no menu entry: (section, what it does, keys)
OTHER_KEYS = (
    ("Notebook cell", "Run the cell", "Ctrl+Return"),
    ("Notebook cell", "Run the cell and go to the next", "Shift+Return"),
    ("Notebook cell", "Leave the editor (command mode)", "Esc"),
    ("Command mode (Esc first)", "Edit the cell", "Return"),
    ("Command mode (Esc first)", "Insert a cell above / below", "A / B"),
    ("Command mode (Esc first)", "Code cell / text cell", "Y / M"),
    ("Command mode (Esc first)", "Cut / copy / paste below / paste above", "X / C / V / Shift+V"),
    ("Command mode (Esc first)", "Delete the cells / bring them back", "D, D / Z"),
    ("Command mode (Esc first)", "Merge the selected cells", "Shift+M"),
    ("Command mode (Esc first)", "Previous / next cell", "Up / Down (K / J)"),
    ("Command mode (Esc first)", "Extend the selection", "Shift+Up / Shift+Down"),
    ("Structure view", "Play or pause the trajectory", "Space"),
    ("Structure view", "Clear picks, measurements and selection", "Esc"),
    ("Structure view", "Pick several atoms (Pick tool)", "Shift+click"),
    ("Minimal mode", "Show the menu bar once", "Alt"),
)


def _native(keys: str) -> str:
    """Each key sequence of ``keys`` ("A / B", "D, D") the platform's way; words stay as they are."""
    parts = []
    for part in keys.split(" / "):
        seq = QtGui.QKeySequence(part)
        text = native_keys(part) if "click" not in part and not seq.isEmpty() else part
        parts.append(text or part)
    return " / ".join(parts)


def menu_shortcuts(menubar: QtWidgets.QMenuBar) -> list[tuple[str, str, str]]:
    """(menu path, entry, keys) of every menu bar entry that has a shortcut."""
    out = []

    def walk(menu, path):
        for act in menu.actions():
            if act.menu() is not None:
                walk(act.menu(), path + [_label(act.text())])
            elif act.text() and not act.shortcut().isEmpty():
                keys = " / ".join(k.toString(QtGui.QKeySequence.NativeText) for k in act.shortcuts())
                out.append((" › ".join(path), _label(act.text()), keys))

    for act in menubar.actions():
        if act.menu() is not None:
            walk(act.menu(), [_label(act.text())])
    return out


class ShortcutsDialog(QtWidgets.QDialog):
    """A searchable table of the shortcuts."""

    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(620, 560)
        lay = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit(placeholderText="Search shortcuts…")
        self.tree = QtWidgets.QTreeWidget(rootIsDecorated=False, alternatingRowColors=True)
        self.tree.setHeaderLabels(["Where", "Command", "Keys"])
        lay.addWidget(self.search)
        lay.addWidget(self.tree, 1)
        rows = menu_shortcuts(window.menuBar()) + [(s, c, _native(k)) for s, c, k in OTHER_KEYS]
        for where, command, keys in rows:
            QtWidgets.QTreeWidgetItem(self.tree, [where, command, keys])
        for col in range(3):
            self.tree.resizeColumnToContents(col)
        self.search.textChanged.connect(self._filter)
        self.search.setFocus()

    def rows(self) -> list[tuple[str, str, str]]:
        return [tuple(self.tree.topLevelItem(i).text(c) for c in range(3)) for i in range(self.tree.topLevelItemCount())]

    def _filter(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(bool(text) and not any(text in item.text(c).lower() for c in range(3)))

