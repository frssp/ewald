"""The left rail of the notebook layout: one button per open notebook, like browser tabs on
their side. Speaks the part of the ``QTabWidget`` API that :class:`NotebookPanel` uses."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

RAIL_WIDTH = 84


class NotebookRail(QtWidgets.QWidget):
    """A vertical rail of notebook buttons next to a stack of pages."""

    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)
    newRequested = QtCore.Signal()
    renameRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.rail = QtWidgets.QFrame(objectName="notebookRail")
        self.rail.setFixedWidth(RAIL_WIDTH)
        rl = QtWidgets.QVBoxLayout(self.rail)
        rl.setContentsMargins(6, 8, 6, 8)
        rl.setSpacing(4)
        self.buttons_box = QtWidgets.QVBoxLayout()
        self.buttons_box.setSpacing(4)
        rl.addLayout(self.buttons_box)
        self.new_button = QtWidgets.QToolButton(text="+", toolTip="New notebook (Ctrl+T)", objectName="railNew")
        self.new_button.clicked.connect(self.newRequested.emit)
        rl.addWidget(self.new_button, 0, QtCore.Qt.AlignHCenter)
        rl.addStretch()
        self.stack = QtWidgets.QStackedWidget()
        lay.addWidget(self.rail)
        lay.addWidget(self.stack, 1)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: list[QtWidgets.QToolButton] = []
        self._texts: list[str] = []
        self.stack.currentChanged.connect(self._stack_changed)

    # ------------------------------------------------------------ QTabWidget subset
    def count(self) -> int:
        return self.stack.count()

    def widget(self, index):
        return self.stack.widget(index)

    def indexOf(self, widget) -> int:
        return self.stack.indexOf(widget)

    def currentIndex(self) -> int:
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.stack.currentWidget()

    def setCurrentIndex(self, index: int):
        self.stack.setCurrentIndex(index)

    def setCurrentWidget(self, widget):
        self.stack.setCurrentWidget(widget)

    def addTab(self, widget, text: str) -> int:
        b = QtWidgets.QToolButton(checkable=True, objectName="railTab")
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        b.setFixedWidth(RAIL_WIDTH - 12)
        b.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        b.customContextMenuRequested.connect(lambda _pos, b=b: self._menu(b))
        b.clicked.connect(lambda _=False, b=b: self.setCurrentIndex(self._buttons.index(b)))
        self.group.addButton(b)
        self._buttons.append(b)
        self._texts.append("")
        self.buttons_box.addWidget(b)
        i = self.stack.addWidget(widget)
        self.setTabText(i, text)
        if self.stack.count() == 1:
            self._stack_changed(0)
        return i

    def removeTab(self, index: int):
        if not 0 <= index < self.count():
            return
        b = self._buttons.pop(index)
        self._texts.pop(index)
        self.group.removeButton(b)
        self.buttons_box.removeWidget(b)
        b.hide()
        b.deleteLater()
        self.stack.removeWidget(self.stack.widget(index))

    def setTabText(self, index: int, text: str):
        self._texts[index] = text
        b = self._buttons[index]
        name, dirty = (text[:-2], True) if text.endswith(" ●") else (text, False)
        b.setText(_wrap(name, b.fontMetrics(), RAIL_WIDTH - 20) + ("\n●" if dirty else ""))
        b.setProperty("dirty", dirty)
        b.style().unpolish(b)
        b.style().polish(b)

    def tabText(self, index: int) -> str:
        return self._texts[index]

    def setTabToolTip(self, index: int, tip: str):
        self._buttons[index].setToolTip(f"{self._texts[index].rstrip(' ●')}\n{tip}")

    def tabButton(self, index: int) -> QtWidgets.QToolButton:
        return self._buttons[index]

    # ------------------------------------------------------------ internals
    def _stack_changed(self, index: int):
        for i, b in enumerate(self._buttons):
            b.setChecked(i == index)
        self.currentChanged.emit(index)

    def _menu(self, button):
        i = self._buttons.index(button)
        m = QtWidgets.QMenu(self)
        m.addAction("Rename…", lambda: self.renameRequested.emit(i))
        m.addAction("Close", lambda: self.tabCloseRequested.emit(i))
        m.exec(button.mapToGlobal(QtCore.QPoint(button.width(), 0)))


def _wrap(name: str, metrics, width: int) -> str:
    """Up to two lines that fit ``width`` pixels, broken after a space, ``_``, ``-`` or ``.``
    where possible; the second line is elided."""
    if metrics.horizontalAdvance(name) <= width:
        return name
    n = 1
    while n < len(name) and metrics.horizontalAdvance(name[:n + 1]) <= width:
        n += 1
    cut = max((k + 1 for k, ch in enumerate(name[:n]) if ch in " _-."), default=n)
    rest = name[cut:].strip()
    return name[:cut].strip() + "\n" + metrics.elidedText(rest, QtCore.Qt.ElideRight, width)
