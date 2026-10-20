"""Notebook tabs along the top of the notebook panel, like browser tabs: one per open
notebook, ``+`` for a new one, right-click to rename or close.

The tab row (:attr:`NotebookTabs.row`) is a separate widget so that it can be the Notebook dock's
title bar (``DockTitleBar.embed``): dragging the empty part of the row moves the panel like any
other panel header, while tabs keep their own clicks and reordering. Double-clicking the empty part
opens a new notebook, as in VS Code.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from . import header


class _TabBar(QtWidgets.QTabBar):
    newRequested = QtCore.Signal()

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        size.setHeight(header.height())  # one row height with the panel headers above and beside it
        return size

    def mousePressEvent(self, event):
        if self.tabAt(event.position().toPoint()) < 0:
            event.ignore()  # empty part of the row: the dock title bar starts a panel drag
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.tabAt(event.position().toPoint()) < 0:
            self.newRequested.emit()
            return
        super().mouseDoubleClickEvent(event)


class NotebookTabs(QtWidgets.QWidget):
    """A tab row and a page stack with the part of the ``QTabWidget`` API the notebook panel uses."""

    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)
    newRequested = QtCore.Signal()
    renameRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("notebookTabs")
        bar = self.bar = _TabBar(objectName="notebookTabBar")
        bar.setDocumentMode(True)
        bar.setTabsClosable(True)
        bar.setMovable(True)
        bar.setDrawBase(False)
        bar.setExpanding(False)
        bar.setElideMode(QtCore.Qt.ElideRight)
        bar.setUsesScrollButtons(True)
        bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.stack = QtWidgets.QStackedWidget(objectName="notebookPages")
        self.new_button = new = header.button("plus", "New notebook (Ctrl+T)")
        new.clicked.connect(self.newRequested.emit)
        self.row = QtWidgets.QWidget(objectName="notebookTabRow")
        row = self.row_layout = QtWidgets.QHBoxLayout(self.row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(bar, 1)
        row.addWidget(new, 0, QtCore.Qt.AlignVCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.row)  # until the dock takes the row as its title bar
        lay.addWidget(self.stack, 1)
        bar.currentChanged.connect(self._current_changed)
        bar.tabCloseRequested.connect(self.tabCloseRequested.emit)
        bar.tabMoved.connect(self._moved)
        bar.newRequested.connect(self.newRequested.emit)
        bar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._menu)
        bar.tabBarDoubleClicked.connect(lambda i: i >= 0 and self.renameRequested.emit(i))

    # --- QTabWidget-like API
    def tabBar(self) -> QtWidgets.QTabBar:
        return self.bar

    def addTab(self, widget, text: str) -> int:
        self.stack.addWidget(widget)  # first, so the bar's currentChanged finds the page
        return self.bar.addTab(text)

    def removeTab(self, index: int):
        widget = self.stack.widget(index)
        if widget is None:
            return
        self.stack.removeWidget(widget)
        self.bar.removeTab(index)
        self.stack.setCurrentIndex(self.bar.currentIndex())

    def count(self) -> int:
        return self.bar.count()

    def widget(self, index: int):
        return self.stack.widget(index)

    def indexOf(self, widget) -> int:
        return self.stack.indexOf(widget)

    def currentIndex(self) -> int:
        return self.bar.currentIndex()

    def currentWidget(self):
        return self.stack.widget(self.bar.currentIndex())

    def setCurrentIndex(self, index: int):
        self.bar.setCurrentIndex(index)

    def setCurrentWidget(self, widget):
        i = self.stack.indexOf(widget)
        if i >= 0:
            self.bar.setCurrentIndex(i)

    def tabText(self, index: int) -> str:
        return self.bar.tabText(index)

    def setTabText(self, index: int, text: str):
        self.bar.setTabText(index, text)

    def setTabToolTip(self, index: int, tip: str):
        self.bar.setTabToolTip(index, tip)

    # ---
    def _current_changed(self, index: int):
        self.stack.setCurrentIndex(index)
        self.currentChanged.emit(index)

    def _moved(self, frm: int, to: int):
        widget = self.stack.widget(frm)
        self.stack.removeWidget(widget)
        self.stack.insertWidget(to, widget)
        self.stack.setCurrentIndex(self.bar.currentIndex())

    def _menu(self, pos):
        i = self.bar.tabAt(pos)
        if i < 0:
            return
        m = QtWidgets.QMenu(self)
        m.addAction("Rename…", lambda: self.renameRequested.emit(i))
        m.addAction("Close", lambda: self.tabCloseRequested.emit(i))
        m.exec(self.bar.mapToGlobal(pos))
