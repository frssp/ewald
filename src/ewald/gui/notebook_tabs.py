"""Notebook tabs along the top of the notebook panel, like browser tabs: one per open
notebook, ``+`` for a new one, right-click to rename or close."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from . import header


class _TabBar(QtWidgets.QTabBar):
    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        size.setHeight(header.height())  # one row height with the panel headers above and beside it
        return size



class _Corner(QtWidgets.QWidget):
    """Holder of ``+``: as tall as the tabs, since Qt sizes the corner from its size hint."""

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), header.height())


class NotebookTabs(QtWidgets.QTabWidget):
    newRequested = QtCore.Signal()
    renameRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(_TabBar(self))
        self.setDocumentMode(True)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabBar().setDrawBase(False)
        self.setObjectName("notebookTabs")
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setUsesScrollButtons(True)
        self.new_button = new = header.button("plus", "New notebook (Ctrl+T)")
        new.clicked.connect(self.newRequested.emit)
        corner = _Corner(objectName="tabCorner")
        lay = QtWidgets.QHBoxLayout(corner)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(new, 0, QtCore.Qt.AlignVCenter)
        self.setCornerWidget(corner, QtCore.Qt.TopRightCorner)
        bar = self.tabBar()
        bar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._menu)
        bar.tabBarDoubleClicked.connect(lambda i: i >= 0 and self.renameRequested.emit(i))

    def _menu(self, pos):
        i = self.tabBar().tabAt(pos)
        if i < 0:
            return
        m = QtWidgets.QMenu(self)
        m.addAction("Rename…", lambda: self.renameRequested.emit(i))
        m.addAction("Close", lambda: self.tabCloseRequested.emit(i))
        m.exec(self.tabBar().mapToGlobal(pos))
