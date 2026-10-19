"""Notebook tabs along the top of the notebook panel, like browser tabs: one per open
notebook, ``+`` for a new one, right-click to rename or close."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class NotebookTabs(QtWidgets.QTabWidget):
    newRequested = QtCore.Signal()
    renameRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent, documentMode=True, tabsClosable=True, movable=True)
        self.setObjectName("notebookTabs")
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setUsesScrollButtons(True)
        new = QtWidgets.QToolButton(text="+", toolTip="New notebook (Ctrl+T)", objectName="tabNew")
        new.clicked.connect(self.newRequested.emit)
        self.setCornerWidget(new, QtCore.Qt.TopRightCorner)
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
