"""Collapsible side panels: a dock title bar with a fold button.

A folded side panel (left or right) shrinks to a thin strip with its title written
vertically; a folded bottom panel keeps only its title bar. Docks shown as tabs fold together.
Clicking the strip, the expand chevron or double-clicking the title unfolds it at its old size.
The bar is a panel header (``header.py``): same height, inset and icon buttons as the tab-group heads.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import header

STRIP = 30  # width of a folded side panel, height of a folded bottom one
_MAX = 16777215  # QWIDGETSIZE_MAX


class _VerticalLabel(QtWidgets.QLabel):
    """A label drawn bottom-to-top, for the strip of a folded side panel."""

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        p.translate(0, self.height())
        p.rotate(-90)
        p.drawText(QtCore.QRect(0, 0, self.height(), self.width()), QtCore.Qt.AlignCenter, self.text())

    def sizeHint(self):
        s = super().sizeHint()
        return QtCore.QSize(s.height(), s.width())

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        return QtCore.QSize(s.height(), s.width())


class DockTitleBar(QtWidgets.QWidget):
    """Title bar of a collapsible dock: title, then fold and float buttons on the right (like the
    workbench's tab-group heads); folded sideways it becomes a strip with the expand button on top."""

    def __init__(self, dock: QtWidgets.QDockWidget):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName("panelHeader")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.collapsed = False
        self._size = None
        self.fold = header.button("chevron_down", f"Collapse the {dock.windowTitle()} panel")
        self.fold.clicked.connect(self.toggle)
        self.title = QtWidgets.QLabel(dock.windowTitle(), objectName="panelTitle")
        self.vtitle = _VerticalLabel(dock.windowTitle(), objectName="panelTitle")
        self.float_button = header.button("float", "Float or dock the panel")
        self.float_button.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        dock.windowTitleChanged.connect(lambda t: (self.title.setText(t), self.vtitle.setText(t)))
        dock.dockLocationChanged.connect(lambda _area: self._sync_icons())
        dock.topLevelChanged.connect(lambda _floating: self._layout(vertical=self.collapsed and self._side()))
        self.embedded = None  # a widget shown instead of the title, see embed
        self.extra = []  # panel actions shown before the fold button, see add_action
        self._layout(vertical=False)

    def add_action(self, button: QtWidgets.QAbstractButton):
        """Show a panel action (e.g. Display's add-visual menu) in the header, left of the fold button."""
        self.extra.append(button)
        self._layout(vertical=self.collapsed and self._side())

    def embed(self, widget: QtWidgets.QWidget):
        """Show ``widget`` (a row the panel would otherwise put at its top, such as the notebook
        tabs) in place of the title. Mouse presses it leaves unhandled start a panel drag as on any
        title bar; folded sideways the bar is still the strip with the panel's name."""
        self.embedded = widget
        self._layout(vertical=self.collapsed and self._side())

    def _sync_icons(self):
        header.set_icon(self.fold, self._fold_icon())

    def _fold_icon(self) -> str:
        """Chevron toward the dock's edge to fold, away from it to unfold; up/down for other docks."""
        window = self.dock.parentWidget()
        area = window.dockWidgetArea(self.dock) if isinstance(window, QtWidgets.QMainWindow) else None
        if area == QtCore.Qt.LeftDockWidgetArea:
            return "chevron_right" if self.collapsed else "chevron_left"
        if area == QtCore.Qt.RightDockWidgetArea:
            return "chevron_left" if self.collapsed else "chevron_right"
        return "chevron_up" if self.collapsed else "chevron_down"

    def _layout(self, vertical: bool):
        old = self.layout()
        if old is not None:
            while old.count():
                old.takeAt(0)
            QtWidgets.QWidget().setLayout(old)  # reparent so a new layout can be set
        h = header.height()
        if vertical:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            lay = QtWidgets.QVBoxLayout(self)
            pad = (STRIP - header.BUTTON) // 2
            lay.setContentsMargins(0, (h - header.BUTTON) // 2, 0, pad)
            lay.setSpacing(8)
        else:
            self.setFixedHeight(h)
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(0 if self.embedded else header.INSET, 0, 6, 0)
            lay.setSpacing(2)
        self.title.setVisible(not vertical and not self.embedded)
        if self.embedded:
            self.embedded.setVisible(not vertical)
        self.vtitle.setVisible(vertical)
        self.float_button.setVisible(not vertical)
        for b in self.extra:
            b.setVisible(not vertical)
        self._sync_icons()
        if vertical:
            lay.addWidget(self.fold, 0, QtCore.Qt.AlignHCenter)
            lay.addWidget(self.vtitle, 0, QtCore.Qt.AlignHCenter)
            lay.addStretch()
        else:
            lay.addWidget(self.embedded or self.title, 1, QtCore.Qt.AlignVCenter)
            for b in self.extra:
                lay.addWidget(b, 0, QtCore.Qt.AlignVCenter)
            lay.addWidget(self.fold, 0, QtCore.Qt.AlignVCenter)
            lay.addWidget(self.float_button, 0, QtCore.Qt.AlignVCenter)

    def mouseDoubleClickEvent(self, event):
        self.toggle()

    def mousePressEvent(self, event):
        if self.collapsed:  # the whole strip unfolds the panel
            self.toggle()
        else:
            super().mousePressEvent(event)

    def toggle(self):
        self.set_collapsed(not self.collapsed)

    def _group(self) -> list:
        window = self.dock.parentWidget()
        if not isinstance(window, QtWidgets.QMainWindow) or self.dock.isFloating():
            return [self.dock]
        return [self.dock] + list(window.tabifiedDockWidgets(self.dock))

    def _side(self) -> bool:
        window = self.dock.parentWidget()
        area = window.dockWidgetArea(self.dock) if isinstance(window, QtWidgets.QMainWindow) else None
        return area in (QtCore.Qt.LeftDockWidgetArea, QtCore.Qt.RightDockWidgetArea)

    def _watch_tabs(self):
        """Clicking a sideways tab of a folded group unfolds it."""
        window = self.dock.parentWidget()
        titles = {d.windowTitle() for d in self._group()}
        for tabs in window.findChildren(QtWidgets.QTabBar):
            if tabs.parent() is window and {tabs.tabText(i) for i in range(tabs.count())} == titles \
                    and not tabs.property("ewaldFoldWatch"):
                tabs.setProperty("ewaldFoldWatch", True)
                tabs.tabBarClicked.connect(lambda _i, t=tabs: self._tab_clicked(t))

    def _tab_clicked(self, tabs):
        for d in self._group():
            bar = d.titleBarWidget()
            if isinstance(bar, DockTitleBar) and bar.collapsed:
                bar.set_collapsed(False)
                return

    def set_collapsed(self, on: bool):
        if on == self.collapsed or (on and self.dock.isFloating()):
            return
        side = self._side()
        window = self.dock.parentWidget()
        for d in self._group():
            bar = d.titleBarWidget()
            if isinstance(bar, DockTitleBar) and d is not self.dock:
                bar.collapsed = on
                bar._layout(vertical=on and side)
            if on:
                if d is self.dock:
                    self._size = d.width() if side else d.height()
                _swap_content(d, folded=True)
                if side:
                    d.setFeatures(d.features() | QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
                    d.setFixedWidth(STRIP)
                else:
                    d.setFixedHeight(STRIP)
            else:
                d.setFeatures(d.features() & ~QtWidgets.QDockWidget.DockWidgetVerticalTitleBar)
                d.setMinimumSize(0, 0)
                d.setMaximumSize(_MAX, _MAX)
                _swap_content(d, folded=False)
        if side and len(self._group()) > 1:  # horizontal tabs would keep the area wide: turn them sideways
            area = window.dockWidgetArea(self.dock)
            if on:
                self._tab_position = window.tabPosition(area)
                window.setTabPosition(area, QtWidgets.QTabWidget.East if area == QtCore.Qt.RightDockWidgetArea
                                      else QtWidgets.QTabWidget.West)
                QtCore.QTimer.singleShot(0, self, self._watch_tabs)
            elif getattr(self, "_tab_position", None) is not None:
                window.setTabPosition(area, self._tab_position)
        self.collapsed = on
        self.fold.setToolTip(f"{'Expand' if on else 'Collapse'} the {self.dock.windowTitle()} panel")
        self._layout(vertical=on and side)
        if on and isinstance(window, QtWidgets.QMainWindow):  # give the space back once the layout settled
            orient = QtCore.Qt.Horizontal if side else QtCore.Qt.Vertical
            QtCore.QTimer.singleShot(0, self, lambda: window.resizeDocks([self.dock], [STRIP], orient))
        if not on and self._size and isinstance(window, QtWidgets.QMainWindow):
            window.resizeDocks([self.dock], [self._size], QtCore.Qt.Horizontal if side else QtCore.Qt.Vertical)


def _swap_content(dock, folded: bool):
    """Put an empty widget in a folded dock (a dock whose content is merely hidden sizes itself
    to its title bar) and the real content back when it unfolds."""
    if folded and not hasattr(dock, "_folded_content"):
        dock._folded_content = dock.widget()
        dock.setWidget(QtWidgets.QWidget())
        if dock._folded_content is not None:
            dock._folded_content.hide()
    elif not folded and hasattr(dock, "_folded_content"):
        placeholder, content = dock.widget(), dock.__dict__.pop("_folded_content")
        if content is not None:
            dock.setWidget(content)
            content.show()
        if placeholder is not None and placeholder is not content:
            placeholder.deleteLater()


def make_collapsible(dock: QtWidgets.QDockWidget) -> DockTitleBar:
    """Give ``dock`` a title bar with a fold button; returns the bar."""
    bar = DockTitleBar(dock)
    dock.setTitleBarWidget(bar)
    return bar
