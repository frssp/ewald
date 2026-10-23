"""How a notebook tab follows the cells the AI chat writes and runs (Preferences › AI › Follow
the AI's cells): ``follow`` scrolls to a written cell only when it is off screen and stops for
the rest of the answer once you scroll yourself; ``end`` scrolls once to the last cell when the
answer is done; ``never`` leaves the scroll alone. Reading cells never scrolls. Whenever the cell
is off screen and the notebook did not go there, a pill at the edge ("AI is on cell 7") jumps to it."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .icons import icon
from .theme import TOKENS

MODES = ("follow", "end", "never")


class AiFollow(QtCore.QObject):
    """One per :class:`~ewald.gui.notebook_panel.NotebookView`. The window calls :meth:`begin` and
    :meth:`end` around each answer; the notebook bridge calls :meth:`touched` for every cell the
    AI inserts, edits, moves or runs."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.mode = "follow"
        self.active = False      # an answer is on its way
        self.following = True    # False once you scrolled during this answer
        self.last = None         # the cell the AI wrote or ran last
        bar = view.scroll.verticalScrollBar()
        bar.actionTriggered.connect(self._scrolled)   # wheel, drag, keys: not programmatic scrolls
        bar.valueChanged.connect(lambda *_: self._update_pill())
        self.pill = QtWidgets.QToolButton(view.scroll, objectName="aiCellPill")
        self.pill.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.pill.setCursor(QtCore.Qt.PointingHandCursor)
        self.pill.clicked.connect(self.jump)
        self.pill.hide()
        view.scroll.viewport().installEventFilter(self)

    # ------------------------------------------------------------ answers
    def begin(self, mode: str = "follow"):
        self.mode = mode if mode in MODES else "follow"
        self.active, self.following, self.last = True, True, None
        self.pill.hide()

    def end(self):
        self.active = False
        if self.mode == "end" and self.last is not None:
            self.jump(flash=False)
        else:
            self._update_pill()

    def touched(self, cell):
        """The AI wrote or ran ``cell``: go there (``follow``) or point at it."""
        if cell is None or cell not in self.view.notebook.cells:
            return
        self.last = cell
        if self.mode == "follow" and self.following:
            QtCore.QTimer.singleShot(0, self.view, lambda: self._scroll_to(cell))   # after the layout
        else:
            QtCore.QTimer.singleShot(0, self.view, self._update_pill)

    def jump(self, flash: bool = True):
        """Show the last cell the AI touched (the pill, or ``end`` mode when the answer is done)."""
        cell, self.following = self.last, True
        self.pill.hide()
        if cell is None:
            return
        if flash:
            self.view.reveal(cell)
        else:
            QtCore.QTimer.singleShot(0, self.view, lambda: self._scroll_to(cell))

    # ------------------------------------------------------------ geometry
    def visible(self, cell) -> bool:
        """Whether enough of ``cell`` is on screen: its top, or half the view."""
        w = self.view.widget_for(cell)
        if w is None:
            return True
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.LayoutRequest)   # cells just rebuilt
        vp = self.view.scroll.viewport()
        top = w.mapTo(vp, QtCore.QPoint(0, 0)).y()
        shown = min(top + w.height(), vp.height()) - max(top, 0)
        return 0 <= top < vp.height() or shown >= min(w.height(), vp.height()) / 2

    def _scroll_to(self, cell):
        w = self.view.widget_for(cell)
        if w is not None and not self.visible(cell):
            self.view.scroll.ensureWidgetVisible(w, 0, 24)
        self._update_pill()

    def _scrolled(self, *_):
        if self.active:
            self.following = False

    def _update_pill(self):
        cell = self.last
        if cell is None or cell not in self.view.notebook.cells or self.visible(cell):
            self.pill.hide()
            return
        n = self.view.notebook.cells.index(cell) + 1
        w = self.view.widget_for(cell)
        vp = self.view.scroll.viewport()
        below = w is not None and w.mapTo(vp, QtCore.QPoint(0, 0)).y() > 0
        self.pill.setText(f"AI is on cell {n}" if self.active else f"AI edited cell {n}")
        self.pill.setIcon(icon("arrow_down" if below else "arrow_up", TOKENS["on_accent"]))
        self.pill.setToolTip("Show the cell the AI wrote or ran last")
        self.pill.adjustSize()
        x = (vp.width() - self.pill.width()) // 2 + vp.x()
        y = vp.y() + (vp.height() - self.pill.height() - 12 if below else 12)
        self.pill.move(max(vp.x(), x), y)
        self.pill.show()
        self.pill.raise_()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize and self.pill.isVisible():
            self._update_pill()
        return False
