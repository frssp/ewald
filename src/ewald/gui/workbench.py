"""The middle of the main window: tab groups you can drag around, as in VS Code.

Two areas, the editor area (3D view at the start) with the timeline under it and the panel
below (console, inspector, plots, log), each a split tree of :class:`TabGroup`\\ s drawn from a
:class:`~ewald.gui.layout_model.Layout`. Drag a tab onto another group's centre to add it
there, onto an edge to split, or onto a tab bar to insert it; Esc cancels. Every move also has
a menu entry and a shortcut (:meth:`Workbench.actions`).

Native GL widgets (the 3D viewport) are never reparented, which can break VTK and the ssh -X
OpenGL viewport: their group holds an empty slot and the widget itself, a child of the
workbench, follows the slot's geometry. A hidden one is parked (``set_parked``) so it builds no
scene, and while a tab is dragged it is replaced by a snapshot so the drop marks can draw over it.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from . import header
from .layout_model import Group, Layout, Split, zone_at
from .palette import native_text
from .theme import TOKENS


def _valid(obj) -> bool:
    """Whether the Qt object behind ``obj`` still exists."""
    try:
        obj.objectName()
        return True
    except RuntimeError:
        return False


@dataclass
class ViewSpec:
    id: str
    widget: QtWidgets.QWidget
    title: str
    home: str = "panel"         # the area it starts in and returns to
    closable: bool = False      # closing removes it (extra plot views); otherwise it goes home
    native: bool = False        # a GL widget: follows a slot instead of being reparented
    keep: bool = False          # closing a closable view keeps its widget (a cell's plot: it can come back)


class TabBar(QtWidgets.QTabBar):
    """A group's tab bar: click selects, drag starts a move, right click opens the view menu."""

    def __init__(self, group: "TabGroup"):
        super().__init__()
        self.group = group
        self.setDocumentMode(True)
        self.setExpanding(False)
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setUsesScrollButtons(True)
        self.setDrawBase(False)  # the panel header draws the line under the tabs
        self._press = None
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        size.setHeight(header.height())  # tabs fill the panel header, like the dock title bars
        return size

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            i = self.tabAt(e.position().toPoint())
            self._press = (e.position().toPoint(), i) if i >= 0 else None
        super().mousePressEvent(e)
        self.group.workbench._activate(self.group)

    def mouseMoveEvent(self, e):
        wb = self.group.workbench
        if wb._drag is not None:
            wb.drag_move(e.globalPosition().toPoint())
            return
        if self._press is not None and e.buttons() & QtCore.Qt.LeftButton:
            if (e.position().toPoint() - self._press[0]).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                view = self.tabData(self._press[1])
                self._press = None
                wb.begin_drag(view)
                wb.drag_move(e.globalPosition().toPoint())
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        wb = self.group.workbench
        if wb._drag is not None and e.button() == QtCore.Qt.LeftButton:
            wb.drag_drop(e.globalPosition().toPoint())
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.tabAt(e.position().toPoint()) >= 0:
            self.group.workbench.toggle_maximize(self.group.model)
        else:
            super().mouseDoubleClickEvent(e)

    def _context_menu(self, pos):
        i = self.tabAt(pos)
        if i < 0:
            return
        menu = QtWidgets.QMenu(self)
        self.group.workbench.fill_view_menu(menu, self.tabData(i))
        menu.exec(self.mapToGlobal(pos))


class TabGroup(QtWidgets.QWidget):
    """One leaf of the layout: a tab bar over a stack showing the current view."""

    def __init__(self, workbench: "Workbench", model: Group):
        super().__init__()
        self.workbench, self.model = workbench, model
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.head = QtWidgets.QWidget(objectName="panelHeader")
        self.head.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.head.setFixedHeight(header.height())
        head = QtWidgets.QHBoxLayout(self.head)
        head.setContentsMargins(0, 0, 6, 0)
        head.setSpacing(2)
        self.bar = TabBar(self)
        self.bar.currentChanged.connect(self._tab_changed)
        self.bar.tabCloseRequested.connect(lambda i: workbench.close_view(self.bar.tabData(i)))
        head.addWidget(self.bar, 1)
        self.buttons = {}
        for key, tip in (("split", native_text("Split the current view to the right (Ctrl+Alt+\\)")),
                         ("maximize", native_text("Maximize this group (Ctrl+Alt+M, or double-click a tab)")),
                         ("fold", native_text("Fold or unfold the panel (Ctrl+Alt+J)")),
                         ("more", "Move, split or close the current view")):
            b = header.button(key, tip, name="groupButton")
            head.addWidget(b, 0, QtCore.Qt.AlignVCenter)
            self.buttons[key] = b
        self.buttons["split"].clicked.connect(lambda: self.current_view() and workbench.split(self.current_view(), "right"))
        self.buttons["maximize"].clicked.connect(lambda: workbench.toggle_maximize(self.model))
        self.buttons["fold"].clicked.connect(lambda: workbench.set_panel_collapsed(not workbench.panel_collapsed))
        self.buttons["more"].clicked.connect(self._menu)
        lay.addWidget(self.head)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = QtWidgets.QLabel("Drag a tab here", alignment=QtCore.Qt.AlignCenter, objectName="muted")
        self.stack.addWidget(self.empty)
        lay.addWidget(self.stack, 1)
        self._syncing = False

    def current_view(self) -> str | None:
        return self.model.current_view

    def sync(self):
        """Show the model's views as tabs (the widgets are moved in, slots stand in for GL widgets)."""
        wb = self.workbench
        self._syncing = True
        try:
            while self.bar.count():
                self.bar.removeTab(0)
            while self.stack.count() > 1:
                self.stack.removeWidget(self.stack.widget(1))
            self.bar.setTabsClosable(any(wb.can_close(v) for v in self.model.views))
            for v in self.model.views:
                spec = wb.specs[v]
                self.stack.addWidget(wb.slot(v) if spec.native else spec.widget)
                i = self.bar.addTab(spec.title)
                self.bar.setTabData(i, v)
                self.bar.setTabToolTip(i, f"{spec.title}: drag to move, right click for more")
                if not wb.can_close(v):
                    self.bar.setTabButton(i, QtWidgets.QTabBar.RightSide, None)
            if self.model.views:
                cur = self.model.views.index(self.model.current_view)
                self.bar.setCurrentIndex(cur)
                self.stack.setCurrentIndex(cur + 1)
            else:
                self.stack.setCurrentIndex(0)
            area = wb.layout.area_of(self.model) if self.model in wb.layout.groups() else "editor"
            self.buttons["fold"].setVisible(area == "panel")
        finally:
            self._syncing = False

    def _tab_changed(self, i):
        if self._syncing or i < 0:
            return
        self.model.current = i
        self.stack.setCurrentIndex(i + 1)
        wb = self.workbench
        if wb.panel_collapsed and wb.layout.area_of(self.model) == "panel":
            wb.set_panel_collapsed(False)
        wb._view_changed(self.model.current_view)

    def _menu(self):
        v = self.current_view()
        if v is None:
            return
        menu = QtWidgets.QMenu(self)
        self.workbench.fill_view_menu(menu, v)
        b = self.buttons["more"]
        menu.exec(b.mapToGlobal(QtCore.QPoint(0, b.height())))


class DropMarks(QtWidgets.QWidget):
    """The drop preview over the workbench while a tab is dragged (ignores the mouse)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.rect_, self.line = None, None

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        accent = QtGui.QColor(TOKENS["accent"])
        if self.rect_ is not None:
            fill = QtGui.QColor(accent)
            fill.setAlpha(46)
            p.fillRect(self.rect_, fill)
            p.setPen(QtGui.QPen(accent, 2))
            p.drawRect(self.rect_.adjusted(1, 1, -1, -1))
        if self.line is not None:
            p.fillRect(self.line, accent)


class Workbench(QtWidgets.QWidget):
    viewShown = QtCore.Signal(str)     # a view became the shown tab of its group
    viewClosed = QtCore.Signal(str)    # a closable view was closed and unregistered
    layoutChanged = QtCore.Signal()

    def __init__(self, timeline: QtWidgets.QWidget | None = None, parent=None):
        super().__init__(parent)
        self.specs: dict[str, ViewSpec] = {}
        self.order: list[str] = []
        self.factory = None            # callable(view_id) -> bool: create a view a saved layout names
        self.layout = Layout()
        self.panel_collapsed = False
        self.maximized: int | None = None
        self._max_hid_panel = False    # a maximized editor group hid the panel (shown again on restore)
        self.active: Group | None = None
        self._groups: dict[int, TabGroup] = {}
        self._splitters: list[tuple[QtWidgets.QSplitter, Split]] = []
        self._slots: dict[str, QtWidgets.QWidget] = {}
        self._page_slots: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QWidget]] = {}  # view -> (page, slot)
        self._watched: list[QtWidgets.QWidget] = []
        self._drag = None
        self._panel_size = 280
        self._sized = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.splitterMoved.connect(self._schedule_sync)
        outer.addWidget(self.vsplit)
        self.pages = QtWidgets.QStackedWidget()   # page 0: editor area + timeline; others: workspaces
        self.editor_page = QtWidgets.QWidget()
        ep = QtWidgets.QVBoxLayout(self.editor_page)
        ep.setContentsMargins(0, 0, 0, 0)
        ep.setSpacing(0)
        self.editor_host = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self.editor_host).setContentsMargins(0, 0, 0, 0)
        ep.addWidget(self.editor_host, 1)
        if timeline is not None:
            ep.addWidget(timeline)
        self.pages.addWidget(self.editor_page)
        self.pages.currentChanged.connect(lambda *_: self._after_visibility())
        self.panel_host = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self.panel_host).setContentsMargins(0, 0, 0, 0)
        self.vsplit.addWidget(self.pages)
        self.vsplit.addWidget(self.panel_host)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 0)
        self.marks = DropMarks(self)
        self.marks.hide()
        self._sync_timer = QtCore.QTimer(self, singleShot=True, interval=0)
        self._sync_timer.timeout.connect(self._sync_native)
        QtWidgets.QApplication.instance().focusChanged.connect(self._focus_changed)
        self._actions = None
        self.extra_view_actions = lambda view: []   # more (label, callable) entries for a view's menu

    # ------------------------------------------------------------------ views
    def register(self, view: str, widget, title: str, home: str = "panel", closable: bool = False,
                 native: bool = False, group: Group | None = None, index: int | None = None, keep: bool = False):
        """Add a view; it goes into ``group`` (or the first group of ``home``) unless the layout
        already places it."""
        self.specs[view] = ViewSpec(view, widget, title, home, closable, native, keep)
        if view not in self.order:
            self.order.append(view)
        if native:
            widget.setParent(self)
            widget.hide()
        if self.layout.find(view) is None:
            g = group or self.layout.groups(home)[0]
            shown = g.current_view
            self.layout.add(view, g, index)
            if shown is not None and index is None:  # appending keeps the shown tab
                g.current = g.views.index(shown)
        self._rebuild()

    def unregister(self, view: str, delete: bool = True):
        """Remove a view; its widget is deleted, or with ``delete=False`` only taken out (hidden,
        no parent) for its owner to keep."""
        spec = self.specs.pop(view, None)
        if view in self.order:
            self.order.remove(view)
        if not self._alive():  # the window is being torn down (a cell's plot left as its notebook went)
            self.layout.remove(view)
            return
        self._capture()
        self.layout.remove(view)
        if spec is not None and _valid(spec.widget):
            spec.widget.setParent(None)
            if delete:
                spec.widget.deleteLater()
        self._rebuild()

    def widget(self, view: str):
        return self.specs[view].widget

    def view_of(self, widget) -> str | None:
        if isinstance(widget, str):
            return widget if widget in self.specs else None
        return next((v for v, s in self.specs.items() if s.widget is widget), None)

    def slot(self, view: str) -> QtWidgets.QWidget:
        if view not in self._slots:
            s = QtWidgets.QWidget()
            s.setObjectName("nativeSlot")
            self._slots[view] = s
        return self._slots[view]

    def lend(self, view: str, page: QtWidgets.QWidget, slot: QtWidgets.QWidget):
        """Show the GL ``view`` over ``slot`` while the workspace ``page`` is the current page
        (the Render page's live 3D view); it goes back to its own tab on the editor page."""
        self._page_slots[view] = (page, slot)
        self._watch()
        self._schedule_sync()

    def _current_slot(self, view: str):
        page, slot = self._page_slots.get(view, (None, None))
        if page is not None and self.pages.currentWidget() is page:
            return slot
        return self._slots.get(view)

    def titles(self, area: str) -> list[str]:
        return [self.specs[v].title for g in self.layout.groups(area) for v in g.views]

    def can_close(self, view) -> bool:
        view = self.view_of(view)
        spec = self.specs[view]
        g = self.layout.find(view)
        return spec.closable or (g is not None and self.layout.area_of(g) != spec.home)

    def default_layout(self) -> Layout:
        lay = Layout()
        for v in self.order:
            lay.groups(self.specs[v].home)[0].views.append(v)
        return lay

    # ------------------------------------------------------------------ queries
    def group_of(self, view) -> Group | None:
        v = self.view_of(view)
        return self.layout.find(v) if v else None

    def is_shown(self, view) -> bool:
        """``view`` is the shown tab of its group and its area is on screen (not folded or hidden)."""
        v = self.view_of(view)
        g = self.layout.find(v) if v else None
        if g is None or g.current_view != v:
            return False
        if self.maximized is not None and self.maximized != g.id:
            return False
        if self.layout.area_of(g) == "panel":
            return not self.panel_collapsed and not self.panel_host.isHidden()
        return self.pages.currentIndex() == 0

    def tab_group(self, group: Group) -> TabGroup:
        return self._groups[group.id]

    # ------------------------------------------------------------------ edits
    def show(self, view):
        """Bring ``view`` to the front of its group, unfolding and showing the panel if needed."""
        v = self.view_of(view)
        g = self.layout.show(v) if v else None
        if g is None:
            return
        if self.maximized is not None and self.maximized != g.id:
            self.maximized = None
        if self.layout.area_of(g) == "panel":
            self.panel_host.show()
            self.panel_collapsed = False
        else:
            self.pages.setCurrentIndex(0)
        self._activate(g)
        self._rebuild()

    def move(self, view, target: Group | str, zone: str = "center", index: int | None = None) -> Group:
        """Move ``view`` into ``target`` (a group, or an area name for its first group)."""
        v = self.view_of(view)
        if isinstance(target, str):
            target = self.layout.groups(target)[0]
        self._capture()
        g = self.layout.move(v, target, zone, index)
        if self.maximized is not None and self.layout.group(self.maximized) is None:
            self.maximized = None
        if self.layout.area_of(g) == "panel":
            self.panel_collapsed = False
            self.panel_host.show()
        self.active = g
        self._rebuild()
        self._view_changed(v)
        return g

    def split(self, view, side: str = "right") -> Group:
        v = self.view_of(view)
        return self.move(v, self.layout.find(v), side)

    def _towards(self, g: Group, direction: str) -> Group | None:
        """The group a move from ``g`` towards ``direction`` lands in (None: a new split)."""
        nb = self.layout.neighbour(g, direction)
        if nb is None and direction in ("top", "bottom"):  # across the panel boundary
            area = self.layout.area_of(g)
            if (area, direction) in (("editor", "bottom"), ("panel", "top")):
                nb = self.layout.groups("panel" if area == "editor" else "editor")[0]
        return nb

    def move_towards(self, view, direction: str) -> Group:
        """Move ``view`` into the next group towards ``direction``, or split that way when there is none."""
        v = self.view_of(view)
        nb = self._towards(self.layout.find(v), direction)
        if nb is not None:
            return self.move(v, nb)
        return self.split(v, direction)

    def can(self, op: str, view=None) -> bool:
        """Whether a View › Layout move (``split:<side>`` or ``towards:<side>``) changes anything
        for ``view``: a view alone in its group cannot split off itself."""
        v = self.view_of(view) if view is not None else self._current_view()
        g = self.layout.find(v) if v else None
        if g is None:
            return False
        kind, side = op.split(":")
        if kind == "towards" and self._towards(g, side) is not None:
            return True
        return len(g.views) > 1

    def return_home(self, view):
        v = self.view_of(view)
        return self.move(v, self.specs[v].home)

    def close_view(self, view):
        """Close a closable view, or send a moved one back to its home area."""
        v = self.view_of(view)
        if v is None:
            return
        if self.specs[v].closable:
            self.unregister(v, delete=not self.specs[v].keep)
            self.viewClosed.emit(v)
        elif self.can_close(v):
            self.return_home(v)

    def toggle_maximize(self, group: Group | None = None):
        group = group or self.active or self.layout.find(self.order[0])
        self._capture()
        self.maximized = None if self.maximized == group.id else group.id
        self._rebuild()

    def set_panel_collapsed(self, collapsed: bool):
        """Fold the panel to its tab bars (a click on a tab unfolds it) or unfold it."""
        if collapsed == self.panel_collapsed:
            return
        if collapsed and self.vsplit.sizes()[1] > 0:
            self._panel_size = self.vsplit.sizes()[1]
        self.panel_collapsed = collapsed
        self._rebuild()

    def set_panel_visible(self, visible: bool):
        self.panel_host.setVisible(visible)
        self._after_visibility()

    def reset(self):
        self.layout = self.default_layout()
        self.maximized = None
        self._max_hid_panel = False
        self.panel_collapsed = False
        self.panel_host.show()
        self._rebuild()
        self.vsplit.setSizes([max(self.height() - self._panel_size, 100), 280])

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        self._capture()
        sizes = self.vsplit.sizes()
        return {"layout": self.layout.to_json(), "panel_size": self._panel_size if self.panel_collapsed else sizes[1],
                "editor_size": sizes[0], "panel_collapsed": self.panel_collapsed}

    def restore_state(self, state: dict) -> bool:
        """Apply :meth:`state`; returns False (and keeps the current layout) when it can't be read."""
        try:
            homes = {v: s.home for v, s in self.specs.items() if not s.closable}
            lay = Layout.from_json(state["layout"], known=self._known, homes=homes)
        except (KeyError, TypeError, ValueError):
            return False
        self.layout = lay
        self.maximized = None
        self.panel_collapsed = bool(state.get("panel_collapsed"))
        self._rebuild()
        e, p = state.get("editor_size"), state.get("panel_size")
        if isinstance(e, int) and isinstance(p, int) and e > 0 and p > 0:
            self._panel_size = p
            if not self.panel_collapsed:
                self.vsplit.setSizes([e, p])
                self._sized = True
        return True

    def _known(self, view) -> bool:
        if view in self.specs:
            return True
        return bool(self.factory and self.factory(view)) and view in self.specs

    # ------------------------------------------------------------------ drawing
    def _alive(self) -> bool:
        """Whether the workbench's Qt objects still exist: when the window is deleted, Qt deletes
        them before the notebook cells, whose docked plots then leave (``unregister``)."""
        return all(_valid(o) for o in (self, self.vsplit, self.editor_host, self.panel_host,
                                       *(sp for sp, _ in self._splitters)))

    def _capture(self):
        """Copy the splitter positions into the model before it changes."""
        for sp, node in self._splitters:
            if not _valid(sp):
                continue
            sizes = sp.sizes()
            if sum(sizes) > 0 and len(sizes) == len(node.children):
                node.sizes = [float(s) for s in sizes]

    def _rebuild(self):
        groups = self.layout.groups()
        live = {g.id for g in groups}
        for gid in [k for k in self._groups if k not in live]:
            tg = self._groups.pop(gid)
            for i in range(tg.stack.count() - 1, 0, -1):   # keep the view widgets alive
                w = tg.stack.widget(i)
                tg.stack.removeWidget(w)
                w.setParent(None)
            tg.setParent(None)
            tg.deleteLater()
        for g in groups:
            if g.id not in self._groups:
                self._groups[g.id] = TabGroup(self, g)
            self._groups[g.id].model = g
        old = self._splitters
        self._splitters = []
        maxi = self.layout.group(self.maximized) if self.maximized is not None else None
        for area, host in (("editor", self.editor_host), ("panel", self.panel_host)):
            if maxi is not None and self.layout.area_of(maxi) == "editor" and area == "editor":
                root = self._groups[maxi.id]
            else:
                root = self._build(self.layout.areas[area])
            lay = host.layout()
            while lay.count():
                w = lay.takeAt(0).widget()
                if w is not None and w is not root and w.parentWidget() is host:
                    w.setParent(None)
            lay.addWidget(root)
            root.show()
        stale = [sp for sp, _ in old if not any(sp is s for s, _ in self._splitters)]
        for g in groups:  # groups not drawn (a maximized group hides the others) leave old splitters
            tg = self._groups[g.id]
            if any(tg.parentWidget() is sp for sp in stale):
                tg.setParent(None)
        for sp in stale:
            sp.setParent(None)
            sp.deleteLater()
        for g in groups:
            tg = self._groups[g.id]
            tg.sync()
            tg.stack.setVisible(not (self.panel_collapsed and self.layout.area_of(g) == "panel"))
        hidden_editor = maxi is not None and self.layout.area_of(maxi) == "panel"
        self.pages.setVisible(not hidden_editor)
        if maxi is not None and not hidden_editor:
            for g in self.layout.groups("editor"):
                if g is not maxi:
                    self._groups[g.id].hide()
        for g in self.layout.groups("panel"):
            self._groups[g.id].setVisible(not hidden_editor or g is maxi)
        editor_max = maxi is not None and not hidden_editor  # the panel goes too, as in "maximize"
        if editor_max and not self._max_hid_panel:
            self._max_hid_panel = not self.panel_host.isHidden()
            self.panel_host.hide()
        elif not editor_max and self._max_hid_panel:
            self._max_hid_panel = False
            self.panel_host.show()
        if self.panel_collapsed:
            h = max(self._groups[g.id].bar.sizeHint().height() for g in self.layout.groups("panel")) + 2
            self.panel_host.setMaximumHeight(max(h, 24))
        else:
            self.panel_host.setMaximumHeight(16777215)
            sizes = self.vsplit.sizes()
            if sum(sizes) and sizes[1] < 60:
                self.vsplit.setSizes([max(sum(sizes) - self._panel_size, 100), self._panel_size])
        self._watch()
        self._after_visibility()
        self.layoutChanged.emit()

    def _build(self, node):
        if isinstance(node, Group):
            return self._groups[node.id]  # shown by the caller once it has a parent
        sp = QtWidgets.QSplitter(QtCore.Qt.Horizontal if node.orient == "h" else QtCore.Qt.Vertical)
        sp.setChildrenCollapsible(False)
        sp.setHandleWidth(7)  # theme.py draws a 1 px rule and a grip in it
        for c in node.children:
            w = self._build(c)
            w.setMinimumSize(160, 90)
            sp.addWidget(w)
            w.show()  # not before: a parentless widget made visible flashes up as its own window
        sp.setSizes([max(int(s * 1000 / (sum(node.sizes) or 1)), 1) for s in node.sizes])
        sp.splitterMoved.connect(self._schedule_sync)
        self._splitters.append((sp, node))
        return sp

    def _watch(self):
        """Follow every move or resize between the GL slots and the workbench."""
        for w in self._watched:
            try:
                w.removeEventFilter(self)
            except RuntimeError:
                pass
        self._watched = []
        for slot in [*self._slots.values(), *(s for _, s in self._page_slots.values())]:
            w = slot
            while w is not None and w is not self:
                w.installEventFilter(self)
                self._watched.append(w)
                w = w.parentWidget()

    def eventFilter(self, obj, e):
        if e.type() in (QtCore.QEvent.Move, QtCore.QEvent.Resize, QtCore.QEvent.Show, QtCore.QEvent.Hide):
            self._schedule_sync()
        return False

    def showEvent(self, e):
        super().showEvent(e)
        if not self._sized:  # the panel's first height (its widgets would ask for much more)
            self._sized = True
            total = sum(self.vsplit.sizes()) or self.height()
            panel = min(self._panel_size, int(total * 0.3))
            self.vsplit.setSizes([max(total - panel, 100), panel])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.marks.setGeometry(self.rect())
        self._schedule_sync()

    def _schedule_sync(self, *_):
        self._sync_timer.start()

    def _after_visibility(self):
        self._sync_native()
        self._schedule_sync()

    def _sync_native(self):
        """Put each GL widget over its slot, or hide (park) it when its slot is not on screen."""
        if self._drag is not None:
            return
        for v in self._slots.keys() | self._page_slots.keys():
            spec = self.specs.get(v)
            if spec is None:
                continue
            w = spec.widget
            slot = self._current_slot(v)
            lent = slot is not self._slots.get(v)
            on = (slot is not None and (lent or self.layout.find(v) is not None) and self.isAncestorOf(slot)
                  and slot.isVisibleTo(self) and slot.width() > 0)
            if on:
                w.setGeometry(QtCore.QRect(slot.mapTo(self, QtCore.QPoint(0, 0)), slot.size()))
                w.show()
                w.raise_()
            else:
                w.hide()
            if hasattr(w, "set_parked"):
                w.set_parked(not on)
        self.marks.raise_()

    def _view_changed(self, view):
        self._after_visibility()
        if view is not None:
            self.viewShown.emit(view)

    def _activate(self, group: TabGroup | Group):
        self.active = group.model if isinstance(group, TabGroup) else group

    def _focus_changed(self, _old, new):
        w = new
        while w is not None and w is not self:
            if isinstance(w, TabGroup):
                self.active = w.model
                return
            v = self.view_of(w)
            if v is not None and self.specs[v].native:
                self.active = self.layout.find(v)
                return
            w = w.parentWidget()

    # ------------------------------------------------------------------ dragging
    def begin_drag(self, view: str):
        self._drag = {"view": view, "target": None, "snaps": []}
        for v, spec in self.specs.items():  # a GL window would hide the drop marks: show a picture
            if spec.native and spec.widget.isVisible():
                snap = getattr(spec.widget, "snapshot", None)
                pix = snap() if snap else spec.widget.grab()
                label = QtWidgets.QLabel(self)
                label.setPixmap(pix)
                label.setScaledContents(True)
                label.setGeometry(spec.widget.geometry())
                label.show()
                spec.widget.hide()
                self._drag["snaps"].append(label)
        self.marks.setGeometry(self.rect())
        self.marks.show()
        self.marks.raise_()
        QtWidgets.QApplication.instance().installEventFilter(self._esc)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DragMoveCursor)

    @property
    def _esc(self):
        if not hasattr(self, "_esc_filter"):
            self._esc_filter = _EscFilter(self)
        return self._esc_filter

    def target_at(self, global_pos: QtCore.QPoint):
        """(group, zone, index) under ``global_pos``: a tab bar gives ("center", insert index),
        a group body its drop zone. None outside every group."""
        for gid, tg in self._groups.items():
            if not tg.isVisible():
                continue
            g = self.layout.group(gid)
            p = tg.bar.mapFromGlobal(global_pos)
            head = QtCore.QRect(0, 0, tg.width(), tg.bar.height())
            if head.contains(p):
                index = tg.bar.count()
                for i in range(tg.bar.count()):
                    if p.x() < tg.bar.tabRect(i).center().x():
                        index = i
                        break
                return g, "center", index
            body = tg.stack.mapFromGlobal(global_pos)
            if tg.stack.isVisible() and tg.stack.rect().contains(body):
                return g, zone_at(tg.stack.width(), tg.stack.height(), body.x(), body.y()), None
        return None

    def drag_move(self, global_pos: QtCore.QPoint):
        if self._drag is None:
            return
        t = self.target_at(global_pos)
        self._drag["target"] = t
        self.marks.rect_ = self.marks.line = None
        if t is not None:
            g, zone, index = t
            tg = self._groups[g.id]
            if index is not None:
                x = tg.bar.tabRect(index).left() if index < tg.bar.count() else (
                    tg.bar.tabRect(index - 1).right() if tg.bar.count() else 0)
                top = tg.bar.mapTo(self, QtCore.QPoint(x, 0))
                self.marks.line = QtCore.QRect(top.x() - 1, top.y(), 3, tg.bar.height())
            else:
                r = QtCore.QRect(tg.stack.mapTo(self, QtCore.QPoint(0, 0)), tg.stack.size())
                w, h = r.width(), r.height()
                r = {"center": r, "left": r.adjusted(0, 0, -w // 2, 0), "right": r.adjusted(w // 2, 0, 0, 0),
                     "top": r.adjusted(0, 0, 0, -h // 2), "bottom": r.adjusted(0, h // 2, 0, 0)}[zone]
                self.marks.rect_ = r
        self.marks.update()

    def drag_drop(self, global_pos: QtCore.QPoint | None = None):
        if self._drag is None:
            return None
        if global_pos is not None:
            self.drag_move(global_pos)
        view, target = self._drag["view"], self._drag["target"]
        self._end_drag()
        if target is None:
            return None
        g, zone, index = target
        return self.move(view, g, zone, index)

    def cancel_drag(self):
        self._end_drag()

    def _end_drag(self):
        if self._drag is None:
            return
        for label in self._drag["snaps"]:
            label.hide()
            label.deleteLater()
        self._drag = None
        self.marks.hide()
        QtWidgets.QApplication.instance().removeEventFilter(self._esc)
        QtWidgets.QApplication.restoreOverrideCursor()
        self._sync_native()

    # ------------------------------------------------------------------ menus
    def fill_view_menu(self, menu: QtWidgets.QMenu, view: str):
        """Move, split, maximize and close entries for ``view`` (tab right click and ⋯)."""
        g = self.layout.find(view)
        area = self.layout.area_of(g)
        menu.addAction(native_text("Split right\tCtrl+Alt+\\"), lambda: self.split(view, "right"))
        menu.addAction("Split down", lambda: self.split(view, "bottom"))
        if area == "panel":
            menu.addAction("Move to editor area", lambda: self.move(view, "editor"))
        else:
            menu.addAction("Move to panel", lambda: self.move(view, "panel"))
        menu.addSeparator()
        menu.addAction(native_text(("Restore group" if self.maximized == g.id else "Maximize group") + "\tCtrl+Alt+M"),
                       lambda: self.toggle_maximize(g))
        for label, fn in self.extra_view_actions(view):
            menu.addAction(label, fn)
        if self.can_close(view):
            menu.addSeparator()
            menu.addAction("Close" if self.specs[view].closable else (
                "Return to panel" if self.specs[view].home == "panel" else "Return to editor area"),
                           lambda: self.close_view(view))

    def _current_view(self):
        """The view the View › Layout moves act on: the front view of the focused (or first) group."""
        g = self.active if self.active in self.layout.groups() else self.layout.find(self.order[0])
        return g.current_view if g else None

    def update_actions(self):
        """Grey out the View › Layout moves that would change nothing (a view alone in its group
        cannot split off itself), naming the view they act on; call when the menu opens."""
        v = self._current_view()
        title = self.specs[v].title if v in self.specs else "the view"
        for a in self.actions():
            op = a.data()
            if op:
                a.setEnabled(self.can(op))
                a.setToolTip(f"Acts on {title}" if a.isEnabled() else
                             f"{title} is alone in its group: click another tab first")

    def enable_actions(self):
        """Every move enabled again (the shortcuts check :meth:`can` themselves)."""
        for a in self.actions():
            a.setEnabled(True)

    def actions(self) -> list[QtGui.QAction]:
        """Window actions for the View › Layout menu and the command palette."""
        if self._actions is None:
            current = self._current_view

            def act(text, fn, keys=None, op=None):
                a = QtGui.QAction(text, self)
                a.triggered.connect(lambda: current() and (op is None or self.can(op)) and fn(current()))
                a.setData(op)
                if keys:
                    a.setShortcut(QtGui.QKeySequence(keys))
                    a.setShortcutContext(QtCore.Qt.WindowShortcut)
                return a

            self._actions = [
                act("Split right", lambda v: self.split(v, "right"), "Ctrl+Alt+\\", "split:right"),
                act("Split down", lambda v: self.split(v, "bottom"), op="split:bottom"),
                act("Move tab left", lambda v: self.move_towards(v, "left"), "Ctrl+Alt+Left", "towards:left"),
                act("Move tab right", lambda v: self.move_towards(v, "right"), "Ctrl+Alt+Right", "towards:right"),
                act("Move tab up", lambda v: self.move_towards(v, "top"), "Ctrl+Alt+Up", "towards:top"),
                act("Move tab down", lambda v: self.move_towards(v, "bottom"), "Ctrl+Alt+Down", "towards:bottom"),
                act("Move tab to editor area", lambda v: self.move(v, "editor")),
                act("Move tab to panel", lambda v: self.move(v, "panel")),
                act("Return tab home", self.close_view, "Ctrl+Alt+W"),
                act("Maximize group", lambda v: self.toggle_maximize(self.layout.find(v)), "Ctrl+Alt+M"),
            ]
            fold = QtGui.QAction("Fold or unfold panel", self)
            fold.setShortcut(QtGui.QKeySequence("Ctrl+Alt+J"))
            fold.triggered.connect(lambda: self.set_panel_collapsed(not self.panel_collapsed))
            reset = QtGui.QAction("Reset layout", self)
            reset.triggered.connect(self.reset)
            self._actions += [fold, reset]
        return self._actions


class _EscFilter(QtCore.QObject):
    """Esc cancels a tab drag; mouse moves anywhere keep the drop marks current."""

    def __init__(self, wb: Workbench):
        super().__init__(wb)
        self.wb = wb

    def eventFilter(self, obj, e):
        if e.type() == QtCore.QEvent.KeyPress and e.key() == QtCore.Qt.Key_Escape:
            self.wb.cancel_drag()
            return True
        return False
