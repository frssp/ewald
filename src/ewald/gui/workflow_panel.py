"""The side bar of one workflow source (:func:`ewald.workflow.workflow_panel`).

Top: the account the source is set up with, and a button to its settings page. Then a box that
takes an id (that workflow alone) or words (a search). Below, the source's workflows, newest
first; expanding one asks the source to describe it and lists its steps (nested, restarts joined)
with their outputs. A double click on a workflow opens it as one notebook with a variable per
step; on a step or an output it opens that path, as File › Open location would. The buttons
insert a ``load(...)`` cell instead or copy the path. Requests run off the GUI thread, so a slow
server never freezes the window.
"""
from __future__ import annotations

import re

from PySide6 import QtCore, QtWidgets

from .. import plugins
from ..workflow import Workflow
from . import workers

ITEM = QtCore.Qt.UserRole          # the Workflow, WorkflowStep or WorkflowOutput an item shows
_ID = re.compile(r"^(\d+|[0-9a-fA-F-]{8,36})$")
#: pages a search follows while they come back empty (a source reads a bounded number per page)
EMPTY_PAGES = 10


class _Job(QtCore.QThread):
    """``fn()`` on another thread; ``done(result, error)`` back on the GUI thread."""

    done = QtCore.Signal(object, str)

    def __init__(self, fn, parent):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), "")
        except Exception as exc:  # a source's failure shows in the side bar, never as a crash
            self.done.emit(None, str(exc) if isinstance(exc, (LookupError, ValueError, OSError))
                           else f"{type(exc).__name__}: {exc}")


class WorkflowPanel(QtWidgets.QWidget):
    def __init__(self, context, source: str, parent=None):
        super().__init__(parent)
        self.context = context
        self.source_name = source
        self.jobs: list[_Job] = []
        self.next_page = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        head = QtWidgets.QHBoxLayout()
        self.who = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        head.addWidget(self.who, 1)
        self.login_button = QtWidgets.QToolButton(text="Log in…", toolTip="Set up where the workflows come from")
        self.login_button.clicked.connect(self.edit_settings)
        head.addWidget(self.login_button)
        lay.addLayout(head)
        row = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit(placeholderText="Id, or search the workflows")
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self.refresh)
        row.addWidget(self.search, 1)
        refresh = QtWidgets.QToolButton(text="Refresh", toolTip="Load the list again")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        lay.addLayout(row)
        self.tree = QtWidgets.QTreeWidget(headerHidden=True, objectName="workflowTree")
        self.tree.setUniformRowHeights(True)
        self.tree.itemExpanded.connect(self._expand)
        self.tree.itemDoubleClicked.connect(lambda item, _col: self.open(item))
        self.tree.currentItemChanged.connect(self._selected)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.tree, 1)
        self.more = QtWidgets.QPushButton("Older workflows", objectName="pluginButton")
        self.more.clicked.connect(lambda: self.refresh(more=True))
        self.more.setVisible(False)
        lay.addWidget(self.more)
        self.path = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        self.path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.path)
        buttons = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for text, slot, tip in (("Open", self.open, "Open it; a workflow opens as one notebook (double click does the same)"),
                                ("Insert cell", self.insert_cell, "A notebook cell pipe = load(...), not run"),
                                ("Copy path", self.copy_path, "The path, for load() or Open location")):
            b = QtWidgets.QPushButton(text, toolTip=tip)
            b.clicked.connect(lambda _=False, slot=slot: slot())
            b.setEnabled(False)
            buttons.addWidget(b)
            self.buttons[text] = b
        lay.addLayout(buttons)
        self.note = QtWidgets.QLabel(objectName="muted", wordWrap=True)
        self.note.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.note)
        context.on("settings", lambda _s: self.refresh())
        self.refresh()

    @property
    def source(self):
        return plugins.workflow_sources().get(self.source_name)

    # ------------------------------------------------------------ requests
    def _run(self, fn, then):
        job = workers.detach(_Job(fn))  # a slow request may outlive the panel
        job.done.connect(then)
        job.finished.connect(lambda: self.jobs.remove(job) if job in self.jobs else None)
        self.jobs.append(job)
        job.start()
        return job

    def wait(self, ms: int = 10000):
        """Block until every request has answered (tests, screenshots)."""
        while self.jobs:
            self.jobs[0].wait(ms)
            QtWidgets.QApplication.processEvents()

    # ------------------------------------------------------------ list
    def refresh(self, more: bool = False):
        source = self.source
        account = problem = ""
        if source is not None:
            try:
                account = source.account()
            except Exception as exc:  # noqa: BLE001 (a broken source says so in the side bar)
                problem = f"{type(exc).__name__}: {exc}"
        if source is None or not account:
            title = source.title if source is not None else self.source_name
            self.who.setText(f"{title} is not set up." if source is not None else f"{title} is not installed.")
            self.login_button.setText("Log in…")
            self.login_button.setVisible(bool(source is not None and source.settings_page))
            self.tree.clear()
            self.more.setVisible(False)
            if source is not None:
                self._say(problem or "Set it up (Log in…) to list your workflows.", error=bool(problem))
            return
        self.who.setText(account)
        self.login_button.setText("Settings…")
        self.login_button.setVisible(bool(source.settings_page))
        text = self.search.text().strip()
        if not more:
            self.tree.clear()
            self.next_page = None
        if text and _ID.match(text) and not more:
            self._say("Looking up…")
            self._run(lambda: source.describe(text), lambda wf, err: self._show([wf] if wf else [], err, None))
        else:
            cursor = self.next_page if more else None
            self._say("Loading…")
            self._run(lambda: self._page(source, cursor, text or None),
                      lambda page, err: self._show(page.workflows if page else [], err, page.next if page else None))

    @staticmethod
    def _page(source, cursor, query):
        page = source.list(query=query, cursor=cursor)
        for _ in range(EMPTY_PAGES - 1):
            if page.workflows or not page.next:
                break
            page = source.list(query=query, cursor=page.next)
        return page

    def _show(self, workflows, error, next_page):
        if error:
            self._say(error, error=True)
            return
        for wf in workflows:
            item = QtWidgets.QTreeWidgetItem([wf.title])
            item.setToolTip(0, "\n".join(x for x in (wf.label, wf.description, wf.ctime, wf.path) if x))
            item.setData(0, ITEM, wf)
            if wf.complete:
                self._fill(item, wf)
            else:
                item.addChild(QtWidgets.QTreeWidgetItem(["…"]))     # filled when expanded
            self.tree.addTopLevelItem(item)
        self.next_page = next_page
        self.more.setVisible(bool(next_page))
        self._say("" if self.tree.topLevelItemCount() else "No workflows found.")

    # ------------------------------------------------------------ one workflow
    def _expand(self, item):
        wf = item.data(0, ITEM)
        if not isinstance(wf, Workflow) or wf.complete or getattr(item, "_asked", False):
            return
        item._asked = True
        source = self.source
        if source is None:
            return
        ref = wf.path or wf.id
        self._run(lambda: source.describe(ref), lambda full, err: self._filled(item, full, err))

    def _filled(self, item, wf, error):
        item.takeChildren()
        if error:
            item.addChild(QtWidgets.QTreeWidgetItem([error]))
            return
        item.setData(0, ITEM, wf)
        item.setText(0, wf.title)
        self._fill(item, wf)
        item.setExpanded(True)

    def _fill(self, item, step):
        """``step``'s steps (nested) and outputs under ``item``."""
        item.takeChildren()
        for s in step.steps:
            label = s.name + (f"  ({s.runs} runs)" if s.runs > 1 else "")
            child = QtWidgets.QTreeWidgetItem([f"{label}  {s.mark()}".rstrip()])
            child.setToolTip(0, s.work_dir or s.path or "")
            child.setData(0, ITEM, s)
            self._fill(child, s)
            item.addChild(child)
        for o in step.outputs:
            child = QtWidgets.QTreeWidgetItem([f"{o.name}  ({o.note})" if o.note else o.name])
            child.setToolTip(0, o.path)
            child.setData(0, ITEM, o)
            item.addChild(child)

    # ------------------------------------------------------------ actions
    def _selected(self, item, _previous=None):
        path = self._path(item)
        self.path.setText(path or "")
        for b in self.buttons.values():
            b.setEnabled(bool(path))
        self.buttons["Open"].setToolTip("Open the whole workflow as one notebook, a variable per step"
                                        if isinstance(self._object(item), Workflow) else "Open it")

    @staticmethod
    def _object(item):
        return item.data(0, ITEM) if item is not None else None

    def _path(self, item=None):
        obj = self._object(item if item is not None else self.tree.currentItem())
        return getattr(obj, "path", "") or None

    def open(self, item=None):
        item = item if item is not None else self.tree.currentItem()
        obj = self._object(item)
        if isinstance(obj, Workflow):
            if not obj.complete:          # describe it first, then open
                source = self.source
                self._say("Reading the workflow…")
                self._run(lambda: source.describe(obj.path or obj.id),
                          lambda wf, err: self._say(err, error=True) if err else self._open_workflow(item, wf))
                return None
            return self._open_workflow(item, obj)
        path = self._path(item)
        return self.context.open(path) if path else None

    def _open_workflow(self, item, wf):
        if item is not None and item.treeWidget() is self.tree:
            self._filled(item, wf, "")
        self._say("")
        return self.context.open_workflow(wf)

    def insert_cell(self):
        path = self._path()
        if path:
            self.context.insert_cell(f"pipe = load({path!r})")

    def copy_path(self):
        path = self._path()
        if path:
            QtWidgets.QApplication.clipboard().setText(path)
            self.context.message(f"copied {path}")

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None or not self._path(item):
            return
        self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Open as notebook" if isinstance(self._object(item), Workflow) else "Open",
                       lambda: self.open(item))
        menu.addAction("Insert load() cell", self.insert_cell)
        menu.addAction("Copy path", self.copy_path)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def edit_settings(self):
        source = self.source
        if source is None or not source.settings_page:
            return
        window = self.context.window
        if hasattr(window, "edit_plugin_settings"):
            window.edit_plugin_settings(source.settings_page)
        else:
            self.context.open_preferences(source.title)
        self.refresh()

    def _say(self, text: str, error: bool = False):
        self.note.setText(text)
        self.note.setObjectName("cellError" if error else "muted")
        self.note.style().polish(self.note)
        self.note.setVisible(bool(text))
