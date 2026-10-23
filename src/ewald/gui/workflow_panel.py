"""The side bar of one workflow source (:func:`ewald.workflow.workflow_panel`).

Top: the account the source is set up with, and a button to its settings page. Then a box that
takes an id (that workflow alone) or words (a search). Below, the source's workflows, newest
first; expanding one asks the source to describe it and lists its steps (nested, restarts joined)
with their outputs. A double click on a workflow opens it as one notebook with a variable per
step; on a step or an output it opens that path, as File › Open from › Location would. The buttons
insert a ``load(...)`` cell instead or copy the path. The right-click menu on one or more selected
workflows adds and runs a cell ``workflow_table(...)``: their steps' numbers in one table, side by
side (:mod:`ewald.workflow_table`). Requests run off the GUI thread, so a slow
server never freezes the window.

When the notebook on screen is one such workflow notebook, every step it loads gets a mark in the
right column: ● is the step ``pipe`` is now (the 3D view and the main Plots tab), ○ another one;
a click on ○ (or a double click on the step) makes it ``pipe`` by rewriting the notebook's
``pipe = ...`` cell and running it. Under such a step its tables are listed ("Plot Bands",
"Plot Steps", ...); a click opens one as its own Plots tab through a view cell
``show(bands, table="Bands", panel=True)``. Both only edit cells, so the notebook stays the record.
"""
from __future__ import annotations

import re

from PySide6 import QtCore, QtWidgets

from .. import plugins
from ..pipeline import Pipeline
from ..workflow import Workflow, reference, step_variables
from . import workers

ITEM = QtCore.Qt.UserRole          # the Workflow, WorkflowStep or WorkflowOutput an item shows
VAR = QtCore.Qt.UserRole + 1       # the variable a workflow notebook loads that step or output into
TABLE = QtCore.Qt.UserRole + 2     # on a "Plot ..." row: the table it opens
ACTIVE, OTHER = "\u25cf", "\u25cb"  # marks of the step that is pipe, and of the other steps loaded
_ID = re.compile(r"^(\d+|[0-9a-fA-F-]{8,36})$")
#: pages a search follows while they come back empty (a source reads a bounded number per page)
EMPTY_PAGES = 10


class _Job(QtCore.QThread):
    """``fn()`` on another thread; ``done(result, error)`` back on the GUI thread."""

    done = QtCore.Signal(object, str)

    def __init__(self, fn, parent=None):
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
        self.browse = QtWidgets.QToolButton(text="Open folder…", toolTip="Open a folder of calculations as a workflow")
        self.browse.clicked.connect(self.open_folder)
        self.browse.setVisible(bool(getattr(getattr(self.source, "source", None), "browse_folders", False)))
        row.addWidget(self.browse)
        refresh = QtWidgets.QToolButton(text="Refresh", toolTip="Load the list again")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        lay.addLayout(row)
        self.tree = QtWidgets.QTreeWidget(headerHidden=True, objectName="workflowTree", columnCount=2)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        head_view = self.tree.header()
        head_view.setStretchLastSection(False)
        head_view.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        head_view.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        head_view.resizeSection(1, 24)
        self.tree.itemExpanded.connect(self._expand)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
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
        context.on("pipeline", lambda _p: self.refresh_marks())
        tabs = getattr(getattr(context.window, "panel", None), "tabs", None)
        if tabs is not None:
            tabs.currentChanged.connect(lambda *_: self.refresh_marks())
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
                self._fill_workflow(item, wf)
            else:
                item.addChild(QtWidgets.QTreeWidgetItem(["…"]))     # filled when expanded
            self.tree.addTopLevelItem(item)
        self.next_page = next_page
        self.more.setVisible(bool(next_page))
        self._say("" if self.tree.topLevelItemCount() else "No workflows found.")
        self.refresh_marks()

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
        self._fill_workflow(item, wf)
        item.setExpanded(True)
        self.refresh_marks()

    def _fill_workflow(self, item, wf):
        names = step_variables(wf)
        steps = {s.path for _, s in wf.walk() if s.path}
        # the workflow's own result gets a mark too, unless it is one of its steps
        item.setData(0, VAR, names.get(wf.path) if wf.path and wf.path not in steps else None)
        self._fill(item, wf, names)

    def _fill(self, item, step, names: dict):
        """``step``'s steps (nested) and outputs under ``item``; ``names`` are the variables a
        notebook of the whole workflow loads them into (:func:`ewald.workflow.step_variables`)."""
        item.takeChildren()
        for s in step.steps:
            label = s.name + (f"  ({s.runs} runs)" if s.runs > 1 else "")
            child = QtWidgets.QTreeWidgetItem([f"{label}  {s.mark()}".rstrip()])
            child.setToolTip(0, s.work_dir or s.path or "")
            child.setData(0, ITEM, s)
            child.setData(0, VAR, names.get(s.path) if s.path else None)
            self._fill(child, s, names)
            item.addChild(child)
        for o in step.outputs:
            child = QtWidgets.QTreeWidgetItem([f"{o.name}  ({o.note})" if o.note else o.name])
            child.setToolTip(0, o.path)
            child.setData(0, ITEM, o)
            child.setData(0, VAR, names.get(o.path) if o.path else None)
            item.addChild(child)

    # ------------------------------------------------------------ the notebook on screen
    def _view(self):
        """The notebook view on screen, or None."""
        return getattr(getattr(self.context.window, "panel", None), "view", None)

    def _items(self) -> list:
        out, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            out.append(it)
            stack.extend(it.child(i) for i in range(it.childCount()))
        return out

    def refresh_marks(self):
        """Mark the steps the notebook on screen loads (● the one that is ``pipe``, ○ the others)
        and list their tables under them; nothing when that notebook is not this workflow's."""
        view = self._view()
        nb = view.notebook if view is not None else None
        lines = set()
        if nb is not None:
            for cell in nb.cells:
                if cell.kind == "code":
                    lines.update(line.split("#", 1)[0].strip() for line in cell.source.splitlines())
        pipe = nb.pipeline if nb is not None else None
        for item in self._items():
            var, obj = item.data(0, VAR), item.data(0, ITEM)
            if not var:
                continue
            loaded = f"{var} = load({str(obj.path)!r})" in lines
            value = nb.namespace.get(var) if loaded else None
            active = loaded and pipe is not None and value is pipe
            item.setText(1, (ACTIVE if active else OTHER) if loaded else "")
            item.setToolTip(1, f"{nb.pipeline_name} = {var}: shown in the 3D view and the Plots tab" if active
                            else f"Click to make {var} the shown step ({nb.pipeline_name} = {var})" if loaded else "")
            self._list_tables(item, value if isinstance(value, Pipeline) else None, view, active)

    def _list_tables(self, item, pipeline, view, active: bool):
        """"Plot <table>" rows under ``item`` for the tables of ``pipeline`` (computed off the GUI
        thread, at its first frame; the frame on screen when it is ``pipe``)."""
        key = id(pipeline) if pipeline is not None else None
        if getattr(item, "_tables_of", None) == key:
            return
        item._tables_of = key
        if pipeline is None:
            self._show_tables(item, key, [])
            return
        data = self.context.data if active else None
        if data is not None:
            self._show_tables(item, key, list(data.tables))
        elif not getattr(view, "running", False):     # cells still running: the next refresh lists them
            self._run(lambda: list(pipeline.compute(0).tables),
                      lambda names, _err: self._show_tables(item, key, names or []))
        else:
            item._tables_of = None

    def _show_tables(self, item, key, names):
        if getattr(item, "_tables_of", None) != key or item.treeWidget() is not self.tree:
            return                                    # the notebook changed meanwhile
        for i in reversed(range(item.childCount())):
            if item.child(i).data(0, TABLE):
                item.takeChild(i)
        var = item.data(0, VAR)
        for k, name in enumerate(names):
            row = QtWidgets.QTreeWidgetItem([f"Plot {name}"])
            row.setData(0, TABLE, name)
            row.setToolTip(0, f"Open show({var}, table={name!r}) as a Plots tab")
            item.insertChild(k, row)

    def _clicked(self, item, column):
        if item.data(0, TABLE):
            self.plot_table(item)
        elif column == 1 and item.text(1) == OTHER:
            self.activate(item)

    def _double_clicked(self, item, _column):
        if item.data(0, TABLE):
            return                                    # the click already opened it
        if item.text(1):
            self.activate(item)
        else:
            self.open(item)

    def activate(self, item):
        """Make the step of ``item`` the notebook's ``pipe``: its ``pipe = ...`` line is rewritten
        (or a cell added) and that cell runs, so the 3D view and the Plots tab follow."""
        from ..notebook import Cell

        view, var = self._view(), item.data(0, VAR)
        if view is None or not var or not var.isidentifier():
            return None
        nb = view.notebook
        name = nb.pipeline_name
        pattern = re.compile(rf"^{re.escape(name)}\s*=\s*[A-Za-z_]\w*\s*(#.*)?$")
        target = next((c for c in reversed(nb.cells) if c.kind == "code"
                       and any(pattern.match(line.strip()) for line in c.source.splitlines())), None)
        new = f"{name} = {var}"
        if target is None:
            target = nb.insert(len(nb.cells), Cell(new))
        else:
            target.source = "\n".join(new if pattern.match(line.strip()) else line
                                       for line in target.source.splitlines())
        nb.dirty = True
        view.rebuild()
        view.history.commit(f"Show {var}")
        view.run(target)
        return target

    def plot_table(self, row):
        """Open the table of a "Plot ..." row as its own Plots tab: a view cell
        ``show(<step>, table=..., panel=True)``, added once and run again when it is there."""
        view, var, table = self._view(), row.parent().data(0, VAR), row.data(0, TABLE)
        if view is None or not var:
            return None
        for cell in view.notebook.cells:
            args = cell.view or {}
            if args.get("source") == var and args.get("table") == table and not args.get("histogram"):
                view.select(cell)
                view.run(cell)
                return cell
        return view.add_view(source=var, table=table, panel=True)

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

    def selected_workflows(self) -> list:
        """The workflows selected in the list (Ctrl/Shift-click picks several), in list order."""
        top = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        return [self._object(it) for it in top if it.isSelected() and isinstance(self._object(it), Workflow)]

    def compare(self, workflows=None):
        """Add and run a cell ``workflow_table(<workflow>, ...)`` in the notebook on screen (a new
        one if none): the numbers of every step, of several workflows side by side. The paths are
        string literals, so a workflow's name or path cannot put code into the cell."""
        workflows = self.selected_workflows() if workflows is None else list(workflows)
        refs = [reference(w) for w in workflows if reference(w)]
        if not refs:
            return None
        view = self._view() or self.context.window.panel.new_notebook()
        cell = view.add_cell("code", "from ewald import workflow_table\n\n"
                                     f"workflow_table({', '.join(repr(str(r)) for r in refs)})")
        self.context.window.notebook_dock.show()
        view.run(cell)
        return cell

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
        if item.text(1) == OTHER:
            menu.addAction(f"Show this step ({item.data(0, VAR)})", lambda: self.activate(item))
        menu.addAction("Open as notebook" if isinstance(self._object(item), Workflow) else "Open",
                       lambda: self.open(item))
        menu.addAction("Insert load() cell", self.insert_cell)
        workflows = self.selected_workflows()
        if workflows:
            menu.addAction(f"Compare {len(workflows)} workflows in a table" if len(workflows) > 1
                           else "Table of the steps' numbers", lambda: self.compare(workflows))
        menu.addAction("Copy path", self.copy_path)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def open_folder(self, path: str | None = None):
        """Pick a folder of calculations and open it as one notebook (sources with ``browse_folders``)."""
        if path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open workflow folder")
        if not path:
            return None
        view = self.context.open_workflow(path)
        self.refresh()
        return view

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
