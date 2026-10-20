"""Notebook view: Python cells (like Jupyter) that build the pipeline the viewport shows.

Each tab is one :class:`ewald.notebook.Notebook` (a percent-format ``.py`` file). Code
cells run in the notebook's namespace and show printed text, values, errors and
matplotlib figures underneath. Step cells (one ``pipe.modifiers.append(...)``) show the
modifier as a parameter form with what it did to the data; their Python source follows
the form, and editing the source re-creates the modifier.

One cell's widget is :class:`ewald.gui.cell_widget.CellWidget` and the code editor
:mod:`ewald.gui.code_editor`; this module runs cells, keeps undo history and holds the tabs.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..notebook import Cell, History, Notebook, new_notebook
from .pipeline_panel import FitWidthScrollArea as _FitWidthScrollArea, FlowLayout, _discover, fill_modifier_menu
from .notebook_tabs import NotebookTabs
from . import header
from .icons import icon
from .theme import TOKENS
# re-exported: the cell widgets and the editor moved to their own modules
from .cell_widget import (  # noqa: F401
    CELL_MIME, GUTTER_WIDTH, CellWidget, MiniPlot, OutputArea, _badge, _call_name, _same_shape, _shape, fill_view_menu,
    result_chips, result_plot)
from .code_editor import MAX_EDITOR_LINES, CodeEditor, PythonHighlighter, mono_font  # noqa: F401

# Seconds the window waits for cells to finish before it goes on and lets them run in the
# background (Interrupt stops them); None waits for every run (tests that read results right away).
CELL_BUDGET: float | None = 0.1


class _ElidedLabel(QtWidgets.QLabel):
    """A one-line label that shrinks with an ellipsis instead of pushing its neighbours."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        text = self.fontMetrics().elidedText(self.text(), QtCore.Qt.ElideMiddle, self.contentsRect().width())
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        p.drawText(self.contentsRect(), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, text)


class _CellWorker(QtCore.QThread):
    """Runs cells off the GUI thread: in-process cells execute here (Interrupt raises into this
    thread), kernel cells only wait here for the kernel's answers."""

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.fn()
        except Exception:  # noqa: BLE001 - kernel errors land in the cell's outputs
            import traceback

            traceback.print_exc()


def _cells_draw_off_screen():
    """pyplot in cells draws with Agg: the cells run on a worker thread, where a Qt figure window
    must not be created (plt.show() only marks figures for the cell's output anyway). The app's own
    plots use matplotlib's Qt canvas directly, not pyplot, so they are unaffected."""
    import matplotlib

    matplotlib.use("Agg", force=True)


class NotebookView(QtWidgets.QWidget):
    """The cells of one notebook, a toolbar and the add-cell bar."""

    changed = QtCore.Signal()                 # pipeline modified: recompute
    pipelineReplaced = QtCore.Signal(object)  # the notebook bound 'pipe' to a new pipeline
    askRequested = QtCore.Signal()
    titleChanged = QtCore.Signal()
    pipRequested = QtCore.Signal(list)        # pip requirements of plugins the notebook needs
    plotCreated = QtCore.Signal(object)       # the live PlotsPanel of a new view cell

    def __init__(self, notebook: Notebook, parent=None):
        super().__init__(parent)
        self.notebook = notebook
        self.history = History(notebook)  # Edit › Undo/Redo, one entry per action below
        self.widgets: list[CellWidget] = []
        self.current = None
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        self._worker = None
        self._marked = []
        run_all = self.run_all_button = QtWidgets.QPushButton("Run all", objectName="primary",
                                                              toolTip="Restart and run every cell (Ctrl+Shift+Enter)")
        run_all.setProperty("ewald_icon", "play")
        run_all.setProperty("ewald_icon_color", "on_accent")
        run_all.setIcon(icon("play", TOKENS["on_accent"]))
        run_all.setIconSize(QtCore.QSize(header.ICON, header.ICON))
        run_all.clicked.connect(self.run_all)
        restart = self.restart_button = QtWidgets.QToolButton(text="Restart",
                                                              toolTip="Clear the namespace (cells keep their code)")
        restart.clicked.connect(self.restart)
        self.interrupt_button = QtWidgets.QToolButton(text="Interrupt", objectName="interrupt",
                                                      toolTip="Stop the running cell")
        self.interrupt_button.setProperty("ewald_icon", "stop")
        self.interrupt_button.setIcon(icon("stop"))
        self.interrupt_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.interrupt_button.clicked.connect(self.interrupt)
        self.interrupt_button.setVisible(False)
        self.py_button = QtWidgets.QToolButton(text=".py", checkable=True, toolTip="Show the notebook as its Python file")
        self.py_button.toggled.connect(self._show_file)
        self.kernel_button = QtWidgets.QToolButton(objectName="kernelButton", autoRaise=True,
                                                   popupMode=QtWidgets.QToolButton.InstantPopup,
                                                   toolTip="Where the cells run: this app's Python or another one")
        self.kernel_button.setProperty("ewald_icon", "settings")
        self.kernel_button.setIcon(icon("settings"))
        self.kernel_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.kernel_menu = QtWidgets.QMenu(self.kernel_button)
        self.kernel_menu.aboutToShow.connect(lambda: self.fill_kernel_menu(self.kernel_menu))
        self.kernel_button.setMenu(self.kernel_menu)
        bar.addWidget(run_all)
        bar.addWidget(restart)
        bar.addWidget(self.interrupt_button)
        bar.addWidget(self.kernel_button)
        self.status = _ElidedLabel(objectName="muted")
        bar.addWidget(self.status, 1)
        bar.addWidget(self.py_button)
        lay.addLayout(bar)
        self.banner = self._make_banner()
        lay.addWidget(self.banner)
        self.packages_bar = self._make_packages_bar()
        lay.addWidget(self.packages_bar)
        self.run_bar = self._make_run_bar()
        lay.addWidget(self.run_bar)
        self.stack = QtWidgets.QStackedWidget()
        self.scroll = _FitWidthScrollArea(objectName="notebookScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # cells follow the width
        self.container = QtWidgets.QWidget(objectName="notebookPage")
        self.cells_layout = QtWidgets.QVBoxLayout(self.container)
        self.cells_layout.setContentsMargins(6, 6, 10, 16)
        self.cells_layout.setSpacing(6)
        self.scroll.setWidget(self.container)
        self.container.setAcceptDrops(True)       # cells dragged by their ⠿ grip
        self.container.installEventFilter(self)
        self.drop_line = QtWidgets.QFrame(self.container, objectName="cellDropLine")
        self.drop_line.setFixedHeight(3)
        self.drop_line.hide()
        self.stack.addWidget(self.scroll)
        self.file_view = CodeEditor()
        self.file_view.setReadOnly(True)
        self.file_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.file_view.textChanged.disconnect(self.file_view._fit)
        self.file_view.setMaximumHeight(16777215)
        self.file_view.setMinimumHeight(0)
        self.file_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.stack.addWidget(self.file_view)
        lay.addWidget(self.stack, 1)
        self.add_bar = self._make_add_bar()
        self._update_kernel_button()
        self.rebuild()

    def _make_banner(self):
        """The bar over a restricted notebook: saved results only, until you trust it."""
        w = QtWidgets.QFrame(objectName="restrictedBanner")
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(10, 6, 8, 6)
        self.banner_text = QtWidgets.QLabel(wordWrap=True)
        row.addWidget(self.banner_text, 1)
        trust = QtWidgets.QPushButton("Trust", toolTip="Allow this notebook's code and plugins to run")
        trust.clicked.connect(lambda: self.trust(ask=False))
        trust_run = QtWidgets.QPushButton("Trust and run all", objectName="primary")
        trust_run.clicked.connect(lambda: self.trust(ask=False) and self.run_all())
        row.addWidget(trust)
        row.addWidget(trust_run)
        self._update_banner(w)
        return w

    def _update_banner(self, banner=None):
        banner = banner or self.banner
        nb = self.notebook
        plugins = [p.get("name") or p.get("id") for p in nb.manifest.get("plugins", []) if p.get("file")]
        self.banner_text.setText(
            "<b>Restricted:</b> saved results only. Cells run once you trust this notebook" +
            (f" and its plugins ({', '.join(plugins)})" if plugins else "") + ".")
        banner.setToolTip("You have not saved or trusted this file before. Its cells are Python code, "
                          "so they run only after you agree.")
        banner.setVisible(nb.restricted)

    def _make_packages_bar(self):
        """The bar over a notebook whose steps come from pip packages this Python lacks."""
        w = QtWidgets.QFrame(objectName="restrictedBanner")
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(10, 6, 8, 6)
        self.packages_text = QtWidgets.QLabel(wordWrap=True)
        row.addWidget(self.packages_text, 1)
        self.packages_button = QtWidgets.QPushButton("Install…", toolTip="Install them with pip (you confirm the command)")
        self.packages_button.clicked.connect(
            lambda: self.pipRequested.emit([e["requirement"] for e in self.missing_packages()]))
        row.addWidget(self.packages_button)
        self.refresh_packages(w)
        return w

    def _make_run_bar(self):
        """The run picker over a notebook that loaded several runs (``runs = load_runs(...)``):
        which run's pipeline ``pipe`` is, so the 3D view and step cells follow it."""
        w = QtWidgets.QFrame(objectName="runBar")
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(10, 4, 8, 4)
        self.run_label = QtWidgets.QLabel("Run")
        self.run_combo = QtWidgets.QComboBox(toolTip="The run the 3D view shows; analyses on runs cover all of them")
        self.run_combo.activated.connect(self._pick_run)
        row.addWidget(self.run_label)
        row.addWidget(self.run_combo, 1)
        w.setVisible(False)
        self._runs = None
        return w

    def _find_runs(self):
        """The (first) :class:`ewald.runs.Runs` in the namespace, and its variable name."""
        from ..runs import Runs

        for name, value in list(self.notebook.namespace.items()):
            if isinstance(value, Runs) and len(value):
                return name, value
        return None, None

    def update_run_bar(self):
        name, runs = self._find_runs()
        self._runs = runs
        self.run_bar.setVisible(runs is not None)
        if runs is None:
            return
        self.run_label.setText(f"Run ({name})")
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItems(runs.labels)
        pipe = self.notebook.pipeline
        current = next((k for k, r in enumerate(runs) if r.pipe is pipe), runs.current)
        self.run_combo.setCurrentIndex(current)
        self.run_combo.blockSignals(False)

    def _pick_run(self, index: int):
        """Bind ``pipe`` to run ``index`` and show it."""
        runs = self._runs
        if runs is None or not 0 <= index < len(runs) or self.running:
            return
        pipe = runs.select(index)
        if self.notebook.pipeline is pipe:
            return
        self.notebook.namespace[self.notebook.pipeline_name] = pipe
        self.pipelineReplaced.emit(pipe)
        self.update_results()

    def missing_packages(self) -> list[dict]:
        from ..notebook_io import missing_packages

        return missing_packages(getattr(self.notebook, "manifest", None) or {})

    def refresh_packages(self, bar=None):
        bar = bar or self.packages_bar
        missing = self.missing_packages()
        names = ", ".join(f"{e['requirement']} ({', '.join(e.get('modifiers', [])) or e['id']})" for e in missing)
        self.packages_text.setText(f"<b>Needs plugins from pip:</b> {names}. Their steps fail until they are installed.")
        bar.setVisible(bool(missing))

    def _make_add_bar(self):
        w = QtWidgets.QWidget(objectName="addBar")
        row = FlowLayout(w, spacing=4)
        row.setContentsMargins(GUTTER_WIDTH + 2, 2, 0, 0)  # lined up with the cell bodies
        code = QtWidgets.QPushButton("+ Code", objectName="addCell", toolTip="Python cell (Ctrl+Shift+N)")
        code.clicked.connect(lambda: self.add_cell("code"))
        step = QtWidgets.QPushButton("+ Step", objectName="addCell", toolTip="Add an analysis or modification step")
        self.step_menu = QtWidgets.QMenu(step)
        self.step_menu.aboutToShow.connect(lambda: fill_modifier_menu(self.step_menu, self.add_step, self._step_context))
        step.setMenu(self.step_menu)
        plugin = QtWidgets.QPushButton("+ Plugin", objectName="addCell", toolTip="Plugin steps, or load a plugin file")
        self.plugin_menu = QtWidgets.QMenu(plugin)
        self.plugin_menu.aboutToShow.connect(self._fill_plugin_menu)
        plugin.setMenu(self.plugin_menu)
        text = QtWidgets.QPushButton("+ Text", objectName="addCell", toolTip="Markdown text cell")
        text.clicked.connect(lambda: self.add_cell("markdown"))
        view = QtWidgets.QPushButton("+ View", objectName="addCell",
                                     toolTip="A plot of a data table (g(r), XRD, DOS, bands, thermo) or a histogram")
        self.view_menu = QtWidgets.QMenu(view)
        self.view_menu.aboutToShow.connect(lambda: fill_view_menu(self.view_menu, self.data, self.add_view))
        view.setMenu(self.view_menu)
        ask = QtWidgets.QPushButton("+ Ask AI", objectName="addCell", toolTip="Ask AI to add steps (Ctrl+J)")
        ask.clicked.connect(self.askRequested.emit)
        for b in (code, step, view, plugin, text, ask):
            row.addWidget(b)
        return w

    def add_view(self, **kwargs) -> Cell:
        """Add a view cell (``show(pipe, **kwargs)``) after the current cell."""
        cell = self.notebook.add_view(self._index_after_current(), **kwargs)
        self.current = cell
        self.run(cell)  # rebuilds: the new cell gets its plot
        w = self.widget_for(cell)
        if w is not None:
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        return cell

    def _fill_plugin_menu(self):
        from .. import plugins

        m = self.plugin_menu
        m.clear()
        found = sorted(plugins.modifiers().items())
        for name, p in found:
            act = m.addAction(name, lambda p=p: self._add_plugin(p.cls))
            act.setToolTip(p.description)
        if not found:
            m.addAction("No plugin steps installed").setEnabled(False)
        m.addSeparator()
        m.addAction("Load plugin file…", self._load_plugin_file)
        m.addAction("Open plugin folder", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(plugins.plugin_dir()))))

    def _add_plugin(self, cls):
        try:
            mod = cls()
        except Exception as exc:  # needs arguments: give the user the code to fill in
            cell = Cell(f"from ewald.modifiers import {cls.__name__}\n"
                        f"{self.notebook.pipeline_name}.modifiers.append({cls.__name__}())  # {exc}")
            self._insert(cell)
            return
        self.add_step(mod)

    def _load_plugin_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load plugin file", "", "Python (*.py);;All files (*)")
        if path:
            cell = Cell(f"from ewald.plugins import load_plugin_file\nload_plugin_file({path!r})")
            self._insert(cell)
            self.run(cell)

    # ----------------------------------------------------------- building
    def rebuild(self):
        """Recreate every cell widget (after cells were added, removed or moved)."""
        keep = self.current
        for w in self.widgets:
            w.setParent(None)
            w.deleteLater()
        self.widgets = []
        while self.cells_layout.count():
            item = self.cells_layout.takeAt(0)
            if item.widget() is not None and item.widget() is not self.add_bar:
                item.widget().deleteLater()
        for cell in self.notebook.cells:
            w = CellWidget(self.notebook, cell)
            w.runRequested.connect(self.run)
            w.selected.connect(self.select)
            w.edited.connect(self._step_edited)
            w.actionRequested.connect(self._cell_action)
            self.widgets.append(w)
            self.cells_layout.addWidget(w)
            if w.plot_panel is not None:
                self.plotCreated.emit(w.plot_panel)
        self.cells_layout.addWidget(self.add_bar)
        # the spare height goes below the cells, not into them: a cell's gutter stretch makes it
        # count as vertically expanding, so a plain addStretch() would share the space with it
        self.cells_layout.addStretch(1)
        self.current = keep if keep in self.notebook.cells else (self.notebook.cells[-1] if self.notebook.cells else None)
        self._mark_current()
        self.update_results()

    def widget_for(self, cell):
        return next((w for w in self.widgets if w.cell is cell), None)

    def select(self, cell):
        self.current = cell
        self._mark_current()

    def _mark_current(self):
        for w in self.widgets:
            w.set_current(w.cell is self.current)

    def update_results(self, data=None):
        if data is not None:
            self.data = data
        self.notebook.refresh_kernel_steps()  # what the kernel's steps did at this frame
        if self.banner.isVisibleTo(self) != self.notebook.restricted:   # trusted elsewhere (console, open)
            self._update_banner()
        for w in self.widgets:
            w.update_state(self.data)
            if w.plot_panel is not None and (data is not None or w.plot_panel.data is None):
                w.set_view_data(self.data, self.notebook.pipeline)
        nb = self.notebook
        pipe = nb.pipeline
        steps = None if pipe is None else len(pipe.modifiers) + sum(len(c.steps) for c in nb.cells)
        # the tab already shows the notebook's name and ●; here only what it does not: steps, unsaved
        parts = [] if pipe is None else [f"{steps} step{'s' if steps != 1 else ''}"]
        if not nb.path:
            parts.append("not saved")
        self.status.setText(" · ".join(parts))
        self.status.setToolTip(nb.path or "Not saved yet (Ctrl+S)")
        if self.py_button.isChecked():
            self.file_view.setPlainText(nb.to_py())

    def _show_file(self, on):
        if on:
            self.file_view.setPlainText(self.notebook.to_py())
        self.stack.setCurrentIndex(1 if on else 0)

    # ----------------------------------------------------------- editing
    def _index_after_current(self):
        cells = self.notebook.cells
        return cells.index(self.current) + 1 if self.current in cells else len(cells)

    def _insert(self, cell, index=None):
        self.notebook.insert(self._index_after_current() if index is None else index, cell)
        self.current = cell
        self.rebuild()
        w = self.widget_for(cell)
        if w is not None and w.editor is not None:
            w.editor.setFocus()
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        self.history.commit("Insert cell")
        self.titleChanged.emit()
        return cell

    def add_cell(self, kind="code", source=""):
        return self._insert(Cell(source, kind=kind))

    def _step_context(self):
        """(current frame, pipeline) for the + Step menu, when the window shows this notebook."""
        win, pipe = self.window(), self.notebook.pipeline
        data = getattr(win, "data", None) if getattr(win, "pipeline", None) is pipe else None
        return data, pipe

    def add_step(self, modifier):
        """A step cell for ``modifier`` after the current cell; the pipeline gets it right away."""
        if self.notebook.pipeline is None:
            QtWidgets.QMessageBox.information(self, "No pipeline yet",
                                              "Open a file first, or run a cell that sets pipe = import_file(...).")
            return None
        cell = self.notebook.add_step(modifier, self._index_after_current())
        self.current = cell
        self.rebuild()
        w = self.widget_for(cell)
        if w is not None:
            w.form_button.setChecked(True)
            QtCore.QTimer.singleShot(0, w, lambda: self.scroll.ensureWidgetVisible(w))
        self.history.commit(f"Add {modifier.name}")
        self.changed.emit()
        self.titleChanged.emit()
        return cell

    def _step_edited(self, cell):
        mod = cell.modifier
        self.history.commit(f"Edit {mod.name if mod is not None else 'step'}", merge=("form", id(cell)))
        if self.notebook.kernel is not None:  # the step lives in the kernel: run its new code there
            self.run(cell)
            return
        self.changed.emit()
        self.titleChanged.emit()

    def _cell_action(self, cell, action):
        nb = self.notebook
        i = nb.cells.index(cell)
        if action == "delete":
            nb.delete(cell)
            self.current = nb.cells[min(i, len(nb.cells) - 1)] if nb.cells else None
            self.rebuild()
            self.changed.emit()
        elif action in ("up", "down"):
            nb.move(cell, i + (-1 if action == "up" else 1))
            self.rebuild()
            self.changed.emit()
        elif action == "code":
            w = self.widget_for(cell)
            w.code_button.setChecked(True)
        elif action == "below":
            self.run(cell, below=True)
        elif action == "clear":
            self.clear_outputs(cell)
            return
        self.history.commit({"delete": "Delete cell", "up": "Move cell", "down": "Move cell"}.get(action, "Edit cell"))
        self.titleChanged.emit()

    def clear_outputs(self, cell=None):
        """Clear outputs of ``cell`` or, with None, of the whole notebook (Notebook › Clear all outputs)."""
        self.notebook.clear_outputs(cell)
        for w in self.widgets if cell is None else [self.widget_for(cell)]:
            if w is not None:
                w.update_state(self.data)
        self.changed.emit()
        self.titleChanged.emit()

    def move_cell(self, cell, before: int):
        """Move ``cell`` in front of the cell at ``before`` (``len(cells)``: to the end)."""
        nb = self.notebook
        i = nb.cells.index(cell)
        to = before - 1 if before > i else before
        if to == i:
            return
        nb.move(cell, to)
        self.current = cell
        self.rebuild()
        self.changed.emit()
        self.history.commit("Move cell")
        self.titleChanged.emit()

    def _drop_index(self, y: float) -> int:
        """Where a cell dropped at height ``y`` of the page goes: before the first cell whose middle is lower."""
        for k, w in enumerate(self.widgets):
            if y < w.geometry().center().y():
                return k
        return len(self.widgets)

    def eventFilter(self, obj, event):
        if obj is self.container and event.type() in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove,
                                                      QtCore.QEvent.DragLeave, QtCore.QEvent.Drop):
            if event.type() == QtCore.QEvent.DragLeave:
                self.drop_line.hide()
                return True
            if not event.mimeData().hasFormat(CELL_MIME):
                return False
            y = event.position().y()
            k = self._drop_index(y)
            if event.type() == QtCore.QEvent.Drop:
                self.drop_line.hide()
                cid = bytes(event.mimeData().data(CELL_MIME)).decode()
                cell = next((c for c in self.notebook.cells if c.id == cid), None)
                if cell is not None:
                    event.acceptProposedAction()
                    QtCore.QTimer.singleShot(0, self, lambda: self.move_cell(cell, k))  # not inside the drag
                return True
            event.acceptProposedAction()
            ref = self.widgets[k].geometry().top() - 5 if k < len(self.widgets) else \
                (self.widgets[-1].geometry().bottom() + 3 if self.widgets else 0)
            self.drop_line.setGeometry(8, max(0, ref), self.container.width() - 16, 3)
            self.drop_line.show()
            self.drop_line.raise_()
            return True
        return super().eventFilter(obj, event)

    # ----------------------------------------------------------- running
    def _busy(self, fn):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return fn()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    @property
    def running(self) -> bool:
        """Cells are running (on a worker thread or in the kernel; the GUI stays usable meanwhile)."""
        return self._worker is not None

    def _execute(self, fn, before, then=None):
        """Run ``fn`` (cells) on a worker thread, so the window keeps responding and Interrupt
        can stop them; a run that ends within :data:`CELL_BUDGET` finishes before this returns."""
        if self.running:
            return
        marked, self._marked = self._marked, []
        if self.notebook.kernel is None:
            _cells_draw_off_screen()
        worker = self._worker = _CellWorker(fn, self)
        self._show_running(marked)
        done = []

        def finished():
            if done:   # already handled synchronously, or the queued signal after that
                return
            done.append(True)
            self._worker = None
            worker.deleteLater()
            self._set_running(False)
            self._after_run(before)
            if then:
                then()

        worker.finished.connect(finished)
        worker.start()
        if CELL_BUDGET is None:
            worker.wait()
        else:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                worker.wait(int(CELL_BUDGET * 1000))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        if worker.isFinished():
            finished()
        else:
            self._set_running(True)

    def _show_running(self, cells):
        """Mark the cells about to run (counter ``*``, like Jupyter) until their results arrive."""
        for c in cells:
            w = self.widget_for(c)
            if w is not None and c.kind != "markdown":
                w.set_running()

    def _set_running(self, on: bool):
        for w in (self.run_all_button, self.restart_button, self.kernel_button):
            w.setEnabled(not on)
        self.interrupt_button.setVisible(on)
        if on:
            kernel = self.notebook.kernel
            self.status.setText(f"Running in {kernel.display_name}…" if kernel is not None else "Running…")
        for w in self.widgets:
            w.update_state(self.data)

    def interrupt(self):
        """Stop the running cell, like Jupyter's ■ (:meth:`ewald.notebook.Notebook.interrupt`)."""
        if self.running:
            self.notebook.interrupt()

    def wait(self, timeout: float = 120.0) -> bool:
        """Process events until kernel cells finish (scripts and tests); False on timeout."""
        import time

        end = time.monotonic() + timeout
        while self.running and time.monotonic() < end:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            if self._worker is not None:
                self._worker.wait(20)
        QtWidgets.QApplication.processEvents()
        return not self.running

    def _after_run(self, before):
        nb = self.notebook
        if len(nb.cells) != len(self.widgets) or any(not _same_shape(w.built_for, _shape(c))
                                                     for w, c in zip(self.widgets, nb.cells)):
            self.rebuild()
        self.history.commit("Run cell")  # what was typed into the cells that ran
        self.update_run_bar()
        if nb.pipeline is not before and nb.pipeline is not None:
            self.pipelineReplaced.emit(nb.pipeline)
        else:
            self.changed.emit()
        self.update_results()
        self.titleChanged.emit()

    def undo(self) -> str | None:
        return self._step_history(self.history.undo)

    def redo(self) -> str | None:
        return self._step_history(self.history.redo)

    def _step_history(self, step):
        if self.running:   # the cells are changing the notebook right now
            return None
        before = self.notebook.pipeline
        label = self._busy(step)
        if label is None:
            return None
        self.rebuild()
        if self.notebook.pipeline is not before and self.notebook.pipeline is not None:
            self.pipelineReplaced.emit(self.notebook.pipeline)
        else:
            self.changed.emit()
        self.titleChanged.emit()
        return label

    def run(self, cell, advance=False, below=False):
        if not self._trusted():
            return
        nb = self.notebook
        before = nb.pipeline
        go = None
        if cell in nb.cells and not nb.cells[nb.cells.index(cell)].kind == "markdown":
            start = nb.cells.index(cell)
            self._marked = nb.cells[start:] if below else [cell]
            if below:
                def go():
                    for k in range(start, len(nb.cells)):
                        if not nb.run_cell(k, run_below=False):
                            break
            else:
                def go():
                    nb.run_cell(cell)
        self._execute(go or (lambda: None), before, (lambda: self._advance(cell)) if advance else None)

    def _advance(self, cell):
        nb = self.notebook
        if cell in nb.cells:
            i = nb.cells.index(cell)
            if i + 1 < len(nb.cells):
                self.select(nb.cells[i + 1])
                w = self.widgets[i + 1]
                if w.editor is not None and w.editor.isVisible():
                    w.editor.setFocus()
            else:
                self.add_cell("code")

    def run_all(self):
        if not self._trusted():
            return
        before = self.notebook.pipeline
        self._marked = list(self.notebook.cells)
        self._execute(self.notebook.run_all, before)

    def _trusted(self) -> bool:
        """A restricted notebook (a file someone sent) runs only after you agree."""
        return not self.notebook.restricted or self.trust(ask=True)

    def trust(self, ask: bool = True) -> bool:
        """Let a restricted notebook run (File › Trust notebook, the banner); ``ask`` confirms first."""
        nb = self.notebook
        if not nb.restricted:
            return True
        if ask:
            plugins = [p.get("name") or p.get("id") for p in nb.manifest.get("plugins", []) if p.get("file")]
            text = (f"You have not saved or trusted {nb.name} before. Its cells are Python code" +
                    (f" and it carries plugins ({', '.join(plugins)})" if plugins else "") +
                    ". Trust it and run?")
            if QtWidgets.QMessageBox.question(self, "Trust notebook", text) != QtWidgets.QMessageBox.Yes:
                return False
        nb.trust()
        self._update_banner()
        self.titleChanged.emit()
        return True

    def restart(self):
        if self.running:
            return
        self.notebook.reset()
        self.update_run_bar()
        self.update_results()

    # ------------------------------------------------------------ kernel
    def _update_kernel_button(self):
        k = self.notebook.kernel
        self.kernel_button.setText(k.display_name if k else "This app")

    def fill_kernel_menu(self, menu):
        """Kernel choices: this app's Python, the installed Jupyter kernels, another Python."""
        from .. import kernel as kmod

        menu.clear()
        current = self.notebook.kernel
        here = menu.addAction("This app's Python", lambda: self.set_kernel(None))
        here.setCheckable(True)
        here.setChecked(current is None)
        menu.addSeparator()
        try:
            specs = kmod.available()
        except Exception:  # noqa: BLE001 - no jupyter_client: only this app and "Other Python…"
            specs = []
        for spec in specs:
            a = menu.addAction(spec["display_name"], lambda sp=spec: self.set_kernel(
                kmod.Kernel(name=sp["name"], display_name=sp["display_name"])))
            a.setCheckable(True)
            a.setChecked(current is not None and current.name == spec["name"])
            a.setToolTip(spec["python"])
        if current is not None and not current.name:
            a = menu.addAction(current.display_name)
            a.setCheckable(True)
            a.setChecked(True)
            a.setToolTip(current.python or "")
        menu.addAction("Other Python…", self._choose_python)

    def _choose_python(self):
        from ..kernel import Kernel

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Python to run the cells", "",
                                                        "Python (python* python*.exe);;All files (*)")
        if path:
            self.set_kernel(Kernel(python=path))

    def set_kernel(self, kernel):
        """Run this notebook's cells in ``kernel`` (None: in this app). The namespace starts over."""
        before = self.notebook.pipeline
        self.notebook.set_kernel(kernel)
        self._update_kernel_button()
        self.update_results()
        self._after_run(before)
        self.titleChanged.emit()


# ---------------------------------------------------------------- tabs
CONFIRM_UNSAVED = True  # ask before closing notebooks with unsaved changes (tests turn it off)


class NotebookPanel(QtWidgets.QWidget):
    """Notebook tabs. Speaks the panel interface the main window uses
    (``set_pipeline``, ``set_data``, ``add_modifier``, ``refresh``, ``changed``)."""

    changed = QtCore.Signal()
    styleChanged = QtCore.Signal()            # unused: display settings live in the View tab
    pipelineReplaced = QtCore.Signal(object)
    askRequested = QtCore.Signal()
    plotCreated = QtCore.Signal(object)       # a view cell's live plot: the main window links it
    pipRequested = QtCore.Signal(list)        # a notebook's bar asks to pip install its plugins
    switched = QtCore.Signal(object, object)  # notebooks of the tab left and the tab shown (either may be None)

    def __init__(self, parent=None):
        super().__init__(parent)
        _discover()
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = NotebookTabs()
        self.tabs.newRequested.connect(lambda: self.new_notebook())
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.renameRequested.connect(self.rename)
        self.tabs.currentChanged.connect(self._tab_changed)
        lay.addWidget(self.tabs)
        self.add_menu = QtWidgets.QMenu(self)  # Pipeline menu / palette: same entries as + Step
        self.add_menu.aboutToShow.connect(lambda: fill_modifier_menu(
            self.add_menu, self.add_modifier, self.view._step_context if self.view is not None else None))
        fill_modifier_menu(self.add_menu, self.add_modifier)
        self._switching = False
        self.plot_style = ("publication", None)   # (style, tokens) of the cells' own matplotlib plots
        self.new_notebook()

    # ----------------------------------------------------------- tabs
    @property
    def view(self) -> NotebookView | None:
        return self.tabs.currentWidget()

    @property
    def notebook(self) -> Notebook | None:
        return self.view.notebook if self.view is not None else None

    @property
    def pipeline(self):
        return self.notebook.pipeline if self.notebook is not None else None

    def views(self) -> list[NotebookView]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def refresh_packages(self):
        """Plugins changed (a pip install): update every notebook's missing-packages bar."""
        for v in self.views():
            v.refresh_packages()

    def _next_name(self):
        names = {v.notebook.name for v in self.views()}
        k = 1
        while f"Notebook {k}" in names:
            k += 1
        return f"Notebook {k}"

    def set_plot_style(self, style: str = "publication", tokens: dict | None = None):
        """Style of the plots cells draw themselves, in every open notebook and new ones
        (``python.plot_style``; see :meth:`ewald.notebook.Notebook.set_plot_style`)."""
        self.plot_style = (style, tokens)
        for v in self.views():
            v.notebook.set_plot_style(style, tokens)

    def add_notebook(self, nb: Notebook) -> NotebookView:
        nb.set_plot_style(*self.plot_style)
        spec = (nb.metadata.get("ewald") or {}).get("kernel")
        if spec and nb.kernel is None:  # saved with a kernel: it starts when a cell runs
            from ..kernel import Kernel

            nb.kernel = Kernel.from_spec(spec)
        view = NotebookView(nb)
        view.changed.connect(self.changed.emit)
        view.pipelineReplaced.connect(self._replaced)
        view.askRequested.connect(self.askRequested.emit)
        view.plotCreated.connect(self.plotCreated.emit)
        for w in view.widgets:  # built before the connection
            if w.plot_panel is not None:
                self.plotCreated.emit(w.plot_panel)
        view.titleChanged.connect(lambda v=view: self._retitle(v))
        view.pipRequested.connect(self.pipRequested.emit)
        self._switching = True
        i = self.tabs.addTab(view, nb.name)
        self.tabs.setCurrentIndex(i)
        self._switching = False
        self._retitle(view)
        return view

    def new_notebook(self, name=None) -> NotebookView:
        nb = new_notebook(name or self._next_name())
        nb.run_all()  # just imports: gives the namespace the modifier names
        nb.dirty = False
        view = self.add_notebook(nb)
        view.update_results()
        return view

    def _retitle(self, view):
        i = self.tabs.indexOf(view)
        if i >= 0:
            self.tabs.setTabText(i, view.notebook.name + (" ●" if view.notebook.dirty else ""))
            self.tabs.setTabToolTip(i, view.notebook.path or "not saved")

    def _is_blank(self, view) -> bool:
        nb = view.notebook
        return nb.pipeline is None and not nb.dirty and nb.path is None

    def rename(self, index=None, name: str | None = None):
        """Rename a notebook (the tab; its file keeps its name until "Save as")."""
        view = self.tabs.widget(self.tabs.currentIndex() if index is None else index)
        if view is None:
            return
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename notebook", "Name:", text=view.notebook.name)
            if not ok or not name.strip():
                return
        view.notebook.name = name.strip()
        self._retitle(view)

    def confirm_discard(self, views=None, title="Close notebook") -> bool:
        """True if ``views`` (default: all) have no unsaved changes or the user lets them go."""
        dirty = [v.notebook.name for v in (self.views() if views is None else views) if v.notebook.dirty]
        if not dirty or not CONFIRM_UNSAVED:
            return True
        return QtWidgets.QMessageBox.question(
            self, title, f"Unsaved changes in {', '.join(dirty)}. Close anyway?") == QtWidgets.QMessageBox.Yes

    def close_tab(self, index):
        view = self.tabs.widget(index)
        if view is None or not self.confirm_discard([view]):
            return
        if view.notebook.kernel is not None:
            view.notebook.kernel.shutdown()
        self.tabs.removeTab(index)
        view.deleteLater()
        if self.tabs.count() == 0:
            self.new_notebook()

    def _tab_changed(self, index):
        old, self._shown = getattr(self, "_shown", None), self.view
        if old is not self._shown:   # also for tabs added or closed (_switching)
            self.switched.emit(None if old is None else old.notebook,
                               None if self._shown is None else self._shown.notebook)
        if self._switching or index < 0:
            return
        pipe = self.pipeline
        if pipe is not None:
            self.pipelineReplaced.emit(pipe)

    def _replaced(self, pipeline):
        if self.sender() is self.view:
            self.pipelineReplaced.emit(pipeline)

    # ----------------------------------------------------------- files
    def open_notebook(self, path, run: bool = False, trusted: bool | None = None) -> NotebookView:
        """Open ``path`` in a tab (or show its tab); ``trusted`` None: whether this user saved or
        trusted that file (:func:`ewald.trust.is_trusted_file`), otherwise it opens restricted."""
        if trusted is None:
            from ..trust import is_trusted_file

            trusted = is_trusted_file(path)
        nb = Notebook.load(path, trusted=trusted)
        for view in self.views():
            if view.notebook.path == nb.path:
                self.tabs.setCurrentWidget(view)
                return view
        blank = self.view if self.view is not None and self._is_blank(self.view) else None
        view = self.add_notebook(nb)
        if blank is not None:
            self.tabs.removeTab(self.tabs.indexOf(blank))
            blank.deleteLater()
        if run:
            view.trust(ask=False)   # the open already asked; also hides the restricted bar
            view.run_all()
        else:
            view.update_results()
        return view

    def save(self, path=None, **options) -> str | None:
        nb = self.notebook
        if nb is None:
            return None
        nb.save(path, **options)
        nb.name = os.path.splitext(os.path.basename(nb.path))[0]
        self._retitle(self.view)
        self.view.update_results()
        return nb.path

    # ----------------------------------------------------------- main window interface
    def set_pipeline(self, pipeline, source_code: str | None = None, name: str | None = None):
        """Show ``pipeline``: its tab if a notebook owns it, else a new notebook for it
        (replacing an untouched empty tab)."""
        for view in self.views():
            if view.notebook.pipeline is pipeline:
                if view is not self.view:
                    self._switching = True
                    self.tabs.setCurrentWidget(view)
                    self._switching = False
                view.update_results()
                return
        if name is None:
            src = getattr(pipeline.source, "paths", None) or [getattr(pipeline.source, "path", "")]
            name = os.path.splitext(os.path.basename(str(src[0])))[0] or self._next_name()
        nb = Notebook.from_pipeline(pipeline, source_code, name=name)
        blank = self.view if self.view is not None and self._is_blank(self.view) else None
        view = self.add_notebook(nb)
        if blank is not None:
            self._switching = True
            self.tabs.removeTab(self.tabs.indexOf(blank))
            blank.deleteLater()
            self._switching = False
        view.update_results()

    def set_data(self, data):
        self.data = data
        if self.view is not None:
            self.view.update_results(data)

    def add_modifier(self, mod):
        if self.view is not None and self.view.add_step(mod) is not None:
            pass

    def refresh(self, select=None):
        """After the pipeline changed outside the cells (console, Ask AI, viewport tools)."""
        view = self.view
        if view is None:
            return
        if view.notebook.sync():
            view.rebuild()
            self._retitle(view)
        view.history.commit("Change pipeline")  # console, Ask AI, viewport tools
        if select is not None:
            mod = select[1] if isinstance(select, tuple) and len(select) == 2 else select
            cell = next((c for c in view.notebook.cells if c.modifier is mod), None)
            if cell is not None:
                view.select(cell)
                w = view.widget_for(cell)
                QtCore.QTimer.singleShot(0, w, lambda: view.scroll.ensureWidgetVisible(w))
        view.update_results()

    def update_status(self):
        if self.view is not None:
            self.view.update_results()

    def undo(self) -> str | None:
        """Undo the last edit of the current notebook; returns what was undone."""
        return self.view.undo() if self.view is not None else None

    def redo(self) -> str | None:
        return self.view.redo() if self.view is not None else None

    def _current(self):
        return None
