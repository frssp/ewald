"""Links between the viewport, plots, Ask AI and the console: picks, selections and bond edits."""
from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import modifiers as mods
from ..io import import_file
from ..llm.notebook_tools import NotebookAccess, notebook_context, place_code, replace_source
from ..notebook import cell_from_source
from . import workers


def _alive(obj) -> bool:
    """Whether the Qt object behind ``obj`` still exists."""
    try:
        obj.objectName()
        return True
    except RuntimeError:
        return False


class _GuiCall(QtCore.QObject):
    """Runs a function on the GUI thread and hands back its result (or exception) to the caller's thread."""

    requested = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.requested.connect(self._run, QtCore.Qt.BlockingQueuedConnection)

    @QtCore.Slot(object)
    def _run(self, box):
        try:
            box["result"] = box["fn"]()
        except BaseException as exc:  # noqa: BLE001 - raised again in the caller's thread
            box["error"] = exc

    def __call__(self, fn):
        if QtCore.QThread.currentThread() is self.thread():
            return fn()
        box = {"fn": fn}
        self.requested.emit(box)
        if "error" in box:
            raise box["error"]
        return box.get("result")


class NotebookBridge(NotebookAccess):
    """The AI chat's notebook tools on one notebook tab (``notebook``; None: the current tab): edits
    happen on the GUI thread, redraw the cells, count as one undo step each ("AI: insert cell", Ctrl+Z)
    and recompute when the pipeline changed; run_cell (Preferences › AI › Let the AI run cells) runs
    like Shift+Enter. Each tab's chat has its own, so an answer still on its way after a switch keeps
    working on its own notebook."""

    def __init__(self, window, notebook=None):
        super().__init__(notebook if notebook is not None else (lambda: window.panel.notebook),
                         can_run=lambda: bool(window.settings["llm.run_cells"]))
        self.window = window
        self.call = _GuiCall()

    def _view(self, nb):
        return next((v for v in self.window.panel.views() if v.notebook is nb), None)

    def changed(self, nb, label, cell=None, pipeline=False):
        view = self._view(nb)
        if view is None:
            return
        if cell is not None and cell in nb.cells:
            view.current = cell
        view.rebuild()
        view.history.commit(label)
        view.titleChanged.emit()
        if pipeline:
            view.changed.emit()
        self.window.notebook_dock.show()
        self.window.message(f"{label} (Ctrl+Z undoes it)")
        view.ai_follow.touched(cell)

    def confirm_host(self, host: str) -> bool:
        """Ask the user (on the GUI thread) whether the AI may open files on SSH host ``host``."""
        def ask():
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Ask AI",
                                        f"The AI wants to open a file on {host} over SSH.",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, self.window)
            box.setInformativeText("Allow it to connect to this host for the rest of this chat? "
                                   "(Preferences › AI › Ask before the AI connects to a host)")
            box.setDefaultButton(QtWidgets.QMessageBox.No)
            return box.exec() == QtWidgets.QMessageBox.Yes

        return bool(self.call(ask))

    def run(self, nb, cell) -> bool:
        view = self.call(lambda: self._view(nb))
        if view is None:
            return nb.run_cell(cell)
        self.call(lambda: view.run(cell))
        while self.call(lambda: view.running):   # the cell runs on the notebook's own worker thread
            time.sleep(0.05)
        self.call(lambda: view.ai_follow.touched(cell))   # its outputs may have moved it off screen
        return cell.state == "ok"


class LinksMixin:
    """Part of :class:`ewald.gui.app.MainWindow`."""

    def toggle_bond(self, i, j):
        """Bond tool: delete the i-j bond if it exists, else add it, via an EditBonds modifier on top."""
        from ..modifiers import EditBonds

        if self.pipeline is None or self.data is None:
            return
        d = self.data
        ids = d["Particle Identifier"] if "Particle Identifier" in d else np.arange(d.count)
        bonded = d.bonds is not None and bool(np.any(np.all(np.sort(d.bonds.topology, axis=1) == sorted((i, j)), axis=1)))
        mods = self.pipeline.modifiers
        if not (mods and isinstance(mods[-1], EditBonds)):
            mods.append(EditBonds())
        mods[-1].toggle(int(ids[i]), int(ids[j]), bonded)
        self.panel.refresh(select=("mod", mods[-1]))
        self.recompute()
        self.message(f"{'Deleted' if bonded else 'Added'} bond {d.types[i]}{int(ids[i])}-{d.types[j]}{int(ids[j])}")

    def _on_pick(self, text):
        self.pick_label.setText(text)
        if text:
            self.statusBar().clearMessage()  # a status message hides the pick label until it times out
        if not self._plot_picking:
            self._each_plot(lambda p: p.forget_selection(frames=False))  # a click in the view replaces a plot range
        self._each_plot(lambda p: p.set_highlight(self._selected_indices()))
        self.ask_panel.update_selection()
        if text:
            self.log.appendPlainText(time.strftime("%H:%M:%S ") + text)

    # ---------------------------------------------------------------- linked plots
    def _connect_plot(self, p):
        """Link a plot (the Plots tab or a notebook view cell) to the viewport and timeline."""
        p.phononModePicked.connect(self.play_phonon_mode)
        p.atomsSelected.connect(self._plot_selected_atoms)
        p.framesSelected.connect(self._plot_selected_frames)
        p.frameRequested.connect(self.timeline._set)
        p.expressionRequested.connect(self._keep_plot_selection)
        p.axisViewRequested.connect(lambda req, p=p: self._align_view_to_plot(p, req))
        p.axisHovered.connect(lambda at, p=p: self._mark_plot_position(p, at))
        p.axisRangeSelected.connect(lambda rng, p=p: self._mark_plot_range(p, rng))
        p.structure = lambda: self.data
        self.timeline.selectionChanged.connect(p.set_frame_selection)

    def plot_panels(self) -> list:
        """The Plots tab and the live plots of the current notebook (view cells, ``show()`` outputs)."""
        view = self.panel.view
        return [self.plots, *self.extra_plots.values()] + ([p for w in view.widgets for p in w.panels()] if view else [])

    def new_plot_view(self, beside: str = "plot:1", view: str | None = None):
        """Another Plots view (a second table or histogram side by side), as a tab next to ``beside``."""
        if view is None:
            n = 2
            while f"plot:{n}" in self.workbench.specs:
                n += 1
            view = f"plot:{n}"
        from .plots import PlotsPanel

        p = PlotsPanel()
        self._connect_plot(p)
        p.pipeline = self.pipeline
        self.extra_plots[view] = p
        group = self.workbench.group_of(beside) if beside in self.workbench.specs else None
        self.workbench.register(view, p, f"Plots {view.split(':')[1]}", closable=True, group=group)
        self.workbench.show(view)
        self._stale_tabs.add(p)
        self._update_bottom_tab()
        return p

    def _make_view(self, view: str) -> bool:
        """Views a saved layout names that don't exist yet: extra plot views."""
        if view.startswith("plot:") and view[5:].isdigit() and view != "plot:1":
            self.new_plot_view(view=view)
            return True
        if view.startswith("plugin:"):     # a plugin's tab under the 3D view
            return self._build_plugin_tab(view[len("plugin:"):])
        return False

    def _plot_view_closed(self, view):
        p = self.extra_plots.pop(view, None)
        self._stale_tabs.discard(p)
        if view in self._plot_docks:   # a cell's plot: its thumbnail stays and opens it again
            self._closed_docks.add(view)

    # ---------------------------------------------------------------- cells' plots in the Plots panel
    def dock_plot(self, slot, show: bool = True):
        """Put the live plot of a cell (``slot``, a :class:`~ewald.gui.cell_widget.PlotSlot`) into
        the Plots panel as its own tab next to "Plots" (:func:`ewald.views.in_panel` says which
        plots go there). A plot of a notebook that is not the shown tab waits for its tab."""
        view = slot.view_id
        old = self._plot_docks.get(view)
        if old is not None and old[0] is not slot:   # the cell ran again: the new plot takes the tab
            self._drop_dock(view, old[1])
            old = None
        if old is None:
            key = id(slot)
            slot.destroyed.connect(lambda *_, v=view, k=key: self._drop_dock(v, k))
            self._plot_docks[view] = (slot, key, slot.panel)
        self._closed_docks.discard(view)
        if not self._in_shown_notebook(slot):
            return
        if view not in self.workbench.specs:
            group = self.workbench.group_of("plot:1") if "plot:1" in self.workbench.specs else None
            self.workbench.register(view, slot.panel, slot.title, closable=True, keep=True, group=group)
        if show:
            self.workbench.show(view)

    def undock_plot(self, slot):
        """Take a cell's plot out of the Plots panel (it goes back under its cell)."""
        entry = self._plot_docks.pop(slot.view_id, None)
        self._closed_docks.discard(slot.view_id)
        if entry is not None and slot.view_id in self.workbench.specs:
            self.workbench.unregister(slot.view_id, delete=False)

    def docked_plots(self) -> list[str]:
        """The tabs of cells' plots now in the Plots panel."""
        return [v for v in self._plot_docks if v in self.workbench.specs]

    def _drop_dock(self, view, key):
        """The cell of a docked plot is gone (or ran again): its tab and plot go too."""
        entry = self._plot_docks.get(view)
        if entry is None or entry[1] != key:
            return
        del self._plot_docks[view]
        self._closed_docks.discard(view)
        if view in self.workbench.specs and self.workbench.specs[view].widget is entry[2]:
            self.workbench.unregister(view)
        elif _alive(entry[2]) and entry[2].parent() is None:
            entry[2].deleteLater()

    def _in_shown_notebook(self, slot) -> bool:
        view = self.panel.view
        return view is not None and _alive(slot) and view.isAncestorOf(slot)

    def _sync_plot_docks(self, *_):
        """Another notebook tab is shown: its cells' plots come into the Plots panel, the others'
        leave it (kept by their cells, for when their tab is shown again)."""
        for view, (slot, key, panel) in list(self._plot_docks.items()):
            if not _alive(slot):
                self._drop_dock(view, key)
            elif self._in_shown_notebook(slot):
                if view not in self.workbench.specs and view not in self._closed_docks:
                    group = self.workbench.group_of("plot:1") if "plot:1" in self.workbench.specs else None
                    self.workbench.register(view, panel, slot.title, closable=True, keep=True, group=group)
            elif view in self.workbench.specs:
                self.workbench.unregister(view, delete=False)

    def _extra_view_actions(self, view):
        if view.startswith("plot:"):
            return [("New plot tab", lambda: self.new_plot_view(view))]
        if view == "viewport" or view.startswith("viewport:"):
            return [("New structure tab", lambda: self.new_view_tab())]
        return []

    def _each_plot(self, fn):
        for p in self.plot_panels():
            fn(p)

    def _selected_indices(self):
        """The viewport picks, else the 'Selection' property of the current frame."""
        indices = self.viewport.picks
        if not indices and self.data is not None and "Selection" in self.data:
            indices = np.nonzero(self.data.selection)[0].tolist()
        return indices

    def _plot_selected_atoms(self, indices):
        self._plot_picking = True
        try:
            self.viewport.set_picks(indices)
        finally:
            self._plot_picking = False

    def _plot_selected_frames(self, frames):
        self.timeline.set_selection(frames)
        if frames and self.frame not in frames:
            self.timeline._set(frames[0])

    def _keep_plot_selection(self, expression):
        from ..modifiers import ExpressionSelection

        if self.pipeline is None:
            return
        m = ExpressionSelection(expression)
        self.pipeline.modifiers.append(m)
        self._each_plot(lambda p: p.forget_selection(frames=False))
        self.viewport.clear_picks()
        self.panel.refresh(select=("mod", m))
        self.recompute()
        self.message(f"Added Expression selection: {expression}")

    # ---------------------------------------------------------------- the 3D view along a profile
    def _align_view_to_plot(self, plot, request):
        """A profile along the cell asks the 3D view to follow it (``(axis, lo, hi)``), or to stop
        (None). One plot at a time: aligning to another one releases the first."""
        if request is None:
            if self._aligned_plot is plot:
                self._aligned_plot = None
            return
        d = plot.space_data()
        if d is None or not d.cell.is_valid:
            return
        old = self._aligned_plot
        if old is not None and old is not plot and _alive(old):
            old.sync.setChecked(False)
        self._aligned_plot = plot
        self.viewport.align_to_axis(d.cell, *request)

    def _view_moved(self):
        """The 3D view's camera moved: an aligned plot shows the same range, or lets go when the
        view was turned away from the profile's direction."""
        plot = self._aligned_plot
        if plot is None or not _alive(plot) or plot.link is None or plot.link.kind != "space":
            self._aligned_plot = None
            return
        d = plot.space_data()
        rng = None if d is None else self.viewport.axis_view_range(d.cell, plot.link.axis)
        if rng is None:
            plot.sync.setChecked(False)
            self.message("3D view turned away from the profile: no longer aligned (Align 3D view aligns it again)")
            return
        plot.follow_view(*rng)

    def _plot_cell(self, plot, axis):
        """The cell a profile's planes span in the 3D view: its frame's, grown across the copies
        drawn side by side (not along the profile); None when the view shows another structure."""
        from ..data import Cell
        from ..replicate import normalize_replicate

        d = plot.space_data()
        if d is None or not d.cell.is_valid:
            return None
        rep = self.viewport.style.replicate
        cell, n = d.cell, normalize_replicate(rep) if rep else None
        if n is not None:
            grow = np.array([1 if a == axis else k for a, k in zip("abc", n)], float)
            cell = Cell(cell.matrix * grow[:, None], cell.origin, cell.pbc)
        return cell

    def _mark_plot_position(self, plot, at):
        """A see-through plane in the 3D view where the cursor is on a profile (None: remove it)."""
        from ..linking import axis_plane

        cell = self._plot_cell(plot, at[0]) if at is not None else None
        self.viewport.set_axis_plane(None if cell is None else axis_plane(cell, *at))

    def _mark_plot_range(self, plot, rng):
        """A see-through slab in the 3D view over the range dragged on a profile (None: remove it)."""
        from ..linking import axis_slab

        cell = self._plot_cell(plot, rng[0]) if rng is not None else None
        self.viewport.set_axis_slab(None if cell is None else axis_slab(cell, *rng))

    # ---------------------------------------------------------------- Ask AI
    def _ask_context(self):
        """(pipeline, frame, indices) for the Ask AI panel: the viewport picks, else the
        'Selection' property of the current frame."""
        return self.pipeline, self.frame, self._selected_indices()

    def ask_claude(self):
        self._show_side_bar(self.ask_dock)  # the AI chat side bar beside the notebook
        self.ask_panel.update_selection()
        self.ask_panel.input.setFocus()

    def ask_about_cell(self, text):
        """Cell menu › Ask AI about this cell: the chat side bar with the cell quoted, ready for the question."""
        self.ask_claude()
        box = self.ask_panel.input
        box.setPlainText(text)
        box.moveCursor(QtGui.QTextCursor.End)

    def _notebook_code(self):
        """The current notebook for the AI chat: its code with cell numbers, states and outputs
        (:func:`~ewald.llm.notebook_tools.notebook_context`; None without one)."""
        nb = self.panel.notebook
        return notebook_context(nb) if nb is not None and nb.cells else None

    def _view_summary(self):
        """How the 3D view colors the atoms of the current frame (:mod:`ewald.color_rule`), for the AI
        chat; None without data."""
        if self.data is None:
            return None
        from ..color_rule import describe, rule, take

        kw = self.style.scene_kwargs()
        return describe(rule(take(self.data, kw, self.frame), self.style.type_colors))

    def _notebook_errors(self):
        """The errors of the current notebook's failed cells, for the AI chat (None without any)."""
        nb = self.panel.notebook
        if nb is None:
            return None
        from ..llm.notebook_tools import failed_cells

        return failed_cells(nb) or None

    def _ai_answer(self, notebook, started: bool):
        """An AI chat answer about ``notebook`` starts or ends: its tab follows the cells the AI
        writes as Preferences › AI › Follow the AI's cells says (:mod:`ewald.gui.ai_follow`)."""
        view = next((v for v in self.panel.views() if v.notebook is notebook), None)
        if view is None:
            return
        if started:
            view.ai_follow.begin(self.settings["llm.follow_cells"])
        else:
            view.ai_follow.end()

    def _show_ai_cell(self, notebook, cell_id):
        """A cell link in the AI chat: show that notebook's tab and the cell, lit up for a moment."""
        view = next((v for v in self.panel.views() if notebook is None or v.notebook is notebook), None)
        cell = None if view is None else next((c for c in view.notebook.cells if c.id == cell_id), None)
        if cell is None:
            self.message("That cell is no longer in the notebook")
            return
        self.panel.tabs.setCurrentWidget(view)
        self.notebook_dock.show()
        self.notebook_dock.raise_()
        view.reveal(cell)

    def _insert_claude_cell(self, code, run=False, replace=False):
        """'Insert as notebook cell' in the AI chat: a code cell after the current one, not run
        unless ``run`` (an answer's code block while Preferences › AI › Let the AI run cells is on).
        With ``replace`` (an answer's code block going in by itself, Code in answers = insert) the code
        takes the place of the one cell that does the same job (the same modifier's step, a plot of
        the same table, the same view tab), so a changed option does not leave the old cell behind."""
        view = self.panel.view
        if view is None:
            view = self.panel.new_notebook()
        nb = view.notebook
        k = place_code(nb, code) if replace else None
        if k is not None:
            cell, new = nb.cells[k], cell_from_source(code)
            if cell.source.strip() != new.source.strip() or (new.frames or None) != (cell.frames or None):
                removed = replace_source(nb, cell, new.source, new.frames)
                if new.title:
                    cell.title = new.title
                view.current = cell
                view.rebuild()
                view.history.commit("AI: edit cell")
                view.titleChanged.emit()
                if removed:
                    view.changed.emit()
            done = f"the AI's code went into cell {k + 1}, which did the same"
        else:
            cell = view.add_cell("code", code)
            done = "inserted the AI's code as a notebook cell"
        self.notebook_dock.show()
        if run and cell is not None and not nb.restricted:
            self._ai_runs = [*getattr(self, "_ai_runs", []), (view, cell)]
            self._run_ai_cells()
            self.message(f"{done[0].upper()}{done[1:]}; running it")
        else:
            self.message(f"{done[0].upper()}{done[1:]} (Shift+Enter runs it)")

    def _run_ai_cells(self):
        """Run the AI's inserted cells in order, each once its notebook is idle."""
        queue = getattr(self, "_ai_runs", [])
        while queue:
            view, cell = queue[0]
            if not workers.alive(view) or cell not in view.notebook.cells:
                queue.pop(0)
            elif view.running:
                QtCore.QTimer.singleShot(50, self, self._run_ai_cells)
                return
            else:
                queue.pop(0)
                view.run(cell)

    def install_packages(self, requirements):
        """pip install ``requirements`` in the Plugins side bar (it asks first and shows pip's output):
        what a notebook's steps need, or an optional component from Preferences › Compute."""
        self._show_side_bar(self.plugins_dock)
        self.plugins_panel.install_requirements(requirements)

    def _create_plugin(self, prompt):
        self.ask_claude()
        self.ask_panel.input.setPlainText(prompt)

    def _after_claude(self):
        self.panel.refresh()
        self.recompute()

    # ---------------------------------------------------------------- console
    def _console_namespace(self):
        return {"np": np, "mods": mods, "viewer": self, "vp": self.viewport, "pipeline": self.pipeline,
                "pipe": self.pipeline, "data": self.data, "import_file": import_file, "style": self.style}

    def _push_console(self):
        if not getattr(self.console, "built", True):
            return  # the console gets the current objects when it is built
        shell = getattr(self.console, "shell", None)
        nb = self.panel.notebook
        ns = {"pipeline": self.pipeline, "pipe": self.pipeline, "data": self.data}
        if nb is not None and nb.kernel is None and hasattr(shell, "init_create_namespaces"):
            # the console runs in the notebook's namespace (one IPython namespace, as a Jupyter
            # console attached to a kernel): cells and console work on the same objects
            from .. import shell as ipython

            ipython.bind(shell, nb.namespace)
            extras = {**self._console_namespace(), "data": self.data, "notebook": nb}
            if nb.pipeline is not None:
                extras["pipe"] = extras["pipeline"] = nb.pipeline
            # console helpers stay out of %who and never hide a notebook variable of the same name
            hidden, ns_ = shell.user_ns_hidden, nb.namespace
            shell.push({k: v for k, v in extras.items()
                        if k not in ns_ or (k in hidden and hidden[k] is ns_[k])},   # unset, or ours from last time
                       interactive=False)
            return
        if nb is not None:   # cells in another Python: the console sees copies of what is here
            ns = {k: v for k, v in nb.namespace.items() if not k.startswith("__")} | ns
            ns["notebook"] = nb
        if hasattr(shell, "push") and hasattr(shell, "user_ns"):
            shell.push(ns)
        elif shell is not None and hasattr(shell, "locals"):
            shell.locals.update(ns)

    def _after_console(self, *args):
        if not getattr(self.console, "built", True):
            return
        shell = getattr(self.console, "shell", None)
        ns = getattr(shell, "user_ns", None) or getattr(shell, "locals", {})
        for key in ("pipe", "pipeline"):
            pipe = ns.get(key)
            if pipe is not None and pipe is not self.pipeline:
                self.set_pipeline(pipe)  # user did pipe = import_file(...)
                return
        if self.pipeline is not None:
            self.panel.refresh()
            self.timeline.set_num_frames(self.pipeline.num_frames, keep_selection=True)
            self.recompute()
