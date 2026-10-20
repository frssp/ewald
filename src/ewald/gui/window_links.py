"""Links between the viewport, plots, Ask AI and the console: picks, selections and bond edits."""
from __future__ import annotations

import time

import numpy as np

from .. import modifiers as mods
from ..io import import_file


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
        self.timeline.selectionChanged.connect(p.set_frame_selection)

    def plot_panels(self) -> list:
        """The Plots tab and the live plots of the current notebook's view cells."""
        view = self.panel.view
        return [self.plots, *self.extra_plots.values()] + ([w.plot_panel for w in view.widgets if w.plot_panel is not None] if view else [])

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
        return False

    def _plot_view_closed(self, view):
        p = self.extra_plots.pop(view, None)
        self._stale_tabs.discard(p)

    def _extra_view_actions(self, view):
        if view.startswith("plot:"):
            return [("New plot view", lambda: self.new_plot_view(view))]
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

    # ---------------------------------------------------------------- Ask AI
    def _ask_context(self):
        """(pipeline, frame, indices) for the Ask AI panel: the viewport picks, else the
        'Selection' property of the current frame."""
        return self.pipeline, self.frame, self._selected_indices()

    def ask_claude(self):
        self._show_side_bar(self.ask_dock)  # the AI chat side bar beside the notebook
        self.ask_panel.update_selection()
        self.ask_panel.input.setFocus()

    def _notebook_code(self):
        """The current notebook as percent-format Python, for the AI chat (None without one)."""
        nb = self.panel.notebook
        return nb.to_py() if nb is not None and nb.cells else None

    def _insert_claude_cell(self, code):
        """'Insert as notebook cell' in the AI chat: a code cell after the current one, not run."""
        view = self.panel.view
        if view is None:
            view = self.panel.new_notebook()
        view.add_cell("code", code)
        self.notebook_dock.show()
        self.message("Inserted the AI's code as a notebook cell (Shift+Enter runs it)")

    def _install_notebook_packages(self, requirements):
        """A notebook's bar asked for the pip packages its steps need: pip runs in the Plugins side bar."""
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
        ns = {"pipeline": self.pipeline, "pipe": self.pipeline, "data": self.data}
        if self.panel.notebook is not None:
            # the console sees the notebook's variables, so cells and console work on the same objects
            ns = {k: v for k, v in self.panel.notebook.namespace.items() if not k.startswith("__")} | ns
            ns["notebook"] = self.panel.notebook
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
