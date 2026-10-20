"""Notebooks: a pipeline written as Python cells, the document behind the GUI's notebook view.

A notebook is a plain ``.py`` file in the percent format (``# %%`` starts a cell) that
Jupyter, VS Code and jupytext open as a notebook and ``python file.py`` runs as a script::

    # %% [markdown]
    # # Cu void annealing

    # %%
    from ewald import load
    from ewald.modifiers import *
    pipe = load("cu_void.lammpstrj")

    # %%
    pipe.add(CoordinationAnalysis(cutoff=3.08))

Cells run in one namespace, like Jupyter. The variable ``pipe`` is the notebook's
pipeline; the viewer shows it. A cell that is a single ``pipe.add(...)`` (or ``pipe.modifiers.append(...)``)
(optionally after imports) is a *step*: the GUI shows it as a parameter form and keeps
its source in sync with the modifier. A cell that is a single ``show(pipe, ...)`` call
(:mod:`ewald.views`) is a *view*: a plot of a data table or a histogram, live and linked to
the 3D view in the GUI. Running any cell again replaces the modifiers it added the last
time at the same place, so re-running never stacks duplicates.

    nb = Notebook.load("analysis.py")
    nb.run_all()
    data = nb.pipeline.compute(0)
"""
from __future__ import annotations

import ast
import contextlib
import contextvars
import io
import json
import linecache
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import Modifier, Pipeline, PythonModifier

PIPELINE_NAME = "pipe"
_DATA_DIRS: contextvars.ContextVar = contextvars.ContextVar("ewald_notebook_data_dirs", default=())
_MARK = re.compile(r"^# %%(.*)$")
# A code cell tagged ``# %% [each frame]`` or ``# %% [all frames]`` is a Python step: its body
# runs at every frame with ``frame`` and ``data``, or once with ``frames`` and ``data``
# (:func:`ewald.pipeline.each_frame` / :func:`~ewald.pipeline.all_frames`).
FRAME_TAGS = {"each": "[each frame]", "all": "[all frames]"}
_FRAME_TAG = re.compile(r"\[(each frame|all frames)\]")


@dataclass
class Output:
    """One result shown under a cell: ``kind`` is 'text' (stdout), 'error', 'value'
    (repr of the last expression) or 'image' (PNG bytes: a matplotlib figure or a rendered image).
    ``show(pipe, ...)`` of the notebook's pipeline makes an image with ``view`` set: the GUI shows
    it as a live plot that follows the frame, everything else (files, Jupyter) as the image."""

    kind: str
    data: object
    view: dict | None = None   # an image drawn by ``show(pipe, ...)``: its arguments, for a live plot


@dataclass
class Cell:
    source: str = ""
    kind: str = "code"                 # 'code' or 'markdown'
    title: str = ""                    # text after '# %%' (besides [markdown] and the frames tag)
    frames: str = ""                   # '' (runs once), 'each' or 'all': the cell's body is a step, see FRAME_TAGS
    outputs: list = field(default_factory=list)
    added: list = field(default_factory=list)   # modifiers this cell's last run put into the pipeline
    count: int | None = None           # execution counter, like Jupyter's In [n]
    state: str = "new"                 # 'new', 'ok', 'error' or 'stale'
    id: str = field(default_factory=lambda: os.urandom(4).hex())   # nbformat cell id, kept in .ipynb
    collapsed: bool = False            # folded in the GUI; kept in .ipynb as Jupyter's source/outputs_hidden
    steps: list = field(default_factory=list)   # in a kernel: summaries of the steps it added (no objects here)
    shadow: object = field(default=None, repr=False, compare=False)  # in a kernel: a local copy of the step's modifier

    @property
    def is_step(self) -> bool:
        return self.kind == "code" and step_call(self.source) is not None

    @property
    def view(self) -> dict | None:
        """The arguments of a view cell (``show(pipe, table=...)``), else None."""
        if self.kind != "code":
            return None
        from .views import view_args

        return view_args(self.source)

    @property
    def is_view(self) -> bool:
        return self.view is not None

    @property
    def modifier(self) -> Modifier | None:
        """The modifier of a step cell (after it ran or was added by the GUI). In a kernel it is
        a local copy (``shadow``) built from the cell's code, for the form and the results."""
        if not self.is_step:
            return None
        if len(self.added) == 1:
            return self.added[0]
        return self.shadow


def _shadow_step(cell: Cell):
    """A local modifier built from a step cell's code (its imports and the ``pipe.add(...)``
    argument), or None when it cannot be built here (a class only the kernel has, a variable)."""
    from . import modifiers, plugins

    arg = step_call(cell.source)
    if arg is None:
        return None
    ns = {n: getattr(modifiers, n) for n in getattr(modifiers, "__all__", ())}
    ns.update({name: p.cls for name, p in plugins.modifiers().items() if getattr(p, "cls", None) is not None})
    try:
        tree = ast.parse(cell.source)
        head = ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[])
        exec(compile(head, "<step>", "exec"), ns)  # noqa: S102 - the cell's own import lines
        mod = eval(compile(ast.Expression(arg), "<step>", "eval"), ns)  # noqa: S307
    except Exception:  # noqa: BLE001 - shown as code then
        return None
    return mod if isinstance(mod, Modifier) else None


def _apply_summary(mod, summary: dict):
    mod.enabled = bool(summary.get("enabled", True))
    mod.status = summary.get("status")
    mod.last_time = summary.get("time")
    out = summary.get("output")
    mod.last_output = out if isinstance(out, dict) else None
    mod._kernel_id = summary.get("id")


def resolve_data_path(path):
    """``path`` itself when it exists (or is not a plain local path); else the same relative
    path, or file name, in the folders of the notebook that is running (its own folder and the
    data an ``.ewd`` carries)."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    from . import plugins

    if plugins.scheme_reader(text) is not None:  # aiida:1234 and other plugin sources
        return path
    if os.path.exists(text) or "://" in text or re.match(r"^[^/\\]+:/", text) or any(c in text for c in "*?["):
        return path
    for folder in _DATA_DIRS.get():
        for candidate in (os.path.join(folder, text), os.path.join(folder, os.path.basename(text))):
            if os.path.exists(candidate):
                return candidate
    return path


def _disables_last(stmt, name: str) -> bool:
    """``<name>.modifiers[-1].enabled = False`` (what :func:`step_source` writes for a step that is off)."""
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and ast.unparse(stmt.targets[0]) == f"{name}.modifiers[-1].enabled"
            and isinstance(stmt.value, ast.Constant) and stmt.value.value is False)


def step_call(source: str, name: str = PIPELINE_NAME) -> ast.expr | None:
    """The argument of ``<name>.add(arg)`` or ``<name>.modifiers.append(arg)`` if ``source`` is only that call,
    optionally preceded by import statements; None otherwise."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    body = list(tree.body)
    if body and _disables_last(body[-1], name):  # a switched-off step: pipe.modifiers[-1].enabled = False
        body.pop()
    *head, last = body or [None]
    if not isinstance(last, ast.Expr) or not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in head):
        return None
    call = last.value
    if not (isinstance(call, ast.Call) and len(call.args) == 1 and not call.keywords
            and isinstance(call.func, ast.Attribute)):
        return None
    target = call.func.value
    if call.func.attr == "add" and isinstance(target, ast.Name) and target.id == name:
        return call.args[0]                                   # pipe.add(X(...))
    if (call.func.attr == "append" and isinstance(target, ast.Attribute) and target.attr == "modifiers"
            and isinstance(target.value, ast.Name) and target.value.id == name):
        return call.args[0]                                   # pipe.modifiers.append(X(...))
    return None


def step_source(modifier: Modifier, namespace: dict | None = None, name: str = PIPELINE_NAME) -> str:
    """Source of a step cell that adds ``modifier``. Classes outside ``ewald.modifiers.__all__``
    (plugins) get their own import line unless the namespace already has them."""
    from . import modifiers

    if isinstance(modifier, PythonModifier):
        fn = modifier.function
        try:
            import inspect
            import textwrap

            src = textwrap.dedent(inspect.getsource(fn)).rstrip()
        except (OSError, TypeError):
            src = f"# {fn.__name__}: Python function defined elsewhere (console); its source is not available"
        params = "".join(f", {k}={v!r}" for k, v in modifier.params.items())
        call = f"{fn.__name__}" if not params else f"PythonModifier({fn.__name__}{params})"
        return f"{src}\n\n{name}.add({call})"
    cls = type(modifier)
    lines = []
    known = namespace is not None and namespace.get(cls.__name__) is cls
    if not known and cls.__name__ not in getattr(modifiers, "__all__", ()):
        module = "ewald.modifiers" if getattr(modifiers, cls.__name__, None) is cls else cls.__module__
        lines.append(f"from {module} import {cls.__name__}")
    lines.append(f"{name}.add({modifier!r})")
    if not modifier.enabled:
        lines.append(f"{name}.modifiers[-1].enabled = False")
    return "\n".join(lines)


def load_code(pipeline: Pipeline, name: str = PIPELINE_NAME) -> str | None:
    """``pipe = load(...)`` reproducing a pipeline's file source (None for other sources)."""
    src = getattr(pipeline, "source", None)
    if src is None or not hasattr(src, "path"):
        return None
    args = [repr(src.path)] + ([f"format={src.format!r}"] if getattr(src, "format", None) else []) + \
        [f"{k}={v!r}" for k, v in getattr(src, "kwargs", {}).items()]
    return f"{name} = load({', '.join(args)})"


def _owned(cells) -> list:
    return [m for c in cells for m in c.added]


class Notebook:
    """Cells, the namespace they run in, and the file they come from."""

    def __init__(self, cells=(), name: str = "Notebook 1", path: str | None = None, pipeline_name: str = PIPELINE_NAME):
        self.cells: list[Cell] = list(cells)
        self.name = name
        self.path = path
        self.pipeline_name = pipeline_name
        self.namespace: dict = {}
        self.counter = 0
        self.dirty = False
        self.metadata: dict = {}      # .ipynb notebook metadata, kept on a round trip
        self.state: dict = {}         # viewer state saved in .ewd files: style, camera, frame
        self.data_dirs: list = []     # more folders where load() looks (an .ewd's bundled data)
        self.bundle_dir: str | None = None   # where an .ewd file was unpacked
        self.restricted = False       # a file not trusted yet (see trust()): cells do not run
        self.manifest: dict = {}      # an .ewd file's manifest
        self.warnings: list = []      # data files that moved or changed (from the manifest)
        self.kernel = None            # an ewald.kernel.Kernel: cells run in another Python
        self.links: list = []         # data files a .py header links (path, size, fingerprint)
        self.py_header: list = []     # other lines of a .py file's jupytext header, kept on save
        self.plot_style = "publication"   # matplotlib style of the cells' own plots ('default': none)
        self.plot_tokens = None           # its colors (a UI theme's tokens); None = print colors
        self._plots = None                # ewald.plotstyle.Session: the cells' rcParams
        self.reset()

    # ------------------------------------------------------------ namespace
    def reset(self):
        """Fresh namespace (like restarting a Jupyter kernel); cells keep their source."""
        self.namespace = {"__name__": "__main__", "__file__": self.path or "<notebook>"}
        self._plots = None   # rcParams a cell set are forgotten too, as on a kernel restart
        if self.kernel is not None:
            self.kernel.reset()
        for c in self.cells:
            c.added, c.steps = [], []
            if c.kind == "code" and c.state != "new":
                c.state = "stale"

    def set_kernel(self, kernel):
        """Run the cells in ``kernel`` (an :class:`ewald.kernel.Kernel`), or here again with None.
        The namespace starts over; the choice is saved with the notebook."""
        if self.kernel is not None and self.kernel is not kernel:
            self.kernel.shutdown()
        self.kernel = kernel
        ewald = self.metadata.setdefault("ewald", {})
        if kernel is None:
            ewald.pop("kernel", None)
        else:
            ewald["kernel"] = kernel.spec
        self.dirty = True
        self.reset()

    @property
    def pipeline(self) -> Pipeline | None:
        p = self.namespace.get(self.pipeline_name)
        return p if isinstance(p, Pipeline) else None

    # ------------------------------------------------------------ running
    def run_all(self, stop_on_error: bool = True) -> bool:
        """Restart and run every cell in order; True if all succeeded."""
        self.reset()
        self.counter = 0
        for k, cell in enumerate(self.cells):
            if not self.run_cell(k, run_below=False) and stop_on_error:
                for later in self.cells[k + 1:]:
                    if later.kind == "code":
                        later.state = "stale"
                return False
        return True

    def run_cell(self, cell, run_below: bool = True) -> bool:
        """Run one cell (a Cell or its index) in the notebook namespace; True on success.

        Modifiers the cell added on its previous run are taken out first and the new ones
        land at the same place. When the cell rebinds ``pipe`` (a new import_file), the
        cells below run again so the new pipeline gets their steps (``run_below``)."""
        index = cell if isinstance(cell, int) else self.cells.index(cell)
        cell = self.cells[index]
        if cell.kind != "code":
            return True
        if self.kernel is not None:
            return self._run_in_kernel(index, run_below)
        pipe = self.pipeline
        slot = None
        if pipe is not None and cell.added:
            idx = [i for i, m in enumerate(pipe.modifiers) if any(m is a for a in cell.added)]
            if idx:
                slot = min(idx)
            for m in cell.added:
                _remove(pipe.modifiers, m)
        if pipe is not None and slot is None:
            slot = self._slot(index, pipe)
        cell.added = []
        before = [] if pipe is None else list(pipe.modifiers)
        self.counter += 1
        cell.count = self.counter
        ok = self._exec(cell, f"<cell {index + 1}>")
        after = self.pipeline
        if after is not None and after is pipe:
            new = [m for m in after.modifiers if not any(m is b for b in before)]
            for m in new:
                _remove(after.modifiers, m)
            for k, m in enumerate(new):
                after.modifiers.insert(min(slot, len(after.modifiers)) + k, m)
            cell.added = new
        elif after is not None:
            cell.added = list(after.modifiers)
            if run_below:
                for k in range(index + 1, len(self.cells)):
                    if not self.run_cell(k, run_below=False):
                        break
            else:
                for later in self.cells[index + 1:]:
                    later.added = []
        cell.state = "ok" if ok else "error"
        return ok

    def _run_in_kernel(self, index: int, run_below: bool) -> bool:
        """Run a cell in the notebook's kernel (the same rules apply there, see
        :mod:`ewald.kernel_bridge`); ``pipe`` here becomes a pipeline over the kernel's one."""
        import base64

        from .kernel import KernelError, KernelSource, kernel_pipeline

        self._require_trust()
        cell = self.cells[index]
        specs = [{"id": c.id, "kind": c.kind, "source": c.source, "frames": c.frames} for c in self.cells]
        try:
            r = self.kernel.run(specs, index, run_below, self.counter, self.pipeline_name, self.path,
                                self._data_dirs(), plot_style=(self.plot_style, self.plot_tokens))
        except KernelError as exc:
            self.counter += 1
            cell.count, cell.state, cell.outputs = self.counter, "error", [Output("error", str(exc))]
            return False
        self.counter = r["counter"]
        by_id = {c.id: c for c in self.cells}
        for rc in r["cells"]:
            c = by_id.get(rc["id"])
            if c is None:
                continue
            c.count, c.state, c.steps = rc["count"], rc["state"], rc["steps"]
            c.shadow = _shadow_step(c) if c.is_step and c.state == "ok" and len(c.steps) == 1 else None
            if c.shadow is not None:
                _apply_summary(c.shadow, c.steps[0])
            c.outputs = [Output(o["kind"], base64.b64decode(o["png"]) if "png" in o else o["text"], o.get("view"))
                         for o in rc["outputs"]]
        for cid, state in r["states"].items():
            if cid in by_id:
                by_id[cid].state = state
        info = r["pipeline"]
        pipe = self.namespace.get(self.pipeline_name)
        if info is None:
            self.namespace.pop(self.pipeline_name, None)
        elif isinstance(pipe, Pipeline) and isinstance(pipe.source, KernelSource) and pipe.source.kernel is self.kernel:
            pipe.source._num_frames = info["num_frames"]
            pipe.invalidate()
        else:
            self.namespace[self.pipeline_name] = kernel_pipeline(self.kernel, self.pipeline_name, info["num_frames"])
        version = info.get("view_version") if info is not None else None
        if version and version[1] and version != getattr(self, "_kernel_view", None):
            # a cell changed the kernel's pipe.view: the GUI viewport follows it
            from .view3d import View

            self._kernel_view = version
            view, settings = self.namespace[self.pipeline_name].view, View(**info["view"]).changed()
            if view.changed() != settings:
                view.reset().update(**settings)
        return r["ok"]

    def refresh_kernel_steps(self):
        """Copy what each step did at the last frame the kernel computed onto its cell's local copy."""
        if self.kernel is None:
            return
        results = getattr(self.kernel, "steps", {}) or {}
        for c in self.cells:
            if c.shadow is not None and c.steps and c.steps[0]["id"] in results:
                c.steps[0] = results[c.steps[0]["id"]]
                _apply_summary(c.shadow, c.steps[0])

    def _slot(self, index: int, pipe: Pipeline) -> int:
        """Where a cell's new modifiers go: after those of the cells above it."""
        above = [m for c in self.cells[:index] for m in c.added]
        positions = [i for i, m in enumerate(pipe.modifiers) if any(m is a for a in above)]
        return max(positions) + 1 if positions else 0 if any(c.added for c in self.cells[index + 1:]) else len(pipe.modifiers)

    def interrupt(self) -> bool:
        """Stop the running cell, like Jupyter's Interrupt; returns whether one was running.

        In a kernel this is a KeyboardInterrupt there. In-process (a cell running on another
        thread, as in the desktop app) :class:`CellInterrupted` is raised in that thread at its
        next Python instruction, so the cell ends with an error and Run all stops; a long call
        into compiled code (numba, C) finishes first."""
        if self.kernel is not None:
            self.kernel.interrupt()
            return True
        with _EXEC_LOCK:
            thread = self.__dict__.get("_exec_thread")
            if thread is None:
                return False
            import ctypes

            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread), ctypes.py_object(CellInterrupted))
            return True

    def trust(self):
        """Allow a restricted notebook (a file someone sent) to run: an ``.ewd``'s bundled
        plugins are loaded and its cells may run. The file is remembered as trusted
        (:func:`ewald.trust.remember_file`), so it opens trusted next time while it is unchanged."""
        if self.restricted and self.bundle_dir:
            from .notebook_io import load_bundled_plugins

            load_bundled_plugins(self)
        if self.restricted and self.path and os.path.isfile(self.path):
            from .trust import remember_file

            remember_file(self.path)
        self.restricted = False

    def _require_trust(self):
        from .trust import require

        require(not self.restricted, f"{self.name} (a notebook file you have not trusted)",
                "call notebook.trust() (Run all in the GUI asks first)")

    def _exec(self, cell: Cell, filename: str) -> bool:
        self._require_trust()
        token = _DATA_DIRS.set(tuple(self._data_dirs()))
        try:
            return self._exec_cell(cell, filename)
        finally:
            _DATA_DIRS.reset(token)

    def _data_dirs(self) -> list:
        dirs = [os.path.dirname(self.path)] if self.path else []
        return dirs + [d for d in self.data_dirs if d not in dirs]

    def set_plot_style(self, style: str = "publication", tokens: dict | None = None):
        """The matplotlib style of plots the cells draw themselves (``plt.plot`` ...):
        'publication' (:mod:`ewald.plotstyle`, in ``tokens`` colors or print colors) or 'default'
        (matplotlib's own). rcParams a cell set stay as the cell set them."""
        self.plot_style, self.plot_tokens = style, tokens
        if self._plots is not None:
            self._plots.configure(style, tokens)

    def _plot_session(self):
        """The cells' rcParams (:class:`ewald.plotstyle.Session`), or a no-op without matplotlib."""
        if self._plots is None:
            try:
                from .plotstyle import Session

                self._plots = Session(self.plot_style, self.plot_tokens)
            except ImportError:
                return contextlib.nullcontext()
        return self._plots.active()

    def _exec_cell(self, cell: Cell, filename: str) -> bool:
        with self._plot_session():
            return self._exec_cell_styled(cell, filename)

    def _exec_cell_styled(self, cell: Cell, filename: str) -> bool:
        cell.outputs = []
        lines = cell.source.splitlines(True)
        linecache.cache[filename] = (len(cell.source), None, lines, filename)
        out = io.StringIO()
        figures_before = _figure_numbers()
        restore_show = _patch_show()
        try:
            tree = ast.parse(cell.source, filename)
            if cell.kind == "code" and cell.frames in FRAME_TAGS:
                if self.pipeline is None:
                    raise NameError(f"a {FRAME_TAGS[cell.frames]} cell adds a step to `{self.pipeline_name}`: "
                                    "load a file first")
                tree, last = _frame_step(tree, cell, self.pipeline_name), None
            else:
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), _interruptible(self):
                exec(compile(tree, filename, "exec"), self.namespace)
                value = None if last is None else eval(compile(ast.Expression(last.value), filename, "eval"),
                                                       self.namespace)
            ok = True
        except BaseException as exc:  # noqa: BLE001 - shown in the cell, like Jupyter
            if isinstance(exc, KeyboardInterrupt):
                raise
            value, ok = None, False
            error = _format_error(exc, filename)
        finally:
            restore_show()
        if out.getvalue():
            cell.outputs.append(Output("text", out.getvalue()))
        if not ok:
            cell.outputs.append(Output("error", error))
        cell.outputs.extend(Output("image", png, view) for png, view in _new_figures(figures_before, self.pipeline))
        if ok and value is not None:
            cell.outputs.append(_display(value))
        return ok

    # ------------------------------------------------------------ editing
    def insert(self, index: int, cell: Cell) -> Cell:
        self.cells.insert(index, cell)
        self.dirty = True
        return cell

    def clear_outputs(self, cell: Cell | None = None):
        """Drop the outputs and run count of ``cell`` (default: every cell), like Jupyter's Clear
        outputs; the next save writes the cells without them. Results and steps stay as they are."""
        for c in self.cells if cell is None else [cell]:
            c.outputs = []
            c.count = None
            if c.state == "error":
                c.state = "new"
        self.dirty = True

    def delete(self, cell):
        """Remove a cell and the modifiers it added."""
        index = cell if isinstance(cell, int) else self.cells.index(cell)
        cell = self.cells.pop(index)
        pipe = self.pipeline
        if pipe is not None:
            for m in cell.added:
                _remove(pipe.modifiers, m)
        self.dirty = True
        return cell

    def set_frames(self, cell: Cell, frames: str) -> bool:
        """Make ``cell`` run once (``''``), as a step at each frame (``'each'``) or once over all
        frames (``'all'``); True if the cell has to run again to add its step. Going back to
        once takes out the step the cell added and leaves the cell stale (its code alone did
        not run yet)."""
        if frames not in ("", *FRAME_TAGS):
            raise ValueError(f"frames must be '', 'each' or 'all', not {frames!r}")
        if cell.frames == frames:
            return False
        cell.frames = frames
        self.dirty = True
        if frames:
            return True
        pipe = self.pipeline
        if pipe is not None and self.kernel is None:
            for m in cell.added:
                _remove(pipe.modifiers, m)
            cell.added = []
        cell.state = "stale"
        return False

    def set_kind(self, cell: Cell, kind: str) -> bool:
        """Turn ``cell`` into a ``'code'`` or ``'markdown'`` cell, keeping its text (Jupyter's Y / M).
        A code cell that becomes text loses its outputs and the steps it added; a text cell that
        becomes code is new (not run yet). True if the kind changed."""
        if kind not in ("code", "markdown"):
            raise ValueError(f"kind must be 'code' or 'markdown', not {kind!r}")
        if cell.kind == kind:
            return False
        if kind == "markdown":
            pipe = self.pipeline
            if pipe is not None and self.kernel is None:
                for m in cell.added:
                    _remove(pipe.modifiers, m)
            cell.added, cell.steps, cell.shadow, cell.frames = [], [], None, ""
            cell.outputs, cell.count, cell.state = [], None, "ok"
        else:
            cell.outputs, cell.count, cell.state = [], None, "new"
        cell.kind = kind
        self.dirty = True
        return True

    def move(self, cell, to: int):
        """Move a cell to position ``to``; the pipeline follows the new cell order."""
        index = cell if isinstance(cell, int) else self.cells.index(cell)
        cell = self.cells.pop(index)
        self.cells.insert(max(0, min(to, len(self.cells))), cell)
        self._reorder()
        self.dirty = True

    def _reorder(self):
        pipe = self.pipeline
        if pipe is None:
            return
        owned = [m for m in _owned(self.cells) if any(m is p for p in pipe.modifiers)]
        rest = [m for m in pipe.modifiers if not any(m is o for o in owned)]
        pipe.modifiers[:] = owned + rest

    def add_step(self, modifier: Modifier, index: int | None = None) -> Cell:
        """Insert a step cell for an already constructed modifier (GUI menus) at ``index``
        (default: the end) and put the modifier into the pipeline at the matching place."""
        index = len(self.cells) if index is None else index
        cell = Cell(step_source(modifier, self.namespace, self.pipeline_name))
        self.insert(index, cell)
        if self.kernel is not None:  # the kernel builds its own modifier from the cell
            self.run_cell(index, run_below=False)
            return cell
        pipe = self.pipeline
        if pipe is not None:
            pipe.modifiers.insert(self._slot(index, pipe), modifier)
            cell.added = [modifier]
            cell.state = "ok"
        return cell

    def add_view(self, index: int | None = None, **kwargs) -> Cell:
        """Insert a view cell (``show(pipe, **kwargs)``) at ``index`` (default: the end)."""
        from .views import view_source

        source = view_source(self.pipeline_name, **kwargs)
        if "show" not in self.namespace:
            source = "from ewald.views import show\n" + source
        cell = Cell(source)
        self.insert(len(self.cells) if index is None else index, cell)
        return cell

    def update_view(self, cell: Cell, **kwargs):
        """Rewrite a view cell for new arguments (the GUI plot's table or column changed)."""
        from .views import view_source

        args = dict(cell.view or {})
        source = args.pop("source", self.pipeline_name)
        args.update(kwargs)
        head = [line for line in cell.source.splitlines() if line.startswith(("import ", "from "))]
        new = "\n".join(head + [view_source(source, **args)])
        if new != cell.source:
            cell.source = new
            self.dirty = True

    def view_cell(self) -> Cell | None:
        """The view cell (``view(pipe, ...)``, the 3D view's display settings), or None."""
        from .view3d import view_cell_args

        for cell in self.cells:
            if cell.kind == "code":
                args = view_cell_args(cell.source)
                if args is not None and args["source"] == self.pipeline_name:
                    return cell
        return None

    def set_view_cell(self, settings: dict) -> Cell | None:
        """Write ``settings`` (:meth:`ewald.view3d.View.changed`) into the view cell, adding one at the
        end when there is none (and something to say). The cell is not run: the pipeline's view
        already holds these settings. Returns the cell, or None when nothing was written."""
        from .view3d import view_source

        cell = self.view_cell()
        if cell is None and not settings:
            return None
        head = [line for line in (cell.source if cell else "").splitlines() if line.startswith(("import ", "from "))]
        if cell is None and "view" not in self.namespace:
            head = ["from ewald import view"]
        new = "\n".join(head + [view_source(self.pipeline_name, **settings)])
        if cell is None:
            cell = self.insert(len(self.cells), Cell(new, state="ok"))
            self.namespace.setdefault("view", __import__("ewald").view)
        elif cell.source != new:
            cell.source = new
            self.dirty = True
        return cell

    def update_step(self, cell: Cell):
        """Rewrite a step cell's source from its modifier (after the form changed it)."""
        if cell.modifier is not None:
            cell.source = step_source(cell.modifier, self.namespace, self.pipeline_name)
            self.dirty = True

    def sync(self) -> bool:
        """Follow pipeline changes made outside the cells (console, Claude, menus): new
        modifiers become step cells after the cell owning the modifier before them, and
        step cells whose modifier left the pipeline are removed. True if cells changed."""
        pipe = self.pipeline
        if pipe is None:
            return False
        changed = False
        for cell in list(self.cells):
            keep = [m for m in cell.added if any(m is p for p in pipe.modifiers)]
            if len(keep) != len(cell.added):
                if cell.is_step and not keep:
                    self.cells.remove(cell)
                else:
                    cell.added = keep
                changed = True
        for i, m in enumerate(pipe.modifiers):
            if any(m is o for o in _owned(self.cells)):
                continue
            prev = pipe.modifiers[i - 1] if i else None
            at = len(self.cells)
            if prev is not None:
                at = next((k + 1 for k, c in enumerate(self.cells) if any(prev is a for a in c.added)), at)
            else:
                at = next((k for k, c in enumerate(self.cells) if c.added and c.added[0] is not None
                           and any(c.added[0] is p for p in pipe.modifiers)), at)
            cell = Cell(step_source(m, self.namespace, self.pipeline_name), added=[m], state="ok")
            self.cells.insert(at, cell)
            changed = True
        for cell in self.cells:  # sources follow parameter edits made elsewhere
            if cell.modifier is not None:
                src = step_source(cell.modifier, self.namespace, self.pipeline_name)
                if src != cell.source:
                    cell.source = src
                    changed = True
        self.dirty |= changed
        return changed

    # ------------------------------------------------------------ undo
    def snapshot(self) -> tuple:
        """The notebook's content as text: (kind, title, source, frames) per cell. Everything
        else (pipeline, outputs) follows from it by running the cells."""
        return tuple((c.kind, c.title, c.source, c.frames) for c in self.cells)

    def restore(self, snap: tuple) -> list[Cell]:
        """Make the cells match ``snap`` (undo/redo). Unchanged cells keep their outputs and
        modifiers; removed cells take their modifiers out of the pipeline. Changed or restored
        step and view cells run again so the pipeline follows; other code cells are only
        marked stale, since running them may have side effects. Returns the cells that changed."""
        import difflib

        old = self.cells
        matcher = difflib.SequenceMatcher(a=self.snapshot(), b=snap, autojunk=False)
        cells, fresh = [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                cells.extend(old[i1:i2])
                continue
            pipe = self.pipeline
            for c in old[i1:i2]:
                if pipe is not None:
                    for m in c.added:
                        _remove(pipe.modifiers, m)
            for kind, title, source, frames in snap[j1:j2]:
                cell = Cell(source, kind=kind, title=title, frames=frames, state="new" if kind == "code" else "ok")
                cells.append(cell)
                fresh.append(cell)
        self.cells = cells
        for cell in fresh:
            if cell.kind != "code":
                continue
            if cell.is_step or cell.is_view or cell.frames:
                self.run_cell(cell, run_below=False)
            else:
                cell.state = "stale"
        self._reorder()
        self.dirty = True
        return fresh

    # ------------------------------------------------------------ files
    def to_py(self, header: bool = False) -> str:
        """The notebook as a percent-format Python file. With ``header`` (what :meth:`save` writes)
        a jupytext-style comment header keeps the kernel, the viewer state (style, frame,
        camera, render settings) and links to the data files (path, size, fingerprint)."""
        parts = [self._py_header()] if header else []
        parts = [p for p in parts if p]
        for cell in self.cells:
            head = "# %%" + (f" {cell.title}" if cell.title else "")
            if cell.kind == "code" and cell.frames in FRAME_TAGS:
                head += " " + FRAME_TAGS[cell.frames]
            if cell.kind == "markdown":
                head += " [markdown]"
                body = "\n".join(("# " + line) if line else "#" for line in cell.source.rstrip("\n").splitlines())
            else:
                body = cell.source.rstrip("\n")
            parts.append(head + ("\n" + body if body else ""))
        return "\n\n".join(parts) + "\n"

    def _py_header(self) -> str:
        from . import notebook_io

        ewald = {k: v for k, v in (self.metadata.get("ewald") or {}).items() if k not in ("version", "state", "data")}
        if self.pipeline_name != PIPELINE_NAME:
            ewald["pipeline"] = self.pipeline_name
        if self.state:
            ewald["state"] = notebook_io._jsonable(self.state)
        base = os.path.dirname(self.path) if self.path else os.getcwd()
        links = [{"path": notebook_io._relative(f, base), "size": os.path.getsize(f),
                  "fingerprint": notebook_io.fingerprint(f)} for f in notebook_io.data_files(self)]
        links = links or self.links   # not run yet: keep the links it was opened with
        if links:
            ewald["data"] = links
        if not ewald and not self.py_header:
            return ""
        lines = ["# ---"]
        rest = list(self.py_header)
        if "# jupyter:" in rest:
            lines += rest[:rest.index("# jupyter:") + 1]
            rest = rest[rest.index("# jupyter:") + 1:]
        else:
            lines.append("# jupyter:")
        if ewald:
            lines.append("#   ewald:")
            lines += [f"#     {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in ewald.items()]
        return "\n".join(lines + rest + ["# ---"])

    def _read_py_header(self, header: list):
        """The ``ewald:`` part of a ``.py`` header back into metadata, state and data links; the
        other header lines (jupytext's own) are kept as they are."""
        ewald, other, inside = {}, [], False
        for line in header:
            if line.rstrip() == "#   ewald:":
                inside = True
                continue
            m = re.match(r"^#     ([A-Za-z_]+): (.*)$", line) if inside else None
            if m:
                try:
                    ewald[m.group(1)] = json.loads(m.group(2))
                    continue
                except ValueError:
                    pass
            inside = False
            other.append(line)
        self.py_header = [ln for ln in other if ln.rstrip() != "# jupyter:" or len(other) > 1]
        self.state = dict(ewald.pop("state", None) or {})
        self.links = list(ewald.pop("data", None) or [])
        if ewald.get("pipeline"):
            self.pipeline_name = ewald["pipeline"]
        if ewald:
            self.metadata.setdefault("ewald", {}).update(ewald)

    @classmethod
    def from_py(cls, text: str, **kwargs) -> "Notebook":
        """Parse a percent-format file; text before the first marker becomes a code cell
        (a plain script is one big cell). A leading ``# ---`` header (jupytext's) is read
        for ewald's kernel, viewer state and data links."""
        cells, current = [], None
        lines = []
        header = []
        rows = text.splitlines()
        if rows and rows[0].rstrip() == "# ---" and "# ---" in [r.rstrip() for r in rows[1:]]:
            end = 1 + [r.rstrip() for r in rows[1:]].index("# ---")
            header, text = rows[1:end], "\n".join(rows[end + 1:])

        def close():
            if current is None and not "".join(lines).strip():
                return
            cell = current or Cell()
            body = "\n".join(lines).strip("\n")
            if cell.kind == "markdown":
                body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            cell.source = body
            cells.append(cell)

        for line in text.splitlines():
            m = _MARK.match(line)
            if m:
                close()
                rest = m.group(1).strip()
                kind = "markdown" if re.search(r"\[(markdown|md)\]", rest) else "code"
                tag = _FRAME_TAG.search(rest) if kind == "code" else None
                frames = "" if tag is None else tag.group(1).split()[0]
                title = _FRAME_TAG.sub("", re.sub(r"\[(markdown|md)\]", "", rest)).strip()
                current, lines = Cell(kind=kind, title=title, frames=frames), []
            else:
                lines.append(line)
        close()
        nb = cls(cells, **kwargs)
        if header:
            nb._read_py_header(header)
        return nb

    @classmethod
    def load(cls, path, trusted: bool | None = None, **kwargs) -> "Notebook":
        """Read a ``.py`` (percent format), ``.ipynb`` or ``.ewd`` notebook.

        ``trusted=False`` opens it restricted (nothing runs until :meth:`trust`), ``True`` lets
        it run, and None (the default) trusts a ``.py``/``.ipynb`` (a script that loads a
        notebook runs what it chose, as ``exec`` would) but an ``.ewd`` only when this user saved
        or trusted that exact file (:func:`ewald.trust.is_trusted_file`). The desktop app passes
        ``is_trusted_file`` for every format, as Jupyter does."""
        from . import notebook_io

        kwargs.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".ewd":
            return notebook_io.load_ewd(path, cls=cls, trusted=trusted, **kwargs)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if ext == ".ipynb":
            nb = notebook_io.from_ipynb(text, cls=cls, path=os.path.abspath(path), **kwargs)
        else:
            nb = cls.from_py(text, path=os.path.abspath(path), **kwargs)
            if nb.links:
                nb.warnings = notebook_io.check_data({"data": nb.links}, os.path.dirname(nb.path), Path())
        nb.restricted = trusted is False
        return nb

    def save(self, path=None, **options):
        """Write the notebook; the extension picks the format: ``.ewd`` (code, outputs, figures,
        small data files and plugins in one file; ``options`` go to
        :func:`ewald.notebook_io.save_ewd`), ``.ipynb`` (code and outputs) or ``.py`` (code)."""
        from . import notebook_io

        path = path or self.path
        if path is None:
            raise ValueError("no file name for this notebook")
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".ewd":
            notebook_io.save_ewd(self, path, **options)
        else:
            self.path = os.path.abspath(path)   # data links are relative to the new file
            text = notebook_io.to_ipynb_text(self) if ext == ".ipynb" else self.to_py(header=True)
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        self.path = os.path.abspath(path)
        self.namespace["__file__"] = self.path
        self.dirty = False
        if not self.restricted:   # your own notebook opens trusted; a saved restricted one stays so
            from .trust import remember_file

            remember_file(self.path)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, source_code: str | None = None, check: bool = True,
                      **kwargs) -> "Notebook":
        """A notebook for a pipeline that already exists (a file opened from the menu):
        an import cell, a load cell (``source_code``, e.g. ``pipe = import_file(...)``) and one
        step per modifier. The pipeline object itself is used, nothing is read again. With
        ``check``, an MD run or a relaxation whose input files say so (:func:`ewald.run_info`)
        gets a ``check_run(pipe)`` cell, run at once, that says whether it converged and
        from which frame it is equilibrated."""
        nb = cls(**kwargs)
        imports = Cell(IMPORTS)
        nb.cells.append(imports)
        nb.run_cell(0)
        source_code = source_code or load_code(pipeline, nb.pipeline_name)
        load = Cell(source_code or f"# {nb.pipeline_name}: built in memory, not reproducible from a file", state="ok")
        nb.cells.append(load)
        nb.namespace[nb.pipeline_name] = pipeline
        for m in pipeline.modifiers:
            nb.cells.append(Cell(step_source(m, nb.namespace, nb.pipeline_name), added=[m], state="ok"))
        if check and _worth_checking(pipeline):
            nb.cells.append(Cell(f"from ewald import check_run\ncheck = check_run({nb.pipeline_name})\ncheck"))
            nb.run_cell(len(nb.cells) - 1, run_below=False)
        nb.dirty = False
        return nb


def _worth_checking(pipeline) -> bool:
    """An MD run or relaxation whose input files say so (a bare structure file is not)."""
    from .runinfo import run_info

    try:
        return run_info(pipeline).kind in ("md", "relax")
    except Exception:  # noqa: BLE001 - never stop a file from opening
        return False


IMPORTS = ("from ewald import import_file, load\nfrom ewald.modifiers import *\nfrom ewald.views import show, view\n"
           "from ewald.paint import paint, hide, isolate, fade, resize")
NEW_NOTEBOOK = '''# %% [markdown]
# # {title}

# %%
import numpy as np
''' + IMPORTS + "\n"


class History:
    """Undo/redo for a notebook, one entry per user action, kept as text snapshots.

    Call :meth:`commit` after every edit; it records an entry when the cells changed since the
    last commit. Edits with the same ``merge`` key in a row (dragging one slider of one step
    form) become one entry.
    """

    def __init__(self, notebook: Notebook, limit: int = 200):
        self.notebook = notebook
        self.limit = limit
        self._undo: list[tuple[str, tuple, tuple]] = []   # (label, before, after)
        self._redo: list[tuple[str, tuple, tuple]] = []
        self._last = notebook.snapshot()
        self._merge = None

    def commit(self, label: str = "Edit", merge=None) -> bool:
        now = self.notebook.snapshot()
        if now == self._last:
            return False
        if merge is not None and merge == self._merge and self._undo:
            lbl, before, _ = self._undo[-1]
            self._undo[-1] = (lbl, before, now)
        else:
            self._undo.append((label, self._last, now))
            del self._undo[:-self.limit]
        self._redo.clear()
        self._last, self._merge = now, merge
        return True

    def reset(self):
        """Forget the history (a notebook was opened or replaced)."""
        self._undo.clear()
        self._redo.clear()
        self._last, self._merge = self.notebook.snapshot(), None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo) or self.notebook.snapshot() != self._last

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        if self.notebook.snapshot() != self._last:
            return "Edit cell"
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1][0] if self._redo else None

    def undo(self) -> str | None:
        """Go back one entry (an edit not committed yet, such as typing in a cell, counts as
        one); returns its label, or None if there was nothing to undo."""
        self.commit("Edit cell")
        if not self._undo:
            return None
        label, before, after = self._undo.pop()
        self.notebook.restore(before)
        self._redo.append((label, before, after))
        self._last, self._merge = before, None
        return label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        label, before, after = self._redo.pop()
        self.notebook.restore(after)
        self._undo.append((label, before, after))
        self._last, self._merge = after, None
        return label


def new_notebook(name: str = "Notebook 1") -> Notebook:
    """An empty notebook with the usual imports."""
    return Notebook.from_py(NEW_NOTEBOOK.format(title=name), name=name)


# ---------------------------------------------------------------- helpers
def _remove(modifiers, m):
    for i, x in enumerate(modifiers):
        if x is m:
            del modifiers[i]
            return


def _format_error(exc: BaseException, filename: str) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    keep = [f for f in frames if f.filename == filename]
    start = next((i for i, f in enumerate(frames) if f.filename == filename), len(frames))
    frames = frames[start:] if keep else []
    text = "".join(traceback.format_list(frames)) if frames else ""
    return (("Traceback (most recent call last):\n" + text) if text else "") + \
        "".join(traceback.format_exception_only(type(exc), exc))


def _frame_step(tree, cell: Cell, pipeline_name: str):
    """The code of a ``[each frame]`` / ``[all frames]`` cell: its body as the step function
    ``f(frame, data)`` / ``f(frames, data)``, added to the pipeline (line numbers kept). The step
    is named after the cell's title; nothing else is left in the notebook's namespace."""
    each = cell.frames == "each"
    name = re.sub(r"\W+", "_", cell.title).strip("_").lower() or ("each_frame" if each else "all_frames") + "_cell"
    template = ast.parse(
        f"from ewald.pipeline import {'each_frame' if each else 'all_frames'} as _ewald_mark\n"
        f"from ewald.notebook import _named as _ewald_named\n"
        f"@_ewald_mark\ndef _ewald_step({'frame' if each else 'frames'}, data):\n    pass\n"
        f"{pipeline_name}.add(_ewald_named(_ewald_step, {name!r}))\n"
        f"del _ewald_step, _ewald_mark, _ewald_named\n")
    func = template.body[2]
    body = tree.body or [ast.Pass(lineno=1, col_offset=0, end_lineno=1, end_col_offset=0)]
    func.body = body
    first = body[0].lineno
    for node in template.body:
        for n in ast.walk(node) if node is not func else [func, *func.decorator_list, func.args]:
            if hasattr(n, "lineno"):   # the wrapper's lines point at the cell's first line
                n.lineno = n.end_lineno = first
    return ast.fix_missing_locations(template)


def _named(function, name: str):
    """``function`` called ``name`` (also in tracebacks)."""
    function.__name__ = function.__qualname__ = name
    try:
        function.__code__ = function.__code__.replace(co_name=name, **(
            {"co_qualname": name} if hasattr(function.__code__, "co_qualname") else {}))
    except (TypeError, ValueError):
        pass
    return function


def _figure_numbers() -> set:
    plt = sys.modules.get("matplotlib.pyplot")
    return set(plt.get_fignums()) if plt is not None else set()


class CellInterrupted(Exception):
    """Raised in a cell that :meth:`Notebook.interrupt` stopped."""


_EXEC_LOCK = threading.Lock()


@contextlib.contextmanager
def _interruptible(notebook):
    """While a cell's code runs, :meth:`Notebook.interrupt` may raise into this thread; an
    interrupt that arrives after the code finished is dropped, never raised in later code."""
    import ctypes

    thread = threading.get_ident()
    with _EXEC_LOCK:
        notebook.__dict__["_exec_thread"] = thread
    try:
        yield
    finally:
        with _EXEC_LOCK:
            notebook.__dict__["_exec_thread"] = None
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread), None)  # a pending one


def _patch_show():
    """plt.show() inside a cell only marks figures for display (inline, like Jupyter)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return lambda: None
    original = plt.show
    plt.show = lambda *a, **k: None

    def restore():
        plt.show = original
    return restore


def _new_figures(before: set, pipeline=None) -> list:
    """``(png, view)`` of each figure a cell made; ``view`` is the arguments of a ``show()`` of
    ``pipeline`` (the notebook's), None for any other figure."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    out = []
    for num in plt.get_fignums():
        if num in before:
            continue
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi="figure", bbox_inches="tight")
        plt.close(fig)
        source, args = getattr(fig, "_ewald_view", (None, None))
        out.append((buf.getvalue(), args if pipeline is not None and source is pipeline else None))
    return out


def _display(value) -> Output:
    import numpy as np

    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] in (3, 4):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(value).save(buf, format="PNG")
        return Output("image", buf.getvalue())
    if type(value).__name__ == "View":   # a view cell: the viewport shows it, the cell says what it set
        return Output("value", repr(value))
    summary = getattr(value, "summary", None)
    if callable(summary) and type(value).__name__ == "DataCollection":  # the summary says more here
        text = summary()
        return Output("value", text if len(text) < 4000 else text[:4000] + " …")
    png = getattr(value, "_repr_png_", None)
    if callable(png):
        try:
            data = png()
            if data:
                return Output("image", data)
        except Exception:  # noqa: BLE001 - fall back to text
            pass
    text = repr(value)
    return Output("value", text if len(text) < 4000 else text[:4000] + " …")
