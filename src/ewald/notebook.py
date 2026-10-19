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
import linecache
import os
import re
import sys
import traceback
from dataclasses import dataclass, field

from .pipeline import Modifier, Pipeline, PythonModifier

PIPELINE_NAME = "pipe"
_DATA_DIRS: contextvars.ContextVar = contextvars.ContextVar("ewald_notebook_data_dirs", default=())
_MARK = re.compile(r"^# %%(.*)$")


@dataclass
class Output:
    """One result shown under a cell: ``kind`` is 'text' (stdout), 'error', 'value'
    (repr of the last expression) or 'image' (PNG bytes: a matplotlib figure or a rendered image)."""

    kind: str
    data: object


@dataclass
class Cell:
    source: str = ""
    kind: str = "code"                 # 'code' or 'markdown'
    title: str = ""                    # text after '# %%' (besides [markdown])
    outputs: list = field(default_factory=list)
    added: list = field(default_factory=list)   # modifiers this cell's last run put into the pipeline
    count: int | None = None           # execution counter, like Jupyter's In [n]
    state: str = "new"                 # 'new', 'ok', 'error' or 'stale'
    id: str = field(default_factory=lambda: os.urandom(4).hex())   # nbformat cell id, kept in .ipynb
    collapsed: bool = False            # folded in the GUI; kept in .ipynb as Jupyter's source/outputs_hidden

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
        """The modifier of a step cell (after it ran or was added by the GUI)."""
        return self.added[0] if self.is_step and len(self.added) == 1 else None


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


def step_call(source: str, name: str = PIPELINE_NAME) -> ast.expr | None:
    """The argument of ``<name>.add(arg)`` or ``<name>.modifiers.append(arg)`` if ``source`` is only that call,
    optionally preceded by import statements; None otherwise."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    *head, last = tree.body or [None]
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
        self.restricted = False       # an .ewd that was not trusted yet: cells do not run
        self.manifest: dict = {}      # an .ewd file's manifest
        self.warnings: list = []      # data files that moved or changed (from the manifest)
        self.reset()

    # ------------------------------------------------------------ namespace
    def reset(self):
        """Fresh namespace (like restarting a Jupyter kernel); cells keep their source."""
        self.namespace = {"__name__": "__main__", "__file__": self.path or "<notebook>"}
        for c in self.cells:
            c.added = []
            if c.kind == "code" and c.state != "new":
                c.state = "stale"

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

    def _slot(self, index: int, pipe: Pipeline) -> int:
        """Where a cell's new modifiers go: after those of the cells above it."""
        above = [m for c in self.cells[:index] for m in c.added]
        positions = [i for i, m in enumerate(pipe.modifiers) if any(m is a for a in above)]
        return max(positions) + 1 if positions else 0 if any(c.added for c in self.cells[index + 1:]) else len(pipe.modifiers)

    def trust(self):
        """Allow a restricted notebook (an ``.ewd`` someone sent) to run: its bundled plugins
        are loaded and its cells may run."""
        if self.restricted and self.bundle_dir:
            from .notebook_io import load_bundled_plugins

            load_bundled_plugins(self)
        self.restricted = False

    def _exec(self, cell: Cell, filename: str) -> bool:
        if self.restricted:
            raise PermissionError(f"{self.name} came from an .ewd file and is not trusted yet: "
                                  "call notebook.trust() (Run all in the GUI asks first)")
        token = _DATA_DIRS.set(tuple(self._data_dirs()))
        try:
            return self._exec_cell(cell, filename)
        finally:
            _DATA_DIRS.reset(token)

    def _data_dirs(self) -> list:
        dirs = [os.path.dirname(self.path)] if self.path else []
        return dirs + [d for d in self.data_dirs if d not in dirs]

    def _exec_cell(self, cell: Cell, filename: str) -> bool:
        cell.outputs = []
        lines = cell.source.splitlines(True)
        linecache.cache[filename] = (len(cell.source), None, lines, filename)
        out = io.StringIO()
        figures_before = _figure_numbers()
        restore_show = _patch_show()
        try:
            tree = ast.parse(cell.source, filename)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
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
        cell.outputs.extend(Output("image", png) for png in _new_figures(figures_before))
        if ok and value is not None:
            cell.outputs.append(_display(value))
        return ok

    # ------------------------------------------------------------ editing
    def insert(self, index: int, cell: Cell) -> Cell:
        self.cells.insert(index, cell)
        self.dirty = True
        return cell

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
        """The notebook's content as text: (kind, title, source) per cell. Everything else
        (pipeline, outputs) follows from it by running the cells."""
        return tuple((c.kind, c.title, c.source) for c in self.cells)

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
            for kind, title, source in snap[j1:j2]:
                cell = Cell(source, kind=kind, title=title, state="new" if kind == "code" else "ok")
                cells.append(cell)
                fresh.append(cell)
        self.cells = cells
        for cell in fresh:
            if cell.kind != "code":
                continue
            if cell.is_step or cell.is_view:
                self.run_cell(cell, run_below=False)
            else:
                cell.state = "stale"
        self._reorder()
        self.dirty = True
        return fresh

    # ------------------------------------------------------------ files
    def to_py(self) -> str:
        """The notebook as a percent-format Python file."""
        parts = []
        for cell in self.cells:
            head = "# %%" + (f" {cell.title}" if cell.title else "")
            if cell.kind == "markdown":
                head += " [markdown]"
                body = "\n".join(("# " + line) if line else "#" for line in cell.source.rstrip("\n").splitlines())
            else:
                body = cell.source.rstrip("\n")
            parts.append(head + ("\n" + body if body else ""))
        return "\n\n".join(parts) + "\n"

    @classmethod
    def from_py(cls, text: str, **kwargs) -> "Notebook":
        """Parse a percent-format file; text before the first marker becomes a code cell
        (a plain script is one big cell)."""
        cells, current = [], None
        lines = []

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
                title = re.sub(r"\[(markdown|md)\]", "", rest).strip()
                current, lines = Cell(kind=kind, title=title), []
            else:
                lines.append(line)
        close()
        return cls(cells, **kwargs)

    @classmethod
    def load(cls, path, **kwargs) -> "Notebook":
        """Read a ``.py`` (percent format), ``.ipynb`` or ``.ewd`` notebook. An ``.ewd`` opens
        restricted (see :func:`ewald.notebook_io.load_ewd`); call :meth:`trust` to run it."""
        from . import notebook_io

        kwargs.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".ewd":
            return notebook_io.load_ewd(path, cls=cls, **kwargs)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if ext == ".ipynb":
            return notebook_io.from_ipynb(text, cls=cls, path=os.path.abspath(path), **kwargs)
        return cls.from_py(text, path=os.path.abspath(path), **kwargs)

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
            text = notebook_io.to_ipynb_text(self) if ext == ".ipynb" else self.to_py()
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        self.path = os.path.abspath(path)
        self.namespace["__file__"] = self.path
        self.dirty = False

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, source_code: str | None = None, **kwargs) -> "Notebook":
        """A notebook for a pipeline that already exists (a file opened from the menu):
        an import cell, a load cell (``source_code``, e.g. ``pipe = import_file(...)``) and one
        step per modifier. The pipeline object itself is used, nothing is read again."""
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
        nb.dirty = False
        return nb


IMPORTS = ("from ewald import import_file, load\nfrom ewald.modifiers import *\nfrom ewald.views import show\n"
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


def _figure_numbers() -> set:
    plt = sys.modules.get("matplotlib.pyplot")
    return set(plt.get_fignums()) if plt is not None else set()


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


def _new_figures(before: set) -> list:
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    out = []
    for num in plt.get_fignums():
        if num in before:
            continue
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        out.append(buf.getvalue())
    return out


def _display(value) -> Output:
    import numpy as np

    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] in (3, 4):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(value).save(buf, format="PNG")
        return Output("image", buf.getvalue())
    png = getattr(value, "_repr_png_", None)
    if callable(png):
        try:
            data = png()
            if data:
                return Output("image", data)
        except Exception:  # noqa: BLE001 - fall back to text
            pass
    summary = getattr(value, "summary", None)
    if callable(summary) and type(value).__name__ == "DataCollection":
        text = summary()
    else:
        text = repr(value)
    return Output("value", text if len(text) < 4000 else text[:4000] + " …")
