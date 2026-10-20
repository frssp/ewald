"""Notebook tools: the AI reads and edits the cells of the user's notebook.

:meth:`Toolset.attach_notebook` adds them; the GUI attaches the notebook of the current tab,
and Python code can attach any :class:`~ewald.notebook.Notebook`::

    ts = Toolset(nb.pipeline, notebook=nb)
    ts.call("insert_cell", {"source": "pipe.add(CoordinationAnalysis(cutoff=3.0))"})

Inserting or editing a cell never runs it: the cell waits, marked not run or out of date,
until the user runs it. ``run_cell`` exists only when the access allows it
(:attr:`NotebookAccess.can_run`; Preferences › AI › "Let the AI run cells" in the GUI),
since a cell is arbitrary Python. Parameters of a step cell change through ``params``,
like the step's form, without running code. Cells are addressed by their 0-based ``index``
in :func:`list_cells` or by their ``id``.
"""
from __future__ import annotations

import base64
from typing import Callable

from ..notebook import Cell, Notebook, _remove
from ..pipeline import PythonModifier
from .tools import _CODE_MODIFIERS, ToolResult, _jsonable, _obj

_MAX_SOURCE = 12000   # characters of one cell's source a tool result carries
_MAX_OUTPUT = 4000    # characters of one text output
_MAX_IMAGES = 3


class NotebookAccess:
    """What the notebook tools act on. ``notebook`` is a :class:`Notebook` or a function
    returning the current one (or None); ``can_run`` says whether ``run_cell`` is offered.

    The GUI subclasses it: :meth:`call` runs each operation on the GUI thread, :meth:`changed`
    redraws the cells and records an undo step, :meth:`run` runs a cell the way Shift+Enter does.
    """

    def __init__(self, notebook: Notebook | Callable[[], Notebook | None] | None = None,
                 can_run: bool | Callable[[], bool] = False):
        self._notebook = notebook
        self._can_run = can_run

    def notebook(self) -> Notebook | None:
        nb = self._notebook
        return nb() if callable(nb) and not isinstance(nb, Notebook) else nb

    @property
    def can_run(self) -> bool:
        return bool(self._can_run() if callable(self._can_run) else self._can_run)

    def call(self, fn: Callable[[], object]):
        """Run ``fn`` where the notebook may be touched (here: right away)."""
        return fn()

    def changed(self, nb: Notebook, label: str, cell: Cell | None = None, pipeline: bool = False):
        """The cells changed (``label`` names the edit for undo); ``pipeline``: its modifiers too."""

    def run(self, nb: Notebook, cell: Cell) -> bool:
        """Run ``cell`` (and, like the notebook, the cells below when it rebinds ``pipe``)."""
        return nb.run_cell(cell)


def cell_summary(nb: Notebook, index: int) -> dict:
    """One line about a cell for list_cells."""
    c = nb.cells[index]
    first = next((line for line in c.source.splitlines() if line.strip()), "")
    kind = "markdown" if c.kind == "markdown" else "step" if c.is_step else "view" if c.is_view else "code"
    out = {"index": index, "id": c.id, "kind": kind, "state": c.state, "lines": len(c.source.splitlines()),
           "first_line": first[:160]}
    if c.title:
        out["title"] = c.title
    if c.count:
        out["run"] = c.count
    if c.outputs:
        out["outputs"] = [o.kind for o in c.outputs]
    mod = c.modifier
    if mod is not None:
        out["modifier"] = mod.name
    return out


def failed_cells(nb: Notebook, max_lines: int = 25, max_chars: int = 6000) -> str:
    """The cells whose last run failed, with the end of each error (what the AI chat sends along)."""
    parts = []
    for k, c in enumerate(nb.cells):
        if c.kind != "code" or c.state != "error":
            continue
        error = "\n".join(str(o.data) for o in c.outputs if o.kind == "error").strip()
        lines = error.splitlines()
        if len(lines) > max_lines:
            lines = ["…"] + lines[-max_lines:]
        first = next((line for line in c.source.splitlines() if line.strip()), "")
        parts.append(f"cell index {k} ({first[:80]}):\n" + "\n".join(lines))
    text = "\n\n".join(parts)
    return text if len(text) <= max_chars else "…\n" + text[-max_chars:]


def _outputs(cell: Cell) -> tuple[list, list[bytes]]:
    items, images = [], []
    for o in cell.outputs:
        if o.kind == "image":
            png = o.data if isinstance(o.data, bytes) else base64.b64decode(o.data)
            if len(images) < _MAX_IMAGES:
                images.append(png)
            items.append({"kind": "image", "shown": len(images) <= _MAX_IMAGES})
        else:
            text = str(o.data)
            if len(text) > _MAX_OUTPUT:
                text = text[:_MAX_OUTPUT // 2] + "\n… (truncated) …\n" + text[-_MAX_OUTPUT // 2:]
            items.append({"kind": o.kind, "text": text})
    return items, images


def register(ts, access: NotebookAccess):
    """Add the notebook tools to Toolset ``ts``."""
    T = ts.tool
    where = {"index": {"type": "integer", "description": "0-based cell index from list_cells"},
             "id": {"type": "string", "description": "cell id from list_cells (instead of index)"}}

    def need() -> Notebook:
        nb = access.notebook()
        if nb is None:
            raise RuntimeError("no notebook is open")
        return nb

    def find(nb: Notebook, index=None, id=None) -> int:
        if id is not None:
            for k, c in enumerate(nb.cells):
                if c.id == id:
                    return k
            raise ValueError(f"no cell with id {id!r}")
        if index is None:
            raise ValueError("pass index or id")
        n = len(nb.cells)
        if not -n <= int(index) < n:
            raise IndexError(f"cell index {index} out of range: the notebook has {n} cells")
        return int(index) % n

    def listing(nb: Notebook) -> dict:
        out = {"notebook": nb.name, "cells": [cell_summary(nb, k) for k in range(len(nb.cells))]}
        if nb.path:
            out["path"] = nb.path
        if nb.restricted:
            out["restricted"] = "an .ewd file that is not trusted yet: its cells do not run"
        return out

    @T("list_cells",
       "The cells of the user's notebook: index, id, kind (code, step = one pipe.add(...), view = one "
       "show(pipe, ...), markdown), state (new = not run, ok, error, stale = out of date), first line "
       "and output kinds. Read a cell's full source and outputs with read_cell.", _obj({}))
    def list_cells():
        return access.call(lambda: listing(need()))

    @T("read_cell",
       "One notebook cell: its full source and outputs (printed text, errors, values; plots come back "
       "as images).", _obj(where))
    def read_cell(index=None, id=None):
        def op():
            nb = need()
            k = find(nb, index, id)
            c = nb.cells[k]
            out = cell_summary(nb, k)
            out.pop("first_line")
            src = c.source
            out["source"] = src if len(src) <= _MAX_SOURCE else src[:_MAX_SOURCE] + "\n# … (truncated)\n"
            items, images = _outputs(c)
            out["outputs"] = items
            mod = c.modifier
            if mod is not None:
                out["params"] = {k: v for k, v in mod.parameters().items() if k not in ("source",)}
            return out, images

        content, images = access.call(op)
        return ToolResult(_jsonable(content), images=images)

    @T("insert_cell",
       "Insert a new cell into the user's notebook (default: at the end; 'index' puts it before the cell "
       "at that index). The cell is not run: the user runs it (or run_cell, when offered). Write one "
       "complete cell that runs in the notebook's namespace (pipe is the pipeline). For a plain analysis "
       "step prefer add_modifier, which adds the step cell and computes it right away.",
       _obj({"source": {"type": "string"},
             "kind": {"type": "string", "enum": ["code", "markdown"], "description": "default code"},
             "index": {"type": "integer", "description": "insert before this cell (default: append)"}},
            ["source"]))
    def insert_cell(source, kind="code", index=None):
        if kind not in ("code", "markdown"):
            raise ValueError("kind is 'code' or 'markdown'")

        def op():
            nb = need()
            n = len(nb.cells)
            at = n if index is None else max(0, min(int(index) if int(index) >= 0 else n + int(index), n))
            cell = nb.insert(at, Cell(source, kind=kind, state="new" if kind == "code" else "ok"))
            access.changed(nb, "AI: insert cell", cell)
            return {"inserted": cell_summary(nb, at), "cells": len(nb.cells)}

        return access.call(op)

    @T("edit_cell",
       "Change a notebook cell. Either replace its whole 'source', or replace 'old_text' (which must occur "
       "exactly once) with 'new_text', or, for a step cell, set the step's 'params' (like its form: the "
       "pipeline follows at once, no code runs). A changed source is not run: the cell is marked out of "
       "date until the user runs it (or run_cell, when offered).",
       _obj({**where, "source": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
             "params": {"type": "object", "description": "step cells: modifier parameters to change"}}))
    def edit_cell(index=None, id=None, source=None, old_text=None, new_text=None, params=None):
        if sum(x is not None for x in (source, old_text, params)) != 1:
            raise ValueError("pass exactly one of source, old_text (+ new_text) or params")
        if old_text is not None and new_text is None:
            raise ValueError("old_text needs new_text")

        def op():
            nb = need()
            k = find(nb, index, id)
            c = nb.cells[k]
            if params is not None:
                return set_params(nb, c, k, params)
            if source is not None:
                new = source
            else:
                hits = c.source.count(old_text)
                if hits != 1:
                    raise ValueError(f"old_text occurs {hits} times in cell {k}; it must occur exactly once")
                new = c.source.replace(old_text, new_text)
            if new == c.source:
                return {"unchanged": cell_summary(nb, k)}
            pipe, removed = nb.pipeline, bool(c.added)
            if pipe is not None:  # the old code's steps leave with it, as when the cell is cleared
                for m in c.added:
                    _remove(pipe.modifiers, m)
            c.added, c.shadow = [], None
            c.source = new
            if c.kind == "code":
                c.state = "stale" if c.state != "new" else "new"
            nb.dirty = True
            access.changed(nb, "AI: edit cell", c, pipeline=removed)
            out = {"edited": cell_summary(nb, k)}
            if removed:
                out["note"] = "the steps this cell added are out of the pipeline until the cell runs again"
            return out

        return access.call(op)

    def set_params(nb: Notebook, c: Cell, k: int, params: dict) -> dict:
        mod = c.modifier
        if mod is None:
            raise ValueError(f"cell {k} is not a step cell that ran; edit its source instead")
        if nb.kernel is not None:
            raise ValueError("this notebook runs in a kernel; edit the step's source instead")
        if isinstance(mod, PythonModifier) or type(mod).__name__ in _CODE_MODIFIERS:   # parameters are code
            raise PermissionError(f"{mod.name} runs Python; its parameters cannot be set from here")
        known = mod.parameters()
        bad = sorted(set(params) - set(known))
        if bad:
            raise ValueError(f"{mod.name} has no parameter(s) {bad}; it has {sorted(known)}")
        old = {p: known[p] for p in params}
        for p, v in params.items():
            setattr(mod, p, v)
        nb.update_step(c)
        access.changed(nb, f"AI: edit {mod.name}", c, pipeline=True)
        return {"edited": cell_summary(nb, k), "before": _jsonable(old), "source": c.source}

    @T("delete_cell", "Delete a notebook cell (and the steps it added to the pipeline).", _obj(where))
    def delete_cell(index=None, id=None):
        def op():
            nb = need()
            k = find(nb, index, id)
            info = cell_summary(nb, k)
            had = bool(nb.cells[k].added)
            nb.delete(k)
            access.changed(nb, "AI: delete cell", None, pipeline=had)
            return {"deleted": info, "cells": len(nb.cells)}

        return access.call(op)

    @T("move_cell", "Move a notebook cell to position 'to' (0-based); the pipeline follows the new order.",
       _obj({**where, "to": {"type": "integer"}}, ["to"]))
    def move_cell(to, index=None, id=None):
        def op():
            nb = need()
            k = find(nb, index, id)
            cell = nb.cells[k]
            nb.move(k, int(to))
            access.changed(nb, "AI: move cell", cell, pipeline=bool(cell.added))
            return listing(nb)

        return access.call(op)

    @T("run_cell",
       "Run a notebook cell, like Shift+Enter, and return its outputs. Offered only when the user lets "
       "the AI run cells.", _obj(where))
    def run_cell(index=None, id=None):
        def find_cell():
            nb = need()
            if nb.restricted:
                raise PermissionError("the notebook is not trusted yet (an .ewd file); the user must trust it first")
            return nb, nb.cells[find(nb, index, id)]

        nb, cell = access.call(find_cell)
        ok = access.run(nb, cell)

        def result():
            k = nb.cells.index(cell) if cell in nb.cells else None
            items, images = _outputs(cell)
            out = {"ok": bool(ok), "state": cell.state, "outputs": items}
            if k is not None:
                out["index"] = k
            pipe = nb.pipeline
            if pipe is not None:
                out["pipeline"] = [repr(m) for m in pipe.modifiers]
            return out, images

        content, images = access.call(result)
        return ToolResult(_jsonable(content), images=images, is_error=not ok)

    ts.tools["run_cell"].available = lambda: access.can_run
