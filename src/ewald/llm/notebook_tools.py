"""Notebook tools: the AI reads and edits the cells of the user's notebook.

Every :class:`Toolset` has them, on the notebook it works on: the GUI attaches the notebook of
the current tab, Python code any :class:`~ewald.notebook.Notebook`, and without one the toolset
uses a session notebook of its own::

    ts = Toolset(notebook=nb)
    ts.call("insert_cell", {"source": "pipe.add(CoordinationAnalysis(cutoff=3.0))"})

A cell the model wrote is arbitrary Python, so it runs only when the access allows it
(:attr:`NotebookAccess.can_run`; Preferences › AI › "Let the AI run cells" in the GUI). Then
``insert_cell`` and ``edit_cell`` run the code cell they wrote at once (``run=false`` only writes
it) and return its outputs, and ``run_cell`` runs any cell. Otherwise the cell waits, marked not
run or out of date, until the user runs it, and ``run_cell`` is not offered. Parameters of a step cell change through
``params``: the step's ``pipe.add(Name(...))`` call is rewritten with the new literal values and
run, so the pipeline changes only by running the cell, and only when the step's arguments are
all literals (no code of the model's runs). Cells are addressed by their 0-based ``index`` in
:func:`list_cells` or by their ``id``.
"""
from __future__ import annotations

import ast
import base64
import inspect
from typing import Callable

from ..notebook import Cell, Notebook, _remove, cell_from_source, split_header, step_call
from ..pipeline import PythonModifier
from .tools import _CODE_MODIFIERS, ToolResult, _jsonable, _obj

#: The tools :func:`register` adds.
CELL_TOOLS = ("list_cells", "read_cell", "insert_cell", "edit_cell", "delete_cell", "move_cell", "run_cell")

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


#: How the chat names what a cell-writing tool did to its cell ("Changed view in cell 6").
_WRITER_VERBS = {"load_structure": "Loaded in", "add_modifier": "Added step in", "remove_modifier": "Removed step in",
                 "set_view": "Changed view in", "add_visual": "Changed view in", "remove_visual": "Changed view in",
                 "add_overlay": "Changed view in", "remove_overlay": "Changed view in", "export_file": "Exported in",
                 "add_analysis": "Added analysis in"}


def touched_cells(name: str, result) -> list[dict]:
    """The cells a tool call wrote, ran or deleted, read from its result (a :class:`ToolResult` or its
    content): ``[{"id", "index", "verb"}]``, ``index`` 0-based as it was then, ``id`` None for a
    deleted cell. The AI chat shows them as links to the cells ("Edited cell 6")."""
    content = result.content if isinstance(result, ToolResult) else result
    if not isinstance(content, dict):
        return []

    def ref(summary, verb):
        if not isinstance(summary, dict) or not isinstance(summary.get("index"), int):
            return []
        return [{"id": summary.get("id") if isinstance(summary.get("id"), str) else None,
                 "index": summary["index"], "verb": verb}]

    ran = " and ran" if isinstance(content.get("run"), dict) else ""
    if "inserted" in content:
        return ref(content["inserted"], "Inserted" + ran)
    if "edited" in content:
        return ref(content["edited"], "Edited" + ran)
    if "deleted" in content:
        return [dict(r, id=None) for r in ref(content["deleted"], "Deleted")]
    if name == "run_cell":
        return ref(content, "Ran")
    if isinstance(content.get("cell_id"), str):
        return ref({"index": content.get("cell"), "id": content["cell_id"]}, _WRITER_VERBS.get(name, "Wrote"))
    return []


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

    execute, then_run = _runner(access)
    run_arg = {"type": "boolean", "description": "run the cell after writing it, when the user lets the AI "
                                                 "run cells (default true)"}

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
       "at that index). When the user lets the AI run cells (run_cell is offered), a code cell runs at "
       "once and its outputs come back under 'run'; otherwise it waits until the user runs it. Write one "
       "complete cell that runs in the notebook's namespace (pipe is the pipeline). For a plain analysis "
       "step prefer add_modifier, which adds the step cell and runs it right away.",
       _obj({"source": {"type": "string"},
             "kind": {"type": "string", "enum": ["code", "markdown"], "description": "default code"},
             "index": {"type": "integer", "description": "insert before this cell (default: append)"},
             "run": run_arg},
            ["source"]))
    def insert_cell(source, kind="code", index=None, run=True):
        if kind not in ("code", "markdown"):
            raise ValueError("kind is 'code' or 'markdown'")

        def op():
            nb = need()
            n = len(nb.cells)
            at = n if index is None else max(0, min(int(index) if int(index) >= 0 else n + int(index), n))
            cell = nb.insert(at, cell_from_source(source, kind=kind, state="new" if kind == "code" else "ok"))
            access.changed(nb, "AI: insert cell", cell)
            return {"inserted": cell_summary(nb, at), "cells": len(nb.cells)}, nb, cell

        return then_run(*access.call(op), run)

    @T("edit_cell",
       "Change a notebook cell. Either replace its whole 'source', or replace 'old_text' (which must occur "
       "exactly once) with 'new_text', or, for a step cell, set the step's 'params': its pipe.add(Name(...)) "
       "is rewritten with the new values and run at once (only when its arguments are literals). A changed "
       "source runs at once when the user lets the AI run cells (run_cell is offered; outputs under 'run'); "
       "otherwise the cell is marked out of date until the user runs it.",
       _obj({**where, "source": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
             "params": {"type": "object", "description": "step cells: modifier parameters to change"},
             "run": run_arg}))
    def edit_cell(index=None, id=None, source=None, old_text=None, new_text=None, params=None, run=True):
        if sum(x is not None for x in (source, old_text, params)) != 1:
            raise ValueError("pass exactly one of source, old_text (+ new_text) or params")
        if old_text is not None and new_text is None:
            raise ValueError("old_text needs new_text")

        if params is not None:
            return set_params(*access.call(lambda: located(index, id)), params)

        def op():
            nb = need()
            k = find(nb, index, id)
            c = nb.cells[k]
            frames = None
            if source is not None:
                new = source
                head = split_header(source) if c.kind == "code" else None
                if head is not None and head[0] == "code":   # a marker line sets the cell's tag and title
                    _, title, frames, new = head
                    if title:
                        c.title = title
            else:
                hits = c.source.count(old_text)
                if hits != 1:
                    raise ValueError(f"old_text occurs {hits} times in cell {k}; it must occur exactly once")
                new = c.source.replace(old_text, new_text)
            if new == c.source and (frames is None or frames == c.frames):
                return {"unchanged": cell_summary(nb, k)}, nb, None
            pipe, removed = nb.pipeline, bool(c.added)
            if pipe is not None:  # the old code's steps leave with it, as when the cell is cleared
                for m in c.added:
                    _remove(pipe.modifiers, m)
            c.added, c.shadow = [], None
            c.source = new
            if frames is not None:
                c.frames = frames
            if c.kind == "code":
                c.state = "stale" if c.state != "new" else "new"
            nb.dirty = True
            access.changed(nb, "AI: edit cell", c, pipeline=removed)
            out = {"edited": cell_summary(nb, k)}
            if removed:
                out["note"] = "the steps this cell added are out of the pipeline until the cell runs again"
            return out, nb, c

        return then_run(*access.call(op), run)

    def located(index, id):
        nb = need()
        k = find(nb, index, id)
        return nb, nb.cells[k], k

    def set_params(nb: Notebook, c: Cell, k: int, params: dict) -> dict:
        mod = c.modifier
        if mod is None:
            raise ValueError(f"cell {k} is not a step cell that ran; edit its source instead")
        if nb.restricted:
            raise PermissionError("the notebook is not trusted yet (an .ewd file); the user must trust it first")
        if isinstance(mod, PythonModifier) or type(mod).__name__ in _CODE_MODIFIERS:   # parameters are code
            raise PermissionError(f"{mod.name} runs Python; its parameters cannot be set from here")
        known = mod.parameters()
        bad = sorted(set(params) - set(known))
        if bad:
            raise ValueError(f"{mod.name} has no parameter(s) {bad}; it has {sorted(known)}")
        old_source = c.source
        new_source = _with_params(old_source, nb.pipeline_name, type(mod), params)
        type(mod)(**{**_call_values(old_source, nb.pipeline_name, type(mod)), **params})   # bad values fail here
        before = {p: known[p] for p in params}
        label = f"AI: edit {mod.name}"

        def write():
            c.source = new_source
            access.changed(nb, label, c)

        access.call(write)
        ok = access.run(nb, c)
        error = "\n".join(str(o.data) for o in c.outputs if o.kind == "error").strip()
        if ok and nb.pipeline is not None and nb.kernel is None:
            try:   # the new values must also compute
                nb.pipeline.compute(ts.frame % max(nb.pipeline.num_frames, 1))
            except Exception as exc:  # noqa: BLE001 - reported to the model below
                ok, error = False, f"{type(exc).__name__}: {exc}"
        if not ok:

            def restore():
                c.source = old_source
                access.changed(nb, f"{label} (failed)", c)

            access.call(restore)
            access.run(nb, c)   # the step as it was (literal arguments only), back in the pipeline
            raise RuntimeError(f"the step failed with the new parameters and was put back:\n{error[-3000:]}")
        return access.call(lambda: {"edited": cell_summary(nb, nb.cells.index(c)), "before": _jsonable(before),
                                    "source": c.source})

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

        content, images, ok = execute(*access.call(find_cell))
        return ToolResult(_jsonable(content), images=images, is_error=not ok)

    ts.tools["run_cell"].available = lambda: access.can_run


def _runner(access: NotebookAccess):
    """``(execute, then_run)``: run a cell and report its outputs; run the cell insert/edit_cell wrote."""
    def execute(nb: Notebook, cell: Cell):
        ok = access.run(nb, cell)

        def result():
            k = nb.cells.index(cell) if cell in nb.cells else None
            items, images = _outputs(cell)
            out = {"ok": bool(ok), "state": cell.state, "outputs": items}
            if k is not None:
                out["index"], out["id"] = k, cell.id
            pipe = nb.pipeline
            if pipe is not None:
                out["pipeline"] = [repr(m) for m in pipe.modifiers]
            return out, images

        content, images = access.call(result)
        return content, images, ok

    def then_run(out: dict, nb: Notebook, cell: Cell | None, run) -> dict | ToolResult:
        if cell is None or cell.kind != "code" or not run:
            return out
        if not access.can_run:
            out["note"] = (out.get("note", "") + "; " if "note" in out else "") + \
                "not run: the user has not let the AI run cells, so the user runs it"
            return out
        if nb.restricted:
            out["note"] = "not run: the notebook is not trusted yet (an .ewd file)"
            return out
        content, images, ok = execute(nb, cell)
        content.pop("index", None)
        content.pop("id", None)
        out["run"] = content
        return ToolResult(_jsonable(out), images=images, is_error=not ok)

    return execute, then_run


def _literal_call(source: str, name: str) -> ast.Call:
    """The ``Name(...)`` call of a step cell when every argument is a literal, else ValueError."""
    arg = step_call(source, name)
    if not isinstance(arg, ast.Call) or not isinstance(arg.func, (ast.Name, ast.Attribute)):
        raise ValueError("this step is not a plain pipe.add(Name(...)) call; edit its source instead")
    try:
        for node in list(arg.args) + [kw.value for kw in arg.keywords]:
            ast.literal_eval(node)
    except ValueError:
        raise ValueError("this step's arguments are code, not plain values; edit its source instead") from None
    if any(kw.arg is None for kw in arg.keywords) or any(isinstance(a, ast.Starred) for a in arg.args):
        raise ValueError("this step passes *args/**kwargs; edit its source instead")
    return arg


def _call_values(source: str, name: str, cls) -> dict:
    """The arguments of a step cell's literal call, by parameter name."""
    call = _literal_call(source, name)
    bound = inspect.signature(cls).bind_partial(*[ast.literal_eval(a) for a in call.args],
                                                **{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
    return dict(bound.arguments)


def _with_params(source: str, name: str, cls, params: dict) -> str:
    """``source`` with its step call's arguments replaced by the old ones updated with ``params``."""
    call = _literal_call(source, name)
    values = {**_call_values(source, name, cls), **params}
    text = f"{ast.get_source_segment(source, call.func)}({', '.join(f'{k}={v!r}' for k, v in values.items())})"
    lines = source.splitlines(True)
    start = sum(len(line) for line in lines[:call.lineno - 1]) + _offset(lines[call.lineno - 1], call.col_offset)
    end = sum(len(line) for line in lines[:call.end_lineno - 1]) + _offset(lines[call.end_lineno - 1],
                                                                          call.end_col_offset)
    return source[:start] + text + source[end:]


def _offset(line: str, col: int) -> int:
    """A character offset in ``line`` from ast's UTF-8 byte offset ``col``."""
    return len(line.encode()[:col].decode(errors="ignore"))
