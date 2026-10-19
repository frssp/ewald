"""The kernel side of a notebook that runs in another Python (:mod:`ewald.kernel`).

The GUI sends the notebook's cells; this module keeps a :class:`~ewald.notebook.Notebook`
whose namespace is the kernel's own (IPython ``user_ns``), runs the requested cell with the
same rules as a local notebook (re-running a cell replaces the steps it added, a new ``pipe``
re-runs the cells below) and writes the outputs, step summaries and the pipeline's frame count
as JSON. Frames go back as ``.npz`` files (:func:`save_frame`, :func:`load_frame`).

Only ewald itself is needed in the kernel's Python, not its GUI dependencies.
"""
from __future__ import annotations

import base64
import json

import numpy as np

_notebook = None


def notebook():
    """The kernel's notebook, created on first use with the kernel's namespace."""
    global _notebook
    if _notebook is None:
        from .notebook import Notebook

        _notebook = Notebook(name="kernel")
        try:
            ip = get_ipython()  # noqa: F821 - defined inside an IPython kernel
            ip.user_ns.update(_notebook.namespace)
            _notebook.namespace = ip.user_ns
        except NameError:
            pass
    return _notebook


def _sync(nb, cells: list):
    """Make the kernel notebook's cells match the GUI's ``cells`` ({id, kind, source}), keeping
    what each unchanged cell added to the pipeline."""
    from .notebook import Cell

    old = {c.id: c for c in nb.cells}
    out = []
    for spec in cells:
        cell = old.get(spec["id"]) or Cell(id=spec["id"])
        cell.kind, cell.source = spec.get("kind", "code"), spec.get("source", "")
        out.append(cell)
    pipe = nb.pipeline
    for gone in set(old) - {c["id"] for c in cells}:  # deleted cells take their steps along
        if pipe is not None:
            for m in old[gone].added:
                if any(m is p for p in pipe.modifiers):
                    pipe.modifiers.remove(m)
    nb.cells = out


def _output(o) -> dict:
    data = o.data
    if isinstance(data, bytes):
        return {"kind": o.kind, "png": base64.b64encode(data).decode("ascii")}
    return {"kind": o.kind, "text": str(data)}


def _step(m) -> dict:
    return {"name": getattr(m, "name", type(m).__name__), "class": type(m).__name__,
            "enabled": bool(getattr(m, "enabled", True)), "status": getattr(m, "status", None),
            "time": getattr(m, "last_time", None), "output": _jsonable(getattr(m, "last_output", None))}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def run(cells: list, index: int, run_below: bool, counter: int, result_path: str, pipeline_name: str = "pipe",
        path: str | None = None, data_dirs: tuple = ()):
    """Run cell ``index`` of ``cells`` and write the result JSON to ``result_path``. ``path`` and
    ``data_dirs`` are where ``load()`` looks for data files (the notebook's folder, an .ewd's data)."""
    nb = notebook()
    nb.pipeline_name, nb.path, nb.data_dirs = pipeline_name, path, list(data_dirs)
    _sync(nb, cells)
    nb.counter = counter
    ok = nb.run_cell(index, run_below=run_below)
    ran = [c for c in nb.cells if c.count is not None and c.count > counter]   # this cell and any re-run below
    pipe = nb.pipeline
    result = {
        "ok": ok, "counter": nb.counter,
        "cells": [{"id": c.id, "state": c.state, "count": c.count, "outputs": [_output(o) for o in c.outputs],
                   "steps": [_step(m) for m in c.added]} for c in ran],
        "states": {c.id: c.state for c in nb.cells},
        "pipeline": None if pipe is None else {"num_frames": pipe.num_frames,
                                               "modifiers": [_step(m) for m in pipe.modifiers]},
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def reset():
    """Forget the cells' steps (the kernel restarted or Run all starts over)."""
    global _notebook
    _notebook = None


# ---------------------------------------------------------------- frames
def save_frame(data, path: str):
    """Write a DataCollection's particles, cell, tables, bonds and simple attributes to ``.npz``."""
    arrays = {"cell": data.cell.matrix, "origin": data.cell.origin, "pbc": np.asarray(data.cell.pbc)}
    for name, a in data.particles.items():
        a = np.asarray(a)
        arrays[f"p/{name}"] = a.astype(str) if a.dtype == object else a
    for table, columns in data.tables.items():
        for col, a in columns.items():
            arrays[f"t/{table}/{col}"] = np.asarray(a)
    if data.bonds is not None:
        for name, a in data.bonds.properties.items():
            arrays[f"b/{name}"] = np.asarray(a)
    attrs = {k: v for k, v in data.attributes.items() if _plain(v)}
    arrays["attributes"] = np.asarray(json.dumps(attrs))
    np.savez(path, **arrays)


def _plain(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def load_frame(path: str):
    """Read a frame written by :func:`save_frame`."""
    from .bonds import Bonds
    from .data import Cell, DataCollection

    with np.load(path, allow_pickle=False) as z:
        d = DataCollection(cell=Cell(z["cell"], z["origin"], z["pbc"]))
        bonds = {}
        for key in z.files:
            kind, _, rest = key.partition("/")
            if kind == "p":
                d.particles[rest] = z[key]
            elif kind == "t":
                table, _, col = rest.rpartition("/")
                d.tables.setdefault(table, {})[col] = z[key]
            elif kind == "b":
                bonds[rest] = z[key]
        if bonds:
            b = Bonds(bonds.pop("Topology"), bonds.pop("Periodic Image", None))
            b.properties.update(bonds)
            d.bonds = b
        d.attributes.update(json.loads(str(z["attributes"])))
    return d
