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
        out = {"kind": o.kind, "png": base64.b64encode(data).decode("ascii")}
        if getattr(o, "view", None) is not None:
            out["view"] = o.view
        return out
    return {"kind": o.kind, "text": str(data)}


def _step(m) -> dict:
    return {"id": id(m), "name": getattr(m, "name", type(m).__name__), "class": type(m).__name__,
            "enabled": bool(getattr(m, "enabled", True)), "status": getattr(m, "status", None),
            "time": getattr(m, "last_time", None), "output": _jsonable(getattr(m, "last_output", None))}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _plain_view(view) -> dict:
    """The settings of ``view`` that survive JSON (a visual element object stays in the kernel)."""
    out = {}
    for k, v in view.changed().items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def run(cells: list, index: int, run_below: bool, counter: int, result_path: str, pipeline_name: str = "pipe",
        path: str | None = None, data_dirs: tuple = (), plot_style=None):
    """Run cell ``index`` of ``cells`` and write the result JSON to ``result_path``. ``path`` and
    ``data_dirs`` are where ``load()`` looks for data files (the notebook's folder, an .ewd's data);
    ``plot_style`` is ``(style, tokens)`` for the cells' own matplotlib plots."""
    nb = notebook()
    nb.pipeline_name, nb.path, nb.data_dirs = pipeline_name, path, list(data_dirs)
    if plot_style:
        nb.set_plot_style(*plot_style)
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
                                               "modifiers": [_step(m) for m in pipe.modifiers],
                                               "view": _plain_view(pipe.view),
                                               "view_version": [id(pipe.view), pipe.view.version]},
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def reset():
    """Forget the cells' steps (the kernel restarted or Run all starts over)."""
    global _notebook
    _notebook = None


# ---------------------------------------------------------------- frames
def frame(pipe, frame: int, path: str):
    """Compute ``frame`` like the GUI does (a failing step is skipped and keeps its message) and
    write it with the steps' results (status, time, output) to ``path``."""
    save_frame(pipe.compute(frame, errors="skip"), path, steps=[_step(m) for m in pipe.modifiers])


def set_enabled(pipe, step_id: int, on: bool):
    """Switch one step (by the id the results carry) on or off."""
    for m in pipe.modifiers:
        if id(m) == step_id:
            m.enabled = bool(on)


def save_frame(data, path: str, steps: list | None = None):
    """Write a DataCollection's particles, cell, tables, bonds, simple attributes and the attribute
    objects in :data:`OBJECTS` to ``.npz``."""
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
    meta = {"surfaces": [], "grids": [], "vectors": {}, "objects": []}
    for i, (name, m) in enumerate(data.surfaces.items()):
        meta["surfaces"].append({"name": name, "transparency": m.transparency, "edge_width": m.edge_width,
                                 "edge_color": [float(c) for c in m.edge_color] if not isinstance(m.edge_color, str)
                                 else m.edge_color})
        arrays[f"s{i}/vertices"], arrays[f"s{i}/faces"] = m.vertices, m.faces
        if m.edges is not None:
            arrays[f"s{i}/edges"] = m.edges
        for prop, a in m.face_properties.items():
            arrays[f"s{i}/f/{prop}"] = np.asarray(a)
        for prop, a in m.vertex_properties.items():
            arrays[f"s{i}/v/{prop}"] = np.asarray(a)
    for i, (name, g) in enumerate(data.grids.items()):
        meta["grids"].append(name)
        arrays[f"g{i}/cell"], arrays[f"g{i}/origin"] = g.domain.matrix, g.domain.origin
        arrays[f"g{i}/pbc"] = np.asarray(g.domain.pbc)
        for prop, a in g.properties.items():
            arrays[f"g{i}/p/{prop}"] = np.asarray(a)
    for prop, style in data.vectors.items():
        if hasattr(style, "to_dict") and _plain(style.to_dict()):
            meta["vectors"][prop] = style.to_dict()
    attrs = {}
    for k, v in data.attributes.items():
        if _plain(v):
            attrs[k] = v
        else:
            _save_object(k, v, arrays, meta)
    arrays["meta"] = np.asarray(json.dumps(meta))
    arrays["attributes"] = np.asarray(json.dumps(attrs))
    if steps is not None:
        arrays["steps"] = np.asarray(json.dumps(steps))
    np.savez(path, **arrays)


# Attribute objects that travel with a frame (the Plots tab draws DOS/bands and phonon bands from
# them, so without them a kernel's frame plots differently): class name -> module. Only these are
# rebuilt on load; nothing is imported by a name that came from the file.
OBJECTS = {"ElectronicStructure": "ewald.electronic", "PhononModes": "ewald.phonon"}


def _save_object(name: str, value, arrays: dict, meta: dict):
    """Store a dataclass attribute of a class in :data:`OBJECTS`: arrays as ``x<i>/<field>``,
    the rest as JSON. Lazy loaders are resolved first (ElectronicStructure projections)."""
    import dataclasses

    cls = type(value).__name__
    if cls not in OBJECTS or not dataclasses.is_dataclass(value):
        return
    i, fields = len(meta["objects"]), {}
    for f in dataclasses.fields(value):
        v = getattr(value, f.name)
        if f.name == "_projections" and v is None:
            v = getattr(value, "projections", None)
        if callable(v) or v is None:
            continue
        if isinstance(v, np.ndarray):
            arrays[f"x{i}/{f.name}"] = v.astype(str) if v.dtype == object else v
        elif _plain(v):
            fields[f.name] = v
    meta["objects"].append({"attribute": name, "class": cls, "fields": fields})


def _load_objects(d, z, meta: dict):
    import dataclasses
    import importlib

    for i, info in enumerate(meta.get("objects", [])):
        module = OBJECTS.get(info.get("class"))
        try:
            cls = getattr(importlib.import_module(module), info["class"]) if module else None
        except ImportError:   # e.g. PhononModes without the ewald-phonopy plugin here
            cls = None
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        values = dict(info.get("fields", {}))
        pre = f"x{i}/"
        values.update({key[len(pre):]: z[key] for key in z.files if key.startswith(pre)})
        names = {f.name for f in dataclasses.fields(cls) if f.init}
        try:
            obj = cls(**{k: v for k, v in values.items() if k in names})
        except TypeError:
            continue
        for k, v in values.items():
            if k not in names:
                setattr(obj, k, v)
        d.attributes[info["attribute"]] = obj


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
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        _load_extras(d, z, meta)
    return d


def _load_extras(d, z, meta: dict):
    """Surfaces, voxel grids and vector styles of a frame written by :func:`save_frame`."""
    from .arrows import VectorStyle
    from .data import Cell, VoxelGrid
    from .mesh import SurfaceMesh

    files = z.files
    for i, info in enumerate(meta.get("surfaces", [])):
        pre = f"s{i}/"
        m = SurfaceMesh(z[pre + "vertices"], z[pre + "faces"], edges=z[pre + "edges"] if pre + "edges" in files else None,
                        transparency=info.get("transparency", 0.0), edge_width=info.get("edge_width", 0.04),
                        edge_color=tuple(info["edge_color"]) if isinstance(info.get("edge_color"), list)
                        else info.get("edge_color", (0.25, 0.25, 0.25)))
        for key in files:
            if key.startswith(pre + "f/"):
                m.face_properties[key[len(pre) + 2:]] = z[key]
            elif key.startswith(pre + "v/"):
                m.vertex_properties[key[len(pre) + 2:]] = z[key]
        d.surfaces[info["name"]] = m
    for i, name in enumerate(meta.get("grids", [])):
        pre = f"g{i}/"
        g = VoxelGrid(Cell(z[pre + "cell"], z[pre + "origin"], z[pre + "pbc"]))
        for key in files:
            if key.startswith(pre + "p/"):
                g[key[len(pre) + 2:]] = z[key]
        d.grids[name] = g
    _load_objects(d, z, meta)
    for prop, style in meta.get("vectors", {}).items():
        try:
            d.vectors[prop] = VectorStyle(**style)
        except TypeError:
            pass
