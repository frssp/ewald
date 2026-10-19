"""The ewald pipeline API as a set of LLM tools.

A :class:`Toolset` is a small stateful session: the loaded pipeline, its
modifier list, rendered figures and a provenance log. Each tool takes a JSON
object and returns a JSON-ready result (plus optional PNG images), so the same
toolset backs the Python :class:`~ewald.llm.Agent`, the MCP server and, later,
the GUI chat panel.

Numbers are always computed by tools; the model only chooses what to compute
and interprets it. Arrays are summarized, never dumped.

>>> ts = Toolset()
>>> ts.call("load_structure", {"path": "XDATCAR"})
>>> ts.call("add_modifier", {"name": "CoordinationAnalysis", "params": {"cutoff": 2.8}})
>>> print(ts.to_script())
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import modifiers as _modifiers
from ..pipeline import Modifier, Pipeline, PythonModifier
from . import describe as _describe


@dataclass
class ToolResult:
    """What a tool returns: JSON-ready ``content`` and optional PNG ``images``."""

    content: object
    images: list[bytes] = field(default_factory=list)
    is_error: bool = False

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else json.dumps(_jsonable(self.content))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    function: Callable

    def definition(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _obj(properties: dict, required=()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


def _downsample(table: dict, max_rows: int = 60) -> dict:
    n = len(next(iter(table.values())))
    idx = np.unique(np.linspace(0, n - 1, min(n, max_rows)).astype(int))
    return {k: [float(f"{v:.4g}") for v in np.asarray(col)[idx]] for k, col in table.items()}


#: Modifiers that run code or take Python objects: only PythonScript, and only with allow_code.
_CODE_MODIFIERS = {"PythonScript"}
_OBJECT_MODIFIERS = {"OvitoModifier"}


def _modifier_classes(allow_code: bool = False) -> dict[str, type]:
    """Every Modifier subclass exported by ``ewald.modifiers`` (new ones appear automatically)."""
    out = {}
    for name in getattr(_modifiers, "__all__", dir(_modifiers)):
        obj = getattr(_modifiers, name, None)
        if name in _OBJECT_MODIFIERS or (name in _CODE_MODIFIERS and not allow_code):
            continue
        if isinstance(obj, type) and issubclass(obj, Modifier) and obj not in (Modifier, PythonModifier):
            out[name] = obj
    return out


def _signature(cls) -> dict:
    params = {}
    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        params[p.name] = None if p.default is p.empty else _jsonable(p.default)
    return params


def _first_paragraph(doc: str | None) -> str:
    return " ".join(textwrap.dedent(doc or "").strip().split("\n\n")[0].split())


class Toolset:
    """Tools over one pipeline session.

    ``allow_code=True`` adds ``run_python``, which executes model-written Python
    in this process (with ``pipe``, ``np`` and ``ewald`` in scope). Leave it off
    unless you trust the model's output as you would your own scripts.
    """

    def __init__(self, pipeline: Pipeline | None = None, *, allow_code: bool = False,
                 workdir: str | None = None, render_size=(1200, 900)):
        self.pipeline = pipeline
        self.allow_code = allow_code
        self.workdir = workdir or tempfile.mkdtemp(prefix="ewald-llm-")
        self.render_size = tuple(render_size)
        self.frame = 0
        self.figures: list[str] = []
        self._source_call: dict | None = None
        self._code_blocks: list[str] = []
        self._namespace: dict = {}
        self.tools: dict[str, Tool] = {}
        self._register()

    # ------------------------------------------------------------- public API
    def definitions(self) -> list[dict]:
        """Tool definitions in Anthropic Messages API format."""
        return [t.definition() for t in self.tools.values()]

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        """Run tool ``name``; errors come back as ``is_error`` results, never raise."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"unknown tool {name!r}; available: {', '.join(self.tools)}", is_error=True)
        try:
            out = tool.function(**(arguments or {}))
        except Exception as exc:  # the model reads the error and corrects itself
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        return out if isinstance(out, ToolResult) else ToolResult(_jsonable(out))

    def to_script(self) -> str:
        """Standalone Python script reproducing the current pipeline."""
        lines = ["from ewald import import_file", ""]
        mods = [m for m in (self.pipeline.modifiers if self.pipeline else [])]
        builtin = sorted({type(m).__name__ for m in mods if not isinstance(m, PythonModifier)})
        if builtin:
            lines.insert(1, f"from ewald.modifiers import {', '.join(builtin)}")
        if self._source_call:
            args = [repr(self._source_call["path"])]
            args += [f"{k}={v!r}" for k, v in self._source_call.get("kwargs", {}).items()]
            lines.append(f"pipe = import_file({', '.join(args)})")
        else:
            lines.append("pipe = ...  # pipeline was passed in from Python")
        for block in self._code_blocks:
            lines += ["", block.rstrip()]
        for m in mods:
            if isinstance(m, PythonModifier):
                lines.append(f"pipe.modifiers.append({m.name})")
            else:
                lines.append(f"pipe.modifiers.append({m!r})")
        lines += ["", f"data = pipe.compute({self.frame})"]
        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------- registration
    def tool(self, name: str, description: str, schema: dict):
        def deco(fn):
            self.tools[name] = Tool(name, description, schema, fn)
            return fn
        return deco

    def _need_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("no structure loaded; call load_structure first")
        return self.pipeline

    def _register(self):
        T = self.tool
        frame_p = {"type": "integer", "description": "frame index (default: current frame; negative counts from the end)"}
        from .. import plugins

        formats = "; ".join(f"'{p.name}': {p.description}" for p in plugins.readers().values()
                            if p.priority is not None or p.origin != "built-in")
        format_help = f"optional; detected from the file when omitted. Readers: {formats}"

        @T("load_structure",
           "Load a structure or trajectory file (VASP POSCAR/CONTCAR/XDATCAR/OUTCAR, LAMMPS dump/data, "
           "extxyz, cif, ... ; wildcards allowed) as the session pipeline, replacing any previous one. "
           "Returns frame count and a description of the first frame.",
           _obj({"path": {"type": "string"},
                 "format": {"type": "string", "description": format_help},
                 "type_map": {"type": "object", "description": "LAMMPS numeric type -> element, e.g. {\"1\": \"Cu\"}",
                              "additionalProperties": {"type": "string"}}}, ["path"]))
        def load_structure(path, format=None, type_map=None):
            from ..io import import_file

            kwargs = {}
            if format:
                kwargs["format"] = format
            if type_map:
                kwargs["type_map"] = {int(k) if str(k).isdigit() else k: v for k, v in type_map.items()}
            self.pipeline = import_file(os.path.expanduser(path), **kwargs)
            self._source_call = {"path": path, "kwargs": kwargs}
            self._code_blocks.clear()
            self.frame = 0
            return {"num_frames": self.pipeline.num_frames,
                    "first_frame": _describe.describe_frame(self.pipeline.compute(0), rdf_peaks_=False, frame=0)}

        @T("describe_frame",
           "Compact description of one frame after the current modifiers: composition, cell, density, "
           "minimum distance, coordination histogram per type, g(r) peaks, property ranges, sanity warnings. "
           "Use this instead of asking for raw coordinates.",
           _obj({"frame": frame_p,
                 "cutoff": {"type": ["number", "string", "null"],
                            "description": "coordination cutoff in Angstrom, 'auto' (first g(r) minimum, default) or null to skip"}}))
        def describe_frame(frame=None, cutoff="auto"):
            pipe = self._need_pipeline()
            f = self._frame(frame)
            return _describe.describe_frame(pipe.compute(f), cutoff=cutoff, frame=f)

        @T("describe_trajectory",
           "Scan the trajectory (source frames, subsampled to max_frames) and return per-frame descriptor "
           "statistics (volume, min distance, MSD and max displacement vs the first frame, coordination) and "
           "'events': frames where a descriptor shifts abruptly. Start here for any 'what happens in this run' question.",
           _obj({"start": {"type": "integer"}, "stop": {"type": "integer"}, "step": {"type": "integer"},
                 "max_frames": {"type": "integer", "description": "default 200"},
                 "cutoff": {"type": ["number", "string", "null"], "description": "coordination cutoff, 'auto' or null"},
                 "threshold": {"type": "number", "description": "change-point threshold in noise units (default 6; lower finds more)"}}))
        def describe_trajectory(start=0, stop=None, step=None, max_frames=200, cutoff="auto", threshold=6.0):
            return _describe.describe_trajectory(self._need_pipeline(), start, stop, step, max_frames,
                                                 cutoff, threshold=threshold)

        @T("event_details",
           "Measure what changes around a frame (e.g. an event from describe_trajectory): which atoms moved "
           "and how far (minimum image, by identifier), by type, local vs system-wide, coordination "
           "transitions, and the top movers with positions before/after.",
           _obj({"frame": {"type": "integer"}, "window": {"type": "integer", "description": "frames each side (default 5)"},
                 "cutoff": {"type": "number"}, "threshold": {"type": "number", "description": "moved-atom threshold in A"}},
                ["frame"]))
        def event_details(frame, window=5, cutoff=None, threshold=None):
            from .narrate import event_details as details

            return details(self._need_pipeline(), int(frame), window, cutoff, threshold)

        @T("list_modifiers", "Available modifiers with their parameters and defaults.", _obj({}))
        def list_modifiers():
            return {name: {"params": _signature(cls), "doc": _first_paragraph(cls.__doc__)}
                    for name, cls in _modifier_classes(self.allow_code).items()}

        @T("add_modifier",
           "Append (or insert at 'index') a modifier to the pipeline. Property expressions use OVITO syntax, "
           "e.g. ExpressionSelection {\"expression\": \"(Coordination < 12) & (ParticleType == 'Cu')\"}. "
           "Returns the new pipeline and what the modifier changed on the current frame.",
           _obj({"name": {"type": "string"}, "params": {"type": "object"}, "index": {"type": "integer"}}, ["name"]))
        def add_modifier(name, params=None, index=None):
            pipe = self._need_pipeline()
            classes = _modifier_classes(self.allow_code)
            if name not in classes:
                raise ValueError(f"unknown modifier {name!r}; available: {', '.join(classes)}")
            mod = classes[name](**(params or {}))
            before = pipe.compute(self.frame)
            if index is None:
                pipe.modifiers.append(mod)
            else:
                pipe.modifiers.insert(index, mod)
            try:
                after = pipe.compute(self.frame)
            except Exception:
                pipe.modifiers.remove(mod)
                raise
            return {"pipeline": self._pipeline_list(), "effect": _diff(before, after)}

        @T("remove_modifier", "Remove the modifier at 'index' (see show_pipeline).",
           _obj({"index": {"type": "integer"}}, ["index"]))
        def remove_modifier(index):
            pipe = self._need_pipeline()
            removed = pipe.modifiers.pop(index)
            return {"removed": repr(removed), "pipeline": self._pipeline_list()}

        @T("show_pipeline", "Current source, modifier list and frame.", _obj({}))
        def show_pipeline():
            pipe = self._need_pipeline()
            return {"source": self._source_call, "num_frames": pipe.num_frames, "frame": self.frame,
                    "modifiers": self._pipeline_list()}

        @T("set_frame", "Set the current frame used by tools that take no frame argument.",
           _obj({"frame": {"type": "integer"}}, ["frame"]))
        def set_frame(frame):
            self.frame = self._frame(frame)
            return {"frame": self.frame}

        @T("evaluate_expression",
           "Evaluate an OVITO-style per-particle expression on a frame and return statistics "
           "(count of true values for conditions; min/max/mean/percentiles for numbers). "
           "Names: Position.X, ParticleType, ParticleIdentifier, Coordination, any computed property; N, CellSize.X.",
           _obj({"expression": {"type": "string"}, "frame": frame_p}, ["expression"]))
        def evaluate_expression(expression, frame=None):
            from ..modifiers.expression import evaluate

            data = self._need_pipeline().compute(self._frame(frame))
            v = np.asarray(evaluate(expression, data, self._frame(frame)))
            if v.dtype == bool:
                idx = np.nonzero(v)[0]
                return {"count": int(v.sum()), "of": int(v.size), "fraction": float(v.mean()) if v.size else 0.0,
                        "first_indices": idx[:20].tolist()}
            if v.ndim == 0:
                return {"value": v.item()}
            v = v.astype(float)
            return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                    "percentiles": dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                            np.percentile(v, [5, 25, 50, 75, 95]).round(4).tolist()))}

        @T("compute_rdf",
           "Radial distribution function g(r) of a frame (total, plus partial pairs). Returns peak positions "
           "and a downsampled table.",
           _obj({"frame": frame_p, "cutoff": {"type": "number", "description": "r_max in Angstrom (default 6)"},
                 "bins": {"type": "integer"}, "partial": {"type": "boolean"}}))
        def compute_rdf(frame=None, cutoff=6.0, bins=200, partial=True):
            from ..kernels import rdf

            data = self._need_pipeline().compute(self._frame(frame))
            types = data.types if partial and len(np.unique(data.types)) > 1 else None
            table = rdf(data.positions, data.cell, cutoff, bins, types=types)
            return {"peaks": _describe.rdf_peaks(data, rmax=cutoff, nbins=bins, partial=partial),
                    "table": _downsample(table)}

        @T("compute_msd",
           "Time-averaged mean squared displacement over source frames (unwrapped, matched by identifier), per "
           "type, with a linear-regime fit D = slope/6. Give timestep_fs (time between stored frames) to get D in "
           "cm^2/s; otherwise D is in A^2 per frame.",
           _obj({"start": {"type": "integer"}, "stop": {"type": "integer"}, "step": {"type": "integer"},
                 "timestep_fs": {"type": "number"},
                 "fit_range": {"type": "array", "items": {"type": "number"},
                               "description": "fraction of lags used for the fit, default [0.1, 0.5]"},
                 "unwrapped": {"type": "boolean", "description": "coordinates are already unwrapped"}}))
        def compute_msd(start=0, stop=None, step=1, timestep_fs=None, fit_range=(0.1, 0.5), unwrapped=False):
            from ..kernels import trajectory_msd

            table = trajectory_msd(self._need_pipeline(), start, stop, step, unwrapped=unwrapped)
            lag = np.asarray(table["lag"], float)
            lo, hi = (int(f * len(lag)) for f in fit_range)
            hi = max(hi, lo + 2)
            fits = {}
            for k, msd in table.items():
                if k == "lag" or len(lag) < 3:
                    continue
                slope, icpt = np.polyfit(lag[lo:hi], np.asarray(msd)[lo:hi], 1)
                d = slope / 6.0
                entry = {"slope_A2_per_frame": float(slope), "intercept_A2": float(icpt)}
                if timestep_fs:
                    entry["D_cm2_per_s"] = float(d / timestep_fs * 1e-1)  # A^2/fs -> cm^2/s
                else:
                    entry["D_A2_per_frame"] = float(d)
                fits[k] = entry
            return {"fit_lags": [float(lag[lo]), float(lag[min(hi, len(lag)) - 1])] if len(lag) else [],
                    "fits": fits, "table": _downsample(table)}

        @T("render_image",
           "Render a frame of the pipeline to PNG and return the image so you can inspect it. "
           "direction: 'x','y','z','-x',...,'iso' or a view vector [x,y,z].",
           _obj({"frame": frame_p, "direction": {"type": ["string", "array"]},
                 "bonds": {"type": ["number", "null"], "description": "draw bonds shorter than this (A)"},
                 "zoom": {"type": "number"}, "width": {"type": "integer"}, "height": {"type": "integer"},
                 "filename": {"type": "string", "description": "also save under this name in the work dir"}}))
        def render_image(frame=None, direction="iso", bonds=None, zoom=1.0, width=None, height=None, filename=None):
            pipe = self._need_pipeline()
            f = self._frame(frame)
            size = (width or self.render_size[0], height or self.render_size[1])
            name = filename or f"frame{f:05d}_{len(self.figures):03d}.png"
            path = os.path.join(self.workdir, os.path.basename(name))
            render_png(pipe.compute(f), path, size=size, direction=direction, bonds=bonds, zoom=zoom)
            self.figures.append(path)
            with open(path, "rb") as fh:
                png = fh.read()
            return ToolResult({"path": path, "frame": f, "size": list(size)}, images=[png])

        writer_names = ", ".join(sorted(plugins.writers()))

        @T("export_file",
           "Write frames of the pipeline (after its modifiers) to a file: POSCAR, LAMMPS data/dump, extxyz, "
           "CIF, XSF, H5MD, XTC, cube/CHGCAR, ASE db, ... The format follows the file name (POSCAR, *.data, "
           "*.extxyz, *.h5md) unless given. One-frame formats (POSCAR, LAMMPS data, CHGCAR, cube) write the "
           "frame they are given (default: the current one). Relative paths go to the work dir.",
           _obj({"path": {"type": "string"},
                 "format": {"type": "string", "description": f"optional writer name or ASE format; writers: {writer_names}"},
                 "frames": {"type": ["string", "array"], "items": {"type": "integer"},
                            "description": "'current' (default), 'all', or a list of frame indices"}}, ["path"]))
        def export_file(path, format=None, frames="current"):
            from ..io import export_file as _export

            pipe = self._need_pipeline()
            target = os.path.expanduser(path)
            if not os.path.isabs(target):
                target = os.path.join(self.workdir, target)
            writer = plugins.find_writer(target, format)
            if frames == "all":
                idx = list(range(pipe.num_frames))
            elif frames in (None, "current"):
                idx = [self.frame]
            else:
                idx = [int(i) for i in frames]
            _export(pipe, target, format=format, frames=idx)
            return {"path": target, "writer": writer.name, "frames": len(idx)}

        @T("get_script", "Standalone Python script that reproduces the current pipeline.", _obj({}))
        def get_script():
            return {"script": self.to_script()}

        if self.allow_code:
            @T("python_api",
               "Reference for the ewald Python API (data model, expression syntax, differences from OVITO, every "
               "modifier signature). Read it before writing code for run_python.", _obj({}))
            def python_api():
                from .reference import api_reference

                return api_reference()

            @T("run_python",
               "Execute Python in the session (variables persist). In scope: pipe (Pipeline), np, ewald, "
               "data (current frame). Define f(frame, data) and set add_as_modifier to append it as a modifier. "
               "Print what you want to see; stdout is returned (truncated).",
               _obj({"code": {"type": "string"}, "add_as_modifier": {"type": "string",
                                                                     "description": "name of a function to append"}},
                    ["code"]))
            def run_python(code, add_as_modifier=None):
                import contextlib
                import io

                import ewald

                ns = self._namespace
                ns.update({"np": np, "ewald": ewald, "pipe": self.pipeline})
                if self.pipeline is not None:
                    ns["data"] = self.pipeline.compute(self.frame)
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        exec(compile(code, "<run_python>", "exec"), ns)
                except Exception:
                    return ToolResult(buf.getvalue()[-4000:] + traceback.format_exc(limit=3), is_error=True)
                self._code_blocks.append(code)
                if add_as_modifier:
                    self._need_pipeline().modifiers.append(ns[add_as_modifier])
                return {"stdout": buf.getvalue()[-4000:], "pipeline": self._pipeline_list() if self.pipeline else []}

    # -------------------------------------------------------------- helpers
    def _frame(self, frame) -> int:
        n = self._need_pipeline().num_frames
        return self.frame if frame is None else int(frame) % n

    def _pipeline_list(self) -> list[str]:
        return [f"{i}: {m!r}" + ("" if m.enabled else " (disabled)")
                for i, m in enumerate(self._need_pipeline().modifiers)]


def _diff(before, after) -> dict:
    """What a modifier changed: particle count, new properties, selection, attributes."""
    out = {"num_particles": [before.count, after.count]}
    new = [k for k in after.particles if k not in before.particles]
    if new:
        out["new_properties"] = {k: _describe._property_summary(after).get(k) for k in new}
    if "Selection" in after.particles:
        out["selected"] = int(after.selection.sum())
    attrs = {k: v for k, v in after.attributes.items()
             if k not in before.attributes and (np.isscalar(v) or isinstance(v, tuple))}
    if attrs:
        out["new_attributes"] = attrs
    tables = [k for k in after.tables if k not in before.tables]
    if tables:
        out["new_tables"] = tables
    return out


def render_png(data, path, **kwargs):
    """Rendering hook (tests replace it; the GUI can point it at its own viewport)."""
    from ..render import render

    render(data, path, **kwargs)
