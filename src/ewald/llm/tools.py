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

import functools
import inspect
import json
import os
import pathlib
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
    available: Callable[[], bool] | None = None   # None: always offered; else asked on every send

    def offered(self) -> bool:
        return self.available is None or bool(self.available())

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


def downsample(table: dict, max_rows: int = 60) -> dict:
    """At most ``max_rows`` evenly spaced rows of a column table, 4 significant digits (for tool results)."""
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
    from .. import plugins

    for name, plugin in plugins.modifiers().items():
        out.setdefault(name, plugin.cls)
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

    ``notebook`` (a :class:`~ewald.notebook.Notebook`, a function returning the current one, or a
    :class:`~ewald.llm.notebook_tools.NotebookAccess`) adds the notebook cell tools
    (:mod:`ewald.llm.notebook_tools`); ``run_cell`` among them only with ``allow_code`` or an
    access that allows running.
    """

    def __init__(self, pipeline: Pipeline | None = None, *, allow_code: bool = False,
                 workdir: str | None = None, render_size=(1200, 900), notebook=None):
        self.pipeline = pipeline
        self.allow_code = allow_code
        self.workdir = workdir or tempfile.mkdtemp(prefix="ewald-llm-")
        self.render_size = tuple(render_size)
        self.frame = 0
        self.pipeline_name = "pipe"   # what the user's notebook calls the pipeline (code in set_view results)
        #: Particle indices the user picked (set by the GUI); describe_selection's default.
        self.selection: list[int] | None = None
        self.figures: list[str] = []
        self._source_call: dict | None = None
        self._code_blocks: list[str] = []
        self._namespace: dict = {}
        self.visuals: list[dict] = []  # plugin visual elements drawn by render_image: {name, params, enabled}
        self.overlays: list[dict] = []  # overlay elements drawn by render_image: {name, params, enabled}
        self.theme: str | None = None   # theme render_image uses (ewald.themes)
        self.tools: dict[str, Tool] = {}
        self.notebook_access = None
        self._register()
        if notebook is not None:
            self.attach_notebook(notebook)

    # ------------------------------------------------------------- public API
    def definitions(self) -> list[dict]:
        """Tool definitions in Anthropic Messages API format."""
        return [t.definition() for t in self.tools.values() if t.offered()]

    def attach_notebook(self, notebook):
        """Add the notebook cell tools over ``notebook`` (see the class docstring); returns the access."""
        from .notebook_tools import NotebookAccess, register

        if not isinstance(notebook, NotebookAccess):
            notebook = NotebookAccess(notebook, can_run=lambda: self.allow_code)
        self.notebook_access = notebook
        register(self, notebook)   # replaces plugin tools of the same name: built-in names win
        return notebook

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        """Run tool ``name``; errors come back as ``is_error`` results, never raise."""
        tool = self.tools.get(name)
        if tool is not None and not tool.offered():
            return ToolResult(f"{name} is not offered in this session (the user has not allowed it)", is_error=True)
        if tool is None:
            offered = ", ".join(n for n, t in self.tools.items() if t.offered())
            return ToolResult(f"unknown tool {name!r}; available: {offered}", is_error=True)
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
        if self.pipeline is not None and self.pipeline.view.changed():
            lines.insert(1, "from ewald import view")
            lines += ["", self.pipeline.view.source("pipe")]
        lines += ["", f"data = pipe.compute({self.frame})"]
        figure = {k: v for k, v in (("visuals", self.visuals), ("overlays", self.overlays), ("theme", self.theme)) if v}
        if figure:
            lines += [f"{k} = {v!r}" for k, v in figure.items()]
            args = ", ".join(f"{k}={k}" for k in figure)
            lines.append(f"# from ewald.render import render; render(data, 'figure.png', {args})")
        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------- registration
    def tool(self, name: str, description: str, schema: dict):
        def deco(fn):
            self.tools[name] = Tool(name, description, schema, fn)
            return fn
        return deco

    def _selection_indices(self, data, indices=None, ids=None, expression=None) -> np.ndarray:
        """Particle indices from explicit indices, Particle Identifiers, an expression, else the user's picks."""
        if indices is not None:
            return np.asarray(indices, int)
        if ids is not None:
            pid = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)
            idx = np.nonzero(np.isin(pid, np.asarray(ids, int)))[0]
            if len(idx) < len(set(ids)):
                raise ValueError(f"unknown Particle Identifier(s): {sorted(set(ids) - set(pid[idx].tolist()))}")
            return idx
        if expression:
            from ..modifiers.expression import evaluate

            mask = np.broadcast_to(np.asarray(evaluate(expression, data), bool), (data.count,))
            return np.nonzero(mask)[0]
        if self.selection is not None and len(self.selection):
            return np.asarray(self.selection, int)
        if "Selection" in data and data.selection.any():
            return np.nonzero(data.selection)[0]
        raise ValueError("nothing is selected; pass indices, ids or expression")

    def selection_image(self, data, indices, radius=6.0) -> bytes | None:
        """PNG close-up of ``indices`` and their surroundings (None when rendering fails)."""
        from .selection import selection_scene

        path = os.path.join(self.workdir, f"selection_{len(self.figures):03d}.png")
        try:
            render_png(selection_scene(data, indices, radius), path, size=(900, 700), direction="iso",
                       show_cell=False, quality="standard")
        except Exception:  # an answer without the picture is still useful
            traceback.print_exc()
            return None
        self.figures.append(path)
        with open(path, "rb") as fh:
            return fh.read()

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
           "Load a structure or trajectory file (VASP POSCAR/CONTCAR/XDATCAR/OUTCAR/vaspout.h5, LAMMPS dump/data, "
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

        @T("describe_selection",
           "Describe particles and their surroundings: per-atom neighbor lists, coordination numbers, structure "
           "types (if CNA/PTM ran), property values against the environment and frame averages, the nearest "
           "defect, and a close-up render with the particles highlighted. Without arguments it describes the "
           "particles the user picked in the viewport.",
           _obj({"frame": frame_p,
                 "indices": {"type": "array", "items": {"type": "integer"}, "description": "0-based particle indices"},
                 "ids": {"type": "array", "items": {"type": "integer"}, "description": "Particle Identifiers"},
                 "expression": {"type": "string", "description": "selection expression, e.g. 'StructureType == 0'"},
                 "radius": {"type": "number", "description": "environment shell in Angstrom (default 6)"},
                 "cutoff": {"type": ["number", "string"], "description": "neighbor cutoff in Angstrom or 'auto'"},
                 "image": {"type": "boolean", "description": "include the close-up render (default true)"}}))
        def describe_selection(frame=None, indices=None, ids=None, expression=None, radius=6.0, cutoff="auto",
                               image=True):
            from . import selection as _selection

            pipe = self._need_pipeline()
            f = self._frame(frame)
            data = pipe.compute(f)
            idx = self._selection_indices(data, indices, ids, expression)
            out = _selection.describe_selection(data, idx, radius=radius, cutoff=cutoff, frame=f)
            if not image:
                return out
            png = self.selection_image(data, idx, radius)
            return ToolResult(_jsonable(out), images=[png] if png else [])

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
                    "table": downsample(table)}

        @T("analyze_symmetry",
           "Space group (Hermann-Mauguin and number), point group, crystal system and the Wyckoff site of every "
           "symmetry-unique atom of a frame (spglib). symprec is the distance tolerance in Angstrom.",
           _obj({"frame": frame_p, "symprec": {"type": "number", "description": "default 0.01"}}))
        def analyze_symmetry(frame=None, symprec=0.01):
            from ..modifiers.symmetry import SymmetryAnalysis

            data = self._need_pipeline().compute(self._frame(frame)).copy()
            SymmetryAnalysis(symprec=symprec).modify(data, None)
            out = {k.split(".", 1)[1]: v for k, v in data.attributes.items() if k.startswith("SymmetryAnalysis.")}
            frac = data.cell.to_fractional(data.positions) % 1.0
            eq = data["Equivalent Atom"]
            out["sites"] = [{"index": int(i), "type": str(data.types[i]), "wyckoff": str(data["Wyckoff Position"][i]),
                             "site_symmetry": str(data["Site Symmetry"][i]),
                             "fractional": np.round(frac[i], 4).tolist()} for i in np.unique(eq)[:40]]
            return out

        @T("compute_diffraction",
           "Powder X-ray or neutron diffraction peaks of a frame (2-theta, d, hkl, relative intensity; strongest "
           "first), or the structure factor S(q) when kind='sq'.",
           _obj({"frame": frame_p, "kind": {"type": "string", "enum": ["xrd", "neutron", "sq"]},
                 "wavelength": {"type": "string", "description": "Angstrom or 'CuKa', 'MoKa', ... (default CuKa)"},
                 "two_theta_max": {"type": "number"}, "q_max": {"type": "number"}}))
        def compute_diffraction(frame=None, kind="xrd", wavelength="CuKa", two_theta_max=90.0, q_max=10.0):
            from .. import diffraction as dif

            data = self._need_pipeline().compute(self._frame(frame))
            if kind == "sq":
                table = dif.structure_factor(data.positions, data.cell, q_max, 200, types=data.types)
                return {"table": downsample(table)}
            p = dif.powder_peaks(data.positions, data.types, data.cell, wavelength,
                                 "neutron" if kind == "neutron" else "xray", (5.0, two_theta_max))
            order = np.argsort(-p["Intensity"])[:25]
            return {"num_peaks": int(len(p["2theta"])), "wavelength": dif.wavelength_of(wavelength),
                    "peaks": [{"2theta": round(float(p["2theta"][i]), 3), "d": round(float(p["d"][i]), 4),
                               "hkl": [int(p["h"][i]), int(p["k"][i]), int(p["l"][i])],
                               "intensity": round(float(p["Intensity"][i]), 2),
                               "multiplicity": int(p["Multiplicity"][i])} for i in order]}

        @T("compute_msd",
           "Time-averaged mean squared displacement over source frames (unwrapped, matched by identifier), per "
           "type, with a linear-regime fit D = slope/6. timestep_fs (time between stored frames) defaults to what "
           "the run's INCAR / log.lammps says (see check_run); with it D is in cm^2/s, else in A^2 per frame.",
           _obj({"start": {"type": "integer"}, "stop": {"type": "integer"}, "step": {"type": "integer"},
                 "timestep_fs": {"type": "number"},
                 "fit_range": {"type": "array", "items": {"type": "number"},
                               "description": "fraction of lags used for the fit, default [0.1, 0.5]"},
                 "unwrapped": {"type": "boolean", "description": "coordinates are already unwrapped"}}))
        def compute_msd(start=0, stop=None, step=1, timestep_fs=None, fit_range=(0.1, 0.5), unwrapped=False):
            from ..kernels import trajectory_msd
            from ..runinfo import run_info

            if timestep_fs is None and run_info(self._need_pipeline()).time_per_frame:
                timestep_fs = run_info(self._need_pipeline()).time_per_frame * 1000.0
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
                    "fits": fits, "table": downsample(table)}

        @T("check_run",
           "What the loaded run is and whether it can be trusted, from its input and log files (INCAR, OSZICAR, "
           "OUTCAR, log.lammps): code, kind (md/relax/static/neb), ensemble, target temperature, MD time step and "
           "time per stored frame, then findings: SCF steps that hit NELM, relaxation not converged, the first "
           "equilibrated frame of MD, temperature off target, energy drift, LAMMPS warnings/errors, atoms too close. "
           "Call it first on a new MD or relaxation output, and cite its findings.",
           _obj({}))
        def check_run():
            from ..runinfo import check_run as _check

            return _check(self._need_pipeline()).to_dict()

        @T("render_image",
           "Render a frame of the pipeline to PNG and return the image so you can inspect it. It draws with the "
           "pipeline's view settings (set_view: style, colors, radii, bonds, cell, camera, ...); the arguments "
           "here override them for this picture only. direction: 'x','y','z','-x',...,'iso' or a view vector [x,y,z].",
           _obj({"frame": frame_p, "direction": {"type": ["string", "array"]},
                 "bonds": {"type": ["number", "null"], "description": "draw bonds shorter than this (A)"},
                 "zoom": {"type": "number"}, "width": {"type": "integer"}, "height": {"type": "integer"},
                 "theme": {"type": ["string", "null"],
                           "description": "theme for this and later renders (see list_styles); null keeps it"},
                 "filename": {"type": "string", "description": "also save under this name in the work dir"}}))
        def render_image(frame=None, direction=None, bonds=None, zoom=None, width=None, height=None, filename=None,
                         theme=None):
            pipe = self._need_pipeline()
            f = self._frame(frame)
            size = (width or self.render_size[0], height or self.render_size[1])
            name = filename or f"frame{f:05d}_{len(self.figures):03d}.png"
            path = os.path.join(self.workdir, os.path.basename(name))
            if theme is not None:
                from ..themes import get_theme

                get_theme(theme)  # unknown names fail here, before rendering
                self.theme = theme
                pipe.view.theme = theme
            data = pipe.compute(f)
            kw = pipe.view.render_kwargs(data, f)
            for key in ("size", "renderer", "quality"):
                kw.pop(key, None)
            if direction is not None:
                kw.pop("camera", None)
                kw["direction"] = direction
            if zoom is not None:
                kw.pop("camera", None)
                kw["zoom"] = zoom
            if bonds is not None:
                kw.pop("bond_pairs", None)
                kw["bonds"] = bonds
            if self.visuals:
                kw["visuals"] = list(kw.get("visuals", [])) + list(self.visuals)
            if self.overlays:
                kw["overlays"] = list(kw.get("overlays", [])) + list(self.overlays)
            if self.theme and "theme" not in kw:
                kw["theme"] = self.theme
            render_png(data, path, size=size, **kw)
            self.figures.append(path)
            with open(path, "rb") as fh:
                png = fh.read()
            return ToolResult({"path": path, "frame": f, "size": list(size)}, images=[png])

        from ..view3d import GROUPS, SETTINGS

        view_props = {k: {**st.schema, "description": st.doc} for k, st in SETTINGS.items()}
        groups = "; ".join(f"{g}: {', '.join(k for k, st in SETTINGS.items() if st.group == g)}" for g in GROUPS)

        @T("get_view",
           "The 3D structure view's settings (the same ones the user's notebook sets with view(pipe, ...) and the "
           "GUI viewport shows): 'changed' lists what differs from the defaults, 'settings' every current value, "
           "'elements' the same values by visual element (Particles, Bonds, Surfaces, Simulation cell, ...), "
           f"'help' what each means. Groups: {groups}.",
           _obj({}))
        def get_view():
            v = self._need_pipeline().view
            return {"changed": v.changed(), "settings": v.settings(), "elements": v.elements.describe(),
                    "help": {k: st.doc for k, st in SETTINGS.items()}, "code": v.source(self.pipeline_name)}

        @T("set_view",
           "Change how the structure is drawn in the GUI viewport, notebook pictures and render_image: style "
           "preset, theme, element colors and radii, atoms on/off, bonds on/off, bond table, cell, background, VESTA "
           "boundary, polyhedra/isosurface look by surface name (surfaces), plugin visuals, vectors, camera "
           "(direction, zoom, rotate), frame, and picture options (renderer, "
           "quality, legends, scale bar, text). Only the settings given change; null resets one to its default. "
           "'rotate' turns the current view by [azimuth, elevation] degrees; 'reset' (true or a list of names) "
           "goes back to defaults first. Per-atom colors by condition are steps instead (add_modifier "
           "ColorCoding / expression selection). Returns the changed settings and the notebook code for them.",
           _obj({**view_props,
                 "rotate": {"type": "array", "items": {"type": "number"},
                            "description": "[azimuth, elevation] in degrees, applied after the other settings"},
                 "reset": {"type": ["boolean", "array"], "items": {"type": "string"},
                           "description": "true: every setting back to its default first; or a list of names"}}))
        def set_view(rotate=None, reset=None, **settings):
            v = self._need_pipeline().view
            if reset is True:
                v.reset()
            elif reset:
                v.reset(*reset)
            v.update(**settings)
            if rotate is not None:
                v.rotate(*rotate)
            if "frame" in settings:
                self.frame = self._frame(v.frame)
            return {"changed": v.changed(), "code": v.source(self.pipeline_name)}

        @T("list_visuals",
           "Visual elements: the built-in ones (Particles, Bonds, Surfaces, Vectors, Trajectory lines, Simulation "
           "cell; configured through render_image and modifiers) and installed plugins, which add_visual draws "
           "on top in every renderer. Lists each with its parameters, plus the ones currently added.",
           _obj({}))
        def list_visuals():
            out = {p.name: {"builtin": p.builtin, "category": p.category, "doc": p.description,
                            "params": _signature(p.cls)} for p in plugins.visuals().values()}
            return {"available": out, "added": list(self.visuals)}

        @T("add_visual",
           "Draw a plugin visual element (see list_visuals) in every later render_image, with optional "
           "constructor parameters. Returns the visuals now added.",
           _obj({"name": {"type": "string"}, "params": {"type": "object"}}, ["name"]))
        def add_visual(name, params=None):
            from ..scene import coerce_visuals, visual_to_dict

            reg = plugins.visuals()
            if name not in reg:
                raise ValueError(f"unknown visual element {name!r}; available: {', '.join(reg)}")
            if reg[name].builtin:
                raise ValueError(f"{name!r} is built in and always drawn; set it up through render_image or "
                                 "a modifier (CreateBonds, VectorVis, ...)")
            (el,) = coerce_visuals([{"name": name, "params": params or {}}])
            self.visuals.append(visual_to_dict(el))
            return {"visuals": list(self.visuals)}

        @T("remove_visual", "Stop drawing the visual element at 'index' of the added list.",
           _obj({"index": {"type": "integer"}}, ["index"]))
        def remove_visual(index):
            removed = self.visuals.pop(index)
            return {"removed": removed, "visuals": list(self.visuals)}

        @T("list_overlays",
           "Overlay elements drawn on top of rendered images (built-in: Coordinate tripod, Attributes; plus "
           "plugins), with their parameters, and the ones currently added. The text label, color legend, "
           "type legend and scale bar are separate render options.",
           _obj({}))
        def list_overlays():
            out = {p.name: {"builtin": p.builtin, "category": p.category, "doc": p.description,
                            "params": _signature(p.cls)} for p in plugins.overlays().values()}
            return {"available": out, "added": list(self.overlays)}

        @T("add_overlay",
           "Draw an overlay element (see list_overlays) on every later render_image, with optional "
           "constructor parameters. Returns the overlays now added.",
           _obj({"name": {"type": "string"}, "params": {"type": "object"}}, ["name"]))
        def add_overlay(name, params=None):
            from ..overlays import coerce_overlays, overlay_to_dict

            (el,) = coerce_overlays([{"name": name, "params": params or {}}])
            self.overlays.append(overlay_to_dict(el))
            return {"overlays": list(self.overlays)}

        @T("remove_overlay", "Stop drawing the overlay at 'index' of the added list.",
           _obj({"index": {"type": "integer"}}, ["index"]))
        def remove_overlay(index):
            removed = self.overlays.pop(index)
            return {"removed": removed, "overlays": list(self.overlays)}

        @T("list_styles",
           "Themes (element colors, background, cell color, radius preset; pass one to render_image) and the "
           "color maps usable as 'cmap' in ColorCoding, GridSlice and VectorVis: common matplotlib names and "
           "plugins, each marked whether it reads in grayscale.",
           _obj({}))
        def list_styles():
            from ..colormaps import is_perceptual, names

            kinds = {k: p.kind for k, p in plugins.colormaps().items()}
            maps = {}
            for n in names():
                maps[n] = {"kind": kinds.get(n, "matplotlib"), "plugin": n in kinds,
                           "grayscale_safe": is_perceptual(n, kinds.get(n))}
            return {"themes": {k: p.description for k, p in plugins.themes().items()}, "current_theme": self.theme,
                    "colormaps": maps, "note": "any other matplotlib colormap name also works"}

        writer_names = ", ".join(sorted(plugins.writers()))

        @T("export_file",
           "Write frames of the pipeline (after its modifiers) to a file: POSCAR, LAMMPS data/dump, extxyz, "
           "CIF, XSF, H5MD, XTC, cube/CHGCAR, ASE db, ... The format follows the file name (POSCAR, *.data, "
           "*.extxyz, *.h5md) unless given. One-frame formats (POSCAR, LAMMPS data, CHGCAR, cube) write the "
           "frame they are given (default: the current one). Relative paths go to the work dir; an existing "
           "file is only replaced with overwrite=true, and only inside the work dir. Outside it, only new "
           "files in folders that are not hidden can be written.",
           _obj({"path": {"type": "string"},
                 "format": {"type": "string", "description": f"optional writer name or ASE format; writers: {writer_names}"},
                 "frames": {"type": ["string", "array"], "items": {"type": "integer"},
                            "description": "'current' (default), 'all', or a list of frame indices"},
                 "overwrite": {"type": "boolean", "description": "replace an existing file (default false)"}},
                ["path"]))
        def export_file(path, format=None, frames="current", overwrite=False):
            from ..io import export_file as _export

            pipe = self._need_pipeline()
            target = os.path.expanduser(path)
            if not os.path.isabs(target):
                target = os.path.join(self.workdir, target)
            target = os.path.realpath(target)
            work = os.path.realpath(self.workdir)
            try:
                inside = os.path.commonpath([target, work]) == work
            except ValueError:  # another drive on Windows
                inside = False
            if not inside:
                # the request may come from text inside a structure file (prompt injection):
                # never replace the user's files or write into ~/.ssh, ~/.config, ...
                if any(part.startswith(".") for part in pathlib.PurePath(target).parts[1:]):
                    raise PermissionError(f"{target}: writing into hidden folders or files is not allowed")
                if os.path.exists(target):
                    raise FileExistsError(f"{target} exists; files outside the work dir {work} are never replaced")
            elif os.path.exists(target) and not overwrite:
                raise FileExistsError(f"{target} exists; pass overwrite=true to replace it")
            os.makedirs(os.path.dirname(target), exist_ok=True)
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

        for plugin in plugins.tools().values():  # @claude_tool plugins; built-in names win
            if plugin.name not in self.tools:
                self.tools[plugin.name] = Tool(plugin.name, plugin.description, plugin.schema,
                                               functools.partial(plugin.function, self))

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
