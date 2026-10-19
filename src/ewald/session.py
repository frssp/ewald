"""Session state files (``.ewald``): pipeline, visual style, camera and render settings as JSON.

Like an OVITO ``.ovito`` session, a session file restores the whole figure::

    from ewald.session import save_session, load_session
    save_session("figure.ewald", pipe, camera=cam, render={"renderer": "raytrace", "size": (2100, 1575)})
    s = load_session("figure.ewald")
    s.pipeline.render("figure.png", camera=s.camera, **s.render_kwargs())

The input file is stored by absolute path and relative to the session file, so a
folder holding both can be moved. In-memory sources (``StaticSource``) are embedded.
Python code (``PythonScript`` modifiers, console functions) is saved as text and only
runs when loaded with ``allow_code=True``; otherwise those modifiers come back disabled.
"""
from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = 1
CODE_MODIFIERS = ("PythonScript", "PythonModifier")


# ---------------------------------------------------------------- JSON encoding of parameter values
def encode(value):
    """JSON-safe form of a parameter value (tuples, arrays, int-keyed dicts are tagged)."""
    from .pipeline import Modifier

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [encode(v) for v in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, Modifier):
        return {"__modifier__": modifier_to_dict(value)}
    raise TypeError(f"cannot store {type(value).__name__} in a session")


def decode(value, allow_code=False):
    if isinstance(value, list):
        return [decode(v, allow_code) for v in value]
    if not isinstance(value, dict):
        return value
    if "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value.get("dtype"))
    if "__tuple__" in value:
        return tuple(decode(v, allow_code) for v in value["__tuple__"])
    if "__set__" in value:
        return {_hashable(decode(v, allow_code)) for v in value["__set__"]}
    if "__dict__" in value:
        return {_hashable(decode(k, allow_code)): decode(v, allow_code) for k, v in value["__dict__"]}
    if "__modifier__" in value:
        return modifier_from_dict(value["__modifier__"], allow_code)[0]
    return {k: decode(v, allow_code) for k, v in value.items()}


def _hashable(k):
    return tuple(k) if isinstance(k, list) else k


# ---------------------------------------------------------------- modifiers
def modifier_to_dict(m) -> dict:
    from .pipeline import PythonModifier

    cls = type(m)
    out = {"type": cls.__name__, "module": cls.__module__, "enabled": bool(m.enabled)}
    if isinstance(m, PythonModifier):
        try:
            if m.function.__name__ == "<lambda>":
                raise TypeError("a lambda has no stand-alone source")
            src = textwrap.dedent(inspect.getsource(m.function))
        except (OSError, TypeError) as exc:
            raise TypeError(f"source of Python function {m.name!r} is not available ({exc}); "
                            "use a PythonScript modifier to keep code in sessions") from exc
        out.update(function=m.function.__name__, source=src, parameters=encode(m.params))
        return out
    params = dict(m.parameters())
    if hasattr(m, "params") and isinstance(m.params, dict):  # **kwargs passed through (PythonScript)
        out["extra"] = encode(m.params)
    out["parameters"] = encode(params)
    return out


def _modifier_class(entry: dict, allow_code: bool):
    from . import modifiers

    name, module = entry["type"], entry.get("module", "")
    if module.startswith("ewald.") or name in modifiers.__all__:
        try:
            return getattr(modifiers, name)
        except AttributeError:
            pass
    if module in sys.modules or (allow_code and module):
        return getattr(importlib.import_module(module), name)
    raise LookupError(f"modifier {module}.{name} is not available (import its package first, "
                      "or load with allow_code=True)")


def modifier_from_dict(entry: dict, allow_code: bool = False):
    """Rebuild one modifier; returns (modifier or None, warning or None)."""
    from .modifiers import PythonScript
    from .pipeline import PythonModifier

    name = entry["type"]
    if name == "PythonModifier":
        params = decode(entry.get("parameters", {}), allow_code)
        if allow_code:
            ns = {"np": np, "__name__": "ewald_session"}
            exec(compile(entry["source"], f"<session:{entry['function']}>", "exec"), ns)
            m = PythonModifier(ns[entry["function"]], **params)
            m.enabled = entry.get("enabled", True)
            return m, None
        # keep the code as an editable, disabled script so saving again does not lose it
        src = entry["source"] + f"\n\ndef modify(frame, data, *args, **kw):\n    return {entry['function']}(frame, data)\n"
        m = PythonScript(src, title=entry["function"], **params)
        m.enabled = False
        return m, f"{entry['function']}: Python code not run (allow_code=False); modifier disabled"
    try:
        cls = _modifier_class(entry, allow_code)
    except (LookupError, ImportError, AttributeError) as exc:
        return None, str(exc)
    params = decode(entry.get("parameters", {}), allow_code)
    extra = decode(entry.get("extra", {}), allow_code)
    sig = inspect.signature(cls.__init__)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        extra = {}
    try:
        m = cls(**params, **extra)
    except Exception as exc:
        return None, f"{name}: could not restore ({type(exc).__name__}: {exc})"
    m.enabled = entry.get("enabled", True)
    warning = None
    if name in CODE_MODIFIERS and not allow_code and m.enabled:
        m.enabled = False
        warning = f"{m.name}: Python code not run (allow_code=False); modifier disabled"
    return m, warning


# ---------------------------------------------------------------- sources
def _data_to_npz(frames) -> str:
    arrays = {}
    for k, d in enumerate(frames):
        for name, arr in d.particles.items():
            arrays[f"{k}/p/{name}"] = np.asarray(arr)
        arrays[f"{k}/cell"] = d.cell.matrix
        arrays[f"{k}/origin"] = d.cell.origin
        arrays[f"{k}/pbc"] = d.cell.pbc
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _data_from_npz(text: str, attributes: list) -> list:
    from .data import Cell, DataCollection

    z = np.load(io.BytesIO(base64.b64decode(text)), allow_pickle=False)
    frames = []
    for k, attrs in enumerate(attributes):
        d = DataCollection(cell=Cell(z[f"{k}/cell"], z[f"{k}/origin"], z[f"{k}/pbc"]))
        prefix = f"{k}/p/"
        for key in z.files:
            if key.startswith(prefix):
                d.particles[key[len(prefix):]] = z[key]
        d.attributes.update(attrs)
        frames.append(d)
    return frames


def source_to_dict(source, base_dir: str) -> dict:
    from .io import FileSource, StaticSource

    if isinstance(source, FileSource):
        path = os.path.abspath(source.path)
        try:
            rel = os.path.relpath(path, base_dir)
        except ValueError:  # other drive on Windows
            rel = None
        out = {"kind": "file", "path": path, "relative_path": rel, "format": source.format,
               "kwargs": encode(source.kwargs)}
        if source.thermo_arg != "auto":  # explicit log.lammps/OSZICAR files, or None
            arg = source.thermo_arg
            out["thermo"] = arg if arg is None or isinstance(arg, str) else [str(a) for a in arg] \
                if not isinstance(arg, os.PathLike) else str(arg)
        return out
    if isinstance(source, StaticSource):
        attrs = []
        for d in source.frames:
            keep = {}
            for k, v in d.attributes.items():
                try:
                    keep[k] = json.loads(json.dumps(encode(v)))
                except TypeError:
                    pass
            attrs.append(keep)
        return {"kind": "static", "attributes": attrs, "npz": _data_to_npz(source.frames)}
    raise TypeError(f"cannot store a {type(source).__name__} source in a session")


def source_from_dict(entry: dict, base_dir: str):
    from .io import FileSource, StaticSource

    if entry["kind"] == "static":
        frames = _data_from_npz(entry["npz"], [decode(a) for a in entry["attributes"]])
        return StaticSource(frames)
    path = entry["path"]
    rel = entry.get("relative_path")
    candidates = ([os.path.normpath(os.path.join(base_dir, rel))] if rel else []) + [path]
    for cand in candidates:
        if os.path.exists(cand) or any(c in cand for c in "*?["):
            try:
                return FileSource(cand, entry.get("format"), thermo=entry.get("thermo", "auto"),
                                  **decode(entry.get("kwargs", {})))
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"session input not found: {path}" + (f" (or {candidates[0]})" if rel else ""))


# ---------------------------------------------------------------- session
@dataclass
class Session:
    pipeline: object = None
    frame: int = 0
    camera: dict | None = None
    style: dict = field(default_factory=dict)     # gui.style.ViewStyle fields
    render: dict = field(default_factory=dict)    # render() keywords plus GUI-only keys
    device: str | None = None
    warnings: list = field(default_factory=list)
    path: str | None = None

    def render_kwargs(self) -> dict:
        """Keywords for :func:`ewald.render.render`: the visual style plus the render settings."""
        from .render import render

        kw = {}
        if self.style:
            from .gui.style import ViewStyle  # plain dataclass, no Qt import

            names = set(ViewStyle.__dataclass_fields__)
            kw.update(ViewStyle(**{k: v for k, v in self.style.items() if k in names}).scene_kwargs())
        allowed = set(inspect.signature(render).parameters) | {
            "type_colors", "type_radii", "bond_pairs", "bond_radius", "atom_style", "cell_width", "cell_color",
            "shading", "trajectory_width", "radius_preset", "oxidation_states", "coordination"}
        kw.update({k: v for k, v in self.render.items()
                   if k in allowed and k not in ("data", "filename", "text", "camera")})
        r = self.render  # keys written by the GUI's Render page
        if str(r.get("quality", "")).lower() == "custom":
            kw.pop("quality", None)
        elif r.get("quality"):  # a preset decides the sampling; the stored field values are only for Custom
            for k in ("samples", "ao_samples", "shadows", "antialiasing", "ambient_occlusion"):
                kw.pop(k, None)
            kw["quality"] = str(r["quality"]).lower()
        if r.get("renderer_name"):
            kw["renderer"] = r["renderer_name"]
        if r.get("scale_bar_on"):
            kw["scale_bar"] = r.get("scale_bar_length") or True
        bg = r.get("background_choice")
        if bg == "transparent":
            kw.update(background="white", transparent_background=True)
        elif bg == "viewport":
            kw["background"] = self.style.get("background", "white")
        elif bg:
            kw["background"] = bg
        if "size" in kw:
            kw["size"] = tuple(kw["size"])
        return kw

    @property
    def text(self) -> str | None:
        """Label template (``[Frame]``, ``[Time]``, ... still unexpanded)."""
        return self.render.get("text_template", self.render.get("text")) or None


def save_session(path, pipeline, *, frame: int = 0, camera: dict | None = None, style=None,
                 render: dict | None = None, device: str | None = None) -> str:
    """Write a session file. ``style`` is a ``ViewStyle`` or its dict; ``render`` holds
    :func:`ewald.render.render` keywords (and any extra GUI keys)."""
    path = os.fspath(path)
    base = os.path.dirname(os.path.abspath(path))
    mods = []
    for m in pipeline.modifiers:
        mods.append(modifier_to_dict(m))
    if style is not None and not isinstance(style, dict):
        style = style.to_dict()
    doc = {"ewald_session": FORMAT_VERSION, "source": source_to_dict(pipeline.source, base), "modifiers": mods,
           "frame": int(frame), "camera": encode(camera) if camera else None, "style": encode(style or {}),
           "render": encode(render or {}), "device": device}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)
    return path


def load_session(path, *, allow_code: bool = False) -> Session:
    """Read a session file. Modifiers that cannot be restored are skipped and listed in
    ``Session.warnings`` (the same messages are issued as Python warnings)."""
    from .pipeline import Pipeline

    path = os.fspath(path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if "ewald_session" not in doc:
        raise ValueError(f"{path} is not an ewald session file")
    if doc["ewald_session"] > FORMAT_VERSION:
        raise ValueError(f"{path} was written by a newer ewald (session format {doc['ewald_session']})")
    base = os.path.dirname(os.path.abspath(path))
    source = source_from_dict(doc["source"], base)
    found = []
    pipe = Pipeline(source)
    for entry in doc.get("modifiers", []):
        m, warning = modifier_from_dict(entry, allow_code)
        if warning:
            found.append(warning)
        if m is not None:
            pipe.modifiers.append(m)
    for w in found:
        warnings.warn(w, stacklevel=2)
    camera = decode(doc.get("camera")) if doc.get("camera") else None
    return Session(pipeline=pipe, frame=int(doc.get("frame", 0)), camera=camera, style=decode(doc.get("style", {})),
                   render=decode(doc.get("render", {})), device=doc.get("device"), warnings=found, path=path)


def is_session_file(path) -> bool:
    if str(path).lower().endswith(".ewald"):
        return True
    try:
        with open(path, encoding="utf-8") as fh:
            return '"ewald_session"' in fh.read(64)
    except (OSError, UnicodeDecodeError):
        return False
