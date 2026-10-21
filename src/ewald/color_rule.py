"""What colors the atoms: the rule behind a 'Color' property, for legends, the View tab and the AI.

A step that writes ``data.particles["Color"]`` records how it chose the colors with :func:`record`
(structure identification: one color per structure type; ColorCoding: a color map over a property;
``paint``/AssignColor: a color where a condition holds). :func:`rule` reads it back, or says that
the colors came from code that recorded nothing, or that atoms have their type colors::

    from ewald import color_rule
    color_rule.rule(pipe.compute())      # {'kind': 'categorical', 'property': 'Structure Type', ...}
    print(color_rule.describe(pipe.compute()))

The record carries a fingerprint of the colors, so a later step that writes other colors without
recording a rule is noticed (the rule is then ``'custom'``). :func:`apply` is the view's
``color_by`` (``view(pipe, color_by="Coordination")``): a display copy colored by a property, or
by particle type, whatever the steps wrote.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

KEY = "Color.rule"          # data.attributes key of the recorded rule
MAX_ENTRIES = 16            # legend rows; more categories (grains, clusters) are summarized
_ACRONYMS = {"FCC", "HCP", "BCC", "ICO", "SC", "BCT", "ISF", "L10", "L12", "B2", "A", "B"}
_CUSTOM_COLORS = 12         # distinct colors listed for a 'custom' rule


def fingerprint(colors) -> str:
    """A short hash of a color array: its shape and up to 4096 evenly spaced rows."""
    a = np.asarray(colors)
    step = max(1, len(a) // 4096)
    sample = np.ascontiguousarray(a[::step], dtype=np.float32)
    h = hashlib.blake2b(repr(a.shape).encode(), digest_size=8)
    h.update(sample.tobytes())
    return h.hexdigest()


def _hex(rgb) -> str:
    from matplotlib.colors import to_hex

    return to_hex(np.clip(np.asarray(rgb, float)[:3], 0, 1)) if not isinstance(rgb, str) else to_hex(rgb)


def record(data, rule: dict) -> None:
    """Store ``rule`` as the way the current 'Color' property was made (call it right after writing it)."""
    rule = {k: v for k, v in rule.items() if k != "fingerprint"}
    rule["fingerprint"] = fingerprint(data.particles["Color"])
    data.attributes[KEY] = rule


def categorical(source: str, property: str, names, colors, note: str | None = None) -> dict:
    """The rule of a step that gives each value 0, 1, 2, ... of ``property`` the color ``colors[value]``."""
    entries = [{"value": i, "name": str(n), "color": _hex(c)} for i, (n, c) in enumerate(zip(names, colors))]
    out = {"kind": "categorical", "source": source, "property": property, "entries": entries}
    if note:
        out["note"] = note
    return out


def continuous(source: str, property: str, cmap: str, lo: float, hi: float, only_selected: bool = False) -> dict:
    out = {"kind": "continuous", "source": source, "property": property, "cmap": cmap,
           "range": [float(lo), float(hi)]}
    if only_selected:
        out["note"] = "only the selected particles; the others keep their colors"
    return out


def painted(base: dict, source: str, where: str, color, count: int) -> dict:
    """``base`` with one more override: ``color`` on the ``count`` particles where ``where`` holds."""
    out = {k: copy.deepcopy(v) for k, v in base.items() if k != "fingerprint"}
    out["painted"] = list(out.get("painted", [])) + [{"source": source, "where": str(where),
                                                      "color": _hex(color), "count": int(count)}]
    return out


def type_rule(data, type_colors: dict | None = None) -> dict:
    """Atoms in their particle type colors (element colors, the theme's or ``type_colors``)."""
    from .render import _rgb, type_color

    entries = []
    if data is not None and data.count:
        types = np.asarray(data.types).astype(str)
        for k, t in enumerate(dict.fromkeys(types.tolist())):
            c = (type_colors or {}).get(t)
            entries.append({"value": t, "name": t, "color": _hex(_rgb(c) if c is not None else type_color(t, k)),
                            "count": int((types == t).sum())})
    return {"kind": "type", "source": "particle type", "property": "Particle Type", "entries": entries}


def rule(data, type_colors: dict | None = None) -> dict:
    """How the atoms of ``data`` are colored now (see the module docstring); categorical entries get
    the number of atoms in each category (``count``)."""
    if data is None or "Color" not in data.particles:
        return type_rule(data, type_colors)
    colors = data.particles["Color"]
    stored = data.attributes.get(KEY)
    if isinstance(stored, dict) and stored.get("fingerprint") == fingerprint(colors):
        out = {k: copy.deepcopy(v) for k, v in stored.items() if k != "fingerprint"}
        prop = out.get("property")
        if out.get("kind") == "categorical" and out.get("entries") and prop in data.particles:
            values = np.asarray(data.particles[prop])
            if values.ndim == 1 and np.issubdtype(values.dtype, np.integer):
                counts = np.bincount(np.clip(values, 0, None), minlength=len(out["entries"]))
                for e in out["entries"]:
                    e["count"] = int(counts[e["value"]]) if e["value"] < len(counts) else 0
        return out
    out = {"kind": "custom", "source": "a 'Color' property set by code that recorded no rule "
                                       "(e.g. data['Color'] = ... in a Python step, or a plugin)"}
    if isinstance(stored, dict):
        out["overwrote"] = stored.get("source")
    rgb = np.round(np.asarray(colors, float)[:, :3], 3)
    if len(rgb):
        uniq, counts = np.unique(rgb, axis=0, return_counts=True)
        if len(uniq) <= _CUSTOM_COLORS:
            order = np.argsort(-counts)
            out["colors"] = [{"color": _hex(uniq[i]), "count": int(counts[i])} for i in order]
        else:
            out["distinct_colors"] = int(len(uniq))
    return out


def legend_entries(r: dict) -> list[tuple[str, np.ndarray]]:
    """(label, RGB) rows of a categorical legend for rule ``r``: the categories that have atoms, then
    the painted overrides; empty for type colors, color maps (a color bar) and custom colors."""
    from .render import _rgb

    if r.get("kind") != "categorical" or not r.get("entries"):
        rows = []
    else:
        rows = [(label(e["name"]), _rgb(e["color"])) for e in r["entries"] if e.get("count", 1)]
    rows += [(p["where"], _rgb(p["color"])) for p in r.get("painted", ()) if p.get("count", 1)]
    return rows[:MAX_ENTRIES]


def label(name: str) -> str:
    """A legend label for an OVITO type name: 'CUBIC_DIAMOND_FIRST_NEIGHBOR' -> 'Cubic diamond (1st
    neighbor)'; acronyms (FCC, BCC, ICO) and mixed-case names stay as they are."""
    name = str(name)
    if not name.isupper():
        return name
    name = name.replace("_FIRST_NEIGHBOR", " (1st neighbor)").replace("_SECOND_NEIGHBOR", " (2nd neighbor)")
    words = name.replace("_", " ").split(" ")
    words = [w if w in _ACRONYMS or not w.isupper() else (w.capitalize() if i == 0 else w.lower())
             for i, w in enumerate(words)]
    return " ".join(words)


def legend_title(r: dict) -> str:
    return str(r.get("property") or "Color")


def describe(r_or_data, type_colors: dict | None = None) -> str:
    """One paragraph saying how the atoms are colored (for the AI and tooltips)."""
    r = r_or_data if isinstance(r_or_data, dict) else rule(r_or_data, type_colors)
    kind = r.get("kind")
    if kind == "type":
        text = "Atoms have their particle type colors: " + ", ".join(
            f"{e['name']} {e['color']}" for e in r.get("entries", [])[:MAX_ENTRIES]) + "."
    elif kind == "categorical":
        entries = r.get("entries")
        if entries:
            parts = [f"{e['value']} {e['name']} = {e['color']}" + (f" ({e['count']} atoms)" if "count" in e else "")
                     for e in entries[:MAX_ENTRIES]]
            text = f"Atoms are colored by '{r['property']}' (from {r['source']}): " + "; ".join(parts) + "."
        else:
            text = f"Atoms are colored by '{r['property']}' (from {r['source']})."
    elif kind == "continuous":
        lo, hi = r["range"]
        text = (f"Atoms are colored by '{r['property']}' with the '{r['cmap']}' color map from {lo:.4g} "
                f"(low end) to {hi:.4g} (high end) (from {r['source']}).")
    else:
        text = f"Atoms are colored by {r.get('source', 'code')}"
        if r.get("overwrote"):
            text += f", over the colors of {r['overwrote']}"
        if r.get("colors"):
            text += ": " + ", ".join(f"{c['color']} ({c['count']} atoms)" for c in r["colors"])
        elif r.get("distinct_colors"):
            text += f": {r['distinct_colors']} distinct colors"
        text += "."
    if r.get("note"):
        text += f" Note: {r['note']}."
    for p in r.get("painted", ()):
        text += f" Then {p['source']} painted {p['count']} atoms where {p['where']} {p['color']}."
    return text


# ------------------------------------------------------------------ the view's color_by
TYPE = "type"


def apply(data, color_by=None, colormap: str = "viridis", color_range=None, frame: int = 0):
    """A display copy of ``data`` colored as the view says: ``color_by`` None keeps the steps' colors,
    ``'type'`` drops them (particle type colors), a property name or expression colors the atoms with
    ``colormap`` over ``color_range`` ``[lo, hi]`` (either may be None: the frame's min/max)."""
    if color_by is None or data is None or not data.count:
        return data
    out = data.copy()
    for k in ("ColorCoding.property", "ColorCoding.cmap", "ColorCoding.range", KEY):
        out.attributes.pop(k, None)
    if color_by == TYPE:
        out.particles.pop("Color", None)
        return out
    from .modifiers.coloring import color_code

    lo, hi = (list(color_range) + [None, None])[:2] if color_range is not None else (None, None)
    color_code(out, str(color_by), colormap or "viridis", lo, hi, frame, source="the view (color_by)")
    return out


def take(data, kwargs: dict, frame: int = 0):
    """Pop ``color_by``/``colormap``/``color_range`` from scene keyword arguments (a view style's
    ``scene_kwargs()``) and return ``data`` colored by them (see :func:`apply`)."""
    color_by, colormap, color_range = (kwargs.pop(k, None) for k in ("color_by", "colormap", "color_range"))
    return apply(data, color_by, colormap or "viridis", color_range, frame)
