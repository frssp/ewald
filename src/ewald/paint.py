"""Color, size and hide particles by condition, from a notebook cell or a Python step::

    from ewald.paint import paint, hide, resize

    @pipe.add                                   # runs on every frame, like any step
    def surface_style(frame, data):
        cn = data["Coordination"]
        paint(data, cn < 12, "crimson")         # any matplotlib color or RGB
        paint(data, cn < 9, "gold")
        hide(data, data["Position"][:, 0] > 9)  # cut the block open
        resize(data, cn < 12, 1.3)

Each call writes a new "Color", "Radius" or "Transparency" property (never edits arrays in
place), and every renderer and the viewport draw them. ``where`` is a boolean array, an index
array or an expression string as in ``ExpressionSelection`` ("Coordination < 12").
"""
from __future__ import annotations

import numpy as np


def _mask(data, where) -> np.ndarray:
    if isinstance(where, str):
        from .modifiers.expression import evaluate

        frame = data.attributes.get("SourceFrame", 0)
        return np.broadcast_to(np.asarray(evaluate(where, data, frame), dtype=bool), (data.count,)).copy()
    where = np.asarray(where)
    if where.dtype == bool:
        return where
    mask = np.zeros(data.count, bool)
    mask[where] = True
    return mask


def paint(data, where, color, label: str | None = None) -> None:
    """Set the color of the particles ``where`` is true (the others keep theirs). ``label`` names
    them in the color legend ("corner"); without it the legend shows ``where`` itself."""
    from . import color_rule
    from .render import _rgb, default_colors

    base = color_rule.rule(data)
    colors = np.array(default_colors(data), float)
    mask = _mask(data, where)
    colors[mask] = _rgb(color)
    data["Color"] = colors
    what = where if isinstance(where, str) else "a mask in code"
    color_rule.record(data, color_rule.painted(base, "paint()", what, _rgb(color), np.count_nonzero(mask), label))


def hide(data, where) -> None:
    """Hide the particles ``where`` is true (Transparency 1; the others keep theirs)."""
    fade(data, where, 1.0)


def isolate(data, where) -> None:
    """Hide every particle except those ``where`` is true."""
    fade(data, ~_mask(data, where), 1.0)


def fade(data, where, transparency: float = 0.7) -> None:
    """Make the particles ``where`` is true see-through (0 opaque, 1 hidden)."""
    t = np.array(data["Transparency"], float) if "Transparency" in data else np.zeros(data.count)
    t[_mask(data, where)] = transparency
    data["Transparency"] = t


def resize(data, where, factor: float) -> None:
    """Scale the radius of the particles ``where`` is true by ``factor``."""
    from .render import default_radii

    radii = np.array(default_radii(data), float)
    radii[_mask(data, where)] *= factor
    data["Radius"] = radii
