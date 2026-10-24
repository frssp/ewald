"""Particle coloring."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from .expression import evaluate


from ..colormaps import to_rgb as _to_rgb


class AssignColor(Modifier):
    """Give the selected particles (all, if nothing selected) one color."""

    def __init__(self, color=(0.3, 0.3, 1.0), keep_selection: bool = False):
        self.color = color
        self.keep_selection = keep_selection

    def modify(self, data, ctx):
        from .. import color_rule
        from ..render import default_colors

        base = color_rule.rule(data)
        colors = np.array(data.particles.get("Color", default_colors(data)), float)
        selected = "Selection" in data
        sel = data.selection if selected else np.ones(data.count, bool)
        colors[sel] = _to_rgb(self.color)
        data.particles["Color"] = colors
        if selected:
            color_rule.record(data, color_rule.painted(base, "the AssignColor step", "selected",
                                                       self.color, np.count_nonzero(sel)))
        else:
            color_rule.record(data, {"kind": "custom", "source": "the AssignColor step (every particle)",
                                     "colors": [{"color": color_rule._hex(_to_rgb(self.color)), "count": data.count}]})
        if not self.keep_selection:
            data.particles.pop("Selection", None)


class ColorCoding(Modifier):
    """Map a scalar property or expression to colors through a color map.

    ``cmap`` is a matplotlib name or a color map plugin (see :mod:`ewald.colormaps`).
    ``range=(lo, hi)`` is the value range of the color map; None, or None for either end, is the
    min/max over the current frame (OVITO's ``start_value``/``end_value`` and the older
    ``start``/``end`` are accepted too). The range used is stored in attributes so a legend can be
    drawn later.
    """

    ALIASES = {"start_value": "start", "end_value": "end"}   # OVITO's names, as attributes too

    @staticmethod
    def _legacy_arguments(kwargs):
        """``start``/``end`` -> ``range``."""
        if "start" not in kwargs and "end" not in kwargs:
            return
        if "range" in kwargs:
            raise TypeError("ColorCoding() got both 'range' and its old form start/end")
        kwargs["range"] = (kwargs.pop("start", None), kwargs.pop("end", None))

    def __init__(self, property: str, cmap: str = "viridis", range=None, only_selected: bool = False):
        self.property = property
        self.cmap = cmap
        self.range = _value_range(range)
        self.only_selected = only_selected

    @property
    def start(self):
        return None if self.range is None else self.range[0]

    @start.setter
    def start(self, value):
        self.range = (value, self.end)

    @property
    def end(self):
        return None if self.range is None else self.range[1]

    @end.setter
    def end(self, value):
        self.range = (self.start, value)

    def modify(self, data, ctx):
        self.range = _value_range(self.range)
        color_code(data, self.property, self.cmap, self.start, self.end, ctx.frame, self.only_selected)


def _value_range(value):
    """``range`` of :class:`ColorCoding`: None or ``(lo, hi)``, either end None for the frame's."""
    if value is not None and (isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2):
        raise TypeError(f"ColorCoding range is (lo, hi) or None, not {value!r}")
    return None if value is None or tuple(value) == (None, None) else tuple(value)


def color_code(data, property: str, cmap: str = "viridis", start=None, end=None, frame: int = 0,
               only_selected: bool = False, source: str = "the ColorCoding step") -> None:
    """Write 'Color' from ``property`` (or an expression) through ``cmap`` over ``start``..``end``
    (None: the frame's min/max), with the ColorCoding attributes and the color rule."""
    from .. import color_rule
    from ..colormaps import get_colormap
    from ..render import default_colors

    if property in data.particles:
        values = np.asarray(data.particles[property], float)
    else:
        values = np.broadcast_to(np.asarray(evaluate(property, data, frame), float), (data.count,))
    if values.ndim == 2:
        values = np.linalg.norm(values, axis=1)
    auto = values if values.size else np.zeros(1)
    lo = np.nanmin(auto) if start is None else start
    hi = np.nanmax(auto) if end is None else end
    t = np.clip((values - lo) / (hi - lo if hi != lo else 1.0), 0, 1)
    rgb = get_colormap(cmap)(t)[:, :3]
    if only_selected:
        colors = np.array(data.particles.get("Color", default_colors(data)), float)
        sel = data.selection
        colors[sel] = rgb[sel]
        rgb = colors
    data.particles["Color"] = rgb
    data.attributes["ColorCoding.property"] = property
    data.attributes["ColorCoding.cmap"] = cmap
    data.attributes["ColorCoding.range"] = (float(lo), float(hi))
    color_rule.record(data, color_rule.continuous(source, property, cmap, lo, hi, only_selected))
