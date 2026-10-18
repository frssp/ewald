"""Particle coloring."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from .expression import evaluate


def _to_rgb(color) -> np.ndarray:
    if isinstance(color, str):
        from matplotlib.colors import to_rgb

        return np.array(to_rgb(color))
    return np.asarray(color, float)[:3]


class AssignColor(Modifier):
    """Give the selected particles (all, if nothing selected) one color."""

    def __init__(self, color=(0.3, 0.3, 1.0), keep_selection: bool = False):
        self.color = color
        self.keep_selection = keep_selection

    def modify(self, data, ctx):
        from ..render import default_colors

        colors = np.array(data.particles.get("Color", default_colors(data)), float)
        sel = data.selection if "Selection" in data else np.ones(data.count, bool)
        colors[sel] = _to_rgb(self.color)
        data.particles["Color"] = colors
        if not self.keep_selection:
            data.particles.pop("Selection", None)


class ColorCoding(Modifier):
    """Map a scalar property or expression to colors through a matplotlib colormap.

    ``start``/``end`` default to the min/max over the current frame. The range used
    is stored in attributes so a legend can be drawn later.
    """

    def __init__(self, property: str, cmap: str = "viridis", start=None, end=None,
                 only_selected: bool = False):
        self.property = property
        self.cmap = cmap
        self.start = start
        self.end = end
        self.only_selected = only_selected

    def modify(self, data, ctx):
        import matplotlib

        from ..render import default_colors

        if self.property in data.particles:
            values = np.asarray(data.particles[self.property], float)
        else:
            values = np.broadcast_to(np.asarray(evaluate(self.property, data, ctx.frame), float),
                                     (data.count,))
        if values.ndim == 2:
            values = np.linalg.norm(values, axis=1)
        auto = values if values.size else np.zeros(1)
        lo = np.nanmin(auto) if self.start is None else self.start
        hi = np.nanmax(auto) if self.end is None else self.end
        t = np.clip((values - lo) / (hi - lo if hi != lo else 1.0), 0, 1)
        rgb = matplotlib.colormaps[self.cmap](t)[:, :3]
        if self.only_selected:
            colors = np.array(data.particles.get("Color", default_colors(data)), float)
            sel = data.selection
            colors[sel] = rgb[sel]
            rgb = colors
        data.particles["Color"] = rgb
        data.attributes["ColorCoding.property"] = self.property
        data.attributes["ColorCoding.cmap"] = self.cmap
        data.attributes["ColorCoding.range"] = (float(lo), float(hi))
