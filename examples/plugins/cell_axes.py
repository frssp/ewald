"""Example visual element plugins: cell axes and a halo around selected atoms.

``CellAxes(length=2.0)`` draws the cell vectors a, b and c as red, green and blue arrows at the
cell origin; ``SelectionHalo(scale=1.6, opacity=0.35)`` draws a see-through sphere around every
selected particle. Each returns renderer-neutral primitives (:mod:`ewald.scene`), so both show up
in the OpenGL viewport and images, the ray tracer and SVG/PDF output alike. See
docs/plugins/visuals.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and both appear
under the GUI's "+ Visual" button and in the Claude ``add_visual`` tool, or draw them from Python::

    from ewald.render import render
    render(data, "figure.png", visuals=["Cell axes", {"name": "Selection halo", "params": {"scale": 2.0}}])
"""
import numpy as np

from ewald.plugins import visual
from ewald.scene import Spheres, VisualElement, make_arrows


@visual(category="Annotations")
class CellAxes(VisualElement):
    """The cell vectors a, b, c as red, green and blue arrows, ``length`` Angstrom long."""

    name = "Cell axes"

    def __init__(self, length: float = 2.0, width: float = 0.25):
        # every constructor argument is stored under its own name: parameters(), the GUI form,
        # sessions and the Claude tools read them back from here
        self.length = length
        self.width = width

    def applies(self, data):
        return data.cell.is_valid

    def build(self, data, ctx):
        m = data.cell.matrix
        unit = m / np.linalg.norm(m, axis=1, keepdims=True)
        base = np.tile(data.cell.origin, (3, 1))
        rgb = [(0.85, 0.15, 0.15), (0.15, 0.65, 0.2), (0.15, 0.3, 0.9)]
        return make_arrows(base, base + self.length * unit, self.width, rgb, name="cell axes")


@visual(category="Annotations")
class SelectionHalo(VisualElement):
    """A see-through sphere ``scale`` times the atom radius around every selected particle."""

    name = "Selection halo"

    def __init__(self, scale: float = 1.6, opacity: float = 0.35, color: tuple = (1.0, 0.85, 0.1)):
        self.scale = scale
        self.opacity = opacity
        self.color = color

    def applies(self, data):
        return "Selection" in data

    def build(self, data, ctx):
        from ewald.render import default_radii

        picked = np.flatnonzero(data.selection)
        if not len(picked):
            return None
        # never write into data's arrays: they are shared with cached frames
        radii = default_radii(data)[picked] * self.scale
        return Spheres(data.positions[picked], radii, self.color, self.opacity, picked)
