"""Per-particle vectors drawn as arrows (OVITO's VectorVis visual element, as a modifier)."""
from __future__ import annotations

from ..pipeline import Modifier


class VectorVis(Modifier):
    """Draw an N x 3 particle property as arrows: 'Force', 'Displacement', 'Velocity',
    'Magnetic Moment' (a scalar moment points along z), or any other vector property.

    ``property=""`` picks the first of Force, Displacement, Velocity, Magnetic Moment.
    ``scaling`` (A per unit) and ``width`` (shaft diameter, A) set the size; ``alignment``
    puts the arrow's 'base', 'center' or 'head' on the atom; ``reverse`` flips it and
    ``offset`` shifts it. ``color`` is used unless ``color_mapping_property`` ("Magnitude"
    for the vector length, another property, or an expression such as "Force.Z") maps
    values through ``cmap`` over ``color_mapping_interval`` (None: this frame's range, shown
    as the color legend when no ColorCoding is present). ``shading``: 'normal' (lit 3D) or 'flat'. Arrows shorter
    than ``min_length`` (A) are skipped. Writes ``data.vectors[property]``; all renderers
    (viewport, ray tracer, SVG/PDF) draw it. Add several VectorVis for several properties.
    """

    def __init__(self, property: str = "", scaling: float = 1.0, width: float = 0.3, alignment: str = "base",
                 reverse: bool = False, color=(1.0, 1.0, 0.0), shading: str = "normal", transparency: float = 0.0,
                 offset=(0.0, 0.0, 0.0), color_mapping_property: str = "", cmap: str = "viridis",
                 color_mapping_interval=None, only_selected: bool = False, min_length: float = 1e-6):
        self.property = property
        self.scaling = scaling
        self.width = width
        self.alignment = alignment
        self.reverse = reverse
        self.color = color
        self.shading = shading
        self.transparency = transparency
        self.offset = offset
        self.color_mapping_property = color_mapping_property
        self.cmap = cmap
        self.color_mapping_interval = color_mapping_interval
        self.only_selected = only_selected
        self.min_length = min_length

    def style(self, data):
        from ..arrows import VectorStyle, auto_property, property_vectors

        name = self.property or auto_property(data)
        property_vectors(data, name)  # clear error now rather than at draw time
        return VectorStyle(property=name, scaling=float(self.scaling), width=float(self.width),
                           alignment=str(self.alignment).lower(), reverse=bool(self.reverse), color=self.color,
                           shading=str(self.shading).lower(), transparency=float(self.transparency),
                           offset=tuple(self.offset), color_mapping_property=self.color_mapping_property or None,
                           cmap=self.cmap, only_selected=bool(self.only_selected), min_length=float(self.min_length),
                           color_mapping_interval=None if self.color_mapping_interval is None
                           else tuple(map(float, self.color_mapping_interval)))

    def modify(self, data, ctx):
        import numpy as np

        from ..arrows import mapping_values

        style = self.style(data)
        if style.color_mapping_property and data.count:
            if style.color_mapping_interval is None:  # fix the range now so arrows and legend agree
                v = mapping_values(data, style.color_mapping_property, ctx.frame)
                style.color_mapping_interval = (float(np.nanmin(v)), float(np.nanmax(v)))
            if "ColorCoding.range" not in data.attributes:
                data.attributes["ColorCoding.property"] = style.color_mapping_property
                data.attributes["ColorCoding.cmap"] = style.cmap
                data.attributes["ColorCoding.range"] = style.color_mapping_interval
        data.vectors = {**data.vectors, style.property: style}
