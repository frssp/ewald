"""Selection, deletion and slicing."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from .expression import evaluate


class ExpressionSelection(Modifier):
    """Select particles where ``expression`` is true (sets 'Selection')."""

    def __init__(self, expression: str):
        self.expression = expression

    def modify(self, data, ctx):
        sel = np.broadcast_to(np.asarray(evaluate(self.expression, data, ctx.frame), dtype=bool),
                              (data.count,))
        data.particles["Selection"] = sel.astype(np.int8)
        data.attributes["SelectExpression.num_selected"] = int(sel.sum())


class SelectType(Modifier):
    """Select particles whose 'Particle Type' (or another property) is in ``types``."""

    def __init__(self, types, property: str = "Particle Type"):
        self.types = [types] if isinstance(types, (str, int)) else list(types)
        self.property = property

    def modify(self, data, ctx):
        values = data.particles[self.property]
        sel = np.isin(values.astype(str), [str(t) for t in self.types])
        data.particles["Selection"] = sel.astype(np.int8)
        data.attributes["SelectType.num_selected"] = int(sel.sum())


class InvertSelection(Modifier):
    """Select the unselected particles and unselect the selected ones."""

    def modify(self, data, ctx):
        data.particles["Selection"] = (~data.selection).astype(np.int8)


class ClearSelection(Modifier):
    """Remove the 'Selection' property."""

    def modify(self, data, ctx):
        data.particles.pop("Selection", None)


class DeleteSelected(Modifier):
    """Delete the selected particles."""

    def modify(self, data, ctx):
        sel = data.selection
        out = data.delete(sel)
        out.particles.pop("Selection", None)
        out.attributes["DeleteSelected.num_particles"] = int(sel.sum())
        return out


class Slice(Modifier):
    """Keep (or select) particles in a slab ``|n.r - distance| <= width/2``.

    ``width=0`` keeps one half-space (``n.r >= distance``, or ``<`` if ``inverse``).
    """

    def __init__(self, normal=(0, 0, 1), distance: float = 0.0, width: float = 0.0,
                 inverse: bool = False, select: bool = False, miller: bool = False):
        self.normal = normal
        self.distance = distance
        self.width = width
        self.inverse = inverse
        self.select = select
        self.miller = miller

    def modify(self, data, ctx):
        n = np.asarray(self.normal, float)
        if self.miller:  # (hkl) in reciprocal-lattice coordinates
            n = n @ np.linalg.inv(data.cell.matrix).T
        n = n / np.linalg.norm(n)
        proj = data.positions @ n
        if self.width > 0:
            inside = np.abs(proj - self.distance) <= self.width / 2
        else:
            inside = proj >= self.distance
        remove = inside if self.inverse else ~inside
        if self.select:
            data.particles["Selection"] = remove.astype(np.int8)
            return None
        return data.delete(remove)
