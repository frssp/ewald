"""Geometric transformations and derived properties."""
from __future__ import annotations

import numpy as np

from ..data import Cell
from ..pipeline import Modifier
from .expression import evaluate


class ComputeProperty(Modifier):
    """Set ``output`` to an expression, e.g. ``ComputeProperty("Height", "Position.Z - 5")``.

    For vector outputs pass a list of expressions, one per component.
    """

    def __init__(self, output: str, expressions, only_selected: bool = False):
        self.output = output
        self.expressions = expressions
        self.only_selected = only_selected

    def modify(self, data, ctx):
        exprs = [self.expressions] if isinstance(self.expressions, str) else list(self.expressions)
        cols = [np.broadcast_to(np.asarray(evaluate(e, data, ctx.frame)), (data.count,)) for e in exprs]
        values = cols[0] if len(cols) == 1 else np.stack(cols, axis=1)
        if self.only_selected and self.output in data.particles:
            old = np.array(data.particles[self.output])
            sel = data.selection
            old[sel] = values[sel]
            values = old
        data.particles[self.output] = np.array(values)


class WrapAtoms(Modifier):
    """Map particles back into the periodic cell."""

    def modify(self, data, ctx):
        data.positions = data.cell.wrap(data.positions)


class AffineTransformation(Modifier):
    """Apply a 3x3 (or 3x4 with translation) matrix to particles and, optionally, the cell."""

    def __init__(self, matrix, transform_cell: bool = True):
        self.matrix = matrix

        self.transform_cell = transform_cell

    def modify(self, data, ctx):
        m = np.asarray(self.matrix, float)
        lin, t = (m[:, :3], m[:, 3]) if m.shape == (3, 4) else (m, np.zeros(3))
        data.positions = data.positions @ lin.T + t
        if self.transform_cell:
            data.cell = Cell(data.cell.matrix @ lin.T, data.cell.origin @ lin.T + t, data.cell.pbc)


class Replicate(Modifier):
    """Tile the periodic cell ``n = (na, nb, nc)`` times."""

    def __init__(self, n=(2, 2, 2), adjust_box: bool = True, unique_ids: bool = True):
        self.n = n
        self.adjust_box = adjust_box
        self.unique_ids = unique_ids

    def modify(self, data, ctx):
        na, nb, nc = self.n
        shifts = np.array([(i, j, k) for i in range(na) for j in range(nb) for k in range(nc)], float)
        shifts = shifts @ data.cell.matrix
        count = data.count
        out = data.take(np.tile(np.arange(count), len(shifts)))
        out.positions = (data.positions[None] + shifts[:, None]).reshape(-1, 3)
        if self.unique_ids and "Particle Identifier" in data:
            ids = data["Particle Identifier"]
            span = int(ids.max()) if count else 0
            out.particles["Particle Identifier"] = (ids[None] + span * np.arange(len(shifts))[:, None]).ravel()
        if self.adjust_box:
            out.cell = Cell(data.cell.matrix * np.array([[na], [nb], [nc]]), data.cell.origin, data.cell.pbc)
        return out
