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
    """Map particles back into the periodic cell (bond periodic images follow)."""

    def modify(self, data, ctx):
        old = data.positions
        data.positions = data.cell.wrap(old)
        if data.bonds is not None and data.bonds.count and data.cell.is_valid:
            shift = np.rint(data.cell.to_fractional(data.positions) - data.cell.to_fractional(old)).astype(np.int32)
            i, j = data.bonds.topology.T
            data.bonds = data.bonds.copy()
            data.bonds.properties["Periodic Image"] = data.bonds.pbc_images - shift[j] + shift[i]


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
        if data.bonds is not None:
            out.bonds = _replicate_bonds(data.bonds, (na, nb, nc), count)
        if self.adjust_box:
            out.cell = Cell(data.cell.matrix * np.array([[na], [nb], [nc]]), data.cell.origin, data.cell.pbc)
        return out


def _replicate_bonds(bonds, n, count):
    """Bonds of a tiled cell: copy c of bond (i, j, image) ends in copy (c + image) mod n of j."""
    from ..bonds import Bonds

    n = np.asarray(n)
    copies = np.array([(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])
    flat = lambda c: (c[..., 0] * n[1] + c[..., 1]) * n[2] + c[..., 2]  # noqa: E731
    target = copies[:, None, :] + bonds.pbc_images[None]                  # (copies, M, 3)
    wrapped = target % n
    i, j = bonds.topology.T
    new_i = (flat(copies)[:, None] * count + i[None]).ravel()
    new_j = (flat(wrapped) * count + j[None]).ravel()
    out = Bonds.__new__(Bonds)
    out.properties = {k: np.tile(v, (len(copies),) + (1,) * (v.ndim - 1)) for k, v in bonds.properties.items()}
    out.properties["Topology"] = np.column_stack([new_i, new_j])
    out.properties["Periodic Image"] = ((target - wrapped) // n).reshape(-1, 3).astype(np.int32)
    return out
