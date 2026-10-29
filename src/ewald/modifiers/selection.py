"""Selection, deletion and slicing."""
from __future__ import annotations

import warnings

import numpy as np

from ..pipeline import Modifier
from .expression import evaluate


class ExpressionSelection(Modifier):
    """Select particles where ``expression`` is true (sets 'Selection').

    ``top=n`` with ``by`` (a property or an expression) keeps only the ``n`` of them with the
    largest ``by`` (``top=-n``: the smallest), e.g. the five Li atoms that move most:
    ``ExpressionSelection("ParticleType == 'Li'", top=5, by="MSD")`` after a
    MeanSquaredDisplacement step (its per-atom 'MSD'), then ``GenerateTrajectoryLines()`` draws
    their paths. ``expression="True"`` ranks every particle.
    """

    def __init__(self, expression: str, top: int = 0, by: str = ""):
        self.expression = expression
        self.top = top
        self.by = by

    def modify(self, data, ctx):
        sel = np.broadcast_to(np.asarray(evaluate(self.expression, data, ctx.frame), dtype=bool),
                              (data.count,))
        top = int(getattr(self, "top", 0) or 0)
        if top:
            by = getattr(self, "by", "")
            if not by:
                raise ValueError("ExpressionSelection: top needs by, the property to rank by (e.g. by='MSD')")
            values = np.broadcast_to(np.asarray(evaluate(by, data, ctx.frame), dtype=float), (data.count,))
            candidates = np.nonzero(sel)[0]
            order = np.argsort(-values[candidates] if top > 0 else values[candidates], kind="stable")
            sel = np.zeros(data.count, dtype=bool)
            sel[candidates[order[:abs(top)]]] = True
        data.particles["Selection"] = sel.astype(np.int8)
        data.attributes["SelectExpression.num_selected"] = int(sel.sum())


class SelectType(Modifier):
    """Select particles whose 'Particle Type' (or another property) is in ``types``."""

    def __init__(self, types, property: str = "Particle Type"):
        self.types = [types] if isinstance(types, (str, int)) else list(types)
        self.property = property

    def modify(self, data, ctx):
        values = data.particles[self.property]
        names = values.astype(str)
        sel = np.isin(names, [str(t) for t in self.types])
        present = np.unique(names).tolist()
        missing = sorted({str(t) for t in self.types} - set(present))
        if missing:
            warnings.warn(f"SelectType: no particles of type {', '.join(missing)} "
                          f"({self.property!r} has {', '.join(present[:12])})", stacklevel=2)
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


class ExpandSelection(Modifier):
    """Grow the selection to the neighbors of selected particles (OVITO's ExpandSelectionModifier).

    ``mode``: 'cutoff' adds every particle within ``cutoff`` of a selected one, 'nearest' the
    ``num_neighbors`` nearest neighbors of each selected particle, 'bonded' the particles bonded
    to a selected one (needs bonds, e.g. CreateBonds). The step is repeated ``iterations``
    times, so ``iterations=2`` reaches second neighbors. Writes 'Selection' and the attributes
    ``ExpandSelection.num_added`` and ``ExpandSelection.num_selected``.
    """

    choice_parameters = {"mode": ["cutoff", "nearest", "bonded"]}

    def __init__(self, mode: str = "cutoff", cutoff: float = 3.2, num_neighbors: int = 1, iterations: int = 1):
        self.mode = mode
        self.cutoff = cutoff
        self.num_neighbors = num_neighbors
        self.iterations = iterations

    def _pairs(self, data, ctx):
        """Directed pairs (i, j): selecting i selects j."""
        mode = self.mode.lower()
        if mode == "cutoff":
            from ..neighbors import find_pairs

            i, j, _, _ = find_pairs(data.positions, data.cell, self.cutoff, ctx.backend)
            return ctx.backend.to_numpy(i), ctx.backend.to_numpy(j)
        if mode == "nearest":
            from ._knn import nearest_neighbors

            k = min(int(self.num_neighbors), data.count - 1)
            if k < 1:
                return np.zeros(0, np.int64), np.zeros(0, np.int64)
            index, _, _ = nearest_neighbors(data.positions, data.cell, k)
            i = np.repeat(np.arange(data.count), k)
            j = np.asarray(index).ravel()
            ok = j >= 0
            return i[ok], j[ok]
        if mode == "bonded":
            if data.bonds is None:
                raise ValueError("ExpandSelection(mode='bonded') needs bonds; add CreateBonds first")
            t = data.bonds.topology
            return np.concatenate([t[:, 0], t[:, 1]]), np.concatenate([t[:, 1], t[:, 0]])
        raise ValueError(f"unknown mode {self.mode!r}; use 'cutoff', 'nearest' or 'bonded'")

    def modify(self, data, ctx):
        sel = data.selection.copy()
        before = int(sel.sum())
        if before and data.count:
            i, j = self._pairs(data, ctx)
            for _ in range(max(int(self.iterations), 0)):
                new = sel.copy()
                new[j[sel[i]]] = True
                if (new == sel).all():
                    break
                sel = new
        data.particles["Selection"] = sel.astype(np.int8)
        data.attributes["ExpandSelection.num_added"] = int(sel.sum()) - before
        data.attributes["ExpandSelection.num_selected"] = int(sel.sum())
