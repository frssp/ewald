"""Example modifier plugin: local chemical composition around every particle.

``LocalComposition(cutoff=3.0, element="Cu")`` adds the property "Local Cu Fraction" (the
fraction of neighbors within ``cutoff`` that are Cu) and the attribute
"LocalComposition.mean" (its average), which is the per-atom input of a Warren-Cowley
short-range order analysis. It is written against ``ctx.xp``, so it runs on NumPy and, with
``device="cuda"``, on CuPy. See docs/plugins/modifiers.md.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and it shows
up in the GUI's Add modification › Analysis menu and in the Claude tools, and imports like a
built-in modifier::

    from ewald.modifiers import LocalComposition
    pipe.modifiers.append(LocalComposition(cutoff=3.0, element="Cu"))
"""
import numpy as np

from ewald import Modifier
from ewald.neighbors import find_pairs
from ewald.plugins import modifier


@modifier(category="Analysis")
class LocalComposition(Modifier):
    """Fraction of neighbors within ``cutoff`` that are ``element``."""

    def __init__(self, cutoff: float = 3.0, element: str = "Cu"):
        # every constructor argument is stored under its own name: parameters(), the GUI form,
        # repr() (which keys caches) and the generated scripts all read them back from here
        self.cutoff = cutoff
        self.element = element

    def modify(self, data, ctx):
        xp, be = ctx.xp, ctx.backend
        i, j, _delta, _dist = find_pairs(data.positions, data.cell, self.cutoff, backend=be)
        is_element = xp.asarray(data.types == self.element, dtype=xp.float64)
        total = xp.bincount(i, minlength=data.count).astype(xp.float64)
        same = xp.bincount(i, weights=is_element[j], minlength=data.count)
        fraction = be.to_numpy(xp.where(total > 0, same / xp.maximum(total, 1), 0.0))
        data[f"Local {self.element} Fraction"] = fraction       # a new array, never written in place
        data.attributes["LocalComposition.mean"] = float(np.mean(fraction))
