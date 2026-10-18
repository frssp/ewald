"""Structure analysis modifiers with CPU/GPU backends."""
from __future__ import annotations

import numpy as np

from ..neighbors import find_pairs
from ..pipeline import Modifier


class CoordinationAnalysis(Modifier):
    """Coordination numbers within ``cutoff`` and the radial distribution function.

    Writes 'Coordination' and the table ``tables['coordination-rdf']`` with
    columns ``r`` and ``g(r)`` (plus partial ``g(r) A-B`` columns if ``partial``).
    Runs on the pipeline's device (``compute(device="cuda")``).
    """

    def __init__(self, cutoff: float = 3.2, number_of_bins: int = 200, partial: bool = False):
        self.cutoff = cutoff
        self.number_of_bins = number_of_bins
        self.partial = partial

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        n = data.count
        i, j, _, dist = find_pairs(data.positions, data.cell, self.cutoff, be)
        data.particles["Coordination"] = be.to_numpy(xp.bincount(i, minlength=n)).astype(np.int64)

        edges = np.linspace(0, self.cutoff, self.number_of_bins + 1)
        r = 0.5 * (edges[1:] + edges[:-1])
        shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        volume = data.cell.volume if data.cell.is_valid else np.prod(np.ptp(data.positions, 0) + 1e-9)
        table = {"r": r}
        hist = be.to_numpy(xp.histogram(dist, bins=xp.asarray(edges))[0])
        table["g(r)"] = hist * volume / (max(n, 1) * max(n - 1, 1) * shell) if n > 1 else hist * 0.0
        if self.partial:
            types = data.types
            uniq = list(dict.fromkeys(types))
            codes = xp.asarray(np.searchsorted(np.array(sorted(uniq)), types))
            order = sorted(uniq)
            ti, tj = codes[i], codes[j]
            for a in range(len(order)):
                for b in range(a, len(order)):
                    na = int((types == order[a]).sum()); nb = int((types == order[b]).sum())
                    m = (ti == a) & (tj == b)
                    h = be.to_numpy(xp.histogram(dist[m], bins=xp.asarray(edges))[0])
                    pairs = na * (nb - 1 if a == b else nb)
                    table[f"g(r) {order[a]}-{order[b]}"] = h * volume / (max(pairs, 1) * shell)
        data.tables["coordination-rdf"] = table


class CalculateDisplacements(Modifier):
    """Displacement from a reference frame (minimum-image convention).

    Particles are matched by 'Particle Identifier' when present, otherwise by index.
    Writes 'Displacement' (vector) and 'Displacement Magnitude', and the attribute
    'MSD' (mean squared displacement).
    """

    def __init__(self, reference_frame: int = 0, minimum_image: bool = True):
        self.reference_frame = reference_frame
        self.minimum_image = minimum_image

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        ref = ctx.source_frame(self.reference_frame)
        ref_pos = ref.positions
        if "Particle Identifier" in data and "Particle Identifier" in ref:
            ref_ids = ref["Particle Identifier"]
            order = np.argsort(ref_ids)
            idx = np.searchsorted(ref_ids[order], data["Particle Identifier"])
            idx = np.clip(idx, 0, len(order) - 1)
            if np.any(ref_ids[order][idx] != data["Particle Identifier"]):
                raise ValueError("some particle identifiers are missing in the reference frame")
            ref_pos = ref_pos[order][idx]
        elif ref.count != data.count:
            raise ValueError("particle count differs from reference and no identifiers to match")
        d = xp.asarray(data.positions) - xp.asarray(ref_pos)
        if self.minimum_image and data.cell.is_valid and data.cell.pbc.any():
            # Use the current cell (OVITO default 'off' affine mapping).
            inv = xp.asarray(np.linalg.inv(data.cell.matrix))
            f = d @ inv
            f = f - xp.round(f) * xp.asarray(data.cell.pbc.astype(float))
            d = f @ xp.asarray(data.cell.matrix)
        disp = be.to_numpy(d)
        data.particles["Displacement"] = disp
        mag = np.linalg.norm(disp, axis=1)
        data.particles["Displacement Magnitude"] = mag
        data.attributes["MSD"] = float(np.mean(mag ** 2)) if len(mag) else 0.0
