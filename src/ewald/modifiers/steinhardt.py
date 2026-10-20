"""Steinhardt bond-orientational order parameters q_l and w_l (Steinhardt, Nelson, Ronchetti 1983),
optionally neighbor-averaged (Lechner and Dellago 2008). Matches freud's ``freud.order.Steinhardt``
(tests/data/make_analysis_reference.py). The array code runs on the pipeline device.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial, pi, sqrt

import numpy as np

from ..neighbors import find_pairs
from ..pipeline import Modifier
from ._knn import nearest_neighbors


class SteinhardtOrderParameters(Modifier):
    """Per-particle q_l (and w_l) for every degree in ``degrees``.

    Neighbors are the ``num_neighbors`` nearest ones, or all within ``cutoff`` when it is set.
    ``average=True`` uses the neighbor-averaged q_lm of Lechner and Dellago (sharper
    solid/liquid separation). ``wl=True`` also writes w_l, normalized by
    (sum_m |q_lm|^2)^(3/2) unless ``wl_normalize=False``. Writes 'Steinhardt Q4', 'Steinhardt Q6', ...
    ('Steinhardt W6', ...) and the means ``SteinhardtOrderParameters.Q6``. Perfect lattices:
    fcc q4 = 0.191, q6 = 0.575; hcp 0.097, 0.485; bcc (8 neighbors) 0.509, 0.629.
    """

    def __init__(self, degrees=(4, 6), num_neighbors: int = 12, cutoff: float | None = None,
                 average: bool = False, wl: bool = False, wl_normalize: bool = True):
        self.degrees = degrees
        self.num_neighbors = num_neighbors
        self.cutoff = cutoff
        self.average = average
        self.wl = wl
        self.wl_normalize = wl_normalize

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        degrees = [int(l) for l in np.atleast_1d(self.degrees)]
        n = data.count
        if self.cutoff:
            i, j, delta, dist = find_pairs(data.positions, data.cell, float(self.cutoff), be)
            i, j, delta = xp.asarray(i), xp.asarray(j), xp.asarray(delta)
        else:
            idx, delta, dist = nearest_neighbors(data.positions, data.cell, int(self.num_neighbors))
            ok = np.isfinite(dist)
            i = xp.asarray(np.nonzero(ok)[0])
            j = xp.asarray(idx[ok])
            delta = xp.asarray(delta[ok])
        nb = xp.bincount(i, minlength=n).astype(float)
        norm = xp.sqrt((delta * delta).sum(1))
        unit = delta / xp.where(norm > 0, norm, 1.0)[:, None]
        for l in degrees:
            qlm = _qlm(xp, l, unit, i, n) / xp.where(nb > 0, nb, 1.0)[:, None]
            if self.average:
                summed = xp.empty_like(qlm)
                for m in range(l + 1):  # m < 0 follows from q_l,-m = (-1)^m conj(q_lm)
                    col = qlm[j, l + m]
                    summed[:, l + m] = (xp.bincount(i, weights=col.real, minlength=n)
                                        + 1j * xp.bincount(i, weights=col.imag, minlength=n))
                    if m:
                        summed[:, l - m] = (-1) ** m * summed[:, l + m].conj()
                qlm = (qlm + summed) / (nb + 1.0)[:, None]
            power = (qlm * qlm.conj()).real.sum(1)
            q = xp.sqrt(4 * pi / (2 * l + 1) * power)
            q = xp.where(nb > 0, q, 0.0)
            data.particles[f"Steinhardt Q{l}"] = be.to_numpy(q).astype(float)
            data.attributes[f"SteinhardtOrderParameters.Q{l}"] = float(be.to_numpy(q).mean()) if n else 0.0
            if self.wl:
                m1, m2, m3, w3j = (xp.asarray(a) for a in _wigner_terms(l))
                w = (w3j[None, :] * qlm[:, m1] * qlm[:, m2] * qlm[:, m3]).sum(1).real
                if self.wl_normalize:
                    w = w / xp.where(power > 0, power, 1.0) ** 1.5
                w = xp.where(nb > 0, w, 0.0)
                data.particles[f"Steinhardt W{l}"] = be.to_numpy(w).astype(float)


def _qlm(xp, l, unit, i, n):
    """sum over the bonds of each particle of Y_l^m(unit), m = -l..l (Condon-Shortley phase),
    shape (n, 2l+1), complex.

    Y_l^m = K_lm P_l^m(z) e^{im phi}, and P_l^m carries a factor sin^m(theta), so
    P_l^m e^{im phi} = Pbar_l^m(z) (x + iy)^m with the sin-free Pbar: no arctan2, exp or
    division. Only m >= 0 is summed; Y_l^-m = (-1)^m conj(Y_l^m), and so are the sums.
    """
    x, y = unit[:, 0], unit[:, 1]
    z = xp.clip(unit[:, 2], -1.0, 1.0)
    out = xp.empty((n, 2 * l + 1), dtype=complex)
    pmm = xp.ones_like(z)
    cr, ci = xp.ones_like(z), xp.zeros_like(z)  # (x + iy)^m
    for m in range(l + 1):
        if m > 0:
            pmm = -(2 * m - 1) * pmm
            cr, ci = cr * x - ci * y, cr * y + ci * x
        # associated Legendre P_l^m / sin^m by the standard upward recurrence in l
        if m == l:
            p = pmm
        else:
            p_prev, p = pmm, z * (2 * m + 1) * pmm
            for ll in range(m + 2, l + 1):
                p_prev, p = p, ((2 * ll - 1) * z * p - (ll + m - 1) * p_prev) / (ll - m)
        w = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m)) * p
        out[:, l + m] = (xp.bincount(i, weights=w * cr, minlength=n)
                         + 1j * xp.bincount(i, weights=w * ci, minlength=n))
        if m:
            out[:, l - m] = (-1) ** m * out[:, l + m].conj()
    return out


@lru_cache(maxsize=None)
def _wigner_terms(l):
    """Index triples (m1, m2, m3) into m = -l..l with m1 + m2 + m3 = 0 and their 3j symbols."""
    m1s, m2s, m3s, vals = [], [], [], []
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            m3 = -m1 - m2
            if -l <= m3 <= l:
                m1s.append(m1 + l)
                m2s.append(m2 + l)
                m3s.append(m3 + l)
                vals.append(_wigner3j(l, l, l, m1, m2, m3))
    return (np.array(m1s), np.array(m2s), np.array(m3s), np.array(vals))


def _wigner3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol by the Racah formula (integer arguments)."""
    f = factorial
    tri = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    pre = sqrt(tri * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    total = 0.0
    for t in range(max(0, j2 - j3 - m1, j1 - j3 + m2), min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1):
        total += (-1) ** t / (f(t) * f(j3 - j2 + t + m1) * f(j3 - j1 + t - m2) * f(j1 + j2 - j3 - t)
                              * f(j1 - t - m1) * f(j2 - t + m2))
    return (-1) ** (j1 - j2 - m3) * pre * total
