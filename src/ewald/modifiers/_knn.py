"""k-nearest-neighbor queries under periodic boundary conditions (triclinic cells).

Used by the structure-identification modifiers.  Periodic cells use a numba
cell list searched in growing shells of bins until the k-th neighbor is provably
found (no image copies, one parallel pass that writes index, delta and distance).
Non-periodic data, or an install without numba, use a scipy cKDTree; for periodic
cells that fallback adds images within a skin layer that grows until every
requested neighbor is guaranteed to be found.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from ..data import Cell

try:
    import numba
except ImportError:  # pragma: no cover - numba is a core dependency
    numba = None


class PeriodicKDTree:
    """Nearest-neighbor search among ``points`` in ``cell``.

    Query results are minimum-image-correct for any cell size: tiny cells simply
    get several layers of images.
    """

    def __init__(self, points, cell: Cell):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.cell = cell
        self.periodic = bool(cell.is_valid and cell.pbc.any())
        if self.periodic:
            self._inv = np.linalg.inv(cell.matrix)
            frac = (self.points - cell.origin) @ self._inv
            frac[:, cell.pbc] %= 1.0
            self._frac = frac
            self._wrapped = frac @ cell.matrix + cell.origin
            self._widths = cell.perpendicular_widths()
        self._skin = -1.0
        self._tree = None
        self._image_index = None
        self._bins = None

    def _build(self, skin: float):
        if not self.periodic:
            self._tree = cKDTree(self.points)
            self._image_index = np.arange(len(self.points))
            self._image_pos = self.points
            self._skin = np.inf
            return
        pbc = self.cell.pbc
        reach = np.where(pbc, np.ceil(skin / self._widths).astype(int), 0)
        skin_f = np.where(pbc, skin / self._widths, 0.0)
        pos, idx = [], []
        for shift in itertools.product(*(range(-r, r + 1) for r in reach)):
            shift = np.array(shift, dtype=float)
            f = self._frac + shift
            if np.any(shift != 0):
                # only periodic directions limit the skin: atoms may sit outside the cell elsewhere
                m = np.all(((f > -skin_f) & (f < 1 + skin_f))[:, pbc], axis=1)
            else:
                m = np.ones(len(f), dtype=bool)
            pos.append(f[m] @ self.cell.matrix + self.cell.origin)
            idx.append(np.nonzero(m)[0])
        self._image_pos = np.concatenate(pos)
        self._image_index = np.concatenate(idx)
        self._tree = cKDTree(self._image_pos)
        self._skin = skin

    def _wrap_queries(self, x):
        if not self.periodic:
            return x
        f = (x - self.cell.origin) @ self._inv
        f[:, self.cell.pbc] %= 1.0
        return f @ self.cell.matrix + self.cell.origin

    def query(self, x, k: int, exclude_self: bool = False):
        """k nearest points to each row of ``x``.

        Returns ``(index, delta, dist)`` with shapes (M, k), (M, k, 3), (M, k);
        ``delta`` points from the query to the neighbor image.  Missing neighbors
        (fewer than k points in total) get index -1 and distance inf.
        If ``exclude_self``, ``x`` must be ``self.points`` and each point's own
        zero-distance entry is dropped.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        kk = k + 1 if exclude_self else k
        xq = self._wrap_queries(x)
        if self.periodic and numba is not None and len(self.points):
            index, delta, dist = self._query_bins(xq, kk)
            if exclude_self:
                index, delta, dist = index[:, 1:], delta[:, 1:], dist[:, 1:]
            return index, delta, dist
        if self._tree is None:
            n = max(len(self.points), 1)
            vol = self.cell.volume if self.cell.is_valid else np.prod(np.ptp(self.points, 0) + 1.0)
            self._build(2.0 * (3.0 * vol * kk / (4.0 * np.pi * n)) ** (1 / 3) + 1e-6)
        while True:
            kq = min(kk, len(self._image_pos))
            dist, img = self._tree.query(xq, k=kq, workers=-1)
            dist = dist.reshape(len(x), kq)
            img = img.reshape(len(x), kq)
            if self._skin == np.inf or (np.isfinite(dist[:, -1]).all() and dist[:, -1].max() <= self._skin
                                        and kq == kk):
                break
            self._build(self._skin * 1.6)
        if kq < kk:
            pad = kk - kq
            dist = np.concatenate([dist, np.full((len(x), pad), np.inf)], 1)
            img = np.concatenate([img, np.full((len(x), pad), -1)], 1)
        if exclude_self:
            img, dist = img[:, 1:], dist[:, 1:]
        valid = img >= 0
        if valid.all():  # the usual case: plain gathers, no masked temporaries
            index = self._image_index[img]
            delta = self._image_pos[img]
            delta -= xq[:, None, :]
            return index, delta, dist
        index = np.where(valid, self._image_index[np.where(valid, img, 0)], -1)
        delta = np.where(valid[..., None], self._image_pos[np.where(valid, img, 0)] - xq[:, None, :], 0.0)
        return index, delta, dist


    def _setup_bins(self, xq, kk: int):
        """Bin the wrapped points; bins are about half the expected k-th neighbor distance."""
        cell = self.cell
        pbc = np.asarray(cell.pbc, bool)
        fq = (xq - cell.origin) @ self._inv
        lo = np.zeros(3)
        span = np.ones(3)
        for d in range(3):  # non-periodic directions: bin the occupied range
            if not pbc[d]:
                v = np.concatenate([self._frac[:, d], fq[:, d]])
                lo[d] = v.min()
                span[d] = max(v.max() - lo[d], 1e-9)
        widths = self._widths * span
        n = len(self.points)
        vol = float(np.prod(widths))
        r_k = (3.0 * vol * max(kk, 1) / (4.0 * np.pi * n)) ** (1 / 3)
        nbins = np.maximum(1, np.floor(widths / (0.5 * r_k))).astype(np.int64)
        while nbins.prod() > 4 * n + 64:  # keep the grid small for sparse or flat data
            nbins = np.maximum(1, nbins // 2)
        f = self._frac - lo
        b = np.minimum(np.maximum(np.floor(f / span * nbins).astype(np.int64), 0), nbins - 1)
        cid = (b[:, 0] * nbins[1] + b[:, 1]) * nbins[2] + b[:, 2]
        order = np.argsort(cid, kind="stable")
        starts = np.zeros(nbins.prod() + 1, np.int64)
        np.cumsum(np.bincount(cid, minlength=nbins.prod()), out=starts[1:])
        self._bins = (lo, span, nbins, widths / nbins, starts, order.astype(np.int64),
                      np.ascontiguousarray(self._wrapped[order]))

    def _query_bins(self, xq, kk: int):
        if self._bins is None:
            self._setup_bins(xq, kk)
        lo, span, nbins, h, starts, order, psorted = self._bins
        fq = (xq - self.cell.origin) @ self._inv
        m = len(xq)
        index = np.empty((m, kk), np.int64)
        delta = np.empty((m, kk, 3))
        dist = np.empty((m, kk))
        _knn_cells(np.ascontiguousarray(fq), np.ascontiguousarray(xq), psorted, order, starts,
                   np.ascontiguousarray(self.cell.matrix, dtype=float), np.asarray(self.cell.pbc, np.bool_),
                   lo, span, nbins, h, index, delta, dist)
        return index, delta, dist


if numba is not None:

    @numba.njit(parallel=True, cache=True)
    def _knn_cells(fq, xq, ps, order, starts, mat, pbc, lo, span, nbins, h, index, delta, dist):
        """k nearest neighbors of every query by shells of bins (Chebyshev rings around the
        query's bin).  A bin is skipped when its slab lower bound exceeds the current k-th
        distance, and the search stops once the next ring cannot hold anything closer.
        Results are sorted by distance; missing neighbors get index -1 and distance inf."""
        m = fq.shape[0]
        kk = dist.shape[1]
        nb0, nb1, nb2 = nbins[0], nbins[1], nbins[2]
        for q in numba.prange(m):
            b = np.empty(3, np.int64)
            u = np.empty(3)
            for d in range(3):
                t = (fq[q, d] - lo[d]) / span[d] * nbins[d]
                if pbc[d]:
                    t = t - np.floor(t / nbins[d]) * nbins[d]
                bd = int(np.floor(t))
                if bd < 0:
                    bd = 0
                if bd > nbins[d] - 1:
                    bd = nbins[d] - 1
                b[d] = bd
                ud = t - bd
                u[d] = min(max(ud, 0.0), 1.0)
            for j in range(kk):
                dist[q, j] = np.inf
                index[q, j] = -1
            sel_j = np.full(kk, -1, np.int64)
            sel_s = np.zeros((kk, 3))
            found = 0
            s = 0
            while True:
                kth = dist[q, kk - 1]
                for o0 in range(-s, s + 1):
                    n0 = b[0] + o0
                    if not pbc[0] and (n0 < 0 or n0 >= nb0):
                        continue
                    l0 = 0.0
                    if o0 > 0:
                        l0 = (o0 - u[0]) * h[0]
                    elif o0 < 0:
                        l0 = (-o0 - 1 + u[0]) * h[0]
                    if l0 * l0 > kth:
                        continue
                    w0 = n0 // nb0
                    c0 = n0 - w0 * nb0
                    for o1 in range(-s, s + 1):
                        n1 = b[1] + o1
                        if not pbc[1] and (n1 < 0 or n1 >= nb1):
                            continue
                        l1 = 0.0
                        if o1 > 0:
                            l1 = (o1 - u[1]) * h[1]
                        elif o1 < 0:
                            l1 = (-o1 - 1 + u[1]) * h[1]
                        if l1 * l1 > kth:
                            continue
                        w1 = n1 // nb1
                        c1 = n1 - w1 * nb1
                        edge01 = abs(o0) == s or abs(o1) == s
                        for o2 in range(-s, s + 1):
                            if not edge01 and abs(o2) != s:
                                continue  # inside the ring: visited by an earlier shell
                            n2 = b[2] + o2
                            if not pbc[2] and (n2 < 0 or n2 >= nb2):
                                continue
                            l2 = 0.0
                            if o2 > 0:
                                l2 = (o2 - u[2]) * h[2]
                            elif o2 < 0:
                                l2 = (-o2 - 1 + u[2]) * h[2]
                            if l2 * l2 > kth:
                                continue
                            w2 = n2 // nb2
                            c2 = n2 - w2 * nb2
                            c = (c0 * nb1 + c1) * nb2 + c2
                            # image shift of this bin in fractional units
                            sh0 = float(w0) if pbc[0] else 0.0
                            sh1 = float(w1) if pbc[1] else 0.0
                            sh2 = float(w2) if pbc[2] else 0.0
                            for p in range(starts[c], starts[c + 1]):
                                # wrapped neighbor + shift - wrapped query, in Cartesian
                                d0 = ps[p, 0] + sh0 * mat[0, 0] + sh1 * mat[1, 0] + sh2 * mat[2, 0] - xq[q, 0]
                                d1 = ps[p, 1] + sh0 * mat[0, 1] + sh1 * mat[1, 1] + sh2 * mat[2, 1] - xq[q, 1]
                                d2 = ps[p, 2] + sh0 * mat[0, 2] + sh1 * mat[1, 2] + sh2 * mat[2, 2] - xq[q, 2]
                                r2 = d0 * d0 + d1 * d1 + d2 * d2
                                kth = dist[q, kk - 1]
                                if r2 >= kth:
                                    continue
                                # insertion into the sorted row (ties keep the earlier entry first)
                                pos = kk - 1
                                while pos > 0 and dist[q, pos - 1] > r2:
                                    dist[q, pos] = dist[q, pos - 1]
                                    sel_j[pos] = sel_j[pos - 1]
                                    sel_s[pos, 0] = sel_s[pos - 1, 0]
                                    sel_s[pos, 1] = sel_s[pos - 1, 1]
                                    sel_s[pos, 2] = sel_s[pos - 1, 2]
                                    pos -= 1
                                dist[q, pos] = r2
                                sel_j[pos] = p
                                sel_s[pos, 0] = d0
                                sel_s[pos, 1] = d1
                                sel_s[pos, 2] = d2
                                if found < kk:
                                    found += 1
                                kth = dist[q, kk - 1]
                # lower bound on any point in ring s + 1, over the directions it can still reach
                nxt = np.inf
                for d in range(3):
                    if pbc[d] or b[d] + s + 1 < nbins[d]:
                        nxt = min(nxt, (s + 1 - u[d]) * h[d])
                    if pbc[d] or b[d] - s - 1 >= 0:
                        nxt = min(nxt, (s + u[d]) * h[d])
                if nxt == np.inf:
                    break
                if found == kk and dist[q, kk - 1] <= nxt * nxt:
                    break
                s += 1
            for j in range(kk):
                if sel_j[j] >= 0:
                    index[q, j] = order[sel_j[j]]
                    dist[q, j] = np.sqrt(dist[q, j])
                    delta[q, j, 0] = sel_s[j, 0]
                    delta[q, j, 1] = sel_s[j, 1]
                    delta[q, j, 2] = sel_s[j, 2]
                else:
                    delta[q, j, 0] = 0.0
                    delta[q, j, 1] = 0.0
                    delta[q, j, 2] = 0.0


def nearest_neighbors(positions, cell: Cell, k: int):
    """k nearest neighbors (excluding self) of every particle, sorted by distance."""
    return PeriodicKDTree(positions, cell).query(positions, k, exclude_self=True)
