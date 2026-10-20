"""k-nearest-neighbor queries under periodic boundary conditions (triclinic cells).

Used by the structure-identification modifiers.  Periodic images of the points
within a skin layer around the cell are added to a scipy cKDTree; the skin grows
until every requested neighbor is guaranteed to be found.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from ..data import Cell


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
                m = np.all((f > -skin_f) & (f < 1 + skin_f), axis=1)
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


def nearest_neighbors(positions, cell: Cell, k: int):
    """k nearest neighbors (excluding self) of every particle, sorted by distance."""
    return PeriodicKDTree(positions, cell).query(positions, k, exclude_self=True)
