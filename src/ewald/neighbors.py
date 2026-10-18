"""Cutoff neighbor search with a vectorized cell list (runs on numpy or CuPy)."""
from __future__ import annotations

import itertools

import numpy as np

from .backend import Backend, get_backend
from .data import Cell


def _minimum_image(xp, d_frac, pbc):
    if any(pbc):
        mask = xp.asarray(np.asarray(pbc, dtype=float))
        d_frac = d_frac - xp.round(d_frac) * mask
    return d_frac


def find_pairs(positions, cell: Cell, cutoff: float, backend: Backend | str | None = None):
    """All ordered pairs (i, j), i != j, with |r_ij| < cutoff under minimum image.

    Returns ``(i, j, delta, dist)`` as backend arrays, ``delta = r_j - r_i``.
    Requires ``cutoff`` < half the cell width along periodic directions; larger
    cutoffs need a replicated cell (not done implicitly).
    """
    be = get_backend(backend)
    xp = be.xp
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n == 0:
        z = xp.zeros(0, dtype=xp.int64)
        return z, z, xp.zeros((0, 3)), xp.zeros(0)

    if cell.is_valid:
        matrix = cell.matrix
        pbc = cell.pbc.copy()
        widths = cell.perpendicular_widths()
    else:  # no cell: bounding box, non-periodic
        lo, hi = pos.min(0), pos.max(0)
        matrix = np.diag(np.maximum(hi - lo, 1.0))
        cell = Cell(matrix, lo, np.zeros(3, bool))
        pbc = cell.pbc
        widths = np.diag(matrix).copy()
    if np.any(pbc & (widths < 2 * cutoff)):
        raise ValueError(f"cutoff {cutoff} exceeds half the periodic cell width "
                         f"(min width {widths[pbc].min():.3f}); replicate the cell first")

    inv = np.linalg.inv(matrix)
    frac = (pos - cell.origin) @ inv
    frac[:, pbc] %= 1.0
    # Bin grid in fractional space; non-periodic directions span the data extent.
    lo = np.where(pbc, 0.0, frac.min(0))
    span = np.where(pbc, 1.0, np.maximum(frac.max(0) - lo, 1e-12))
    nbins = np.maximum(1, np.floor(span * widths / cutoff).astype(int))
    if np.any(pbc & (nbins < 3)) or n < 64:
        return _brute_force(be, frac, matrix, pbc, cutoff)

    f = xp.asarray(frac)
    mat = xp.asarray(matrix)
    nb = xp.asarray(nbins)
    b = xp.floor((f - xp.asarray(lo)) / xp.asarray(span) * nb).astype(xp.int64)
    b = xp.minimum(xp.maximum(b, 0), nb - 1)
    cell_id = (b[:, 0] * nbins[1] + b[:, 1]) * nbins[2] + b[:, 2]
    order = xp.argsort(cell_id)
    sorted_ids = cell_id[order]
    ncells = int(np.prod(nbins))
    counts = xp.bincount(sorted_ids, minlength=ncells)
    starts = xp.concatenate([xp.zeros(1, dtype=xp.int64), xp.cumsum(counts)[:-1]])
    m = int(counts.max())
    # Padded (ncells, m) table of atom indices, -1 = empty slot.
    slot = xp.arange(n, dtype=xp.int64) - starts[sorted_ids]
    table = -xp.ones((ncells, m), dtype=xp.int64)
    table[sorted_ids, slot] = order

    cells = xp.arange(ncells, dtype=xp.int64)
    cz = cells % nbins[2]
    cy = (cells // nbins[2]) % nbins[1]
    cx = cells // (nbins[1] * nbins[2])
    out_i, out_j, out_d = [], [], []
    for off in itertools.product((-1, 0, 1), repeat=3):
        nx, ny, nz = cx + off[0], cy + off[1], cz + off[2]
        valid = xp.ones(ncells, dtype=bool)
        coords = []
        for c, nbd, p in ((nx, nbins[0], pbc[0]), (ny, nbins[1], pbc[1]), (nz, nbins[2], pbc[2])):
            if p:
                coords.append(c % nbd)
            else:
                valid &= (c >= 0) & (c < nbd)
                coords.append(xp.clip(c, 0, nbd - 1))
        other = (coords[0] * nbins[1] + coords[1]) * nbins[2] + coords[2]
        ai = table[cells][:, :, None]          # (ncells, m, 1)
        aj = table[other][:, None, :]          # (ncells, 1, m)
        ok = (ai >= 0) & (aj >= 0) & valid[:, None, None] & (ai != aj)
        ii = xp.broadcast_to(ai, ok.shape)[ok]
        jj = xp.broadcast_to(aj, ok.shape)[ok]
        if ii.size == 0:
            continue
        d = _minimum_image(xp, f[jj] - f[ii], pbc) @ mat
        dist = xp.sqrt((d * d).sum(1))
        keep = dist < cutoff
        out_i.append(ii[keep]); out_j.append(jj[keep]); out_d.append(d[keep])
    if not out_i:
        z = xp.zeros(0, dtype=xp.int64)
        return z, z, xp.zeros((0, 3)), xp.zeros(0)
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    srt = xp.lexsort(xp.stack([j, i]))
    i, j, d = i[srt], j[srt], d[srt]
    return i, j, d, xp.sqrt((d * d).sum(1))


def _brute_force(be: Backend, frac, matrix, pbc, cutoff, chunk_bytes=2**27):
    xp = be.xp
    f = xp.asarray(frac)
    mat = xp.asarray(matrix)
    n = len(frac)
    chunk = max(1, int(chunk_bytes // (24 * n)))
    out_i, out_j, out_d = [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _minimum_image(xp, f[None, :, :] - f[s:e, None, :], pbc) @ mat  # (c, n, 3)
        dist = xp.sqrt((d * d).sum(-1))
        rows = xp.arange(s, e)
        dist[xp.arange(e - s), rows] = xp.inf
        ii, jj = xp.nonzero(dist < cutoff)
        out_i.append(ii + s); out_j.append(jj); out_d.append(d[ii, jj])
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    return i, j, d, xp.sqrt((d * d).sum(1))
