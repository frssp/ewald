"""Cutoff neighbor search with a cell list.

Three implementations share the same binning and return the same pairs:

``"numba"``  compiled, multithreaded two-pass kernel (CPU default when numba is installed)
``"cuda"``   the same two-pass kernel in CUDA C via CuPy ``RawKernel`` (GPU default)
``"xp"``     vectorized array code that runs unchanged on numpy or CuPy (GPU fallback)
``"brute"``  all-pairs in chunks (small systems, or fewer than 3 bins along a periodic axis)
"""
from __future__ import annotations

import itertools

import numpy as np

from .backend import Backend, get_backend
from .data import Cell

try:
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None
    HAVE_NUMBA = False

METHODS = ("auto", "numba", "cuda", "xp", "brute")


def _minimum_image(xp, d_frac, pbc):
    if any(pbc):
        mask = xp.asarray(np.asarray(pbc, dtype=float))
        d_frac = d_frac - xp.round(d_frac) * mask
    return d_frac


def _empty(xp):
    z = xp.zeros(0, dtype=xp.int64)
    return z, z, xp.zeros((0, 3)), xp.zeros(0)


def find_pairs(positions, cell: Cell, cutoff: float, backend: Backend | str | None = None,
               half: bool = False, method: str = "auto"):
    """All pairs (i, j), i != j, with |r_ij| < cutoff under minimum image.

    Returns ``(i, j, delta, dist)`` as backend arrays sorted by (i, j), with
    ``delta = r_j - r_i``. By default both (i, j) and (j, i) are listed; with
    ``half=True`` only i < j. ``method`` picks the implementation (see module
    docstring); ``"auto"`` uses numba on CPU and the array code on GPU.

    Requires ``cutoff`` < half the cell width along periodic directions; larger
    cutoffs need a replicated cell (not done implicitly).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    be = get_backend(backend)
    xp = be.xp
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n == 0:
        return _empty(xp)

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
    nbins = np.maximum(1, np.floor(span * widths / cutoff).astype(np.int64))
    # Cap the grid so that sparse systems (e.g. a cluster in a large vacuum box)
    # don't allocate far more cells than atoms.
    while np.prod(nbins) > 4 * n + 27 and np.any(nbins > 3):
        nbins = np.where(nbins > 3, np.maximum(3, nbins // 2), nbins)

    if method == "auto":
        if np.any(pbc & (nbins < 3)) or n < 64:
            method = "brute"
        elif xp is np and HAVE_NUMBA:
            method = "numba"
        elif be.raw_kernels:
            method = "cuda"
        else:
            method = "xp"
    if method != "brute" and np.any(pbc & (nbins < 3)):
        method = "brute"  # fewer than 3 periodic bins would visit a cell twice

    if method == "brute":
        out = _brute_force(be, frac, matrix, pbc, cutoff, half=half)
    elif method == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("method='numba' needs numba: pip install numba")
        if xp is not np:
            raise ValueError("method='numba' runs on the CPU backend only")
        out = _cell_list_numba(frac, matrix, pbc, cutoff, lo, span, nbins, half)
    elif method == "cuda":
        if not be.raw_kernels:
            raise ValueError(f"method='cuda' needs a CUDA backend, not {be.name!r}")
        out = _cell_list_cuda(be, frac, matrix, pbc, cutoff, lo, span, nbins, half)
    else:
        out = _cell_list_xp(be, frac, matrix, pbc, cutoff, lo, span, nbins, half)
    return out


def _bin_index(frac, lo, span, nbins):
    b = np.floor((frac - lo) / span * nbins).astype(np.int64)
    return np.minimum(np.maximum(b, 0), nbins - 1)


# --------------------------------------------------------------------------- numba

if HAVE_NUMBA:

    @numba.njit(parallel=True, cache=True)
    def _numba_pairs(fs, bs, starts, order, mat, pbc, nbins, cutoff2, half, offsets,
                     out_i, out_j, out_d, fill):
        """Pass 1 (fill=False) counts neighbors per sorted atom into ``offsets``;
        pass 2 (fill=True) writes them at ``offsets`` (now a prefix sum)."""
        n = fs.shape[0]
        nb0, nb1, nb2 = nbins[0], nbins[1], nbins[2]
        for k in numba.prange(n):
            i = order[k]
            cx, cy, cz = bs[k, 0], bs[k, 1], bs[k, 2]
            pos = offsets[k] if fill else 0
            first = pos
            for ox in range(-1, 2):
                nx = cx + ox
                if pbc[0]:
                    nx = nx % nb0
                elif nx < 0 or nx >= nb0:
                    continue
                for oy in range(-1, 2):
                    ny = cy + oy
                    if pbc[1]:
                        ny = ny % nb1
                    elif ny < 0 or ny >= nb1:
                        continue
                    for oz in range(-1, 2):
                        nz = cz + oz
                        if pbc[2]:
                            nz = nz % nb2
                        elif nz < 0 or nz >= nb2:
                            continue
                        c = (nx * nb1 + ny) * nb2 + nz
                        for s in range(starts[c], starts[c + 1]):
                            j = order[s]
                            if j == i or (half and j < i):
                                continue
                            f0 = fs[s, 0] - fs[k, 0]
                            f1 = fs[s, 1] - fs[k, 1]
                            f2 = fs[s, 2] - fs[k, 2]
                            if pbc[0]:
                                f0 -= np.rint(f0)
                            if pbc[1]:
                                f1 -= np.rint(f1)
                            if pbc[2]:
                                f2 -= np.rint(f2)
                            d0 = f0 * mat[0, 0] + f1 * mat[1, 0] + f2 * mat[2, 0]
                            d1 = f0 * mat[0, 1] + f1 * mat[1, 1] + f2 * mat[2, 1]
                            d2 = f0 * mat[0, 2] + f1 * mat[1, 2] + f2 * mat[2, 2]
                            r2 = d0 * d0 + d1 * d1 + d2 * d2
                            if r2 < cutoff2:
                                if fill:
                                    out_i[pos] = i
                                    out_j[pos] = j
                                    out_d[pos, 0] = d0
                                    out_d[pos, 1] = d1
                                    out_d[pos, 2] = d2
                                pos += 1
            if fill:
                # insertion sort of this row by j (rows are short)
                for a in range(first + 1, pos):
                    jj = out_j[a]
                    t0, t1, t2 = out_d[a, 0], out_d[a, 1], out_d[a, 2]
                    b = a - 1
                    while b >= first and out_j[b] > jj:
                        out_j[b + 1] = out_j[b]
                        out_d[b + 1, 0] = out_d[b, 0]
                        out_d[b + 1, 1] = out_d[b, 1]
                        out_d[b + 1, 2] = out_d[b, 2]
                        b -= 1
                    out_j[b + 1] = jj
                    out_d[b + 1, 0] = t0
                    out_d[b + 1, 1] = t1
                    out_d[b + 1, 2] = t2
            else:
                offsets[k] = pos


def _sorted_bins(frac, lo, span, nbins):
    """Cell-sorted copies for the per-atom kernels: positions, bin coords,
    cell start offsets and the sorted -> original index map."""
    b = _bin_index(frac, lo, span, nbins)
    cell_id = (b[:, 0] * nbins[1] + b[:, 1]) * nbins[2] + b[:, 2]
    order = np.argsort(cell_id, kind="stable").astype(np.int64)
    starts = np.zeros(int(np.prod(nbins)) + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell_id, minlength=len(starts) - 1), out=starts[1:])
    return np.ascontiguousarray(frac[order]), np.ascontiguousarray(b[order]), starts, order


def _cell_list_cuda(be: Backend, frac, matrix, pbc, cutoff, lo, span, nbins, half):
    from .cuda_kernels import cell_list_pairs

    xp = be.xp
    fs, bs, starts, order = _sorted_bins(frac, lo, span, nbins)
    i, j, d = cell_list_pairs(xp, fs, bs, starts, order, matrix, pbc, nbins, cutoff, half)
    return i, j, d, xp.sqrt((d * d).sum(1))


def _cell_list_numba(frac, matrix, pbc, cutoff, lo, span, nbins, half):
    n = len(frac)
    fs, bs, starts, order = _sorted_bins(frac, lo, span, nbins)
    mat = np.ascontiguousarray(matrix, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=np.bool_)
    counts = np.zeros(n, dtype=np.int64)
    dummy_i = np.zeros(0, dtype=np.int64)
    dummy_d = np.zeros((0, 3))
    args = (fs, bs, starts, order, mat, pbc, nbins.astype(np.int64), cutoff * cutoff, half)
    _numba_pairs(*args, counts, dummy_i, dummy_i, dummy_d, False)
    # Output rows follow original atom order, so offsets are a prefix sum over i.
    row_len = np.empty(n, dtype=np.int64)
    row_len[order] = counts
    row_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_len, out=row_start[1:])
    total = int(row_start[-1])
    out_i = np.empty(total, dtype=np.int64)
    out_j = np.empty(total, dtype=np.int64)
    out_d = np.empty((total, 3))
    _numba_pairs(*args, row_start[order], out_i, out_j, out_d, True)
    return out_i, out_j, out_d, np.sqrt(np.einsum("ij,ij->i", out_d, out_d))


# --------------------------------------------------------------------------- array code (numpy / CuPy)

def _cell_list_xp(be: Backend, frac, matrix, pbc, cutoff, lo, span, nbins, half,
                  chunk_elems=2**24):
    xp = be.xp
    n = len(frac)
    f = xp.asarray(frac)
    mat = xp.asarray(matrix)
    b = xp.asarray(_bin_index(frac, lo, span, nbins))
    cell_id = (b[:, 0] * int(nbins[1]) + b[:, 1]) * int(nbins[2]) + b[:, 2]
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

    nb = [int(x) for x in nbins]
    # Visit occupied cells only, in chunks that bound the (chunk, m, m) candidates.
    occupied = xp.nonzero(counts)[0].astype(xp.int64)
    chunk = max(1, chunk_elems // (m * m))
    out_i, out_j, out_d = [], [], []
    for c0 in range(0, len(occupied), chunk):
        cells = occupied[c0:c0 + chunk]
        cz = cells % nb[2]
        cy = (cells // nb[2]) % nb[1]
        cx = cells // (nb[1] * nb[2])
        ai = table[cells][:, :, None]          # (chunk, m, 1)
        for off in itertools.product((-1, 0, 1), repeat=3):
            valid = xp.ones(len(cells), dtype=bool)
            coords = []
            for c, o, nbd, p in zip((cx, cy, cz), off, nb, pbc):
                c = c + o
                if p:
                    coords.append(c % nbd)
                else:
                    valid &= (c >= 0) & (c < nbd)
                    coords.append(xp.clip(c, 0, nbd - 1))
            other = (coords[0] * nb[1] + coords[1]) * nb[2] + coords[2]
            aj = table[other][:, None, :]      # (chunk, 1, m)
            ok = (ai >= 0) & (aj >= 0) & valid[:, None, None]
            ok &= (ai < aj) if half else (ai != aj)
            ii = xp.broadcast_to(ai, ok.shape)[ok]
            jj = xp.broadcast_to(aj, ok.shape)[ok]
            if ii.size == 0:
                continue
            d = _minimum_image(xp, f[jj] - f[ii], pbc) @ mat
            dist2 = (d * d).sum(1)
            keep = dist2 < cutoff * cutoff
            out_i.append(ii[keep]); out_j.append(jj[keep]); out_d.append(d[keep])
    if not out_i:
        return _empty(xp)
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    srt = xp.argsort(i * n + j)
    i, j, d = i[srt], j[srt], d[srt]
    return i, j, d, xp.sqrt((d * d).sum(1))


def _brute_force(be: Backend, frac, matrix, pbc, cutoff, chunk_bytes=2**27, half=False):
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
        if half:
            dist[xp.arange(n)[None, :] < rows[:, None]] = xp.inf
        ii, jj = xp.nonzero(dist < cutoff)
        out_i.append(ii + s); out_j.append(jj); out_d.append(d[ii, jj])
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    return i, j, d, xp.sqrt((d * d).sum(1))
