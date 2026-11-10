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

from . import backend as _backend
from .backend import CPU, Backend, format_bytes, get_backend
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

    A ``cutoff`` beyond half the cell width along a periodic direction (a small cell,
    e.g. ``rdf(pipe, cutoff=6)`` on a 108-atom box) is searched over periodic images
    (:func:`_image_pairs`): then a pair (i, j) is listed once per image of j within reach,
    with that image's ``delta``, and an atom is its own neighbor (i == j) once the cutoff
    passes a lattice vector, as in OVITO.

    A pair list that would not fit into memory (a cutoff far too large for the number of
    atoms) raises ``MemoryError`` before it is allocated, with the cutoff that would fit.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    return _search(get_backend(backend), np.asarray(positions, dtype=float), cell, cutoff, half, method)


#: Bytes of one listed pair: i and j (int64), delta (three float64) and dist.
PAIR_BYTES = 48


def expected_pairs(n: int, volume: float, cutoff: float, open_extents=(), images: bool = False) -> float:
    """Ordered pairs within ``cutoff`` among ``n`` atoms spread evenly over ``volume``: each atom
    times the atoms in a sphere around it (its own periodic images too with ``images``, a cutoff
    beyond the cell), less what the two ends of an open (non-periodic) direction of length
    ``open_extents[k]`` cut off. Real structures list about this many or more (voids, surfaces
    and clusters put atoms closer together), so a search refused on this number would not have fit."""
    if not np.isfinite(cutoff):
        return float("inf")
    pairs = n * (n if images else n - 1.0) * 4.0 / 3.0 * np.pi * cutoff ** 3 / volume
    for extent in open_extents:
        pairs *= 1.0 - cutoff / (2 * extent) if cutoff < extent else extent / (2 * cutoff)
    return pairs


class _MemoryGuard:
    """Refuses a pair list that will not fit into memory before it is allocated: once on the
    estimate of :func:`expected_pairs`, and again by the kernels on the pairs they counted or
    found so far. ``copies`` is how many times the list is held at its peak; ``why`` says in the
    message what makes the list long (the periodic images of a cutoff beyond the cell).
    ``half``: the list holds each pair once, so an atom has twice the listed pairs per atom as
    neighbors. ``estimate(cutoff)`` gives the expected length of the list at another cutoff,
    for the advice in the message: with open directions the list does not grow as cutoff^3."""

    def __init__(self, be: Backend, n: int, cutoff: float, copies: int = 1, why: str = "", half: bool = False,
                 estimate=None):
        self.be, self.n, self.cutoff, self.copies, self.why = be, n, cutoff, copies, why
        self.half, self.estimate = half, estimate

    def check(self, pairs: float, about: str = ""):
        """``pairs`` is the exact length of the list, an estimate (``about="about "``) or what a
        search has found so far (``"over "``)."""
        need = pairs * PAIR_BYTES * self.copies
        short = self.be.short_of_memory(need)
        if short is None:
            return
        budget, have = short
        fits = self._fits(budget, need)
        neighbors = pairs * (2 if self.half else 1) / max(self.n, 1)
        raise MemoryError(
            f"cutoff {self.cutoff:g} needs too much memory: {about}{pairs:.3g} pairs"
            f"{' (each listed once)' if self.half else ''} for {self.n} atoms "
            f"({about}{neighbors:.0f} neighbors each{self.why}) take {format_bytes(need)}, "
            f"and {have}. Use a smaller cutoff: {'less than' if about == 'over ' else 'about'} {fits:.3g} "
            "would fit.")

    def _fits(self, budget: float, need: float) -> float:
        """The cutoff whose list fits into ``budget`` bytes, 5% under the limit since the count
        is an estimate: the largest one the estimate allows (found by bisection), and no more
        than cutoff^3 scaling from the list that was refused gives."""
        cubic = self.cutoff * (budget / need) ** (1.0 / 3.0) if np.isfinite(self.cutoff) else np.inf
        if self.estimate is None:
            return 0.95 * cubic

        def fits(cutoff: float) -> bool:
            return self.estimate(cutoff) * PAIR_BYTES * self.copies <= budget

        high = self.cutoff if np.isfinite(self.cutoff) else 1.0
        for _ in range(200):       # an infinite cutoff: a finite one that does not fit either
            if not fits(high):
                break
            high *= 2.0
        else:
            return 0.95 * cubic
        low = 0.0
        for _ in range(60):
            mid = 0.5 * (low + high)
            low, high = (mid, high) if fits(mid) else (low, mid)
        return 0.95 * min(low, cubic)


def _search(be: Backend, pos, cell: Cell, cutoff: float, half: bool, method: str, rows=None, guard=None):
    """:func:`find_pairs` on a float array. ``rows = (first, last)`` lists only the pairs that
    start at the atoms ``first <= i < last``, in no particular order (the periodic-image search
    asks for the cell's own atoms among their images and sorts them itself; CPU methods only),
    and ``guard`` is the caller's memory guard."""
    xp = be.xp
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
        return _image_pairs(be, pos, cell, matrix, pbc, widths, cutoff, half, method)

    inv = np.linalg.inv(matrix)
    frac = (pos - cell.origin) @ inv
    frac[:, pbc] %= 1.0
    # Bin grid in fractional space; non-periodic directions span the data extent.
    lo = np.where(pbc, 0.0, frac.min(0))
    span = np.where(pbc, 1.0, np.maximum(frac.max(0) - lo, 1e-12))
    # A list of every pair there is (the most minimum image can give) that is too small to be
    # refused needs no estimate: per-frame analyses of small cells call this in a loop.
    if guard is None and (_backend.MEMORY_BUDGET is not None or n * (n - 1.0) * PAIR_BYTES >= _backend.SMALL_ALLOCATION):
        volume, opens = abs(np.linalg.det(matrix)) * span.prod(), (span * widths)[~pbc]

        def listed(c: float) -> float:
            pairs = min(expected_pairs(n, volume, c, opens), n * (n - 1.0))   # a pair once each way at most
            return pairs / 2 if half else pairs

        guard = _MemoryGuard(be, n, cutoff, half=half, estimate=listed)
        guard.check(listed(cutoff), "about ")
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
        out = _brute_force(be, frac, matrix, pbc, cutoff, half=half, rows=rows, guard=guard)
    elif method == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("method='numba' needs numba: pip install numba")
        if xp is not np:
            raise ValueError("method='numba' runs on the CPU backend only")
        out = _cell_list_numba(frac, matrix, pbc, cutoff, lo, span, nbins, half, rows=rows, guard=guard)
    elif method == "cuda":
        if not be.raw_kernels:
            raise ValueError(f"method='cuda' needs a CUDA backend, not {be.name!r}")
        out = _cell_list_cuda(be, frac, matrix, pbc, cutoff, lo, span, nbins, half)
    else:
        out = _cell_list_xp(be, frac, matrix, pbc, cutoff, lo, span, nbins, half, rows=rows, guard=guard)
    return out


#: Most atoms (the cell's atoms times its periodic images) :func:`_image_pairs` searches.
MAX_IMAGE_ATOMS = 2_000_000


def _image_pairs(be: Backend, pos, cell: Cell, matrix, pbc, widths, cutoff, half, method):
    """Pairs within a cutoff beyond half the cell width: the wrapped atoms and their periodic
    images within reach go through the non-periodic search, which lists the pairs that start at
    an atom of the cell itself (j mapped back to its atom). Pairs between two images are never
    listed: with a cutoff of several cell widths they are a thousand times the real ones."""
    xp = be.xp
    n = len(pos)
    if not np.isfinite(cutoff):
        raise ValueError(f"cutoff {cutoff} in a periodic cell would list every atom with infinitely many "
                         "periodic images: give a finite cutoff")
    reps = np.where(pbc, np.ceil(cutoff / np.where(pbc, widths, 1.0)).astype(np.int64), 0)
    nshifts = int(np.prod(2 * reps + 1, dtype=float))    # counted before the list of them is built
    if n * nshifts > MAX_IMAGE_ATOMS:
        raise ValueError(f"cutoff {cutoff} exceeds half the periodic cell width (min width "
                         f"{widths[pbc].min():.3f}) and the {nshifts} periodic images it needs hold more than "
                         f"{MAX_IMAGE_ATOMS} atoms; use a smaller cutoff or replicate the cell first")
    frac = (pos - cell.origin) @ np.linalg.inv(matrix)
    frac[:, pbc] %= 1.0
    span = np.where(pbc, 1.0, np.maximum(np.ptp(frac, axis=0), 1e-12))
    # the search result and its sorted copy are held together below, hence two copies
    volume, opens = abs(np.linalg.det(matrix)) * span.prod(), (span * widths)[~pbc]

    def listed(c: float) -> float:
        return expected_pairs(n, volume, c, opens, images=True)

    guard = _MemoryGuard(CPU, n, cutoff, copies=2, estimate=listed, why=(
        f"; the cutoff is {cutoff / widths[pbc].min():.3g} times the periodic cell width "
        f"{widths[pbc].min():.4g}, so atoms also pair with up to {nshifts - 1} periodic images of the cell"))
    guard.check(listed(cutoff), "about ")
    shifts = np.array(list(itertools.product(*[range(-r, r + 1) for r in reps])), dtype=float)
    wrapped = frac @ matrix + cell.origin
    center = int(np.flatnonzero(~shifts.any(1))[0])      # the image with no shift: the cell itself
    ext = (wrapped[None] + (shifts @ matrix)[:, None]).reshape(-1, 3)
    inner = "auto" if method == "cuda" else method        # small cells: the images are searched on the CPU
    i, j, delta, dist = (be.to_numpy(a) for a in _search(
        CPU, ext, Cell(np.zeros((3, 3))), cutoff, False, inner, rows=(center * n, (center + 1) * n), guard=guard))
    i = i - center * n
    image, j = j // n, j % n
    if half:   # one of (i, j, image) and (j, i, -image): i < j, and an atom's own images once
        mirror = len(shifts) - 1 - image                 # shifts are symmetric: index k <-> -shift
        keep = (i < j) | ((i == j) & (image > mirror))
        i, j, image, delta, dist = i[keep], j[keep], image[keep], delta[keep], dist[keep]
    order = np.lexsort((image, j, i))
    return tuple(xp.asarray(a[order]) for a in (i, j, delta, dist))


def _bin_index(frac, lo, span, nbins):
    b = np.floor((frac - lo) / span * nbins).astype(np.int64)
    return np.minimum(np.maximum(b, 0), nbins - 1)


# --------------------------------------------------------------------------- numba

if HAVE_NUMBA:

    # nogil: the GUI's compute worker searches bonds while the window keeps painting
    @numba.njit(parallel=True, cache=True, nogil=True)
    def _numba_pairs(fs, bs, starts, order, mat, pbc, nbins, cutoff2, half, first_row, last_row, sort_rows,
                     offsets, out_i, out_j, out_d, fill):
        """Pass 1 (fill=False) counts neighbors per sorted atom into ``offsets``;
        pass 2 (fill=True) writes them at ``offsets`` (now a prefix sum). Only the atoms
        ``first_row <= i < last_row`` get a row (the others keep the count 0 they came with),
        sorted by j unless the caller sorts anyway (``sort_rows``)."""
        n = fs.shape[0]
        nb0, nb1, nb2 = nbins[0], nbins[1], nbins[2]
        for k in numba.prange(n):
            i = order[k]
            if i < first_row or i >= last_row:
                continue
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
            if fill and sort_rows:
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
            elif not fill:
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


def _cell_list_numba(frac, matrix, pbc, cutoff, lo, span, nbins, half, rows=None, guard=None):
    n = len(frac)
    fs, bs, starts, order = _sorted_bins(frac, lo, span, nbins)
    mat = np.ascontiguousarray(matrix, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=np.bool_)
    counts = np.zeros(n, dtype=np.int64)
    dummy_i = np.zeros(0, dtype=np.int64)
    dummy_d = np.zeros((0, 3))
    first, last = (0, n) if rows is None else rows
    # the image search sorts its pairs itself, and its rows (thousands of neighbors once the cutoff
    # is several cell widths) are far too long for the kernel's insertion sort
    args = (fs, bs, starts, order, mat, pbc, nbins.astype(np.int64), cutoff * cutoff, half, first, last,
            rows is None)
    _numba_pairs(*args, counts, dummy_i, dummy_i, dummy_d, False)
    # Output rows follow original atom order, so offsets are a prefix sum over i.
    row_len = np.empty(n, dtype=np.int64)
    row_len[order] = counts
    row_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_len, out=row_start[1:])
    total = int(row_start[-1])
    if guard is not None:
        # the exact size, known before the arrays exist: np.empty of more than the machine has can
        # succeed (Linux overcommits) and the fill pass then gets the process killed
        guard.check(total)
    out_i = np.empty(total, dtype=np.int64)
    out_j = np.empty(total, dtype=np.int64)
    out_d = np.empty((total, 3))
    _numba_pairs(*args, row_start[order], out_i, out_j, out_d, True)
    return out_i, out_j, out_d, np.sqrt(np.einsum("ij,ij->i", out_d, out_d))


# --------------------------------------------------------------------------- array code (numpy / CuPy)

def _cell_table(xp, cell_id, ncells, first=0):
    """Padded ``(ncells, m)`` table of the atoms in each cell (atom ``first + k`` is in cell
    ``cell_id[k]``), -1 = empty slot, and the cells that hold any."""
    order = xp.argsort(cell_id)
    sorted_ids = cell_id[order]
    counts = xp.bincount(sorted_ids, minlength=ncells)
    starts = xp.concatenate([xp.zeros(1, dtype=xp.int64), xp.cumsum(counts)[:-1]])
    slot = xp.arange(len(cell_id), dtype=xp.int64) - starts[sorted_ids]
    table = -xp.ones((ncells, int(counts.max())), dtype=xp.int64)
    table[sorted_ids, slot] = order + first
    return table, xp.nonzero(counts)[0].astype(xp.int64)


def _cell_list_xp(be: Backend, frac, matrix, pbc, cutoff, lo, span, nbins, half,
                  chunk_elems=2**24, rows=None, guard=None):
    xp = be.xp
    n = len(frac)
    f = xp.asarray(frac)
    mat = xp.asarray(matrix)
    b = xp.asarray(_bin_index(frac, lo, span, nbins))
    cell_id = (b[:, 0] * int(nbins[1]) + b[:, 1]) * int(nbins[2]) + b[:, 2]
    ncells = int(np.prod(nbins))
    table, occupied = _cell_table(xp, cell_id, ncells)
    own = table                            # the atoms pairs start at (i): all of them,
    if rows is not None:                   # or a table of just the rows asked for
        own, occupied = _cell_table(xp, cell_id[rows[0]:rows[1]], ncells, rows[0])

    nb = [int(x) for x in nbins]
    # Visit occupied cells only, in chunks that bound the (chunk, m, m) candidates.
    chunk = max(1, chunk_elems // (own.shape[1] * table.shape[1]))
    out_i, out_j, out_d = [], [], []
    found = 0
    for c0 in range(0, len(occupied), chunk):
        cells = occupied[c0:c0 + chunk]
        cz = cells % nb[2]
        cy = (cells // nb[2]) % nb[1]
        cx = cells // (nb[1] * nb[2])
        ai = own[cells][:, :, None]            # (chunk, m, 1)
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
            found += len(out_i[-1])
            if guard is not None:
                guard.check(found, "over ")    # too many already: stop before finding the rest
    if not out_i:
        return _empty(xp)
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    srt = xp.argsort(i * n + j)
    i, j, d = i[srt], j[srt], d[srt]
    return i, j, d, xp.sqrt((d * d).sum(1))


def _brute_force(be: Backend, frac, matrix, pbc, cutoff, chunk_bytes=2**27, half=False, rows=None, guard=None):
    xp = be.xp
    f = xp.asarray(frac)
    mat = xp.asarray(matrix)
    n = len(frac)
    chunk = max(1, int(chunk_bytes // (24 * n)))
    first, last = (0, n) if rows is None else rows
    out_i, out_j, out_d = [], [], []
    found = 0
    for s in range(first, last, chunk):
        e = min(last, s + chunk)
        d = _minimum_image(xp, f[None, :, :] - f[s:e, None, :], pbc) @ mat  # (c, n, 3)
        dist = xp.sqrt((d * d).sum(-1))
        own = xp.arange(s, e)
        dist[xp.arange(e - s), own] = xp.inf
        if half:
            dist[xp.arange(n)[None, :] < own[:, None]] = xp.inf
        ii, jj = xp.nonzero(dist < cutoff)
        out_i.append(ii + s); out_j.append(jj); out_d.append(d[ii, jj])
        found += len(ii)
        if guard is not None:
            guard.check(found, "over ")
    i = xp.concatenate(out_i); j = xp.concatenate(out_j); d = xp.concatenate(out_d)
    return i, j, d, xp.sqrt((d * d).sum(1))
