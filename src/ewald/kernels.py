"""Backend-agnostic analysis kernels (numpy on CPU, CuPy on GPU).

Everything here takes a ``backend`` (``"cpu"``, ``"cuda"``, ``"auto"`` or a
:class:`~ewald.backend.Backend`) and is written against its array module, so the
GPU path runs the same code that the CPU tests check.

>>> from ewald.kernels import rdf, trajectory_msd
>>> table = rdf(data.positions, data.cell, cutoff=6.0, types=data.types, backend="cuda")
>>> msd = trajectory_msd(pipeline, device="cuda")   # {"lag": ..., "MSD": ..., "MSD Cu": ...}
"""
from __future__ import annotations

import numpy as np

from .backend import Backend, get_backend
from .data import Cell
from .neighbors import find_pairs


# --------------------------------------------------------------------------- RDF

def rdf_from_pairs(dist, n: int, volume: float, cutoff: float, nbins: int, backend=None,
                   half: bool = False, i=None, j=None, types=None):
    """g(r) table from a pair list (as returned by :func:`find_pairs`).

    ``half`` says whether the list holds each pair once (i < j) or twice.
    With ``types`` (one label per particle) and ``i``/``j``, partial
    ``g(r) A-B`` columns are added for every unordered type pair.
    """
    be = get_backend(backend)
    xp = be.xp
    edges = np.linspace(0.0, cutoff, nbins + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    unordered = 1.0 if half else 0.5     # pair-list entries per unordered pair
    dr = cutoff / nbins
    b = xp.minimum((dist / dr).astype(xp.int64), nbins - 1)
    table = {"r": r}
    hist = be.to_numpy(xp.bincount(b, minlength=nbins)) * unordered
    npairs = n * (n - 1) / 2.0
    table["g(r)"] = hist * volume / (npairs * shell) if npairs > 0 else hist * 0.0
    if types is not None:
        if i is None or j is None:
            raise ValueError("partial g(r) needs the pair indices i and j")
        types = np.asarray(types)
        names, codes = np.unique(types, return_inverse=True)
        nt = len(names)
        counts = np.bincount(codes, minlength=nt)
        c = xp.asarray(codes)
        ti, tj = c[i], c[j]
        key = (xp.minimum(ti, tj) * nt + xp.maximum(ti, tj)) * nbins + b
        h = be.to_numpy(xp.bincount(key, minlength=nt * nt * nbins)).reshape(nt, nt, nbins) * unordered
        for a in range(nt):
            for bb in range(a, nt):
                pairs = counts[a] * (counts[a] - 1) / 2.0 if a == bb else counts[a] * counts[bb]
                table[f"g(r) {names[a]}-{names[bb]}"] = h[a, bb] * volume / (max(pairs, 1) * shell)
    return table


def rdf(positions, cell: Cell, cutoff: float, nbins: int = 200, types=None, backend=None,
        method: str = "auto"):
    """Radial distribution function (total, plus partials when ``types`` is given)."""
    be = get_backend(backend)
    pos = np.asarray(positions, dtype=float)
    i, j, _, dist = find_pairs(pos, cell, cutoff, be, half=True, method=method)
    volume = cell.volume if cell.is_valid else float(np.prod(np.ptp(pos, 0) + 1e-9))
    return rdf_from_pairs(dist, len(pos), volume, cutoff, nbins, be, half=True,
                          i=i, j=j, types=types)


# --------------------------------------------------------------------------- MSD

def unwrap(positions, cells=None, pbc=(True, True, True), backend=None):
    """Undo periodic wrapping of a trajectory ``(T, N, 3)``.

    Frame-to-frame steps are taken under minimum image in each frame's own cell
    (``cells`` is ``(3, 3)`` or ``(T, 3, 3)``), so particles must move less than
    half a cell width between stored frames. Returns a backend array.
    """
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(positions, dtype=xp.float64)
    if cells is None or not np.any(pbc) or len(x) < 2:
        return x
    cells = np.asarray(cells, dtype=float)
    if cells.ndim == 2:
        cells = np.broadcast_to(cells, (len(x), 3, 3))
    mats = xp.asarray(cells[1:])
    invs = xp.asarray(np.linalg.inv(cells[1:]))
    step = x[1:] - x[:-1]                               # (T-1, N, 3)
    f = xp.matmul(step, invs)
    f = f - xp.round(f) * xp.asarray(np.asarray(pbc, dtype=float))
    step = xp.matmul(f, mats)
    out = xp.empty_like(x)
    out[0] = x[0]
    out[1:] = x[0][None] + xp.cumsum(step, axis=0)
    return out


def msd_fft(positions, backend=None, chunk_bytes=2**28):
    """Time-averaged mean squared displacement for all lags of an unwrapped
    trajectory ``(T, N, 3)``, using the FFT algorithm (O(T log T) per atom).

    Returns a length-T array: ``msd[m] = < |x(t+m) - x(t)|^2 >_{t, atoms}``.
    """
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(positions, dtype=xp.float64)
    T, n = x.shape[0], x.shape[1]
    if n == 0 or T == 0:
        return np.zeros(T)
    lags = xp.arange(T, 0, -1, dtype=xp.float64)        # T - m
    # S1 term is linear in |x|^2, so sum over atoms first.
    D = (x * x).sum(axis=(1, 2))                        # (T,)
    csum = xp.cumsum(D)
    rsum = xp.cumsum(D[::-1])
    Q = 2.0 * D.sum() - xp.concatenate([xp.zeros(1), csum[:-1] + rsum[:-1]])
    s1 = Q / lags
    # S2: autocorrelation of each coordinate via FFT, chunked over atoms.
    nfft = 2 * T
    chunk = max(1, int(chunk_bytes // (nfft * 3 * 16)))
    acf = xp.zeros(T)
    for s in range(0, n, chunk):
        F = xp.fft.rfft(x[:, s:s + chunk], n=nfft, axis=0)
        acf += xp.fft.irfft(F * F.conj(), n=nfft, axis=0)[:T].sum(axis=(1, 2))
    s2 = acf / lags
    return be.to_numpy((s1 - 2.0 * s2) / n)


def trajectory_msd(pipeline, start: int = 0, stop: int | None = None, step: int = 1,
                   device=None, by_type: bool = True, unwrapped: bool = False):
    """Time-averaged MSD over the source frames of ``pipeline``.

    Particles are matched across frames by 'Particle Identifier' when present.
    Positions are unwrapped with :func:`unwrap` unless ``unwrapped=True``
    (e.g. LAMMPS ``xu yu zu`` columns). Returns a table with ``lag`` (in frames,
    multiplied by ``step``), ``MSD`` and, with ``by_type``, ``MSD <type>``.
    """
    be = get_backend(device or pipeline.device)
    stop = pipeline.num_frames if stop is None else stop
    frames = range(start, stop, step)
    pos, cells = [], []
    types = pbc = None
    for k in frames:
        data = pipeline.source_frame(k)
        p = np.asarray(data.positions, dtype=float)
        if "Particle Identifier" in data:
            order = np.argsort(data["Particle Identifier"], kind="stable")
            p = p[order]
            t = np.asarray(data.types)[order]
        else:
            t = np.asarray(data.types)
        if pos and len(p) != len(pos[0]):
            raise ValueError(f"frame {k} has {len(p)} particles, expected {len(pos[0])}")
        pos.append(p)
        cells.append(data.cell.matrix)
        if types is None:
            types, pbc = t, data.cell.pbc.copy()
    x = np.stack(pos)
    if not unwrapped:
        x = unwrap(x, np.stack(cells), pbc, be)
    table = {"lag": np.arange(len(frames)) * step, "MSD": msd_fft(x, be)}
    if by_type:
        names = np.unique(types)
        if len(names) > 1:
            xp = be.xp
            for name in names:
                sel = xp.asarray(np.nonzero(types == name)[0])
                table[f"MSD {name}"] = msd_fft(xp.asarray(x)[:, sel], be)
    return table


# --------------------------------------------------------------------------- clusters

def _scatter_min(xp, out, idx, vals):
    """``out[idx] = min(out[idx], vals)`` with repeated indices, using only
    sort/gather/scatter so it runs on any array module."""
    if idx.size == 0:
        return out
    order = xp.lexsort(xp.stack([vals, idx]))
    idx, vals = idx[order], vals[order]
    first = xp.ones(idx.shape, dtype=bool)
    first[1:] = idx[1:] != idx[:-1]
    idx, vals = idx[first], vals[first]
    out[idx] = xp.minimum(out[idx], vals)
    return out


def connected_components(n: int, i, j, backend=None, method: str = "auto"):
    """Label the connected components of the graph with ``n`` nodes and edges (i, j).

    Returns ``(labels, count)``: labels are 0..count-1, numbered in order of
    each component's lowest node index, as a backend array. ``method="xp"``
    (GPU default) uses min-label hooking with pointer jumping on the array
    module; on CPU ``"auto"`` uses scipy's union-find.
    """
    be = get_backend(backend)
    xp = be.xp
    if method == "auto":
        method = "scipy" if xp is np else "xp"
    if method == "scipy":
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as cc

        ii, jj = be.to_numpy(i), be.to_numpy(j)
        g = coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n, n))
        count, labels = cc(g, directed=False)
        # renumber by lowest member, like the xp path
        first = np.full(count, n, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(n))
        rank = np.empty(count, dtype=np.int64)
        rank[np.argsort(first)] = np.arange(count)
        return xp.asarray(rank[labels]), int(count)
    if method != "xp":
        raise ValueError(f"unknown method {method!r}; use 'auto', 'scipy' or 'xp'")
    i = xp.asarray(i, dtype=xp.int64)
    j = xp.asarray(j, dtype=xp.int64)
    parent = xp.arange(n, dtype=xp.int64)
    while True:
        pi, pj = parent[i], parent[j]
        lo, hi = xp.minimum(pi, pj), xp.maximum(pi, pj)
        new = _scatter_min(xp, parent.copy(), hi, lo)    # hook larger root onto smaller
        while True:                                       # pointer jumping to roots
            nxt = new[new]
            if bool((nxt == new).all()):
                break
            new = nxt
        if bool((new == parent).all()):
            break
        parent = new
    roots = parent == xp.arange(n, dtype=xp.int64)
    rank = xp.cumsum(roots.astype(xp.int64)) - 1          # roots are their component's min index
    return rank[parent], int(roots.sum())


def unwrap_clusters(positions, i, j, delta, labels, count: int, backend=None, method: str = "auto"):
    """Make each cluster contiguous across periodic boundaries.

    Walks the bonds (i, j) with vectors ``delta = r_j - r_i`` (a half or full
    pair list) outward from each cluster's lowest-index particle, as OVITO
    does. ``"scipy"`` (CPU default) is a breadth-first search; ``"xp"`` (GPU
    default) expands all clusters one bond layer per step and takes the
    lowest-index parent, which gives the same coordinates unless a cluster
    percolates through the cell. Returns a backend array.
    """
    be = get_backend(backend)
    xp = be.xp
    if method == "auto":
        method = "scipy" if xp is np else "xp"
    labels = be.to_numpy(labels) if method == "scipy" else xp.asarray(labels)
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if method == "scipy":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order

        ii, jj, dd = be.to_numpy(i), be.to_numpy(j), be.to_numpy(delta)
        src, dst = np.concatenate([ii, jj]), np.concatenate([jj, ii])
        step = np.concatenate([dd, -dd])
        seeds = np.full(count, n, dtype=np.int64)
        np.minimum.at(seeds, labels, np.arange(n))
        # a virtual root linked to every seed turns the forest into one BFS tree
        rows = np.concatenate([src, np.full(count, n)])
        cols = np.concatenate([dst, seeds])
        g = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n + 1, n + 1))
        order, pred = breadth_first_order(g, n, directed=True, return_predecessors=True)
        nodes = order[1:]
        parent = pred[nodes]
        tree = parent != n
        nodes, parent = nodes[tree], parent[tree]
        key = src * n + dst
        srt = np.argsort(key)
        stp = step[srt[np.searchsorted(key[srt], parent * n + nodes)]]
        out = pos.copy()
        done = np.zeros(n, bool)
        done[seeds] = True
        while len(nodes):                    # resolve parents before children
            ready = done[parent]
            out[nodes[ready]] = out[parent[ready]] + stp[ready]
            done[nodes[ready]] = True
            nodes, parent, stp = nodes[~ready], parent[~ready], stp[~ready]
        return xp.asarray(out)
    if method != "xp":
        raise ValueError(f"unknown method {method!r}; use 'auto', 'scipy' or 'xp'")
    i = xp.asarray(i, dtype=xp.int64)
    j = xp.asarray(j, dtype=xp.int64)
    d = xp.asarray(delta)
    src, dst = xp.concatenate([i, j]), xp.concatenate([j, i])
    step = xp.concatenate([d, -d])
    out = xp.asarray(pos).copy()
    known = xp.zeros(n, dtype=bool)
    seeds = _scatter_min(xp, xp.full(count, n, dtype=xp.int64), labels,
                         xp.arange(n, dtype=xp.int64))
    known[seeds] = True
    while True:
        alive = ~known[dst]
        src, dst, step = src[alive], dst[alive], step[alive]
        front = known[src]
        if not bool(front.any()):
            break
        s, t, dd = src[front], dst[front], step[front]
        order = xp.argsort(t * n + s)
        s, t, dd = s[order], t[order], dd[order]
        first = xp.ones(t.shape, dtype=bool)
        first[1:] = t[1:] != t[:-1]
        s, t, dd = s[first], t[first], dd[first]
        out[t] = out[s] + dd
        known[t] = True
    return out


def cluster_properties(unwrapped, labels, count: int, masses=None, backend=None):
    """Per-cluster size, center of mass, gyration tensor and radius of gyration
    from contiguous (unwrapped) coordinates, mass-weighted if ``masses`` given.

    Returns numpy arrays ``(size, com, gyration_tensor, rg)``; the tensor
    columns are XX, YY, ZZ, XY, XZ, YZ. Clusters with zero total mass get 0.
    """
    be = get_backend(backend)
    xp = be.xp
    lab = xp.asarray(labels, dtype=xp.int64)
    x = xp.asarray(unwrapped, dtype=xp.float64)
    n = len(x)
    w = xp.ones(n) if masses is None else xp.asarray(np.asarray(masses, dtype=float))
    size = xp.bincount(lab, minlength=count)
    msum = xp.bincount(lab, weights=w, minlength=count)
    msafe = xp.where(msum > 0, msum, 1.0)
    com = xp.stack([xp.bincount(lab, weights=w * x[:, k], minlength=count) for k in range(3)], 1)
    com = com / msafe[:, None]
    com[msum <= 0] = 0.0
    d = x - com[lab]
    comps = (d[:, 0] * d[:, 0], d[:, 1] * d[:, 1], d[:, 2] * d[:, 2],
             d[:, 0] * d[:, 1], d[:, 0] * d[:, 2], d[:, 1] * d[:, 2])
    gyr = xp.stack([xp.bincount(lab, weights=w * c, minlength=count) for c in comps], 1)
    gyr = gyr / msafe[:, None]
    rg = xp.sqrt(gyr[:, :3].sum(1))
    return be.to_numpy(size), be.to_numpy(com), be.to_numpy(gyr), be.to_numpy(rg)
