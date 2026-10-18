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
