"""Backend-agnostic analysis kernels (numpy on CPU, CuPy on GPU).

Everything here takes a ``backend`` (``"cpu"``, ``"cuda"``, ``"auto"`` or a
:class:`~ewald.backend.Backend`) and is written against its array module, so the
GPU path runs the same code that the CPU tests check.

>>> from ewald.kernels import rdf, trajectory_msd
>>> table = rdf(data.positions, data.cell, cutoff=6.0, types=data.types, backend="cuda")
>>> msd = trajectory_msd(pipeline, device="cuda")   # {"lag": ..., "MSD": ..., "MSD Cu": ...}
>>> D, _, _ = diffusion_fit(msd["lag"] * dt, msd["MSD"])
"""
from __future__ import annotations

import numpy as np

from .backend import get_backend
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


def _bins(bins, kwargs, old: str = "nbins"):
    """``bins``, also accepted under its old name (``nbins``) in ``kwargs``; default 200."""
    if old in kwargs:
        if bins is not None:
            raise TypeError(f"give bins or {old}, not both")
        bins = kwargs.pop(old)
    if kwargs:
        raise TypeError(f"unexpected keyword argument {next(iter(kwargs))!r}")
    return 200 if bins is None else int(bins)


def rdf(positions, cell: Cell, cutoff: float, bins: int | None = None, types=None, backend=None,
        method: str = "auto", **kwargs):
    """Radial distribution function (total, plus partials when ``types`` is given) in ``bins``
    shells up to ``cutoff`` (default 200; ``nbins`` is the old name)."""
    nbins = _bins(bins, kwargs)
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


def _acf_sum(x, xp, chunk_bytes):
    """sum over atoms and components of sum_t x(t) x(t+m), for every lag m, via FFT."""
    T = x.shape[0]
    nfft = 2 * T
    per_atom = int(np.prod(x.shape[2:], dtype=np.int64)) or 1
    chunk = max(1, int(chunk_bytes // (nfft * per_atom * 16)))
    acf = xp.zeros(T)
    for s in range(0, x.shape[1], chunk):
        F = xp.fft.rfft(x[:, s:s + chunk], n=nfft, axis=0)
        acf += xp.fft.irfft(F * F.conj(), n=nfft, axis=0)[:T].reshape(T, -1).sum(axis=1)
    return acf


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
    s2 = _acf_sum(x, xp, chunk_bytes) / lags             # position autocorrelation
    return be.to_numpy((s1 - 2.0 * s2) / n)


def particle_msd(positions, lags=None, backend=None, chunk_bytes=2**28):
    """Each particle's own time-averaged MSD of an unwrapped trajectory ``(T, N, 3)``, averaged
    over the lags ``lags`` (a slice of lag indices; all lags but 0 by default), as a length-N
    array: ``<|x_i(t+m) - x_i(t)|^2>_{t, m in lags}``. Its mean over the particles is the mean of
    :func:`msd_fft` over the same lags. Same FFT algorithm, in chunks of particles."""
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(positions, dtype=xp.float64)
    T, n = x.shape[0], x.shape[1]
    if n == 0 or T < 2:
        return np.zeros(n)
    m = np.arange(T)[lags if lags is not None else slice(1, None)]
    if len(m) == 0:
        raise ValueError(f"no lags in {lags} for {T} frames")
    nfft = 2 * T
    chunk = max(1, int(chunk_bytes // (nfft * 3 * 16)))
    count = xp.asarray((T - m).astype(np.float64))[:, None]
    m_x = xp.asarray(m)
    out = []
    for s in range(0, n, chunk):
        c = x[:, s:s + chunk]                                    # (T, k, 3)
        D = (c * c).sum(axis=2)                                  # (T, k)
        csum, rsum = xp.cumsum(D, axis=0), xp.cumsum(D[::-1], axis=0)
        Q = 2.0 * D.sum(axis=0)[None] - xp.concatenate([xp.zeros((1, c.shape[1])), csum[:-1] + rsum[:-1]])
        F = xp.fft.rfft(c, n=nfft, axis=0)
        acf = xp.fft.irfft(F * F.conj(), n=nfft, axis=0)[:T].sum(axis=2)   # (T, k)
        out.append(((Q[m_x] - 2.0 * acf[m_x]) / count).mean(axis=0))
    return be.to_numpy(xp.concatenate(out))


def fit_lags(n: int, fit_range=(0.2, 0.8), time=None, fit_time=None) -> slice:
    """The lag indices :func:`diffusion_fit` fits over for ``n`` lags: the fraction ``fit_range``
    of them, or with ``fit_time=(t0, t1)`` the lags whose ``time`` lies in ``t0..t1``, so runs of
    different length fit the same window."""
    if fit_time is not None:
        t = np.arange(n, dtype=float) if time is None else np.asarray(time, dtype=float)
        t0, t1 = (float(v) for v in fit_time)
        lo, hi = int(np.searchsorted(t, t0, "left")), int(np.searchsorted(t, t1, "right"))
        if hi - lo < 2:
            raise ValueError(f"fit_time={tuple(fit_time)} holds {max(hi - lo, 0)} lag(s); "
                             f"the lags run from {t[0]:g} to {t[-1]:g} in steps of {t[1] - t[0]:g}")
        return slice(lo, hi)
    lo = min(int(fit_range[0] * n), n - 2)
    return slice(max(lo, 0), max(int(np.ceil(fit_range[1] * n)), lo + 2))


def autocorrelation_fft(vectors, backend=None, chunk_bytes=2**28):
    """Time-averaged autocorrelation ``c[m] = < v(t) . v(t+m) >_{t, atoms}`` of a
    ``(T, N, 3)`` series (e.g. velocities for the VACF), via FFT. Returns length T."""
    be = get_backend(backend)
    xp = be.xp
    v = xp.asarray(vectors, dtype=xp.float64)
    T, n = v.shape[0], v.shape[1]
    if n == 0 or T == 0:
        return np.zeros(T)
    lags = xp.arange(T, 0, -1, dtype=xp.float64)
    return be.to_numpy(_acf_sum(v, xp, chunk_bytes) / lags / n)


def diffusion_fit(time, msd, fit_range=(0.2, 0.8), dimensions: int = 3, fit_time=None):
    """Einstein relation: fit ``msd = 2 d D t + c`` over the fraction ``fit_range`` of the lags,
    or over the lags with ``time`` in ``fit_time=(t0, t1)`` when given.

    Returns ``(D, intercept, (t_first, t_last))``; D is in length^2 per unit of ``time``.
    The first lags (ballistic) and the last ones (few time origins) are excluded by default.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(msd, dtype=float)
    if len(t) < 3:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    window = fit_lags(len(t), fit_range, t, fit_time)
    lo, hi = window.start, window.stop
    slope, icpt = np.polyfit(t[lo:hi], y[lo:hi], 1)
    return float(slope / (2 * dimensions)), float(icpt), (float(t[lo]), float(t[hi - 1]))


def green_kubo(time, vacf, dimensions: int = 3):
    """Green-Kubo diffusion coefficient ``D = (1/d) int_0^t <v(0).v(t)> dt`` (trapezoid rule).

    Returns the running integral (same length as ``time``); its plateau is D.
    """
    t = np.asarray(time, dtype=float)
    c = np.asarray(vacf, dtype=float)
    if len(t) < 2:
        return np.zeros(len(t))
    steps = 0.5 * (c[1:] + c[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(steps)]) / dimensions


def path_statistics(unwrapped, backend=None):
    """Per-particle path metrics of an unwrapped trajectory ``(T, N, 3)``.

    Returns numpy arrays ``(path_length, net_displacement, max_displacement)``:
    summed frame-to-frame step lengths, ``|x(T-1) - x(0)|`` and ``max_t |x(t) - x(0)|``.
    """
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(unwrapped, dtype=xp.float64)
    if x.shape[0] == 0:
        z = np.zeros(x.shape[1])
        return z, z.copy(), z.copy()
    path = xp.sqrt(((x[1:] - x[:-1]) ** 2).sum(-1)).sum(0)
    disp = xp.sqrt(((x - x[0][None]) ** 2).sum(-1))
    return be.to_numpy(path), be.to_numpy(disp[-1]), be.to_numpy(disp.max(0))


def collect_trajectory(frames, properties=("Position",), particle_ids=None, indices=None):
    """Stack per-particle ``properties`` of an iterable of frames into ``(T, N, ...)`` arrays.

    Reads one frame at a time (lazy LAMMPS dump / XDATCAR readers stay lazy). Particles are
    matched across frames by 'Particle Identifier' when present, in ascending id order, else
    by index. ``particle_ids`` (or ``indices`` without identifiers) restricts the output to
    those particles. Returns a dict with the stacked properties plus ``types`` (N,),
    ``Particle Identifier`` (N,) or None, ``Index`` (N,: indices in the first frame), ``cells`` (T, 3, 3), ``pbc`` and ``Timestep`` (T,).
    """
    out = {name: [] for name in properties}
    cells, steps = [], []
    first = None
    for k, data in enumerate(frames):
        # identifiers match particles only when the first frame has them too; else by index
        if "Particle Identifier" in data and (first is None or ident is not None):
            ids = np.asarray(data["Particle Identifier"])
            if first is None:
                want = np.unique(ids if particle_ids is None else np.asarray(particle_ids))
            order = np.argsort(ids, kind="stable")
            pos = np.searchsorted(ids[order], want)
            pos = np.clip(pos, 0, max(len(ids) - 1, 0))
            if len(ids) == 0 and len(want) or np.any(ids[order][pos] != want):
                raise ValueError(f"frame {k} is missing some particle identifiers of the first frame")
            take = order[pos]
        else:
            if first is None:
                want = np.arange(data.count) if indices is None else np.asarray(indices)
            elif ident is not None:
                raise ValueError(f"frame {k} has no 'Particle Identifier' but the first frame has")
            elif data.count != n0:
                raise ValueError(f"frame {k} has {data.count} particles, expected {n0} "
                                 "(no 'Particle Identifier' to match them)")
            take = want
        if first is None:
            first, n0, index = data, data.count, take
            types = np.asarray(data.types)[take]
            ident = np.asarray(data["Particle Identifier"])[take] if "Particle Identifier" in data else None
        for name in properties:
            if name not in data:
                raise KeyError(f"frame {k} has no property {name!r}")
            out[name].append(np.asarray(data[name])[take])
        cells.append(data.cell.matrix)
        steps.append(data.attributes.get("Timestep", k))
    if first is None:
        raise ValueError("no frames to analyze")
    result = {name: np.stack(v) for name, v in out.items()}
    result.update({"types": types, "Particle Identifier": ident, "Index": index, "cells": np.stack(cells),
                   "pbc": first.cell.pbc.copy(), "Timestep": np.asarray(steps)})
    return result


def _per_type(fn, x, types, be, prefix, table):
    """Add ``<prefix> <type>`` columns computed by ``fn`` on each type's particles."""
    names = np.unique(types)
    if len(names) > 1:
        xp = be.xp
        x = xp.asarray(x)
        for name in names:
            sel = xp.asarray(np.nonzero(types == name)[0])
            table[f"{prefix} {name}"] = fn(x[:, sel], be)
    return table


def msd_axes(fn, x, be, column, table):
    """Add ``<column> x``, ``<column> y`` and ``<column> z``: ``fn`` on each Cartesian component of
    ``x`` (the MSD along one axis; an isotropic diffuser has a third of the total on each)."""
    xp = be.xp
    x = xp.asarray(x)
    for k, axis in enumerate("xyz"):
        table[f"{column} {axis}"] = fn(x[:, :, k:k + 1], be)
    return table


def trajectory_msd(pipeline, start: int = 0, stop: int | None = None, step: int = 1,
                   device=None, by_type: bool = True, unwrapped: bool = False, by_axis: bool = False):
    """Time-averaged MSD over the source frames of ``pipeline``.

    Particles are matched across frames by 'Particle Identifier' when present.
    Positions are unwrapped with :func:`unwrap` unless ``unwrapped=True``
    (e.g. LAMMPS ``xu yu zu`` columns). Returns a table with ``lag`` (in frames,
    multiplied by ``step``), ``MSD`` and, with ``by_type``, ``MSD <type>``; ``by_axis``
    adds ``<column> x|y|z`` for each (:func:`msd_axes`).
    """
    be = get_backend(device or pipeline.device)
    stop = pipeline.num_frames if stop is None else stop
    traj = collect_trajectory(pipeline.source_frame(k) for k in range(start, stop, step))
    x = traj["Position"]
    if not unwrapped:
        x = unwrap(x, traj["cells"], traj["pbc"], be)
    table = {"lag": np.arange(len(traj["cells"])) * step, "MSD": msd_fft(x, be)}
    if by_type:
        _per_type(msd_fft, x, traj["types"], be, "MSD", table)
    if by_axis:
        _add_axes(table, x, traj["types"], be)
    return table


def _add_axes(table, x, types, be):
    """``msd_axes`` of every ``MSD`` / ``MSD <type>`` column of ``table``."""
    names = np.unique(types)
    xp = be.xp
    for col in [c for c in table if c == "MSD" or c.startswith("MSD ")]:
        if col == "MSD":
            msd_axes(msd_fft, x, be, col, table)
        else:
            sel = xp.asarray(np.nonzero(types == col[4:])[0])
            if len(names) > 1:
                msd_axes(msd_fft, xp.asarray(x)[:, sel], be, col, table)
    return table


def finite_difference_velocities(unwrapped, dt: float = 1.0, backend=None):
    """Velocities ``(x(t+1) - x(t)) / dt`` at the T-1 midpoints of an unwrapped trajectory."""
    be = get_backend(backend)
    x = be.xp.asarray(unwrapped, dtype=be.xp.float64)
    return (x[1:] - x[:-1]) / dt


def trajectory_vacf(pipeline, start: int = 0, stop: int | None = None, step: int = 1,
                    device=None, by_type: bool = True, time_per_frame: float = 1.0,
                    velocities: str = "auto"):
    """Velocity autocorrelation over the source frames of ``pipeline``.

    ``velocities="property"`` uses the 'Velocity' property (LAMMPS ``vx vy vz``),
    ``"positions"`` differentiates unwrapped positions (XDATCAR has no velocities),
    ``"auto"`` picks the property when the first frame has it. ``time_per_frame`` is the
    time between stored frames before ``step``. Returns a table with ``Time``, ``VACF``,
    ``VACF (normalized)`` and, with ``by_type``, ``VACF <type>``.
    """
    be = get_backend(device or pipeline.device)
    stop = pipeline.num_frames if stop is None else stop
    frames = range(start, stop, step)
    if velocities == "auto":
        velocities = "property" if "Velocity" in pipeline.source_frame(start) else "positions"
    return vacf_table((pipeline.source_frame(k) for k in frames), time_per_frame * step,
                      velocities, be, by_type)


def vacf_table(frames, dt: float, velocities: str, backend, by_type: bool = True):
    """VACF table from an iterable of frames ``dt`` apart (see :func:`trajectory_vacf`)."""
    be = get_backend(backend)
    if velocities == "property":
        traj = collect_trajectory(frames, ("Velocity",))
        v = traj["Velocity"]
    elif velocities == "positions":
        traj = collect_trajectory(frames)
        x = unwrap(traj["Position"], traj["cells"], traj["pbc"], be)
        v = finite_difference_velocities(x, dt, be)
    else:
        raise ValueError(f"velocities must be 'auto', 'property' or 'positions', not {velocities!r}")
    c = autocorrelation_fft(v, be)
    table = {"Time": np.arange(len(c)) * dt, "VACF": c,
             "VACF (normalized)": c / c[0] if len(c) and c[0] != 0 else c * 0.0}
    if by_type:
        _per_type(autocorrelation_fft, v, traj["types"], be, "VACF", table)
    return table


def trajectory_rdf(pipeline, cutoff: float, bins: int | None = None, start: int = 0, stop: int | None = None,
                   step: int = 1, partial: bool = False, device=None, **kwargs):
    """Time-averaged g(r) (total, plus partials with ``partial``) over the source frames, in
    ``bins`` shells (default 200; ``nbins`` is the old name)."""
    nbins = _bins(bins, kwargs)
    be = get_backend(device or pipeline.device)
    stop = pipeline.num_frames if stop is None else stop
    return average_tables(
        rdf(d.positions, d.cell, cutoff, nbins, types=d.types if partial else None, backend=be)
        for d in (pipeline.source_frame(k) for k in range(start, stop, step)))


def average_tables(tables):
    """Column-wise mean of tables with the same columns (the first column, e.g. ``r``,
    is kept from the first table). Columns missing from some tables average over the others."""
    sums, counts, first = {}, {}, None
    for t in tables:
        if first is None:
            first = t
        for k, v in t.items():
            v = np.asarray(v, dtype=float)
            if k in sums:
                if sums[k].shape != v.shape:
                    raise ValueError(f"column {k!r} changes shape between frames")
                sums[k] = sums[k] + v
            else:
                sums[k] = v.copy()
            counts[k] = counts.get(k, 0) + 1
    if first is None:
        raise ValueError("no frames to average")
    keys = list(first) + [k for k in sums if k not in first]
    key0 = keys[0]
    return {k: (np.asarray(first[k]) if k == key0 else sums[k] / counts[k]) for k in keys}


# --------------------------------------------------------------------------- ion transport

# e^2 / (A * ps * J): converts q^2 [e^2] * D [A^2/ps] / (V [A^3] * kB T [J]) to S/m
_SIGMA_SI = 1.602176634e-19 ** 2 * 1e-20 / 1e-12 / 1e-30
_KB = 1.380649e-23


def conductivity_from_slope(diffusivity, volume: float, temperature: float) -> float:
    """Ionic conductivity in S/cm from ``diffusivity`` = slope / 6 of a charge MSD
    (e^2 A^2/ps), the cell ``volume`` (A^3) and ``temperature`` (K): sigma = D_q / (V kB T)."""
    return float(diffusivity) * _SIGMA_SI / (volume * _KB * temperature) / 100.0


def displacement_moments(unwrapped, max_origins: int | None = 200, backend=None):
    """``<|dr|^2>`` and ``<|dr|^4>`` for every lag of an unwrapped trajectory ``(T, N, 3)``,
    averaged over atoms and up to ``max_origins`` evenly spaced time origins (all if None).
    Returns numpy arrays of length T (lag 0 is 0)."""
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(unwrapped, dtype=xp.float64)
    T, n = x.shape[0], x.shape[1]
    r2, r4, count = xp.zeros(T), xp.zeros(T), np.zeros(T)
    if T == 0 or n == 0:
        return np.zeros(T), np.zeros(T)
    for o in _origins(T - 1, max_origins):
        d2 = ((x[o:] - x[o][None]) ** 2).sum(-1)           # (T-o, N)
        r2[:T - o] += d2.sum(1)
        r4[:T - o] += (d2 * d2).sum(1)
        count[:T - o] += n
    count = np.maximum(count, 1)
    return be.to_numpy(r2) / count, be.to_numpy(r4) / count


def non_gaussian_parameter(r2, r4, dimensions: int = 3):
    """alpha_2(t) = d <dr^4> / ((d + 2) <dr^2>^2) - 1 (0 for Gaussian displacements; NaN at t=0)."""
    r2, r4 = np.asarray(r2, float), np.asarray(r4, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = dimensions * r4 / ((dimensions + 2) * r2 * r2) - 1.0
    return np.where(r2 > 0, a, np.nan)


def _origins(last: int, max_origins: int | None):
    """Time origins 0..last (inclusive), evenly thinned to at most ``max_origins``."""
    if last < 0:
        return np.zeros(0, int)
    if max_origins is None or last + 1 <= max_origins:
        return np.arange(last + 1)
    return np.unique(np.linspace(0, last, int(max_origins)).round().astype(int))


def van_hove_self(unwrapped, lags, r_max: float, nbins: int = 100, max_origins: int | None = 100,
                  backend=None):
    """Self part of the van Hove function for each lag (frames) in ``lags``.

    Returns ``(r, P)`` with bin centers ``r`` and ``P`` of shape ``(len(lags), nbins)``: the
    radial probability density ``4 pi r^2 G_s(r, t)`` of the displacement length, normalized
    so its integral up to ``r_max`` is the fraction of displacements shorter than ``r_max``.
    """
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(unwrapped, dtype=xp.float64)
    T, n = x.shape[0], x.shape[1]
    edges = np.linspace(0.0, r_max, nbins + 1)
    dr = edges[1] - edges[0]
    out = np.zeros((len(lags), nbins))
    for row, m in enumerate(lags):
        m = int(m)
        if m >= T or n == 0:
            out[row] = np.nan
            continue
        origins = _origins(T - 1 - m, max_origins)
        o = xp.asarray(origins)
        d = xp.sqrt(((x[o + m] - x[o]) ** 2).sum(-1)).ravel()
        h = xp.histogram(d, bins=xp.asarray(edges))[0]
        out[row] = be.to_numpy(h) / (len(origins) * n * dr)
    return 0.5 * (edges[1:] + edges[:-1]), out


def van_hove_distinct(positions, cells, pbc, lags, r_max: float, nbins: int = 100,
                      max_origins: int | None = 20, backend=None):
    """Distinct part ``G_d(r, t)`` for each lag in ``lags``, normalized like g(r)
    (``G_d(r, 0) = g(r)``, tending to 1 at large r).

    ``positions`` is a wrapped or unwrapped trajectory ``(T, N, 3)`` (minimum image is taken in
    the cell of the origin frame); ``r_max`` is clipped to half the narrowest periodic width.
    Returns ``(r, G)`` with ``G`` of shape ``(len(lags), nbins)``.
    """
    be = get_backend(backend)
    x = np.asarray(positions, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if cells.ndim == 2:
        cells = np.broadcast_to(cells, (len(x), 3, 3))
    T, n = x.shape[0], x.shape[1]
    edges = np.linspace(0.0, r_max, nbins + 1)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    out = np.zeros((len(lags), nbins))
    for row, m in enumerate(lags):
        m = int(m)
        if m >= T or n < 2:
            out[row] = np.nan
            continue
        hist = np.zeros(nbins)
        origins = _origins(T - 1 - m, max_origins)
        volume = 0.0
        for o in origins:
            hist += _cross_histogram(x[o], x[o + m], cells[o], pbc, r_max, nbins, be)
            volume += abs(np.linalg.det(cells[o]))
        rho = (n - 1) / (volume / len(origins))       # n (n - 1) ordered pairs, as in rdf()
        out[row] = hist / (len(origins) * n * rho * shell)
    return 0.5 * (edges[1:] + edges[:-1]), out


def _cross_histogram(a, b, matrix, pbc, r_max: float, nbins: int, be, chunk_bytes=2**26):
    """Histogram of the minimum-image distances |b_j - a_i|, i != j, below ``r_max``."""
    xp = be.xp
    inv = xp.asarray(np.linalg.inv(matrix))
    mat = xp.asarray(matrix)
    per = xp.asarray(np.asarray(pbc, dtype=float))
    fa, fb = xp.matmul(xp.asarray(a), inv), xp.matmul(xp.asarray(b), inv)
    n = len(a)
    hist = xp.zeros(nbins, dtype=xp.int64)
    chunk = max(1, int(chunk_bytes // (len(b) * 3 * 8 * 4)))
    cols = xp.arange(len(b))
    for s in range(0, n, chunk):
        f = fb[None] - fa[s:s + chunk, None]
        f = f - xp.round(f) * per
        d = xp.sqrt((xp.matmul(f, mat) ** 2).sum(-1))
        d = xp.where(cols[None] == xp.arange(s, min(s + chunk, n))[:, None], r_max, d)   # j == i
        k = (d / (r_max / nbins)).astype(xp.int64)
        k = k[k < nbins]
        hist += xp.bincount(k, minlength=nbins)[:nbins]
    return be.to_numpy(hist).astype(float)


def fibonacci_directions(count: int):
    """``count`` nearly uniform unit vectors on the upper half sphere (``q`` and ``-q`` give the
    same self intermediate scattering function)."""
    k = np.arange(count) + 0.5
    z = 1.0 - k / count                                   # (0, 1]
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def self_intermediate_scattering(unwrapped, q, directions: int = 16, backend=None, chunk_bytes=2**27):
    """Self intermediate scattering function ``F_s(q, t) = < exp(i q.(r(t0+t) - r(t0))) >`` for
    every lag of an unwrapped trajectory ``(T, N, 3)``, averaged over atoms, every time origin
    (FFT) and ``directions`` orientations of ``q`` (powder average). ``q`` is a number or a
    sequence of moduli (1/length). Returns an array ``(len(q), T)``."""
    be = get_backend(backend)
    xp = be.xp
    x = xp.asarray(unwrapped, dtype=xp.float64)
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    T, n = x.shape[0], x.shape[1]
    if T == 0 or n == 0:
        return np.zeros((len(qs), T))
    dirs = xp.asarray(fibonacci_directions(max(1, int(directions))))
    proj = xp.matmul(x, dirs.T)                           # (T, N, D)
    nfft = 2 * T
    chunk = max(1, int(chunk_bytes // (nfft * proj.shape[2] * 32)))
    lags = xp.arange(T, 0, -1, dtype=xp.float64)
    out = np.zeros((len(qs), T))
    for row, qv in enumerate(qs):
        acc = xp.zeros(T)
        for s in range(0, n, chunk):
            z = xp.exp(1j * qv * proj[:, s:s + chunk])
            F = xp.fft.fft(z, n=nfft, axis=0)
            acc += xp.fft.ifft(F * F.conj(), n=nfft, axis=0)[:T].real.reshape(T, -1).sum(axis=1)
        out[row] = be.to_numpy(acc / lags) / (n * proj.shape[2])
    return out


def decay_time(time, values, level: float = np.exp(-1.0)) -> float:
    """First time at which ``values`` falls below ``level`` (linear interpolation), else NaN."""
    t, y = np.asarray(time, float), np.asarray(values, float)
    below = np.nonzero(y < level)[0]
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    k = below[0]
    return float(t[k - 1] + (y[k - 1] - level) / (y[k - 1] - y[k]) * (t[k] - t[k - 1]))


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
