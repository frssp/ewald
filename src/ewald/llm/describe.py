"""Compact, JSON-ready descriptions of structures and trajectories.

An LLM cannot read 10^6 coordinates, and should not have to. These functions
compress a snapshot (or a trajectory) into a few hundred tokens of physically
meaningful numbers: composition, cell, density, nearest-neighbor distances,
coordination statistics, RDF peaks, property ranges and sanity warnings.
Everything here is plain numpy and works without any LLM.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..data import DataCollection
from ..kernels import rdf
from ..neighbors import find_pairs

#: Below this distance (Angstrom) two atoms are reported as overlapping.
OVERLAP_DISTANCE = 0.5


def _round(x, nd=4):
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    x = float(x)
    return float(f"{x:.{nd}g}") if np.isfinite(x) else None


def cell_parameters(cell) -> dict:
    """Lattice lengths (Angstrom), angles (degrees), volume and periodicity."""
    if not cell.is_valid:
        return {"valid": False, "pbc": [bool(p) for p in cell.pbc]}
    a, b, c = cell.matrix
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))

    def ang(u, v):
        return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))

    return {"a": _round(la), "b": _round(lb), "c": _round(lc),
            "alpha": _round(ang(b, c)), "beta": _round(ang(a, c)), "gamma": _round(ang(a, b)),
            "volume": _round(cell.volume), "pbc": [bool(p) for p in cell.pbc]}


def _masses(types) -> np.ndarray | None:
    from ase.data import atomic_masses, chemical_symbols

    lookup = {s: atomic_masses[i] for i, s in enumerate(chemical_symbols)}
    m = np.array([lookup.get(str(t), np.nan) for t in types])
    return None if np.isnan(m).any() else m


def suggest_cutoff(data: DataCollection, rmax: float = 6.0, nbins: int = 240) -> float | None:
    """First minimum of g(r) after its first peak: a natural first-shell cutoff."""
    if data.count < 2:
        return None
    rmax = min(rmax, 0.499 * float(np.min(data.cell.perpendicular_widths()))) \
        if data.cell.is_valid and data.cell.pbc.all() else rmax
    table = rdf(data.positions, data.cell, rmax, nbins)
    r, g = np.asarray(table["r"]), np.asarray(table["g(r)"])
    if not np.any(g > 0):
        return None
    g = np.convolve(g, np.ones(3) / 3, mode="same")
    gmax = g.max()  # first clear peak (not the global max: diamond's 2nd shell is higher)
    ipk = next(k for k in range(1, len(g) - 1)
               if g[k] >= 0.3 * gmax and g[k] >= g[k - 1] and g[k] >= g[k + 1])
    after = g[ipk:]
    # first local minimum after the peak that is clearly below it
    for k in range(1, len(after) - 1):
        if after[k] <= after[k - 1] and after[k] <= after[k + 1] and after[k] < 0.6 * after[0]:
            end = k  # crystals: g(r) sits on a flat zero between shells -> take the gap's middle
            while end + 1 < len(after) and after[end + 1] <= after[k] + 1e-9:
                end += 1
            return float(r[ipk + (k + end) // 2])
    return None


def coordination_summary(data: DataCollection, cutoff: float) -> dict:
    """Coordination-number histogram per particle type within ``cutoff``."""
    i, j, _, _ = find_pairs(data.positions, data.cell, cutoff, half=True)
    i, j = np.asarray(i), np.asarray(j)
    cn = np.bincount(i, minlength=data.count) + np.bincount(j, minlength=data.count)
    types = np.asarray(data.types)
    out = {"cutoff": _round(cutoff), "mean": _round(cn.mean()) if data.count else None, "by_type": {}}
    for t in np.unique(types):
        c = cn[types == t]
        hist = Counter(int(x) for x in c)
        out["by_type"][str(t)] = {"mean": _round(c.mean()),
                                  "histogram": {str(k): v for k, v in sorted(hist.items())}}
    return out


def rdf_peaks(data: DataCollection, rmax: float = 6.0, nbins: int = 240, max_peaks: int = 4,
              partial: bool = True) -> dict:
    """Positions and heights of the main g(r) peaks (total and per pair)."""
    rmax = min(rmax, 0.499 * float(np.min(data.cell.perpendicular_widths()))) \
        if data.cell.is_valid and data.cell.pbc.all() else rmax
    types = data.types if partial and len(np.unique(data.types)) > 1 else None
    table = rdf(data.positions, data.cell, rmax, nbins, types=types)
    r = np.asarray(table["r"])
    out = {}
    for name, g in table.items():
        if name == "r":
            continue
        g = np.convolve(np.asarray(g, float), np.ones(3) / 3, mode="same")
        peaks = [k for k in range(1, len(g) - 1) if g[k] > g[k - 1] and g[k] >= g[k + 1] and g[k] > 1.2]
        peaks = sorted(peaks, key=lambda k: -g[k])[:max_peaks]
        out[name] = [{"r": _round(r[k], 3), "g": _round(g[k], 3)} for k in sorted(peaks)]
    return out


def health_check(data: DataCollection) -> list[str]:
    """Cheap sanity checks; returns human-readable warnings (empty if all is fine)."""
    warn = []
    if data.count == 0:
        return ["structure contains no particles"]
    pos = data.positions
    if not np.all(np.isfinite(pos)):
        warn.append(f"{int((~np.isfinite(pos)).any(axis=1).sum())} particles have NaN/inf positions")
        return warn
    if data.count > 1:
        i, j, _, d = find_pairs(pos, data.cell, OVERLAP_DISTANCE, half=True)
        if len(d):
            warn.append(f"{len(d)} atom pairs closer than {OVERLAP_DISTANCE} A (overlapping atoms; "
                        "duplicated frame, wrong cell, or exploded simulation)")
    if data.cell.is_valid:
        f = data.cell.to_fractional(pos)
        outside = ((f < -0.01) | (f > 1.01))[:, data.cell.pbc].any(axis=1)
        if outside.any():
            warn.append(f"{int(outside.sum())} particles lie outside the periodic cell (unwrapped coordinates?)")
    if "Particle Identifier" in data:
        ids = np.asarray(data["Particle Identifier"])
        if len(np.unique(ids)) != len(ids):
            warn.append("duplicate particle identifiers")
    if np.all(np.asarray(data.types) == "X"):
        warn.append("no element names (types are unknown); pass type_map={1: 'Cu', ...} for LAMMPS files")
    return warn


def _property_summary(data: DataCollection, max_props: int = 24) -> dict:
    out = {}
    for name, arr in list(data.particles.items())[:max_props]:
        a = np.asarray(arr)
        if a.dtype.kind in "biuf" and a.size:
            v = np.linalg.norm(a.reshape(len(a), -1), axis=1) if a.ndim == 2 else a
            entry = {"components": int(a.shape[1]) if a.ndim == 2 else 1,
                     "min": _round(np.nanmin(v)), "max": _round(np.nanmax(v)), "mean": _round(np.nanmean(v))}
            if a.ndim == 2:
                entry["note"] = "statistics of the vector magnitude"
            out[name] = entry
        elif a.dtype.kind in "US":
            out[name] = {"values": dict(Counter(str(x) for x in a).most_common(8))}
    return out


def describe_frame(data: DataCollection, cutoff: float | str | None = "auto", rdf_peaks_: bool = True,
                   frame: int | None = None) -> dict:
    """Describe one snapshot in a few hundred tokens.

    ``cutoff`` sets the coordination cutoff in Angstrom; ``"auto"`` takes the
    first minimum of g(r); ``None`` skips coordination analysis.
    """
    types = np.asarray(data.types)
    comp = dict(sorted(Counter(str(t) for t in types).items(), key=lambda kv: -kv[1]))
    desc: dict = {"num_particles": data.count, "composition": comp, "cell": cell_parameters(data.cell)}
    if frame is not None:
        desc["frame"] = frame
    if data.cell.is_valid and data.count:
        desc["number_density_per_A3"] = _round(data.count / data.cell.volume)
        m = _masses(types)
        if m is not None:
            desc["density_g_cm3"] = _round(m.sum() * 1.66053906660 / data.cell.volume)
    warnings = health_check(data)
    if data.count > 1 and not any("NaN" in w for w in warnings):
        i, j, _, d = find_pairs(data.positions, data.cell, 4.0, half=True)
        d = np.asarray(d)
        if len(d):
            k = int(np.argmin(d))
            pair = sorted((str(types[int(np.asarray(i)[k])]), str(types[int(np.asarray(j)[k])])))
            desc["min_distance"] = {"value": _round(d[k]), "pair": "-".join(pair)}
        else:
            desc["min_distance"] = {"value": None, "note": "no pairs within 4 A"}
        if cutoff == "auto":
            cutoff = suggest_cutoff(data)
            if cutoff is not None:
                desc["cutoff_source"] = "first minimum of g(r)"
        if cutoff:
            desc["coordination"] = coordination_summary(data, float(cutoff))
        if rdf_peaks_:
            desc["rdf_peaks"] = rdf_peaks(data)
    desc["properties"] = _property_summary(data)
    attrs = {k: (_round(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v)
             for k, v in data.attributes.items() if np.isscalar(v)}
    if attrs:
        desc["attributes"] = attrs
    desc["warnings"] = warnings
    return desc


# ------------------------------------------------------------------ trajectories

def detect_change_points(series, window: int = 5, threshold: float = 6.0) -> list[int]:
    """Indices where the mean of ``series`` shifts abruptly.

    Sliding two-window test: at each index the means of the ``window`` samples
    before and after are compared, Welch-style, against the spread inside the
    two windows (floored by the series' global noise from the MAD of first
    differences, so flat stretches don't divide by zero). Using the local
    spread keeps noisy regimes (a liquid) and steady ramps (diffusive MSD) from
    firing; peaks above ``threshold`` survive non-maximum suppression.
    """
    x = np.asarray(series, float)
    n = len(x)
    if n < 2 * window + 1 or not np.all(np.isfinite(x)):
        return []
    dx = np.diff(x)
    noise = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2)
    floor = max(noise, 1e-3 * (np.max(np.abs(x)) + 1e-12), 1e-12)
    score = np.zeros(n)
    for t in range(window, n - window + 1):
        b, a = x[t - window:t], x[t:t + window]
        spread = np.sqrt((b.var(ddof=1) + a.var(ddof=1)) / window) if window > 1 else 0.0
        score[t] = abs(a.mean() - b.mean()) / max(spread, floor * np.sqrt(2.0 / window))
    out = []
    for t in np.argsort(-score):
        if score[t] < threshold:
            break
        if all(abs(t - s) > window for s in out):
            out.append(int(t))
    return sorted(out)


def trajectory_series(pipeline, start: int = 0, stop: int | None = None, step: int | None = None,
                      max_frames: int = 200, cutoff: float | None = None) -> dict:
    """Per-frame descriptor time series over the *source* frames of ``pipeline``.

    Columns: frame, volume, min_distance, displacement statistics against the
    first sampled frame (minimum image, matched by identifier), and with
    ``cutoff`` the mean coordination and the fraction of particles whose
    coordination differs from the first frame's most common value.
    """
    stop = pipeline.num_frames if stop is None else min(stop, pipeline.num_frames)
    if step is None:
        step = max(1, (stop - start) // max_frames)
    frames = list(range(start, stop, step))
    cols: dict[str, list] = {"frame": [], "volume": [], "min_distance": [], "msd": [], "max_displacement": []}
    if cutoff:
        cols.update({"mean_coordination": [], "fraction_off_mode": []})
    ref = None
    mode = None
    for f in frames:
        data = pipeline.source_frame(f)
        pos = np.asarray(data.positions, float)
        if "Particle Identifier" in data:
            pos = pos[np.argsort(data["Particle Identifier"], kind="stable")]
        cols["frame"].append(f)
        cols["volume"].append(data.cell.volume if data.cell.is_valid else np.nan)
        if data.count > 1:
            _, _, _, d = find_pairs(data.positions, data.cell, 4.0, half=True)
            cols["min_distance"].append(float(np.min(d)) if len(d) else np.nan)
        else:
            cols["min_distance"].append(np.nan)
        if ref is None:
            ref = pos
        if len(pos) == len(ref):
            dr = pos - ref
            if data.cell.is_valid:
                fr = dr @ np.linalg.inv(data.cell.matrix)
                fr[:, data.cell.pbc] -= np.round(fr[:, data.cell.pbc])
                dr = fr @ data.cell.matrix
            d2 = (dr ** 2).sum(1)
            cols["msd"].append(float(d2.mean()))
            cols["max_displacement"].append(float(np.sqrt(d2.max())))
        else:
            cols["msd"].append(np.nan)
            cols["max_displacement"].append(np.nan)
        if cutoff:
            i, j, _, _ = find_pairs(data.positions, data.cell, cutoff, half=True)
            cn = np.bincount(np.asarray(i), minlength=data.count) + np.bincount(np.asarray(j), minlength=data.count)
            if mode is None:
                mode = int(np.bincount(cn).argmax())
            cols["mean_coordination"].append(float(cn.mean()))
            cols["fraction_off_mode"].append(float(np.mean(cn != mode)))
    return {k: np.asarray(v) for k, v in cols.items()}


def describe_trajectory(pipeline, start: int = 0, stop: int | None = None, step: int | None = None,
                        max_frames: int = 200, cutoff: float | str | None = "auto",
                        window: int = 5, threshold: float = 6.0) -> dict:
    """Summarize a trajectory and flag frames where something abrupt happens.

    Returns per-series statistics (start/end/min/max), the sampled frame
    indices, and ``events``: change points with the series that jumped and the
    values before/after. The events are what an LLM (or a person) should look
    at first; render the frames around them.
    """
    first = pipeline.source_frame(start)
    if cutoff == "auto":
        cutoff = suggest_cutoff(first)
    ts = trajectory_series(pipeline, start, stop, step, max_frames, cutoff)
    frames = ts["frame"]
    summary = {}
    events: dict[int, dict] = {}
    for name, x in ts.items():
        if name == "frame" or not np.isfinite(x).any():
            continue
        summary[name] = {"start": _round(x[0]), "end": _round(x[-1]),
                         "min": _round(np.nanmin(x)), "max": _round(np.nanmax(x))}
        w = max(2, min(window, len(x) // 4))
        for t in detect_change_points(x, w, threshold):
            if name in ("msd", "max_displacement") and t <= w:
                continue  # the rise away from the reference frame itself is not an event
            before, after = float(np.mean(x[t - w:t])), float(np.mean(x[t:t + w]))
            ev = events.setdefault(int(frames[t]), {"frame": int(frames[t]), "changes": {}})
            ev["changes"][name] = {"before": _round(before), "after": _round(after)}
    return {"num_frames": pipeline.num_frames, "sampled_frames": len(frames),
            "frame_step": int(frames[1] - frames[0]) if len(frames) > 1 else 1,
            "coordination_cutoff": _round(cutoff) if cutoff else None,
            "series": summary,
            "events": [events[k] for k in sorted(events)],
            "first_frame_warnings": health_check(first)}
