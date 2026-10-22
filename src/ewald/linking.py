"""Linked selection between plots and the 3D view.

A plot of a data table (or a histogram of a particle property) is linked to the particles or
frames it was computed from, so a range dragged on the plot selects them and a selection in
the viewport can be drawn back on the plot:

- property histograms (``histogram[<property>]``, the Plots tab's "histogram of", CSP, PTM
  RMSD): particles whose value is in the range;
- g(r) (``coordination-rdf``, also time-averaged): particles in a pair at that distance, for
  partial columns ``g(r) A-B`` only A-B pairs;
- ``bond-length-distr``: the two particles of every bond with that length;
- ``bond-angle-distr``: the vertex particle of every bond angle in the range;
- tables with one row per frame or timestep (``time-series``, ``plumed``, ``colvar``, the
  ``Steps`` table of log files, MSD/VACF): the frames whose points lie in the dragged box;
- 2D free energy surfaces (``fes``): the frames whose CV pair (from ``colvar`` or ``plumed``)
  lies in the box.

Everything here is GUI-free so the mapping can be tested and reused from scripts.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

# tables whose x column is a particle property under another name
_TABLE_PROPERTY = {"csp-centrosymmetry": "Centrosymmetry", "ptm-rmsd": "RMSD"}
_FRAME_COLUMNS = ("Frame",)
_STEP_COLUMNS = ("Step", "Timestep")


@dataclass(frozen=True)
class Link:
    """What a plot is linked to.

    ``kind``: "property", "pairs", "bonds", "angles" (particle selections from an x range),
    "frames" (frames from points in a box) or "frames2d" (frames from a CV pair in a box).
    ``name`` is the property (for "property") or the table; ``types`` restricts pairs, bonds
    and angles to a partial column's particle types (vertex in the middle for angles).
    """

    kind: str
    name: str
    column: str | None = None
    types: tuple = ()

    @property
    def selects_atoms(self) -> bool:
        return self.kind in ("property", "pairs", "bonds", "angles")


def link_for(data, table: str | None = None, column: str | None = None,
             hist: str | None = None, num_frames: int | None = None) -> Link | None:
    """The :class:`Link` for the plot of ``data.tables[table]`` (y ``column``, None or "(all)" for
    every column) or for the histogram of particle property ``hist``; None if not linkable."""
    if data is None:
        return None
    if hist:
        return Link("property", hist) if _has_property(data, hist) else None
    t = data.tables.get(table) if table else None
    if not t:
        return None
    keys = list(t)
    x = keys[0]
    col = column if column in t else None
    if grid_map(t):
        cv = _cv_table(data, keys[:2])
        return Link("frames2d", table, cv) if cv else None
    prop = _TABLE_PROPERTY.get(table) or (table[10:-1] if table.startswith("histogram[") else x)
    if _has_property(data, prop) and x not in _FRAME_COLUMNS + _STEP_COLUMNS:
        return Link("property", prop, col)
    if x == "r" and any(k.startswith("g(r)") for k in keys):
        return Link("pairs", table, col, _partial(col, "g(r) ", 2))
    if x == "Bond Length":
        return Link("bonds", table, col, _partial(col, "Count ", 2))
    if x in ("Angle", "Cosine") and table.startswith("bond-angle-distr"):
        return Link("angles", table, col, _partial(col, "Count ", 3))
    n = len(np.asarray(t[x]))
    if x in _FRAME_COLUMNS + _STEP_COLUMNS or table == "colvar" or (num_frames and n == num_frames > 1) \
            or any(c in t for c in _FRAME_COLUMNS):  # e.g. the NEB profile: one row per image frame
        return Link("frames", table, col)
    return None


def _has_property(data, name) -> bool:
    from .modifiers.statistics import property_values

    try:
        v, _ = property_values(data.particles, name)
    except (KeyError, ValueError):
        return False
    return v.ndim == 1 and v.dtype.kind in "biuf"


def _partial(column, prefix, parts) -> tuple:
    if not column or not column.startswith(prefix):
        return ()
    names = tuple(column[len(prefix):].split("-"))
    return names if len(names) == parts else ()


def grid_map(table):
    """``(names, (x, y, z))`` when ``table`` is a map on a regular 2D grid (three columns, the
    first two covering every combination of their values, e.g. a 2D free energy surface)."""
    keys = list(table)
    if len(keys) != 3:
        return None
    x, y, z = (np.asarray(table[k], float) for k in keys)
    ux, uy = np.unique(x), np.unique(y)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(z):
        return None
    ix, iy = np.searchsorted(ux, x), np.searchsorted(uy, y)
    grid = np.full((len(uy), len(ux)), np.nan)
    grid[iy, ix] = z
    if np.isnan(grid).any():
        return None
    return keys, (ux, uy, grid)


def _cv_table(data, cvs):
    for name in ("colvar", "plumed"):
        t = data.tables.get(name)
        if t and all(c in t for c in cvs):
            return name
    return None


# ------------------------------------------------------------------ particles
def atoms_in_range(data, link: Link, lo: float, hi: float) -> np.ndarray:
    """Indices of the particles that ``link``'s x range ``lo``..``hi`` selects."""
    lo, hi = sorted((float(lo), float(hi)))
    n = data.count
    if link.kind == "property":
        from .modifiers.statistics import property_values

        v = property_values(data.particles, link.name)[0].astype(float)
        return np.nonzero((v >= lo) & (v <= hi))[0]
    if link.kind == "pairs":
        i, j, d = _pairs(data, hi)
        keep = (d >= lo) & (d <= hi) & _type_mask(data, i, j, link.types)
        return np.unique(np.concatenate([i[keep], j[keep]])) if n else np.zeros(0, int)
    if link.kind == "bonds":
        if data.bonds is None:
            return np.zeros(0, int)
        length = data.bonds.lengths(data)
        i, j = data.bonds.topology.T
        keep = (length >= lo) & (length <= hi) & _type_mask(data, i, j, link.types)
        return np.unique(np.concatenate([i[keep], j[keep]]))
    if link.kind == "angles":
        v, _, x = _angles(data, link)
        return np.unique(v[(x >= lo) & (x <= hi)])
    raise ValueError(f"{link.kind!r} plots select frames, not particles")


def values_of(data, link: Link, indices, xmax: float | None = None) -> np.ndarray:
    """The x values ``indices`` contribute to ``link``'s plot (their property values, the
    distances, bond lengths or angles they take part in), to draw a selection on the plot."""
    idx = np.asarray(indices, dtype=int)
    if not len(idx):
        return np.zeros(0)
    sel = np.zeros(data.count, bool)
    sel[idx[idx < data.count]] = True
    if link.kind == "property":
        from .modifiers.statistics import property_values

        return property_values(data.particles, link.name)[0][sel].astype(float)
    if link.kind == "pairs":
        if not xmax:
            return np.zeros(0)
        i, j, d = _pairs(data, xmax)
        keep = (sel[i] | sel[j]) & _type_mask(data, i, j, link.types)
        return d[keep]
    if link.kind == "bonds":
        if data.bonds is None:
            return np.zeros(0)
        i, j = data.bonds.topology.T
        keep = (sel[i] | sel[j]) & _type_mask(data, i, j, link.types)
        return data.bonds.lengths(data)[keep]
    if link.kind == "angles":
        v, _, x = _angles(data, link)
        return x[sel[v]]
    return np.zeros(0)


def selection_expression(link: Link, lo: float, hi: float) -> str | None:
    """An :class:`~ewald.modifiers.ExpressionSelection` expression for a property range, so the
    selection can be kept in the pipeline (re-evaluated on every frame); None for other links."""
    if link is None or link.kind != "property":
        return None
    lo, hi = sorted((float(lo), float(hi)))
    key = link.name.replace(" ", "")
    return f"({key} >= {lo:.6g}) & ({key} <= {hi:.6g})"


def _pairs(data, cutoff):
    from .neighbors import find_pairs

    if data.count < 2 or cutoff <= 0:
        e = np.zeros(0, int)
        return e, e, np.zeros(0)
    widths = data.cell.perpendicular_widths()[data.cell.pbc] if data.cell.is_valid else ()
    if len(widths):
        cutoff = min(cutoff, 0.4999 * float(np.min(widths)))
    i, j, _, d = find_pairs(data.positions, data.cell, cutoff, half=True)
    return np.asarray(i), np.asarray(j), np.asarray(d)


def _type_mask(data, i, j, types):
    if not types:
        return np.ones(len(i), bool)
    t = np.asarray(data.types).astype(str)
    a, b = types
    return ((t[i] == a) & (t[j] == b)) | ((t[i] == b) & (t[j] == a))


def _angles(data, link):
    """(vertex, ends, x) of every bond angle, x in degrees or as cosine like the plot's axis."""
    from .modifiers.topology import bond_angles

    if data.bonds is None:
        e = np.zeros(0, int)
        return e, np.zeros((0, 2), int), np.zeros(0)
    vertex, other, _, a, b, cos = bond_angles(data, data.bonds)
    v, ends = vertex[a], np.stack([other[a], other[b]], 1)
    if link.types:
        t = np.asarray(data.types).astype(str)
        e1, mid, e2 = link.types
        keep = (t[v] == mid) & (((t[ends[:, 0]] == e1) & (t[ends[:, 1]] == e2))
                                | ((t[ends[:, 0]] == e2) & (t[ends[:, 1]] == e1)))
        v, ends, cos = v[keep], ends[keep], cos[keep]
    x = cos if link.name in data.tables and "Cosine" in data.tables[link.name] else np.degrees(np.arccos(cos))
    return v, ends, x


# ------------------------------------------------------------------ frames
def frame_rows(pipeline, data, table: str) -> np.ndarray:
    """For every frame of ``pipeline``, the row of ``data.tables[table]`` that belongs to it
    (-1 when none): by a 'Frame' column, by timestep for 'Step'/'Timestep' tables, through the
    pipeline's LoadColvar modifier (ewald-plumed plugin; any enabled modifier with a ``row(frame)`` method
    that writes the 'colvar' table) for 'colvar', else row k for frame k."""
    nf = pipeline.num_frames if pipeline is not None else 1
    t = data.tables[table]
    x0 = next(iter(t))
    nrows = len(np.asarray(t[x0]))
    rows = np.full(nf, -1, dtype=np.int64)
    fc = x0 if x0 in _FRAME_COLUMNS else next((c for c in _FRAME_COLUMNS if c in t), None)
    if fc is not None:
        f = np.asarray(t[fc]).astype(np.int64)
        ok = (f >= 0) & (f < nf)
        rows[f[ok][::-1]] = np.nonzero(ok)[0][::-1]  # first row of each frame wins
        return rows
    if table == "colvar" and pipeline is not None:
        m = next((m for m in pipeline.modifiers
                  if type(m).__name__ == "LoadColvar" and m.enabled and callable(getattr(m, "row", None))), None)
        if m is not None:
            for k in range(nf):
                r = m.row(k)
                rows[k] = r if r is not None and r < nrows else -1
            return rows
    if x0 in _STEP_COLUMNS:
        steps = np.asarray(t[x0], dtype=float)
        fsteps = frame_timesteps(pipeline, data)
        if fsteps is not None and len(steps):
            order = np.argsort(steps, kind="stable")
            pos = np.clip(np.searchsorted(steps[order], fsteps), 0, len(steps) - 1)
            prev = np.clip(pos - 1, 0, len(steps) - 1)
            take = np.where(np.abs(steps[order][prev] - fsteps) < np.abs(steps[order][pos] - fsteps), prev, pos)
            rows[:] = order[take]
            return rows
    k = min(nf, nrows)
    rows[:k] = np.arange(k)
    return rows


_TIMESTEPS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def frame_timesteps(pipeline, data=None):
    """The 'Timestep' of every frame, assuming frames are evenly spaced in timestep (read from
    the first and last frame only, so it stays cheap on long trajectories); None without one."""
    if pipeline is None:
        ts = data.attributes.get("Timestep") if data is not None else None
        return None if ts is None else np.array([float(ts)])
    nf = pipeline.num_frames
    key = (id(pipeline.source), nf)
    hit = _TIMESTEPS.get(pipeline)
    if hit is not None and hit[0] == key:
        return hit[1]
    t0 = pipeline.source_frame(0).attributes.get("Timestep")
    t1 = pipeline.source_frame(nf - 1).attributes.get("Timestep") if nf > 1 else t0
    out = None if t0 is None or t1 is None else np.linspace(float(t0), float(t1), nf)
    _TIMESTEPS[pipeline] = (key, out)
    return out


def frames_in_box(pipeline, data, link: Link, x1, x2, y1=-np.inf, y2=np.inf,
                  columns=None) -> list[int]:
    """Frames whose plotted points lie in the box: the table's x column against each of
    ``columns`` (default: the link's column, or every column) for "frames"; the two CVs of a
    free energy surface for "frames2d"."""
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    if link.kind == "frames2d":
        table = link.column
        keys = list(data.tables[link.name])[:2]
        t = data.tables[table]
        a, b = (np.asarray(t[k], float) for k in keys)
        hit = (a >= x1) & (a <= x2) & (b >= y1) & (b <= y2)
    else:
        table = link.name
        t = data.tables[table]
        keys = list(t)
        x = np.asarray(t[keys[0]], float)
        cols = columns or ([link.column] if link.column else keys[1:])
        hit = np.zeros(len(x), bool)
        for c in cols:
            y = np.asarray(t[c], float)
            hit |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    rows = frame_rows(pipeline, data, table)
    ok = rows >= 0
    frames = np.nonzero(ok)[0]
    return frames[hit[rows[ok]]].tolist()


def frame_at(pipeline, data, link: Link, x: float) -> int | None:
    """The frame nearest to ``x`` on a "frames" plot's x axis (a click on a time series)."""
    if link is None or link.kind != "frames":
        return None
    t = data.tables[link.name]
    xs = np.asarray(t[next(iter(t))], float)
    rows = frame_rows(pipeline, data, link.name)
    frames = np.nonzero(rows >= 0)[0]
    if not len(frames):
        return None
    return int(frames[np.argmin(np.abs(xs[rows[frames]] - float(x)))])


def frame_xy(pipeline, data, link: Link):
    """``(frames, x, y)`` of every frame's point on a "frames2d" plot (the two CVs), to find
    the frame nearest to a click."""
    if link is None or link.kind != "frames2d":
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    t = data.tables[link.column]
    keys = list(data.tables[link.name])[:2]
    rows = frame_rows(pipeline, data, link.column)
    frames = np.nonzero(rows >= 0)[0]
    r = rows[frames]
    return frames, np.asarray(t[keys[0]], float)[r], np.asarray(t[keys[1]], float)[r]


def frame_points(pipeline, data, link: Link, frames, column=None):
    """(x, y) of the plotted points that belong to ``frames``, to mark a frame selection."""
    if link is None or link.kind not in ("frames", "frames2d") or not len(frames):
        return np.zeros(0), np.zeros(0)
    if link.kind == "frames2d":
        t = data.tables[link.column]
        keys = list(data.tables[link.name])[:2]
        table = link.column
    else:
        t = data.tables[link.name]
        keys = [next(iter(t)), column or link.column or list(t)[1]]
        table = link.name
    rows = frame_rows(pipeline, data, table)
    f = np.asarray([k for k in frames if 0 <= k < len(rows)], dtype=int)
    r = rows[f] if len(f) else np.zeros(0, int)
    r = r[r >= 0]
    return np.asarray(t[keys[0]], float)[r], np.asarray(t[keys[1]], float)[r]
