"""Trajectory modifiers: analyses that read every frame of a range, not just the current one.

They evaluate the pipeline above themselves at each frame (``ctx.input_frame``), so upstream
selections and transformations apply, and read one frame at a time, so the lazy LAMMPS dump
and XDATCAR readers stay lazy. The result is cached on the modifier and reused at every frame
until a parameter, an upstream modifier or the source changes.
"""
from __future__ import annotations

from itertools import chain

import numpy as np

from ..kernels import (_per_type, autocorrelation_fft, average_tables, collect_trajectory, diffusion_fit,
                       finite_difference_velocities, green_kubo, msd_fft, path_statistics, unwrap)
from ..pipeline import Modifier


class _TrajectoryModifier(Modifier):
    start: int = 0
    stop: int | None = None
    step: int = 1

    def _frames(self, ctx) -> range:
        n = ctx.pipeline.num_frames
        stop = n if self.stop is None else min(self.stop, n)
        frames = range(self.start, stop, self.step)
        if len(frames) == 0:
            raise ValueError(f"empty frame range start={self.start}, stop={self.stop}, step={self.step} "
                             f"for {n} frames")
        return frames

    def _cached(self, ctx, compute):
        key = (ctx.upstream_key(), repr(self))
        cache = self.__dict__.get("_result")
        if cache is None or cache[0] != key:
            cache = (key, compute())
            self._result = cache
        return cache[1]

    def _input(self, ctx, only_selected: bool = False, properties=("Position",)):
        """Stacked trajectory of the frame range; ``only_selected`` keeps the particles
        selected in the first frame of the range (all of them if nothing is selected)."""
        frames = self._frames(ctx)
        first = ctx.input_frame(frames[0])
        ids = idx = None
        if only_selected and "Selection" in first:
            idx = np.nonzero(np.asarray(first["Selection"]) != 0)[0]
            if "Particle Identifier" in first:
                ids, idx = np.asarray(first["Particle Identifier"])[idx], None
        rest = (ctx.input_frame(k) for k in frames[1:])
        return collect_trajectory(chain([first], rest), properties, particle_ids=ids, indices=idx)

    def _dt(self):
        """Time between analyzed frames, and whether it is a real time (else frames)."""
        tpf = getattr(self, "time_per_frame", 0.0) or 0.0
        return (tpf * self.step, True) if tpf > 0 else (float(self.step), False)


def _scatter_to_frame(data, traj, values, fill=0.0):
    """Per-particle ``values`` (in the order of a :func:`collect_trajectory` result ``traj``)
    placed onto the particles of ``data``: by identifier when there are identifiers, else by
    index in the first analyzed frame."""
    out = np.full((data.count,) + values.shape[1:], fill, dtype=float)
    ids = traj["Particle Identifier"]
    if ids is None or "Particle Identifier" not in data:
        index = np.asarray(traj["Index"])
        ok = index < data.count
        out[index[ok]] = values[ok]
        return out
    here = np.asarray(data["Particle Identifier"])
    if len(ids) == 0:
        return out
    pos = np.clip(np.searchsorted(ids, here), 0, len(ids) - 1)
    hit = ids[pos] == here
    out[hit] = values[pos[hit]]
    return out


class MeanSquaredDisplacement(_TrajectoryModifier):
    """Time-averaged mean squared displacement over a frame range and the diffusion coefficient.

    Positions are unwrapped across periodic boundaries (skip with ``unwrapped=True`` for LAMMPS
    ``xu yu zu``) and matched by 'Particle Identifier'. Every frame is a time origin (FFT
    algorithm). Writes ``tables['msd']`` with ``Time``, ``MSD``, per-type ``MSD <type>`` and the
    Einstein fit ``MSD fit``, and attributes ``MeanSquaredDisplacement.diffusion_coefficient``
    (``.<type>`` per type) from a linear fit over ``fit_range`` (fraction of the lags).
    ``time_per_frame`` is the time between stored frames (e.g. ps, giving D in A^2/ps;
    1 A^2/ps = 1e-4 cm^2/s); 0 reports lags in frames. ``subtract_drift`` removes the
    center-of-mass motion; ``only_selected`` uses the particles selected at frame ``start``.
    """

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1, time_per_frame: float = 0.0,
                 fit_range: tuple = (0.2, 0.8), by_type: bool = True, unwrapped: bool = False,
                 subtract_drift: bool = False, only_selected: bool = False):
        self.start = start
        self.stop = stop
        self.step = step
        self.time_per_frame = time_per_frame
        self.fit_range = fit_range
        self.by_type = by_type
        self.unwrapped = unwrapped
        self.subtract_drift = subtract_drift
        self.only_selected = only_selected

    def modify(self, data, ctx):
        table, attrs = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["msd"] = dict(table)
        data.attributes.update(attrs)

    def _compute(self, ctx):
        be = ctx.backend
        traj = self._input(ctx, self.only_selected)
        x = traj["Position"]
        if not self.unwrapped:
            x = unwrap(x, traj["cells"], traj["pbc"], be)
        if self.subtract_drift:
            x = x - x.mean(axis=1, keepdims=True)
        dt, _ = self._dt()
        table = {"Time": np.arange(x.shape[0]) * dt, "MSD": msd_fft(x, be)}
        if self.by_type:
            _per_type(msd_fft, x, traj["types"], be, "MSD", table)
        attrs = {}
        name = "MeanSquaredDisplacement.diffusion_coefficient"
        for col in [c for c in table if c.startswith("MSD")]:
            d, icpt, _ = diffusion_fit(table["Time"], table[col], self.fit_range)
            attrs[name + col[3:].replace(" ", ".")] = d
            if col == "MSD":
                table["MSD fit"] = 6.0 * d * table["Time"] + icpt
        return table, attrs


class VelocityAutocorrelation(_TrajectoryModifier):
    """Velocity autocorrelation function and the Green-Kubo diffusion coefficient.

    ``velocities``: 'property' uses 'Velocity' (LAMMPS ``vx vy vz``), 'positions' differentiates
    unwrapped positions (XDATCAR), 'auto' picks the property when present. Writes
    ``tables['vacf']`` with ``Time``, ``VACF``, ``VACF (normalized)``, per-type ``VACF <type>`` and
    the running integral ``D(t)`` = (1/3) int VACF dt, and the attribute
    ``VelocityAutocorrelation.diffusion_coefficient``: ``D(t)`` averaged over the second half of
    ``0 .. max_lag_fraction`` of the lags. Velocities must be in length per unit of ``time_per_frame``.
    """

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1, time_per_frame: float = 0.0,
                 velocities: str = "auto", by_type: bool = True, max_lag_fraction: float = 0.5,
                 only_selected: bool = False):
        self.start = start
        self.stop = stop
        self.step = step
        self.time_per_frame = time_per_frame
        self.velocities = velocities
        self.by_type = by_type
        self.max_lag_fraction = max_lag_fraction
        self.only_selected = only_selected

    def modify(self, data, ctx):
        table, attrs = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["vacf"] = dict(table)
        data.attributes.update(attrs)

    def _compute(self, ctx):
        be = ctx.backend
        source = self.velocities
        if source == "auto":
            source = "property" if "Velocity" in ctx.input_frame(self._frames(ctx)[0]) else "positions"
        dt, _ = self._dt()
        if source == "property":
            traj = self._input(ctx, self.only_selected, ("Velocity",))
            v = traj["Velocity"]
        elif source == "positions":
            traj = self._input(ctx, self.only_selected)
            x = unwrap(traj["Position"], traj["cells"], traj["pbc"], be)
            v = finite_difference_velocities(x, dt, be)
        else:
            raise ValueError(f"velocities must be 'auto', 'property' or 'positions', not {self.velocities!r}")
        c = autocorrelation_fft(v, be)
        if len(c) == 0:
            raise ValueError("need at least two frames to differentiate positions")
        t = np.arange(len(c)) * dt
        table = {"Time": t, "VACF": c, "VACF (normalized)": c / c[0] if c[0] != 0 else c * 0.0}
        if self.by_type:
            _per_type(autocorrelation_fft, v, traj["types"], be, "VACF", table)
        running = green_kubo(t, c)
        table["D(t)"] = running
        hi = max(2, int(self.max_lag_fraction * len(running)))
        d = float(np.mean(running[hi // 2:hi])) if len(running) > 1 else float("nan")
        return table, {"VelocityAutocorrelation.diffusion_coefficient": d}


class TimeAveraging(_TrajectoryModifier):
    """Average upstream results over a frame range (OVITO's TimeAveragingModifier).

    ``operate_on`` names one or more inputs: ``'table:<name>'`` (e.g. ``'table:coordination-rdf'``
    after CoordinationAnalysis gives the time-averaged g(r)) writes ``tables['<name>[average]']``;
    ``'attribute:<name>'`` writes the attribute ``'<name> (Average)'``; ``'property:<name>'``
    writes the per-particle property ``'<name> (Average)'`` (particles matched by identifier).
    """

    def __init__(self, operate_on="table:coordination-rdf", start: int = 0, stop: int | None = None,
                 step: int = 1):
        self.operate_on = operate_on
        self.start = start
        self.stop = stop
        self.step = step

    def _targets(self):
        items = [self.operate_on] if isinstance(self.operate_on, str) else list(self.operate_on)
        out = []
        for item in items:
            kind, _, name = item.partition(":")
            if kind not in ("table", "attribute", "property") or not name:
                raise ValueError(f"operate_on entries look like 'table:<name>', 'attribute:<name>' or "
                                 f"'property:<name>', not {item!r}")
            out.append((kind, name))
        return out

    def modify(self, data, ctx):
        results = self._cached(ctx, lambda: self._compute(ctx))
        for (kind, name), value in results.items():
            if kind == "table":
                data.tables[f"{name}[average]"] = dict(value)
            elif kind == "attribute":
                data.attributes[f"{name} (Average)"] = value
            else:
                traj, mean = value
                data.particles[f"{name} (Average)"] = _scatter_to_frame(data, traj, mean, np.nan)

    def _compute(self, ctx):
        targets = self._targets()
        tables = {name: [] for kind, name in targets if kind == "table"}
        attrs = {name: [] for kind, name in targets if kind == "attribute"}
        props = [name for kind, name in targets if kind == "property"]
        prop_frames = []

        def frames():
            for k in self._frames(ctx):
                d = ctx.input_frame(k)
                for name in tables:
                    if name not in d.tables:
                        raise KeyError(f"frame {k} has no table {name!r}; tables: {list(d.tables)}")
                    tables[name].append(d.tables[name])
                for name in attrs:
                    if name not in d.attributes:
                        raise KeyError(f"frame {k} has no attribute {name!r}")
                    attrs[name].append(float(d.attributes[name]))
                yield d

        if props:
            traj = collect_trajectory(frames(), props)
            for name in props:
                prop_frames.append((name, (traj, np.asarray(traj[name], dtype=float).mean(axis=0))))
        else:
            for _ in frames():
                pass
        out = {("table", n): average_tables(v) for n, v in tables.items()}
        out.update({("attribute", n): float(np.mean(v)) for n, v in attrs.items()})
        out.update({("property", n): v for n, v in prop_frames})
        return {t: out[t] for t in targets}


class TimeSeries(_TrajectoryModifier):
    """Global attributes as a function of time (OVITO's TimeSeriesModifier).

    ``operate_on`` is an attribute name or a tuple of names; None takes every numeric
    attribute of the first frame. Writes ``tables['time-series']`` with ``Frame``,
    ``Timestep`` and one column per attribute.
    """

    def __init__(self, operate_on=None, start: int = 0, stop: int | None = None, step: int = 1):
        self.operate_on = operate_on
        self.start = start
        self.stop = stop
        self.step = step

    def modify(self, data, ctx):
        data.tables["time-series"] = dict(self._cached(ctx, lambda: self._compute(ctx)))

    def _compute(self, ctx):
        frames = self._frames(ctx)
        names = [self.operate_on] if isinstance(self.operate_on, str) else self.operate_on
        cols = {"Frame": [], "Timestep": []}
        for k in frames:
            d = ctx.input_frame(k)
            if names is None:
                names = [a for a, v in d.attributes.items() if a not in ("Timestep", "Frame")
                         and isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
            cols["Frame"].append(k)
            cols["Timestep"].append(d.attributes.get("Timestep", k))
            for a in names:
                cols.setdefault(a, []).append(float(d.attributes[a]) if a in d.attributes else np.nan)
        return {k: np.asarray(v) for k, v in cols.items()}


class GenerateTrajectoryLines(_TrajectoryModifier):
    """Particle paths over a frame range, drawn as lines, plus path metrics.

    Traces the particles selected at frame ``start`` (all of them when nothing is selected or
    ``only_selected`` is False). Writes ``tables['trajectory-lines']`` with one row per particle
    and frame (``Position``, ``Particle Identifier``, ``Time``), and the particle properties
    'Path Length' (summed step lengths), 'Net Displacement' and 'Max Displacement' (from the
    position at ``start``); untraced particles get 0. ``unwrap_trajectories`` keeps paths
    continuous across periodic boundaries.
    """

    def __init__(self, only_selected: bool = True, start: int = 0, stop: int | None = None, step: int = 1,
                 unwrap_trajectories: bool = True):
        self.only_selected = only_selected
        self.start = start
        self.stop = stop
        self.step = step
        self.unwrap_trajectories = unwrap_trajectories

    def modify(self, data, ctx):
        lines, traj, stats = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["trajectory-lines"] = dict(lines)
        for name, values in stats.items():
            data.particles[name] = _scatter_to_frame(data, traj, values)

    def _compute(self, ctx):
        be = ctx.backend
        traj = self._input(ctx, self.only_selected)
        x = traj["Position"]
        unwrapped = unwrap(x, traj["cells"], traj["pbc"], be)
        path, net, peak = path_statistics(unwrapped, be)
        shown = be.to_numpy(unwrapped) if self.unwrap_trajectories else x
        T, n = shown.shape[:2]
        ids = traj["Particle Identifier"]
        line_ids = ids if ids is not None else np.arange(n)
        frames = np.asarray(list(self._frames(ctx)))
        lines = {"Position": shown.transpose(1, 0, 2).reshape(-1, 3),
                 "Particle Identifier": np.repeat(line_ids, T),
                 "Time": np.tile(traj["Timestep"] if len(traj["Timestep"]) == T else frames, n)}
        stats = {"Path Length": path, "Net Displacement": net, "Max Displacement": peak}
        index = {k: traj[k] for k in ("Particle Identifier", "Index")}
        return lines, index, stats


class OverlayFrames(_TrajectoryModifier):
    """Every frame of ``start:stop:step`` drawn at once: the current frame opaque, the others
    as see-through copies of the atoms (``transparency`` 0..1), e.g. all images of a NEB band
    over the image shown on the timeline.

    Only atoms that sit more than ``min_displacement`` (Angstrom) away from where they are in
    the current frame get a copy, so the atoms that move leave a translucent trail and the rest
    of the structure is drawn once. Copies are placed at the minimum image next to the current
    position, matched by 'Particle Identifier' when there are identifiers, else by index.
    They are appended after the real particles with the current frame's properties (type,
    color, radius), 'Transparency', 'Overlay Frame' (the frame each copy comes from; -1 for
    the real particles), no selection and zero vector properties, so put this modifier last.
    """

    def __init__(self, transparency: float = 0.7, min_displacement: float = 0.1, start: int = 0,
                 stop: int | None = None, step: int = 1):
        self.transparency = transparency
        self.min_displacement = min_displacement
        self.start = start
        self.stop = stop
        self.step = step

    def modify(self, data, ctx):
        frames = self._cached(ctx, lambda: self._compute(ctx))
        here = ctx.frame % ctx.pipeline.num_frames
        x = np.asarray(data.positions, float)
        ids = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else None
        src, pos, origin = [], [], []
        for k, (kids, kx) in frames.items():
            if k == here:
                continue
            if ids is not None and kids is not None:
                order = np.argsort(kids)
                at = np.clip(np.searchsorted(kids, ids, sorter=order), 0, len(kids) - 1)
                found = kids[order[at]] == ids
                own, theirs = np.flatnonzero(found), order[at[found]]
            elif len(kx) == len(x):
                own = theirs = np.arange(len(x))
            else:
                raise ValueError(f"frame {k} has {len(kx)} atoms and frame {here} has {len(x)}, and there "
                                 f"are no particle identifiers to match them")
            d = _mic(kx[theirs] - x[own], data.cell)
            moved = np.linalg.norm(d, axis=1) > self.min_displacement
            src.append(own[moved])
            pos.append(x[own[moved]] + d[moved])
            origin.append(np.full(int(moved.sum()), k))
        src = np.concatenate(src) if src else np.zeros(0, int)
        n, m = data.count, len(src)
        new = {}
        for name, v in data.particles.items():
            v = np.asarray(v)
            if name == "Position":
                extra = np.concatenate(pos) if m else np.zeros((0, 3))
            elif name == "Selection" or name in data.vectors:
                extra = np.zeros((m,) + v.shape[1:], v.dtype)
            else:
                extra = v[src]
            new[name] = np.concatenate([v, extra])
        base = np.asarray(data["Transparency"], float) if "Transparency" in data else np.zeros(n)
        new["Transparency"] = np.concatenate([base, np.full(m, float(np.clip(self.transparency, 0, 1)))])
        new["Overlay Frame"] = np.concatenate([np.full(n, -1), np.concatenate(origin) if m else np.zeros(0, int)])
        data.particles = new
        data.attributes["OverlayFrames.count"] = m

    def _compute(self, ctx):
        out = {}
        for k in self._frames(ctx):
            d = ctx.input_frame(k)
            ids = np.asarray(d["Particle Identifier"]) if "Particle Identifier" in d else None
            out[k] = (ids, np.asarray(d.positions, float))
        return out


def _mic(delta, cell):
    """Displacements wrapped to the nearest periodic image of ``cell``."""
    if not cell.is_valid or not cell.pbc.any():
        return delta
    frac = delta @ np.linalg.inv(cell.matrix)
    frac[:, cell.pbc] -= np.round(frac[:, cell.pbc])
    return frac @ cell.matrix


def _match_identifiers(ref_ids, ids, what: str = "reference frame"):
    """Index into ``ref_ids`` of every entry of ``ids`` (-1 where it is missing)."""
    order = np.argsort(ref_ids, kind="stable")
    pos = np.clip(np.searchsorted(ref_ids[order], ids), 0, max(len(order) - 1, 0))
    if not len(order):
        return np.full(len(ids), -1, np.int64)
    found = ref_ids[order][pos] == ids
    return np.where(found, order[pos], -1)


class FreezeProperty(Modifier):
    """Copy a particle property as it was at frame ``freeze_at`` to every frame
    (OVITO's FreezePropertyModifier).

    Typical use: keep the selection, the structure type or the coordination of the starting
    configuration, e.g. ``FreezeProperty("Structure Type", "Initial Structure", freeze_at=0)``
    after CNA, then select the atoms that were fcc at the start. The frozen values come from
    the pipeline above this modifier at ``freeze_at``. Particles are matched by
    'Particle Identifier' when present, else by index; particles that did not exist at
    ``freeze_at`` get zero. ``destination_property`` defaults to ``source_property``.
    """

    def __init__(self, source_property: str = "Selection", destination_property: str | None = None,
                 freeze_at: int = 0):
        self.source_property = source_property
        self.destination_property = destination_property
        self.freeze_at = freeze_at

    def _frozen(self, data, ctx):
        frame = self.freeze_at % ctx.pipeline.num_frames
        key = (ctx.upstream_key(), self.source_property, frame)
        cache = self.__dict__.get("_cache")
        if cache is None or cache[0] != key:
            ref = data if frame == ctx.frame else ctx.input_frame(frame)
            if self.source_property not in ref:
                raise KeyError(f"no particle property {self.source_property!r} at frame {frame}")
            ids = np.asarray(ref["Particle Identifier"]) if "Particle Identifier" in ref else None
            cache = (key, np.asarray(ref[self.source_property]), ids)
            self._cache = cache
        return cache[1], cache[2]

    def modify(self, data, ctx):
        values, ids = self._frozen(data, ctx)
        if ids is not None and "Particle Identifier" in data:
            index = _match_identifiers(ids, np.asarray(data["Particle Identifier"]))
            out = np.zeros((data.count,) + values.shape[1:], values.dtype)
            hit = index >= 0
            out[hit] = values[index[hit]]
        elif len(values) == data.count:
            out = values
        else:
            raise ValueError(f"frame {ctx.frame} has {data.count} particles, frame {self.freeze_at} had "
                             f"{len(values)}, and there are no identifiers to match them")
        data.particles[self.destination_property or self.source_property] = out


class UnwrapTrajectories(Modifier):
    """Undo the wrapping of particles at periodic boundaries, so trajectories are continuous
    (OVITO's UnwrapTrajectoriesModifier).

    With a 'Periodic Image' property (LAMMPS ``ix iy iz``) the positions are shifted by those
    image counts and the counts reset to zero. Otherwise every crossing is detected by following
    the trajectory from frame 0 (a particle must move less than half a cell width between
    stored frames); particles are matched by 'Particle Identifier', else by index. Bond periodic
    images are adjusted so bonds stay the same. Checkpoints along the trajectory keep a jump to
    a distant frame from replaying it from the start.
    """

    max_checkpoints = 16

    def modify(self, data, ctx):
        cell = data.cell
        if not (cell.is_valid and cell.pbc.any()):
            return None
        if "Periodic Image" in data:
            img = np.asarray(data["Periodic Image"]).astype(np.int64)
            data.particles["Periodic Image"] = np.zeros_like(data["Periodic Image"])
        else:
            img = self._images(data, ctx)
        img = img * cell.pbc.astype(np.int64)
        if not img.any():
            return None
        data.positions = data.positions + img @ cell.matrix
        if data.bonds is not None and data.bonds.count:
            i, j = data.bonds.topology.T
            data.bonds.properties["Periodic Image"] = (data.bonds.pbc_images + img[i] - img[j]).astype(np.int32)
        return None

    @staticmethod
    def _state(d, ids):
        """Positions of ``d`` in the order of the reference identifiers ``ids`` (None: by index)."""
        if ids is None:
            return np.asarray(d.positions, float), None
        index = _match_identifiers(np.asarray(d["Particle Identifier"]), ids)
        if (index < 0).any() or d.count != len(ids):
            raise ValueError("UnwrapTrajectories: particles appear or vanish between frames")
        return np.asarray(d.positions, float)[index], index

    def _images(self, data, ctx):
        n = ctx.pipeline.num_frames
        key = ctx.upstream_key()
        st = self.__dict__.get("_unwrap")
        if st is None or st["key"] != key:
            first = data if ctx.frame == 0 else ctx.input_frame(0)
            ids = np.asarray(first["Particle Identifier"]) if "Particle Identifier" in first else None
            st = {"key": key, "ids": ids, "stride": max(1, -(-n // self.max_checkpoints)),
                  "points": {0: (np.asarray(first.positions, float), np.zeros((first.count, 3), np.int64))}}
            self._unwrap = st
        ids = st["ids"]
        start = max(k for k in st["points"] if k <= ctx.frame)
        prev, img = st["points"][start]
        img = img.copy()
        for k in range(start + 1, ctx.frame + 1):
            d = data if k == ctx.frame else ctx.input_frame(k)
            pos, _ = self._state(d, ids)
            if ids is None and len(pos) != len(prev):
                raise ValueError("UnwrapTrajectories: the particle count changes and there are no identifiers")
            f = (pos - prev) @ np.linalg.inv(d.cell.matrix)
            img -= np.round(f).astype(np.int64) * d.cell.pbc.astype(np.int64)
            prev = pos
            if k % st["stride"] == 0 or k == ctx.frame:
                st["points"][k] = (pos, img.copy())
        # drop the previous "last frame" point unless it is a checkpoint
        for k in [k for k in st["points"] if k % st["stride"] and k != ctx.frame]:
            del st["points"][k]
        if ids is None:
            if len(img) != data.count:
                raise ValueError("UnwrapTrajectories: the particle count changes and there are no identifiers")
            return img
        _, index = self._state(data, ids)  # data order -> reference order
        out = np.empty_like(img)
        out[index] = img
        return out
