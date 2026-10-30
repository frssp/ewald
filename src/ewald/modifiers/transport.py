"""Ion transport: ionic conductivity and Haven ratio, van Hove correlation, self intermediate
scattering, the non-Gaussian parameter and site-to-site hopping.

Like the other trajectory modifiers they evaluate the pipeline above themselves at every frame
of ``start:stop:step`` (``ctx.input_frame``), cache the result and show it at every frame.
Lengths are in Angstrom, ``time_per_frame`` is the time between stored frames (ps for the
conductivity in S/cm); with ``time_per_frame=0`` times are in frames.
"""
from __future__ import annotations

import numpy as np

from ..kernels import (conductivity_from_slope, decay_time, diffusion_fit, displacement_moments, msd_fft,
                       non_gaussian_parameter, self_intermediate_scattering, unwrap, van_hove_distinct,
                       van_hove_self)
from .trajectory import _scatter_to_frame, _TrajectoryModifier


def _selected(traj, only_selected: bool):
    """Mask of the particles of a trajectory that were selected in its first frame (all of them
    when ``only_selected`` is off or nothing is selected)."""
    n = traj["Position"].shape[1]
    if only_selected and "Selection" in traj:
        sel = np.asarray(traj["Selection"][0]) != 0
        if sel.any():
            return sel
    return np.ones(n, bool)


def _lags(times, n_frames: int):
    """Lags in analyzed frames: the given ones, or 4 roughly log-spaced lags."""
    if times is None or (not np.isscalar(times) and len(times) == 0):
        top = max(1, n_frames // 2)
        return sorted({max(1, int(round(top ** f))) for f in (0.0, 1 / 3, 2 / 3, 1.0)})
    return [int(t) for t in np.atleast_1d(times)]


def _masses(types):
    from ase.data import atomic_masses, atomic_numbers

    from ..radii import element_of

    out = np.ones(len(types))
    for name in np.unique(types):
        sym = element_of(name)
        if sym:
            out[np.asarray(types) == name] = atomic_masses[atomic_numbers[sym]]
    return out


class IonicConductivity(_TrajectoryModifier):
    """Ionic conductivity from the collective charge displacement, its Nernst-Einstein estimate
    and the Haven ratio.

    sigma = lim d/dt < |sum_i q_i dr_i(t)|^2 > / (6 V kB T) (Einstein-Helfand, every frame a
    time origin) and sigma_NE = sum_i q_i^2 < |dr_i(t)|^2 > / (6 V kB T), which leaves out the
    cross correlations between ions; the Haven ratio is sigma_NE / sigma. Both slopes come from
    a linear fit over ``fit_range`` (fraction of the lags), or over the lags between
    ``fit_time=(t0, t1)`` (units of ``time_per_frame``) when given. ``charges`` maps type names to
    charges in e; None takes the 'Charge' property of frame ``start``, else formal oxidation
    states guessed from the types (as the ionic radii do). ``only_selected`` sums over the
    particles selected at ``start`` (e.g. only Li, giving its own conductivity and Haven ratio).
    ``subtract_drift`` removes the center-of-mass motion of all particles first.

    Writes ``tables['conductivity']`` (``Time``, ``Charge MSD`` in e^2 A^2, ``Charge MSD fit``,
    ``Nernst-Einstein MSD``) and attributes ``IonicConductivity.conductivity`` and
    ``.nernst_einstein_conductivity`` (S/cm, NaN without ``time_per_frame`` in ps),
    ``.haven_ratio``, ``.volume`` (A^3, mean over frames) and ``.temperature`` (K).
    """

    def __init__(self, temperature: float = 300.0, time_per_frame: float = 0.0, charges: dict | None = None,
                 start: int = 0, stop: int | None = None, step: int = 1, fit_range: tuple = (0.2, 0.8),
                 unwrapped: bool = False, subtract_drift: bool = True, only_selected: bool = False,
                 fit_time: tuple | None = None):
        self.temperature = temperature
        self.time_per_frame = time_per_frame
        self.charges = charges
        self.start = start
        self.stop = stop
        self.step = step
        self.fit_range = fit_range
        self.unwrapped = unwrapped
        self.subtract_drift = subtract_drift
        self.only_selected = only_selected
        self.fit_time = fit_time

    def modify(self, data, ctx):
        table, attrs = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["conductivity"] = dict(table)
        data.attributes.update(attrs)

    def _charges(self, first, traj):
        types = np.asarray(traj["types"]).astype(str)
        if self.charges is not None:
            given = {str(k): float(v) for k, v in dict(self.charges).items()}
            missing = sorted(set(types) - set(given))
            if missing:
                raise ValueError(f"charges has no value for types {missing}")
            return np.array([given[t] for t in types])
        if "Charge" in traj:
            return np.asarray(traj["Charge"][0], float)
        from ..radii import guess_oxidation_states

        states = guess_oxidation_states(first)
        unknown = sorted(t for t in set(types) if states.get(t, (None,))[0] is None)
        if unknown:
            raise ValueError(f"no charge for types {unknown}: pass charges={{type: charge}} or a 'Charge' property")
        return np.array([float(states[t][0]) for t in types])

    def _compute(self, ctx):
        be = ctx.backend
        xp = be.xp
        first = ctx.input_frame(self._frames(ctx)[0])
        props = ["Position"] + [p for p in ("Selection", "Charge") if p in first]
        traj = self._input(ctx, False, tuple(props))
        q = self._charges(first, traj)
        x = traj["Position"]
        if not self.unwrapped:
            x = unwrap(x, traj["cells"], traj["pbc"], be)
        x = xp.asarray(x)
        if self.subtract_drift:
            w = xp.asarray(_masses(traj["types"]))
            x = x - (xp.tensordot(x, w, axes=([1], [0])) / w.sum())[:, None, :]
        sel = _selected(traj, self.only_selected)
        xs, qs = x[:, xp.asarray(np.nonzero(sel)[0])], xp.asarray(q[sel])
        dt, real_time = self._dt()
        t = np.arange(x.shape[0]) * dt
        total = (xs * qs[None, :, None]).sum(axis=1)[:, None, :]       # (T, 1, 3) charge displacement
        charge_msd = msd_fft(total, be)
        ne_msd = msd_fft(xs * qs[None, :, None], be) * int(sel.sum())
        fit_time = getattr(self, "fit_time", None)
        d_sigma, icpt, _ = diffusion_fit(t, charge_msd, self.fit_range, fit_time=fit_time)
        d_ne, _, _ = diffusion_fit(t, ne_msd, self.fit_range, fit_time=fit_time)
        volume = float(np.mean(np.abs(np.linalg.det(traj["cells"]))))
        table = {"Time": t, "Charge MSD": charge_msd, "Charge MSD fit": 6.0 * d_sigma * t + icpt,
                 "Nernst-Einstein MSD": ne_msd}

        def sigma(d):
            if not real_time or self.temperature <= 0 or volume <= 0:
                return float("nan")
            return conductivity_from_slope(d, volume, self.temperature)

        name = "IonicConductivity."
        attrs = {name + "conductivity": sigma(d_sigma), name + "nernst_einstein_conductivity": sigma(d_ne),
                 name + "haven_ratio": d_ne / d_sigma if d_sigma > 0 else float("nan"),
                 name + "volume": volume, name + "temperature": float(self.temperature)}
        return table, attrs


class VanHoveCorrelation(_TrajectoryModifier):
    """Van Hove correlation function at a few times: the self part (how far one ion gets,
    with peaks at the hop distances) and the distinct part (where the other ions are).

    ``times`` are lags in analyzed frames (None: 4 log-spaced lags up to half the range).
    ``part`` is 'self', 'distinct' or 'both'. Writes ``tables['van-hove']`` with ``r`` and, for
    every lag, ``4πr²Gs(t=...)`` (radial probability density of the displacement length,
    integrating to 1), ``Gs(t=...)`` and ``Gd(t=...)`` (normalized like g(r); g(r) at t=0).
    Time labels are in units of ``time_per_frame`` (frames when it is 0). Displacements use
    up to ``max_origins`` time origins; the distinct part, which needs a pair search per
    origin, uses up to a fifth of them. ``only_selected`` uses the particles selected at ``start``.
    """

    ALIASES = {"nbins": "bins"}

    def __init__(self, times: tuple | None = None, part: str = "self", r_max: float = 6.0, bins: int = 120,
                 time_per_frame: float = 0.0, start: int = 0, stop: int | None = None, step: int = 1,
                 max_origins: int = 100, unwrapped: bool = False, only_selected: bool = False):
        self.times = times
        self.part = part
        self.r_max = r_max
        self.bins = bins
        self.time_per_frame = time_per_frame
        self.start = start
        self.stop = stop
        self.step = step
        self.max_origins = max_origins
        self.unwrapped = unwrapped
        self.only_selected = only_selected

    def modify(self, data, ctx):
        data.tables["van-hove"] = dict(self._cached(ctx, lambda: self._compute(ctx)))

    def _compute(self, ctx):
        if self.part not in ("self", "distinct", "both"):
            raise ValueError(f"part must be 'self', 'distinct' or 'both', not {self.part!r}")
        be = ctx.backend
        traj = self._input(ctx, self.only_selected)
        x = traj["Position"]
        lags = _lags(self.times, x.shape[0])
        dt, _ = self._dt()
        table = {}
        if self.part in ("self", "both"):
            u = x if self.unwrapped else unwrap(x, traj["cells"], traj["pbc"], be)
            r, p = van_hove_self(u, lags, self.r_max, self.bins, self.max_origins, be)
            table["r"] = r
            with np.errstate(divide="ignore", invalid="ignore"):
                for m, row in zip(lags, p):
                    table[f"4πr²Gs(t={m * dt:g})"] = row
                    table[f"Gs(t={m * dt:g})"] = row / (4 * np.pi * r * r)
        if self.part in ("distinct", "both"):
            widths = [np.min(_widths(c, traj["pbc"])) for c in traj["cells"]]
            r_max = min(self.r_max, 0.5 * min(widths) * 0.999)
            if "r" in table and r_max < self.r_max:
                raise ValueError(f"r_max={self.r_max} is more than half the cell width ({2 * r_max:.3g}); "
                                 f"lower it for part='both'")
            r, g = van_hove_distinct(x, traj["cells"], traj["pbc"], [0] + [m for m in lags if m != 0], r_max,
                                     self.bins, max(1, self.max_origins // 5), be)
            table.setdefault("r", r)
            for m, row in zip([0] + [m for m in lags if m != 0], g):
                table[f"Gd(t={m * dt:g})"] = row
        return table


def _widths(matrix, pbc):
    from ..data import Cell

    w = Cell(np.asarray(matrix), pbc=pbc).perpendicular_widths()
    return np.where(np.asarray(pbc, bool), w, np.inf)


class SelfIntermediateScattering(_TrajectoryModifier):
    """Self intermediate scattering function F_s(q, t) and its relaxation time.

    F_s(q, t) = < exp(i q . dr(t)) >, averaged over the analyzed particles, every time origin
    (FFT) and ``directions`` orientations of q (powder average). ``q`` is a modulus in 1/A or a
    tuple of them; None uses 2 pi / d with d = (V / N)^(1/3) the mean spacing of the analyzed
    particles, the scale of one hop. Writes ``tables['intermediate-scattering']`` (``Time`` and
    ``Fs(q=...)`` per q) and ``SelfIntermediateScattering.relaxation_time`` (time at which F_s
    of the first q drops below 1/e; ``.relaxation_time.<i>`` for the i-th q when there are
    several) and ``.q``. ``only_selected`` uses the particles selected at ``start``.
    """

    def __init__(self, q=None, directions: int = 16, time_per_frame: float = 0.0, start: int = 0,
                 stop: int | None = None, step: int = 1, unwrapped: bool = False, only_selected: bool = False):
        self.q = q
        self.directions = directions
        self.time_per_frame = time_per_frame
        self.start = start
        self.stop = stop
        self.step = step
        self.unwrapped = unwrapped
        self.only_selected = only_selected

    def modify(self, data, ctx):
        table, attrs = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["intermediate-scattering"] = dict(table)
        data.attributes.update(attrs)

    def _compute(self, ctx):
        be = ctx.backend
        traj = self._input(ctx, self.only_selected)
        x = traj["Position"]
        if self.q is None:
            volume = float(np.mean(np.abs(np.linalg.det(traj["cells"]))))
            qs = [2 * np.pi / (volume / max(x.shape[1], 1)) ** (1 / 3)]
        else:
            qs = [float(v) for v in np.atleast_1d(self.q)]
        u = x if self.unwrapped else unwrap(x, traj["cells"], traj["pbc"], be)
        fs = self_intermediate_scattering(u, qs, self.directions, be)
        dt, _ = self._dt()
        t = np.arange(x.shape[0]) * dt
        table = {"Time": t}
        name = "SelfIntermediateScattering."
        attrs = {name + "q": qs[0]}
        for k, (qv, row) in enumerate(zip(qs, fs)):
            table[f"Fs(q={qv:.4g})"] = row
            tau = decay_time(t, row)
            if k == 0:
                attrs[name + "relaxation_time"] = tau
            if len(qs) > 1:
                attrs[f"{name}relaxation_time.{k}"] = tau
        return table, attrs


class NonGaussianParameter(_TrajectoryModifier):
    """Non-Gaussian parameter alpha_2(t) = 3 <dr^4> / (5 <dr^2>^2) - 1 of the displacements.

    It is 0 for Gaussian (Fickian) motion and peaks at the time scale of hops or caging.
    Averages over the analyzed particles and up to ``max_origins`` time origins. Writes
    ``tables['non-gaussian']`` (``Time``, ``MSD``, ``MQD`` = <dr^4>, ``alpha2``) and attributes
    ``NonGaussianParameter.max`` and ``.peak_time``. ``only_selected`` uses the particles
    selected at ``start``.
    """

    def __init__(self, time_per_frame: float = 0.0, start: int = 0, stop: int | None = None, step: int = 1,
                 max_origins: int = 200, unwrapped: bool = False, only_selected: bool = False):
        self.time_per_frame = time_per_frame
        self.start = start
        self.stop = stop
        self.step = step
        self.max_origins = max_origins
        self.unwrapped = unwrapped
        self.only_selected = only_selected

    def modify(self, data, ctx):
        table, attrs = self._cached(ctx, lambda: self._compute(ctx))
        data.tables["non-gaussian"] = dict(table)
        data.attributes.update(attrs)

    def _compute(self, ctx):
        be = ctx.backend
        traj = self._input(ctx, self.only_selected)
        x = traj["Position"]
        u = x if self.unwrapped else unwrap(x, traj["cells"], traj["pbc"], be)
        r2, r4 = displacement_moments(u, self.max_origins, be)
        a2 = non_gaussian_parameter(r2, r4)
        dt, _ = self._dt()
        t = np.arange(len(r2)) * dt
        table = {"Time": t, "MSD": r2, "MQD": r4, "alpha2": a2}
        ok = np.isfinite(a2)
        k = int(np.nanargmax(np.where(ok, a2, -np.inf))) if ok.any() else 0
        return table, {"NonGaussianParameter.max": float(a2[k]) if ok.any() else float("nan"),
                       "NonGaussianParameter.peak_time": float(t[k]) if ok.any() else float("nan")}


class SiteHopping(_TrajectoryModifier):
    """Site occupancy, hops between sites and residence times of mobile ions.

    The mobile ions are the particles selected at frame ``start`` (all particles if nothing is
    selected). ``sites`` are the positions of those ions at ``start`` (None), an ``(M, 3)``
    array of Cartesian positions in the cell of frame ``start``, or a structure file whose atoms
    (of type ``site_type`` when given) are the sites, e.g. the ideal Li sublattice with its
    vacancies. Sites keep their fractional coordinates when the cell changes. An ion belongs to
    its nearest site; with ``site_radius`` > 0 it changes site only once it comes closer than
    ``site_radius`` to a new one, which ignores rattling across the boundary between sites.

    Writes ``tables['site-hops']`` (one row per hop: ``Frame``, ``Time``, ``Particle
    Identifier``, ``From Site``, ``To Site``, ``Distance``), ``tables['residence-times']``
    (histogram of completed stays) and ``tables['site-occupancy']`` (fraction of the analyzed
    frames each site is occupied), the particle properties 'Site Index' (at the current frame,
    -1 outside the analyzed frames or for ions not analyzed) and 'Hop Count', and attributes
    ``SiteHopping.hops``, ``.hop_rate`` (hops per ion per time), ``.mean_residence_time``
    (mean completed stay) and ``.jump_diffusion_coefficient`` (sum of squared hop lengths /
    (6 N t), the uncorrelated random-walk D to compare with the tracer D for the correlation
    factor; A^2 per unit of ``time_per_frame``).
    """

    ALIASES = {"nbins": "bins"}

    def __init__(self, sites=None, site_type: str = "", site_radius: float = 0.0, time_per_frame: float = 0.0,
                 start: int = 0, stop: int | None = None, step: int = 1, only_selected: bool = False,
                 bins: int = 50):
        self.sites = sites
        self.site_type = site_type
        self.site_radius = site_radius
        self.time_per_frame = time_per_frame
        self.start = start
        self.stop = stop
        self.step = step
        self.only_selected = only_selected
        self.bins = bins

    def modify(self, data, ctx):
        res = self._cached(ctx, lambda: self._compute(ctx))
        for name in ("site-hops", "residence-times", "site-occupancy"):
            data.tables[name] = dict(res[name])
        data.attributes.update(res["attributes"])
        frames = list(self._frames(ctx))
        here = ctx.frame % ctx.pipeline.num_frames
        current = res["assign"][frames.index(here)] if here in frames else np.full(len(res["hops"]), -1)
        data.particles["Site Index"] = _scatter_to_frame(data, res["index"], current.astype(float), -1).astype(int)
        data.particles["Hop Count"] = _scatter_to_frame(data, res["index"], res["hops"].astype(float)).astype(int)

    def _site_positions(self, traj, ions):
        cell0 = traj["cells"][0]
        if self.sites is None:
            return np.asarray(traj["Position"][0][ions], float)
        if isinstance(self.sites, (str, bytes)) or hasattr(self.sites, "__fspath__"):
            from .. import import_file

            d = import_file(self.sites).compute(0)
            pos = np.asarray(d.positions, float)
            if self.site_type:
                pos = pos[np.asarray(d.types).astype(str) == str(self.site_type)]
            if d.cell.is_valid:  # same fractional coordinates in this trajectory's cell
                pos = pos @ np.linalg.inv(d.cell.matrix) @ cell0
            return pos
        pos = np.asarray(self.sites, float).reshape(-1, 3)
        return pos

    def _compute(self, ctx):
        from scipy.spatial import cKDTree

        first = ctx.input_frame(self._frames(ctx)[0])
        props = ("Position", "Selection") if "Selection" in first else ("Position",)
        traj = self._input(ctx, False, props)
        ions = np.nonzero(_selected(traj, self.only_selected))[0]
        x = np.asarray(traj["Position"][:, ions], float)
        cells, pbc = traj["cells"], np.asarray(traj["pbc"], bool)
        sites = self._site_positions(traj, ions)
        if len(sites) == 0:
            raise ValueError("no sites")
        site_frac = sites @ np.linalg.inv(cells[0])
        site_frac[:, pbc] %= 1.0
        shifts = np.array(np.meshgrid(*[[-1, 0, 1] if p else [0] for p in pbc], indexing="ij")).reshape(3, -1).T
        T, n, M = x.shape[0], x.shape[1], len(sites)
        assign = np.zeros((T, n), int)
        for k in range(T):
            frac = x[k] @ np.linalg.inv(cells[k])
            frac[:, pbc] %= 1.0
            images = (site_frac[None] + shifts[:, None]).reshape(-1, 3) @ cells[k]
            dist, idx = cKDTree(images).query(frac @ cells[k])
            near = idx % M
            if k and self.site_radius > 0:
                near = np.where(dist < self.site_radius, near, assign[k - 1])
            assign[k] = near
        dt, _ = self._dt()
        frames = np.asarray(list(self._frames(ctx)))
        moved = assign[1:] != assign[:-1]                       # (T-1, n)
        step_k, ion_k = np.nonzero(moved)
        a, b = assign[step_k, ion_k], assign[step_k + 1, ion_k]
        d = (site_frac[b] - site_frac[a])
        d[:, pbc] -= np.round(d[:, pbc])
        length = np.linalg.norm(d @ cells[0], axis=1)
        ids = traj["Particle Identifier"]
        pid = (ids[ions] if ids is not None else np.asarray(traj["Index"])[ions])[ion_k]
        hops = {"Frame": frames[step_k + 1], "Time": (step_k + 1) * dt, "Particle Identifier": pid,
                "From Site": a, "To Site": b, "Distance": length}
        stays = []
        for i in range(n):
            when = np.nonzero(moved[:, i])[0]
            if len(when) > 1:
                stays.append(np.diff(when) * dt)
        stays = np.concatenate(stays) if stays else np.zeros(0)
        hi = max(float(stays.max()) if len(stays) else dt, dt)
        counts, edges = np.histogram(stays, bins=max(1, int(self.bins)), range=(0.0, hi))
        occupied = np.zeros(M)
        for k in range(T):
            occupied[np.unique(assign[k])] += 1
        total_time = (T - 1) * dt
        count = len(step_k)
        attrs = {"SiteHopping.hops": int(count), "SiteHopping.sites": int(M),
                 "SiteHopping.hop_rate": count / (n * total_time) if n and total_time > 0 else float("nan"),
                 "SiteHopping.mean_residence_time": float(stays.mean()) if len(stays) else float("nan"),
                 "SiteHopping.jump_diffusion_coefficient":
                     float((length ** 2).sum() / (6 * n * total_time)) if n and total_time > 0 else float("nan")}
        index = {"Particle Identifier": ids[ions] if ids is not None else None,
                 "Index": np.asarray(traj["Index"])[ions]}
        return {"site-hops": hops, "attributes": attrs, "assign": assign, "hops": moved.sum(0),
                "index": index, "residence-times": {"Time": 0.5 * (edges[1:] + edges[:-1]), "Count": counts},
                "site-occupancy": {"Site": np.arange(M), "Occupancy": occupied / T}}
