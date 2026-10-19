"""Nudged elastic band: energy profile and barrier of a band whose images are the frames."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier

_ENERGY_NAMES = ("Energy", "energy", "Potential Energy", "PotEng", "Free Energy")


class NebEnergyProfile(Modifier):
    """Energy along a NEB band, with the barrier, from frames that are the band's images.

    The band is the last ``images`` frames (None: every frame; an ASE NEB trajectory holds
    every optimizer step, so ``images=7`` takes the final band of a 7-image run). Image
    energies come from the attribute ``energy`` (eV; ``Energy`` from OUTCAR/vasprun or the
    LAMMPS NEB log, else ``energy``/``PotEng``/... when present). The reaction coordinate is
    the summed distance between neighbouring images in 3N-dimensional space (Angstrom,
    minimum image), as VTST's nebbarrier.pl computes it. When the frames carry ``Force``,
    the force along the improved tangent (Henkelman & Jonsson 2000) gives the slope at each
    image and the profile is VTST's cubic spline (nebspline.pl); without forces it is a
    natural cubic spline through the energies.

    Writes ``tables['NEB']`` (``Reaction Coordinate``, ``Energy`` relative to the first
    image, ``Tangent Force``, ``Image``, ``Frame``), ``tables['NEB spline']`` (``spline_points``
    points of the profile), the attributes ``NEB.Barrier`` (highest image minus the first),
    ``NEB.Reverse Barrier`` (highest image minus the last), ``NEB.Reaction Energy`` (last
    minus first), ``NEB.Spline Barrier`` (maximum of the spline), ``NEB.Saddle Image``,
    ``NEB.Saddle Coordinate``, ``NEB.Path Length`` and ``NEB.Reference Energy``, and on
    frames of the band ``NEB.Image``, ``NEB.Reaction Coordinate`` and ``NEB.Relative Energy``.
    The GUI's Plots tab draws the profile with the current image marked; clicking an image
    there jumps to its frame.
    """

    def __init__(self, energy: str = "Energy", images: int | None = None, spline_points: int = 200):
        self.energy = energy
        self.images = images
        self.spline_points = spline_points

    def _frames(self, n: int) -> range:
        if self.images is None or self.images <= 0:
            return range(n)
        if self.images < 2 or self.images > n:
            raise ValueError(f"images={self.images}: a band needs 2..{n} images (frames)")
        return range(n - self.images, n)

    def modify(self, data, ctx):
        key = (ctx.upstream_key(), repr(self))
        cache = self.__dict__.get("_result")
        if cache is None or cache[0] != key:
            cache = (key, self._compute(ctx))
            self._result = cache
        table, spline, attrs = cache[1]
        data.tables["NEB"] = dict(table)
        if spline is not None:
            data.tables["NEB spline"] = dict(spline)
        data.attributes.update(attrs)
        frame = ctx.frame % ctx.pipeline.num_frames
        hit = np.nonzero(table["Frame"] == frame)[0]
        if len(hit):
            i = int(hit[0])
            data.attributes["NEB.Image"] = i
            data.attributes["NEB.Reaction Coordinate"] = float(table["Reaction Coordinate"][i])
            e = float(table["Energy"][i])
            if e == e:
                data.attributes["NEB.Relative Energy"] = e

    def _energy(self, d) -> float:
        names = (self.energy,) + (_ENERGY_NAMES if self.energy == "Energy" else ())
        for name in names:
            v = d.attributes.get(name)
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                return float(v)
        return np.nan

    def _compute(self, ctx):
        frames = self._frames(ctx.pipeline.num_frames)
        pos, cells, energies, forces, order = [], [], [], [], None
        for k in frames:
            d = ctx.input_frame(k)
            ids = np.asarray(d["Particle Identifier"]) if "Particle Identifier" in d else None
            if order is None:
                order = None if ids is None else np.sort(ids)
            idx = np.arange(d.count) if ids is None or order is None else _index_of(ids, order, k)
            if pos and len(idx) != len(pos[0]):
                raise ValueError(f"frame {k} has {len(idx)} atoms, frame {frames[0]} has {len(pos[0])}: "
                                 f"images of a band must have the same atoms")
            pos.append(np.asarray(d.positions, dtype=float)[idx])
            cells.append(d.cell)
            energies.append(self._energy(d))
            forces.append(np.asarray(d["Force"], dtype=float)[idx] if "Force" in d else None)
        energies = np.asarray(energies)
        if np.isnan(energies).all():
            raise ValueError(f"no image has an energy attribute {self.energy!r}")
        steps = [_mic(pos[i + 1] - pos[i], cells[i]) for i in range(len(pos) - 1)]
        s = np.concatenate([[0.0], np.cumsum([np.linalg.norm(v) for v in steps])])
        ref = energies[np.nonzero(~np.isnan(energies))[0][0]]
        rel = energies - ref
        f_par = np.full(len(pos), np.nan)
        for i, f in enumerate(forces):
            if f is not None:
                tau = _tangent(steps, rel, i)
                if tau is not None:
                    f_par[i] = float(np.sum(f * tau))
        table = {"Reaction Coordinate": s, "Energy": rel, "Tangent Force": f_par,
                 "Image": np.arange(len(pos)), "Frame": np.asarray(list(frames))}
        ok = ~np.isnan(rel)
        top = int(np.nanargmax(rel))
        attrs = {"NEB.Barrier": float(rel[top]), "NEB.Saddle Image": top, "NEB.Path Length": float(s[-1]),
                 "NEB.Reference Energy": float(ref), "NEB.Saddle Coordinate": float(s[top])}
        if ok[-1]:
            attrs["NEB.Reverse Barrier"] = float(rel[top] - rel[-1])
            attrs["NEB.Reaction Energy"] = float(rel[-1])
        spline = None
        if ok.sum() >= 2 and self.spline_points >= 2 and s[-1] > 0:
            xs, ys = _spline(s[ok], rel[ok], f_par[ok], self.spline_points)
            spline = {"Reaction Coordinate": xs, "Energy": ys}
            j = int(np.argmax(ys))
            attrs["NEB.Spline Barrier"] = float(ys[j])
            attrs["NEB.Saddle Coordinate"] = float(xs[j])
        return table, spline, attrs


def _index_of(ids, order, frame):
    pos = np.searchsorted(order, ids)
    if len(ids) != len(order) or (pos >= len(order)).any() or (order[np.minimum(pos, len(order) - 1)] != ids).any():
        raise ValueError(f"frame {frame}: particle identifiers differ from the first image")
    idx = np.empty(len(ids), dtype=int)
    idx[pos] = np.arange(len(ids))
    return idx


def _mic(delta, cell):
    """Displacements wrapped to the nearest periodic image of ``cell``."""
    if not cell.is_valid or not cell.pbc.any():
        return delta
    frac = delta @ np.linalg.inv(cell.matrix)
    frac[:, cell.pbc] -= np.round(frac[:, cell.pbc])
    return frac @ cell.matrix


def _tangent(steps, e, i):
    """Unit tangent at image ``i``: the improved tangent of Henkelman & Jonsson (2000)
    inside the band, the step to the neighbour at the ends."""
    n = len(steps) + 1
    plus = steps[i] if i < n - 1 else None
    minus = steps[i - 1] if i > 0 else None
    if plus is None or minus is None:
        tau = plus if plus is not None else minus
    elif np.isnan(e[i - 1:i + 2]).any():
        tau = plus / max(np.linalg.norm(plus), 1e-12) + minus / max(np.linalg.norm(minus), 1e-12)
    elif e[i + 1] > e[i] > e[i - 1]:
        tau = plus
    elif e[i + 1] < e[i] < e[i - 1]:
        tau = minus
    else:
        dmax = max(abs(e[i + 1] - e[i]), abs(e[i - 1] - e[i]))
        dmin = min(abs(e[i + 1] - e[i]), abs(e[i - 1] - e[i]))
        tau = plus * dmax + minus * dmin if e[i + 1] > e[i - 1] else plus * dmin + minus * dmax
    norm = np.linalg.norm(tau)
    return tau / norm if norm > 1e-12 else None


def _spline(s, e, f_par, n):
    """Profile through the images: cubic Hermite with slopes -F_parallel (VTST) when every
    image has a tangent force, else a natural cubic spline (linear for two images)."""
    from scipy.interpolate import CubicHermiteSpline, CubicSpline

    xs = np.linspace(s[0], s[-1], n)
    if not np.isnan(f_par).any():
        return xs, CubicHermiteSpline(s, e, -f_par)(xs)
    if len(s) >= 3:
        return xs, CubicSpline(s, e, bc_type="natural")(xs)
    return xs, np.interp(xs, s, e)
