"""Bonds between particles (OVITO's bonds container) and the bond search used to create them.

A bond is stored once, as ``Topology`` (i, j) plus ``Periodic Image`` (the cell
image of j seen from i), so the bond vector is ``x_j - x_i + image @ cell.matrix``.
That stays correct across periodic boundaries and in cells smaller than twice
the bond length, where a minimum-image convention would pick the wrong partner.
"""
from __future__ import annotations

import numpy as np


class Bonds:
    """Per-bond properties, keyed by OVITO names: 'Topology' (M x 2 particle indices),
    'Periodic Image' (M x 3 int), and optionally 'Bond Type', 'Color', 'Selection', 'Width', ...

    Like ``DataCollection.particles``, arrays are shared between copies: replace them,
    never write into them.
    """

    def __init__(self, topology=None, periodic_image=None, **properties):
        topo = np.zeros((0, 2), np.int64) if topology is None else np.asarray(topology, np.int64).reshape(-1, 2)
        img = (np.zeros((len(topo), 3), np.int32) if periodic_image is None
               else np.asarray(periodic_image, np.int32).reshape(-1, 3))
        self.properties: dict[str, np.ndarray] = {"Topology": topo, "Periodic Image": img}
        for name, value in properties.items():  # bond_type=... -> 'Bond Type'
            self[name.replace("_", " ").title()] = value

    @property
    def count(self) -> int:
        return len(self.properties["Topology"])

    def __len__(self) -> int:
        return self.count

    @property
    def topology(self) -> np.ndarray:
        return self.properties["Topology"]

    @property
    def pbc_images(self) -> np.ndarray:
        return self.properties["Periodic Image"]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.properties[name]

    def __setitem__(self, name: str, value) -> None:
        value = np.asarray(value)
        if value.ndim == 0:
            value = np.full(self.count, value)
        if len(value) != self.count:
            raise ValueError(f"bond property '{name}' has length {len(value)}, expected {self.count}")
        self.properties[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self.properties

    def copy(self) -> "Bonds":
        new = Bonds.__new__(Bonds)
        new.properties = dict(self.properties)
        return new

    def take(self, index) -> "Bonds":
        """The bonds selected by ``index`` (mask or indices)."""
        new = Bonds.__new__(Bonds)
        new.properties = {k: v[index] for k, v in self.properties.items()}
        return new

    def extend(self, other: "Bonds") -> "Bonds":
        """Bonds of ``self`` followed by ``other``; properties missing on one side are zero-filled."""
        new = Bonds.__new__(Bonds)
        new.properties = {}
        for name in dict.fromkeys([*self.properties, *other.properties]):
            parts = []
            for b in (self, other):
                if name in b.properties:
                    parts.append(b.properties[name])
                else:
                    ref = (other if b is self else self).properties[name]
                    parts.append(np.zeros((b.count,) + ref.shape[1:], ref.dtype))
            new.properties[name] = np.concatenate(parts)
        return new

    def remap_particles(self, index, n_old: int) -> "Bonds":
        """Bonds after ``DataCollection.take(index)``: indices renumbered, bonds to dropped particles removed.
        A particle taken more than once keeps the bonds of its first copy."""
        if isinstance(index, slice):
            index = np.arange(n_old)[index]
        index = np.asarray(index)
        old = np.arange(n_old)[index] if index.dtype == bool else index.astype(np.int64) % max(n_old, 1)
        new_of_old = np.full(n_old, -1, np.int64)
        new_of_old[old[::-1]] = np.arange(len(old))[::-1]
        t = new_of_old[self.topology] if self.count else np.zeros((0, 2), np.int64)
        keep = (t >= 0).all(axis=1)
        out = self.take(keep)
        out.properties["Topology"] = t[keep]
        return out

    def vectors(self, data) -> np.ndarray:
        """Bond vectors x_j - x_i + image @ cell (M x 3)."""
        i, j = self.topology.T
        d = data.positions[j] - data.positions[i]
        if data.cell.is_valid and self.count:
            d = d + self.pbc_images @ data.cell.matrix
        return d

    def lengths(self, data) -> np.ndarray:
        return np.linalg.norm(self.vectors(data), axis=1)

    def neighbor_counts(self, n: int) -> np.ndarray:
        return np.bincount(self.topology.ravel(), minlength=n)

    def __repr__(self):
        return f"<Bonds M={self.count} props=[{', '.join(self.properties)}]>"


# ---------------------------------------------------------------------------- bond search
def minimum_images(data, i, j) -> np.ndarray:
    """Periodic image of j that is closest to i, for bonds given only by topology (M x 3 int)."""
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    cell = data.cell
    if not cell.is_valid or not len(i):
        return np.zeros((len(i), 3), np.int32)
    f = cell.to_fractional(data.positions[j]) - cell.to_fractional(data.positions[i])
    return (-np.rint(f) * cell.pbc).astype(np.int32)


def normalize_pairs(pairs):
    """VESTA-style bond rules ``(A, B, dmin, dmax, pbc)`` from rows ``(A, B, dmin, dmax[, pbc])``,
    ``(A, B, dmax)`` triples or an OVITO-style ``{(A, B): dmax}`` dict. '*' matches any type.

    How the rules combine (see :func:`find_bonds`): the most specific rules that name a pair decide
    it, A-B before A-* before *-*, so ``[('*', '*', 0, 3), ('Ti', 'Ti', 0, 0)]`` bonds everything up
    to 3 A except Ti-Ti. ``pbc`` (default True) False keeps only bonds inside the cell, none through
    a periodic boundary."""
    if pairs is None:
        return []
    if isinstance(pairs, dict):
        pairs = [(a, b, 0.0, r) for (a, b), r in pairs.items()]
    out = []
    for p in pairs:
        p = tuple(p)
        if len(p) == 3:
            p = (p[0], p[1], 0.0, p[2])
        if len(p) not in (4, 5):
            raise ValueError(f"bond rule {p!r}: use (A, B, dmax), (A, B, dmin, dmax) or (A, B, dmin, dmax, pbc)")
        a, b, lo, hi = p[:4]
        out.append((str(a), str(b), float(lo), float(hi), _flag(p[4]) if len(p) == 5 else True))
    return out


def _flag(v) -> bool:
    if isinstance(v, str):   # 'pbc' / 'no pbc' as typed in a table, or JSON booleans as text
        v = v.strip().lower()
        if v in ("true", "yes", "on", "1", "pbc"):
            return True
        if v in ("false", "no", "off", "0", "", "none", "no pbc"):
            return False
        raise ValueError(f"bond rule pbc must be True or False, not {v!r}")
    return bool(v)


def rule_rows(pairs) -> list:
    """Bond rules as plain JSON lists, ``[A, B, dmin, dmax]`` plus ``False`` only for rules without
    periodic bonds, so tables without the option stay as they were."""
    return [[a, b, lo, hi] if pbc else [a, b, lo, hi, False] for a, b, lo, hi, pbc in normalize_pairs(pairs)]


# Not bonded to each other by default in compounds (metal-metal contacts would clutter oxides etc.)
NONMETALS = frozenset("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())


def covalent_pairs(types, factor: float = 1.15):
    """One row per type pair with dmax = factor * (r_A + r_B) from covalent radii (VESTA's default).
    In compounds metal-metal pairs are left out (SrTiO3 gets Ti-O and Sr-O, not Sr-Ti or Sr-Sr);
    a pure metal or alloy keeps them."""
    from ase.data import chemical_symbols, covalent_radii

    from .radii import atomic_number

    uniq = sorted(set(str(t) for t in types))
    z = {t: atomic_number(t) for t in uniq}
    rad = {t: covalent_radii[z[t]] if z[t] else 0.8 for t in uniq}
    metal = {t: bool(z[t]) and chemical_symbols[z[t]] not in NONMETALS for t in uniq}
    compound = not all(metal.values())
    return [(a, b, 0.0, round(factor * (rad[a] + rad[b]), 4)) for k, a in enumerate(uniq) for b in uniq[k:]
            if not (compound and metal[a] and metal[b])]


def _pair_mask(ti, tj, dist, img, pairs):
    """Which pairs the rules bond: per pair only the most specific matching rules count (A-B: 2,
    A-*: 1, *-*: 0 named types); any of those whose length range (and pbc) fits bonds it."""
    rank = np.full(len(dist), -1, np.int8)
    hits = []
    inside = ~np.any(img != 0, axis=1) if len(dist) else np.zeros(0, bool)
    for a, b, lo, hi, pbc in pairs:
        ok_t = (((ti == a) | (a == "*")) & ((tj == b) | (b == "*"))) | \
               (((ti == b) | (b == "*")) & ((tj == a) | (a == "*")))
        level = (a != "*") + (b != "*")
        rank = np.where(ok_t, np.maximum(rank, level), rank)
        ok = ok_t & (dist >= lo) & (dist <= hi)
        hits.append((level, ok if pbc else ok & inside))
    keep = np.zeros(len(dist), bool)
    for level, ok in hits:
        keep |= ok & (rank == level)
    return keep


def _half_pairs(data, rmax, backend=None):
    """Every pair (i, j, image) with |x_j - x_i + image @ H| < rmax, each listed once."""
    from .backend import get_backend
    from .neighbors import find_pairs

    cell, pos = data.cell, data.positions
    if cell.is_valid and np.any(cell.pbc & (cell.perpendicular_widths() < 2 * rmax)):
        return _half_pairs_small_cell(pos, cell, rmax)
    be = get_backend(backend)
    i, j, d, dist = (be.to_numpy(x) for x in find_pairs(pos, cell, rmax, be, half=True))
    i, j = i.astype(np.int64), j.astype(np.int64)
    img = np.zeros((len(i), 3), np.int32)
    if cell.is_valid and len(i):
        img = np.rint((d - (pos[j] - pos[i])) @ np.linalg.inv(cell.matrix)).astype(np.int32)
    return i, j, img, dist


def _half_pairs_small_cell(pos, cell, rmax, chunk: int = 256):
    """Brute force over periodic images for cells thinner than 2 * rmax (e.g. primitive cells)."""
    H = cell.matrix
    frac = cell.to_fractional(pos)
    shift = np.where(cell.pbc, np.floor(frac), 0.0)
    xw = pos - shift @ H
    nimg = np.where(cell.pbc, np.ceil(rmax / cell.perpendicular_widths()).astype(int) + 1, 0)
    shifts = np.array([(a, b, c) for a in range(-nimg[0], nimg[0] + 1) for b in range(-nimg[1], nimg[1] + 1)
                       for c in range(-nimg[2], nimg[2] + 1)])
    n = len(pos)
    out_i, out_j, out_s, out_d = [], [], [], []
    for s in shifts:
        positive = next((v > 0 for v in s if v != 0), False)   # self-images: keep one of s, -s
        off = s @ H
        for i0 in range(0, n, chunk):
            d = xw[None, :, :] - xw[i0:i0 + chunk, None, :] + off
            dist = np.sqrt((d * d).sum(-1))
            ii, jj = np.nonzero(dist < rmax)
            ii += i0
            keep = (ii < jj) | ((ii == jj) & positive)
            ii, jj = ii[keep], jj[keep]
            out_i.append(ii); out_j.append(jj)
            out_s.append(np.broadcast_to(s, (len(ii), 3)))
            out_d.append(dist[ii - i0, jj])
    i = np.concatenate(out_i).astype(np.int64)
    j = np.concatenate(out_j).astype(np.int64)
    img = (np.concatenate(out_s) - shift[j] + shift[i]).astype(np.int32)
    dist = np.concatenate(out_d)
    order = np.lexsort((j, i))
    return i[order], j[order], img[order], dist[order]


def find_bonds(data, cutoff: float | None = None, pairs=None, lower_cutoff: float = 0.0, backend=None) -> Bonds:
    """Bonds shorter than ``cutoff`` or, with ``pairs`` (bond rules, see :func:`normalize_pairs`),
    bonds whose types and length match the most specific rules for their pair."""
    pairs = normalize_pairs(pairs)
    rmax = max((p[3] for p in pairs), default=0.0) if pairs else float(cutoff or 0.0)
    if data.count == 0 or rmax <= 0:
        return Bonds()
    i, j, img, dist = _half_pairs(data, rmax, backend)
    keep = dist >= lower_cutoff
    if pairs:
        types = np.asarray(data.types).astype(str)
        keep &= _pair_mask(types[i], types[j], dist, img, pairs)
    return Bonds(np.column_stack([i[keep], j[keep]]), img[keep])


def bond_segments(data, bonds: Bonds, colors: np.ndarray):
    """Half-bond cylinders for drawing: (start, end, rgb, bond index), two per bond. Each half
    takes the color of its atom unless the bond has a 'Color' property."""
    i, j = bonds.topology.T
    d = bonds.vectors(data)
    pos = data.positions
    start = np.concatenate([pos[i], pos[j]])
    end = np.concatenate([pos[i] + 0.5 * d, pos[j] - 0.5 * d])
    if "Color" in bonds:
        rgb = np.tile(np.clip(np.asarray(bonds["Color"], float), 0, 1), (2, 1))
    else:
        rgb = np.concatenate([colors[i], colors[j]])
    return start, end, rgb, np.tile(np.arange(bonds.count), 2)
