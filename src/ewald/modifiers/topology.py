"""Bond-network analysis: bond angle and bond length distributions, ring statistics.

The distributions match OVITO 3.16's Bond Angle / Bond Length Distribution modifiers bin for bin
(tests/data/make_analysis_reference.py). FindRings finds shortest-path rings (Franzblau 1991).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..pipeline import Modifier

_PARTITIONS = ("off", "particle_type", "particle_selection", "bond_type", "bond_selection")


def _require_bonds(data, who):
    if data.bonds is None or not data.bonds.count:
        raise ValueError(f"{who} needs bonds; add CreateBonds (or load a file with bonds) first")
    return data.bonds


def _partition(data, partition):
    """``(per-particle class or None, per-bond class or None, class names)`` for a partition mode."""
    p = str(partition).lower().replace(" ", "_")
    p = {"byparticletype": "particle_type", "byparticleselection": "particle_selection",
         "bybondtype": "bond_type", "bybondselection": "bond_selection"}.get(p.replace("_", ""), p)
    if p not in _PARTITIONS:
        raise ValueError(f"partition must be one of {', '.join(_PARTITIONS)}")
    if p == "off":
        return None, None, None
    if p.endswith("selection"):
        source = data.particles if p.startswith("particle") else data.bonds.properties
        if "Selection" not in source:
            raise ValueError(f"partition={p!r} needs a 'Selection' property")
        return ((np.asarray(source["Selection"]) != 0).astype(np.int64),
                p.startswith("particle"), ["Other", "Selected"])
    source = data.particles if p.startswith("particle") else data.bonds.properties
    name = "Particle Type" if p.startswith("particle") else "Bond Type"
    if name not in source:
        raise ValueError(f"partition={p!r} needs a '{name}' property")
    values = np.asarray(source[name])
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(first)            # classes in order of first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse.ravel()], p.startswith("particle"), [str(values[first[k]]) for k in order]


def _pair_ids(a, b, n):
    """Index of the unordered pair (a, b), a, b < n, in the order (0,0), (0,1), ..., (1,1), ..."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo * n - lo * (lo - 1) // 2 + (hi - lo)


def _pair_names(names):
    return [f"{names[a]}-{names[b]}" for a in range(len(names)) for b in range(a, len(names))]


def bond_angles(data, bonds):
    """Every angle between two bonds at a shared particle. Returns ``(vertex, other, bond, a, b,
    cos)``: the bond ends are listed from both sides (``vertex[k]``, ``other[k]``, ``bond[k]``),
    and each angle is the pair of half-bonds ``a``, ``b`` with cosine ``cos``."""
    vec = bonds.vectors(data)
    i, j = bonds.topology.T
    # Each bond seen from both ends: the half-bond from vertex i points along +vec, from j along -vec.
    vertex = np.concatenate([i, j])
    other = np.concatenate([j, i])
    hv = np.concatenate([vec, -vec])
    bond = np.concatenate([np.arange(bonds.count)] * 2)
    order = np.argsort(vertex, kind="stable")
    vertex, other, hv, bond = vertex[order], other[order], hv[order], bond[order]
    starts = np.searchsorted(vertex, np.arange(data.count + 1))
    deg = np.diff(starts)
    # All unordered pairs of half-bonds sharing a vertex.
    a_list, b_list = [], []
    for k in np.unique(deg[deg > 1]):
        ia, ib = np.triu_indices(k, 1)
        s = starts[:-1][deg == k]
        a_list.append((s[:, None] + ia[None, :]).ravel())
        b_list.append((s[:, None] + ib[None, :]).ravel())
    a = np.concatenate(a_list) if a_list else np.zeros(0, np.int64)
    b = np.concatenate(b_list) if b_list else np.zeros(0, np.int64)
    va, vb = hv[a], hv[b]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1))
    ok = np.isfinite(cos)
    cos = np.clip(cos[ok], -1.0, 1.0)
    a, b = a[ok], b[ok]
    return vertex, other, bond, a, b, cos


class BondAngleDistribution(Modifier):
    """Histogram of the angles between pairs of bonds at each particle.

    ``bins`` bins over 0-180 degrees, or over cos(angle) from -1 to 1 with ``cosine_mode``.
    ``partition``: "off", "particle_type" (one column per type triplet "A-B-C", B the vertex),
    "particle_selection", "bond_type" (one column per bond-type pair) or "bond_selection".
    Writes ``tables['bond-angle-distr']`` with 'Angle' (or 'Cosine') and 'Count' or
    'Count <class>' columns.
    """

    def __init__(self, bins: int = 180, cosine_mode: bool = False, partition: str = "off"):
        self.bins = bins
        self.cosine_mode = cosine_mode
        self.partition = partition

    def modify(self, data, ctx):
        bonds = _require_bonds(data, "BondAngleDistribution")
        nbins = max(1, int(self.bins))
        vertex, other, bond, a, b, cos = bond_angles(data, bonds)
        if self.cosine_mode:
            idx = ((cos + 1.0) * (nbins / 2.0)).astype(np.int64)
            x = -1.0 + (np.arange(nbins) + 0.5) * (2.0 / nbins)
            xname = "Cosine"
        else:
            angle = np.degrees(np.arccos(cos))
            idx = (angle * (nbins / 180.0)).astype(np.int64)
            x = (np.arange(nbins) + 0.5) * (180.0 / nbins)
            xname = "Angle"
        idx = np.clip(idx, 0, nbins - 1)
        cls, per_particle, names = _partition(data, self.partition)
        table = {xname: x}
        if cls is None:
            table["Count"] = np.bincount(idx, minlength=nbins).astype(np.int64)
        else:
            nc = len(names)
            if per_particle:
                ends = _pair_ids(cls[other[a]], cls[other[b]], nc)
                npair = nc * (nc + 1) // 2
                key = cls[vertex[a]] * npair + ends
                labels = [f"{names[e1]}-{names[v]}-{names[e2]}" for v in range(nc)
                          for e1 in range(nc) for e2 in range(e1, nc)]
            else:
                key = _pair_ids(cls[bond[a]], cls[bond[b]], nc)
                labels = _pair_names(names)
            hist = np.bincount(key * nbins + idx, minlength=len(labels) * nbins).reshape(len(labels), nbins)
            for label, h in zip(labels, hist):
                table[f"Count {label}"] = h.astype(np.int64)
        data.tables["bond-angle-distr"] = table


class BondLengthDistribution(Modifier):
    """Histogram of bond lengths from 0 to ``length_cutoff`` in ``bins`` bins (longer bonds are skipped).

    ``partition`` as in :class:`BondAngleDistribution` (particle-type pairs "A-B", or one column
    per bond type). Writes ``tables['bond-length-distr']`` with 'Bond Length' and 'Count' or
    'Count <class>' columns.
    """

    def __init__(self, bins: int = 100, length_cutoff: float = 4.0, partition: str = "off"):
        self.bins = bins
        self.length_cutoff = length_cutoff
        self.partition = partition

    def modify(self, data, ctx):
        bonds = _require_bonds(data, "BondLengthDistribution")
        nbins = max(1, int(self.bins))
        rmax = float(self.length_cutoff)
        length = bonds.lengths(data)
        keep = length <= rmax
        idx = np.clip((length[keep] * (nbins / rmax)).astype(np.int64), 0, nbins - 1)
        table = {"Bond Length": (np.arange(nbins) + 0.5) * (rmax / nbins)}
        cls, per_particle, names = _partition(data, self.partition)
        if cls is None:
            table["Count"] = np.bincount(idx, minlength=nbins).astype(np.int64)
        else:
            nc = len(names)
            if per_particle:
                i, j = bonds.topology[keep].T
                key, labels = _pair_ids(cls[i], cls[j], nc), _pair_names(names)
            else:
                key, labels = cls[keep], names
            hist = np.bincount(key * nbins + idx, minlength=len(labels) * nbins).reshape(len(labels), nbins)
            for label, h in zip(labels, hist):
                table[f"Count {label}"] = h.astype(np.int64)
        data.tables["bond-length-distr"] = table


class FindRings(Modifier):
    """Rings of ``minimum_ring_size``..``maximum_ring_size`` bonds in the bond network.

    A ring counts when it is a shortest-path ring (Franzblau, Phys. Rev. B 44, 4925 (1991)):
    between any two of its atoms, the shorter way around the ring is a shortest path in the
    whole network. A chain that closes on itself only through the periodic boundary (its bonds'
    periodic images do not add up to zero, so it winds around the cell instead of enclosing a
    region) is not a ring and is left out; ``periodic_rings=True`` keeps it, as OVITO does, and
    ``FindRings.periodic_rings`` counts how many were left out. Writes ``tables['ring-size-histogram']`` ('Ring Size', 'Count'),
    ``tables['<N>-rings']`` with 'Particle Indices' (one ring per row, starting at its smallest
    index), the per-particle 'Ring Count' and ``FindRings.count.<N>``. OVITO's Find Rings modifier
    finds the same rings, except that it keeps fewer of the near-maximum-size rings that
    differ from another ring of the same size by a smaller ring.
    """

    def __init__(self, minimum_ring_size: int = 4, maximum_ring_size: int = 8, periodic_rings: bool = False):
        self.minimum_ring_size = minimum_ring_size
        self.maximum_ring_size = maximum_ring_size
        self.periodic_rings = periodic_rings

    def modify(self, data, ctx):
        bonds = _require_bonds(data, "FindRings")
        lo, hi = int(self.minimum_ring_size), int(self.maximum_ring_size)
        if lo < 3 or hi < lo:
            raise ValueError("ring sizes must satisfy 3 <= minimum_ring_size <= maximum_ring_size")
        rings = shortest_path_rings(bonds.topology, data.count, hi)
        if getattr(self, "periodic_rings", False):
            wrapping = set()
        else:
            images = bond_images(bonds)
            wrapping = {r for r in rings if lo <= len(r) and winds_around_cell(r, images)}
        rings = rings - wrapping
        data.attributes["FindRings.periodic_rings"] = len(wrapping)
        sizes = np.arange(lo, hi + 1)
        by_size = {n: sorted(r for r in rings if len(r) == n) for n in sizes}
        data.tables["ring-size-histogram"] = {"Ring Size": sizes,
                                              "Count": np.array([len(by_size[n]) for n in sizes], np.int64)}
        count = np.zeros(data.count, np.int64)
        for n in sizes:
            arr = np.array(by_size[n], np.int64).reshape(-1, n)
            data.tables[f"{n}-rings"] = {"Particle Indices": arr}
            data.attributes[f"FindRings.count.{n}"] = len(arr)
            np.add.at(count, arr.ravel(), 1)
        data.particles["Ring Count"] = count


def bond_images(bonds) -> dict:
    """``{(i, j): periodic image of j seen from i}`` for both directions of every bond."""
    images = {}
    for (i, j), img in zip(bonds.topology.tolist(), bonds.pbc_images.tolist()):
        images.setdefault((i, j), img)
        images.setdefault((j, i), [-v for v in img])
    return images


def winds_around_cell(ring, images) -> bool:
    """Whether the closed bond path ``ring`` winds around the periodic cell: the periodic images
    of its bonds (``images`` from :func:`bond_images`), taken in walking order, do not add up to zero."""
    total = [0, 0, 0]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        img = images.get((a, b), (0, 0, 0))
        total = [t + v for t, v in zip(total, img)]
    return any(total)


def _adjacency(topology, n):
    i, j = np.asarray(topology, np.int64).T
    keep = i != j
    a = np.concatenate([i[keep], j[keep]])
    b = np.concatenate([j[keep], i[keep]])
    pairs = np.unique(np.stack([a, b], 1), axis=0)
    starts = np.searchsorted(pairs[:, 0], np.arange(n + 1))
    nb = pairs[:, 1].tolist()
    return [nb[starts[k]:starts[k + 1]] for k in range(n)]


def _bfs(adj, s, depth):
    dist = {s: 0}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        du = dist[u]
        if du >= depth:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = du + 1
                queue.append(v)
    return dist


def shortest_path_rings(topology, n: int, max_size: int) -> set[tuple]:
    """All shortest-path rings of up to ``max_size`` atoms, each as a tuple that starts at its
    smallest index and continues towards the smaller of that atom's two ring neighbors."""
    adj = _adjacency(topology, n)
    half = max_size // 2
    dist = [_bfs(adj, s, half + 1) for s in range(n)]
    rings = set()
    for s in range(n):
        ds = dist[s]
        paths = {}

        def shortest(t):
            # every shortest path s -> t through atoms > s (s is the ring's smallest atom)
            if t not in paths:
                if t == s:
                    paths[t] = [[s]]
                else:
                    d = ds[t]
                    paths[t] = [p + [t] for u in adj[t] if u >= s and ds.get(u, -1) == d - 1
                                for p in shortest(u)]
            return paths[t]

        for t, k in ds.items():
            if t <= s or 2 * k > max_size:
                continue
            ps = shortest(t)
            for x in range(len(ps)):          # even rings: two disjoint shortest paths s -> t
                inner = set(ps[x][1:-1])
                for y in range(x + 1, len(ps)):
                    if inner.isdisjoint(ps[y][1:-1]):
                        _accept(ps[x] + ps[y][-2:0:-1], dist, rings)
            if 2 * k + 1 <= max_size:          # odd rings: shortest paths to the two ends of a bond
                for t2 in adj[t]:
                    if t2 > t and ds.get(t2) == k:
                        for pa in ps:
                            inner = set(pa[1:])
                            for pb in shortest(t2):
                                if inner.isdisjoint(pb[1:]):
                                    _accept(pa + pb[:0:-1], dist, rings)
    return rings


def _accept(ring, dist, rings):
    size = len(ring)
    for a in range(size):
        da = dist[ring[a]]
        for b in range(a + 2, size):
            if da.get(ring[b], size) < min(b - a, size - b + a):
                return
    if ring[1] > ring[-1]:
        ring = [ring[0]] + ring[:0:-1]
    rings.add(tuple(ring))
