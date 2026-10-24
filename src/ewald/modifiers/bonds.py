"""Bond creation and editing, and coordination polyhedra built from bonds."""
from __future__ import annotations

import numpy as np

from ..bonds import Bonds, covalent_pairs, find_bonds, minimum_images, normalize_pairs
from ..mesh import SurfaceMesh
from ..pipeline import Modifier

_MODES = ("uniform", "pairwise", "covalent")


class CreateBonds(Modifier):
    """Create bonds between nearby particles (OVITO's CreateBondsModifier, plus VESTA's bond search).

    ``mode``: 'uniform' bonds every pair closer than ``cutoff``; 'pairwise' uses the pair
    table ``pairs``, VESTA-style rules ``(A, B, dmin, dmax)`` or ``(A, B, dmax)`` with '*' for any
    type, or a ``{(A, B): dmax}`` dict (also filled by :meth:`set_pairwise_cutoff`). The most specific
    rules for a pair decide it (A-B before A-* before *-*; dmax 0 turns a pair off), and a fifth item
    ``False`` (``(A, B, dmin, dmax, False)``) keeps that rule's bonds inside the cell, none through
    periodic boundaries (molecules, slabs, unwrapped trajectories). 'covalent' bonds A-B
    pairs shorter than ``factor * (r_A + r_B)`` with covalent radii, VESTA's default.
    Pairs shorter than ``lower_cutoff`` are skipped. Bonds are appended to existing ones and
    stored in ``data.bonds`` ('Topology', 'Periodic Image'); attribute 'CreateBonds.num_bonds'.
    Works in cells smaller than twice the cutoff (primitive cells).
    """

    def __init__(self, cutoff: float = 3.2, mode: str = "uniform", pairs=None, lower_cutoff: float = 0.0,
                 factor: float = 1.15):
        self.cutoff = cutoff
        self.mode = mode
        self.pairs = pairs
        self.lower_cutoff = lower_cutoff
        self.factor = factor

    def set_pairwise_cutoff(self, type_a, type_b, cutoff: float):
        """OVITO API: bond ``type_a``-``type_b`` pairs shorter than ``cutoff`` (switches to 'pairwise')."""
        rows = [p for p in normalize_pairs(self.pairs) if {p[0], p[1]} != {str(type_a), str(type_b)}]
        if cutoff > 0:
            rows.append((str(type_a), str(type_b), 0.0, float(cutoff)))
        self.pairs = rows
        self.mode = "pairwise"

    def modify(self, data, ctx):
        if self.mode not in _MODES:
            raise ValueError(f"unknown mode {self.mode!r}; use one of {_MODES}")
        if self.mode == "uniform":
            self._check_positive("cutoff")
            new = find_bonds(data, cutoff=self.cutoff, lower_cutoff=self.lower_cutoff, backend=ctx.backend)
        else:
            pairs = covalent_pairs(data.types, self.factor) if self.mode == "covalent" else self.pairs
            if not normalize_pairs(pairs):
                raise ValueError("mode 'pairwise' needs a pair table, e.g. pairs=[('Ti', 'O', 2.4)]")
            new = find_bonds(data, pairs=pairs, lower_cutoff=self.lower_cutoff, backend=ctx.backend)
        data.bonds = new if data.bonds is None else data.bonds.extend(new)
        data.attributes["CreateBonds.num_bonds"] = new.count


def _identifiers(data):
    return np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)


def _rows_of(data, ids):
    """Row index of each particle identifier (-1 when absent from this frame)."""
    here = _identifiers(data)
    ids = np.asarray(ids).ravel()
    if not len(here):
        return np.full(len(ids), -1)
    order = np.argsort(here, kind="stable")
    rows = order[np.clip(np.searchsorted(here[order], ids), 0, len(here) - 1)]
    return np.where(here[rows] == ids, rows, -1)


class EditBonds(Modifier):
    """Add or delete individual bonds by hand, as in VESTA's bond editing.

    ``add`` and ``remove`` are lists of ``(a, b)`` particle identifiers ('Particle Identifier',
    or the index when there is none), so edits follow the atoms through a trajectory. Added
    bonds use the shortest periodic image; ``remove`` deletes every bond between a and b.
    The GUI's bond tool appends to these lists. Attribute 'EditBonds.num_bonds'.
    """

    def __init__(self, add=(), remove=()):
        self.add = [tuple(p) for p in add]
        self.remove = [tuple(p) for p in remove]

    def toggle(self, a, b, bonded: bool):
        """Record an edit: delete the a-b bond if ``bonded``, add it otherwise. Undoing an earlier
        edit of the same pair just drops that edit."""
        key = {a, b}
        had_add = any(set(p) == key for p in self.add)
        had_remove = any(set(p) == key for p in self.remove)
        self.add = [p for p in self.add if set(p) != key]
        self.remove = [p for p in self.remove if set(p) != key]
        if bonded and not had_add:
            self.remove.append((a, b))
        elif not bonded and not had_remove:
            self.add.append((a, b))

    def modify(self, data, ctx):
        bonds = data.bonds if data.bonds is not None else Bonds()
        if self.remove and bonds.count:
            ids = _identifiers(data)
            t = ids[bonds.topology]
            lo, hi = np.minimum(t[:, 0], t[:, 1]), np.maximum(t[:, 0], t[:, 1])
            gone = np.zeros(bonds.count, bool)
            for a, b in self.remove:
                gone |= (lo == min(a, b)) & (hi == max(a, b))
            bonds = bonds.take(~gone)
        if self.add:
            rows = _rows_of(data, np.asarray(self.add).reshape(-1, 2)).reshape(-1, 2)
            rows = rows[(rows >= 0).all(axis=1) & (rows[:, 0] != rows[:, 1])]
            if bonds.count:  # skip pairs that are bonded already
                have = {tuple(p) for p in np.sort(bonds.topology, axis=1).tolist()}
                rows = rows[[tuple(sorted(r)) not in have for r in rows.tolist()]].reshape(-1, 2)
            if len(rows):
                bonds = bonds.extend(Bonds(rows, minimum_images(data, rows[:, 0], rows[:, 1])))
        data.bonds = bonds
        data.attributes["EditBonds.num_bonds"] = bonds.count


class CoordinationPolyhedra(Modifier):
    """Polyhedra spanned by the bonded neighbors of each center particle (OVITO's
    CoordinationPolyhedraModifier, VESTA's polyhedral style). Uses the bonds from upstream
    (CreateBonds); without any it finds them as ``CreateBonds(mode="covalent")`` would, for
    the polyhedra only (``data.bonds`` stays empty).

    Centers are particles of ``center_types`` (e.g. ``["Ti"]``) or, when None, the selected
    particles. Each polyhedron is the convex hull of the neighbor positions (unwrapped across
    the cell); three or more coplanar neighbors give a flat polygon. Faces take the center's
    color; ``transparency`` and ``edges`` (outline the polyhedron edges) control the look.
    Writes ``data.surfaces['coordination-polyhedra']`` with face properties 'Color' and 'Region'
    (index of the center particle) and the attribute 'CoordinationPolyhedra.count'.
    """

    def __init__(self, center_types=None, transparency: float = 0.25, edges: bool = True):
        self.center_types = center_types
        self.transparency = transparency
        self.edges = edges

    def modify(self, data, ctx):
        from ..render import default_colors

        if self.center_types is not None:
            types = [self.center_types] if isinstance(self.center_types, str) else list(self.center_types)
            centers = np.flatnonzero(np.isin(np.asarray(data.types).astype(str), [str(t) for t in types]))
        else:
            centers = np.flatnonzero(data.selection)
            if not len(centers):
                raise ValueError("no center particles: select them or set center_types, e.g. ['Ti']")
        bonds = data.bonds
        if bonds is None:  # e.g. the GUI's display-only bonds: search covalent bonds like VESTA
            bonds = find_bonds(data, pairs=covalent_pairs(data.types), backend=getattr(ctx, "backend", None))
        i, j = bonds.topology.T
        d = bonds.vectors(data)
        # neighbor vectors seen from each end of every bond
        src = np.concatenate([i, j])
        vec = np.concatenate([d, -d])
        order = np.argsort(src, kind="stable")
        src, vec = src[order], vec[order]
        starts = np.searchsorted(src, centers)
        ends = np.searchsorted(src, centers, side="right")
        colors = np.clip(default_colors(data), 0, 1)
        verts, faces, regions, edges = [], [], [], []
        nv = 0
        for c, s, e in zip(centers, starts, ends):
            if e - s < 3:
                continue
            pts = data.positions[c] + vec[s:e]
            tri, edg = _hull(pts)
            if tri is None:
                continue
            verts.append(pts)
            faces.append(tri + nv)
            edges.append(edg + nv)
            regions.append(np.full(len(tri), c))
            nv += len(pts)
        if faces:
            region = np.concatenate(regions)
            mesh = SurfaceMesh(np.concatenate(verts), np.concatenate(faces),
                               {"Color": colors[region], "Region": region},
                               np.concatenate(edges) if self.edges else None, transparency=self.transparency)
        else:
            mesh = SurfaceMesh(transparency=self.transparency)
        data.surfaces["coordination-polyhedra"] = mesh
        data.attributes["CoordinationPolyhedra.count"] = len(faces)


def _hull(pts):
    """Triangles and outline edges of the convex hull of ``pts`` (flat polygon if coplanar)."""
    from scipy.spatial import ConvexHull, QhullError

    centered = pts - pts.mean(0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    scale = max(sv[0], 1e-12)
    if sv[1] / scale < 1e-6:
        return None, None                                # collinear
    if len(pts) < 4 or sv[2] / scale < 1e-3:             # planar: fan around the centroid
        uv = centered @ vt[:2].T
        ring = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
        try:
            ring = ring[np.isin(ring, ConvexHull(uv).vertices)]
        except (QhullError, ValueError):
            pass
        if len(ring) < 3:
            return None, None
        tri = np.column_stack([np.full(len(ring) - 2, ring[0]), ring[1:-1], ring[2:]])
        return tri, np.column_stack([ring, np.roll(ring, -1)])
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None, None
    tri = hull.simplices.copy()
    # orient triangles outward
    normals = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    flip = np.einsum("ij,ij->i", normals, pts[tri[:, 0]] - pts.mean(0)) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    # edges between non-coplanar neighbor triangles (drop diagonals of flat faces)
    eq = hull.equations[:, :3]
    edges = set()
    for t, nb in enumerate(hull.neighbors):
        for k, u in enumerate(nb):
            if t < u and eq[t] @ eq[u] < 1 - 1e-6:
                a, b = np.delete(hull.simplices[t], k)
                edges.add((min(a, b), max(a, b)))
    return tri, np.array(sorted(edges), np.int64).reshape(-1, 2)
