"""VESTA-style boundary: periodic images of atoms on the cell faces, edges and corners.

VESTA draws every atom whose fractional coordinates fall inside a range (by default 0..1, so an
atom at x = 0 also shows at x = 1) and completes the bonds and coordination polyhedra the cell
cuts. :func:`expand` does the same for drawing: it returns a copy of the data with the extra
periodic images ("ghost atoms") appended, bonds between displayed atoms made whole, and
polyhedra repeated on the ghost centers. It is display only: modifiers and analysis never see
the ghosts, and every ghost knows its real atom (``source``), so picking selects that atom.

``render(..., boundary=(-0.1, 1.1))`` and the GUI's Simulation cell panel use it; see
:func:`ewald.scene.build`.
"""
from __future__ import annotations

import numpy as np

#: what happens to a bond whose partner lies outside the boundary range
BOND_MODES = {
    "half": "half bond pointing out of the range (as without a boundary)",
    "complete": "add the partner atom, so every bond of a displayed atom is whole (VESTA's 'search additional atoms')",
    "none": "drop the bond",
}
#: meshes whose face 'Region' is the center particle; repeated on that particle's ghosts
PARTICLE_MESHES = ("coordination-polyhedra",)


def normalize_boundary(boundary) -> np.ndarray | None:
    """``boundary`` as a 3 x 2 array of fractional (lo, hi) per cell vector, or None.

    Accepts None/False (off), True (0..1), a margin ``m`` (-m..1+m), one ``(lo, hi)`` pair for all
    three axes, or three pairs."""
    if boundary is None or boundary is False:
        return None
    if boundary is True:
        boundary = (0.0, 1.0)
    b = np.asarray(boundary, float)
    if b.ndim == 0:
        b = np.array([-b, 1.0 + b])
    if b.shape == (2,):
        b = np.tile(b, (3, 1))
    if b.shape != (3, 2):
        raise ValueError(f"boundary must be a margin, (lo, hi) or three (lo, hi) pairs, got {boundary!r}")
    if np.any(b[:, 1] < b[:, 0]):
        raise ValueError(f"boundary lo must not exceed hi: {b.tolist()}")
    return b


def expand(data, boundary, bonds=None, *, bond_mode: str = "half", tol: float = 1e-4):
    """``(display, source)``: ``data`` plus the periodic images inside ``boundary`` (fractional
    range, see :func:`normalize_boundary`), and the real atom of every displayed one.

    The real atoms come first, unchanged (also those outside the range); ghosts follow with every
    particle property copied from their atom. ``bonds`` (default ``data.bonds``) become bonds
    between displayed atoms; a bond whose partner is outside the range follows ``bond_mode`` (see
    :data:`BOND_MODES`), except that the neighbors of polyhedron centers are always added so every
    drawn polyhedron has its corner atoms. Half bonds carry the bond property 'Half Bond'.
    Only periodic directions get images. Returns ``(data, None)`` when there is nothing to do.
    """
    rng = normalize_boundary(boundary)
    if bond_mode not in BOND_MODES:
        raise ValueError(f"unknown bond_mode {bond_mode!r}; use one of {', '.join(BOND_MODES)}")
    cell = data.cell
    if rng is None or not cell.is_valid or data.count == 0:
        return data, None
    n = data.count
    H = cell.matrix
    pbc = np.asarray(cell.pbc, bool)
    frac = cell.to_fractional(data.positions)
    lo = np.where(pbc, np.ceil(rng[:, 0] - tol - frac), 0).astype(np.int64)    # N x 3 image range
    hi = np.where(pbc, np.floor(rng[:, 1] + tol - frac), 0).astype(np.int64)
    src, shift = [np.arange(n)], [np.zeros((n, 3), np.int64)]
    if (hi >= lo).any():
        smin, smax = np.minimum(lo, hi).min(0), np.maximum(lo, hi).max(0)
        for s in np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(smin, smax)], indexing="ij"),
                          -1).reshape(-1, 3):
            if not s.any():
                continue
            hit = np.flatnonzero(((lo <= s) & (s <= hi)).all(1))
            src.append(hit)
            shift.append(np.broadcast_to(s, (len(hit), 3)))
    src, shift = np.concatenate(src), np.concatenate(shift).astype(np.int64)

    bonds = data.bonds if bonds is None else bonds
    disp_bonds = None
    if bonds is not None:
        u, v, img = _directed(bonds)
        n_in = len(src)  # displayed because they are inside the range (or real)
        # partners outside the range: added for polyhedron centers (or every atom in 'complete' mode)
        grow = np.ones(n, bool) if bond_mode == "complete" else _mesh_centers(data, n)
        if grow.any() and len(u):
            c, p, k = _partners(src, shift, u, v, img, n)
            want = (p < 0) & grow[src[c]]
            if want.any():
                extra = np.unique(np.column_stack([v[k[want]], shift[c[want]] + img[k[want]]]), axis=0)
                src = np.concatenate([src, extra[:, 0]])
                shift = np.concatenate([shift, extra[:, 1:]])
        disp_bonds = _display_bonds(bonds, src, shift, u, v, img, n, np.arange(len(src)) < n_in, bond_mode)

    disp = data.copy()
    disp.particles = {k: np.asarray(a)[src] for k, a in data.particles.items()}
    disp.particles["Position"] = data.positions[src] + shift @ H
    disp.bonds = disp_bonds
    disp.surfaces = dict(data.surfaces)
    for name in PARTICLE_MESHES:
        mesh = data.surfaces.get(name)
        if mesh is not None and mesh.face_count and "Region" in mesh.face_properties:
            disp.surfaces[name] = _repeat_mesh(mesh, src, shift, H)
    return disp, src


def _mesh_centers(data, n):
    centers = np.zeros(n, bool)
    for name in PARTICLE_MESHES:
        mesh = data.surfaces.get(name)
        if mesh is not None and "Region" in mesh.face_properties:
            r = np.asarray(mesh.face_properties["Region"], np.int64)
            centers[r[(r >= 0) & (r < n)]] = True
    return centers


def _directed(bonds):
    """Every bond in both directions: (u, v, image) with x_v - x_u + image @ H the bond vector."""
    i, j = bonds.topology.T
    img = bonds.pbc_images.astype(np.int64)
    return np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([img, -img])


def _codes(src, shift, n):
    """One integer per (particle, image shift) for lookups."""
    s = shift + 512
    return ((src.astype(np.int64) * 1024 + s[:, 0]) * 1024 + s[:, 1]) * 1024 + s[:, 2]


def _partners(src, shift, u, v, img, n):
    """For every displayed copy ``c`` of each directed bond's u: the displayed index ``p`` of its
    partner (-1 when absent) and the directed bond ``k``."""
    order = np.argsort(src, kind="stable")
    starts = np.searchsorted(src[order], np.arange(n))
    counts = np.bincount(src, minlength=n)
    k = np.repeat(np.arange(len(u)), counts[u])
    first = np.repeat(starts[u], counts[u])
    offset = np.arange(len(k)) - np.repeat(np.cumsum(counts[u]) - counts[u], counts[u])
    c = order[first + offset]
    p = v[k].copy()                                  # real atom to real atom: the partner itself
    look = np.flatnonzero((c >= n) | img[k].any(1))  # everything else is looked up
    if len(look):
        want = _codes(v[k[look]], shift[c[look]] + img[k[look]], n)
        have = _codes(src, shift, n)
        sorter = np.argsort(have)
        q = sorter[np.clip(np.searchsorted(have, want, sorter=sorter), 0, len(have) - 1)]
        p[look] = np.where(have[q] == want, q, -1)
    return c, p, k


def _display_bonds(bonds, src, shift, u, v, img, n, in_range, bond_mode):
    from .bonds import Bonds

    if not len(u):
        return Bonds()
    c, p, k = _partners(src, shift, u, v, img, n)
    full = (p >= 0) & (c < p)                       # each whole bond is found from both ends
    half = (p < 0) & in_range[c] & (bond_mode == "half")
    m = bonds.count
    bidx = np.concatenate([k[full], k[half]]) % m   # directed bond -> stored bond (its properties)
    out = bonds.take(bidx)
    # half bonds point at the real partner, through the image that gives the right bond vector
    topo = np.concatenate([np.column_stack([c[full], p[full]]), np.column_stack([c[half], v[k[half]]])])
    image = np.concatenate([np.zeros((full.sum(), 3), np.int64), shift[c[half]] + img[k[half]]])
    out.properties["Topology"] = topo.astype(np.int64)
    out.properties["Periodic Image"] = image.astype(np.int32)
    out.properties["Half Bond"] = np.r_[np.zeros(full.sum(), bool), np.ones(half.sum(), bool)]
    return out


def _repeat_mesh(mesh, src, shift, H):
    """``mesh`` plus a translated copy of each center's faces for every ghost of that center."""
    region = np.asarray(mesh.face_properties["Region"], np.int64)
    ghosts = np.flatnonzero(shift.any(1))
    verts, faces, edges = [mesh.vertices], [mesh.faces], [] if mesh.edges is None else [mesh.edges]
    fprops = {k: [np.asarray(a)] for k, a in mesh.face_properties.items()}
    vprops = {k: [np.asarray(a)] for k, a in mesh.vertex_properties.items()}
    nv = mesh.vertex_count
    if len(ghosts):
        keys = np.unique(shift[ghosts], axis=0)
        for s in keys:
            g = ghosts[(shift[ghosts] == s).all(1)]
            fmask = np.isin(region, src[g])
            if not fmask.any():
                continue
            used = np.unique(mesh.faces[fmask])
            remap = np.full(mesh.vertex_count, -1, np.int64)
            remap[used] = nv + np.arange(len(used))
            verts.append(mesh.vertices[used] + s @ H)
            faces.append(remap[mesh.faces[fmask]])
            for k, a in mesh.face_properties.items():
                fprops[k].append(np.asarray(a)[fmask])
            for k, a in mesh.vertex_properties.items():
                vprops[k].append(np.asarray(a)[used])
            if mesh.edges is not None:
                e = remap[mesh.edges]
                edges.append(e[(e >= 0).all(1)])
            nv += len(used)
    out = mesh.copy()
    out.vertices = np.concatenate(verts)
    out.faces = np.concatenate(faces)
    out.face_properties = {k: np.concatenate(v) for k, v in fprops.items()}
    out.vertex_properties = {k: np.concatenate(v) for k, v in vprops.items()}
    out.edges = None if mesh.edges is None else np.concatenate(edges)
    return out
