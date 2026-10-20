"""Replicated cell display: ``na x nb x nc`` periodic copies of the structure on screen.

OVITO makes a supercell picture with a Replicate modifier; VESTA draws the boundary range over
several cells. :func:`tile` is the display-only form of both: a copy of the data with the cell
tiled ``n`` times (bonds joined across the copies, polyhedra and isosurfaces repeated, the
drawn cell grown), and the real atom of every drawn one. Modifiers, analysis and export never
see the copies; :func:`ewald.scene.build` (``replicate=``) uses it before the boundary range,
so ``boundary`` then applies to the replicated cell.
"""
from __future__ import annotations

import numpy as np

#: which cell edges a replicated view draws
CELL_MODES = {
    "super": "the outline of the whole replicated block",
    "unit": "the original cell only",
    "all": "every copied cell (a lattice grid)",
}


def normalize_replicate(replicate) -> tuple[int, int, int] | None:
    """``replicate`` as three ints >= 1, or None when off (None, False, 1 or (1, 1, 1))."""
    if replicate is None or replicate is False:
        return None
    n = np.asarray(replicate)
    if n.ndim == 0:
        n = np.repeat(n, 3)
    if n.shape != (3,) or not np.allclose(n, np.rint(n)) or (np.rint(n) < 1).any():
        raise ValueError(f"replicate must be n or (na, nb, nc) with integers >= 1, got {replicate!r}")
    out = tuple(int(x) for x in np.rint(n))
    return None if out == (1, 1, 1) else out


def tile(data, replicate, bonds=None):
    """``(display, source)``: ``data`` tiled ``replicate`` times along a, b, c and the real atom
    of every displayed one (copy-major: copy ``c`` of atom ``i`` is ``c * N + i``).

    ``bonds`` (default ``data.bonds``) are joined across the copies (a bond leaving the block
    keeps a periodic image of the grown cell); meshes are repeated, polyhedra on their center's
    copy. Returns ``(data, None)`` when there is nothing to tile."""
    from .data import Cell
    from .modifiers.transform import _replicate_bonds

    n = normalize_replicate(replicate)
    cell = data.cell
    if n is None or not cell.is_valid or data.count == 0:
        return data, None
    count = data.count
    copies = np.array([(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])
    shift = copies @ cell.matrix
    source = np.tile(np.arange(count), len(copies))
    disp = data.copy()
    disp.particles = {k: np.asarray(a)[source] for k, a in data.particles.items()}
    disp.particles["Position"] = (np.asarray(data.positions)[None] + shift[:, None]).reshape(-1, 3)
    bonds = data.bonds if bonds is None else bonds
    disp.bonds = None if bonds is None else _replicate_bonds(bonds, n, count)
    disp.cell = Cell(cell.matrix * np.array(n)[:, None], cell.origin, cell.pbc)
    disp.surfaces = {k: _tile_mesh(m, shift, count) for k, m in data.surfaces.items()}
    return disp, source


def _tile_mesh(mesh, shift, count):
    """A translated copy of ``mesh`` per cell copy; a 'Region' (center atom) points at its copy."""
    if not mesh.vertex_count:
        return mesh
    k, nv = len(shift), mesh.vertex_count
    out = mesh.copy()
    out.vertices = (mesh.vertices[None] + shift[:, None]).reshape(-1, 3)
    out.faces = (mesh.faces[None] + nv * np.arange(k)[:, None, None]).reshape(-1, 3)
    if mesh.edges is not None:
        out.edges = (mesh.edges[None] + nv * np.arange(k)[:, None, None]).reshape(-1, 2)
    out.face_properties = {n: np.concatenate([np.asarray(v)] * k) for n, v in mesh.face_properties.items()}
    if "Region" in mesh.face_properties:
        region = np.asarray(mesh.face_properties["Region"], np.int64)
        out.face_properties["Region"] = np.where(region[None] >= 0, region[None] + count * np.arange(k)[:, None],
                                                 region[None]).ravel()
    out.vertex_properties = {n: np.concatenate([np.asarray(v)] * k) for n, v in mesh.vertex_properties.items()}
    return out


def cell_edges(unit, n, mode: str = "super"):
    """Corner points and edge index pairs of the cell lines a replicated view draws."""
    from .scene import _CELL_EDGES

    if mode not in CELL_MODES:
        raise ValueError(f"unknown replicate_cell {mode!r}; use one of {', '.join(CELL_MODES)}")
    a, b, c = unit.matrix
    if mode == "unit":
        blocks, size = [(0, 0, 0)], (1, 1, 1)
    elif mode == "super":
        blocks, size = [(0, 0, 0)], n
    else:
        blocks = [(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
        size = (1, 1, 1)
    pts, edges = [], []
    for blk in blocks:
        o = unit.origin + np.asarray(blk) @ unit.matrix
        A, B, C = a * size[0], b * size[1], c * size[2]
        base = len(pts)
        pts += [o, o + A, o + B, o + C, o + A + B, o + A + C, o + B + C, o + A + B + C]
        edges += [[base + i, base + j] for i, j in _CELL_EDGES]
    return np.array(pts), edges
