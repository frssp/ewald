"""CPU ray tracer for publication figures: spheres + cylinders, ambient occlusion, soft shadows.

Plays the role of OVITO's Tachyon renderer. Pure numba (no OSPRay needed): a
uniform grid accelerates ray casts, each pixel takes ``samples`` jittered rays,
and every hit is shaded with a key light (shadow ray), hemisphere ambient
occlusion (``ao_samples`` rays limited to ``ao_distance``) and a Blinn-Phong
highlight. Scenes come from the same data the OpenGL renderer uses, so colors,
radii, bonds and the camera match the viewport.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SPHERE, CYLINDER, TRIANGLE = 0, 1, 2


# ----------------------------------------------------------------------------- scene
class Scene:
    """Primitives in world coordinates (Angstrom)."""

    def __init__(self):
        self.sph_c = np.zeros((0, 3)); self.sph_r = np.zeros(0); self.sph_rgb = np.zeros((0, 3))
        self.cyl_a = np.zeros((0, 3)); self.cyl_b = np.zeros((0, 3)); self.cyl_r = np.zeros(0)
        self.cyl_rgb = np.zeros((0, 3))
        self.tri_a = np.zeros((0, 3)); self.tri_b = np.zeros((0, 3)); self.tri_c = np.zeros((0, 3))
        self.tri_rgb = np.zeros((0, 3)); self.tri_alpha = np.zeros(0)
        # per-vertex normals and colours (M x 3 x 3), for smooth isosurfaces and colour-mapped slices
        self.tri_n = np.zeros((0, 3, 3)); self.tri_vrgb = np.zeros((0, 3, 3))

    def add_spheres(self, centers, radii, colors):
        self.sph_c = np.vstack([self.sph_c, np.asarray(centers, float).reshape(-1, 3)])
        self.sph_r = np.concatenate([self.sph_r, np.broadcast_to(np.asarray(radii, float), (len(centers),))])
        self.sph_rgb = np.vstack([self.sph_rgb, np.broadcast_to(np.asarray(colors, float), (len(centers), 3))])

    def add_cylinders(self, a, b, radii, colors):
        a = np.asarray(a, float).reshape(-1, 3)
        self.cyl_a = np.vstack([self.cyl_a, a])
        self.cyl_b = np.vstack([self.cyl_b, np.asarray(b, float).reshape(-1, 3)])
        self.cyl_r = np.concatenate([self.cyl_r, np.broadcast_to(np.asarray(radii, float), (len(a),))])
        self.cyl_rgb = np.vstack([self.cyl_rgb, np.broadcast_to(np.asarray(colors, float), (len(a), 3))])

    def add_triangles(self, a, b, c, colors, alpha=1.0, normals=None, vertex_colors=None):
        """Two-sided triangles; ``alpha`` < 1 makes them see-through (no shadows, no AO).
        ``normals``/``vertex_colors`` (M x 3 x 3, one row per corner) are interpolated across
        each triangle; without them it is flat-shaded in its face colour."""
        a = np.asarray(a, float).reshape(-1, 3)
        b = np.asarray(b, float).reshape(-1, 3)
        c = np.asarray(c, float).reshape(-1, 3)
        rgb = np.broadcast_to(np.asarray(colors, float), (len(a), 3))
        if normals is None:
            nf = np.cross(b - a, c - a)
            nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-300)
            normals = np.repeat(nf[:, None], 3, axis=1)
        if vertex_colors is None:
            vertex_colors = np.repeat(rgb[:, None], 3, axis=1)
        self.tri_a = np.vstack([self.tri_a, a])
        self.tri_b = np.vstack([self.tri_b, b])
        self.tri_c = np.vstack([self.tri_c, c])
        self.tri_rgb = np.vstack([self.tri_rgb, rgb])
        self.tri_n = np.concatenate([self.tri_n, np.asarray(normals, float).reshape(-1, 3, 3)])
        self.tri_vrgb = np.concatenate([self.tri_vrgb, np.asarray(vertex_colors, float).reshape(-1, 3, 3)])
        self.tri_alpha = np.concatenate([self.tri_alpha, np.broadcast_to(np.asarray(alpha, float), (len(a),))])

    def bounds(self):
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        if len(self.sph_c):
            lo = np.minimum(lo, (self.sph_c - self.sph_r[:, None]).min(0))
            hi = np.maximum(hi, (self.sph_c + self.sph_r[:, None]).max(0))
        if len(self.cyl_a):
            for p in (self.cyl_a, self.cyl_b):
                lo = np.minimum(lo, (p - self.cyl_r[:, None]).min(0))
                hi = np.maximum(hi, (p + self.cyl_r[:, None]).max(0))
        for p in (self.tri_a, self.tri_b, self.tri_c):
            if len(p):
                lo = np.minimum(lo, p.min(0)); hi = np.maximum(hi, p.max(0))
        return lo, hi


def scene_from_data(data, *, radius_scale=1.0, type_colors=None, type_radii=None, bonds=None, bond_pairs=None,
                    bond_radius=0.15, show_cell=True, cell_color=(0.0, 0.0, 0.0), cell_radius=None,
                    radius_preset="covalent", oxidation_states=None, coordination=None, vectors=None,
                    arrow_resolution=24, view_direction=None, trajectory_width=2.0, visuals=None, frame=0,
                    boundary=None, boundary_bonds="half", replicate=None, replicate_cell="super", **_ignored):
    """Build a Scene with the same colors/radii/bonds/arrows as :func:`ewald.render.build_scene`.

    ``view_direction`` (scene -> camera) lets 'flat' arrows face the viewer evenly lit.
    ``visuals``: extra visual elements (plugins), see :mod:`ewald.scene`.
    """
    from . import scene as ir

    scene = ir.build(data, ctx=ir.SceneContext(frame=frame, view_direction=view_direction), visuals=visuals,
                     radius_scale=radius_scale, type_colors=type_colors, type_radii=type_radii,
                     radius_preset=radius_preset, oxidation_states=oxidation_states, coordination=coordination,
                     bonds=bonds, bond_pairs=bond_pairs, bond_radius=bond_radius, vectors=vectors,
                     trajectory_width=trajectory_width, show_cell=show_cell, cell_radius=cell_radius,
                     boundary=boundary, boundary_bonds=boundary_bonds, replicate=replicate,
                     replicate_cell=replicate_cell)
    return scene_from_ir(scene, arrow_resolution=arrow_resolution, view_direction=view_direction, ink=cell_color)


def scene_from_ir(scene, *, arrow_resolution=24, view_direction=None, ink=(0.0, 0.0, 0.0)) -> Scene:
    """Ray-tracer primitives for every layer of an :class:`ewald.scene.Scene`. The ray tracer is
    opaque except for triangles: sphere and cylinder ``alpha`` is ignored. Lines without a
    ``radius`` (screen-space only, such as trajectory lines) are not drawn."""
    from .render import _rgb

    sc = Scene()
    for layer in scene.layers:
        prim = layer.primitive
        if layer.kind == "spheres":
            if prim.count:
                c, r, rgb, a = prim.centers, prim.sizes(), prim.colors(), prim.alpha
                solid = a >= 1
                if solid.all():
                    sc.add_spheres(c, r, rgb)
                else:  # only triangles can be see-through: tessellate those spheres
                    if solid.any():
                        sc.add_spheres(c[solid], r[solid], rgb[solid])
                    _add_glass_spheres(sc, c[~solid], r[~solid], rgb[~solid], a[~solid])
        elif layer.kind == "cylinders":
            if prim.count:
                sc.add_cylinders(prim.start, prim.end, prim.radius, prim.rgb)
                if prim.cap_end:
                    sc.add_spheres(prim.end, prim.radius, prim.rgb)  # close the open tube end
        elif layer.kind == "mesh":
            mesh = prim
            if mesh.face_count:
                v, f = mesh.vertices, mesh.faces
                vn, vc = mesh.vertex_properties.get("Normal"), mesh.vertex_properties.get("Color")
                sc.add_triangles(v[f[:, 0]], v[f[:, 1]], v[f[:, 2]], np.clip(mesh.face_colors(), 0, 1),
                                 1.0 - mesh.transparency, None if vn is None else np.asarray(vn, float)[f],
                                 None if vc is None else np.clip(np.asarray(vc, float), 0, 1)[f])
                if mesh.edges is not None and len(mesh.edges):
                    e = mesh.edges
                    sc.add_cylinders(v[e[:, 0]], v[e[:, 1]], 0.5 * mesh.edge_width, _rgb(mesh.edge_color))
        elif layer.kind == "arrows":
            _add_arrows(sc, prim, arrow_resolution, view_direction)
        elif layer.kind == "lines":
            if prim.radius is None or not prim.count:
                continue
            seg = prim.segments()
            if not len(seg):
                continue
            if prim.rgb is None:
                seg_rgb = _rgb(ink)
            else:
                strip_of = np.concatenate([np.full(max(len(s) - 1, 0), k) for k, s in enumerate(prim.strips)])
                seg_rgb = prim.rgb[strip_of]
            sc.add_cylinders(prim.points[seg[:, 0]], prim.points[seg[:, 1]], prim.radius, seg_rgb)
            # round joints so corners look closed
            joints = np.unique(seg)
            if prim.rgb is None:
                sc.add_spheres(prim.points[joints], prim.radius, seg_rgb)
            else:
                rgb = np.empty((len(prim.points), 3))
                rgb[seg[:, 1]] = seg_rgb
                rgb[seg[:, 0]] = seg_rgb
                sc.add_spheres(prim.points[joints], prim.radius, rgb[joints])
    return sc


def _unit_sphere(level: int = 2):
    """Vertices and faces of a subdivided icosahedron on the unit sphere."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, out = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                p = verts[i] + verts[j]
                verts.append(p / np.linalg.norm(p))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = out
    return np.array(verts), np.array(f)


def _add_glass_spheres(sc, centers, radii, rgb, alpha, level=2):
    """See-through spheres as smooth triangle meshes (the ray tracer's transparent primitive)."""
    v, f = _unit_sphere(level)
    tri = v[f]                                          # F x 3 x 3 unit corners = normals
    keep = alpha > 0
    centers, radii, rgb, alpha = centers[keep], radii[keep], rgb[keep], alpha[keep]
    if not len(centers):
        return
    p = (centers[:, None, None] + radii[:, None, None, None] * tri[None]).reshape(-1, 3, 3)
    nf = len(f)
    sc.add_triangles(p[:, 0], p[:, 1], p[:, 2], np.repeat(rgb, nf, axis=0), np.repeat(alpha, nf),
                     np.tile(tri, (len(centers), 1, 1)), None)


def _add_arrows(sc, arrows, sides, view_direction):
    """Arrow shafts as cylinders, base/head discs and cones as smooth-normal triangles."""
    from .arrows import arrow_mesh

    if not arrows.count:
        return
    see_through = arrows.alpha < 1  # only triangles can be see-through
    verts, faces, normals, owner = arrow_mesh(arrows, sides, shaft=see_through)
    if arrows.shading == "flat" and view_direction is not None:
        toward = np.asarray(view_direction, float)
        normals = np.broadcast_to(toward / np.linalg.norm(toward), normals.shape)
    if not see_through:
        sc.add_cylinders(arrows.base, arrows.neck, arrows.radius, arrows.rgb)
    sc.add_triangles(verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]], arrows.rgb[owner[faces[:, 0]]],
                     arrows.alpha, normals[faces], None)


# ----------------------------------------------------------------------------- grid
def build_grid(scene: Scene, target_per_cell: float = 2.0):
    lo, hi = scene.bounds()
    pad = 1e-3 * max(float(np.max(hi - lo)), 1.0)
    lo, hi = lo - pad, hi + pad
    n_obj = len(scene.sph_r) + len(scene.cyl_r) + len(scene.tri_alpha)
    extent = hi - lo
    vol = float(np.prod(np.maximum(extent, 1e-6)))
    cell = (vol * target_per_cell / max(n_obj, 1)) ** (1 / 3)
    if len(scene.tri_alpha):  # surfaces fill a 2D sheet, not the volume: size cells by the triangles
        edge = float(np.median(np.linalg.norm(scene.tri_b - scene.tri_a, axis=1)))
        cell = min(cell, max(2.0 * edge, (vol / 2 ** 24) ** (1 / 3)))
    dims = np.clip(np.ceil(extent / cell).astype(np.int64), 1, 512)
    # object AABBs
    boxes_lo, boxes_hi, kinds, idx = [], [], [], []
    if len(scene.sph_r):
        boxes_lo.append(scene.sph_c - scene.sph_r[:, None]); boxes_hi.append(scene.sph_c + scene.sph_r[:, None])
        kinds.append(np.full(len(scene.sph_r), SPHERE)); idx.append(np.arange(len(scene.sph_r)))
    if len(scene.cyl_r):
        boxes_lo.append(np.minimum(scene.cyl_a, scene.cyl_b) - scene.cyl_r[:, None])
        boxes_hi.append(np.maximum(scene.cyl_a, scene.cyl_b) + scene.cyl_r[:, None])
        kinds.append(np.full(len(scene.cyl_r), CYLINDER)); idx.append(np.arange(len(scene.cyl_r)))
    if len(scene.tri_alpha):
        boxes_lo.append(np.minimum(np.minimum(scene.tri_a, scene.tri_b), scene.tri_c))
        boxes_hi.append(np.maximum(np.maximum(scene.tri_a, scene.tri_b), scene.tri_c))
        kinds.append(np.full(len(scene.tri_alpha), TRIANGLE)); idx.append(np.arange(len(scene.tri_alpha)))
    blo, bhi = np.vstack(boxes_lo), np.vstack(boxes_hi)
    kinds, idx = np.concatenate(kinds), np.concatenate(idx)
    size = extent / dims
    c0 = np.clip(((blo - lo) / size).astype(np.int64), 0, dims - 1)
    c1 = np.clip(((bhi - lo) / size).astype(np.int64), 0, dims - 1)
    starts, items = _fill_grid(c0, c1, dims, kinds * (1 << 40) + idx)
    return lo, size, dims, starts, items


@njit(cache=True)
def _fill_grid(c0, c1, dims, codes):
    ncell = dims[0] * dims[1] * dims[2]
    counts = np.zeros(ncell + 1, np.int64)
    for o in range(len(codes)):
        for x in range(c0[o, 0], c1[o, 0] + 1):
            for y in range(c0[o, 1], c1[o, 1] + 1):
                for z in range(c0[o, 2], c1[o, 2] + 1):
                    counts[(x * dims[1] + y) * dims[2] + z + 1] += 1
    starts = np.cumsum(counts)
    fill = starts[:-1].copy()
    items = np.empty(starts[-1], np.int64)
    for o in range(len(codes)):
        for x in range(c0[o, 0], c1[o, 0] + 1):
            for y in range(c0[o, 1], c1[o, 1] + 1):
                for z in range(c0[o, 2], c1[o, 2] + 1):
                    c = (x * dims[1] + y) * dims[2] + z
                    items[fill[c]] = codes[o]
                    fill[c] += 1
    return starts, items


# ----------------------------------------------------------------------------- intersection
@njit(cache=True, inline="always")
def _hit_sphere(o, d, c, r):
    ox, oy, oz = o[0] - c[0], o[1] - c[1], o[2] - c[2]
    b = ox * d[0] + oy * d[1] + oz * d[2]
    cc = ox * ox + oy * oy + oz * oz - r * r
    disc = b * b - cc
    if disc < 0:
        return -1.0
    s = math.sqrt(disc)
    t = -b - s
    if t > 1e-6:
        return t
    t = -b + s
    return t if t > 1e-6 else -1.0


@njit(cache=True, inline="always")
def _hit_cylinder(o, d, a, b, r):
    """Finite open cylinder (caps omitted; bond ends sit inside spheres)."""
    ax, ay, az = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    L = math.sqrt(ax * ax + ay * ay + az * az)
    if L < 1e-12:
        return -1.0
    ax /= L; ay /= L; az /= L
    ox, oy, oz = o[0] - a[0], o[1] - a[1], o[2] - a[2]
    dd = d[0] * ax + d[1] * ay + d[2] * az
    od = ox * ax + oy * ay + oz * az
    px, py, pz = d[0] - dd * ax, d[1] - dd * ay, d[2] - dd * az
    qx, qy, qz = ox - od * ax, oy - od * ay, oz - od * az
    A = px * px + py * py + pz * pz
    if A < 1e-14:
        return -1.0
    B = px * qx + py * qy + pz * qz
    C = qx * qx + qy * qy + qz * qz - r * r
    disc = B * B - A * C
    if disc < 0:
        return -1.0
    s = math.sqrt(disc)
    for t in ((-B - s) / A, (-B + s) / A):
        if t > 1e-6:
            h = od + t * dd
            if 0.0 <= h <= L:
                return t
    return -1.0


@njit(cache=True, inline="always")
def _hit_triangle(o, d, a, b, c):
    """Moller-Trumbore, two-sided."""
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    px, py, pz = d[1] * e2z - d[2] * e2y, d[2] * e2x - d[0] * e2z, d[0] * e2y - d[1] * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-14:
        return -1.0
    inv = 1.0 / det
    tx, ty, tz = o[0] - a[0], o[1] - a[1], o[2] - a[2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return -1.0
    qx, qy, qz = ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    return t if t > 1e-6 else -1.0


@njit(cache=True, inline="always")
def _dda_axis(o, d, lo, size, dim, t0):
    """Start cell, step, first boundary crossing and crossing interval of one axis."""
    c = int((o + d * (t0 + 1e-9) - lo) / size)
    c = min(max(c, 0), dim - 1)
    if d > 0:
        return c, 1, (lo + (c + 1) * size - o) / d, size / d
    if d < 0:
        return c, -1, (lo + c * size - o) / d, -size / d
    return c, 0, 1e300, 1e300


@njit(cache=True)
def _trace(o, d, tmax, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b, cyl_r,
           tri_a, tri_b, tri_c, tri_alpha, any_hit):
    """3D-DDA through the grid. Returns (t, code) of the nearest hit (code -1: miss).
    Shadow/AO rays (``any_hit``) pass through see-through triangles.
    The walk state lives in scalars: this runs for every primary, shadow and AO ray, and
    small per-call arrays cost a heap allocation each."""
    # clip ray to grid box
    t0, t1 = 0.0, tmax
    for k in range(3):
        hi = lo[k] + size[k] * dims[k]
        if abs(d[k]) < 1e-12:
            if o[k] < lo[k] or o[k] > hi:
                return -1.0, -1
        else:
            ta = (lo[k] - o[k]) / d[k]
            tb = (hi - o[k]) / d[k]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
    if t0 > t1:
        return -1.0, -1
    dx, dy, dz = dims[0], dims[1], dims[2]
    cx, sx, nx, ex = _dda_axis(o[0], d[0], lo[0], size[0], dx, t0)
    cy, sy, ny, ey = _dda_axis(o[1], d[1], lo[1], size[1], dy, t0)
    cz, sz, nz, ez = _dda_axis(o[2], d[2], lo[2], size[2], dz, t0)
    best_t, best = 1e300, -1
    while True:
        c = (cx * dy + cy) * dz + cz
        for s in range(starts[c], starts[c + 1]):
            code = items[s]
            kind = code >> 40
            i = code & ((1 << 40) - 1)
            if kind == 0:
                t = _hit_sphere(o, d, sph_c[i], sph_r[i])
            elif kind == 1:
                t = _hit_cylinder(o, d, cyl_a[i], cyl_b[i], cyl_r[i])
            elif any_hit and tri_alpha[i] < 0.999:
                t = -1.0
            else:
                t = _hit_triangle(o, d, tri_a[i], tri_b[i], tri_c[i])
            if t > 0 and t < best_t and t < tmax:
                best_t, best = t, code
                if any_hit:
                    return best_t, best
        tcell = min(nx, min(ny, nz))
        if best >= 0 and best_t <= tcell:
            return best_t, best
        if tcell > t1:
            return (best_t, best) if best >= 0 else (-1.0, -1)
        if ny < nx and ny <= nz:  # the axis whose boundary comes first (ties: x, then y)
            cy += sy
            if cy < 0 or cy >= dy:
                return (best_t, best) if best >= 0 else (-1.0, -1)
            ny += ey
        elif nz < nx and nz < ny:
            cz += sz
            if cz < 0 or cz >= dz:
                return (best_t, best) if best >= 0 else (-1.0, -1)
            nz += ez
        else:
            cx += sx
            if cx < 0 or cx >= dx:
                return (best_t, best) if best >= 0 else (-1.0, -1)
            nx += ex


@njit(cache=True, inline="always")
def _rand(state):
    # xorshift64*, per-pixel state array of length 1
    x = state[0]
    x ^= (x >> np.uint64(12)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27))
    state[0] = x
    return ((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _shade(p, n, d, cr, cg, cb, light, ambient, diffuse, specular, shininess, shadows, ao_samples, ao_dist,
           state, od, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b, cyl_r, tri_a, tri_b, tri_c,
           tri_alpha, aos):
    """Key light with shadow ray, cosine-weighted hemisphere AO and a Blinn-Phong highlight.

    AO directions follow the R2 low-discrepancy sequence, shifted per pixel (``aos``: two
    random offsets and a running index), so the AO rays of all samples of a pixel spread
    evenly over the hemisphere: 128 rays per pixel are as smooth as 512 random ones."""
    ndl = n[0] * light[0] + n[1] * light[1] + n[2] * light[2]
    lit = 0.0
    spec = 0.0
    if ndl > 0:
        vis = 1.0
        if shadows:
            ts, cs = _trace(p, light, 1e30, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b,
                            cyl_r, tri_a, tri_b, tri_c, tri_alpha, True)
            if cs >= 0:
                vis = 0.0
        lit = ndl * vis
        hx = light[0] - d[0]; hy = light[1] - d[1]; hz = light[2] - d[2]
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        ndh = (n[0] * hx + n[1] * hy + n[2] * hz) / hn
        if ndh > 0:
            spec = vis * ndh ** shininess
    ao = 1.0
    if ao_samples > 0:
        # tangent frame
        if abs(n[0]) > 0.9:
            tx, ty, tz = 0.0, 1.0, 0.0
        else:
            tx, ty, tz = 1.0, 0.0, 0.0
        bx = n[1] * tz - n[2] * ty; by = n[2] * tx - n[0] * tz; bz = n[0] * ty - n[1] * tx
        bn = math.sqrt(bx * bx + by * by + bz * bz)
        bx /= bn; by /= bn; bz /= bn
        cx = n[1] * bz - n[2] * by; cy = n[2] * bx - n[0] * bz; cz = n[0] * by - n[1] * bx
        hits = 0
        for a in range(ao_samples):
            g = aos[2]
            aos[2] += 1.0
            r1 = (aos[0] + g * 0.7548776662466927) % 1.0
            r2 = (aos[1] + g * 0.5698402909980532) % 1.0
            phi = 2 * math.pi * r1
            sr = math.sqrt(r2)
            x = math.cos(phi) * sr; y = math.sin(phi) * sr; z = math.sqrt(max(0.0, 1 - r2))
            od[0] = x * bx + y * cx + z * n[0]
            od[1] = x * by + y * cy + z * n[1]
            od[2] = x * bz + y * cz + z * n[2]
            ta, ca = _trace(p, od, ao_dist, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b,
                            cyl_r, tri_a, tri_b, tri_c, tri_alpha, True)
            if ca >= 0:
                hits += 1
        ao = 1.0 - hits / ao_samples
    shade = ambient * ao + diffuse * lit
    return (min(1.0, cr * shade + specular * spec), min(1.0, cg * shade + specular * spec),
            min(1.0, cb * shade + specular * spec))


@njit(cache=True, parallel=True)
def _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, samples, ao_samples, ao_dist,
            light, ambient, diffuse, specular, shininess, shadows, bg,
            lo, size, dims, starts, items, sph_c, sph_r, sph_rgb, cyl_a, cyl_b, cyl_r, cyl_rgb,
            tri_a, tri_b, tri_c, tri_n, tri_vrgb, tri_alpha, seed, rows):
    img = np.zeros((height, width, 4))
    for j in prange(height):
        py = rows[j]
        state = np.empty(1, np.uint64)
        o = np.empty(3); d = np.empty(3); n = np.empty(3); p = np.empty(3); od = np.empty(3)
        aos = np.zeros(3)
        for px in range(width):
            state[0] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(py * width + px + 1)
            aos[0] = _rand(state); aos[1] = _rand(state); aos[2] = 0.0
            acc_r = acc_g = acc_b = acc_a = 0.0
            for s in range(samples):
                jx = _rand(state) if samples > 1 else 0.5
                jy = _rand(state) if samples > 1 else 0.5
                u = ((px + jx) / width * 2.0 - 1.0) * half_w
                v = (1.0 - (py + jy) / height * 2.0) * half_h
                for k in range(3):
                    if ortho:
                        o[k] = eye[k] + u * right[k] + v * up[k]
                        d[k] = fwd[k]
                    else:
                        o[k] = eye[k]
                        d[k] = fwd[k] + u * right[k] + v * up[k]
                nd = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                for k in range(3):
                    d[k] /= nd
                trans = 1.0  # light still passing through see-through triangles in front
                for layer in range(16):
                    t, code = _trace(o, d, 1e30, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b, cyl_r,
                                     tri_a, tri_b, tri_c, tri_alpha, False)
                    if code < 0:
                        acc_r += trans * bg[0]; acc_g += trans * bg[1]; acc_b += trans * bg[2]
                        acc_a += trans * bg[3]
                        break
                    kind = code >> 40
                    i = code & ((1 << 40) - 1)
                    for k in range(3):
                        p[k] = o[k] + t * d[k]
                    alpha = 1.0
                    if kind == 0:
                        for k in range(3):
                            n[k] = (p[k] - sph_c[i, k]) / sph_r[i]
                        cr, cg, cb = sph_rgb[i, 0], sph_rgb[i, 1], sph_rgb[i, 2]
                    elif kind == 1:
                        ax = cyl_b[i, 0] - cyl_a[i, 0]; ay = cyl_b[i, 1] - cyl_a[i, 1]
                        az = cyl_b[i, 2] - cyl_a[i, 2]
                        L2 = ax * ax + ay * ay + az * az
                        h = ((p[0] - cyl_a[i, 0]) * ax + (p[1] - cyl_a[i, 1]) * ay + (p[2] - cyl_a[i, 2]) * az) / L2
                        for k, a_k in ((0, ax), (1, ay), (2, az)):
                            n[k] = p[k] - (cyl_a[i, k] + h * a_k)
                        nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                        for k in range(3):
                            n[k] /= nn
                        cr, cg, cb = cyl_rgb[i, 0], cyl_rgb[i, 1], cyl_rgb[i, 2]
                    else:
                        # barycentric weights of the hit point, then interpolate normal and colour
                        e1x = tri_b[i, 0] - tri_a[i, 0]; e1y = tri_b[i, 1] - tri_a[i, 1]; e1z = tri_b[i, 2] - tri_a[i, 2]
                        e2x = tri_c[i, 0] - tri_a[i, 0]; e2y = tri_c[i, 1] - tri_a[i, 1]; e2z = tri_c[i, 2] - tri_a[i, 2]
                        qx = p[0] - tri_a[i, 0]; qy = p[1] - tri_a[i, 1]; qz = p[2] - tri_a[i, 2]
                        d11 = e1x * e1x + e1y * e1y + e1z * e1z; d12 = e1x * e2x + e1y * e2y + e1z * e2z
                        d22 = e2x * e2x + e2y * e2y + e2z * e2z
                        dq1 = qx * e1x + qy * e1y + qz * e1z; dq2 = qx * e2x + qy * e2y + qz * e2z
                        den = d11 * d22 - d12 * d12
                        wb = (d22 * dq1 - d12 * dq2) / den if den > 0 else 0.0
                        wc = (d11 * dq2 - d12 * dq1) / den if den > 0 else 0.0
                        wa = 1.0 - wb - wc
                        for k in range(3):
                            n[k] = wa * tri_n[i, 0, k] + wb * tri_n[i, 1, k] + wc * tri_n[i, 2, k]
                        nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                        if nn < 1e-12:  # degenerate normals: use the face normal
                            n[0] = e1y * e2z - e1z * e2y; n[1] = e1z * e2x - e1x * e2z; n[2] = e1x * e2y - e1y * e2x
                            nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                        for k in range(3):
                            n[k] /= nn
                        cr = wa * tri_vrgb[i, 0, 0] + wb * tri_vrgb[i, 1, 0] + wc * tri_vrgb[i, 2, 0]
                        cg = wa * tri_vrgb[i, 0, 1] + wb * tri_vrgb[i, 1, 1] + wc * tri_vrgb[i, 2, 1]
                        cb = wa * tri_vrgb[i, 0, 2] + wb * tri_vrgb[i, 1, 2] + wc * tri_vrgb[i, 2, 2]
                        alpha = tri_alpha[i]
                    if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0:
                        for k in range(3):
                            n[k] = -n[k]
                    eps = 1e-4 * (1.0 + abs(t))
                    for k in range(3):
                        p[k] += n[k] * eps
                    sr, sg, sb = _shade(p, n, d, cr, cg, cb, light, ambient, diffuse, specular, shininess, shadows,
                                        ao_samples, ao_dist, state, od, lo, size, dims, starts, items, sph_c, sph_r,
                                        cyl_a, cyl_b, cyl_r, tri_a, tri_b, tri_c, tri_alpha, aos)
                    w = trans * alpha
                    acc_r += w * sr; acc_g += w * sg; acc_b += w * sb; acc_a += w
                    trans *= 1.0 - alpha
                    if trans < 1e-3:
                        break
                    for k in range(3):  # continue behind the see-through surface
                        o[k] = p[k] - 2.0 * n[k] * eps
            img[py, px, 0] = acc_r / samples
            img[py, px, 1] = acc_g / samples
            img[py, px, 2] = acc_b / samples
            img[py, px, 3] = acc_a / samples
    return img


# ----------------------------------------------------------------------------- public API
def raytrace(scene: Scene, camera: dict, size=(1600, 1200), *, samples: int = 8, ao_samples: int = 16,
             ao_distance: float | None = None, shadows: bool = True, background=(1.0, 1.0, 1.0),
             transparent_background: bool = False, ambient: float = 0.55, diffuse: float = 0.6,
             specular: float = 0.25, shininess: float = 40.0, light_direction=(-0.4, 0.6, 1.0), seed: int = 1):
    """Ray-trace ``scene`` seen through ``camera`` (see :func:`ewald.render.camera_state`).

    Returns an (H, W, 4) float RGBA image in [0, 1]. ``light_direction`` is given
    in camera space (x right, y up, z toward the viewer).
    """
    from .render import _rgb

    width, height = int(size[0]), int(size[1])
    eye = np.asarray(camera["position"], float)
    focal = np.asarray(camera["focal_point"], float)
    fwd = focal - eye
    dist = np.linalg.norm(fwd)
    fwd /= dist
    up = np.asarray(camera["up"], float)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    aspect = width / height
    ortho = bool(camera.get("parallel_projection", True))
    if ortho:
        half_h = float(camera["parallel_scale"])
        # start rays well behind the scene so nothing is clipped
        lo, hi = scene.bounds()
        eye = eye - fwd * (np.linalg.norm(hi - lo) + dist)
    else:
        half_h = math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2)
    half_w = half_h * aspect
    ld = np.asarray(light_direction, float)
    light = ld[0] * right + ld[1] * up - ld[2] * fwd
    light /= np.linalg.norm(light)
    if ao_distance is None:
        r = np.concatenate([scene.sph_r, scene.cyl_r])
        ao_distance = 4.0 * float(np.median(r)) if len(r) else 1.0
    bg = np.zeros(4) if transparent_background else np.append(_rgb(background), 1.0)
    if not (len(scene.sph_r) or len(scene.cyl_r) or len(scene.tri_alpha)):
        return np.broadcast_to(bg, (height, width, 4)).copy()
    lo, gsize, dims, starts, items = build_grid(scene)
    return _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, int(samples), int(ao_samples),
                   float(ao_distance), light, ambient, diffuse, specular, shininess, bool(shadows), bg,
                   lo, gsize, dims, starts, items, scene.sph_c, scene.sph_r, scene.sph_rgb,
                   scene.cyl_a, scene.cyl_b, scene.cyl_r, scene.cyl_rgb,
                   scene.tri_a, scene.tri_b, scene.tri_c, np.ascontiguousarray(scene.tri_n),
                   np.ascontiguousarray(scene.tri_vrgb), scene.tri_alpha, int(seed), _row_order(height))


def _row_order(height: int, stride: int = 64):
    """Rows interleaved (0, 64, 128, ..., 1, 65, ...): prange hands each thread one contiguous
    block of the loop, and with plain row order the threads that get the empty margins finish
    long before the ones with the structure. Pixels do not depend on the order."""
    return np.concatenate([np.arange(k, height, stride) for k in range(min(stride, height))]).astype(np.int64)
