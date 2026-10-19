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

SPHERE, CYLINDER = 0, 1


# ----------------------------------------------------------------------------- scene
class Scene:
    """Primitives in world coordinates (Angstrom)."""

    def __init__(self):
        self.sph_c = np.zeros((0, 3)); self.sph_r = np.zeros(0); self.sph_rgb = np.zeros((0, 3))
        self.cyl_a = np.zeros((0, 3)); self.cyl_b = np.zeros((0, 3)); self.cyl_r = np.zeros(0)
        self.cyl_rgb = np.zeros((0, 3))

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

    def bounds(self):
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        if len(self.sph_c):
            lo = np.minimum(lo, (self.sph_c - self.sph_r[:, None]).min(0))
            hi = np.maximum(hi, (self.sph_c + self.sph_r[:, None]).max(0))
        if len(self.cyl_a):
            for p in (self.cyl_a, self.cyl_b):
                lo = np.minimum(lo, (p - self.cyl_r[:, None]).min(0))
                hi = np.maximum(hi, (p + self.cyl_r[:, None]).max(0))
        return lo, hi


def scene_from_data(data, *, radius_scale=1.0, type_colors=None, type_radii=None, bonds=None, bond_pairs=None,
                    bond_radius=0.15, show_cell=True, cell_color=(0.0, 0.0, 0.0), cell_radius=None, **_ignored):
    """Build a Scene with the same colors/radii/bonds as :func:`ewald.render.build_scene`."""
    from .render import _rgb, bond_list, default_colors, default_radii

    sc = Scene()
    if data.count:
        colors = np.clip(default_colors(data, type_colors), 0, 1)
        radii = default_radii(data, type_radii=type_radii) * radius_scale
        keep = np.asarray(data["Transparency"]) < 1 if "Transparency" in data else np.ones(data.count, bool)
        sc.add_spheres(data.positions[keep], radii[keep], colors[keep])
        if bonds or bond_pairs:
            i, j, d = bond_list(data, bonds, bond_pairs)
            if len(i):
                start = data.positions[i]
                sc.add_cylinders(start, start + 0.5 * d, bond_radius, colors[i])
                sc.add_spheres(start + 0.5 * d, bond_radius, colors[i])  # close the open tube end
    if show_cell and data.cell.is_valid:
        o, (a, b, c) = data.cell.origin, data.cell.matrix
        corners = np.array([o, o + a, o + b, o + c, o + a + b, o + a + c, o + b + c, o + a + b + c])
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        r = cell_radius or 0.0015 * float(np.linalg.norm(a + b + c))
        sc.add_cylinders(corners[[e[0] for e in edges]], corners[[e[1] for e in edges]], r, _rgb(cell_color))
        # round joints so corners look closed
        sc.add_spheres(corners, r, _rgb(cell_color))
    return sc


# ----------------------------------------------------------------------------- grid
def build_grid(scene: Scene, target_per_cell: float = 2.0):
    lo, hi = scene.bounds()
    pad = 1e-3 * max(float(np.max(hi - lo)), 1.0)
    lo, hi = lo - pad, hi + pad
    n_obj = len(scene.sph_r) + len(scene.cyl_r)
    extent = hi - lo
    vol = float(np.prod(np.maximum(extent, 1e-6)))
    cell = (vol * target_per_cell / max(n_obj, 1)) ** (1 / 3)
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


@njit(cache=True)
def _trace(o, d, tmax, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b, cyl_r, any_hit):
    """3D-DDA through the grid. Returns (t, code) of the nearest hit (code -1: miss)."""
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
    cell = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    tnext = np.empty(3)
    tdelta = np.empty(3)
    for k in range(3):
        p = o[k] + d[k] * (t0 + 1e-9)
        c = int((p - lo[k]) / size[k])
        cell[k] = min(max(c, 0), dims[k] - 1)
        if d[k] > 0:
            step[k] = 1
            tnext[k] = (lo[k] + (cell[k] + 1) * size[k] - o[k]) / d[k]
            tdelta[k] = size[k] / d[k]
        elif d[k] < 0:
            step[k] = -1
            tnext[k] = (lo[k] + cell[k] * size[k] - o[k]) / d[k]
            tdelta[k] = -size[k] / d[k]
        else:
            step[k] = 0
            tnext[k] = 1e300
            tdelta[k] = 1e300
    best_t, best = 1e300, -1
    while True:
        c = (cell[0] * dims[1] + cell[1]) * dims[2] + cell[2]
        for s in range(starts[c], starts[c + 1]):
            code = items[s]
            kind = code >> 40
            i = code & ((1 << 40) - 1)
            if kind == 0:
                t = _hit_sphere(o, d, sph_c[i], sph_r[i])
            else:
                t = _hit_cylinder(o, d, cyl_a[i], cyl_b[i], cyl_r[i])
            if t > 0 and t < best_t and t < tmax:
                best_t, best = t, code
                if any_hit:
                    return best_t, best
        tcell = min(tnext[0], min(tnext[1], tnext[2]))
        if best >= 0 and best_t <= tcell:
            return best_t, best
        if tcell > t1:
            return (best_t, best) if best >= 0 else (-1.0, -1)
        k = 0
        if tnext[1] < tnext[k]:
            k = 1
        if tnext[2] < tnext[k]:
            k = 2
        cell[k] += step[k]
        if cell[k] < 0 or cell[k] >= dims[k]:
            return (best_t, best) if best >= 0 else (-1.0, -1)
        tnext[k] += tdelta[k]


@njit(cache=True, inline="always")
def _rand(state):
    # xorshift64*, per-pixel state array of length 1
    x = state[0]
    x ^= (x >> np.uint64(12)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27))
    state[0] = x
    return ((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, parallel=True)
def _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, samples, ao_samples, ao_dist,
            light, ambient, diffuse, specular, shininess, shadows, bg,
            lo, size, dims, starts, items, sph_c, sph_r, sph_rgb, cyl_a, cyl_b, cyl_r, cyl_rgb, seed):
    img = np.zeros((height, width, 4))
    for py in prange(height):
        state = np.empty(1, np.uint64)
        o = np.empty(3); d = np.empty(3); n = np.empty(3); p = np.empty(3); od = np.empty(3)
        for px in range(width):
            state[0] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(py * width + px + 1)
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
                t, code = _trace(o, d, 1e30, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b, cyl_r, False)
                if code < 0:
                    acc_r += bg[0]; acc_g += bg[1]; acc_b += bg[2]; acc_a += bg[3]
                    continue
                kind = code >> 40
                i = code & ((1 << 40) - 1)
                for k in range(3):
                    p[k] = o[k] + t * d[k]
                if kind == 0:
                    for k in range(3):
                        n[k] = (p[k] - sph_c[i, k]) / sph_r[i]
                    cr, cg, cb = sph_rgb[i, 0], sph_rgb[i, 1], sph_rgb[i, 2]
                else:
                    ax = cyl_b[i, 0] - cyl_a[i, 0]; ay = cyl_b[i, 1] - cyl_a[i, 1]; az = cyl_b[i, 2] - cyl_a[i, 2]
                    L2 = ax * ax + ay * ay + az * az
                    h = ((p[0] - cyl_a[i, 0]) * ax + (p[1] - cyl_a[i, 1]) * ay + (p[2] - cyl_a[i, 2]) * az) / L2
                    for k, a_k in ((0, ax), (1, ay), (2, az)):
                        n[k] = p[k] - (cyl_a[i, k] + h * a_k)
                    nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                    for k in range(3):
                        n[k] /= nn
                    cr, cg, cb = cyl_rgb[i, 0], cyl_rgb[i, 1], cyl_rgb[i, 2]
                if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0:
                    for k in range(3):
                        n[k] = -n[k]
                eps = 1e-4 * (1.0 + abs(t))
                for k in range(3):
                    p[k] += n[k] * eps
                # key light with shadow
                ndl = n[0] * light[0] + n[1] * light[1] + n[2] * light[2]
                lit = 0.0
                spec = 0.0
                if ndl > 0:
                    vis = 1.0
                    if shadows:
                        ts, cs = _trace(p, light, 1e30, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b,
                                        cyl_r, True)
                        if cs >= 0:
                            vis = 0.0
                    lit = ndl * vis
                    hx = light[0] - d[0]; hy = light[1] - d[1]; hz = light[2] - d[2]
                    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
                    ndh = (n[0] * hx + n[1] * hy + n[2] * hz) / hn
                    if ndh > 0:
                        spec = vis * ndh ** shininess
                # ambient occlusion: cosine-weighted hemisphere
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
                        r1 = _rand(state); r2 = _rand(state)
                        phi = 2 * math.pi * r1
                        sr = math.sqrt(r2)
                        x = math.cos(phi) * sr; y = math.sin(phi) * sr; z = math.sqrt(max(0.0, 1 - r2))
                        od[0] = x * bx + y * cx + z * n[0]
                        od[1] = x * by + y * cy + z * n[1]
                        od[2] = x * bz + y * cz + z * n[2]
                        ta, ca = _trace(p, od, ao_dist, lo, size, dims, starts, items, sph_c, sph_r, cyl_a, cyl_b,
                                        cyl_r, True)
                        if ca >= 0:
                            hits += 1
                    ao = 1.0 - hits / ao_samples
                shade = ambient * ao + diffuse * lit
                acc_r += min(1.0, cr * shade + specular * spec)
                acc_g += min(1.0, cg * shade + specular * spec)
                acc_b += min(1.0, cb * shade + specular * spec)
                acc_a += 1.0
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
    if not (len(scene.sph_r) or len(scene.cyl_r)):
        return np.broadcast_to(bg, (height, width, 4)).copy()
    lo, gsize, dims, starts, items = build_grid(scene)
    return _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, int(samples), int(ao_samples),
                   float(ao_distance), light, ambient, diffuse, specular, shininess, bool(shadows), bg,
                   lo, gsize, dims, starts, items, scene.sph_c, scene.sph_r, scene.sph_rgb,
                   scene.cyl_a, scene.cyl_b, scene.cyl_r, scene.cyl_rgb, int(seed))
