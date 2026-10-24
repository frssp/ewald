"""CPU ray tracer for publication figures: spheres + cylinders, ambient occlusion, soft shadows.

Plays the role of OVITO's Tachyon renderer. Pure numba (no OSPRay needed): a
bounding volume hierarchy (binned SAH) accelerates ray casts, each pixel takes up to
``samples`` jittered rays (adaptive sampling stops converged pixels early), and every hit is shaded with a key light (shadow ray), hemisphere ambient
occlusion (``ao_samples`` rays limited to ``ao_distance``) and a Blinn-Phong
highlight. Scenes come from the same data the OpenGL renderer uses, so colors,
radii, bonds and the camera match the viewport.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SPHERE, CYLINDER, TRIANGLE = 0, 1, 2
# grid and BVH item code: kind << 40 | NO_SHADOW (objects shadow and AO rays pass through) | index
NO_SHADOW = 1 << 39
INDEX_MASK = NO_SHADOW - 1
PROGRESS_STEPS = 20          # row batches when raytrace() reports progress


class RenderCancelled(Exception):
    """Raised by :func:`raytrace` when its ``progress`` callback returns ``False``."""


# ----------------------------------------------------------------------------- scene
class Scene:
    """Primitives in world coordinates (Angstrom)."""

    def __init__(self):
        self.sph_c = np.zeros((0, 3)); self.sph_r = np.zeros(0); self.sph_rgb = np.zeros((0, 3))
        self.cyl_a = np.zeros((0, 3)); self.cyl_b = np.zeros((0, 3)); self.cyl_r = np.zeros(0)
        self.cyl_rgb = np.zeros((0, 3))
        self.cyl_cap = np.zeros((0, 2))   # 1: a flat disk closes the start / end (round caps are spheres)
        self.tri_a = np.zeros((0, 3)); self.tri_b = np.zeros((0, 3)); self.tri_c = np.zeros((0, 3))
        self.tri_rgb = np.zeros((0, 3)); self.tri_alpha = np.zeros(0)
        # per-vertex normals and colours (M x 3 x 3), for smooth isosurfaces and colour-mapped slices
        self.tri_n = np.zeros((0, 3, 3)); self.tri_vrgb = np.zeros((0, 3, 3))
        # False: seen by camera rays but skipped by shadow and AO rays (cell edges)
        self.sph_cast = np.zeros(0, bool); self.cyl_cast = np.zeros(0, bool); self.tri_cast = np.zeros(0, bool)
        # True: part of a surface mesh (isosurface, slice, polyhedra), traced through the BVH when
        # large; False: tessellated solids (glass spheres, arrow heads) that fill the volume
        self.tri_surface = np.zeros(0, bool)
        self.legends = []   # color bars of the elements (ewald.scene.Scene.legends), drawn as overlays

    def add_spheres(self, centers, radii, colors, cast_shadows=True):
        self.sph_c = np.vstack([self.sph_c, np.asarray(centers, float).reshape(-1, 3)])
        self.sph_r = np.concatenate([self.sph_r, np.broadcast_to(np.asarray(radii, float), (len(centers),))])
        self.sph_rgb = np.vstack([self.sph_rgb, np.broadcast_to(np.asarray(colors, float), (len(centers), 3))])
        self.sph_cast = np.concatenate([self.sph_cast, np.full(len(centers), bool(cast_shadows))])

    def add_cylinders(self, a, b, radii, colors, cast_shadows=True, flat_caps=(False, False)):
        """Open cylinders from ``a`` to ``b``; ``flat_caps`` (start, end), one pair or one per
        cylinder, closes those ends with disks."""
        a = np.asarray(a, float).reshape(-1, 3)
        self.cyl_cap = np.vstack([self.cyl_cap, np.broadcast_to(np.asarray(flat_caps, float), (len(a), 2))])
        self.cyl_a = np.vstack([self.cyl_a, a])
        self.cyl_b = np.vstack([self.cyl_b, np.asarray(b, float).reshape(-1, 3)])
        self.cyl_r = np.concatenate([self.cyl_r, np.broadcast_to(np.asarray(radii, float), (len(a),))])
        self.cyl_rgb = np.vstack([self.cyl_rgb, np.broadcast_to(np.asarray(colors, float), (len(a), 3))])
        self.cyl_cast = np.concatenate([self.cyl_cast, np.full(len(a), bool(cast_shadows))])

    def add_triangles(self, a, b, c, colors, alpha=1.0, normals=None, vertex_colors=None, cast_shadows=True,
                      surface=False):
        """Two-sided triangles; ``alpha`` < 1 makes them see-through (no shadows, no AO).
        ``normals``/``vertex_colors`` (M x 3 x 3, one row per corner) are interpolated across
        each triangle; without them it is flat-shaded in its face colour. ``surface``: the
        triangles form a surface mesh (see :class:`Accel`)."""
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
        self.tri_cast = np.concatenate([self.tri_cast, np.full(len(a), bool(cast_shadows))])
        self.tri_surface = np.concatenate([self.tri_surface, np.full(len(a), bool(surface))])

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
                    arrow_resolution=24, view_direction=None, trajectory_width=2.0, trajectory_color_by=None,
                    trajectory_colormap="viridis", trajectory_color_range=None, trajectory_time_range=None,
                    trajectory_trail=None, trajectory_radius=None, visuals=None, frame=0, boundary=None, boundary_bonds="half", replicate=None, replicate_cell="super",
                    show_atoms=True, surfaces=None, type_transparency=None, selection_color=None, measurements=None,
                    pixel=None, flat=False, **_ignored):
    """Build a Scene with the same colors/radii/bonds/arrows as :func:`ewald.render.build_scene`.

    ``view_direction`` (scene -> camera) lets 'flat' arrows face the viewer evenly lit.
    ``pixel``: ``(size of a pixel in Angstrom, image height)`` sizes lines as in the other renderers.
    ``flat``: the flat (ASE) style's outlines (see :func:`scene_from_ir`).
    ``visuals``: extra visual elements (plugins), see :mod:`ewald.scene`.
    """
    from . import scene as ir

    scene = ir.build(data, ctx=ir.SceneContext(frame=frame, view_direction=view_direction), visuals=visuals,
                     radius_scale=radius_scale, type_colors=type_colors, type_radii=type_radii,
                     radius_preset=radius_preset, oxidation_states=oxidation_states, coordination=coordination,
                     bonds=bonds, bond_pairs=bond_pairs, bond_radius=bond_radius, vectors=vectors,
                     trajectory_width=trajectory_width, trajectory_color_by=trajectory_color_by,
                     trajectory_colormap=trajectory_colormap, trajectory_color_range=trajectory_color_range,
                     trajectory_time_range=trajectory_time_range, trajectory_trail=trajectory_trail,
                     trajectory_radius=trajectory_radius, show_cell=show_cell, cell_radius=cell_radius,
                     boundary=boundary, boundary_bonds=boundary_bonds, replicate=replicate,
                     replicate_cell=replicate_cell, show_atoms=show_atoms, surfaces=surfaces,
                     type_transparency=type_transparency, selection_color=selection_color,
                     measurements=measurements)
    return scene_from_ir(scene, arrow_resolution=arrow_resolution, view_direction=view_direction, ink=cell_color,
                         pixel=pixel, flat=flat)


def scene_from_ir(scene, *, arrow_resolution=24, view_direction=None, ink=(0.0, 0.0, 0.0), pixel=None,
                  flat=False) -> Scene:
    """Ray-tracer primitives for every layer of an :class:`ewald.scene.Scene`. The ray tracer is
    opaque except for triangles: sphere and cylinder ``alpha`` is ignored. Lines are tubes of their
    ``radius``, else as wide as :func:`ewald.scene.line_pixels` with ``pixel`` = ``(size of a pixel in
    Angstrom, image height)``; without either they are not drawn.

    ``flat`` (with ``view_direction``): the flat (ASE) style's dark outlines, as the OpenGL renderer draws
    them (render it unlit): a dark rim inside each atom, ``max(0.8 pixel, 5 %)`` of its radius, and one
    around each stick, 30 % of its radius. Both are dark copies pushed back along the view, so only the
    rim shows past the colored primitive (the OpenGL inverted hull)."""
    from .render import FLAT_OUTLINE, STICK_OUTLINE, _rgb
    from .scene import line_pixels

    toward = None
    if flat and view_direction is not None:
        toward = np.asarray(view_direction, float) / np.linalg.norm(view_direction)
    dark = np.array([FLAT_OUTLINE] * 3)

    def outline(centers, radii, start=None, end=None):
        """Dark spheres (``centers``) or cylinders (``start``, ``end``) of ``radii``, pushed back by their radius."""
        back = np.asarray(radii, float)[:, None] * toward
        if start is None:
            sc.add_spheres(centers - back, radii, dark, False)
        else:
            sc.add_cylinders(start - back, end - back, radii, dark, False)

    sc = Scene()
    sc.legends = list(getattr(scene, "legends", []))
    for layer in scene.layers:
        prim = layer.primitive
        if layer.kind == "spheres":
            if prim.count:
                c, r, rgb, a = prim.centers, prim.sizes(), prim.colors(), prim.alpha
                if toward is not None:   # the rim: a dark sphere behind, the colored one that much smaller
                    rim = np.maximum(0.05 * r, 0.8 * pixel[0] if pixel is not None else 0.0)
                    outline(c, r)
                    r = np.maximum(r - rim, 0.5 * r)
                solid = a >= 1
                if solid.all():
                    sc.add_spheres(c, r, rgb)
                else:  # only triangles can be see-through: tessellate those spheres
                    if solid.any():
                        sc.add_spheres(c[solid], r[solid], rgb[solid])
                    _add_glass_spheres(sc, c[~solid], r[~solid], rgb[~solid], a[~solid])
        elif layer.kind == "cylinders":
            if prim.count:
                sc.add_cylinders(prim.start, prim.end, prim.radius, prim.rgb,
                                 flat_caps=np.column_stack([prim.start_cap == 1, prim.end_cap == 1]))
                center, _, radius, rgb, _, _ = prim.caps("round")
                if len(center):  # a sphere of the tube radius shows as a hemisphere past the end
                    sc.add_spheres(center, radius, rgb)
                if toward is not None:
                    wide = (1.0 + STICK_OUTLINE) * np.broadcast_to(prim.radius, (prim.count,))
                    outline(None, wide, prim.start, prim.end)
                    if len(center):
                        outline(center, (1.0 + STICK_OUTLINE) * np.asarray(radius, float))
        elif layer.kind == "mesh":
            mesh = prim
            if mesh.face_count:
                v, f = mesh.vertices, mesh.faces
                vn, vc = mesh.vertex_properties.get("Normal"), mesh.vertex_properties.get("Color")
                sc.add_triangles(v[f[:, 0]], v[f[:, 1]], v[f[:, 2]], np.clip(mesh.face_colors(), 0, 1),
                                 1.0 - mesh.transparency, None if vn is None else np.asarray(vn, float)[f],
                                 None if vc is None else np.clip(np.asarray(vc, float), 0, 1)[f], surface=True)
                if mesh.edges is not None and len(mesh.edges):
                    e = mesh.edges
                    sc.add_cylinders(v[e[:, 0]], v[e[:, 1]], 0.5 * mesh.edge_width, _rgb(mesh.edge_color))
        elif layer.kind == "arrows":
            _add_arrows(sc, prim, arrow_resolution, view_direction)
        elif layer.kind == "lines":
            radius = prim.radius
            if radius is None and pixel is not None:
                radius = 0.5 * line_pixels(prim.width, pixel[1]) * pixel[0]
            if radius is None or not prim.count:
                continue
            seg = prim.segments()
            if not len(seg):
                continue
            if prim.rgb is None and prim.point_rgb is None:
                seg_rgb = _rgb(ink)
            else:
                seg_rgb = prim.segment_colors()
            cast = prim.cast_shadows
            sc.add_cylinders(prim.points[seg[:, 0]], prim.points[seg[:, 1]], radius, seg_rgb, cast)
            # round joints so corners look closed
            joints = np.unique(seg)
            if prim.rgb is None and prim.point_rgb is None:
                sc.add_spheres(prim.points[joints], radius, seg_rgb, cast)
            elif prim.point_rgb is not None:
                sc.add_spheres(prim.points[joints], radius, np.asarray(prim.point_rgb, float)[joints], cast)
            else:
                rgb = np.empty((len(prim.points), 3))
                rgb[seg[:, 1]] = seg_rgb
                rgb[seg[:, 0]] = seg_rgb
                sc.add_spheres(prim.points[joints], radius, rgb[joints], cast)
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


# ----------------------------------------------------------------------------- acceleration
# Atoms, bonds, cell edges, tessellated solids (glass spheres, arrow heads) and coarse meshes
# (polyhedra) fill the volume evenly and go into a uniform grid: the short AO and shadow rays
# from an atom cross only a few cells there, and a BVH walk from the root costs more (1.5-2x on
# 108k fcc Cu, 1.2x on 1728 TiO6 octahedra). Fine surface meshes (isosurfaces, slices: triangles
# much smaller than the grid cells the atoms need) go into a BVH from BVH_MIN_TRIANGLES on:
# in the grid they force cells as small as their triangles, and are 1.4-2.2x slower to trace.
BVH_MIN_TRIANGLES = 1024
LEAF_SIZE = 4                # a node with this many primitives or fewer is always a leaf
MAX_LEAF = 16                # leaves up to this size when the SAH finds no better split
SAH_BINS = 16


class Accel:
    """Acceleration structures of a :class:`Scene` for :func:`raytrace`.

    Grid (spheres, cylinders, triangles of solids and of small meshes): ``grid`` (origin, cell
    size, cell counts) and CSR cell lists ``starts``/``items``. BVH (surface meshes of at least
    ``BVH_MIN_TRIANGLES`` triangles, binned SAH): ``nodes`` (N x 8: box lo, box hi, then the left child's index for an interior node, the right one follows it, or the leaf's first
    slot in ``bvh_items``; then the leaf's primitive count, 0 for interior nodes) and
    ``bvh_items``; ``depth`` sizes the traversal stack. Item codes are
    ``kind << 40 | NO_SHADOW | index`` into ``sph``/``cyl``/``tri``, the primitives one row each.
    """

    def __init__(self, scene: Scene):
        sheet = _fine_surface(scene)
        lo, size, dims, self.starts, self.items = build_grid(scene, triangles=~sheet)
        self.grid = np.concatenate([lo, size, dims]).astype(float)
        self.nodes, self.bvh_items, self.depth = build_bvh(scene, triangles=sheet)
        # one row per primitive for the ray casts (fewer arrays per call, no row views)
        self.sph = np.ascontiguousarray(np.column_stack([scene.sph_c, scene.sph_r]))
        self.cyl = np.ascontiguousarray(np.column_stack([scene.cyl_a, scene.cyl_b, scene.cyl_r, scene.cyl_cap]))
        self.tri = np.ascontiguousarray(np.column_stack([scene.tri_a, scene.tri_b, scene.tri_c, scene.tri_alpha]))


def _fine_surface(scene: Scene):
    """Mask of the surface-mesh triangles that go into the BVH: all of them when there are at
    least ``BVH_MIN_TRIANGLES`` and their median edge is under half the grid cell the other
    primitives need (where :func:`build_grid` would shrink its cells for them), else none."""
    surf = scene.tri_surface
    if int(surf.sum()) < BVH_MIN_TRIANGLES:
        return np.zeros(len(surf), bool)
    lo, hi = scene.bounds()
    vol = float(np.prod(np.maximum(hi - lo, 1e-6)))
    n_volume = len(scene.sph_r) + len(scene.cyl_r) + int((~surf).sum())
    cell = (vol * 2.0 / max(n_volume, 1)) ** (1 / 3)
    edge = float(np.median(np.linalg.norm(scene.tri_b[surf] - scene.tri_a[surf], axis=1)))
    return surf if 2.0 * edge < cell else np.zeros(len(surf), bool)


def primitive_boxes(scene: Scene, kinds=(SPHERE, CYLINDER, TRIANGLE), triangles=None):
    """Axis-aligned boxes (lo, hi) and item codes of the scene's primitives of ``kinds``
    (``triangles``: a mask to take only some of the triangles)."""
    boxes_lo, boxes_hi, codes = [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros(0, np.int64)]

    def add(kind, lo, hi, cast, index=None):
        index = np.arange(len(lo), dtype=np.int64) if index is None else index
        boxes_lo.append(lo[index]); boxes_hi.append(hi[index])
        codes.append(kind * (1 << 40) + np.where(cast[index], 0, NO_SHADOW) + index)

    if SPHERE in kinds and len(scene.sph_r):
        add(SPHERE, scene.sph_c - scene.sph_r[:, None], scene.sph_c + scene.sph_r[:, None], scene.sph_cast)
    if CYLINDER in kinds and len(scene.cyl_r):
        add(CYLINDER, np.minimum(scene.cyl_a, scene.cyl_b) - scene.cyl_r[:, None],
            np.maximum(scene.cyl_a, scene.cyl_b) + scene.cyl_r[:, None], scene.cyl_cast)
    if TRIANGLE in kinds and len(scene.tri_alpha):
        add(TRIANGLE, np.minimum(np.minimum(scene.tri_a, scene.tri_b), scene.tri_c),
            np.maximum(np.maximum(scene.tri_a, scene.tri_b), scene.tri_c), scene.tri_cast,
            None if triangles is None else np.flatnonzero(triangles))
    return np.vstack(boxes_lo), np.vstack(boxes_hi), np.concatenate(codes).astype(np.int64)


def build_bvh(scene: Scene, kinds=(TRIANGLE,), triangles=None):
    """BVH over the primitives of ``kinds`` (``triangles``: mask of the triangles to take):
    (nodes, items, depth), see :class:`Accel`."""
    blo, bhi, codes = primitive_boxes(scene, kinds, triangles)
    if not len(codes):
        return np.zeros((0, 8)), codes, 1
    # pad flat boxes (axis-aligned triangles) a little so they are never missed edge-on
    pad = 1e-9 * max(float(np.max(bhi.max(0) - blo.min(0))), 1.0)
    nodes, order, depth = _build_bvh(np.ascontiguousarray(blo - pad), np.ascontiguousarray(bhi + pad))
    return nodes, codes[order], depth


@njit(cache=True)
def _area(ex, ey, ez):
    return ex * ey + ey * ez + ez * ex


@njit(cache=True)
def _build_bvh(blo, bhi):
    """Top-down binned-SAH build (Wald 2007): every node tries ``SAH_BINS`` centroid bins on
    each axis and splits where ``1 + (A_l N_l + A_r N_r) / A`` is smallest; a node stays a
    leaf when that is not below its primitive count (or it has at most ``LEAF_SIZE``)."""
    n = len(blo)
    cen = 0.5 * (blo + bhi)
    idx = np.arange(n)
    cap = 2 * n
    box = np.empty((cap, 8))   # one 64-byte row per node: lo, hi, child or first item, count
    stack = np.empty((cap + 1, 4), np.int64)   # node, begin, end, depth
    stack[0, 0] = 0; stack[0, 1] = 0; stack[0, 2] = n; stack[0, 3] = 1
    sp = 1
    nodes = 1
    max_depth = 1
    bin_lo = np.empty((SAH_BINS, 3)); bin_hi = np.empty((SAH_BINS, 3)); bin_n = np.zeros(SAH_BINS, np.int64)
    right_area = np.empty(SAH_BINS); right_n = np.empty(SAH_BINS, np.int64)
    while sp > 0:
        sp -= 1
        node, b, e, depth = stack[sp, 0], stack[sp, 1], stack[sp, 2], stack[sp, 3]
        max_depth = max(max_depth, depth)
        lo0 = lo1 = lo2 = 1e300; hi0 = hi1 = hi2 = -1e300
        cl0 = cl1 = cl2 = 1e300; ch0 = ch1 = ch2 = -1e300
        for s in range(b, e):
            i = idx[s]
            lo0 = min(lo0, blo[i, 0]); lo1 = min(lo1, blo[i, 1]); lo2 = min(lo2, blo[i, 2])
            hi0 = max(hi0, bhi[i, 0]); hi1 = max(hi1, bhi[i, 1]); hi2 = max(hi2, bhi[i, 2])
            cl0 = min(cl0, cen[i, 0]); cl1 = min(cl1, cen[i, 1]); cl2 = min(cl2, cen[i, 2])
            ch0 = max(ch0, cen[i, 0]); ch1 = max(ch1, cen[i, 1]); ch2 = max(ch2, cen[i, 2])
        box[node, 0] = lo0; box[node, 1] = lo1; box[node, 2] = lo2
        box[node, 3] = hi0; box[node, 4] = hi1; box[node, 5] = hi2
        m = e - b
        if m <= LEAF_SIZE:
            box[node, 6] = b; box[node, 7] = m
            continue
        parent_area = _area(hi0 - lo0, hi1 - lo1, hi2 - lo2)
        best_cost, best_axis, best_bin = 1e300, -1, 0
        for axis in range(3):
            c_lo = cl0 if axis == 0 else (cl1 if axis == 1 else cl2)
            c_hi = ch0 if axis == 0 else (ch1 if axis == 1 else ch2)
            if c_hi - c_lo <= 1e-12 * (1.0 + abs(c_lo)):
                continue
            scale = SAH_BINS / (c_hi - c_lo)
            bin_n[:] = 0
            bin_lo[:] = 1e300; bin_hi[:] = -1e300
            for s in range(b, e):
                i = idx[s]
                k = min(int((cen[i, axis] - c_lo) * scale), SAH_BINS - 1)
                bin_n[k] += 1
                for q in range(3):
                    bin_lo[k, q] = min(bin_lo[k, q], blo[i, q]); bin_hi[k, q] = max(bin_hi[k, q], bhi[i, q])
            # sweep from the right: area and count of bins k..end
            r0 = r1 = r2 = 1e300; s0 = s1 = s2 = -1e300; rn = 0
            for k in range(SAH_BINS - 1, 0, -1):
                if bin_n[k]:
                    r0 = min(r0, bin_lo[k, 0]); r1 = min(r1, bin_lo[k, 1]); r2 = min(r2, bin_lo[k, 2])
                    s0 = max(s0, bin_hi[k, 0]); s1 = max(s1, bin_hi[k, 1]); s2 = max(s2, bin_hi[k, 2])
                    rn += bin_n[k]
                right_n[k] = rn
                right_area[k] = _area(s0 - r0, s1 - r1, s2 - r2) if rn else 0.0
            l0 = l1 = l2 = 1e300; t0 = t1 = t2 = -1e300; ln = 0
            for k in range(SAH_BINS - 1):  # split between bin k and k + 1
                if bin_n[k]:
                    l0 = min(l0, bin_lo[k, 0]); l1 = min(l1, bin_lo[k, 1]); l2 = min(l2, bin_lo[k, 2])
                    t0 = max(t0, bin_hi[k, 0]); t1 = max(t1, bin_hi[k, 1]); t2 = max(t2, bin_hi[k, 2])
                    ln += bin_n[k]
                if ln == 0 or right_n[k + 1] == 0:
                    continue
                cost = 1.0 + (_area(t0 - l0, t1 - l1, t2 - l2) * ln + right_area[k + 1] * right_n[k + 1]) \
                    / max(parent_area, 1e-300)
                if cost < best_cost:
                    best_cost, best_axis, best_bin = cost, axis, k
        if best_axis < 0:  # every centroid in one spot: split by index so leaves stay small
            if m <= MAX_LEAF:
                box[node, 6] = b; box[node, 7] = m
                continue
            mid = (b + e) // 2
        else:
            if best_cost >= m and m <= MAX_LEAF:
                box[node, 6] = b; box[node, 7] = m
                continue
            c_lo = cl0 if best_axis == 0 else (cl1 if best_axis == 1 else cl2)
            c_hi = ch0 if best_axis == 0 else (ch1 if best_axis == 1 else ch2)
            scale = SAH_BINS / (c_hi - c_lo)
            i0, j0 = b, e - 1
            while i0 <= j0:
                if min(int((cen[idx[i0], best_axis] - c_lo) * scale), SAH_BINS - 1) <= best_bin:
                    i0 += 1
                else:
                    tmp = idx[i0]; idx[i0] = idx[j0]; idx[j0] = tmp
                    j0 -= 1
            mid = i0
            if mid == b or mid == e:
                mid = (b + e) // 2
        box[node, 6] = nodes; box[node, 7] = 0
        stack[sp, 0] = nodes; stack[sp, 1] = b; stack[sp, 2] = mid; stack[sp, 3] = depth + 1
        stack[sp + 1, 0] = nodes + 1; stack[sp + 1, 1] = mid; stack[sp + 1, 2] = e; stack[sp + 1, 3] = depth + 1
        sp += 2
        nodes += 2
    return box[:nodes].copy(), idx, max_depth


def build_grid(scene: Scene, target_per_cell: float = 2.0, kinds=(SPHERE, CYLINDER, TRIANGLE), triangles=None):
    """Uniform grid over the primitives of ``kinds`` (``triangles``: mask of the triangles to
    take): (lo, cell size, dims, starts, items)."""
    blo, bhi, codes = primitive_boxes(scene, kinds, triangles)
    if not len(codes):
        return np.zeros(3), np.ones(3), np.ones(3, np.int64), np.zeros(2, np.int64), codes
    lo, hi = blo.min(0), bhi.max(0)
    pad = 1e-3 * max(float(np.max(hi - lo)), 1.0)
    lo, hi = lo - pad, hi + pad
    n_obj = len(codes)
    extent = hi - lo
    vol = float(np.prod(np.maximum(extent, 1e-6)))
    cell = (vol * target_per_cell / max(n_obj, 1)) ** (1 / 3)
    tri = np.ones(len(scene.tri_alpha), bool) if triangles is None else np.asarray(triangles, bool)
    if TRIANGLE in kinds and tri.any():  # surfaces fill a 2D sheet: size cells by the triangles
        edge = float(np.median(np.linalg.norm(scene.tri_b[tri] - scene.tri_a[tri], axis=1)))
        cell = min(cell, max(2.0 * edge, (vol / 2 ** 24) ** (1 / 3)))
    dims = np.clip(np.ceil(extent / cell).astype(np.int64), 1, 512)
    size = extent / dims
    c0 = np.clip(((blo - lo) / size).astype(np.int64), 0, dims - 1)
    c1 = np.clip(((bhi - lo) / size).astype(np.int64), 0, dims - 1)
    starts, items = _fill_grid(c0, c1, dims, codes)
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
def _hit_sphere(o, d, sph, i):
    """Sphere ``i`` of ``sph`` (rows: center, radius)."""
    ox, oy, oz = o[0] - sph[i, 0], o[1] - sph[i, 1], o[2] - sph[i, 2]
    r = sph[i, 3]
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
def _hit_cylinder(o, d, cyl, i):
    """Finite cylinder ``i`` of ``cyl`` (rows: start, end, radius, flat start cap, flat end cap):
    the side, plus a disk at each end whose flag is set."""
    ax, ay, az = cyl[i, 3] - cyl[i, 0], cyl[i, 4] - cyl[i, 1], cyl[i, 5] - cyl[i, 2]
    r = cyl[i, 6]
    L = math.sqrt(ax * ax + ay * ay + az * az)
    if L < 1e-12:
        return -1.0
    ax /= L; ay /= L; az /= L
    ox, oy, oz = o[0] - cyl[i, 0], o[1] - cyl[i, 1], o[2] - cyl[i, 2]
    dd = d[0] * ax + d[1] * ay + d[2] * az
    od = ox * ax + oy * ay + oz * az
    px, py, pz = d[0] - dd * ax, d[1] - dd * ay, d[2] - dd * az
    qx, qy, qz = ox - od * ax, oy - od * ay, oz - od * az
    best = -1.0
    if abs(dd) > 1e-12:  # flat caps: the planes h = 0 and h = L, inside the rim
        for col, h in ((7, 0.0), (8, L)):
            if cyl[i, col] > 0.5:
                t = (h - od) / dd
                if t > 1e-6 and (best < 0 or t < best):
                    wx, wy, wz = qx + t * px, qy + t * py, qz + t * pz
                    if wx * wx + wy * wy + wz * wz <= r * r:
                        best = t
    A = px * px + py * py + pz * pz
    if A < 1e-14:
        return best
    B = px * qx + py * qy + pz * qz
    C = qx * qx + qy * qy + qz * qz - r * r
    disc = B * B - A * C
    if disc < 0:
        return best
    s = math.sqrt(disc)
    for t in ((-B - s) / A, (-B + s) / A):
        if t > 1e-6:
            h = od + t * dd
            if 0.0 <= h <= L:
                return t if best < 0 or t < best else best
    return best


@njit(cache=True, inline="always")
def _hit_triangle(o, d, tri, i):
    """Moller-Trumbore, two-sided; triangle ``i`` of ``tri`` (rows: corners a, b, c, alpha)."""
    e1x, e1y, e1z = tri[i, 3] - tri[i, 0], tri[i, 4] - tri[i, 1], tri[i, 5] - tri[i, 2]
    e2x, e2y, e2z = tri[i, 6] - tri[i, 0], tri[i, 7] - tri[i, 1], tri[i, 8] - tri[i, 2]
    px, py, pz = d[1] * e2z - d[2] * e2y, d[2] * e2x - d[0] * e2z, d[0] * e2y - d[1] * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-14:
        return -1.0
    inv = 1.0 / det
    tx, ty, tz = o[0] - tri[i, 0], o[1] - tri[i, 1], o[2] - tri[i, 2]
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
def _hit_box(box, k, ox, oy, oz, ix, iy, iz, tmax):
    """Entry distance of the ray into node ``k``'s box, or -1 when it misses within (0, tmax)."""
    ta = (box[k, 0] - ox) * ix; tb = (box[k, 3] - ox) * ix
    tn = min(ta, tb); tf = max(ta, tb)
    ta = (box[k, 1] - oy) * iy; tb = (box[k, 4] - oy) * iy
    tn = max(tn, min(ta, tb)); tf = min(tf, max(ta, tb))
    ta = (box[k, 2] - oz) * iz; tb = (box[k, 5] - oz) * iz
    tn = max(tn, min(ta, tb)); tf = min(tf, max(ta, tb))
    tn = max(tn, 0.0)
    return tn if tn <= tf and tn < tmax else -1.0


@njit(cache=True, inline="always")
def _inv(x):
    # 1/x without infinities: (lo - o) * inv stays finite (no inf * 0 = nan in the slab test)
    if abs(x) < 1e-300:
        return 1e300
    return 1.0 / x


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


@njit(cache=True, inline="always")
def _trace_grid(o, d, tmax, grid, starts, items, sph, cyl, tri, any_hit):
    """3D-DDA through the grid. Returns (t, code) of the nearest hit
    (code -1: miss) closer than ``tmax``. The walk state lives in scalars: this runs for every primary, shadow and AO ray, and
    small per-call arrays cost a heap allocation each."""
    if len(items) == 0:
        return -1.0, -1
    # clip ray to grid box (grid: origin, cell size, cell counts)
    t0, t1 = 0.0, tmax
    for k in range(3):
        hi = grid[k] + grid[3 + k] * grid[6 + k]
        if abs(d[k]) < 1e-12:
            if o[k] < grid[k] or o[k] > hi:
                return -1.0, -1
        else:
            ta = (grid[k] - o[k]) / d[k]
            tb = (hi - o[k]) / d[k]
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
    if t0 > t1:
        return -1.0, -1
    dx, dy, dz = int(grid[6]), int(grid[7]), int(grid[8])
    cx, sx, nx, ex = _dda_axis(o[0], d[0], grid[0], grid[3], dx, t0)
    cy, sy, ny, ey = _dda_axis(o[1], d[1], grid[1], grid[4], dy, t0)
    cz, sz, nz, ez = _dda_axis(o[2], d[2], grid[2], grid[5], dz, t0)
    best_t, best = 1e300, -1
    while True:
        c = (cx * dy + cy) * dz + cz
        for s in range(starts[c], starts[c + 1]):
            code = items[s]
            if any_hit and code & NO_SHADOW:
                continue
            kind = code >> 40
            i = code & INDEX_MASK
            if kind == 0:
                t = _hit_sphere(o, d, sph, i)
            elif kind == 1:
                t = _hit_cylinder(o, d, cyl, i)
            elif any_hit and tri[i, 9] < 0.999:
                t = -1.0
            else:
                t = _hit_triangle(o, d, tri, i)
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


@njit(cache=True)
def _trace_bvh(o, d, tmax, box, items, stack, stack_t, sph, cyl, tri, any_hit):
    """Nearest hit closer than ``tmax`` through the BVH: returns (t, code), code -1 on a miss.
    Children are visited near first and skipped once a closer hit is known. Shadow/AO rays
    (``any_hit``) stop at the first opaque hit and pass through see-through triangles.
    ``stack``/``stack_t`` are scratch arrays of the caller (no allocation per ray)."""
    if len(box) == 0:
        return -1.0, -1
    ox, oy, oz = o[0], o[1], o[2]
    ix, iy, iz = _inv(d[0]), _inv(d[1]), _inv(d[2])
    best_t, best = tmax, -1
    t_root = _hit_box(box, 0, ox, oy, oz, ix, iy, iz, best_t)
    if t_root < 0:
        return -1.0, -1
    stack[0] = 0; stack_t[0] = t_root
    sp = 1
    while sp > 0:
        sp -= 1
        if stack_t[sp] >= best_t:
            continue
        node = stack[sp]
        n = int(box[node, 7])
        if n > 0:
            first = int(box[node, 6])
            for s in range(first, first + n):
                code = items[s]
                if any_hit and code & NO_SHADOW:
                    continue
                kind = code >> 40
                i = code & INDEX_MASK
                if kind == 0:
                    t = _hit_sphere(o, d, sph, i)
                elif kind == 1:
                    t = _hit_cylinder(o, d, cyl, i)
                elif any_hit and tri[i, 9] < 0.999:
                    t = -1.0
                else:
                    t = _hit_triangle(o, d, tri, i)
                if t > 0 and t < best_t:
                    best_t, best = t, code
                    if any_hit:
                        return best_t, best
        else:
            left = int(box[node, 6])
            tl = _hit_box(box, left, ox, oy, oz, ix, iy, iz, best_t)
            tr = _hit_box(box, left + 1, ox, oy, oz, ix, iy, iz, best_t)
            if tl >= 0 and tr >= 0:
                if tl <= tr:  # far child below, near child on top
                    stack[sp] = left + 1; stack_t[sp] = tr
                    stack[sp + 1] = left; stack_t[sp + 1] = tl
                else:
                    stack[sp] = left; stack_t[sp] = tl
                    stack[sp + 1] = left + 1; stack_t[sp + 1] = tr
                sp += 2
            elif tl >= 0:
                stack[sp] = left; stack_t[sp] = tl
                sp += 1
            elif tr >= 0:
                stack[sp] = left + 1; stack_t[sp] = tr
                sp += 1
    return (best_t, best) if best >= 0 else (-1.0, -1)


@njit(cache=True)
def _trace(o, d, tmax, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, any_hit):
    """Nearest hit of the grid and the BVH (see :class:`Accel`): (t, code), code -1 on a miss.
    The BVH walk only looks closer than the grid's hit."""
    tg, cg = -1.0, -1
    if len(items):
        tg, cg = _trace_grid(o, d, tmax, grid, starts, items, sph, cyl, tri, any_hit)
        if cg >= 0:
            if any_hit:
                return tg, cg
            tmax = tg
    if len(nodes):  # skipped without triangles: a call with this many arrays is not free
        tb, cb = _trace_bvh(o, d, tmax, nodes, bvh_items, stack, stack_t, sph, cyl, tri, any_hit)
        if cb >= 0:
            return tb, cb
    return tg, cg


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
           state, od, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, aos):
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
            ts, cs = _trace(p, light, 1e30, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, True)
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
            ta, ca = _trace(p, od, ao_dist, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, True)
            if ca >= 0:
                hits += 1
        ao = 1.0 - hits / ao_samples
    shade = ambient * ao + diffuse * lit
    return (min(1.0, cr * shade + specular * spec), min(1.0, cg * shade + specular * spec),
            min(1.0, cb * shade + specular * spec))


@njit(cache=True, parallel=True)
def _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, s1, samples, ao_samples, ao_dist,
            light, ambient, diffuse, specular, shininess, shadows, bg,
            grid, starts, items, nodes, bvh_items, depth, sph, cyl, tri, sph_c, sph_r, sph_rgb,
            cyl_a, cyl_b, cyl_rgb, tri_a, tri_b, tri_c, tri_n, tri_vrgb, tri_alpha, seed, rows, active,
            acc, sq, nsamp, rng, ao_next):
    """Bring every ``active`` pixel of ``rows`` from ``nsamp`` samples to ``s1``, adding them to
    ``acc`` (sums of RGBA) and ``sq`` (sums of squares, for the noise estimate). A pixel that
    has samples continues its random stream (``rng``) and AO sequence (``ao_next``) where the
    last pass left it, so its samples are the same however they are split into passes.
    ``sq``/``nsamp``/``rng``/``ao_next`` of shape (1, 1) are not kept (one pass from 0)."""
    keep = sq.shape[0] == height
    for j in prange(len(rows)):
        py = rows[j]
        state = np.empty(1, np.uint64)
        o = np.empty(3); d = np.empty(3); n = np.empty(3); p = np.empty(3); od = np.empty(3)
        aos = np.zeros(3)
        stack = np.empty(depth + 2, np.int64); stack_t = np.empty(depth + 2)
        for px in range(width):
            if not active[py, px]:
                continue
            state[0] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(py * width + px + 1)
            aos[0] = _rand(state); aos[1] = _rand(state); aos[2] = 0.0
            jox = _rand(state); joy = _rand(state)
            s0 = nsamp[py, px] if keep else 0
            if s0 > 0:
                state[0] = rng[py, px]; aos[2] = ao_next[py, px]
            acc_r, acc_g, acc_b, acc_a = acc[py, px, 0], acc[py, px, 1], acc[py, px, 2], acc[py, px, 3]
            acc_q = sq[py, px] if keep else 0.0
            for s in range(s0, s1):
                # sub-pixel positions follow the R2 sequence (shifted per pixel) like the AO rays: evenly
                # spread for any sample count, so lines one pixel wide come out solid, not dotted
                jx = (jox + s * 0.7548776662466927) % 1.0 if samples > 1 else 0.5
                jy = (joy + s * 0.5698402909980532) % 1.0 if samples > 1 else 0.5
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
                vr = vg = vb = va = 0.0
                trans = 1.0  # light still passing through see-through triangles in front
                for layer in range(16):
                    t, code = _trace(o, d, 1e30, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, False)
                    if code < 0:
                        vr += trans * bg[0]; vg += trans * bg[1]; vb += trans * bg[2]
                        va += trans * bg[3]
                        break
                    kind = code >> 40
                    i = code & INDEX_MASK
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
                        # a flat cap: the hit is closer to the end's plane than to the side
                        cap = cyl[i, 7] if h < 0.5 else cyl[i, 8]
                        if cap > 0.5 and min(abs(h), abs(1.0 - h)) * math.sqrt(L2) < cyl[i, 6] - nn:
                            sign = (-1.0 if h < 0.5 else 1.0) / math.sqrt(L2)
                            n[0], n[1], n[2] = sign * ax, sign * ay, sign * az
                        else:
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
                                        ao_samples, ao_dist, state, od, grid, starts, items, nodes, bvh_items, stack, stack_t, sph, cyl, tri, aos)
                    w = trans * alpha
                    vr += w * sr; vg += w * sg; vb += w * sb; va += w
                    trans *= 1.0 - alpha
                    if trans < 1e-3:
                        break
                    for k in range(3):  # continue behind the see-through surface
                        o[k] = p[k] - 2.0 * n[k] * eps
                acc_r += vr; acc_g += vg; acc_b += vb; acc_a += va
                acc_q += vr * vr + vg * vg + vb * vb + va * va
            acc[py, px, 0] = acc_r; acc[py, px, 1] = acc_g; acc[py, px, 2] = acc_b; acc[py, px, 3] = acc_a
            if keep:
                sq[py, px] = acc_q; nsamp[py, px] = s1; rng[py, px] = state[0]; ao_next[py, px] = aos[2]


# ----------------------------------------------------------------------------- public API
def raytrace(scene: Scene, camera: dict, size=(1600, 1200), *, samples: int = 8, ao_samples: int = 16,
             ao_distance: float | None = None, shadows: bool = True, background=(1.0, 1.0, 1.0),
             transparent_background: bool = False, ambient: float = 0.55, diffuse: float = 0.6,
             specular: float = 0.25, shininess: float = 40.0, light_direction=(-0.4, 0.6, 1.0), seed: int = 1,
             noise_threshold: float = 0.0, min_samples: int | None = None, progress=None, device=None):
    """Ray-trace ``scene`` seen through ``camera`` (see :func:`ewald.render.camera_state`).

    Returns an (H, W, 4) float RGBA image in [0, 1]. ``light_direction`` is given
    in camera space (x right, y up, z toward the viewer).

    Adaptive sampling: with ``noise_threshold`` > 0, every pixel first takes ``min_samples``
    rays (default ``max(8, samples // 4)``), then pixels whose estimated error (the standard
    error of the pixel mean, RGBA summed, in 0-1 units) is still above the threshold, and
    their neighbours, take more in doubling passes up to ``samples``. Flat background and
    evenly lit surfaces stop early; edges, shadows and AO keep the full count. 0 (default)
    gives every pixel ``samples`` rays. A pixel's samples do not depend on the passes, so a
    pixel that runs to ``samples`` matches the non-adaptive image exactly.

    ``progress(done, total)`` is called after each batch of rows; when it returns ``False`` the
    trace stops and :class:`RenderCancelled` is raised. The image does not depend on it.

    ``device``: ``"cpu"`` (default, numba on every core), ``"cuda"`` (an NVIDIA GPU through CuPy),
    ``"opencl"`` (an Intel, AMD, NVIDIA or Apple GPU through pyopencl) or ``"auto"`` (CUDA, else an
    OpenCL GPU, else the CPU); :mod:`ewald.raytrace_gpu`. Single precision, so a GPU image matches
    the CPU one to within a few 1e-3 rather than exactly.
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
    accel = Accel(scene)
    gpu = _gpu_tracer(scene, accel, device)
    samples = max(1, int(samples))
    first = samples
    if noise_threshold > 0:
        first = min(samples, max(1, int(min_samples)) if min_samples is not None else max(8, samples // 4))
    adaptive = first < samples
    acc = np.zeros((height, width, 4))
    if adaptive:
        sq, nsamp = np.zeros((height, width)), np.zeros((height, width), np.int64)
        rng, ao_next = np.zeros((height, width), np.uint64), np.zeros((height, width))
    else:
        sq, nsamp, rng, ao_next = np.zeros((1, 1)), np.zeros((1, 1), np.int64), np.zeros((1, 1), np.uint64), \
            np.zeros((1, 1))
    active = np.ones((height, width), bool)
    if gpu is not None:
        params = np.concatenate([eye, fwd, right, up, [half_w, half_h], light,
                                 [ambient, diffuse, specular, shininess], bg, [ao_distance]])
        ints = [width, height, int(ortho), 0, samples, int(ao_samples), int(bool(shadows)), int(seed), 0,
                int(adaptive)]
        acc, sq, nsamp, rng, ao_next = (gpu.upload(a) for a in (acc, sq, nsamp, rng, ao_next))
    rows = _row_order(height)
    work_total, work = height * samples, 0     # row-samples; converged pixels finish early
    s0, s1 = 0, first
    while True:
        pass_rows = rows if s0 == 0 else rows[active[rows].any(1)]
        # with a progress callback, trace in batches (each one still spread over every thread)
        batches = np.array_split(pass_rows, min(PROGRESS_STEPS, len(pass_rows))) if progress is not None \
            else [pass_rows]
        if gpu is not None:
            dev_active = gpu.upload(active.astype(np.uint8))
        for batch in batches:
            if gpu is not None:
                ints[3] = s1
                gpu.trace(params, ints, batch, dev_active, acc, sq, nsamp, rng, ao_next)
            else:
                _render(width, height, eye, fwd, right, up, half_w, half_h, ortho, s1, samples, int(ao_samples),
                        float(ao_distance), light, ambient, diffuse, specular, shininess, bool(shadows), bg,
                        accel.grid, accel.starts, accel.items, accel.nodes, accel.bvh_items, accel.depth,
                        accel.sph, accel.cyl, accel.tri, scene.sph_c, scene.sph_r, scene.sph_rgb,
                        scene.cyl_a, scene.cyl_b, scene.cyl_rgb,
                        scene.tri_a, scene.tri_b, scene.tri_c, np.ascontiguousarray(scene.tri_n),
                        np.ascontiguousarray(scene.tri_vrgb), scene.tri_alpha, int(seed), batch, active,
                        acc, sq, nsamp, rng, ao_next)
            work += len(batch) * (s1 - s0)
            if progress is not None and progress(min(height, work * height // work_total), height) is False:
                raise RenderCancelled("ray trace cancelled")
        if not adaptive:
            break
        if s1 >= samples:
            break
        if gpu is not None:
            n_host = gpu.download(nsamp)
            active = _still_noisy(gpu.download(acc), gpu.download(sq), n_host, noise_threshold) & (n_host < samples)
        else:
            active = _still_noisy(acc, sq, nsamp, noise_threshold) & (nsamp < samples)
        if not active.any():
            break
        s0, s1 = s1, min(samples, 2 * s1)   # pixels that stopped earlier and are woken again catch up
    if progress is not None and work < work_total:
        progress(height, height)
    if gpu is not None:
        acc, nsamp = gpu.download(acc), gpu.download(nsamp)
    return acc / (nsamp[..., None] if adaptive else samples)


GPU_REAL = "float"   # precision of the GPU tracers; tests set "double" to compare with the CPU


def _gpu_tracer(scene, accel, device):
    """A GPU tracer from :mod:`ewald.raytrace_gpu` when ``device`` names a GPU, else None."""
    from .raytrace_gpu import tracer

    return tracer(scene, accel, device, GPU_REAL)


def _still_noisy(acc, sq, nsamp, threshold, grow=2):
    """Pixels whose mean is not yet within ``threshold`` (standard error, RGBA summed), grown by
    ``grow`` pixels: a thin edge or line that every first ray of a pixel missed still shows as
    noise in a pixel nearby, and wakes it."""
    n = nsamp.astype(float)
    mean_sq = (acc ** 2).sum(-1) / n ** 2
    var = np.maximum(sq / n - mean_sq, 0.0) * n / np.maximum(n - 1, 1)   # per-sample variance
    noisy = var / n > threshold ** 2
    for _ in range(grow):
        grown = noisy.copy()
        grown[1:] |= noisy[:-1]; grown[:-1] |= noisy[1:]
        grown[:, 1:] |= noisy[:, :-1]; grown[:, :-1] |= noisy[:, 1:]
        grown[1:, 1:] |= noisy[:-1, :-1]; grown[:-1, :-1] |= noisy[1:, 1:]
        grown[1:, :-1] |= noisy[:-1, 1:]; grown[:-1, 1:] |= noisy[1:, :-1]
        noisy = grown
    return noisy


def _row_order(height: int, stride: int = 64):
    """Rows interleaved (0, 64, 128, ..., 1, 65, ...): prange hands each thread one contiguous
    block of the loop, and with plain row order the threads that get the empty margins finish
    long before the ones with the structure. Pixels do not depend on the order."""
    return np.concatenate([np.arange(k, height, stride) for k in range(min(stride, height))]).astype(np.int64)
