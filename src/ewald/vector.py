"""Vector renderer: SVG / PDF / EPS figures with true circles, bond sticks and vector text.

Atoms are drawn as shaded discs and half-bonds as projected sticks, painted far to
near (painter's algorithm, the way VESTA exports vector images). Overlays (text,
color legend, type legend, scale bar) stay editable text and shapes, so a figure
can be finished in Illustrator or Inkscape. Uses matplotlib only, so it needs no
OpenGL context. Colors, radii, bonds and the camera match the other renderers.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np

from .render import FLAT_OUTLINE, STICK_OUTLINE

# Measured cost of one drawn shape (render_vector, 4-core container; benchmarks/vector_size.py):
# seconds and bytes per atom disc (four circles when shaded), per stick (a half-bond, trajectory or
# cell line piece, arrow part) and per mesh triangle or edge. Grows linearly with the shape count.
SHAPE_SECONDS = {"atom": 0.8e-3, "stick": 0.47e-3, "face": 0.43e-3}
SHAPE_BYTES = {".svg": {"atom": 2600, "stick": 390, "face": 990},
               ".pdf": {"atom": 570, "stick": 55, "face": 60}}
SHAPE_BYTES[".eps"] = SHAPE_BYTES[".ps"] = SHAPE_BYTES[".svg"]  # PostScript text is about as verbose as SVG
# Above either estimate vector output warns (render(large_vector='warn')) or asks (the GUI):
# a raster image is then far smaller and faster.
LARGE_VECTOR_SECONDS = 10.0
LARGE_VECTOR_BYTES = 10e6


class LargeVectorWarning(UserWarning):
    """Vector output of a scene with many shapes: slow and a large file (see :func:`vector_cost`)."""


class LargeVectorError(RuntimeError):
    """Raised by ``render(..., large_vector='raise')`` before drawing a large vector figure;
    ``estimate`` is the :func:`vector_cost` dict, so a caller can ask the user and render again."""

    def __init__(self, estimate):
        super().__init__(large_vector_message(estimate))
        self.estimate = estimate


def vector_cost(scene, filename=None) -> dict:
    """Shapes a vector figure of ``scene`` draws and the estimated time and file size: a dict with
    ``atoms``, ``sticks``, ``faces``, ``shapes`` (their sum), ``seconds``, ``bytes`` (None unless
    ``filename`` ends in .svg/.pdf/.eps/.ps), ``format`` and ``large`` (over ``LARGE_VECTOR_SECONDS``
    or ``LARGE_VECTOR_BYTES``)."""
    n = {"atom": 0, "stick": 0, "face": 0}
    for layer in scene.layers:
        prim = layer.primitive
        if layer.kind == "spheres":
            n["atom"] += int(np.count_nonzero(np.asarray(prim.alpha) > 0)) if prim.count else 0
        elif layer.kind == "cylinders":
            n["stick"] += int(prim.count)
        elif layer.kind == "arrows":
            n["stick"] += 3 * int(prim.count)
        elif layer.kind == "lines":
            n["stick"] += len(prim.segments())
        elif layer.kind == "mesh":
            n["face"] += int(prim.face_count) + (len(prim.edges) if prim.edges is not None else 0)
    fmt = os.path.splitext(str(filename or ""))[1].lower()
    per = SHAPE_BYTES.get(fmt)
    seconds = sum(SHAPE_SECONDS[k] * v for k, v in n.items())
    size = None if per is None else sum(per[k] * v for k, v in n.items())
    return {"atoms": n["atom"], "sticks": n["stick"], "faces": n["face"], "shapes": sum(n.values()),
            "seconds": seconds, "bytes": size, "format": fmt.lstrip(".") or None,
            "large": seconds > LARGE_VECTOR_SECONDS or (size or 0) > LARGE_VECTOR_BYTES}


def _caller_level() -> int:
    """``stacklevel`` of the first frame outside ewald, so the warning names the user's line."""
    import sys

    root = os.path.dirname(os.path.abspath(__file__))
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(root):
        frame, level = frame.f_back, level + 1
    return level


def large_vector_message(est) -> str:
    """One sentence on a large vector figure: what it holds, the expected size and time, what to do."""
    parts = [f"{est['atoms']:,} atoms"] if est["atoms"] else []
    if est["sticks"]:
        parts.append(f"{est['sticks']:,} bond and line pieces")
    if est["faces"]:
        parts.append(f"{est['faces']:,} mesh faces and edges")
    size = f"a {est['bytes'] / 1e6:.0f} MB {est['format'].upper()} file and " if est["bytes"] else ""
    return (f"Vector output draws {est['shapes']:,} shapes ({', '.join(parts)}): expect {size}"
            f"about {max(1, round(est['seconds'])):,} s (estimate). A raster image (PNG with renderer="
            f"'opengl' or 'raytrace') is much smaller and faster; pass large_vector='ignore' to draw it anyway.")

def _basis(camera):
    """Camera eye and orthonormal (right, up, forward) axes."""
    eye = np.asarray(camera["position"], float)
    fwd = np.asarray(camera["focal_point"], float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(camera["up"], float))
    right /= np.linalg.norm(right)
    return eye, right, np.cross(right, fwd), fwd


def _project(points, camera, size):
    """World -> (screen x, screen y, depth, pixels per Angstrom at that depth)."""
    w, h = size
    eye, right, up, fwd = _basis(camera)
    rel = np.asarray(points, float).reshape(-1, 3) - eye
    x, y, z = rel @ right, rel @ up, rel @ fwd
    if camera.get("parallel_projection", True):
        s = np.full(len(z), (h / 2) / float(camera["parallel_scale"]))
    else:
        t = math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2)
        s = (h / 2) / (np.maximum(z, 1e-9) * t)
    return w / 2 + x * s, h / 2 - y * s, z, s


def _mix(c, white):
    return c + (1 - c) * white


def _hull(pts):
    """Convex hull of 2D points (Andrew's monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, np.round(pts, 6)))
    if len(pts) < 3:
        return pts

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return half(pts) + half(pts[::-1])


def _arrow_prims(data, vectors, camera, size, pt, shading, sides=16, arrow_sets=None):
    """Arrows as their true silhouettes: hulls of the projected shaft rings, then of the head rim
    and tip (an arrow pointing at the viewer becomes a disc, as it should). The shaft is split
    where it passes its atom, and each piece is painted in front of the atom only if it is
    nearer than the atom center, so arrows run through atoms instead of lying on top."""
    from matplotlib.patches import Polygon

    from .arrows import _frame, scene_arrows

    if arrow_sets is None:
        arrow_sets = scene_arrows(data, vectors)
    prims = []
    ang = 2 * np.pi * np.arange(sides) / sides
    cs, sn = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]
    for arrows in arrow_sets:
        m = arrows.count
        d = arrows.direction
        e1, e2 = _frame(d)
        ring = cs * e1[:, None] + sn * e2[:, None]
        neck = arrows.neck
        h = np.clip(np.einsum("ij,ij->i", arrows.anchor - arrows.base, d), 0, None)
        h = np.minimum(h, np.linalg.norm(neck - arrows.base, axis=1))
        mid = arrows.base + h[:, None] * d  # where the shaft passes its atom
        r, R = arrows.radius[:, None, None], arrows.head_radius[:, None, None]
        pts = np.concatenate([arrows.base[:, None] + r * ring, mid[:, None] + r * ring, neck[:, None] + r * ring,
                              neck[:, None] + R * ring, arrows.tip[:, None]], axis=1)  # m x (4s+1) x 3
        px, py, pz, _ = _project(pts.reshape(-1, 3), camera, size)
        px, py, pz = (a.reshape(m, -1) for a in (px, py, pz))
        _, _, za, _ = _project(arrows.anchor, camera, size)
        _, _, zc, _ = _project(np.concatenate([arrows.base, mid, neck, arrows.tip]), camera, size)
        zb, zm, zn, zt = zc.reshape(4, m)
        flat = arrows.shading == "flat" or shading == "flat"
        a = arrows.alpha
        s = sides
        for k in range(m):
            if not camera.get("parallel_projection", True) and pz[k].min() <= 0:
                continue
            xy = np.column_stack([px[k], py[k]])
            c = arrows.rgb[k]
            edge, lw = (*(c * 0.45), a), 0.6 * pt
            body = (*(c if flat else c * 0.85), a)
            L = np.linalg.norm(neck[k] - arrows.base[k])
            # (outline points, depth of the piece's end away from the atom, fill)
            pieces = [(xy[:2 * s], zb[k], body, h[k] > 1e-9), (xy[s:3 * s], zn[k], body, h[k] < L - 1e-9),
                      (xy[3 * s:], zt[k] if h[k] < L + 0.5 * arrows.head_length[k] else zn[k], (*c, a), True)]
            for ring_xy, z_far, fill, present in pieces:
                if not present:
                    continue
                depth = z_far - 1e-6 if z_far < za[k] else za[k] + 1e-6
                prims.append((depth, [Polygon(_hull(ring_xy), closed=True)], [fill], [edge], [lw]))
    return prims


def scene_prims(scene, camera, size, pt, shading="gradient", ink="black", light=None):
    """Every layer of an :class:`ewald.scene.Scene` as depth-tagged matplotlib patches
    ``(depth, patches, facecolors, edgecolors, linewidths)``, to be painted far to near; ``light`` moves
    the highlights (see :func:`ewald.render.light_params`)."""
    prims = []
    for layer in scene.layers:
        prim = layer.primitive
        if layer.kind == "spheres":
            prims += _sphere_prims(prim, camera, size, pt, shading, light)
        elif layer.kind == "cylinders":
            prims += _cylinder_prims(prim, camera, size, pt, shading, light)
        elif layer.kind == "arrows":
            prims += _arrow_prims(None, None, camera, size, pt, shading, arrow_sets=[prim])
        elif layer.kind == "mesh":
            prims += _mesh_prims(prim, camera, size, pt, light)
        elif layer.kind == "lines":
            prims += _line_prims(prim, camera, size, ink)
    return prims


def _highlight(light):
    """Screen offset (x right, y down) of the highlight towards the key light (the default one when None)."""
    from .render import light_params

    d, _ = light_params(light)
    return 1.6 * d[0], -1.6 * d[1]


def _sphere_prims(sph, camera, size, pt, shading, light=None):
    """Shaded discs (or flat discs with an outline), lit from ``light`` (see :func:`ewald.render.light_params`)."""
    from matplotlib.patches import Circle

    prims = []
    if not sph.count:
        return prims
    colors, radii, alpha = sph.colors(), sph.sizes(), sph.alpha
    keep = np.flatnonzero(alpha > 0)
    sx, sy, z, s = _project(sph.centers[keep], camera, size)
    hx, hy = _highlight(light)
    for k, i in enumerate(keep):
        if z[k] <= 0 and not camera.get("parallel_projection", True):
            continue
        c, r, a = colors[i], radii[i] * s[k], alpha[i]
        x, y = sx[k], sy[k]
        edge = (*(c * 0.45), a)
        if shading == "flat":  # the OpenGL flat shader: a dark rim of max(0.8 px, 5 %) inside the disc
            rim = max(0.8, 0.05 * r)
            prims.append((z[k], [Circle((x, y), r), Circle((x, y), max(r - rim, 0.5 * r))],
                          [(FLAT_OUTLINE, FLAT_OUTLINE, FLAT_OUTLINE, a), (*c, a)], ["none", "none"], [0, 0]))
        else:
            prims.append((z[k], [Circle((x, y), r), Circle((x + 0.06 * hx * r, y + 0.06 * hy * r), 0.86 * r),
                                 Circle((x + 0.2 * hx * r, y + 0.2 * hy * r), 0.52 * r),
                                 Circle((x + 0.3 * hx * r, y + 0.3 * hy * r), 0.2 * r)],
                          [(*(c * 0.62), a), (*c, a), (*_mix(c, 0.3), a), (*_mix(c, 0.7), a)],
                          [edge, "none", "none", "none"], [max(0.3, 0.03 * r) * pt, 0, 0, 0]))
    return prims


def _cylinder_prims(cyl, camera, size, pt, shading, light=None):
    """Projected sticks with a highlight stripe, each painted just before its start (a half-bond's
    own atom hides the stick's root), and their caps: a round cap is a disc of the stick radius, a
    flat cap the projected ellipse; one on the near end is painted over the stick, a round one on
    the far end under it (a flat one there is hidden)."""
    from matplotlib.patches import Circle, Ellipse, Polygon

    prims = []
    keep = np.flatnonzero(cyl.alpha > 0)
    if not len(keep):
        return prims
    ax0, ay0, z0, s0 = _project(cyl.start[keep], camera, size)
    ax1, ay1, z1, s1 = _project(cyl.end[keep], camera, size)
    hx, hy = _highlight(light)
    for k, i in enumerate(keep):
        dx, dy = ax1[k] - ax0[k], ay1[k] - ay0[k]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length
        c, a = cyl.rgb[i], cyl.alpha[i]
        r0, r1 = cyl.radius[i] * s0[k], cyl.radius[i] * s1[k]
        quad = [(ax0[k] + nx * r0, ay0[k] + ny * r0), (ax1[k] + nx * r1, ay1[k] + ny * r1),
                (ax1[k] - nx * r1, ay1[k] - ny * r1), (ax0[k] - nx * r0, ay0[k] - ny * r0)]
        patches, fc, ec, lw = [Polygon(quad, closed=True)], [(*(c * 0.85), a)], [(*(c * 0.45), a)], \
            [max(0.3, 0.06 * r0) * pt]
        body = fc[0], ec[0], lw[0]   # caps match the stick
        if shading == "flat":  # unlit, inside a dark outline 30 % of the radius wide (the OpenGL hull)
            w0, w1 = (1 + STICK_OUTLINE) * r0, (1 + STICK_OUTLINE) * r1
            hull = [(ax0[k] + nx * w0, ay0[k] + ny * w0), (ax1[k] + nx * w1, ay1[k] + ny * w1),
                    (ax1[k] - nx * w1, ay1[k] - ny * w1), (ax0[k] - nx * w0, ay0[k] - ny * w0)]
            patches, fc, ec, lw = [Polygon(hull, closed=True), patches[0]], [(FLAT_OUTLINE, FLAT_OUTLINE, FLAT_OUTLINE, a), (*c, a)], \
                ["none", "none"], [0, 0]
            body = fc[1], "none", 0
        if shading != "flat":  # highlight stripe toward the light (upper left by default)
            side = 1.0 if nx * hx + ny * hy > 0 else -1.0
            o0, o1 = 0.35 * r0 * side, 0.35 * r1 * side
            hw0, hw1 = 0.25 * r0, 0.25 * r1
            patches.append(Polygon([(ax0[k] + nx * (o0 + hw0), ay0[k] + ny * (o0 + hw0)),
                                    (ax1[k] + nx * (o1 + hw1), ay1[k] + ny * (o1 + hw1)),
                                    (ax1[k] + nx * (o1 - hw1), ay1[k] + ny * (o1 - hw1)),
                                    (ax0[k] + nx * (o0 - hw0), ay0[k] + ny * (o0 - hw0))], closed=True))
            fc.append((*_mix(c, 0.35), a)); ec.append("none"); lw.append(0)
        prims.append((z0[k] + 1e-6, patches, fc, ec, lw))
        span = float(np.linalg.norm(cyl.end[i] - cyl.start[i]))
        tilt = abs(z1[k] - z0[k]) / span if span > 0 else 0.0   # |cos| of the axis to the view direction
        for x, y, z, rr, code, near in ((ax0[k], ay0[k], z0[k], r0, cyl.start_cap[i], z0[k] < z1[k]),
                                        (ax1[k], ay1[k], z1[k], r1, cyl.end_cap[i], z1[k] < z0[k])):
            if code == 2:  # outlined disc under the stick, plain one over its end line
                prims.append((z0[k] + 2e-6, [Circle((x, y), rr)], [body[0]], [body[1]], [body[2]]))
                prims.append((min(z, z0[k]) - 1e-7, [Circle((x, y), rr - 0.5 * body[2] / pt)], [body[0]], ["none"], [0]))
            elif code == 1 and near:
                cap = Ellipse((x, y), 2 * rr, 2 * rr * tilt, angle=math.degrees(math.atan2(ny, nx)))
                prims.append((z - 1e-7, [cap], [body[0]], [body[1]], [body[2]]))
    return prims


def _mesh_prims(mesh, camera, size, pt, light=None):
    """Lit triangles (two-sided, like the OpenGL faces) and edge strips just in front of them."""
    from matplotlib.patches import Polygon

    from .render import _rgb, light_params

    prims = []
    if not mesh.face_count:
        return prims
    v, f = mesh.vertices, mesh.faces
    vx, vy, vz, vs = _project(v, camera, size)
    _, right, up, fwd = _basis(camera)
    d, strength = light_params(light)  # the ray tracer's key light
    light = d @ np.array([right, up, -fwd])
    light /= np.linalg.norm(light)
    vn, vc = mesh.vertex_properties.get("Normal"), mesh.vertex_properties.get("Color")
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]) if vn is None \
        else np.asarray(vn, float)[f].mean(1)  # smooth isosurface normals when given
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lit = np.minimum(1.0, 0.55 + 0.45 * strength * np.abs(n @ light))
    if vc is not None:  # color-mapped slices keep their exact colormap (unlit)
        fcol = np.clip(np.asarray(vc, float), 0, 1)[f].mean(1)
    else:
        fcol = np.clip(mesh.face_colors(), 0, 1) * lit[:, None]
    a = 1.0 - mesh.transparency
    for k, tri in enumerate(f):
        poly = Polygon(np.column_stack([vx[tri], vy[tri]]), closed=True)
        # the face color as a hairline outline closes anti-aliasing seams between triangles
        prims.append((vz[tri].mean(), [poly], [(*fcol[k], a)], [(*fcol[k], a)], [0.3 * pt]))
    if mesh.edges is not None and len(mesh.edges):
        ec = _rgb(mesh.edge_color)
        for i, j in mesh.edges:
            dx, dy = vx[j] - vx[i], vy[j] - vy[i]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            hw = max(0.5, 0.5 * mesh.edge_width * 0.5 * (vs[i] + vs[j]))
            nx, ny = -dy / length * hw, dx / length * hw
            quad = [(vx[i] + nx, vy[i] + ny), (vx[j] + nx, vy[j] + ny), (vx[j] - nx, vy[j] - ny),
                    (vx[i] - nx, vy[i] - ny)]
            prims.append((0.5 * (vz[i] + vz[j]) - 1e-3, [Polygon(quad, closed=True)], [(*ec, 1.0)], ["none"], [0]))
    return prims


def _line_prims(lines, camera, size, ink):
    """Line pieces as screen-space strips as wide as :func:`ewald.scene.line_pixels`, extended by half a
    width so corners close."""
    from matplotlib.patches import Polygon

    from .render import _rgb

    prims = []
    seg = lines.segments()
    if not len(seg):
        return prims
    cx, cy, cz, _ = _project(lines.points, camera, size)
    from .scene import line_pixels

    half = 0.5 * line_pixels(lines.width, size[1])
    seg_rgb = lines.segment_colors(_rgb(ink))
    for (i, j), cc in zip(seg, seg_rgb):
        dx, dy = cx[j] - cx[i], cy[j] - cy[i]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx, ny = -dy / length * half, dx / length * half
        ex, ey = dx / length * half, dy / length * half
        quad = [(cx[i] - ex + nx, cy[i] - ey + ny), (cx[j] + ex + nx, cy[j] + ey + ny),
                (cx[j] + ex - nx, cy[j] + ey - ny), (cx[i] - ex - nx, cy[i] - ey - ny)]
        prims.append((0.5 * (cz[i] + cz[j]), [Polygon(quad, closed=True)], [(*cc, 1.0)], ["none"], [0]))
    return prims


def render_vector(data, filename=None, *, camera, size=(1600, 1200), background="white",
                  transparent_background=False, radius_scale=1.0, type_colors=None, type_radii=None,
                  bonds=None, bond_pairs=None, bond_radius=0.15, show_cell=True, cell_color=None,
                  cell_width=2.0, shading="gradient", text=None, legend=True, type_legend=False,
                  scale_bar=None, dpi=None, radius_preset="covalent", oxidation_states=None, coordination=None,
                  vectors=None, trajectory_width=2.0, trajectory_color_by=None, trajectory_colormap="viridis",
                  trajectory_color_range=None, trajectory_time_range=None, trajectory_trail=None,
                  trajectory_radius=None, visuals=None, frame=0, overlays=None, boundary=None,
                  boundary_bonds="half", replicate=None, replicate_cell="super", show_atoms=True, surfaces=None,
                  type_transparency=None, selection_color=None, light=None, measurements=None,
                  large_vector="warn", **_ignored):
    """Draw ``data`` as a vector figure and save it (format from the extension).

    ``shading``: 'gradient' (shaded discs) or 'flat' (one color plus outline, VESTA's
    flat style). ``dpi`` sets the physical page size (``size / dpi`` inches; 100 when
    not given). ``large_vector``: what a scene with many shapes does (a large, slow file; see
    :func:`vector_cost`): 'warn' (a :class:`LargeVectorWarning`, then draws it), 'ignore' or 'raise'
    (:class:`LargeVectorError` before drawing). ``light``: the key light (see :func:`ewald.render.light_params`) that places the highlights.
    ``visuals``: extra visual elements (plugins), see :mod:`ewald.scene`;
    ``overlays``: overlay elements, see :mod:`ewald.overlays`.
    Returns the rasterized image as an RGB(A) uint8 array.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    from . import scene as ir
    from .overlays import cell_line_color, draw_items_mpl, ink_color, overlay_layout
    from .render import _rgb

    w, h = int(size[0]), int(size[1])
    dpi = float(dpi or 100)
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    bg = _rgb(background)
    fig.patch.set_facecolor((*bg, 0.0) if transparent_background else bg)
    ax.patch.set_alpha(0.0)
    pt = 72.0 / dpi  # points per pixel

    scene = ir.build(data, ctx=ir.SceneContext(frame=frame), visuals=visuals, radius_scale=radius_scale,
                     type_colors=type_colors, type_radii=type_radii, radius_preset=radius_preset,
                     oxidation_states=oxidation_states, coordination=coordination, bonds=bonds,
                     bond_pairs=bond_pairs, bond_radius=bond_radius, vectors=vectors,
                     trajectory_width=trajectory_width, trajectory_color_by=trajectory_color_by,
                     trajectory_colormap=trajectory_colormap, trajectory_color_range=trajectory_color_range,
                     trajectory_time_range=trajectory_time_range, trajectory_trail=trajectory_trail,
                     trajectory_radius=trajectory_radius, show_cell=show_cell, cell_color=cell_color,
                     cell_width=cell_width, boundary=boundary, boundary_bonds=boundary_bonds,
                     replicate=replicate, replicate_cell=replicate_cell, show_atoms=show_atoms,
                     surfaces=surfaces, type_transparency=type_transparency, selection_color=selection_color,
                     measurements=measurements)
    if large_vector != "ignore":
        est = vector_cost(scene, filename)
        if est["large"]:
            if large_vector == "raise":
                raise LargeVectorError(est)
            warnings.warn(large_vector_message(est), LargeVectorWarning, stacklevel=_caller_level())
    ink = cell_line_color(bg, transparent_background)
    prims = scene_prims(scene, camera, (w, h), pt, shading, ink, light)
    prims.sort(key=lambda p: -p[0])  # far first
    patches, fc, ec, lw = [], [], [], []
    for _, p, f, e, l in prims:
        patches += p; fc += f; ec += e; lw += l
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors=ec, linewidths=lw,
                                          match_original=False, antialiased=True))

    lay = overlay_layout(data, (w, h), text=text, legend=legend, type_legend=type_legend, scale_bar=scale_bar,
                         camera=camera, type_colors=type_colors, overlays=overlays, frame=frame,
                         background=background, transparent=transparent_background, colorbars=scene.legends)
    ink = ink_color(background, transparent_background)
    fs, small = lay["font_size"] * pt, lay["small"] * pt
    if "text" in lay:
        t = lay["text"]
        ax.text(t["x"], t["y"], t["text"], color=ink, fontsize=fs, va="top", ha="left")
    for row in lay.get("types", []):
        d = row["d"]
        ax.add_patch(Circle((row["x"] + d / 2, row["y"] + d / 2), d / 2, facecolor=row["rgb"], edgecolor=ink,
                            linewidth=max(1, d // 14) * pt))
        ax.text(row["x"] + 1.5 * d, row["y"] + d / 2, row["name"], color=ink, fontsize=fs, va="center", ha="left")
    if "scale_bar" in lay:
        sb = lay["scale_bar"]
        th = sb["thickness"]
        ax.add_patch(Rectangle((sb["x0"], sb["y"] - th), sb["x1"] - sb["x0"], th, facecolor=ink, edgecolor="none"))
        ax.text((sb["x0"] + sb["x1"]) / 2, sb["y"] - th - 0.3 * lay["font_size"], sb["label"], color=ink,
                fontsize=small, va="bottom", ha="center")

    if "items" in lay:
        draw_items_mpl(ax, lay["items"], ink, pt, lay["font_size"])

    if filename:
        fig.savefig(filename, dpi=dpi, transparent=transparent_background, facecolor=fig.get_facecolor())
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba()).copy()
    return img if transparent_background else img[..., :3]
