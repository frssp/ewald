"""Vector renderer: SVG / PDF / EPS figures with true circles, bond sticks and vector text.

Atoms are drawn as shaded discs and half-bonds as projected sticks, painted far to
near (painter's algorithm, the way VESTA exports vector images). Overlays (text,
color legend, type legend, scale bar) stay editable text and shapes, so a figure
can be finished in Illustrator or Inkscape. Uses matplotlib only, so it needs no
OpenGL context. Colors, radii, bonds and the camera match the other renderers.
"""
from __future__ import annotations

import math

import numpy as np

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


def scene_prims(scene, camera, size, pt, shading="gradient", ink="black"):
    """Every layer of an :class:`ewald.scene.Scene` as depth-tagged matplotlib patches
    ``(depth, patches, facecolors, edgecolors, linewidths)``, to be painted far to near."""
    prims = []
    for layer in scene.layers:
        prim = layer.primitive
        if layer.kind == "spheres":
            prims += _sphere_prims(prim, camera, size, pt, shading)
        elif layer.kind == "cylinders":
            prims += _cylinder_prims(prim, camera, size, pt, shading)
        elif layer.kind == "arrows":
            prims += _arrow_prims(None, None, camera, size, pt, shading, arrow_sets=[prim])
        elif layer.kind == "mesh":
            prims += _mesh_prims(prim, camera, size, pt)
        elif layer.kind == "lines":
            prims += _line_prims(prim, camera, size, ink)
    return prims


def _sphere_prims(sph, camera, size, pt, shading):
    """Shaded discs (or flat discs with an outline)."""
    from matplotlib.patches import Circle

    prims = []
    if not sph.count:
        return prims
    colors, radii, alpha = sph.colors(), sph.sizes(), sph.alpha
    keep = np.flatnonzero(alpha > 0)
    sx, sy, z, s = _project(sph.centers[keep], camera, size)
    for k, i in enumerate(keep):
        if z[k] <= 0 and not camera.get("parallel_projection", True):
            continue
        c, r, a = colors[i], radii[i] * s[k], alpha[i]
        x, y = sx[k], sy[k]
        edge = (*(c * 0.45), a)
        if shading == "flat":
            prims.append((z[k], [Circle((x, y), r)], [(*c, a)], [edge], [max(0.4, 0.06 * r) * pt]))
        else:
            prims.append((z[k], [Circle((x, y), r), Circle((x - 0.06 * r, y - 0.06 * r), 0.86 * r),
                                 Circle((x - 0.2 * r, y - 0.2 * r), 0.52 * r),
                                 Circle((x - 0.3 * r, y - 0.3 * r), 0.2 * r)],
                          [(*(c * 0.62), a), (*c, a), (*_mix(c, 0.3), a), (*_mix(c, 0.7), a)],
                          [edge, "none", "none", "none"], [max(0.3, 0.03 * r) * pt, 0, 0, 0]))
    return prims


def _cylinder_prims(cyl, camera, size, pt, shading):
    """Projected sticks with a highlight stripe, each painted just before its start (a half-bond's
    own atom hides the stick's root)."""
    from matplotlib.patches import Polygon

    prims = []
    keep = np.flatnonzero(cyl.alpha > 0)
    if not len(keep):
        return prims
    ax0, ay0, z0, s0 = _project(cyl.start[keep], camera, size)
    ax1, ay1, z1, s1 = _project(cyl.end[keep], camera, size)
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
        if shading != "flat":  # highlight stripe toward the light (upper left)
            side = 1.0 if nx + ny < 0 else -1.0
            o0, o1 = 0.35 * r0 * side, 0.35 * r1 * side
            hw0, hw1 = 0.25 * r0, 0.25 * r1
            patches.append(Polygon([(ax0[k] + nx * (o0 + hw0), ay0[k] + ny * (o0 + hw0)),
                                    (ax1[k] + nx * (o1 + hw1), ay1[k] + ny * (o1 + hw1)),
                                    (ax1[k] + nx * (o1 - hw1), ay1[k] + ny * (o1 - hw1)),
                                    (ax0[k] + nx * (o0 - hw0), ay0[k] + ny * (o0 - hw0))], closed=True))
            fc.append((*_mix(c, 0.35), a)); ec.append("none"); lw.append(0)
        prims.append((z0[k] + 1e-6, patches, fc, ec, lw))
    return prims


def _mesh_prims(mesh, camera, size, pt):
    """Lit triangles (two-sided, like the OpenGL faces) and edge strips just in front of them."""
    from matplotlib.patches import Polygon

    from .render import _rgb

    prims = []
    if not mesh.face_count:
        return prims
    v, f = mesh.vertices, mesh.faces
    vx, vy, vz, vs = _project(v, camera, size)
    _, right, up, fwd = _basis(camera)
    light = np.array([-0.4, 0.6, 1.0]) @ np.array([right, up, -fwd])  # same key light as the ray tracer
    light /= np.linalg.norm(light)
    vn, vc = mesh.vertex_properties.get("Normal"), mesh.vertex_properties.get("Color")
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]) if vn is None \
        else np.asarray(vn, float)[f].mean(1)  # smooth isosurface normals when given
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lit = 0.55 + 0.45 * np.abs(n @ light)
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
    """Line pieces as screen-space strips, ``width`` scaled with the image height, extended by half a
    width so corners close."""
    from matplotlib.patches import Polygon

    from .render import _rgb

    prims = []
    seg = lines.segments()
    if not len(seg):
        return prims
    cx, cy, cz, _ = _project(lines.points, camera, size)
    half = max(0.5, lines.width * size[1] / 1600)
    if lines.rgb is None:
        seg_rgb = np.tile(_rgb(ink), (len(seg), 1))
    else:
        seg_rgb = lines.rgb[np.concatenate([np.full(max(len(s) - 1, 0), k) for k, s in enumerate(lines.strips)])]
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
                  vectors=None, trajectory_width=2.0, visuals=None, frame=0, overlays=None, **_ignored):
    """Draw ``data`` as a vector figure and save it (format from the extension).

    ``shading``: 'gradient' (shaded discs) or 'flat' (one color plus outline, VESTA's
    flat style). ``dpi`` sets the physical page size (``size / dpi`` inches; 100 when
    not given). ``visuals``: extra visual elements (plugins), see :mod:`ewald.scene`;
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
                     trajectory_width=trajectory_width, show_cell=show_cell, cell_color=cell_color,
                     cell_width=cell_width)
    ink = cell_line_color(bg, transparent_background)
    prims = scene_prims(scene, camera, (w, h), pt, shading, ink)
    prims.sort(key=lambda p: -p[0])  # far first
    patches, fc, ec, lw = [], [], [], []
    for _, p, f, e, l in prims:
        patches += p; fc += f; ec += e; lw += l
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors=ec, linewidths=lw,
                                          match_original=False, antialiased=True))

    lay = overlay_layout(data, (w, h), text=text, legend=legend, type_legend=type_legend, scale_bar=scale_bar,
                         camera=camera, type_colors=type_colors, overlays=overlays, frame=frame,
                         background=background, transparent=transparent_background)
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
    if "colorbar" in lay:
        from .colormaps import get_colormap

        cb = lay["colorbar"]
        x0, y0, x1, y1 = cb["box"]
        cmap = get_colormap(cb["cmap"])
        n = 64
        stripes = [Rectangle((x0, y0 + (y1 - y0) * k / n), x1 - x0, (y1 - y0) / n + 0.5) for k in range(n)]
        ax.add_collection(PatchCollection(stripes, facecolors=[cmap(1 - (k + 0.5) / n) for k in range(n)],
                                          edgecolors="none", linewidths=0))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ink, linewidth=pt))
        ax.text(cb["title_x"], y0 - 0.4 * lay["font_size"], cb["title"], color=ink, fontsize=small, va="bottom", ha="left")
        for y, label in cb["ticks"]:
            ax.plot([x1, x1 + cb["tick_len"]], [y, y], color=ink, linewidth=pt)
            ax.text(cb["label_x"], y, label, color=ink, fontsize=small, va="center", ha="left")
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
