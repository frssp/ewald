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


def render_vector(data, filename=None, *, camera, size=(1600, 1200), background="white",
                  transparent_background=False, radius_scale=1.0, type_colors=None, type_radii=None,
                  bonds=None, bond_pairs=None, bond_radius=0.15, show_cell=True, cell_color=None,
                  cell_width=2.0, shading="gradient", text=None, legend=True, type_legend=False,
                  scale_bar=None, dpi=None, radius_preset="covalent", oxidation_states=None, coordination=None,
                  **_ignored):
    """Draw ``data`` as a vector figure and save it (format from the extension).

    ``shading``: 'gradient' (shaded discs) or 'flat' (one color plus outline, VESTA's
    flat style). ``dpi`` sets the physical page size (``size / dpi`` inches; 100 when
    not given). Returns the rasterized image as an RGB(A) uint8 array.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon, Rectangle

    from .overlays import ink_color, overlay_layout
    from .bonds import bond_segments
    from .render import _rgb, default_colors, default_radii, scene_bonds

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

    prims = []  # (depth, patches, facecolors, edgecolors, linewidths)
    if data.count:
        colors = np.clip(default_colors(data, type_colors), 0, 1)
        radii = default_radii(data, type_radii=type_radii, radius_preset=radius_preset,
                              oxidation_states=oxidation_states, coordination=coordination) * radius_scale
        alpha = 1 - np.clip(np.asarray(data["Transparency"], float), 0, 1) if "Transparency" in data \
            else np.ones(data.count)
        keep = np.flatnonzero(alpha > 0)
        sx, sy, z, s = _project(data.positions[keep], camera, (w, h))
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
        if bonds or bond_pairs or data.bonds is not None:
            b = scene_bonds(data, bonds, bond_pairs)
            seg_start, seg_end, seg_rgb, _ = bond_segments(data, b, colors) if b.count else ([], [], [], [])
            if len(seg_start):
                i_, j_ = b.topology.T
                bi = np.concatenate([i_, j_])  # atom each half-bond starts at
                keep_b = alpha[bi] > 0
                bi, seg_start, seg_end, seg_rgb = bi[keep_b], seg_start[keep_b], seg_end[keep_b], seg_rgb[keep_b]
                ax0, ay0, z0, s0 = _project(seg_start, camera, (w, h))
                ax1, ay1, z1, s1 = _project(seg_end, camera, (w, h))
                for k, i in enumerate(bi):
                    dx, dy = ax1[k] - ax0[k], ay1[k] - ay0[k]
                    length = math.hypot(dx, dy)
                    if length < 1e-6:
                        continue
                    nx, ny = -dy / length, dx / length
                    c, a = seg_rgb[k], alpha[i]
                    r0, r1 = bond_radius * s0[k], bond_radius * s1[k]
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
                    # paint just before the bond's own atom, so the sphere hides the stick's root
                    prims.append((z0[k] + 1e-6, patches, fc, ec, lw))
    for mesh in getattr(data, "surfaces", {}).values():  # polyhedra and other triangle meshes
        if not mesh.face_count:
            continue
        v, f = mesh.vertices, mesh.faces
        vx, vy, vz, vs = _project(v, camera, (w, h))
        _, right, up, fwd = _basis(camera)
        light = np.array([-0.4, 0.6, 1.0]) @ np.array([right, up, -fwd])  # same key light as the ray tracer
        light /= np.linalg.norm(light)
        vn, vc = mesh.vertex_properties.get("Normal"), mesh.vertex_properties.get("Color")
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]) if vn is None \
            else np.asarray(vn, float)[f].mean(1)  # smooth isosurface normals when given
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        lit = 0.55 + 0.45 * np.abs(n @ light)  # two-sided, like the OpenGL faces
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
                # just in front of the faces it borders
                prims.append((0.5 * (vz[i] + vz[j]) - 1e-3, [Polygon(quad, closed=True)], [(*ec, 1.0)], ["none"], [0]))
    if show_cell and data.cell.is_valid:
        if cell_color is None:
            cell_color = "#AEB4BE" if sum(bg) < 1.5 and not transparent_background else "black"
        o, (a_, b_, c_) = data.cell.origin, data.cell.matrix
        corners = np.array([o, o + a_, o + b_, o + c_, o + a_ + b_, o + a_ + c_, o + b_ + c_, o + a_ + b_ + c_])
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        cx, cy, cz, _ = _project(corners, camera, (w, h))
        half = max(0.5, cell_width * h / 1600)
        cc = _rgb(cell_color)
        for i, j in edges:
            dx, dy = cx[j] - cx[i], cy[j] - cy[i]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            nx, ny = -dy / length * half, dx / length * half
            # extend by half a width so corners close
            ex, ey = dx / length * half, dy / length * half
            quad = [(cx[i] - ex + nx, cy[i] - ey + ny), (cx[j] + ex + nx, cy[j] + ey + ny),
                    (cx[j] + ex - nx, cy[j] + ey - ny), (cx[i] - ex - nx, cy[i] - ey - ny)]
            prims.append((0.5 * (cz[i] + cz[j]), [Polygon(quad, closed=True)], [(*cc, 1.0)], ["none"], [0]))

    prims.sort(key=lambda p: -p[0])  # far first
    patches, fc, ec, lw = [], [], [], []
    for _, p, f, e, l in prims:
        patches += p; fc += f; ec += e; lw += l
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors=ec, linewidths=lw,
                                          match_original=False, antialiased=True))

    lay = overlay_layout(data, (w, h), text=text, legend=legend, type_legend=type_legend, scale_bar=scale_bar,
                         camera=camera, type_colors=type_colors)
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
        import matplotlib

        cb = lay["colorbar"]
        x0, y0, x1, y1 = cb["box"]
        cmap = matplotlib.colormaps[cb["cmap"]]
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

    if filename:
        fig.savefig(filename, dpi=dpi, transparent=transparent_background, facecolor=fig.get_facecolor())
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba()).copy()
    return img if transparent_background else img[..., :3]
