"""Offscreen / interactive rendering with VTK (via PyVista).

Atoms are drawn as sphere meshes for publication-quality shading, switching to
GPU sphere impostors (one vertex per atom) for very large systems.
"""
from __future__ import annotations

import numpy as np

from .data import DataCollection

_FALLBACK_PALETTE = np.array([
    [0.97, 0.38, 0.26], [0.25, 0.52, 0.93], [0.36, 0.78, 0.36], [0.98, 0.80, 0.20],
    [0.65, 0.40, 0.85], [0.20, 0.80, 0.80], [0.95, 0.55, 0.75], [0.60, 0.60, 0.60]])

_DIRECTIONS = {
    "x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0),
    "z": (0, 0, 1), "-z": (0, 0, -1), "top": (0, 0, 1), "front": (0, -1, 0),
    "iso": (1, 1, 1), "persp": (-1, -1, -1)
}


def _element(t: str):
    from ase.data import atomic_numbers

    return atomic_numbers.get(str(t))


def type_color(t: str, index: int = 0) -> np.ndarray:
    """Default color of a particle type: Jmol for elements, a palette otherwise."""
    from ase.data.colors import jmol_colors

    z = _element(t)
    return np.array(jmol_colors[z] if z else _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)], float)


def type_radius(t: str, scale: float = 0.8) -> float:
    from ase.data import covalent_radii

    z = _element(t)
    return float(covalent_radii[z] * scale) if z else 0.5


def default_colors(data: DataCollection, type_colors: dict | None = None) -> np.ndarray:
    """Per-particle RGB: the 'Color' property, else per-type colors (overrides first)."""
    if "Color" in data:
        return np.asarray(data["Color"], float)
    types = data.types
    out = np.zeros((data.count, 3))
    for k, t in enumerate(dict.fromkeys(types)):
        c = (type_colors or {}).get(t)
        out[types == t] = _rgb(c) if c is not None else type_color(t, k)
    return out


def default_radii(data: DataCollection, scale: float = 0.8, type_radii: dict | None = None) -> np.ndarray:
    if "Radius" in data:
        return np.asarray(data["Radius"], float)
    types = data.types
    out = np.full(data.count, 0.5)
    for t in dict.fromkeys(types):
        r = (type_radii or {}).get(t)
        out[types == t] = r if r is not None else type_radius(t, scale)
    return out


def _rgb(c) -> np.ndarray:
    if isinstance(c, str):
        from matplotlib.colors import to_rgb

        return np.array(to_rgb(c))
    c = np.asarray(c, float)[:3]
    return c / 255.0 if c.max() > 1 else c


def _cell_lines(cell):
    import pyvista as pv

    o, (a, b, c) = cell.origin, cell.matrix
    corners = np.array([o, o + a, o + b, o + c, o + a + b, o + a + c, o + b + c, o + a + b + c])
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    lines = np.hstack([[2, i, j] for i, j in edges])
    return pv.PolyData(corners, lines=lines)


def build_scene(plotter, data: DataCollection, *, radius_scale: float = 1.0, show_cell: bool = True,
                cell_color="black", cell_width: float = 2.0, bonds: float | None = None,
                bond_radius: float = 0.15, only_visible=None, atom_style: str = "auto",
                mesh_limit: int = 150_000, sphere_resolution: int | None = None,
                type_colors: dict | None = None, type_radii: dict | None = None,
                bond_pairs: list | None = None):
    """Add atoms (and optionally cell + bonds) of ``data`` to a PyVista plotter.

    ``atom_style``: 'mesh' (sphere meshes, publication shading), 'impostor' (GPU
    point sprites for very large systems) or 'auto' (mesh up to ``mesh_limit`` atoms).
    ``type_colors``/``type_radii`` override per-type defaults, e.g. ``{"Cu": "#c8794a"}``.
    ``bond_pairs``: VESTA-style list of ``(A, B, dmin, dmax)``; '*' matches any type.
    """
    import pyvista as pv

    if data.count:
        colors = np.clip(default_colors(data, type_colors), 0, 1)
        radii = default_radii(data, type_radii=type_radii) * radius_scale
        if only_visible is not None:
            keep = np.asarray(only_visible, bool)
        elif "Transparency" in data:
            keep = np.asarray(data["Transparency"]) < 1
        else:
            keep = slice(None)
        cloud = pv.PolyData(np.ascontiguousarray(data.positions[keep]))
        cloud["rgb"] = (colors[keep] * 255).astype(np.uint8)
        cloud["radius"] = radii[keep]
        if atom_style == "auto":
            atom_style = "mesh" if cloud.n_points <= mesh_limit else "impostor"
        if atom_style == "mesh":
            # Real sphere meshes: best shading for figures, fine up to ~1e5 atoms.
            res = sphere_resolution or (20 if cloud.n_points < 20_000 else 12)
            sphere = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res)
            atoms = cloud.glyph(geom=sphere, scale="radius", orient=False)
            plotter.add_mesh(atoms, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.25,
                             diffuse=0.8, specular=0.3, specular_power=40, name="atoms", render=False)
        else:
            # GPU sphere impostors: one vertex per atom, scales to millions of atoms.
            actor = plotter.add_mesh(cloud, style="points_gaussian", scalars="rgb", rgb=True,
                                     emissive=False, render_points_as_spheres=True, name="atoms", render=False)
            mapper = actor.GetMapper()
            mapper.SetScaleArray("radius")
            mapper.SetScaleFactor(1.0)
        if bonds or bond_pairs:
            _add_bonds(plotter, data, colors, bonds, bond_radius, bond_pairs)
    if show_cell and data.cell.is_valid:
        plotter.add_mesh(_cell_lines(data.cell), color=cell_color, line_width=cell_width, name="cell", render=False)


def bond_list(data, cutoff=None, pairs=None):
    """Bonds as (i, j, delta); both directions (i, j) and (j, i) are listed (half-bond drawing).

    With ``pairs`` [(A, B, dmin, dmax), ...] only pairs whose types and length
    match a row are kept (VESTA's bond search table); otherwise ``cutoff``.
    """
    from .neighbors import find_pairs

    rmax = max(p[3] for p in pairs) if pairs else cutoff
    i, j, d, dist = find_pairs(data.positions, data.cell, rmax)
    i, j, d, dist = (np.asarray(x) for x in (i, j, d, dist))
    if pairs:
        types = data.types
        ti, tj = types[i], types[j]
        keep = np.zeros(len(i), bool)
        for a, b, lo, hi in pairs:
            ok_t = (((ti == a) | (a == "*")) & ((tj == b) | (b == "*"))) | \
                   (((ti == b) | (b == "*")) & ((tj == a) | (a == "*")))
            keep |= ok_t & (dist >= lo) & (dist <= hi)
        i, j, d = i[keep], j[keep], d[keep]
    return i, j, d


def _add_bonds(plotter, data, colors, cutoff, bond_radius, pairs=None):
    """Half-bonds from each atom along the minimum-image vector (correct across PBC)."""
    import pyvista as pv

    i, j, d = bond_list(data, cutoff, pairs)
    if not len(i):
        return
    start = data.positions[i]
    pts = np.empty((2 * len(i), 3))
    pts[0::2] = start
    pts[1::2] = start + 0.5 * d
    lines = np.column_stack([np.full(len(i), 2), np.arange(0, 2 * len(i), 2),
                             np.arange(1, 2 * len(i), 2)]).ravel()
    poly = pv.PolyData(pts, lines=lines)
    poly.cell_data["rgb"] = (colors[i] * 255).astype(np.uint8)
    tubes = poly.tube(radius=bond_radius, n_sides=12, capping=False)
    plotter.add_mesh(tubes, scalars="rgb", rgb=True, name="bonds", smooth_shading=True, render=False,
                     ambient=0.25, diffuse=0.8, specular=0.2, specular_power=30)


def _setup_camera(plotter, data, direction, orthographic, zoom):
    d = np.asarray(_DIRECTIONS.get(direction, direction) if isinstance(direction, str) else direction, float)
    d = d / np.linalg.norm(d)
    if data.cell.is_valid:
        o, m = data.cell.origin, data.cell.matrix
        corners = o + np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ m
        pts = np.vstack([corners, data.positions]) if data.count else corners
    else:
        pts = data.positions
    center = 0.5 * (pts.min(0) + pts.max(0))
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    plotter.camera.focal_point = center
    plotter.camera.position = center + d * 10 * np.ptp(pts, 0).max()
    plotter.camera.up = up
    if orthographic:
        plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(zoom)


def camera_state(plotter) -> dict:
    """Serializable camera of a plotter (for GUI -> offscreen render and sessions)."""
    cam = plotter.camera
    return {"position": tuple(cam.position), "focal_point": tuple(cam.focal_point),
            "up": tuple(cam.up), "parallel_projection": bool(cam.parallel_projection),
            "parallel_scale": float(cam.parallel_scale), "view_angle": float(cam.view_angle)}


def apply_camera_state(plotter, state: dict):
    cam = plotter.camera
    cam.position = state["position"]
    cam.focal_point = state["focal_point"]
    cam.up = state["up"]
    if state.get("parallel_projection"):
        plotter.enable_parallel_projection()
    else:
        plotter.disable_parallel_projection()
    cam.parallel_scale = state.get("parallel_scale", cam.parallel_scale)
    cam.view_angle = state.get("view_angle", cam.view_angle)
    plotter.reset_camera_clipping_range()


def setup_camera(plotter, data, direction="iso", orthographic=True, zoom=1.0):
    _setup_camera(plotter, data, direction, orthographic, zoom)


def render(data: DataCollection, filename=None, *, size=(1600, 1200), direction="iso",
           orthographic: bool = True, zoom: float = 1.0, background="white",
           transparent_background: bool = False, ambient_occlusion: bool = False,
           legend: bool = True, show_cell: bool = True, bonds: float | None = None,
           radius_scale: float = 1.0, antialiasing: bool = True, text: str | None = None,
           camera: dict | None = None, renderer: str = "opengl", samples: int = 8, ao_samples: int = 16,
           shadows: bool = True, **scene_kwargs):
    """Render one DataCollection to ``filename`` (PNG/JPG/TIFF) or return an RGB array.

    ``direction`` is a view vector *from* the scene to the camera, or one of
    'x', 'y', 'z', '-x', ..., 'iso'. ``bonds=2.8`` draws bonds shorter than 2.8 A.
    ``camera`` (see :func:`camera_state`) reproduces an interactive view exactly.
    ``renderer="raytrace"`` uses the built-in ray tracer (ambient occlusion with
    ``ao_samples`` rays, ``shadows``, ``samples`` rays per pixel) instead of OpenGL.
    """
    import pyvista as pv

    if renderer == "raytrace":
        return _render_raytraced(data, filename, size=size, direction=direction, orthographic=orthographic,
                                 zoom=zoom, background=background, transparent_background=transparent_background,
                                 legend=legend, show_cell=show_cell, bonds=bonds, radius_scale=radius_scale,
                                 text=text, camera=camera, samples=samples, ao_samples=ao_samples,
                                 shadows=shadows, **scene_kwargs)
    if renderer != "opengl":
        raise ValueError(f"unknown renderer {renderer!r}; use 'opengl' or 'raytrace'")

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    p.set_background(background)
    build_scene(p, data, radius_scale=radius_scale, show_cell=show_cell, bonds=bonds, **scene_kwargs)
    if legend and "ColorCoding.range" in data.attributes:
        _add_colorbar(p, data)
    if text:
        p.add_text(text, font_size=12, color="black" if background == "white" else "white")
    if camera is not None:
        apply_camera_state(p, camera)
    else:
        _setup_camera(p, data, direction, orthographic, zoom)
    if antialiasing:
        try:
            p.enable_anti_aliasing("ssaa")
        except Exception:
            pass
    if ambient_occlusion:
        try:
            radius = float(np.median(default_radii(data))) if data.count else 1.0  # AO kernel ~ atom size
            p.enable_ssao(radius=radius, kernel_size=128)
        except Exception:
            pass
    img = p.screenshot(filename, transparent_background=transparent_background, return_img=True)
    p.close()
    return img


def _render_raytraced(data, filename, *, size, direction, orthographic, zoom, background, transparent_background,
                      legend, show_cell, bonds, radius_scale, text, camera, samples, ao_samples, shadows,
                      **scene_kwargs):
    import pyvista as pv

    from .raytrace import raytrace, scene_from_data

    scene_kwargs.pop("atom_style", None); scene_kwargs.pop("sphere_resolution", None)
    scene_kwargs.pop("mesh_limit", None); scene_kwargs.pop("antialiasing", None)
    scene_kwargs.pop("ambient_occlusion", None); scene_kwargs.pop("cell_width", None)
    if camera is None:  # same framing as the OpenGL path
        p = pv.Plotter(off_screen=True, window_size=tuple(size))
        build_scene(p, data, radius_scale=radius_scale, show_cell=show_cell, sphere_resolution=6,
                    **{k: v for k, v in scene_kwargs.items() if k in ("type_colors", "type_radii")})
        _setup_camera(p, data, direction, orthographic, zoom)
        camera = camera_state(p)
        p.close()
    cell_color = "#AEB4BE" if sum(_rgb(background)) < 1.5 and not transparent_background else "black"
    scene = scene_from_data(data, radius_scale=radius_scale, bonds=bonds, show_cell=show_cell,
                            cell_color=scene_kwargs.pop("cell_color", cell_color), **scene_kwargs)
    img = raytrace(scene, camera, size, samples=samples, ao_samples=ao_samples, shadows=shadows,
                   background=background, transparent_background=transparent_background)
    rgba = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    rgba = _draw_overlays(rgba, data, text if text else None, legend, background, transparent_background)
    if filename:
        from PIL import Image

        im = Image.fromarray(rgba, "RGBA")
        if not transparent_background and not str(filename).lower().endswith(".png"):
            im = im.convert("RGB")
        im.save(filename)
    return rgba[..., :3] if not transparent_background else rgba


def _draw_overlays(rgba, data, text, legend, background, transparent):
    """Text label and color legend for ray-traced images (PIL, sized relative to the image)."""
    if not text and not (legend and "ColorCoding.range" in data.attributes):
        return rgba
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    h, w = rgba.shape[:2]
    im = Image.fromarray(rgba, "RGBA")
    draw = ImageDraw.Draw(im)
    dark = not transparent and sum(_rgb(background)) < 1.5
    ink = (235, 235, 238, 255) if dark else (20, 20, 20, 255)
    fs = max(10, int(h * 0.03))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
        small = ImageFont.truetype("DejaVuSans.ttf", int(fs * 0.8))
    except OSError:
        font = small = ImageFont.load_default()
    if text:
        draw.text((int(0.02 * w), int(0.02 * h)), text, fill=ink, font=font)
    if legend and "ColorCoding.range" in data.attributes:
        lo, hi = data.attributes["ColorCoding.range"]
        cmap = matplotlib.colormaps[data.attributes.get("ColorCoding.cmap", "viridis")]
        x0, x1 = int(0.88 * w), int(0.91 * w)
        y0, y1 = int(0.15 * h), int(0.85 * h)
        for y in range(y0, y1):
            c = cmap(1 - (y - y0) / max(1, y1 - y0 - 1))
            draw.line([(x0, y), (x1, y)], fill=tuple(int(255 * v) for v in c[:3]) + (255,))
        draw.rectangle([x0, y0, x1, y1 - 1], outline=ink)
        title = str(data.attributes.get("ColorCoding.property", ""))
        draw.text((x0, y0 - int(1.4 * fs)), title, fill=ink, font=small)
        for frac in (0, 0.25, 0.5, 0.75, 1):
            y = int(y1 - 1 - frac * (y1 - y0 - 1))
            draw.line([(x1, y), (x1 + int(0.006 * w), y)], fill=ink)
            draw.text((x1 + int(0.01 * w), y - fs // 2), f"{lo + frac * (hi - lo):.3g}", fill=ink, font=small)
    return np.asarray(im)


def _add_colorbar(plotter, data, font_size=None):
    import pyvista as pv

    lo, hi = data.attributes["ColorCoding.range"]
    dummy = pv.PolyData(np.zeros((2, 3)))
    dummy["v"] = np.array([lo, hi])
    actor = plotter.add_mesh(dummy, scalars="v", cmap=data.attributes.get("ColorCoding.cmap", "viridis"),
                             clim=(lo, hi), opacity=0.0, show_scalar_bar=False, name="_legend")
    actor.VisibilityOff()
    plotter.add_scalar_bar(title=str(data.attributes.get("ColorCoding.property", "")), mapper=actor.mapper,
                           vertical=True, position_x=0.86, position_y=0.1, height=0.8, width=0.06,
                           color="black", fmt="%.3g",
                           **({"title_font_size": font_size + 2, "label_font_size": font_size} if font_size else {}))


def render_animation(pipeline, filename, *, frames=None, fps: int = 10, device=None, **kwargs):
    """Render frames of a pipeline into a movie (.mp4/.gif) via imageio."""
    import imageio.v2 as imageio

    idx = range(pipeline.num_frames) if frames is None else frames
    with imageio.get_writer(filename, fps=fps) as writer:
        for f in idx:
            writer.append_data(render(pipeline.compute(f, device=device), None, **kwargs))
