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
                bond_pairs: list | None = None, trajectory_width: float = 2.0):
    """Add atoms (and optionally cell + bonds) of ``data`` to a PyVista plotter.

    ``atom_style``: 'mesh' (sphere meshes, publication shading), 'impostor' (GPU
    point sprites for very large systems) or 'auto' (mesh up to ``mesh_limit`` atoms).
    ``type_colors``/``type_radii`` override per-type defaults, e.g. ``{"Cu": "#c8794a"}``.
    ``bond_pairs``: VESTA-style list of ``(A, B, dmin, dmax)``; '*' matches any type.
    Bonds in ``data.bonds`` (CreateBonds, EditBonds) are drawn whenever present and take
    precedence over ``bonds``/``bond_pairs``; ``data.surfaces`` meshes (polyhedra) are drawn too.
    ``trajectory_width``: line width of ``tables['trajectory-lines']`` (GenerateTrajectoryLines); 0 hides them.
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
            # VTK's built-in lit sprites have no ambient term (black rims), so shade them ourselves;
            # with custom splat code VTK skips its own sprite lighting.
            actor = plotter.add_mesh(cloud, style="points_gaussian", scalars="rgb", rgb=True,
                                     emissive=False, render_points_as_spheres=True, name="atoms",
                                     render=False)
            mapper = actor.GetMapper()
            mapper.SetScaleArray("radius")
            mapper.SetScaleFactor(1.0)
            mapper.SetSplatShaderCode(_IMPOSTOR_SHADER)
        if bonds or bond_pairs or data.bonds is not None:
            _add_bonds(plotter, data, colors, bonds, bond_radius, bond_pairs)
    _add_surfaces(plotter, data)
    if trajectory_width > 0 and "trajectory-lines" in data.tables:
        _add_trajectory_lines(plotter, data, trajectory_width)
    if show_cell and data.cell.is_valid:
        plotter.add_mesh(_cell_lines(data.cell), color=cell_color, line_width=cell_width, name="cell", render=False)


# Fragment code for sphere impostors: discard outside the disc, then Blinn-Phong with a
# fixed head light plus ambient, matching the mesh material (ambient 0.25, diffuse 0.8).
_IMPOSTOR_SHADER = """
//VTK::Color::Impl
float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
if (r2 > 1.0) { discard; }
vec3 n = vec3(offsetVCVSOutput.xy, sqrt(1.0 - r2));
vec3 l = normalize(vec3(-0.25, 0.35, 1.0));
float diff = max(dot(n, l), 0.0);
float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
diffuseColor = diffuseColor * (0.3 + 0.75 * diff) + vec3(0.3 * spec);
ambientColor = vec3(0.0);
"""


def bond_list(data, cutoff=None, pairs=None):
    """Bonds as (i, j, delta); both directions (i, j) and (j, i) are listed (half-bond drawing).

    With ``pairs`` [(A, B, dmin, dmax), ...] only pairs whose types and length
    match a row are kept (VESTA's bond search table); otherwise ``cutoff``.
    """
    from .bonds import find_bonds

    b = find_bonds(data, cutoff, pairs)
    i, j = b.topology.T
    d = b.vectors(data)
    return np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([d, -d])


def scene_bonds(data, cutoff=None, pairs=None):
    """The bonds a renderer draws: ``data.bonds`` if present, else a search by ``cutoff``/``pairs``."""
    from .bonds import find_bonds

    if data.bonds is not None:
        return data.bonds
    return find_bonds(data, cutoff, pairs)


def _add_bonds(plotter, data, colors, cutoff, bond_radius, pairs=None):
    """Half-bonds from each atom along its periodic-image bond vector (correct across PBC)."""
    import pyvista as pv

    from .bonds import bond_segments

    b = scene_bonds(data, cutoff, pairs)
    if not b.count:
        return
    start, end, rgb, _ = bond_segments(data, b, colors)
    m = len(start)
    pts = np.empty((2 * m, 3))
    pts[0::2] = start
    pts[1::2] = end
    lines = np.column_stack([np.full(m, 2), np.arange(0, 2 * m, 2), np.arange(1, 2 * m, 2)]).ravel()
    poly = pv.PolyData(pts, lines=lines)
    poly.cell_data["rgb"] = (rgb * 255).astype(np.uint8)
    tubes = poly.tube(radius=bond_radius, n_sides=12, capping=False)
    plotter.add_mesh(tubes, scalars="rgb", rgb=True, name="bonds", smooth_shading=True, render=False,
                     ambient=0.25, diffuse=0.8, specular=0.2, specular_power=30)


def _add_surfaces(plotter, data):
    """Every ``data.surfaces`` mesh as one actor (faces) plus its outline edges."""
    import pyvista as pv

    for name, mesh in getattr(data, "surfaces", {}).items():
        if not mesh.face_count:
            continue
        poly = mesh.to_pyvista()
        poly.cell_data["rgb"] = (np.clip(mesh.face_colors(), 0, 1) * 255).astype(np.uint8)
        plotter.add_mesh(poly, scalars="rgb", rgb=True, opacity=1.0 - mesh.transparency, name=f"surface:{name}",
                         ambient=0.3, diffuse=0.7, specular=0.2, render=False, smooth_shading=False)
        if mesh.edges is not None and len(mesh.edges):
            lines = np.column_stack([np.full(len(mesh.edges), 2), mesh.edges]).ravel()
            plotter.add_mesh(pv.PolyData(mesh.vertices, lines=lines), color=_rgb(mesh.edge_color), line_width=1.5,
                             name=f"surface-edges:{name}", render=False)


def _add_trajectory_lines(plotter, data, width):
    """Polylines through ``tables['trajectory-lines']``, one per particle, in its atom color."""
    import pyvista as pv

    table = data.tables["trajectory-lines"]
    pts = np.asarray(table["Position"], dtype=float)
    ids = np.asarray(table["Particle Identifier"])
    if len(pts) < 2:
        return
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    sizes = np.diff(np.r_[starts, len(ids)])
    lines = np.concatenate([np.r_[m, np.arange(s, s + m)] for s, m in zip(starts, sizes)])
    poly = pv.PolyData(pts, lines=lines)
    colors = np.tile([0.3, 0.3, 0.3], (len(starts), 1))
    if data.count:
        here = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)
        order = np.argsort(here)
        pos = np.clip(np.searchsorted(here[order], ids[starts]), 0, data.count - 1)
        hit = here[order][pos] == ids[starts]
        colors[hit] = np.clip(default_colors(data), 0, 1)[order][pos[hit]]
    poly.cell_data["rgb"] = (colors * 255).astype(np.uint8)
    plotter.add_mesh(poly, scalars="rgb", rgb=True, line_width=width, name="trajectories", render=False)


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


VECTOR_SUFFIXES = (".svg", ".pdf", ".eps", ".ps")

# Output quality presets (file output puts quality first; the interactive viewport has its own settings).
# samples: ray-traced rays per pixel; ao_samples: ambient-occlusion rays; antialiasing: OpenGL SSAA;
# ambient_occlusion: OpenGL SSAO; sphere_resolution: OpenGL sphere mesh detail; scale: pixel size factor.
QUALITY_PRESETS = {
    "draft": {"samples": 1, "ao_samples": 0, "shadows": False, "antialiasing": False,
              "ambient_occlusion": False, "sphere_resolution": 8, "scale": 0.5},
    "standard": {"samples": 4, "ao_samples": 8, "shadows": True, "antialiasing": True,
                 "ambient_occlusion": False, "sphere_resolution": 16, "scale": 1.0},
    "publication": {"samples": 16, "ao_samples": 32, "shadows": True, "antialiasing": True,
                    "ambient_occlusion": True, "sphere_resolution": 28, "scale": 1.0},
}


def quality_settings(quality: str | None = "publication", **explicit) -> dict:
    """Preset values with every explicitly given (non-None) setting taking precedence."""
    name = (quality or "publication").lower()
    if name not in QUALITY_PRESETS:
        raise ValueError(f"unknown quality {quality!r}; use one of {', '.join(QUALITY_PRESETS)}")
    out = dict(QUALITY_PRESETS[name])
    out.update({k: v for k, v in explicit.items() if v is not None})
    return out


def render(data: DataCollection, filename=None, *, size=(1600, 1200), direction="iso",
           orthographic: bool = True, zoom: float = 1.0, background="white",
           transparent_background: bool = False, ambient_occlusion: bool | None = None,
           legend: bool = True, show_cell: bool = True, bonds: float | None = None,
           radius_scale: float = 1.0, antialiasing: bool | None = None, text: str | None = None,
           camera: dict | None = None, renderer: str = "opengl", samples: int | None = None,
           ao_samples: int | None = None, shadows: bool | None = None, type_legend: bool = False,
           scale_bar: bool | float | None = None, dpi: float | None = None, quality: str = "publication",
           scale: float | None = None, **scene_kwargs):
    """Render one DataCollection to ``filename`` or return an RGB array.

    ``direction`` is a view vector *from* the scene to the camera, or one of
    'x', 'y', 'z', '-x', ..., 'iso'. ``bonds=2.8`` draws bonds shorter than 2.8 A.
    ``camera`` (see :func:`camera_state`) reproduces an interactive view exactly.

    ``renderer``: 'opengl' (VTK, fast), 'raytrace' (built-in ray tracer: ambient
    occlusion with ``ao_samples`` rays, ``shadows``, ``samples`` rays per pixel) or
    'vector' (SVG/PDF/EPS with real circles and editable text, see :mod:`ewald.vector`).
    Raster renderers can also write .pdf/.svg/.eps, embedding the image.

    ``quality``: 'draft', 'standard' or 'publication' (default) fills in every sampling
    setting left as None (``samples``, ``ao_samples``, ``shadows``, ``antialiasing``,
    ``ambient_occlusion``, sphere mesh detail and ``scale``, a factor on ``size``);
    see ``QUALITY_PRESETS``. Explicit arguments always win.

    Overlays: ``text`` label, ``legend`` (ColorCoding color bar), ``type_legend``
    (one swatch per particle type), ``scale_bar`` (True = round length, or a length
    in Angstrom; exact in orthographic views). ``dpi`` is stored in PNG/TIFF files
    and sets the page size of vector output.
    """
    import pyvista as pv

    if renderer not in ("opengl", "raytrace", "vector"):
        raise ValueError(f"unknown renderer {renderer!r}; use 'opengl', 'raytrace' or 'vector'")
    q = quality_settings(quality, samples=samples, ao_samples=ao_samples, shadows=shadows,
                         antialiasing=antialiasing, ambient_occlusion=ambient_occlusion,
                         sphere_resolution=scene_kwargs.pop("sphere_resolution", None), scale=scale)
    samples, ao_samples, shadows = q["samples"], q["ao_samples"], q["shadows"]
    antialiasing, ambient_occlusion = q["antialiasing"], q["ambient_occlusion"]
    if q["scale"] != 1.0:
        size = (max(16, round(size[0] * q["scale"])), max(16, round(size[1] * q["scale"])))
    overlays = {"text": text, "legend": legend, "type_legend": type_legend, "scale_bar": scale_bar,
                "type_colors": scene_kwargs.get("type_colors")}
    if renderer == "vector":
        from .vector import render_vector

        if camera is None:
            camera = default_camera(data, size, direction, orthographic, zoom, radius_scale, scene_kwargs)
        for k in ("atom_style", "sphere_resolution", "mesh_limit", "trajectory_width"):
            scene_kwargs.pop(k, None)
        return render_vector(data, filename, camera=camera, size=size, background=background,
                             transparent_background=transparent_background, radius_scale=radius_scale,
                             bonds=bonds, show_cell=show_cell, dpi=dpi,
                             **{k: v for k, v in overlays.items() if k != "type_colors"}, **scene_kwargs)
    if renderer == "raytrace":
        return _render_raytraced(data, filename, size=size, direction=direction, orthographic=orthographic,
                                 zoom=zoom, background=background, transparent_background=transparent_background,
                                 show_cell=show_cell, bonds=bonds, radius_scale=radius_scale, camera=camera,
                                 samples=samples, ao_samples=ao_samples, shadows=shadows, overlays=overlays,
                                 dpi=dpi, **scene_kwargs)

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    p.set_background(background)
    res = q["sphere_resolution"] if data.count <= 20_000 else min(q["sphere_resolution"], 16)
    build_scene(p, data, radius_scale=radius_scale, show_cell=show_cell, bonds=bonds, sphere_resolution=res,
                **scene_kwargs)
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
    img = p.screenshot(None, transparent_background=transparent_background, return_img=True)
    cam = camera_state(p)
    p.close()
    return _finish(img, data, filename, cam, overlays, background, transparent_background, dpi)


def default_camera(data, size=(1600, 1200), direction="iso", orthographic=True, zoom=1.0, radius_scale=1.0,
                   scene_kwargs=None) -> dict:
    """Camera that frames ``data`` like the OpenGL renderer (no window is shown)."""
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    build_scene(p, data, radius_scale=radius_scale, show_cell=True, sphere_resolution=6,
                **{k: v for k, v in (scene_kwargs or {}).items() if k in ("type_colors", "type_radii")})
    _setup_camera(p, data, direction, orthographic, zoom)
    cam = camera_state(p)
    p.close()
    return cam


def _finish(img, data, filename, camera, overlays, background, transparent, dpi):
    """Draw overlays on a rendered raster image, save it, return RGB (or RGBA if transparent)."""
    from .overlays import draw_overlays

    rgba = draw_overlays(np.asarray(img, np.uint8), data, camera=camera, background=background,
                         transparent=transparent, **overlays)
    if filename:
        save_image(rgba, filename, transparent=transparent, dpi=dpi)
    return rgba if transparent else rgba[..., :3]


def save_image(rgba, filename, *, transparent=False, dpi=None):
    """Save an RGB(A) uint8 image; .pdf/.svg/.eps embed it in a vector page of the same size."""
    from PIL import Image

    rgba = np.asarray(rgba, np.uint8)
    name = str(filename)
    if name.lower().endswith(VECTOR_SUFFIXES):
        from matplotlib.figure import Figure

        d = float(dpi or 100)
        fig = Figure(figsize=(rgba.shape[1] / d, rgba.shape[0] / d), dpi=d)
        fig.figimage(rgba if transparent else rgba[..., :3], resize=False)
        fig.savefig(name, dpi=d, transparent=transparent)
        return
    im = Image.fromarray(rgba)
    if not (transparent and name.lower().endswith((".png", ".tif", ".tiff", ".webp"))):
        im = im.convert("RGB")
    kw = {"dpi": (float(dpi), float(dpi))} if dpi and name.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")) else {}
    im.save(name, **kw)


def _render_raytraced(data, filename, *, size, direction, orthographic, zoom, background, transparent_background,
                      show_cell, bonds, radius_scale, camera, samples, ao_samples, shadows, overlays, dpi,
                      **scene_kwargs):
    from .raytrace import raytrace, scene_from_data

    for k in ("atom_style", "sphere_resolution", "mesh_limit", "antialiasing", "ambient_occlusion", "cell_width"):
        scene_kwargs.pop(k, None)
    if camera is None:  # same framing as the OpenGL path
        camera = default_camera(data, size, direction, orthographic, zoom, radius_scale, scene_kwargs)
    cell_color = "#AEB4BE" if sum(_rgb(background)) < 1.5 and not transparent_background else "black"
    scene = scene_from_data(data, radius_scale=radius_scale, bonds=bonds, show_cell=show_cell,
                            cell_color=scene_kwargs.pop("cell_color", cell_color), **scene_kwargs)
    img = raytrace(scene, camera, size, samples=samples, ao_samples=ao_samples, shadows=shadows,
                   background=background, transparent_background=transparent_background)
    rgba = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return _finish(rgba, data, filename, camera, overlays, background, transparent_background, dpi)


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


def render_animation(pipeline, filename, *, frames=None, fps: float = 10, device=None, text=None,
                     time_per_frame: float | None = None, progress=None, **kwargs):
    """Render frames of a pipeline into a movie; see :func:`ewald.movie.render_animation`."""
    from .movie import render_animation as _render_animation

    return _render_animation(pipeline, filename, frames=frames, fps=fps, device=device, text=text,
                             time_per_frame=time_per_frame, progress=progress, **kwargs)
