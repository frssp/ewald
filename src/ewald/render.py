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


def default_colors(data: DataCollection) -> np.ndarray:
    from ase.data.colors import jmol_colors

    if "Color" in data:
        return np.asarray(data["Color"], float)
    types = data.types
    out = np.zeros((data.count, 3))
    for k, t in enumerate(dict.fromkeys(types)):
        z = _element(t)
        out[types == t] = jmol_colors[z] if z else _FALLBACK_PALETTE[k % len(_FALLBACK_PALETTE)]
    return out


def default_radii(data: DataCollection, scale: float = 0.8) -> np.ndarray:
    from ase.data import covalent_radii

    if "Radius" in data:
        return np.asarray(data["Radius"], float)
    types = data.types
    out = np.full(data.count, 0.5)
    for t in dict.fromkeys(types):
        z = _element(t)
        if z:
            out[types == t] = covalent_radii[z] * scale
    return out


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
                mesh_limit: int = 150_000, sphere_resolution: int | None = None):
    """Add atoms (and optionally cell + bonds) of ``data`` to a PyVista plotter.

    ``atom_style``: 'mesh' (sphere meshes, publication shading), 'impostor' (GPU
    point sprites for very large systems) or 'auto' (mesh up to ``mesh_limit`` atoms).
    """
    import pyvista as pv

    if data.count:
        colors = np.clip(default_colors(data), 0, 1)
        radii = default_radii(data) * radius_scale
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
        if bonds:
            _add_bonds(plotter, data, colors, bonds, bond_radius)
    if show_cell and data.cell.is_valid:
        plotter.add_mesh(_cell_lines(data.cell), color=cell_color, line_width=cell_width, name="cell", render=False)


def _add_bonds(plotter, data, colors, cutoff, bond_radius):
    """Half-bonds from each atom along the minimum-image vector (correct across PBC)."""
    import pyvista as pv

    from .neighbors import find_pairs

    i, j, d, _ = find_pairs(data.positions, data.cell, cutoff)
    i, d = np.asarray(i), np.asarray(d)
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
           camera: dict | None = None, **scene_kwargs):
    """Render one DataCollection to ``filename`` (PNG/JPG/TIFF) or return an RGB array.

    ``direction`` is a view vector *from* the scene to the camera, or one of
    'x', 'y', 'z', '-x', ..., 'iso'. ``bonds=2.8`` draws bonds shorter than 2.8 A.
    ``camera`` (see :func:`camera_state`) reproduces an interactive view exactly.
    """
    import pyvista as pv

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
            radius = float(np.median(default_radii(data))) if data.count else 1.0
            p.enable_ssao(radius=radius, kernel_size=128)
        except Exception:
            pass
    img = p.screenshot(filename, transparent_background=transparent_background, return_img=True)
    p.close()
    return img


def _add_colorbar(plotter, data):
    import pyvista as pv

    lo, hi = data.attributes["ColorCoding.range"]
    dummy = pv.PolyData(np.zeros((2, 3)))
    dummy["v"] = np.array([lo, hi])
    actor = plotter.add_mesh(dummy, scalars="v", cmap=data.attributes.get("ColorCoding.cmap", "viridis"),
                             clim=(lo, hi), opacity=0.0, show_scalar_bar=False, name="_legend")
    actor.VisibilityOff()
    plotter.add_scalar_bar(title=str(data.attributes.get("ColorCoding.property", "")), mapper=actor.mapper,
                           vertical=True, position_x=0.86, position_y=0.1, height=0.8, width=0.06,
                           color="black", fmt="%.3g")


def render_animation(pipeline, filename, *, frames=None, fps: int = 10, device=None, **kwargs):
    """Render frames of a pipeline into a movie (.mp4/.gif) via imageio."""
    import imageio.v2 as imageio

    idx = range(pipeline.num_frames) if frames is None else frames
    with imageio.get_writer(filename, fps=fps) as writer:
        for f in idx:
            writer.append_data(render(pipeline.compute(f, device=device), None, **kwargs))
