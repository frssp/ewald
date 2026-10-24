"""Offscreen / interactive rendering with VTK (via PyVista).

Atoms are drawn as sphere meshes for publication-quality shading, switching to
GPU sphere impostors (one vertex per atom) for very large systems.
"""
from __future__ import annotations

import numpy as np

from .colormaps import to_rgb
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

    from .radii import element_of

    return atomic_numbers.get(element_of(t))


_RADIUS_KEYS = ("type_radii", "radius_preset", "oxidation_states", "coordination")

# The key light of every renderer, in the camera frame (x right, y up, z towards the viewer): upper left,
# in front. The ray tracer and vector output use it as is; OpenGL keeps VTK's light kit unless a ``light``
# is given (the impostor shader has its own head light, IMPOSTOR_LIGHT).
DEFAULT_LIGHT = (-0.4, 0.6, 1.0)
IMPOSTOR_LIGHT = (-0.25, 0.35, 1.0, 1.0)
LIGHT_KEYS = ("direction", "azimuth", "elevation", "intensity")


def light_params(light) -> tuple[np.ndarray, float]:
    """``(direction, intensity)`` of a ``light`` setting: None (the default key light), a direction
    ``[x, y, z]`` in the camera frame (x right, y up, z towards the viewer) or a dict with ``direction`` or
    ``azimuth``/``elevation`` (degrees; azimuth turns the light right of the viewer, elevation up) and
    ``intensity`` (1 = default; scales diffuse and specular light, not the ambient part)."""
    import math

    if light is None:
        light = {}
    elif not isinstance(light, dict):
        light = {"direction": light}
    unknown = set(light) - set(LIGHT_KEYS)
    if unknown:
        raise KeyError(f"light: unknown {sorted(unknown)}; use {', '.join(LIGHT_KEYS)}")
    if "direction" in light and ("azimuth" in light or "elevation" in light):
        raise ValueError("light: give direction or azimuth/elevation, not both")
    if "azimuth" in light or "elevation" in light:
        d0 = np.asarray(DEFAULT_LIGHT, float)
        az = math.radians(float(light.get("azimuth", math.degrees(math.atan2(d0[0], d0[2])))))
        el = math.radians(float(light.get("elevation", math.degrees(math.asin(d0[1] / np.linalg.norm(d0))))))
        d = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    else:
        d = np.asarray(light.get("direction", DEFAULT_LIGHT), float)
    n = np.linalg.norm(d) if d.shape == (3,) else 0.0
    if not n or not np.isfinite(n):
        raise ValueError(f"light direction must be a nonzero [x, y, z], not {light.get('direction')!r}")
    k = float(light.get("intensity", 1.0))
    if not k >= 0:
        raise ValueError(f"light intensity must be >= 0, not {k}")
    return d / n, k


def apply_light(plotter, light=None):
    """Light a PyVista plotter: ``light`` (see :func:`light_params`) replaces its lights with a camera key
    light plus a weak head light (fill), and sets the impostor shader's light; None puts VTK's light kit
    back if a light was set before."""
    import pyvista as pv

    ren = plotter.renderer
    if light is None:
        if getattr(ren, "_ewald_light", None) is not None:
            plotter.remove_all_lights()
            plotter.enable_lightkit()
            ren._ewald_light = None
        return
    d, k = light_params(light)
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=tuple(d), focal_point=(0, 0, 0), light_type="camera light", intensity=k))
    plotter.add_light(pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), light_type="camera light",
                               intensity=0.3 * k))
    ren._ewald_light = (*d, k)


def type_color(t: str, index: int = 0) -> np.ndarray:
    """Default color of a particle type: Jmol for elements, a palette otherwise."""
    from ase.data.colors import jmol_colors

    z = _element(t)
    return np.array(jmol_colors[z] if z else _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)], float)


def type_radius(t: str, scale: float = 0.8, preset: str = "covalent") -> float:
    """Default radius of a particle type: ``scale`` x the ``preset`` radius (see :mod:`ewald.radii`)."""
    from .radii import element_radius

    r = element_radius(str(t), preset)
    return float(r * scale) if r is not None else 0.5


_last_groups: tuple = (None, None)  # (types array, result): one scene build asks several times


def type_groups(types, max_scan: int = 32):
    """Distinct types in order of first appearance and each particle's index into them.

    One vectorized comparison per type: ~20x faster than ``dict.fromkeys`` or
    ``np.unique`` on millions of string labels; many types fall back to ``np.unique``.
    The last result is kept for the same array object (arrays of a DataCollection are never
    changed in place), since radii, colors and the scene all group the same frame's types.
    """
    global _last_groups
    if isinstance(types, np.ndarray) and _last_groups[0] is types and max_scan == 32:
        names, inverse = _last_groups[1]
        return list(names), inverse
    out = _type_groups(np.asarray(types), max_scan)
    if isinstance(types, np.ndarray) and max_scan == 32:
        _last_groups = (types, (list(out[0]), out[1]))
    return out


def _type_groups(types, max_scan):
    inverse = np.zeros(len(types), dtype=np.int32)
    names: list = []
    left = np.ones(len(types), bool)
    while len(types) and len(names) < max_scan:
        i = int(np.argmax(left))
        if not left[i]:
            break
        match = types == types[i]
        inverse[match] = len(names)
        names.append(types[i].item())
        left &= ~match
    rest = np.flatnonzero(left)
    if not len(rest):
        return names, inverse
    uniq, first, inv = np.unique(types[rest], return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)
    inverse[rest] = len(names) + rank[inv.ravel()]
    names.extend(uniq[order].tolist())
    return names, inverse


def _color_table(names, type_colors=None) -> np.ndarray:
    return np.array([_rgb(c) if (c := (type_colors or {}).get(t)) is not None else type_color(t, k)
                     for k, t in enumerate(names)], float).reshape(-1, 3)


def _radius_table(names, scale=0.8, type_radii=None, data=None, radius_preset="covalent",
                  oxidation_states=None, coordination=None) -> np.ndarray:
    """Radius per type name: ``type_radii`` first, else ``scale`` x the preset radius (0.5 if unknown)."""
    type_radii = type_radii or {}
    table = {}
    if data is not None and any(type_radii.get(t) is None for t in names):
        from .radii import resolve

        table = resolve(data, radius_preset, oxidation_states, coordination).radii
    return np.array([r if (r := type_radii.get(t)) is not None
                     else (table[str(t)] * scale if str(t) in table else 0.5) for t in names], float)


def default_colors(data: DataCollection, type_colors: dict | None = None) -> np.ndarray:
    """Per-particle RGB: the 'Color' property, else per-type colors (overrides first)."""
    if "Color" in data:
        return np.asarray(data["Color"], float)
    names, inverse = type_groups(data.types)
    return _color_table(names, type_colors)[inverse]


def default_radii(data: DataCollection, scale: float = 0.8, type_radii: dict | None = None,
                  radius_preset: str = "covalent", oxidation_states: dict | None = None,
                  coordination: dict | None = None) -> np.ndarray:
    """Per-particle radii: the 'Radius' property, else ``type_radii``, else ``scale`` x the preset radius.

    ``radius_preset`` is one of :data:`ewald.radii.PRESETS` ('covalent', 'atomic', 'vdW', 'ionic', ...);
    the ionic presets use ``oxidation_states``/``coordination`` per type, or guess them.
    """
    if "Radius" in data:
        return np.asarray(data["Radius"], float)
    names, inverse = type_groups(data.types)
    return _radius_table(names, scale, type_radii, data, radius_preset, oxidation_states, coordination)[inverse]


_rgb = to_rgb  # many modules import it from here


def atom_cloud(data: DataCollection, *, radius_scale: float = 1.0, only_visible=None,
               type_colors: dict | None = None, type_radii: dict | None = None,
               radius_preset: str = "covalent", oxidation_states: dict | None = None,
               coordination: dict | None = None):
    """Visible atoms as a ``pv.PolyData`` with 'rgb' and 'radius' point arrays, plus the
    per-particle colors of all atoms (bonds reuse them)."""
    from .scene import ParticlesVis, SceneContext

    ctx = SceneContext(only_visible=only_visible)
    ((_, spheres),) = ParticlesVis(radius_scale, type_colors, type_radii, radius_preset, oxidation_states,
                                   coordination).build(data, ctx)
    return spheres_cloud(spheres), ctx.particle_colors


def spheres_cloud(spheres):
    """A :class:`ewald.scene.Spheres` as a ``pv.PolyData`` with 'rgb' (uint8) and 'radius' point arrays."""
    import pyvista as pv

    cloud = pv.PolyData(np.ascontiguousarray(spheres.centers))
    if spheres.rgb is None:  # per-type table: one conversion per type, not per atom
        cloud["rgb"] = (np.clip(spheres.palette, 0, 1) * 255).astype(np.uint8)[spheres.palette_index]
    else:
        cloud["rgb"] = (np.clip(spheres.rgb, 0, 1) * 255).astype(np.uint8)
    if (spheres.alpha < 1).any():  # see-through spheres: RGBA scalars
        cloud["rgb"] = np.column_stack([cloud["rgb"], (np.clip(spheres.alpha, 0, 1) * 255).astype(np.uint8)])
    cloud["radius"] = spheres.radii if spheres.radii is not None else spheres.radius_table[spheres.palette_index]
    return cloud


_SOFTWARE_GL = ("llvmpipe", "softpipe", "swrast", "software rasterizer", "microsoft basic render", "swiftshader")


def opengl_info(render_window) -> dict:
    """The OpenGL driver a (rendered) VTK window uses: 'vendor', 'renderer', 'version' and
    'software' (True for Mesa llvmpipe and other CPU rasterizers, which are slow)."""
    info = {"vendor": "", "renderer": "", "version": ""}
    for line in render_window.ReportCapabilities().splitlines():
        key, _, value = line.partition(" string:")
        if key.startswith("OpenGL ") and key[7:] in info:
            info[key[7:]] = value.strip()
    info["software"] = any(s in info["renderer"].lower() for s in _SOFTWARE_GL)
    return info


def _split_by_alpha(cloud):
    """``[(cloud, name suffix)]``: when some spheres are see-through and some are not, the
    opaque ones (RGB) and, as ``":glass"``, the see-through ones (RGBA); else the cloud as is."""
    import pyvista as pv

    rgb = cloud["rgb"]
    if rgb.shape[1] == 3 or (rgb[:, 3] < 255).all():
        return [(cloud, "")]
    out = []
    for mask, suffix, cols in ((rgb[:, 3] == 255, "", 3), (rgb[:, 3] < 255, ":glass", 4)):
        if mask.any():
            part = pv.PolyData(np.ascontiguousarray(cloud.points[mask]))
            part["rgb"] = np.ascontiguousarray(rgb[mask, :cols])
            part["radius"] = cloud["radius"][mask]
            out.append((part, suffix))
    return out


def needs_depth_peeling(scene) -> bool:
    """Whether an OpenGL scene has see-through layers (meshes, spheres or arrows)."""
    for layer in scene.layers:
        p = layer.primitive
        if (layer.kind == "mesh" and p.transparency > 0) or (layer.kind == "spheres" and (p.alpha < 1).any()) \
                or (layer.kind == "arrows" and p.alpha < 1):
            return True
    return False


def add_atoms(plotter, cloud, *, atom_style: str = "auto", mesh_limit: int = 150_000,
              sphere_resolution: int | None = None, name: str = "atoms"):
    """Add an :func:`atom_cloud` as sphere meshes or GPU impostors; returns the actor."""
    import pyvista as pv

    if atom_style == "auto":
        atom_style = "mesh" if cloud.n_points <= mesh_limit else "impostor"
    if atom_style == "mesh":
        # Real sphere meshes: best shading for figures, fine up to ~1e5 atoms.
        res = sphere_resolution or (20 if cloud.n_points < 20_000 else 12)
        sphere = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res)
        atoms = cloud.glyph(geom=sphere, scale="radius", orient=False)
        return plotter.add_mesh(atoms, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.25,
                                diffuse=0.8, specular=0.3, specular_power=40, name=name, render=False)
    # GPU sphere impostors: one vertex per atom, scales to millions of atoms.
    # VTK's built-in lit sprites have no ambient term (black rims), so shade them ourselves;
    # with custom splat code VTK skips its own sprite lighting.
    actor = plotter.add_mesh(cloud, style="points_gaussian", scalars="rgb", rgb=True,
                             emissive=False, render_points_as_spheres=True, name=name,
                             render=False)
    mapper = actor.GetMapper()
    mapper.SetScaleArray("radius")
    mapper.SetScaleFactor(1.0)
    mapper.SetSplatShaderCode(_FLAT_SHADER if atom_style == "flat" else _IMPOSTOR_SHADER)
    if atom_style == "flat":
        actor.GetProperty().LightingOff()  # the shader's color is final
    ren = plotter.renderer
    if not getattr(ren, "_impostor_observer", False):
        ren.AddObserver("StartEvent", _impostor_camera)
        ren._impostor_observer = True
    _impostor_camera(ren)
    return actor


def _impostor_camera(renderer, *_):
    """Pass the projection to every impostor actor of ``renderer`` (on each render), so their
    fragments get the depth of the sphere surface."""
    cam = renderer.GetActiveCamera()
    m = cam.GetProjectionTransformMatrix(renderer)
    width = renderer.GetSize()[0]
    proj = [m.GetElement(0, 0) * width / 2, m.GetElement(2, 2), m.GetElement(2, 3),
            0.0 if cam.GetParallelProjection() else 1.0]
    light = list(getattr(renderer, "_ewald_light", None) or IMPOSTOR_LIGHT)
    actors = renderer.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        mapper = actor.GetMapper()
        code = mapper.GetSplatShaderCode() if hasattr(mapper, "GetSplatShaderCode") else None
        if code in (_IMPOSTOR_SHADER, _FLAT_SHADER):
            uniforms = actor.GetShaderProperty().GetFragmentCustomUniforms()
            uniforms.SetUniform4f("impostorProj", proj)
            if code == _IMPOSTOR_SHADER:
                uniforms.SetUniform4f("impostorLight", light)


def build_scene(plotter, data: DataCollection, *, radius_scale: float = 1.0, show_cell: bool = True,
                cell_color="black", cell_width: float = 2.0, bonds: float | None = None,
                bond_radius: float = 0.15, only_visible=None, atom_style: str = "auto",
                mesh_limit: int = 150_000, sphere_resolution: int | None = None,
                type_colors: dict | None = None, type_radii: dict | None = None,
                bond_pairs: list | None = None, trajectory_width: float = 2.0, trajectory_color_by=None,
                trajectory_colormap: str = "viridis", trajectory_color_range=None, trajectory_time_range=None,
                trajectory_trail=None, trajectory_radius=None, radius_preset: str = "covalent", oxidation_states: dict | None = None,
                coordination: dict | None = None, vectors=None, arrow_resolution: int | None = None,
                visuals=None, frame: int = 0, boundary=None, boundary_bonds: str = "half",
                replicate=None, replicate_cell: str = "super", show_atoms: bool = True, surfaces=None,
                cell_radius=None, type_transparency=None, selection_color=None, light=None, measurements=None):
    """Add atoms (and optionally cell + bonds) of ``data`` to a PyVista plotter; returns the
    :class:`ewald.scene.Scene` it drew.

    ``atom_style``: 'mesh' (sphere meshes, publication shading), 'impostor' (GPU
    point sprites for very large systems), 'auto' (mesh up to ``mesh_limit`` atoms) or 'flat'
    (ASE-GUI look: unshaded discs with a dark outline and unshaded bonds; impostors, so fast at any size).
    ``type_colors``/``type_radii`` override per-type defaults, e.g. ``{"Cu": "#c8794a"}``.
    ``radius_preset`` picks the default radii: 'covalent' (default), 'atomic', 'atomic (calc.)', 'vdW',
    'vdW (Bondi)', 'ionic' or 'ionic (crystal)' (Shannon; ``oxidation_states={"Ti": 4, "O": -2}`` and
    ``coordination={"Ti": 6}`` per type, else guessed from the structure; see :mod:`ewald.radii`).
    ``bond_pairs``: VESTA-style list of ``(A, B, dmin, dmax)``; '*' matches any type.
    Bonds in ``data.bonds`` (CreateBonds, EditBonds) are drawn whenever present and take
    precedence over ``bonds``/``bond_pairs``; ``data.surfaces`` meshes (polyhedra) are drawn too.
    ``trajectory_width``: line width of ``tables['trajectory-lines']`` (GenerateTrajectoryLines); 0 hides them.
    ``trajectory_color_by`` ('Time', 'Frame', 'Step Length', a table column or a particle property) colors
    them with ``trajectory_colormap`` over ``trajectory_color_range``; ``trajectory_time_range`` ``(t0, t1)``
    and ``trajectory_trail`` (True: up to the current frame, a number: that much Time before it) cut them;
    ``trajectory_radius`` sizes ray-traced tubes (see :class:`ewald.scene.TrajectoryLinesVis`).
    ``vectors``: particle properties to draw as arrows on top of ``data.vectors`` (VectorVis), e.g.
    ``"Force"``, ``["Force", "Velocity"]`` or ``{"Force": {"scaling": 2, "color": "red"}}`` (see
    :class:`ewald.arrows.VectorStyle`). ``arrow_resolution``: sides per arrow (default from the
    sphere detail); above ``mesh_limit`` arrows they are drawn as lines.
    ``visuals``: extra visual elements (plugins), see :mod:`ewald.scene`.
    ``boundary``: VESTA-style boundary range in fractional coordinates, e.g. ``(-0.1, 1.1)``: also
    draws the periodic images inside it, with cut bonds and polyhedra completed (display only;
    ``boundary_bonds`` 'half', 'complete' or 'none', see :mod:`ewald.boundary`).
    ``replicate``: ``(na, nb, nc)`` periodic copies of the cell drawn side by side (display only;
    ``replicate_cell`` 'super', 'unit' or 'all' picks the cell lines, see :mod:`ewald.replicate`).
    ``show_atoms=False`` leaves the atoms out (bonds keep their colors); ``surfaces`` restyles
    ``data.surfaces`` by name (``{name: {"visible", "color", "transparency", "edges"}}``).
    ``type_transparency`` ``{type: 0..1}`` fades whole types; ``selection_color`` draws the particles of
    a 'Selection' property in that color (OVITO's red). ``light`` sets the plotter's key light (see
    :func:`apply_light`); None keeps VTK's light kit. ``measurements`` draws the lines of distances, angles
    and dihedrals (:mod:`ewald.measure`; the values are an overlay).
    """
    from .scene import SceneContext, build

    scene = build(data, ctx=SceneContext(frame=frame, only_visible=only_visible), visuals=visuals,
                  radius_scale=radius_scale, type_colors=type_colors, type_radii=type_radii,
                  radius_preset=radius_preset, oxidation_states=oxidation_states, coordination=coordination,
                  bonds=bonds, bond_pairs=bond_pairs, bond_radius=bond_radius, vectors=vectors,
                  trajectory_width=trajectory_width, trajectory_color_by=trajectory_color_by,
                  trajectory_colormap=trajectory_colormap, trajectory_color_range=trajectory_color_range,
                  trajectory_time_range=trajectory_time_range, trajectory_trail=trajectory_trail,
                  trajectory_radius=trajectory_radius, show_cell=show_cell, cell_width=cell_width,
                  boundary=boundary, boundary_bonds=boundary_bonds, replicate=replicate,
                  replicate_cell=replicate_cell, show_atoms=show_atoms, surfaces=surfaces, cell_radius=cell_radius,
                  type_transparency=type_transparency, selection_color=selection_color, measurements=measurements)
    draw_scene(plotter, scene, atom_style=atom_style, mesh_limit=mesh_limit, sphere_resolution=sphere_resolution,
               arrow_resolution=arrow_resolution or max(6, (sphere_resolution or 16) // 2 + 2), ink=cell_color)
    apply_light(plotter, light)
    return scene


def draw_scene(plotter, scene, *, atom_style: str = "auto", mesh_limit: int = 150_000,
               sphere_resolution: int | None = None, arrow_resolution: int = 10, ink="black"):
    """Add every layer of an :class:`ewald.scene.Scene` to a PyVista plotter, one actor per layer
    named after it. ``ink`` colors lines that have no color of their own (the cell)."""
    for layer in scene.layers:
        prim, name = layer.primitive, layer.name
        if layer.kind == "spheres":
            if prim.count:
                # see-through spheres get their own actor: RGBA on the opaque ones too spoils their shading
                for part, suffix in _split_by_alpha(spheres_cloud(prim)):
                    style = atom_style
                    if style == "impostor" and part["rgb"].shape[1] == 4 and part.n_points <= mesh_limit:
                        style = "mesh"  # see-through impostors vanish under depth peeling on macOS
                    add_atoms(plotter, part, atom_style=style, mesh_limit=mesh_limit,
                              sphere_resolution=sphere_resolution, name=name + suffix)
        elif layer.kind == "cylinders":
            flat = atom_style == "flat"
            impostor = not flat and (atom_style == "impostor" or (atom_style == "auto" and prim.count > mesh_limit))
            _add_cylinders(plotter, prim, name, flat=flat, impostor=impostor)
        elif layer.kind == "mesh":
            _add_mesh(plotter, prim, name)
        elif layer.kind == "arrows":
            _add_arrows(plotter, prim, name, arrow_resolution, mesh_limit)
        elif layer.kind == "lines":
            _add_lines(plotter, prim, name, ink)


# Fragment code for sphere impostors: discard outside the disc, then Blinn-Phong with a
# key light ``impostorLight`` (camera frame direction, intensity; IMPOSTOR_LIGHT or the plotter's light)
# plus ambient, matching the mesh material (ambient 0.25, diffuse 0.8).
# The sprite sits at the sphere center; moving each fragment's depth to the sphere surface makes
# overlapping spheres intersect and bonds enter them like real meshes. The world radius comes from
# the sprite's size in pixels (screen derivative of the offset) and ``impostorProj`` =
# (pixels per eye unit at unit depth or in ortho, P22, P23, perspective) from _impostor_camera.
_IMPOSTOR_SHADER = """
//VTK::Color::Impl
float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
if (r2 > 1.0) { discard; }
float nz = sqrt(1.0 - r2);
vec3 n = vec3(offsetVCVSOutput.xy, nz);
vec3 l = normalize(impostorLight.xyz);
float diff = max(dot(n, l), 0.0);
float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
diffuseColor = diffuseColor * (0.3 + 0.75 * impostorLight.w * diff) + vec3(0.3 * impostorLight.w * spec);
ambientColor = vec3(0.0);
float rpix = 1.0 / max(length(vec2(dFdx(offsetVCVSOutput.x), dFdy(offsetVCVSOutput.x))), 1e-6);
float zn = 2.0 * gl_FragCoord.z - 1.0;
bool persp = impostorProj.w > 0.5;
float ze = persp ? -impostorProj.z / (zn + impostorProj.y) : (zn - impostorProj.z) / impostorProj.y;
ze += nz * rpix / (persp ? impostorProj.x / -ze : impostorProj.x);
zn = persp ? (impostorProj.y * ze + impostorProj.z) / -ze : impostorProj.y * ze + impostorProj.z;
gl_FragDepth = 0.5 * zn + 0.5;
"""

# The 'flat' style (ASE GUI): the atom's own color, unlit, inside a dark outline (1 pixel or 5% of
# the radius, antialiased), with the same sphere-surface depth as _IMPOSTOR_SHADER.
_FLAT_SHADER = _IMPOSTOR_SHADER.replace("""vec3 l = normalize(impostorLight.xyz);
float diff = max(dot(n, l), 0.0);
float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
diffuseColor = diffuseColor * (0.3 + 0.75 * impostorLight.w * diff) + vec3(0.3 * impostorLight.w * spec);
ambientColor = vec3(0.0);
float rpix = 1.0 / max(length(vec2(dFdx(offsetVCVSOutput.x), dFdy(offsetVCVSOutput.x))), 1e-6);
""", """float rpix = 1.0 / max(length(vec2(dFdx(offsetVCVSOutput.x), dFdy(offsetVCVSOutput.x))), 1e-6);
float edge = clamp((1.0 - sqrt(r2)) * rpix - max(0.8, 0.05 * rpix), 0.0, 1.0);
diffuseColor = mix(vec3(0.08), diffuseColor, edge);
ambientColor = vec3(0.0);
""")


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
    """The bonds a renderer draws: ``data.bonds`` if present, else a search by ``cutoff``/``pairs``.
    The see-through copies of ``OverlayFrames`` ('Overlay Frame' >= 0) get no bonds: they sit a
    fraction of a bond length from their atoms and would bond to every neighbour of it."""
    from .bonds import Bonds, find_bonds

    if data.bonds is not None:
        return data.bonds
    if "Overlay Frame" in data:
        real = np.flatnonzero(np.asarray(data["Overlay Frame"]) < 0)
        if len(real) < data.count:
            b = find_bonds(data.take(real), cutoff, pairs)
            return Bonds(real[b.topology], b.pbc_images) if b.count else b
    return find_bonds(data, cutoff, pairs)


TUBE_SIDES = 12


def tube_mesh(start, end, radius: float, sides: int = TUBE_SIDES):
    """Open cylinders from ``start`` to ``end`` as triangles: ``(verts, faces, normals)``, ``sides``
    vertices around each end (segment ``i`` owns vertices ``2*sides*i ...``). Vectorized numpy, about
    ten times faster than ``vtkTubeFilter``, which made bonds most of a frame's build time."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    k = len(start)
    axis = end - start
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    t = np.divide(axis, length, out=np.tile([0.0, 0.0, 1.0], (k, 1)), where=length > 0)
    ref = np.zeros_like(t)
    ref[np.arange(k), np.argmin(np.abs(t), axis=1)] = 1.0  # the axis least parallel to t
    u = np.cross(t, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(t, u)
    ang = 2 * np.pi * np.arange(sides) / sides
    normals = np.empty((k, 2, sides, 3))
    np.multiply(np.cos(ang)[:, None], u[:, None], out=normals[:, 0])
    normals[:, 0] += np.sin(ang)[:, None] * v[:, None]
    normals[:, 1] = normals[:, 0]
    verts = radius * normals
    verts[:, 0] += start[:, None]
    verts[:, 1] += end[:, None]
    a = np.arange(sides)
    b = (a + 1) % sides
    quad = np.concatenate([np.column_stack([a, b, a + sides]), np.column_stack([b, b + sides, a + sides])])
    faces = (quad[None] + (2 * sides * np.arange(k))[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), faces, normals.reshape(-1, 3)


CAP_RINGS = 3  # latitude rings of a round (hemisphere) cap mesh, pole excluded


def cap_mesh(center, axis, radius, kind: str, sides: int = TUBE_SIDES):
    """Caps closing tube ends as triangles: ``(verts, faces, normals)``, one block of vertices per
    cap. ``axis`` points out of the tube; ``kind`` is "flat" (a disk) or "round" (a hemisphere).
    The rim matches :func:`tube_mesh`'s vertices, so caps close the tubes without cracks."""
    center, axis = np.asarray(center, float).reshape(-1, 3), np.asarray(axis, float).reshape(-1, 3)
    k = len(center)
    ref = np.zeros_like(axis)
    ref[np.arange(k), np.argmin(np.abs(axis), axis=1)] = 1.0
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axis, u)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = np.cos(ang)[None, :, None] * u[:, None] + np.sin(ang)[None, :, None] * v[:, None]   # k, sides, 3
    a = np.arange(sides)
    b = (a + 1) % sides
    if kind == "flat":
        normals = np.repeat(axis[:, None], sides + 1, axis=1)
        verts = np.concatenate([radius[:, None, None] * ring, np.zeros((k, 1, 3))], axis=1) + center[:, None]
        tri = np.column_stack([np.full(sides, sides), a, b])
    else:
        lat = 0.5 * np.pi * np.arange(CAP_RINGS) / CAP_RINGS
        normals = np.concatenate([(np.cos(lat)[None, :, None, None] * ring[:, None]
                                   + np.sin(lat)[None, :, None, None] * axis[:, None, None]).reshape(k, -1, 3),
                                  axis[:, None]], axis=1)
        verts = radius[:, None, None] * normals + center[:, None]
        pole = CAP_RINGS * sides
        tri = [np.column_stack([a + j * sides, b + j * sides, b + (j + 1) * sides]) for j in range(CAP_RINGS - 1)]
        tri += [np.column_stack([a + j * sides, b + (j + 1) * sides, a + (j + 1) * sides]) for j in range(CAP_RINGS - 1)]
        tri.append(np.column_stack([a + (CAP_RINGS - 1) * sides, b + (CAP_RINGS - 1) * sides, np.full(sides, pole)]))
        tri = np.concatenate(tri)
    per = verts.shape[1]
    faces = (tri[None] + (per * np.arange(k))[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), faces, normals.reshape(-1, 3)


def _tube_caps(cyl, pick, sides: int = TUBE_SIDES):
    """Flat and round cap meshes of cylinders ``pick`` (one radius): ``(verts, faces, normals, rgb)``
    blocks to append to their tubes."""
    out = []
    mask = np.zeros(cyl.count, bool)
    mask[pick] = True
    for kind in ("flat", "round"):
        center, axis, radius, rgb, _, idx = cyl.caps(kind)
        sel = mask[idx]
        if sel.any():
            verts, faces, normals = cap_mesh(center[sel], axis[sel], radius[sel], kind, sides)
            out.append((verts, faces, normals, np.repeat(rgb[sel], len(verts) // int(sel.sum()), axis=0)))
    return out


def _round_cap_spheres(cyl):
    """Round caps as spheres of the cylinder radius (for GPU stick impostors, whose ends are flat)."""
    from .scene import Spheres

    center, _, radius, rgb, alpha, _ = cyl.caps("round")
    return Spheres(center, radius, rgb, alpha) if len(center) else None


def _add_cylinders(plotter, cyl, name, flat: bool = False, impostor: bool = False):
    """Cylinders as tubes (half-bonds: one per atom, in its color) with their caps; ``flat``: unshaded
    (ASE-GUI look); ``impostor``: GPU ray-cast cylinders (:func:`add_sticks`), one vertex each
    instead of a tube, plus sphere impostors for round caps."""
    import pyvista as pv

    if not cyl.count:
        return
    if impostor:
        add_sticks(plotter, cyl, name)
        caps = _round_cap_spheres(cyl)
        if caps is not None:
            add_atoms(plotter, spheres_cloud(caps), atom_style="impostor", name=f"{name}:caps")
        return
    radii = np.unique(cyl.radius)
    for r in radii:  # one tube actor per radius (bonds have one)
        pick = np.flatnonzero(cyl.radius == r) if len(radii) > 1 else np.arange(cyl.count)
        verts, faces, normals = tube_mesh(cyl.start[pick], cyl.end[pick], float(r))
        rgb = np.repeat(cyl.rgb[pick], 2 * TUBE_SIDES, axis=0)
        parts = [(verts, faces, normals, rgb)] + _tube_caps(cyl, pick)
        if len(parts) > 1:
            offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
            verts = np.concatenate([p[0] for p in parts])
            faces = np.concatenate([p[1] + o for p, o in zip(parts, offsets)])
            normals = np.concatenate([p[2] for p in parts])
            rgb = np.concatenate([p[3] for p in parts])
        tubes = pv.PolyData.from_regular_faces(verts, faces)
        tubes.point_data["Normals"] = normals
        tubes.point_data.active_normals_name = "Normals"
        tubes.point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        light = dict(ambient=1.0, diffuse=0.0, specular=0.0) if flat else _TUBE_LIGHT
        actor_name = name if len(radii) == 1 else f"{name}:{r:g}"
        plotter.add_mesh(tubes, scalars="rgb", rgb=True, name=actor_name, smooth_shading=not flat, render=False,
                         **light)
        if flat:
            _add_stick_outline(plotter, cyl.start[pick], cyl.end[pick], float(r), f"{actor_name}:outline")


_TUBE_LIGHT = dict(ambient=0.25, diffuse=0.8, specular=0.2, specular_power=30)


def stick_data(cyl):
    """The input of :func:`add_sticks`: one float32 point per cylinder at its middle, with its unit
    axis ("orient"), (length, radius, radius) ("scale", what ``vtkOpenGLStickMapper`` reads), its
    8-bit color ("rgb") and its index ("selection"). The mapper reads that index during picking
    and crashes without it, and only a ``vtkIdTypeArray`` will do (an int64 array from numpy is a
    ``vtkLongLongArray``)."""
    import pyvista as pv
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import VTK_ID_TYPE

    start, end = cyl.start.astype(np.float32), cyl.end.astype(np.float32)
    axis = end - start
    length = np.sqrt(np.einsum("ij,ij->i", axis, axis))
    unit = np.divide(axis, length[:, None], out=np.tile(np.float32([0, 0, 1]), (cyl.count, 1)), where=length[:, None] > 0)
    middle = start + end
    middle *= 0.5
    sticks = pv.PolyData(middle)
    sticks.point_data["orient"] = unit
    sticks.point_data["scale"] = np.column_stack([length, cyl.radius, cyl.radius]).astype(np.float32)
    sticks.point_data["rgb"] = (np.clip(cyl.rgb, 0, 1) * 255).astype(np.uint8)
    ids = numpy_to_vtk(np.arange(cyl.count, dtype=np.int64), deep=True, array_type=VTK_ID_TYPE)
    ids.SetName("selection")
    sticks.GetPointData().AddArray(ids)
    return sticks


def add_sticks(plotter, cyl, name):
    """Cylinders as GPU impostors (Sigg et al. 2006): VTK's stick mapper, the one its molecule
    mapper draws bonds with, ray-casts each cylinder in the fragment shader from one point, so a
    million half-bonds build in about a second and draw without a triangle each. Both ends are
    flat caps (round ones are added as spheres by :func:`_add_cylinders`); the material matches
    the tubes'."""
    from vtkmodules.vtkRenderingCore import vtkActor
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLStickMapper

    mapper = vtkOpenGLStickMapper()
    mapper.SetInputData(stick_data(cyl))
    mapper.SetOrientationArray("orient")
    mapper.SetScaleArray("scale")
    mapper.SetSelectionIdArray("selection")
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("rgb")
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()
    actor = vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetAmbient(_TUBE_LIGHT["ambient"])
    prop.SetDiffuse(_TUBE_LIGHT["diffuse"])
    prop.SetSpecular(_TUBE_LIGHT["specular"])
    prop.SetSpecularPower(_TUBE_LIGHT["specular_power"])
    plotter.add_actor(actor, name=name, render=False, reset_camera=False)
    return actor


STICK_OUTLINE = 0.3  # outline width of flat sticks, as a fraction of the stick radius


def _add_stick_outline(plotter, start, end, radius, name):
    """Dark outline of flat sticks (ASE GUI look) as an inverted hull: a slightly wider tube that
    shows only its back faces, so a dark rim appears around each stick from any direction. The mesh
    is static and drawn by the GPU; ``vtkPolyDataSilhouette`` recomputed the edges on the CPU on
    every camera move, which made the whole style several times slower to rotate."""
    import pyvista as pv

    verts, faces, _ = tube_mesh(start, end, radius * (1.0 + STICK_OUTLINE))
    hull = pv.PolyData.from_regular_faces(verts, faces)
    actor = plotter.add_mesh(hull, color=(0.08, 0.08, 0.08), name=name, render=False, lighting=False)
    actor.GetProperty().FrontfaceCullingOn()  # tube_mesh faces wind outward


def _add_mesh(plotter, mesh, name):
    """A triangle mesh (faces) plus its outline edges. Meshes with vertex "Normal"s (isosurfaces) are
    smooth-shaded; vertex "Color"s (grid slices) are interpolated."""
    import pyvista as pv

    if not mesh.face_count:
        return
    poly = mesh.to_pyvista()
    kw = dict(opacity=1.0 - mesh.transparency, name=name, ambient=0.3, diffuse=0.7,
              specular=0.2, render=False, smooth_shading=False)
    normals = mesh.vertex_properties.get("Normal")
    if normals is not None:
        poly.point_data["Normals"] = np.asarray(normals, float)
        poly.point_data.active_normals_name = "Normals"
        kw.update(smooth_shading=True, specular=0.25, specular_power=30)
    vcolors = mesh.vertex_properties.get("Color")
    if vcolors is not None:
        poly.point_data["rgb"] = (np.clip(vcolors, 0, 1) * 255).astype(np.uint8)
        kw.update(ambient=0.6, diffuse=0.45, specular=0.0)  # colour maps should read true, not shaded
    else:
        poly.cell_data["rgb"] = (np.clip(mesh.face_colors(), 0, 1) * 255).astype(np.uint8)
    plotter.add_mesh(poly, scalars="rgb", rgb=True, **kw)
    if mesh.edges is not None and len(mesh.edges):
        lines = np.column_stack([np.full(len(mesh.edges), 2), mesh.edges]).ravel()
        edges = name.replace("surface:", "surface-edges:", 1) if name.startswith("surface:") else f"{name}:edges"
        plotter.add_mesh(pv.PolyData(mesh.vertices, lines=lines), color=_rgb(mesh.edge_color), line_width=1.5,
                         name=edges, render=False)


def _add_arrows(plotter, arrows, name, sides, line_limit):
    """One arrow set: cylinder + cone meshes, or colored lines above ``line_limit`` arrows so the
    interactive viewport stays fast."""
    import pyvista as pv

    from .arrows import arrow_mesh

    m = arrows.count
    if not m:
        return
    flat = arrows.shading == "flat"
    light = dict(ambient=1.0, diffuse=0.0, specular=0.0) if flat else \
        dict(ambient=0.25, diffuse=0.8, specular=0.3, specular_power=40)
    if m > line_limit:
        pts = np.empty((2 * m, 3))
        pts[0::2], pts[1::2] = arrows.base, arrows.tip
        lines = np.column_stack([np.full(m, 2), np.arange(0, 2 * m, 2), np.arange(1, 2 * m, 2)]).ravel()
        poly = pv.PolyData(pts, lines=lines)
        poly.cell_data["rgb"] = (arrows.rgb * 255).astype(np.uint8)
        plotter.add_mesh(poly, scalars="rgb", rgb=True, line_width=2, opacity=arrows.alpha,
                         name=name, render=False, lighting=False)
        return
    verts, faces, normals, owner = arrow_mesh(arrows, sides)
    poly = pv.PolyData(verts, faces=np.column_stack([np.full(len(faces), 3), faces]).ravel())
    poly.point_data["Normals"] = normals
    poly.point_data.active_normals_name = "Normals"
    poly.point_data["rgb"] = (arrows.rgb[owner] * 255).astype(np.uint8)
    plotter.add_mesh(poly, scalars="rgb", rgb=True, opacity=arrows.alpha, smooth_shading=not flat,
                     name=name, render=False, **light)


def _add_lines(plotter, lines, name, ink):
    """Polylines ``width`` pixels wide, each in its color (or ``ink``)."""
    import pyvista as pv

    if not lines.count or len(lines.points) < 2:
        return
    cells = np.concatenate([np.r_[len(s), s] for s in lines.strips])
    poly = pv.PolyData(lines.points, lines=cells)
    if lines.point_rgb is not None:   # a color map along the lines, blended along each piece
        poly.point_data["rgb"] = (np.clip(lines.point_rgb, 0, 1) * 255).astype(np.uint8)
        plotter.add_mesh(poly, scalars="rgb", rgb=True, line_width=lines.width, name=name, render=False)
        return
    if lines.rgb is None:
        plotter.add_mesh(poly, color=ink, line_width=lines.width, name=name, render=False)
        return
    poly.cell_data["rgb"] = (np.clip(lines.rgb, 0, 1) * 255).astype(np.uint8)
    plotter.add_mesh(poly, scalars="rgb", rgb=True, line_width=lines.width, name=name, render=False)


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
    plotter.reset_camera(render=False)
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
RENDERERS = {
    "opengl": "VTK, fast; SSAO and anti-aliasing; GPU impostors for large systems",
    "raytrace": "built-in numba ray tracer: ambient occlusion, shadows, anti-aliasing (publication figures)",
    "vector": "SVG/PDF/EPS with editable shapes and text (matplotlib)",
}

# Output quality presets (file output puts quality first; the interactive viewport has its own settings).
# samples: ray-traced rays per pixel (the most a pixel takes with adaptive sampling); ao_samples:
# ambient-occlusion rays; noise_threshold: ray tracer adaptive sampling, pixels whose standard error is
# below it stop early (0 = off; see ewald.raytrace.raytrace); antialiasing: OpenGL SSAA;
# ambient_occlusion: OpenGL SSAO; sphere_resolution: OpenGL sphere mesh detail; scale: pixel size factor.
QUALITY_PRESETS = {
    "draft": {"samples": 1, "ao_samples": 0, "noise_threshold": 0.0, "shadows": False, "antialiasing": False,
              "ambient_occlusion": False, "sphere_resolution": 8, "scale": 0.5},
    "standard": {"samples": 4, "ao_samples": 4, "noise_threshold": 0.0, "shadows": True, "antialiasing": True,
                 "ambient_occlusion": False, "sphere_resolution": 16, "scale": 1.0},
    "publication": {"samples": 16, "ao_samples": 8, "noise_threshold": 0.01, "shadows": True, "antialiasing": True,
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
           orthographic: bool = True, zoom: float = 1.0, background=None,
           transparent_background: bool = False, ambient_occlusion: bool | None = None,
           legend: bool = True, show_cell: bool = True, bonds: float | None = None,
           radius_scale: float = 1.0, antialiasing: bool | None = None, text: str | None = None,
           camera: dict | None = None, renderer: str = "opengl", samples: int | None = None,
           ao_samples: int | None = None, noise_threshold: float | None = None, shadows: bool | None = None,
           type_legend: bool = False,
           scale_bar: bool | float | None = None, dpi: float | None = None, quality: str = "publication",
           scale: float | None = None, theme=None, overlays=None, region=None, labels=None, light=None,
           progress=None, device=None, **scene_kwargs):
    """Render one DataCollection to ``filename`` or return an RGB array.

    ``direction`` is a view vector *from* the scene to the camera, or one of
    'x', 'y', 'z', '-x', ..., 'iso'. ``bonds=2.8`` draws bonds shorter than 2.8 A.
    ``camera`` (see :func:`camera_state`) reproduces an interactive view exactly.
    ``region=(x, y, height)`` renders only that part of the view: the GUI Render frame, as fractions of
    the view height (see :mod:`ewald.framing`); None shows the whole view height, centered.

    ``renderer``: 'opengl' (VTK, fast), 'raytrace' (built-in ray tracer: ambient
    occlusion with ``ao_samples`` rays, ``shadows``, ``samples`` rays per pixel, of which pixels
    that converge below ``noise_threshold`` take fewer: adaptive sampling, 0 = off) or
    'vector' (SVG/PDF/EPS with real circles and editable text, see :mod:`ewald.vector`).
    Raster renderers can also write .pdf/.svg/.eps, embedding the image.

    ``quality``: 'draft', 'standard' or 'publication' (default) fills in every sampling
    setting left as None (``samples``, ``ao_samples``, ``noise_threshold``, ``shadows``, ``antialiasing``,
    ``ambient_occlusion``, sphere mesh detail and ``scale``, a factor on ``size``);
    see ``QUALITY_PRESETS``. Explicit arguments always win.

    Overlays: ``text`` label, ``legend`` (the atoms' color bar or category swatches), ``type_legend``
    (one swatch per particle type), ``scale_bar`` (True = round length, or a length
    in Angstrom; exact in orthographic views), and ``overlays``: overlay elements
    (Coordinate tripod, Attributes, Atom labels, plugins; see :mod:`ewald.overlays`). ``labels`` puts a
    label on each atom: True (its type), a template such as ``"{type}{index}"`` or ``"{Charge:.2f}"``, or a
    dict with ``text``, ``types``, ``size``, ``color`` (:class:`ewald.overlays.AtomLabels`). ``measurements``
    ``[[i, j], [i, j, k], [i, j, k, l]]`` draws distances, angles and dihedrals with their values
    (:mod:`ewald.measure`). ``dpi`` is stored
    in PNG/TIFF files and sets the page size of vector output.

    ``boundary``: VESTA-style boundary range, e.g. ``(-0.1, 1.1)`` (fractional; or three pairs): also
    draws the periodic images of atoms inside it and completes bonds and polyhedra cut by the cell;
    display only (``boundary_bonds``: 'half', 'complete' or 'none'; see :mod:`ewald.boundary`).
    ``replicate=(2, 2, 2)`` draws that many periodic copies of the cell (OVITO's Replicate as a
    display setting; ``replicate_cell`` 'super', 'unit' or 'all'); the data stays unchanged.
    ``show_atoms=False`` leaves the atoms out (a stick model); ``surfaces={"coordination-polyhedra":
    {"visible": True, "color": "teal", "transparency": 0.5, "edges": False}}`` restyles ``data.surfaces``
    meshes by name without changing their step. ``type_transparency={"O": 0.6}`` fades whole types (1
    hides them); ``selection_color="red"`` highlights a 'Selection' property, as OVITO does.
    ``color_by`` recolors the atoms for the picture only: 'type' (type colors, whatever the steps wrote) or a
    property/expression mapped through ``colormap`` over ``color_range`` (see :func:`ewald.color_rule.apply`).

    ``light``: the key light, the same in every renderer: a direction ``[x, y, z]`` in the camera frame
    (x right, y up, z towards the viewer) or ``{"direction" or "azimuth"/"elevation" (degrees), "intensity"}``
    (see :func:`light_params`); None: each renderer's default lighting.

    ``progress(done, total)``: called while the ray tracer runs (other renderers ignore it); returning
    ``False`` stops it with :class:`ewald.raytrace.RenderCancelled`. ``device``: where the ray tracer runs,
    ``None``/'cpu' (every core), 'cuda' (an NVIDIA GPU through CuPy) or 'auto' (the GPU when there is one);
    the other renderers ignore it.

    ``theme``: a theme name ('VESTA', 'Journal', 'ASE', 'Dark slide', plugins; see :mod:`ewald.themes`)
    that sets element colors, background, cell color and radius preset; explicit arguments win.
    ``background`` defaults to white.
    """
    import pyvista as pv

    if renderer not in RENDERERS:
        raise ValueError(f"unknown renderer {renderer!r}; use 'opengl', 'raytrace' or 'vector'")
    from .color_rule import take

    data = take(data, scene_kwargs, scene_kwargs.get("frame", 0))
    if labels is not None and labels is not False:
        from .overlays import label_params

        overlays = [*_overlay_list(overlays), {"name": "Atom labels", "params": label_params(labels)}]
    if scene_kwargs.get("measurements"):
        overlays = [*_overlay_list(overlays), {"name": "Measurements",
                                               "params": {"measurements": scene_kwargs["measurements"]}}]
    if theme is not None:
        from .themes import apply_theme

        themed = apply_theme(theme, data, {"background": background, **{
            k: scene_kwargs.get(k) for k in ("type_colors", "cell_color", "radius_preset")}})
        background = themed.pop("background")
        scene_kwargs.update({k: v for k, v in themed.items() if v is not None})
    background = "white" if background is None else background
    if scene_kwargs.get("atom_style") == "flat" and ambient_occlusion is None:
        ambient_occlusion = False  # SSAO would shade the flat discs
    q = quality_settings(quality, samples=samples, ao_samples=ao_samples, noise_threshold=noise_threshold,
                         shadows=shadows,
                         antialiasing=antialiasing, ambient_occlusion=ambient_occlusion,
                         sphere_resolution=scene_kwargs.pop("sphere_resolution", None), scale=scale)
    samples, ao_samples, shadows, noise_threshold = q["samples"], q["ao_samples"], q["shadows"], q["noise_threshold"]
    antialiasing, ambient_occlusion = q["antialiasing"], q["ambient_occlusion"]
    if q["scale"] != 1.0:
        size = (max(16, round(size[0] * q["scale"])), max(16, round(size[1] * q["scale"])))
    if region is not None:
        from .framing import region_camera

        if camera is None:
            camera = default_camera(data, size, direction, orthographic, zoom, radius_scale, scene_kwargs)
        camera = region_camera(camera, region)
    overlays = {"text": text, "legend": legend, "type_legend": type_legend, "scale_bar": scale_bar,
                "type_colors": scene_kwargs.get("type_colors"), "overlays": overlays,
                "frame": scene_kwargs.get("frame", 0)}
    if renderer == "vector":
        from .vector import render_vector

        if camera is None:
            camera = default_camera(data, size, direction, orthographic, zoom, radius_scale, scene_kwargs)
        if scene_kwargs.get("atom_style") == "flat":
            scene_kwargs.setdefault("shading", "flat")  # one color plus outline, as on screen
        for k in ("atom_style", "sphere_resolution", "mesh_limit", "arrow_resolution"):
            scene_kwargs.pop(k, None)
        return render_vector(data, filename, camera=camera, size=size, background=background,
                             transparent_background=transparent_background, radius_scale=radius_scale,
                             bonds=bonds, show_cell=show_cell, dpi=dpi, light=light,
                             **{k: v for k, v in overlays.items() if k not in ("type_colors", "frame")},
                             **scene_kwargs)
    if renderer == "raytrace":
        return _render_raytraced(data, filename, size=size, direction=direction, orthographic=orthographic,
                                 zoom=zoom, background=background, transparent_background=transparent_background,
                                 show_cell=show_cell, bonds=bonds, radius_scale=radius_scale, camera=camera,
                                 samples=samples, ao_samples=ao_samples, shadows=shadows, overlays=overlays,
                                 dpi=dpi, light=light, progress=progress, noise_threshold=noise_threshold,
                                 device=device, **scene_kwargs)

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    p.set_background(background)
    res = q["sphere_resolution"] if data.count <= 20_000 else min(q["sphere_resolution"], 16)
    scene = build_scene(p, data, radius_scale=radius_scale, show_cell=show_cell, bonds=bonds,
                        sphere_resolution=res, light=light, **scene_kwargs)
    if any(s.transparency > 0 for s in data.surfaces.values()) or needs_depth_peeling(scene):
        p.enable_depth_peeling()
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
    overlays["colorbars"] = scene.legends
    return _finish(img, data, filename, cam, overlays, background, transparent_background, dpi)


def _overlay_list(overlays) -> list:
    if not overlays:
        return []
    return [overlays] if isinstance(overlays, (str, dict)) or not isinstance(overlays, (list, tuple)) else list(overlays)


# scene keywords that change what the default camera frames (boundary ghosts and their bond partners)
_CAMERA_KEYS = _RADIUS_KEYS + ("type_colors", "vectors", "visuals", "boundary", "boundary_bonds", "bond_pairs",
                                 "replicate", "replicate_cell")


def default_camera(data, size=(1600, 1200), direction="iso", orthographic=True, zoom=1.0, radius_scale=1.0,
                   scene_kwargs=None) -> dict:
    """Camera that frames ``data`` like the OpenGL renderer (no window is shown)."""
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    build_scene(p, data, radius_scale=radius_scale, show_cell=True, sphere_resolution=6,
                **{k: v for k, v in (scene_kwargs or {}).items() if k in _CAMERA_KEYS})
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
                      light=None, progress=None, noise_threshold=0.0, device=None, **scene_kwargs):
    from .overlays import cell_line_color
    from .raytrace import raytrace, scene_from_data

    for k in ("atom_style", "sphere_resolution", "mesh_limit", "antialiasing", "ambient_occlusion", "cell_width"):
        scene_kwargs.pop(k, None)
    if camera is None:  # same framing as the OpenGL path
        camera = default_camera(data, size, direction, orthographic, zoom, radius_scale, scene_kwargs)
    cell_color = cell_line_color(background, transparent_background)
    view = np.asarray(camera["position"], float) - np.asarray(camera["focal_point"], float)
    scene_kwargs.setdefault("arrow_resolution", 16 if samples <= 4 else 24)
    scene = scene_from_data(data, radius_scale=radius_scale, bonds=bonds, show_cell=show_cell,
                            cell_color=scene_kwargs.pop("cell_color", cell_color), view_direction=view,
                            **scene_kwargs)
    lit = {}
    if light is not None:
        d, k = light_params(light)
        lit = {"light_direction": tuple(d), "diffuse": 0.6 * k, "specular": 0.25 * k}
    img = raytrace(scene, camera, size, samples=samples, ao_samples=ao_samples, shadows=shadows,
                   noise_threshold=noise_threshold,
                   background=background, transparent_background=transparent_background, progress=progress,
                   device=device, **lit)
    rgba = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    overlays = {**overlays, "colorbars": scene.legends}
    return _finish(rgba, data, filename, camera, overlays, background, transparent_background, dpi)


def render_animation(pipeline, filename, *, frames=None, fps: float = 10, device=None, text=None,
                     time_per_frame: float | None = None, progress=None, **kwargs):
    """Render frames of a pipeline into a movie; see :func:`ewald.movie.render_animation`."""
    from .movie import render_animation as _render_animation

    return _render_animation(pipeline, filename, frames=frames, fps=fps, device=device, text=text,
                             time_per_frame=time_per_frame, progress=progress, **kwargs)
