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
    actors = renderer.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        mapper = actor.GetMapper()
        if hasattr(mapper, "GetSplatShaderCode") and mapper.GetSplatShaderCode() in (_IMPOSTOR_SHADER, _FLAT_SHADER):
            actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniform4f("impostorProj", proj)


def build_scene(plotter, data: DataCollection, *, radius_scale: float = 1.0, show_cell: bool = True,
                cell_color="black", cell_width: float = 2.0, bonds: float | None = None,
                bond_radius: float = 0.15, only_visible=None, atom_style: str = "auto",
                mesh_limit: int = 150_000, sphere_resolution: int | None = None,
                type_colors: dict | None = None, type_radii: dict | None = None,
                bond_pairs: list | None = None, trajectory_width: float = 2.0,
                radius_preset: str = "covalent", oxidation_states: dict | None = None,
                coordination: dict | None = None, vectors=None, arrow_resolution: int | None = None,
                visuals=None, frame: int = 0, boundary=None, boundary_bonds: str = "half",
                replicate=None, replicate_cell: str = "super", show_atoms: bool = True, surfaces=None,
                cell_radius=None, type_transparency=None, selection_color=None):
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
    a 'Selection' property in that color (OVITO's red).
    """
    from .scene import SceneContext, build

    scene = build(data, ctx=SceneContext(frame=frame, only_visible=only_visible), visuals=visuals,
                  radius_scale=radius_scale, type_colors=type_colors, type_radii=type_radii,
                  radius_preset=radius_preset, oxidation_states=oxidation_states, coordination=coordination,
                  bonds=bonds, bond_pairs=bond_pairs, bond_radius=bond_radius, vectors=vectors,
                  trajectory_width=trajectory_width, show_cell=show_cell, cell_width=cell_width,
                  boundary=boundary, boundary_bonds=boundary_bonds, replicate=replicate,
                  replicate_cell=replicate_cell, show_atoms=show_atoms, surfaces=surfaces, cell_radius=cell_radius,
                  type_transparency=type_transparency, selection_color=selection_color)
    draw_scene(plotter, scene, atom_style=atom_style, mesh_limit=mesh_limit, sphere_resolution=sphere_resolution,
               arrow_resolution=arrow_resolution or max(6, (sphere_resolution or 16) // 2 + 2), ink=cell_color)
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
                    add_atoms(plotter, part, atom_style=atom_style, mesh_limit=mesh_limit,
                              sphere_resolution=sphere_resolution, name=name + suffix)
        elif layer.kind == "cylinders":
            _add_cylinders(plotter, prim, name, flat=atom_style == "flat")
        elif layer.kind == "mesh":
            _add_mesh(plotter, prim, name)
        elif layer.kind == "arrows":
            _add_arrows(plotter, prim, name, arrow_resolution, mesh_limit)
        elif layer.kind == "lines":
            _add_lines(plotter, prim, name, ink)


# Fragment code for sphere impostors: discard outside the disc, then Blinn-Phong with a
# fixed head light plus ambient, matching the mesh material (ambient 0.25, diffuse 0.8).
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
vec3 l = normalize(vec3(-0.25, 0.35, 1.0));
float diff = max(dot(n, l), 0.0);
float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
diffuseColor = diffuseColor * (0.3 + 0.75 * diff) + vec3(0.3 * spec);
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
_FLAT_SHADER = _IMPOSTOR_SHADER.replace("""vec3 l = normalize(vec3(-0.25, 0.35, 1.0));
float diff = max(dot(n, l), 0.0);
float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
diffuseColor = diffuseColor * (0.3 + 0.75 * diff) + vec3(0.3 * spec);
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
    """The bonds a renderer draws: ``data.bonds`` if present, else a search by ``cutoff``/``pairs``."""
    from .bonds import find_bonds

    if data.bonds is not None:
        return data.bonds
    return find_bonds(data, cutoff, pairs)


def _add_cylinders(plotter, cyl, name, flat: bool = False):
    """Cylinders as tubes (half-bonds: one per atom, in its color); ``flat``: unshaded (ASE-GUI look)."""
    import pyvista as pv

    if not cyl.count:
        return
    radii = np.unique(cyl.radius)
    for r in radii:  # one tube actor per radius (bonds have one)
        pick = np.flatnonzero(cyl.radius == r) if len(radii) > 1 else np.arange(cyl.count)
        k = len(pick)
        pts = np.empty((2 * k, 3))
        pts[0::2] = cyl.start[pick]
        pts[1::2] = cyl.end[pick]
        lines = np.column_stack([np.full(k, 2), np.arange(0, 2 * k, 2), np.arange(1, 2 * k, 2)]).ravel()
        poly = pv.PolyData(pts, lines=lines)
        poly.cell_data["rgb"] = (np.clip(cyl.rgb[pick], 0, 1) * 255).astype(np.uint8)
        tubes = poly.tube(radius=float(r), n_sides=12, capping=False)
        light = dict(ambient=1.0, diffuse=0.0, specular=0.0) if flat else \
            dict(ambient=0.25, diffuse=0.8, specular=0.2, specular_power=30)
        actor_name = name if len(radii) == 1 else f"{name}:{r:g}"
        plotter.add_mesh(tubes, scalars="rgb", rgb=True, name=actor_name, smooth_shading=not flat, render=False,
                         **light)
        if flat:
            _add_stick_outline(plotter, cyl.start[pick], cyl.end[pick], float(r), f"{actor_name}:outline")


def _add_stick_outline(plotter, start, end, radius, name):
    """Dark silhouette edges of sticks (ASE GUI look), recomputed by VTK whenever the camera moves.
    The two halves of a bond (same midpoint ``end``) become one tube, so no ring is drawn between them;
    a half without its partner (a bond across the periodic boundary) keeps its real end."""
    import pyvista as pv
    import vtk

    _, inv, cnt = np.unique(np.round(end / 1e-4).astype(np.int64), axis=0, return_inverse=True,
                            return_counts=True)
    inv = inv.ravel()
    paired = cnt[inv] == 2
    halves = np.flatnonzero(paired)[np.argsort(inv[paired], kind="stable")].reshape(-1, 2)
    single = np.flatnonzero(~paired)
    a = np.concatenate([start[halves[:, 0]], start[single]])
    b = np.concatenate([start[halves[:, 1]], end[single]])
    k = len(a)
    pts = np.empty((2 * k, 3))
    pts[0::2], pts[1::2] = a, b
    lines = np.column_stack([np.full(k, 2), np.arange(0, 2 * k, 2), np.arange(1, 2 * k, 2)]).ravel()
    tubes = pv.PolyData(pts, lines=lines).tube(radius=radius, n_sides=12, capping=False).triangulate()
    sil = vtk.vtkPolyDataSilhouette()  # needs polygons, hence triangulate (tube makes strips)
    sil.SetInputData(tubes)
    sil.SetCamera(plotter.renderer.GetActiveCamera())
    sil.SetDirectionToCameraVector()
    actor = plotter.add_mesh(sil, color=(0.08, 0.08, 0.08), line_width=1.5, name=name, render=False,
                             lighting=False)
    m = actor.GetMapper()  # the lines lie on the tube surface: pull them in front of it
    m.SetResolveCoincidentTopologyToPolygonOffset()
    m.SetRelativeCoincidentTopologyLineOffsetParameters(-2, -4)


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
# samples: ray-traced rays per pixel; ao_samples: ambient-occlusion rays; antialiasing: OpenGL SSAA;
# ambient_occlusion: OpenGL SSAO; sphere_resolution: OpenGL sphere mesh detail; scale: pixel size factor.
QUALITY_PRESETS = {
    "draft": {"samples": 1, "ao_samples": 0, "shadows": False, "antialiasing": False,
              "ambient_occlusion": False, "sphere_resolution": 8, "scale": 0.5},
    "standard": {"samples": 4, "ao_samples": 4, "shadows": True, "antialiasing": True,
                 "ambient_occlusion": False, "sphere_resolution": 16, "scale": 1.0},
    "publication": {"samples": 16, "ao_samples": 8, "shadows": True, "antialiasing": True,
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
           ao_samples: int | None = None, shadows: bool | None = None, type_legend: bool = False,
           scale_bar: bool | float | None = None, dpi: float | None = None, quality: str = "publication",
           scale: float | None = None, theme=None, overlays=None, region=None, labels=None, **scene_kwargs):
    """Render one DataCollection to ``filename`` or return an RGB array.

    ``direction`` is a view vector *from* the scene to the camera, or one of
    'x', 'y', 'z', '-x', ..., 'iso'. ``bonds=2.8`` draws bonds shorter than 2.8 A.
    ``camera`` (see :func:`camera_state`) reproduces an interactive view exactly.
    ``region=(x, y, height)`` renders only that part of the view: the GUI Render frame, as fractions of
    the view height (see :mod:`ewald.framing`); None shows the whole view height, centered.

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
    in Angstrom; exact in orthographic views), and ``overlays``: overlay elements
    (Coordinate tripod, Attributes, Atom labels, plugins; see :mod:`ewald.overlays`). ``labels`` puts a
    label on each atom: True (its type), a template such as ``"{type}{index}"`` or ``"{Charge:.2f}"``, or a
    dict with ``text``, ``types``, ``size``, ``color`` (:class:`ewald.overlays.AtomLabels`). ``dpi`` is stored
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

    ``theme``: a theme name ('VESTA', 'Journal', 'ASE', 'Dark slide', plugins; see :mod:`ewald.themes`)
    that sets element colors, background, cell color and radius preset; explicit arguments win.
    ``background`` defaults to white.
    """
    import pyvista as pv

    if renderer not in RENDERERS:
        raise ValueError(f"unknown renderer {renderer!r}; use 'opengl', 'raytrace' or 'vector'")
    if labels is not None and labels is not False:
        from .overlays import label_params

        overlays = [*_overlay_list(overlays), {"name": "Atom labels", "params": label_params(labels)}]
    if theme is not None:
        from .themes import apply_theme

        themed = apply_theme(theme, data, {"background": background, **{
            k: scene_kwargs.get(k) for k in ("type_colors", "cell_color", "radius_preset")}})
        background = themed.pop("background")
        scene_kwargs.update({k: v for k, v in themed.items() if v is not None})
    background = "white" if background is None else background
    if scene_kwargs.get("atom_style") == "flat" and ambient_occlusion is None:
        ambient_occlusion = False  # SSAO would shade the flat discs
    q = quality_settings(quality, samples=samples, ao_samples=ao_samples, shadows=shadows,
                         antialiasing=antialiasing, ambient_occlusion=ambient_occlusion,
                         sphere_resolution=scene_kwargs.pop("sphere_resolution", None), scale=scale)
    samples, ao_samples, shadows = q["samples"], q["ao_samples"], q["shadows"]
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
        for k in ("atom_style", "sphere_resolution", "mesh_limit", "trajectory_width", "arrow_resolution"):
            scene_kwargs.pop(k, None)
        return render_vector(data, filename, camera=camera, size=size, background=background,
                             transparent_background=transparent_background, radius_scale=radius_scale,
                             bonds=bonds, show_cell=show_cell, dpi=dpi,
                             **{k: v for k, v in overlays.items() if k not in ("type_colors", "frame")},
                             **scene_kwargs)
    if renderer == "raytrace":
        return _render_raytraced(data, filename, size=size, direction=direction, orthographic=orthographic,
                                 zoom=zoom, background=background, transparent_background=transparent_background,
                                 show_cell=show_cell, bonds=bonds, radius_scale=radius_scale, camera=camera,
                                 samples=samples, ao_samples=ao_samples, shadows=shadows, overlays=overlays,
                                 dpi=dpi, **scene_kwargs)

    p = pv.Plotter(off_screen=True, window_size=tuple(size))
    p.set_background(background)
    res = q["sphere_resolution"] if data.count <= 20_000 else min(q["sphere_resolution"], 16)
    scene = build_scene(p, data, radius_scale=radius_scale, show_cell=show_cell, bonds=bonds,
                        sphere_resolution=res, **scene_kwargs)
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
                      **scene_kwargs):
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
    img = raytrace(scene, camera, size, samples=samples, ao_samples=ao_samples, shadows=shadows,
                   background=background, transparent_background=transparent_background)
    rgba = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return _finish(rgba, data, filename, camera, overlays, background, transparent_background, dpi)


def _add_colorbar(plotter, data, font_size=None):
    import pyvista as pv

    from .colormaps import get_colormap

    lo, hi = data.attributes["ColorCoding.range"]
    dummy = pv.PolyData(np.zeros((2, 3)))
    dummy["v"] = np.array([lo, hi])
    actor = plotter.add_mesh(dummy, scalars="v", cmap=get_colormap(data.attributes.get("ColorCoding.cmap", "viridis")),
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
