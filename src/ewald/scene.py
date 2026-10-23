"""Renderer-neutral scene: what to draw, separate from how each renderer draws it.

A :class:`VisualElement` (OVITO's "visual element") reads a :class:`~ewald.data.DataCollection`
and returns primitives; every renderer (OpenGL viewport and images, the ray tracer, the vector
renderer) draws every primitive kind. So a new element, built in or a plugin, appears in all
renderers without touching them:

==================  ============================================================================
:class:`Spheres`     centers, radii, colors, opacity (atoms, markers)
:class:`Cylinders`   start/end, radius, colors (half-bonds, sticks)
:class:`Lines`       polylines with a width in pixels (cell, trajectories); a world ``radius``
                     lets renderers without lines (the ray tracer) draw them as tubes
``SurfaceMesh``      triangle meshes with edges (:class:`ewald.mesh.SurfaceMesh`)
``Arrows``           shaft + cone arrows (:class:`ewald.arrows.Arrows`)
==================  ============================================================================

Positions are world coordinates in Angstrom; primitives know nothing about the camera.
:func:`build` runs the built-in elements (configured from the ``render()`` keyword arguments)
and then any extra elements given as ``visuals=[...]``.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from .arrows import Arrows
from .mesh import SurfaceMesh


# ----------------------------------------------------------------------------- primitives
@dataclass
class Spheres:
    """``centers`` (N x 3), ``radii`` (N), ``rgb`` (N x 3, 0..1), ``alpha`` (N, 1 = opaque) and the
    particle ``index`` of each sphere. Large particle sets may instead give per-type tables
    ``palette`` (T x 3) and ``radius_table`` (T) with ``palette_index`` (N); :attr:`rgb` and
    :attr:`radii` then expand on first use, and the OpenGL renderer never expands them."""

    centers: np.ndarray
    radii: np.ndarray | None = None
    rgb: np.ndarray | None = None
    alpha: np.ndarray | None = None
    index: np.ndarray | None = None
    palette: np.ndarray | None = None
    radius_table: np.ndarray | None = None
    palette_index: np.ndarray | None = None

    def __post_init__(self):
        self.centers = np.asarray(self.centers, float).reshape(-1, 3)
        n = len(self.centers)
        if self.rgb is None and self.palette is None:
            self.rgb = np.full((n, 3), 0.6)
        if self.radii is None and self.radius_table is None:
            self.radii = np.full(n, 0.5)
        if self.radii is not None:
            self.radii = np.broadcast_to(np.asarray(self.radii, float), (n,))
        if self.rgb is not None:
            self.rgb = np.broadcast_to(np.asarray(self.rgb, float), (n, 3))
        if self.alpha is None:
            self.alpha = np.ones(n)
        self.alpha = np.broadcast_to(np.asarray(self.alpha, float), (n,))
        if self.index is None:
            self.index = np.arange(n)

    @property
    def count(self) -> int:
        return len(self.centers)

    def colors(self) -> np.ndarray:
        return self.rgb if self.rgb is not None else self.palette[self.palette_index]

    def sizes(self) -> np.ndarray:
        return self.radii if self.radii is not None else self.radius_table[self.palette_index]


@dataclass
class Cylinders:
    """Cylinders from ``start`` to ``end`` (M x 3) with ``radius`` (M) and ``rgb`` (M x 3).
    ``owner`` is the particle each one belongs to (the vector renderer paints a half-bond just
    behind its atom), ``alpha`` its opacity (renderers without transparency skip only 0), and
    ``cap_end`` closes the end with a sphere in the ray tracer."""

    start: np.ndarray
    end: np.ndarray
    radius: np.ndarray | float = 0.15
    rgb: np.ndarray | None = None
    alpha: np.ndarray | None = None
    owner: np.ndarray | None = None
    cap_end: bool = True

    def __post_init__(self):
        self.start = np.asarray(self.start, float).reshape(-1, 3)
        self.end = np.asarray(self.end, float).reshape(-1, 3)
        m = len(self.start)
        self.radius = np.broadcast_to(np.asarray(self.radius, float), (m,))
        self.rgb = np.broadcast_to(np.asarray((0.6, 0.6, 0.6) if self.rgb is None else self.rgb, float), (m, 3))
        self.alpha = np.broadcast_to(np.asarray(1.0 if self.alpha is None else self.alpha, float), (m,))

    @property
    def count(self) -> int:
        return len(self.start)


@dataclass
class Lines:
    """Polylines: ``points`` (P x 3) and ``strips``, one array of point indices per line, with one
    ``rgb`` per strip (S x 3) or ``None`` for the renderer's ink color (black on light backgrounds).
    ``point_rgb`` (P x 3), when given, colors every point instead (a color map along the line):
    renderers blend it along each piece where they can, else give a piece the mean of its ends.
    ``width`` is in pixels of a 1600-pixel-high image; ``radius`` (Angstrom), when given, lets
    renderers that have no lines draw them as tubes with round joints. Lines are guides, so those
    tubes cast no shadows and no ambient occlusion unless ``cast_shadows`` is True."""

    points: np.ndarray
    strips: list
    rgb: np.ndarray | None = None
    width: float = 2.0
    radius: float | None = None
    cast_shadows: bool = False
    point_rgb: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, float).reshape(-1, 3)
        self.strips = [np.asarray(s, np.int64) for s in self.strips]
        if self.rgb is not None:
            self.rgb = np.broadcast_to(np.asarray(self.rgb, float), (len(self.strips), 3))
        if self.point_rgb is not None:
            self.point_rgb = np.broadcast_to(np.asarray(self.point_rgb, float), (len(self.points), 3))

    @property
    def count(self) -> int:
        return len(self.strips)

    def segments(self) -> np.ndarray:
        """Every line piece as a pair of point indices (K x 2), and the strip of each."""
        pairs = [np.column_stack([s[:-1], s[1:]]) for s in self.strips if len(s) > 1]
        return np.vstack(pairs) if pairs else np.zeros((0, 2), np.int64)

    def vertex_colors(self, ink=None) -> np.ndarray | None:
        """One RGB per point (P x 3): ``point_rgb``, else each strip's ``rgb`` on its points, else
        ``ink`` (None when there is neither color nor ink)."""
        if self.point_rgb is not None:
            return np.asarray(self.point_rgb, float)
        if self.rgb is None:
            return None if ink is None else np.broadcast_to(np.asarray(ink, float), (len(self.points), 3))
        out = np.zeros((len(self.points), 3))
        for k, s in enumerate(self.strips):
            out[s] = self.rgb[k]
        return out

    def segment_colors(self, ink=(0.0, 0.0, 0.0)) -> np.ndarray:
        """One RGB per piece of :meth:`segments` (K x 3): its strip's color, the mean of its two
        ``point_rgb`` ends, or ``ink``."""
        seg = self.segments()
        if self.point_rgb is not None:
            pr = np.asarray(self.point_rgb, float)
            return 0.5 * (pr[seg[:, 0]] + pr[seg[:, 1]])
        if self.rgb is None:
            return np.tile(np.asarray(ink, float), (len(seg), 1))
        return self.rgb[np.concatenate([np.full(max(len(s) - 1, 0), k) for k, s in enumerate(self.strips)]
                                       or [np.zeros(0, np.int64)])]


def make_arrows(base, tip, width: float = 0.2, rgb=(0.85, 0.2, 0.2), alpha: float = 1.0,
                shading: str = "normal", name: str = "") -> Arrows:
    """Arrows from ``base`` to ``tip`` (M x 3) with OVITO's head proportions, for plugin elements."""
    from .arrows import HEAD_LENGTH, HEAD_RADIUS

    base = np.asarray(base, float).reshape(-1, 3)
    tip = np.asarray(tip, float).reshape(-1, 3)
    m = len(base)
    length = np.linalg.norm(tip - base, axis=1)
    r = np.full(m, 0.5 * float(width))
    head_len = np.minimum(HEAD_LENGTH * r, length)
    head_rad = HEAD_RADIUS * r * np.where(r > 0, head_len / np.maximum(HEAD_LENGTH * r, 1e-300), 1.0)
    rgb = np.broadcast_to(np.asarray(rgb, float), (m, 3)).copy()
    return Arrows(base, tip, r, head_rad, head_len, rgb, float(alpha), np.arange(m), base.copy(), shading, name)


KINDS = {Spheres: "spheres", Cylinders: "cylinders", Lines: "lines", SurfaceMesh: "mesh", Arrows: "arrows"}


@dataclass
class Layer:
    """One primitive set and the element that made it. ``name`` is unique in a scene and names the
    OpenGL actor ('atoms', 'bonds', 'cell', 'surface:<name>', 'vectors:<property>', ...)."""

    name: str
    primitive: object
    element: str = ""

    @property
    def kind(self) -> str:
        return KINDS[type(self.primitive)]


@dataclass
class Scene:
    """Everything one image shows, in drawing order, and the color bars its elements ask for
    (``legends``: ``{"title", "cmap", "range"}`` dicts, drawn next to a ColorCoding bar)."""

    layers: list = field(default_factory=list)
    legends: list = field(default_factory=list)

    def add(self, name: str, primitive, element: str = "") -> Layer:
        if type(primitive) not in KINDS:
            raise TypeError(f"{type(primitive).__name__} is not a scene primitive; use one of "
                            + ", ".join(k.__name__ for k in KINDS))
        taken = {lay.name for lay in self.layers}
        base, k = name, 2
        while name in taken:
            name, k = f"{base}:{k}", k + 1
        layer = Layer(name, primitive, element)
        self.layers.append(layer)
        return layer

    def of(self, kind: str) -> list[Layer]:
        return [lay for lay in self.layers if lay.kind == kind]

    def __getitem__(self, name: str) -> Layer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return any(lay.name == name for lay in self.layers)

    def summary(self) -> str:
        return "; ".join(f"{lay.name} ({lay.kind}, {getattr(lay.primitive, 'count', getattr(lay.primitive, 'face_count', 0))})"
                         for lay in self.layers) or "empty"


@dataclass
class SceneContext:
    """What an element may know besides the data: the ``frame``, the direction toward the camera
    (``view_direction``, may be None), ``only_visible`` (a particle mask the viewport uses for its
    coarse level of detail) and ``particle_colors``, the per-particle RGB of the Particles element
    (Bonds and other per-particle elements reuse it)."""

    frame: int = 0
    view_direction: np.ndarray | None = None
    only_visible: np.ndarray | None = None
    particle_colors: np.ndarray | None = None
    particle_radii: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- elements
class VisualElement:
    """Base class of everything that draws: ``build(data, ctx)`` returns primitives.

    Store every constructor argument as an attribute of the same name (as for modifiers):
    :meth:`parameters`, sessions, the GUI form and the Claude tools read them from there.
    ``build`` returns one primitive, a list of them, or ``(name, primitive)`` pairs, or None;
    it must not modify ``data``. ``enabled=False`` keeps an element in a session without drawing.
    """

    #: shown in the GUI and in ``ewald plugins``; defaults to the class name
    name: str = ""
    enabled: bool = True

    def applies(self, data) -> bool:
        """Whether there is anything to draw in ``data`` (cheap; default True)."""
        return True

    def build(self, data, ctx: SceneContext):
        raise NotImplementedError

    def legend(self, data, ctx: SceneContext) -> dict | None:
        """A color bar for what :meth:`build` drew, ``{"title", "cmap", "range": (lo, hi)}``, or None
        (default). Called after ``build``; every renderer draws it with the overlays."""
        return None

    def parameters(self) -> dict:
        sig = inspect.signature(type(self).__init__)
        return {k: getattr(self, k) for k in list(sig.parameters)[1:]
                if hasattr(self, k) and sig.parameters[k].kind not in (inspect.Parameter.VAR_KEYWORD,
                                                                       inspect.Parameter.VAR_POSITIONAL)}

    @property
    def title(self) -> str:
        return self.name or type(self).__name__

    def __repr__(self):
        return f"{type(self).__name__}(" + ", ".join(f"{k}={v!r}" for k, v in self.parameters().items()) + ")"


class ParticlesVis(VisualElement):
    """Atoms as spheres: the 'Color' and 'Radius' properties, else per-type colors (``type_colors``
    first) and radii (``type_radii``, else ``radius_preset``, see :mod:`ewald.radii`), times
    ``radius_scale``. Particles with Transparency 1 are left out. ``selection_color`` (e.g. red, as in
    OVITO) draws the particles of a 'Selection' property in that color; None leaves them as they are."""

    name = "Particles"

    def __init__(self, radius_scale: float = 1.0, type_colors: dict | None = None, type_radii: dict | None = None,
                 radius_preset: str = "covalent", oxidation_states: dict | None = None,
                 coordination: dict | None = None, selection_color=None):
        self.radius_scale = radius_scale
        self.type_colors = type_colors
        self.type_radii = type_radii
        self.radius_preset = radius_preset
        self.oxidation_states = oxidation_states
        self.coordination = coordination
        self.selection_color = selection_color

    def applies(self, data) -> bool:
        return data.count > 0

    def build(self, data, ctx):
        from .render import _color_table, _radius_table, type_groups

        if ctx.only_visible is not None:
            keep = np.flatnonzero(np.asarray(ctx.only_visible, bool))
        elif "Transparency" in data:
            keep = np.flatnonzero(np.asarray(data["Transparency"]) < 1)
        else:
            keep = None
        sel = slice(None) if keep is None else keep
        alpha = (1 - np.clip(np.asarray(data["Transparency"], float)[sel], 0, 1)) if "Transparency" in data else None
        # Per-type tables gathered by type index: no per-atom clipping or float math on 1e6+ atoms.
        groups = type_groups(data.types) if "Color" not in data or "Radius" not in data else None
        colors = rgb = palette = radii = radius_table = None
        if "Color" in data:
            colors = np.clip(np.asarray(data["Color"], float), 0, 1)
            rgb = colors[sel]
        else:
            palette = np.clip(_color_table(groups[0], self.type_colors), 0, 1)
        if self.selection_color is not None and "Selection" in data:
            chosen = np.asarray(data["Selection"]) != 0
            if chosen.any():
                from .render import _rgb

                colors = colors.copy() if colors is not None else palette[groups[1]]
                colors[chosen] = _rgb(self.selection_color)
                rgb, palette = colors[sel], None
        if "Radius" in data:
            radii = np.asarray(data["Radius"], float)[sel] * self.radius_scale
        else:
            # ionic radii guess coordination from the structure: use the real atoms, not boundary ghosts
            radius_table = _radius_table(groups[0], 0.8, self.type_radii, ctx.extras.get("real_data", data),
                                         self.radius_preset,
                                         self.oxidation_states, self.coordination) * self.radius_scale
        prim = Spheres(np.ascontiguousarray(data.positions[sel]), radii, rgb, alpha,
                       np.arange(data.count) if keep is None else keep, palette, radius_table,
                       None if groups is None else groups[1][sel])
        ctx.particle_colors = colors if colors is not None else palette[groups[1]]
        return [("atoms", prim)]


class BondsVis(VisualElement):
    """Half-bonds as cylinders in their atoms' colors (or the bonds' 'Color'): ``data.bonds``
    when present, else a search by ``cutoff`` or VESTA-style ``pairs`` [(A, B, dmin, dmax)]."""

    name = "Bonds"

    def __init__(self, cutoff: float | None = None, pairs: list | None = None, radius: float = 0.15):
        self.cutoff = cutoff
        self.pairs = pairs
        self.radius = radius

    def applies(self, data) -> bool:
        return data.count > 0 and bool(self.cutoff or self.pairs or data.bonds is not None)

    def build(self, data, ctx):
        from .bonds import bond_segments
        from .render import default_colors, scene_bonds

        b = scene_bonds(data, self.cutoff, self.pairs)
        if not b.count:
            return None
        colors = ctx.particle_colors if ctx.particle_colors is not None else np.clip(default_colors(data), 0, 1)
        start, end, rgb, _ = bond_segments(data, b, colors)
        i, j = b.topology.T
        owner = np.concatenate([i, j])
        if "Half Bond" in b:  # boundary display: only the half at the displayed atom
            keep = np.r_[np.ones(b.count, bool), ~np.asarray(b["Half Bond"], bool)]
            start, end, rgb, owner = start[keep], end[keep], rgb[keep], owner[keep]
        alpha = 1 - np.clip(np.asarray(data["Transparency"], float), 0, 1)[owner] if "Transparency" in data else None
        if alpha is not None:  # a hidden atom (Transparency 1) takes its half-bonds with it
            keep = alpha > 0
            start, end, rgb, owner, alpha = start[keep], end[keep], rgb[keep], owner[keep], alpha[keep]
            if not len(owner):
                return None
        return [("bonds", Cylinders(start, end, self.radius, np.clip(rgb, 0, 1), alpha, owner, cap_end=True))]


class SurfacesVis(VisualElement):
    """Every ``data.surfaces`` mesh (polyhedra, isosurfaces, grid slices). ``styles`` restyles them by
    name on top of what their step made: ``{name: {"visible", "color", "transparency", "edges"}}``."""

    name = "Surfaces"

    def __init__(self, styles: dict | None = None):
        self.styles = styles

    def applies(self, data) -> bool:
        return bool(getattr(data, "surfaces", None))

    def build(self, data, ctx):
        styles = self.styles or {}
        return [(f"surface:{name}", _styled_mesh(mesh, styles[name]) if name in styles else mesh)
                for name, mesh in data.surfaces.items()
                if mesh.face_count and styles.get(name, {}).get("visible", True)]


def _styled_mesh(mesh, style):
    """A shallow copy of ``mesh`` with ``style``'s color, transparency and edges (data stays unchanged)."""
    import copy

    from .render import _rgb

    out = copy.copy(mesh)
    if style.get("color") is not None:
        out.face_properties = {**mesh.face_properties, "Color": np.tile(_rgb(style["color"]), (mesh.face_count, 1))}
        out.vertex_properties = {k: v for k, v in mesh.vertex_properties.items() if k != "Color"}
    if style.get("transparency") is not None:
        out.transparency = float(style["transparency"])
    if style.get("edges") is False:
        out.edges = None
    return out


class VectorsVis(VisualElement):
    """Per-particle vectors as arrows: ``data.vectors`` (VectorVis) plus ``vectors`` (a property
    name, a list, or ``{name: {VectorStyle fields}}``, see :class:`ewald.arrows.VectorStyle`)."""

    name = "Vectors"

    def __init__(self, vectors=None):
        self.vectors = vectors

    def applies(self, data) -> bool:
        return bool(getattr(data, "vectors", None) or self.vectors)

    def build(self, data, ctx):
        from .arrows import scene_arrows

        return [(f"vectors:{a.name}", a) for a in scene_arrows(data, self.vectors, ctx.frame)]


TRAJECTORY_COLOR_BY = ("Time", "Frame", "Step Length")


class TrajectoryLinesVis(VisualElement):
    """``tables['trajectory-lines']`` (GenerateTrajectoryLines) as one polyline per particle, in its atom
    color or a color map by Time, Frame, step length or a property, over a Time window or a trail.

    ``width`` in pixels, 0 hides them; ``radius`` (Angstrom) sizes the ray-traced tubes (None: 0.04 x width).

    ``color_by`` None draws each line in its atom's color. Otherwise the lines follow the color map
    ``colormap`` point by point: 'Time' (the table's Time, the timestep), 'Frame', 'Step Length' (the
    distance moved since the previous point), another table column, or a particle property of the
    current frame ('Path Length', 'Net Displacement', ...) that colors a whole line. ``color_range``
    ``(lo, hi)`` fixes the ends of the map (values outside take the end colors); None spans the values in
    the Time window (all of a trail, also the part not drawn yet, so a color keeps its meaning).

    ``time_range`` ``(t0, t1)`` draws only the part of the paths inside that Time window (either end
    may be None). ``trail`` follows the current frame: True draws the paths up to it, a number only
    the last ``trail`` Time units before it (a comet tail). ``colorbar`` adds the map's color bar.
    """

    name = "Trajectory lines"
    type_colors = None   # the view's per-type colors, for lines in their atom's color (set by builtin_elements)

    def __init__(self, width: float = 2.0, color_by: str | None = None, colormap: str = "viridis",
                 color_range=None, time_range=None, trail=None, radius: float | None = None,
                 colorbar: bool = True):
        self.width = width
        self.color_by = color_by
        self.colormap = colormap
        self.color_range = color_range
        self.time_range = time_range
        self.trail = trail
        self.radius = radius
        self.colorbar = colorbar

    def applies(self, data) -> bool:
        return self.width > 0 and "trajectory-lines" in data.tables

    def build(self, data, ctx):
        from .render import default_colors

        self._bar = None
        table = data.tables["trajectory-lines"]
        pts = np.asarray(table["Position"], dtype=float)
        ids = np.asarray(table["Particle Identifier"])
        if len(pts) < 2:
            return None
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        sizes = np.diff(np.r_[starts, len(ids)])
        line_of = np.repeat(np.arange(len(starts)), sizes)
        time = np.asarray(table["Time"], float) if "Time" in table else \
            np.concatenate([np.arange(m, dtype=float) for m in sizes])
        keep = self._window(table, time, data, ctx)
        values = None if self.color_by is None else self._values(table, data, ids, starts, sizes, line_of)
        colors = np.tile([0.3, 0.3, 0.3], (len(starts), 1))
        if values is None and data.count:
            here = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)
            order = np.argsort(here)
            pos = np.clip(np.searchsorted(here[order], ids[starts]), 0, data.count - 1)
            hit = here[order][pos] == ids[starts]
            colors[hit] = np.clip(default_colors(data, self.type_colors), 0, 1)[order][pos[hit]]
        # a strip per run of kept points of one line, at least two points long
        strips, owner = [], []
        for k, (s, m) in enumerate(zip(starts, sizes)):
            idx = np.arange(s, s + m)
            if keep is not None:
                idx = idx[keep[s:s + m]]
            if len(idx) < 2:
                continue
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(run) > 1:
                    strips.append(run)
                    owner.append(k)
        if not strips:
            return None
        radius = self.radius if self.radius else 0.04 * float(self.width)
        if values is None:
            return [("trajectories", Lines(pts, strips, colors[owner], self.width, radius))]
        from .colormaps import get_colormap

        # the map spans the whole Time window, not just the trail drawn so far: colors keep their meaning
        # while the trail grows frame by frame
        span = self._window(table, time, data, ctx, trail=False)
        lo, hi = self._limits(values if span is None else values[span])
        span = hi - lo if hi > lo else 1.0
        rgb = np.asarray(get_colormap(self.colormap)(np.clip((values - lo) / span, 0, 1)), float)[:, :3]
        rgb[~np.isfinite(values)] = 0.5
        if self.colorbar:
            self._bar = {"title": str(self.color_by), "cmap": self.colormap, "range": (float(lo), float(hi))}
        return [("trajectories", Lines(pts, strips, None, self.width, radius, point_rgb=rgb))]

    def legend(self, data, ctx):
        return getattr(self, "_bar", None)

    def _limits(self, shown):
        finite = shown[np.isfinite(shown)]
        auto = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        rng = self.color_range
        if rng is None:
            return auto
        lo, hi = rng
        return (auto[0] if lo is None else float(lo)), (auto[1] if hi is None else float(hi))

    def _window(self, table, time, data, ctx, trail: bool = True):
        """Which points to draw (None: all): ``time_range`` and (``trail``) the trail."""
        keep = None
        if self.time_range is not None:
            t0, t1 = self.time_range
            keep = np.ones(len(time), bool)
            if t0 is not None:
                keep &= time >= float(t0)
            if t1 is not None:
                keep &= time <= float(t1)
        if trail and self.trail is not None and self.trail is not False:
            now = self._current_time(table, time, data, ctx)
            tail = time <= now + 1e-9 * max(1.0, abs(now))
            if self.trail is not True:
                tail &= time >= now - float(self.trail)
            keep = tail if keep is None else keep & tail
        return keep

    @staticmethod
    def _current_time(table, time, data, ctx) -> float:
        """The Time of the frame shown (the data's SourceFrame, else ``ctx.frame``): interpolated from the
        table's Frame column, else the data's Timestep."""
        frame = int(data.attributes.get("SourceFrame", ctx.frame))
        if "Frame" in table:
            frames = np.asarray(table["Frame"], float)
            first = np.flatnonzero(np.r_[True, np.diff(frames) <= 0])   # one particle's rows
            n = first[1] if len(first) > 1 else len(frames)
            return float(np.interp(frame, frames[:n], time[:n]))
        return float(data.attributes.get("Timestep", frame))

    def _values(self, table, data, ids, starts, sizes, line_of):
        """The value of ``color_by`` at every point of the table."""
        name = self.color_by
        if name == "Step Length":
            pts = np.asarray(table["Position"], float)
            step = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)]
            step[starts] = step[np.minimum(starts + 1, len(step) - 1)]   # a line's first point: its first step
            return step
        if name in table and name not in ("Position", "Particle Identifier"):
            col = np.asarray(table[name])
            if col.ndim == 1 and np.issubdtype(col.dtype, np.number):
                return col.astype(float)
        if name == "Frame" and "Frame" not in table:
            return np.concatenate([np.arange(m, dtype=float) for m in sizes])
        if name in data.particles:
            prop = np.asarray(data[name])
            if prop.ndim == 1 and np.issubdtype(prop.dtype, np.number):
                here = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)
                order = np.argsort(here)
                pos = np.clip(np.searchsorted(here[order], ids[starts]), 0, max(data.count - 1, 0))
                hit = here[order][pos] == ids[starts]
                per_line = np.where(hit, prop.astype(float)[order][pos], np.nan)
                return per_line[line_of]
        options = [*TRAJECTORY_COLOR_BY, *(c for c in table if c not in ("Position", "Particle Identifier")
                                           and c not in TRAJECTORY_COLOR_BY)]
        raise KeyError(f"trajectory lines cannot be colored by {name!r}; use one of {', '.join(options)} "
                       f"or a numeric particle property")


_CELL_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]


class SimulationCellVis(VisualElement):
    """The cell edges as lines, ``width`` pixels wide; ``color`` None uses the renderer's ink
    (black, or light grey on dark backgrounds). ``radius`` (Angstrom) sizes the ray-traced tubes,
    None = 0.15 % of the cell diagonal."""

    name = "Simulation cell"

    def __init__(self, color=None, width: float = 2.0, radius: float | None = None):
        self.color = color
        self.width = width
        self.radius = radius

    def applies(self, data) -> bool:
        return data.cell.is_valid

    def build(self, data, ctx):
        from .render import _rgb

        o, (a, b, c) = data.cell.origin, data.cell.matrix
        corners = np.array([o, o + a, o + b, o + c, o + a + b, o + a + c, o + b + c, o + a + b + c])
        edges = [list(e) for e in _CELL_EDGES]
        if "replicate_cell" in ctx.extras:  # a replicated view: the block, the unit cell or every copy
            from .replicate import cell_edges

            corners, edges = cell_edges(*ctx.extras["replicate_cell"])
        rgb = None if self.color is None else _rgb(self.color)
        radius = self.radius or 0.0015 * float(np.linalg.norm(a + b + c))
        return [("cell", Lines(corners, edges, rgb, self.width, radius))]


MEASUREMENT_COLOR = "#E0701F"


class MeasurementsVis(VisualElement):
    """The view's ``measurements`` (:mod:`ewald.measure`): lines through 2, 3 or 4 atoms (nearest periodic
    images), ``width`` pixels wide in ``color``; the values are the Measurements overlay
    (:class:`ewald.overlays.MeasurementLabels`). Indices are those of the real atoms, also in a
    boundary or replicated view."""

    name = "Measurements"

    def __init__(self, measurements=None, color=MEASUREMENT_COLOR, width: float = 3.0):
        self.measurements = measurements or []
        self.color = color
        self.width = width

    def applies(self, data) -> bool:
        return bool(self.measurements) and self.width > 0

    def build(self, data, ctx):
        from . import measure
        from .render import _rgb

        real = ctx.extras.get("source_data", data)
        items = [m for m in measure.check(self.measurements) if max(m) < real.count]
        if not items:
            return None
        pts, strips = [], []
        for idx in items:
            strips.append(np.arange(len(idx)) + sum(len(s) for s in strips))
            pts.append(measure.points(real, idx))
        radius = 0.04 * float(np.median(np.linalg.norm(np.diff(np.vstack(pts), axis=0), axis=1)) or 1.0)
        return [("measurements", Lines(np.vstack(pts), strips, _rgb(self.color), self.width, radius))]


BUILTIN = (ParticlesVis, BondsVis, SurfacesVis, VectorsVis, TrajectoryLinesVis, SimulationCellVis, MeasurementsVis)


def builtin_elements(*, radius_scale=1.0, type_colors=None, type_radii=None, radius_preset="covalent",
                     oxidation_states=None, coordination=None, bonds=None, bond_pairs=None, bond_radius=0.15,
                     vectors=None, trajectory_width=2.0, trajectory_color_by=None, trajectory_colormap="viridis",
                     trajectory_color_range=None, trajectory_time_range=None, trajectory_trail=None,
                     trajectory_radius=None, show_cell=True, cell_color=None, cell_width=2.0,
                     cell_radius=None, show_atoms=True, surfaces=None, selection_color=None, measurements=None,
                     **_ignored) -> list[VisualElement]:
    """The built-in elements configured from ``render()`` keyword arguments, in drawing order."""
    particles = ParticlesVis(radius_scale, type_colors, type_radii, radius_preset, oxidation_states, coordination,
                             selection_color)
    particles.enabled = bool(show_atoms)
    lines = TrajectoryLinesVis(trajectory_width, trajectory_color_by, trajectory_colormap, trajectory_color_range,
                               trajectory_time_range, trajectory_trail, trajectory_radius)
    lines.type_colors = type_colors
    out = [particles, BondsVis(bonds, bond_pairs, bond_radius), SurfacesVis(surfaces or None), VectorsVis(vectors),
           lines]
    if show_cell:
        out.append(SimulationCellVis(cell_color, cell_width, cell_radius))
    if measurements:
        out.append(MeasurementsVis(measurements))
    return out


def coerce_visuals(visuals) -> list[VisualElement]:
    """``visuals`` as accepted by ``render()``: elements, registered names, or ``{"name": ..., "params":
    {...}}`` dicts (the session form)."""
    if not visuals:
        return []
    if isinstance(visuals, (VisualElement, str, dict)):
        visuals = [visuals]
    out = []
    for v in visuals:
        if isinstance(v, VisualElement):
            out.append(v)
            continue
        from .plugins import visuals as registry

        name, params = (v, {}) if isinstance(v, str) else (v["name"], dict(v.get("params") or {}))
        reg = registry()
        if name not in reg:
            raise KeyError(f"no visual element {name!r}; registered: {', '.join(reg) or 'none'}")
        enabled = params.pop("enabled", True) if isinstance(v, dict) else True
        el = reg[name].cls(**params)
        el.enabled = bool(v.get("enabled", enabled)) if isinstance(v, dict) else True
        out.append(el)
    return out


def visual_to_dict(el: VisualElement) -> dict:
    """``{"name", "params", "enabled"}``, the form sessions, the GUI style and the Claude tools store."""
    from .plugins import visual_name

    return {"name": visual_name(type(el)), "params": el.parameters(), "enabled": bool(el.enabled)}


def build(data, elements=None, ctx: SceneContext | None = None, *, visuals=None, boundary=None,
          boundary_bonds: str = "half", replicate=None, replicate_cell: str = "super", type_transparency=None,
          **render_kwargs) -> Scene:
    """Run ``elements`` (default: :func:`builtin_elements` from ``render_kwargs``) and then ``visuals``
    on ``data``; returns the :class:`Scene` every renderer draws.

    ``boundary`` (VESTA's boundary range, fractional: ``(-0.1, 1.1)``, three pairs, or a margin)
    also draws the periodic images inside that range, with bonds and coordination polyhedra
    completed across the cell; ``boundary_bonds`` says what happens to bonds that leave it
    ('half', 'complete', 'none'; see :mod:`ewald.boundary`). The elements then see those ghost
    atoms as particles, but each primitive's particle ``index``/``owner`` is the real atom.

    ``replicate`` ``(na, nb, nc)`` (or ``n``) draws that many periodic copies of the cell, with bonds
    joined across them and polyhedra and isosurfaces repeated (OVITO's Replicate, as a display
    setting; see :mod:`ewald.replicate`); ``replicate_cell`` draws the outline of the whole block
    ('super'), the original cell ('unit') or every copy ('all'). The boundary range then applies to
    the replicated block. Both are display only: the data and every analysis stay unchanged.

    ``type_transparency`` ``{type: 0..1}`` makes whole particle types see-through (1 hides them, with
    their half-bonds), on top of a 'Transparency' property (the larger value wins)."""
    ctx = ctx or SceneContext()
    ctx.extras["source_data"] = data   # the real atoms, before boundary ghosts and replicas (measurements)
    if type_transparency:
        data = _with_type_transparency(data, type_transparency)
    elements = list(builtin_elements(**render_kwargs) if elements is None else elements) + coerce_visuals(visuals)
    source = None
    if replicate is not None and replicate is not False:
        data, source = _with_replicate(data, elements, ctx, replicate, replicate_cell)
    if boundary is not None and boundary is not False:
        data, ghosts = _with_boundary(data, elements, ctx, boundary, boundary_bonds)
        if ghosts is not None:
            source = ghosts if source is None else source[ghosts]
    scene = Scene()
    for el in elements:
        if not getattr(el, "enabled", True) or not el.applies(data):
            if isinstance(el, ParticlesVis) and el.applies(data):
                el.build(data, ctx)   # hidden atoms still give their bonds their colors
            continue
        out = el.build(data, ctx)
        bar = el.legend(data, ctx) if out is not None else None
        if bar:
            scene.legends.append(dict(bar))
        if out is None:
            continue
        if not isinstance(out, (list, tuple)) or (len(out) == 2 and isinstance(out[0], str)):
            out = [out]
        builtin = isinstance(el, BUILTIN)
        slug = el.title.lower().replace(" ", "-")
        for item in out:
            name, prim = item if isinstance(item, tuple) else (slug, item)
            # plugin layers get their own actor namespace, so they never replace a built-in one
            scene.add(name if builtin else f"visual:{slug}" + ("" if name == slug else f":{name}"), prim, el.title)
    if source is not None:  # ghosts pick, hover and select as their real atom
        for lay in scene.layers:
            prim = lay.primitive
            if isinstance(prim, (Spheres, Arrows)) and prim.index is not None and len(prim.index):
                prim.index = source[np.asarray(prim.index, np.int64)]
            elif isinstance(prim, Cylinders) and prim.owner is not None and len(prim.owner):
                prim.owner = source[np.asarray(prim.owner, np.int64)]
    return scene


def _with_type_transparency(data, table):
    """A display copy of ``data`` whose 'Transparency' includes the per-type values of ``table``."""
    from .render import type_groups

    names, inverse = type_groups(data.types)
    per_type = np.array([float(table.get(str(n), 0.0)) for n in names])
    if not data.count or not per_type.any():
        return data
    t = np.clip(per_type, 0, 1)[inverse]
    if "Transparency" in data:
        t = np.maximum(t, np.asarray(data["Transparency"], float))
    out = data.copy()
    out["Transparency"] = t
    return out


def _element_bonds(data, elements):
    """The bonds the Bonds element would draw on ``data`` (found on the real atoms)."""
    from .render import scene_bonds

    for el in elements:
        if isinstance(el, BondsVis) and getattr(el, "enabled", True) and el.applies(data):
            return scene_bonds(data, el.cutoff, el.pairs)
    return data.bonds


def _with_replicate(data, elements, ctx, replicate, cell_mode):
    """The data tiled ``replicate`` times (see :func:`ewald.replicate.tile`) and the real atom of each."""
    from .replicate import CELL_MODES, normalize_replicate, tile

    if cell_mode not in CELL_MODES:
        raise ValueError(f"unknown replicate_cell {cell_mode!r}; use one of {', '.join(CELL_MODES)}")
    n = normalize_replicate(replicate)
    if n is None or not data.cell.is_valid or not data.count:
        return data, None
    disp, source = tile(data, n, _element_bonds(data, elements))
    if disp.bonds is None and any(isinstance(el, BondsVis) for el in elements):
        from .bonds import Bonds

        disp.bonds = Bonds()  # bonds were searched on the real atoms (none found)
    ctx.extras["real_data"] = data
    ctx.extras["replicate_cell"] = (data.cell, n, cell_mode)
    if ctx.only_visible is not None:
        ctx.only_visible = np.asarray(ctx.only_visible, bool)[source]
    return disp, source


def _with_boundary(data, elements, ctx, boundary, bond_mode):
    """The data with boundary ghosts (see :func:`ewald.boundary.expand`) and the real atom of each.
    Bonds are found on the real atoms first, so ghosts never bond through the periodic cell."""
    from .boundary import expand

    if not data.cell.is_valid or not data.count:
        return data, None
    disp, source = expand(data, boundary, _element_bonds(data, elements), bond_mode=bond_mode)
    if source is None:
        return data, None
    if disp.bonds is None and any(isinstance(el, BondsVis) for el in elements):
        from .bonds import Bonds

        disp.bonds = Bonds()  # never let BondsVis search the ghosts through the periodic cell
    ctx.extras.setdefault("real_data", data)
    if ctx.only_visible is not None:
        ctx.only_visible = np.asarray(ctx.only_visible, bool)[source]
    return disp, source
