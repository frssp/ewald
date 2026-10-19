"""Per-particle vector arrows (OVITO's VectorVis): forces, displacements, velocities, magnetic moments.

A :class:`VectorStyle` in ``data.vectors[property]`` says how to draw one N x 3 particle
property; the :class:`ewald.modifiers.VectorVis` modifier puts it there, or ``render(...,
vectors="Force")`` passes it directly. Geometry is built from the *final* collection at draw
time, so Slice, Replicate or DeleteSelected after VectorVis keep arrows on their atoms.

Every renderer uses :func:`arrow_geometry`: OpenGL draws a cylinder + cone mesh
(:func:`arrow_mesh`), the ray tracer the same triangles plus a cylinder shaft, and the vector
renderer the projected silhouettes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ALIGNMENTS = ("base", "center", "head")
SHADINGS = ("normal", "flat")

# Head proportions relative to the shaft radius r = width / 2 (OVITO's arrow primitive).
HEAD_RADIUS = 2.5
HEAD_LENGTH = 4.5

# Properties VectorVis() shows when no name is given, first match wins.
AUTO_PROPERTIES = ("Force", "Displacement", "Velocity", "Magnetic Moment", "Dipole Orientation")


@dataclass
class VectorStyle:
    """How to draw one vector property; field names follow OVITO's VectorVis.

    ``scaling`` multiplies the vectors (Angstrom per unit), ``width`` is the shaft diameter (A),
    ``alignment`` places the arrow's 'base', 'center' or 'head' on the particle, ``reverse``
    flips it, ``offset`` shifts it (A). ``color`` is one RGB for every arrow unless
    ``color_mapping_property`` ("Magnitude" for this vector's length, another property, or an
    expression) maps values through ``cmap`` over ``color_mapping_interval`` (None: this
    frame's min/max). ``shading`` 'normal' is lit 3D, 'flat' is a solid 2D-looking arrow.
    ``only_selected`` draws arrows of selected particles only; arrows shorter than
    ``min_length`` (A, after scaling) are skipped.
    """

    property: str = "Force"
    scaling: float = 1.0
    width: float = 0.3
    alignment: str = "base"
    reverse: bool = False
    color: tuple = (1.0, 1.0, 0.0)
    shading: str = "normal"
    transparency: float = 0.0
    offset: tuple = (0.0, 0.0, 0.0)
    color_mapping_property: str | None = None
    cmap: str = "viridis"
    color_mapping_interval: tuple | None = None
    only_selected: bool = False
    min_length: float = 1e-6

    def __post_init__(self):
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}, not {self.alignment!r}")
        if self.shading not in SHADINGS:
            raise ValueError(f"shading must be one of {SHADINGS}, not {self.shading!r}")
        if self.color_mapping_property and self.color_mapping_property.lower() == "magnitude":
            self.color_mapping_property = self.property

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Arrows:
    """Arrow geometry in world coordinates: ``base`` and ``tip`` (M x 3), shaft ``radius`` and
    ``head_radius``/``head_length`` (M), ``rgb`` (M x 3), ``alpha``, the ``index`` of each
    arrow's particle, its ``anchor`` (particle position plus offset) and the style's ``shading``."""

    base: np.ndarray
    tip: np.ndarray
    radius: np.ndarray
    head_radius: np.ndarray
    head_length: np.ndarray
    rgb: np.ndarray
    alpha: float
    index: np.ndarray
    anchor: np.ndarray
    shading: str = "normal"
    name: str = ""

    @property
    def count(self) -> int:
        return len(self.base)

    @property
    def neck(self) -> np.ndarray:
        """Where the shaft ends and the head begins."""
        return self.tip - self.direction * self.head_length[:, None]

    @property
    def direction(self) -> np.ndarray:
        v = self.tip - self.base
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)


def property_vectors(data, name: str) -> np.ndarray:
    """``data[name]`` as N x 3 vectors. A scalar property (collinear magnetic moments, a
    charge) becomes a vector along z, so spin up/down shows as up/down arrows."""
    if name not in data:
        have = [k for k, v in data.particles.items() if np.ndim(v) == 2 and np.shape(v)[1] == 3 and k != "Position"]
        raise KeyError(f"no particle property {name!r}; vector properties here: {have or 'none'}")
    v = np.asarray(data[name], float)
    if v.ndim == 1:
        return np.column_stack([np.zeros_like(v), np.zeros_like(v), v])
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError(f"property {name!r} has shape {v.shape}; arrows need N x 3 (or N) values")
    return np.column_stack([v, np.zeros(len(v))]) if v.shape[1] == 2 else v


def auto_property(data) -> str:
    """The vector property VectorVis shows when none is named."""
    for name in AUTO_PROPERTIES:
        if name in data:
            return name
    for name, v in data.particles.items():
        if name not in ("Position", "Color", "Periodic Image") and np.ndim(v) == 2 and np.shape(v)[1] == 3:
            return name
    raise KeyError("no vector property to show (looked for " + ", ".join(AUTO_PROPERTIES) + ")")


def mapping_values(data, prop: str, frame: int = 0) -> np.ndarray:
    """Per-particle scalars for color mapping: a property (vectors by magnitude) or an expression."""
    if prop in data.particles:
        v = np.asarray(data.particles[prop], float)
    else:
        from .modifiers.expression import evaluate

        v = np.broadcast_to(np.asarray(evaluate(prop, data, frame), float), (data.count,))
    return np.linalg.norm(v, axis=1) if v.ndim == 2 else v


def coerce_styles(vectors) -> list[VectorStyle]:
    """``vectors`` as accepted by the renderers: a property name, a VectorStyle, a list of
    either, or ``{name: {style fields}}``."""
    if vectors is None or vectors is False:
        return []
    if isinstance(vectors, (str, VectorStyle)):
        vectors = [vectors]
    if isinstance(vectors, dict):
        vectors = [VectorStyle(property=k, **(v or {})) if not isinstance(v, VectorStyle) else v
                   for k, v in vectors.items()]
    return [v if isinstance(v, VectorStyle) else VectorStyle(property=str(v)) for v in vectors]


def scene_styles(data, vectors=None) -> list[VectorStyle]:
    """Styles a renderer draws: ``data.vectors`` (from VectorVis), then ``vectors`` (render argument,
    which replaces a ``data.vectors`` entry of the same property)."""
    out = dict(getattr(data, "vectors", {}) or {})
    for s in coerce_styles(vectors):
        out[s.property] = s
    return list(out.values())


def arrow_geometry(data, style: VectorStyle, frame: int = 0) -> Arrows:
    """World-space arrows for ``style`` on ``data`` (particles with Transparency 1 are left out)."""
    from .render import _rgb

    n = data.count
    v = property_vectors(data, style.property) * float(style.scaling) if n else np.zeros((0, 3))
    if style.reverse:
        v = -v
    length = np.linalg.norm(v, axis=1)
    keep = length > max(float(style.min_length), 1e-12)
    if style.only_selected:
        keep &= data.selection
    if "Transparency" in data:
        keep &= np.asarray(data["Transparency"], float) < 1
    idx = np.flatnonzero(keep)
    v, length = v[idx], length[idx]
    pos = data.positions[idx] + np.asarray(style.offset, float)
    shift = {"base": 0.0, "center": 0.5, "head": 1.0}[style.alignment]
    base = pos - shift * v
    tip = base + v
    r = np.full(len(idx), 0.5 * float(style.width))
    # A short arrow keeps its head proportions but never grows a head longer than itself.
    head_len = np.minimum(HEAD_LENGTH * r, length)
    head_rad = HEAD_RADIUS * r * np.where(HEAD_LENGTH * r > 0, head_len / np.maximum(HEAD_LENGTH * r, 1e-300), 1.0)
    if style.color_mapping_property:
        import matplotlib

        values = mapping_values(data, style.color_mapping_property, frame)
        if style.color_mapping_interval is not None:
            lo, hi = map(float, style.color_mapping_interval)
        else:
            lo, hi = (float(np.nanmin(values)), float(np.nanmax(values))) if n else (0.0, 1.0)
        t = np.clip((values[idx] - lo) / (hi - lo if hi != lo else 1.0), 0, 1)
        rgb = matplotlib.colormaps[style.cmap](t)[:, :3]
    else:
        rgb = np.tile(np.clip(_rgb(style.color), 0, 1), (len(idx), 1))
    return Arrows(base, tip, r, head_rad, head_len, rgb, 1.0 - float(style.transparency), idx, pos,
                  style.shading, style.property)


def scene_arrows(data, vectors=None, frame: int = 0) -> list[Arrows]:
    """Every arrow set a renderer draws (see :func:`scene_styles`); empty sets are dropped."""
    out = []
    for style in scene_styles(data, vectors):
        a = arrow_geometry(data, style, frame)
        if a.count:
            out.append(a)
    return out


def _frame(d):
    """Two unit vectors perpendicular to each row of ``d`` (M x 3, unit)."""
    helper = np.where(np.abs(d[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(d, e1)


def arrow_mesh(arrows: Arrows, sides: int = 12, shaft: bool = True):
    """Closed triangle mesh of every arrow: base disc, shaft, head disc, cone (``shaft=False``
    leaves the shaft out, for renderers with a cylinder primitive).

    Returns ``(vertices V x 3, faces F x 3, normals V x 3, arrow index of each vertex)``. Rings
    have their own vertices per surface, so normals stay sharp at the edges and smooth around.
    """
    m, s = arrows.count, int(max(3, sides))
    d = arrows.direction
    e1, e2 = _frame(d)
    ang = 2 * np.pi * np.arange(s) / s
    ring = np.cos(ang)[None, :, None] * e1[:, None] + np.sin(ang)[None, :, None] * e2[:, None]  # m x s x 3
    neck = arrows.neck
    r, R, hl = arrows.radius[:, None, None], arrows.head_radius[:, None, None], arrows.head_length
    base, tip = arrows.base[:, None], arrows.tip[:, None]
    # cone side normals tilt toward the tip by the half-angle
    slope = (arrows.head_radius / np.maximum(hl, 1e-300))[:, None, None]
    cone_n = ring + slope * d[:, None]
    cone_n /= np.linalg.norm(cone_n, axis=2, keepdims=True)
    back = np.broadcast_to(-d[:, None], (m, s, 3))
    groups = [  # (points m x s x 3, normals m x s x 3)
        (base + r * ring, back),                 # 0 base disc rim
        (base + r * ring, ring),                 # 1 shaft bottom
        (neck[:, None] + r * ring, ring),        # 2 shaft top
        (neck[:, None] + R * ring, back),        # 3 head disc rim (r ring inside it is covered)
        (neck[:, None] + R * ring, cone_n),      # 4 cone rim
        (np.broadcast_to(tip, (m, s, 3)), cone_n),  # 5 cone tip (one per side, for smooth normals)
    ]
    centers = [(arrows.base, -d), (neck, -d)]  # 6 base disc center, 7 head disc center
    per = 6 * s + 2
    verts = np.empty((m, per, 3), np.float32)  # half the memory of float64; plenty for display
    norms = np.empty((m, per, 3), np.float32)
    for g, (p, nrm) in enumerate(groups):
        verts[:, g * s:(g + 1) * s] = p
        norms[:, g * s:(g + 1) * s] = nrm
    for c, (p, nrm) in enumerate(centers):
        verts[:, 6 * s + c] = p
        norms[:, 6 * s + c] = nrm
    k = np.arange(s)
    k1 = (k + 1) % s
    G = lambda g, kk: g * s + kk
    tris = [np.column_stack([np.full(s, 6 * s), G(0, k1), G(0, k)]),         # base disc
            np.column_stack([np.full(s, 6 * s + 1), G(3, k1), G(3, k)]),      # head disc
            np.column_stack([G(4, k), G(4, k1), G(5, k)])]                    # cone
    if shaft:
        tris += [np.column_stack([G(1, k), G(1, k1), G(2, k1)]), np.column_stack([G(1, k), G(2, k1), G(2, k)])]
    local = np.vstack(tris)
    faces = (local[None] + per * np.arange(m)[:, None, None]).reshape(-1, 3)
    owner = np.repeat(np.arange(m), per)
    return verts.reshape(-1, 3), faces, norms.reshape(-1, 3), owner


def summary_line(name: str, style: VectorStyle) -> str:
    color = f"mapped by {style.color_mapping_property!r}" if style.color_mapping_property else f"color {style.color}"
    return f"vectors {name!r}: scaling {style.scaling:g}, width {style.width:g}, {style.alignment}, {color}"
