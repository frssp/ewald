"""The 3D structure view as code: every display setting of a pipeline in one :class:`View`.

``pipe.view`` belongs to the pipeline. Notebook cells, the GUI and Claude all set it, and
``pipe.render()``, ``snapshot(pipe)``, the Jupyter picture and the GUI viewport all draw with it::

    view(pipe, style="Ball & stick", colors={"O": "red"}, radii={"Ti": 0.6},
         bonds=True, cell=False, background="white", direction="z", zoom=1.3)

    pipe.view.bonds = False                  # the same, one setting at a time
    pipe.view.color("O", "crimson")
    pipe.view.look("x").rotate(azimuth=30, elevation=15)
    pipe.view.render("figure.png", renderer="raytrace")

A cell that is a single ``view(pipe, key=literal, ...)`` call is the notebook's *view cell*:
the GUI applies it when it runs and writes what you change in the viewport, the Particles and
Bonds editors or the style menu back into it, so the notebook stays the record of the picture.
:data:`SETTINGS` lists every setting with its default and meaning (``View.help()`` prints it);
:data:`STYLE_PRESETS` are the viewport style presets. Settings only change how the structure is
drawn; per-atom colors, sizes and hiding by condition are steps (:mod:`ewald.paint`) and
polyhedra, isosurfaces and vector properties come from modifiers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STYLE_PRESETS = {
    # name: (radius_scale, bonds, atom_style)
    "Ball & stick": (0.5, True, "auto"),
    "Space-filling": (1.0, False, "auto"),
    "Simple (ASE)": (0.7, True, "flat"),  # ASE GUI look: flat outlined discs, outlined sticks
    "Points": (0.25, False, "impostor"),
}
# Presets that also apply an ewald.themes theme when picked (ASE's white background).
PRESET_THEMES = {"Simple (ASE)": "ASE"}
DEFAULT_PRESET = "Ball & stick"
ATOM_STYLES = ("auto", "mesh", "impostor", "flat")
BOND_AUTO_LIMIT = 100_000   # automatic bonds stop above this many atoms (GUI: bonds.auto_limit)
DIRECTIONS = ("iso", "x", "-x", "y", "-y", "z", "-z", "top", "front", "persp")


@dataclass(frozen=True)
class Setting:
    name: str
    default: object
    group: str
    doc: str
    choices: tuple | None = None
    schema: dict | None = None   # JSON schema of the value (Claude's set_view tool)


def _s(name, default, group, doc, choices=None, **schema):
    if not schema:
        kind = {bool: "boolean", int: "number", float: "number", str: "string", dict: "object",
                list: "array"}.get(type(default))
        schema = {"type": kind} if kind else {}
    if choices:
        schema = {**schema, "enum": list(choices)}
    return Setting(name, default, group, doc, choices, schema)


_COLOR = {"type": ["string", "array", "null"], "items": {"type": "number"}}
SETTINGS: dict[str, Setting] = {s.name: s for s in [
    # style
    _s("style", DEFAULT_PRESET, "style", "Viewport style preset; sets radius_scale, bonds and atom_style "
       "('Simple (ASE)' also the ASE theme) unless those are given too.", tuple(STYLE_PRESETS)),
    _s("theme", None, "style", "Theme name (ewald.themes: 'VESTA', 'Journal', 'ASE', 'Dark slide', plugins): "
       "element colors, background, cell color and radius preset; explicit settings win.",
       type=["string", "null"]),
    _s("atom_style", "auto", "style", "How atoms are drawn: 'auto' (meshes, impostors for big systems), "
       "'mesh', 'impostor' (fast GPU spheres) or 'flat' (outlined discs, like ASE).", ATOM_STYLES),
    _s("background", None, "style", "Background color (any matplotlib color); None: the theme's, else the "
       "viewport's dark grey on screen and white in rendered images.", type=["string", "array", "null"]),
    # atoms
    _s("radius_scale", 0.5, "atoms", "Factor on every atom radius (the style preset sets it)."),
    _s("colors", {}, "atoms", "Color per particle type, e.g. {'O': 'red', 'Ti': '#7f7fff'}; others keep the "
       "element (or theme) color.", type="object", additionalProperties=_COLOR),
    _s("radii", {}, "atoms", "Radius per particle type in Angstrom (before radius_scale); wins over "
       "radius_preset.", type="object", additionalProperties={"type": "number"}),
    _s("radius_preset", "covalent", "atoms", "Radius table for the other types (ewald.radii).",
       ("covalent", "atomic", "atomic (calc.)", "vdW", "vdW (Bondi)", "ionic", "ionic (crystal)")),
    _s("oxidation_states", {}, "atoms", "Oxidation state per type for the ionic presets, e.g. {'Ti': 4}.",
       type="object", additionalProperties={"type": "number"}),
    # bonds
    _s("bonds", True, "bonds", "Draw bonds: True (VESTA-style pairs, or bond_pairs; off above "
       f"{BOND_AUTO_LIMIT:,} atoms), False, or a cutoff in Angstrom. Bonds a step created (CreateBonds) "
       "are always drawn.", type=["boolean", "number"]),
    _s("bond_pairs", [], "bonds", "Bond table [[A, B, dmin, dmax], ...]; empty: one row per type pair with "
       "dmax = bond_factor * (r_A + r_B), metal-metal pairs left out in compounds.",
       type="array", items={"type": "array"}),
    _s("bond_factor", 1.15, "bonds", "dmax factor of the automatic bond table (VESTA)."),
    _s("bond_radius", 0.15, "bonds", "Stick radius in Angstrom."),
    # cell
    _s("cell", True, "cell", "Draw the simulation cell."),
    _s("cell_width", 2.0, "cell", "Cell line width in pixels."),
    _s("cell_color", None, "cell", "Cell line color; None: black, or light grey on dark backgrounds.",
       type=["string", "array", "null"]),
    # boundary
    _s("boundary", None, "boundary", "VESTA boundary (fractional), e.g. [-0.1, 1.1] or three pairs: also draws "
       "periodic images inside it and completes bonds and polyhedra cut by the cell. Display only.",
       type=["array", "null"]),
    _s("boundary_bonds", "half", "boundary", "Bonds leaving the boundary: half, complete or none.",
       ("half", "complete", "none")),
    # elements
    _s("visuals", [], "elements", "Plugin visual elements drawn on top: names or {'name', 'params', 'enabled'}.",
       type="array"),
    _s("vectors", None, "elements", "Per-atom vector properties as arrows: a property name ('Force'), a list, "
       "or {name: {VectorStyle fields}} (scaling, width, color, ...).", type=["string", "array", "object", "null"]),
    _s("trajectory_width", 2.0, "elements", "Width in pixels of trajectory lines (GenerateTrajectoryLines); 0 hides them."),
    # camera
    _s("frame", 0, "camera", "Trajectory frame shown.", type="integer"),
    _s("direction", "iso", "camera", "View direction, from the structure towards the camera: 'x', '-y', 'z', "
       "'iso', ... or a vector [x, y, z].", type=["string", "array"]),
    _s("zoom", 1.0, "camera", "Zoom factor on the framed structure (2 = twice as big)."),
    _s("orthographic", True, "camera", "Parallel projection (True) or perspective."),
    _s("camera", None, "camera", "An exact camera (position, focal_point, up, parallel_projection, "
       "parallel_scale, view_angle) as the GUI saves it; wins over direction and zoom.", type=["object", "null"]),
    _s("region", None, "camera", "Part of the view a picture shows (the Render frame): [x, y, height], the "
       "frame's center offset from the view center and its height, as fractions of the view height; its width "
       "follows the picture size. None: the whole view height, centered.", type=["array", "null"]),
    # picture
    _s("renderer", None, "picture", "Renderer for pictures: 'opengl', 'raytrace' or 'vector'; None: the "
       "caller's default.", ("opengl", "raytrace", "vector", None)),
    _s("quality", None, "picture", "'draft', 'standard' or 'publication' sampling; None: the caller's default.",
       ("draft", "standard", "publication", None)),
    _s("size", None, "picture", "Picture size [width, height] in pixels; None: the caller's default.",
       type=["array", "null"]),
    _s("ambient_occlusion", None, "picture", "Ambient occlusion (soft contact shadows); None: from quality.",
       type=["boolean", "null"]),
    _s("shadows", None, "picture", "Cast shadows (ray tracer); None: from quality.", type=["boolean", "null"]),
    _s("transparent_background", False, "picture", "Transparent PNG background."),
    _s("text", None, "picture", "Text label; [Frame], [Timestep], [Time] and [attribute] are filled in.",
       type=["string", "null"]),
    _s("legend", True, "picture", "Color bar of a ColorCoding step."),
    _s("type_legend", False, "picture", "One swatch per particle type."),
    _s("scale_bar", None, "picture", "Scale bar: True (a round length) or a length in Angstrom.",
       type=["boolean", "number", "null"]),
    _s("overlays", [], "picture", "Overlay elements: 'Coordinate tripod', 'Attributes', plugins; names or "
       "{'name', 'params'}.", type="array"),
]}
# keys of render() that are not named like the setting
_RENDER_NAMES = {"colors": "type_colors", "radii": "type_radii", "cell": "show_cell"}
# settings a style preset implies (index into STYLE_PRESETS values)
_PRESET_KEYS = {"radius_scale": 0, "bonds": 1, "atom_style": 2}
GROUPS = ("style", "atoms", "bonds", "cell", "boundary", "elements", "camera", "picture")


class View:
    """How a pipeline's structure is drawn: style, colors, radii, bonds, cell, boundary, camera and
    picture settings (see :data:`SETTINGS`). Read and set them as attributes (``view.bonds``,
    ``view.colors``), several at once with :meth:`update`, and draw with :meth:`render`.
    Only what was set is stored (:meth:`changed`); the rest follows the style preset and defaults."""

    def __init__(self, pipeline=None, **settings):
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "version", 0)   # bumps on every change (the GUI follows it)
        self.update(**settings)

    # ------------------------------------------------------------ values
    def __getattr__(self, name):
        if name.startswith("_") or name not in SETTINGS:
            raise AttributeError(f"View has no setting {name!r}; settings: {', '.join(SETTINGS)}")
        values = self._values
        if name in values:
            return values[name]
        if name in _PRESET_KEYS:
            return STYLE_PRESETS[values.get("style", DEFAULT_PRESET)][_PRESET_KEYS[name]]
        if name == "theme":
            return PRESET_THEMES.get(values.get("style", DEFAULT_PRESET))
        default = SETTINGS[name].default
        return type(default)() if isinstance(default, (dict, list)) else default

    def __setattr__(self, name, value):
        if name in ("pipeline", "version"):
            object.__setattr__(self, name, value)
            return
        self.update(**{name: value})

    def __dir__(self):
        return list(super().__dir__()) + list(SETTINGS)

    def update(self, **settings) -> "View":
        """Set several settings at once; ``style`` goes first, so ``update(style='Points', bonds=True)``
        keeps the bonds. ``None`` (or the default) for a setting goes back to its default. Returns self."""
        for name in settings:
            if name not in SETTINGS:
                raise KeyError(f"no view setting {name!r}; settings: {', '.join(SETTINGS)}")
        order = sorted(settings, key=lambda n: n != "style")
        values = dict(self._values)
        for name in order:
            value = _check(name, settings[name])
            if name == "style":
                for k in _PRESET_KEYS:   # a preset sets these, like picking it in the viewport
                    values.pop(k, None)
                if value is None or value == DEFAULT_PRESET:
                    values.pop("style", None)
                else:
                    values["style"] = value
                if "theme" not in settings and values.get("theme") in PRESET_THEMES.values():
                    values.pop("theme")
                continue
            if name == "camera" and value is not None and ("direction" not in settings and "zoom" not in settings):
                values.pop("direction", None)
                values.pop("zoom", None)
            if name in ("direction", "zoom") and "camera" not in settings:
                values.pop("camera", None)   # a new direction or zoom frames the structure again
            values[name] = value
            if _same(value, self._implied(name, values)):
                values.pop(name)
        if values != self._values:
            object.__setattr__(self, "_values", values)
            object.__setattr__(self, "version", self.version + 1)
        return self

    def _implied(self, name, values):
        """What ``name`` would be without an explicit value, given the other ``values``."""
        if name in _PRESET_KEYS:
            return STYLE_PRESETS[values.get("style", DEFAULT_PRESET)][_PRESET_KEYS[name]]
        if name == "theme":
            return PRESET_THEMES.get(values.get("style", DEFAULT_PRESET))
        return SETTINGS[name].default

    def reset(self, *names) -> "View":
        """Back to the defaults: every setting, or only ``names``."""
        values = {} if not names else {k: v for k, v in self._values.items() if k not in names}
        if values != self._values:
            object.__setattr__(self, "_values", values)
            object.__setattr__(self, "version", self.version + 1)
        return self

    def changed(self) -> dict:
        """The settings that differ from the defaults, in :data:`SETTINGS` order (what a view cell writes)."""
        return {k: _copy(self._values[k]) for k in SETTINGS if k in self._values}

    def settings(self) -> dict:
        """Every setting with its current value."""
        return {k: _copy(getattr(self, k)) for k in SETTINGS}

    def copy(self, pipeline=None) -> "View":
        return View(pipeline if pipeline is not None else self.pipeline, **self.changed())

    @staticmethod
    def help() -> str:
        """The settings, their defaults and what they do, by group."""
        lines = []
        for group in GROUPS:
            lines.append(f"[{group}]")
            lines += [f"  {s.name} = {s.default!r}: {s.doc}" for s in SETTINGS.values() if s.group == group]
        return "\n".join(lines)

    # ------------------------------------------------------------ helpers
    def color(self, types, color) -> "View":
        """Color particle type(s) ``types`` (a name or a list); ``None`` gives back the default color."""
        colors = dict(self.colors)
        for t in [types] if isinstance(types, str) else types:
            if color is None:
                colors.pop(str(t), None)
            else:
                colors[str(t)] = _color(color)
        return self.update(colors=colors)

    def radius(self, types, radius) -> "View":
        """Radius in Angstrom of particle type(s) ``types``; ``None`` gives back the preset's."""
        radii = dict(self.radii)
        for t in [types] if isinstance(types, str) else types:
            if radius is None:
                radii.pop(str(t), None)
            else:
                radii[str(t)] = float(radius)
        return self.update(radii=radii)

    def look(self, direction="iso", zoom: float | None = None) -> "View":
        """Look along ``direction`` ('x', '-z', 'iso', [1, 1, 0], ...), framing the structure again."""
        kw = {"direction": direction}
        if zoom is not None:
            kw["zoom"] = zoom
        return self.update(**kw)

    def rotate(self, azimuth: float = 0.0, elevation: float = 0.0) -> "View":
        """Turn the view by ``azimuth`` degrees around the vertical (z) axis and tilt it by ``elevation``
        degrees up; works on the direction, or on an exact camera when one is set."""
        if self.camera is not None:
            cam = dict(self.camera)
            focal = np.asarray(cam["focal_point"], float)
            d = np.asarray(cam["position"], float) - focal
            d = _turn(d, azimuth, elevation, np.asarray(cam.get("up", (0, 0, 1)), float))
            cam["position"] = tuple(float(x) for x in focal + d)
            return self.update(camera=cam)
        d = _turn(direction_vector(self.direction), azimuth, elevation, np.array([0, 0, 1.0]))
        return self.update(direction=[round(float(x), 6) for x in d])

    def zoom_by(self, factor: float) -> "View":
        """Zoom in (``factor`` > 1) or out."""
        if self.camera is not None:
            cam = dict(self.camera)
            cam["parallel_scale"] = float(cam.get("parallel_scale", 1.0)) / factor
            if not cam.get("parallel_projection", True):
                focal = np.asarray(cam["focal_point"], float)
                cam["position"] = tuple(float(x) for x in focal + (np.asarray(cam["position"], float) - focal) / factor)
            return self.update(camera=cam)
        return self.update(zoom=round(self.zoom * factor, 6))

    # ------------------------------------------------------------ drawing
    def render_kwargs(self, data=None, frame: int | None = None) -> dict:
        """Keyword arguments of :func:`ewald.render.render` for this view (``data`` fills in the automatic
        bond table and the text label). Settings left at None are left out, so the caller's defaults hold."""
        kw = {}
        for name in SETTINGS:
            if name in ("style", "frame", "bonds", "bond_pairs", "bond_factor", "text"):
                continue
            value = getattr(self, name)
            if value is None or (isinstance(value, (dict, list)) and not value):
                continue
            kw[_RENDER_NAMES.get(name, name)] = _copy(value)
        if "type_colors" in kw:
            kw["type_colors"] = {k: _color(v) for k, v in kw["type_colors"].items()}
        if kw.get("camera") is not None:
            kw.pop("direction", None)
            kw.pop("zoom", None)
        bonds = self.bonds
        if bonds is True:
            if self.bond_pairs:
                kw["bond_pairs"] = [tuple(p) for p in self.bond_pairs]
            elif data is not None and 0 < data.count <= BOND_AUTO_LIMIT:
                from .bonds import covalent_pairs

                kw["bond_pairs"] = covalent_pairs(np.unique(np.asarray(data.types)), self.bond_factor)
        elif bonds not in (False, None, 0):
            kw["bonds"] = float(bonds)
        if self.text:
            from .overlays import format_label

            f = self.frame if frame is None else frame
            kw["text"] = format_label(self.text, f, data)
        return kw

    def _data(self, frame=None):
        if self.pipeline is None:
            raise ValueError("this view belongs to no pipeline: use pipe.view, or render(data, **view.render_kwargs(data))")
        return self.pipeline.compute(self.frame if frame is None else frame)

    def render(self, filename=None, frame: int | None = None, **overrides):
        """Draw the pipeline with this view (:func:`ewald.render.render`); ``overrides`` win for this call."""
        from .render import render

        data = self._data(frame)
        return render(data, filename, **merge(self.render_kwargs(data, frame), overrides))

    def snapshot(self, frame: int | None = None, **overrides):
        """A :class:`ewald.views.Snapshot` (an image under the cell) of this view."""
        from .views import Snapshot

        return Snapshot(self.pipeline, self.frame if frame is None else frame, **overrides)

    def _repr_png_(self):
        if self.pipeline is None:
            return None
        return self.snapshot()._repr_png_()

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.changed().items())
        return f"View({args})"

    def source(self, pipeline_name: str = "pipe") -> str:
        """The view cell for this view: ``view(pipe, style=..., ...)``."""
        return view_source(pipeline_name, **self.changed())


def merge(kwargs: dict, overrides: dict) -> dict:
    """``kwargs`` of :meth:`View.render_kwargs` with ``overrides`` (render() keywords) on top: an override
    also drops what it replaces (``bonds=2.8`` the bond table, ``direction=`` the exact camera, ...)."""
    out = dict(kwargs)
    for key, drops in (("bonds", ("bond_pairs",)), ("bond_pairs", ("bonds",)), ("camera", ("direction", "zoom")),
                       ("direction", ("camera",)), ("zoom", ("camera",))):
        if key in overrides:
            for d in drops:
                out.pop(d, None)
    out.update(overrides)
    return out


def view(pipe, **settings) -> View:
    """Set display settings of ``pipe``'s 3D view (see :data:`SETTINGS`, ``View.help()``) and return
    ``pipe.view``; as the last line of a cell it shows the picture. Settings not named keep their
    value. ``view(pipe)`` alone just shows it."""
    v = pipe.view if hasattr(pipe, "view") else View()
    return v.update(**settings)


def view_source(pipeline_name: str = "pipe", **settings) -> str:
    """Source of a view cell, one setting per line when there are several."""
    items = [f"{k}={_literal(v)}" for k, v in settings.items()]
    if not items:
        return f"view({pipeline_name})"
    one = f"view({pipeline_name}, {', '.join(items)})"
    if len(one) <= 100:
        return one
    return f"view({pipeline_name},\n" + "".join(f"     {item},\n" for item in items) + ")"


def view_cell_args(source: str) -> dict | None:
    """Settings of a cell that is a single ``view(<name>, key=literal, ...)`` call (optionally after
    imports), with ``source`` the pipeline's name; None for any other cell."""
    from .views import view_args

    args = view_args(source, function="view")
    if args is None or any(k not in SETTINGS for k in args if k != "source"):
        return None
    return args


# ---------------------------------------------------------------- helpers
def direction_vector(direction) -> np.ndarray:
    from .render import _DIRECTIONS

    d = np.asarray(_DIRECTIONS.get(direction, direction) if isinstance(direction, str) else direction, float)
    n = np.linalg.norm(d)
    if d.shape != (3,) or not n:
        raise ValueError(f"direction must be one of {DIRECTIONS} or a nonzero [x, y, z], not {direction!r}")
    return d / n


def _turn(d, azimuth, elevation, up):
    up = up / (np.linalg.norm(up) or 1.0)
    out = _rodrigues(d, up, math.radians(azimuth))
    side = np.cross(up, out)
    if np.linalg.norm(side) < 1e-9:   # looking straight down the up axis: tilt towards -y, like the viewport
        side = np.cross(up, [0.0, -1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0, 0])
    if elevation:
        out = _rodrigues(out, side / np.linalg.norm(side), -math.radians(elevation))
    return out


def _rodrigues(v, k, angle):
    return v * math.cos(angle) + np.cross(k, v) * math.sin(angle) + k * np.dot(k, v) * (1 - math.cos(angle))


def _color(c):
    """A color as stored in a view: names and hex strings stay, RGB becomes a float tuple."""
    if isinstance(c, str):
        return c
    rgb = np.asarray(c, float)
    if rgb.shape not in ((3,), (4,)):
        raise ValueError(f"not a color: {c!r}")
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return tuple(round(float(x), 4) for x in rgb[:3])


def _check(name, value):
    s = SETTINGS[name]
    if value is None:
        return s.default if not isinstance(s.default, (dict, list)) else type(s.default)()
    if s.choices is not None and value not in s.choices:
        raise ValueError(f"{name} must be one of {[c for c in s.choices if c is not None]}, not {value!r}")
    if name == "theme":
        from .themes import get_theme

        get_theme(value)   # unknown names fail here
    elif name == "direction":
        direction_vector(value)
        if not isinstance(value, str):
            value = [float(x) for x in value]
    elif name in ("colors",):
        if not isinstance(value, dict):
            raise TypeError("colors is {type: color}")
        value = {str(k): _color(v) for k, v in value.items()}
    elif name in ("radii", "oxidation_states"):
        if not isinstance(value, dict):
            raise TypeError(f"{name} is {{type: number}}")
        value = {str(k): (float(v) if name == "radii" else v) for k, v in value.items()}
    elif name == "bond_pairs":
        value = [[str(p[0]), str(p[1]), float(p[2]), float(p[3])] for p in value]
    elif name == "bonds":
        if not isinstance(value, bool):
            value = float(value)
            if value <= 0:
                value = False
    elif name in ("radius_scale", "bond_factor", "bond_radius", "cell_width", "zoom", "trajectory_width"):
        value = float(value)
        if value < 0 or (value == 0 and name in ("radius_scale", "zoom", "bond_factor")):
            raise ValueError(f"{name} must be positive, not {value}")
    elif name == "frame":
        value = int(value)
    elif name in ("cell", "orthographic", "legend", "type_legend", "transparent_background"):
        value = bool(value)
    elif name == "size":
        value = [int(value[0]), int(value[1])]
    elif name == "region":
        from .framing import as_region

        value = list(as_region(value))
    elif name == "boundary":
        value = [list(map(float, p)) for p in value] if np.ndim(value) == 2 else [float(x) for x in np.ravel(value)]
    elif name == "camera":
        missing = {"position", "focal_point", "up"} - set(value)
        if missing:
            raise ValueError(f"camera needs {sorted(missing)}")
        value = {k: (tuple(float(x) for x in v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                 for k, v in value.items()}
    elif name in ("visuals", "overlays"):
        value = [v if isinstance(v, str) else dict(v) for v in ([value] if isinstance(value, (str, dict)) else value)]
    elif name in ("background", "cell_color") and not isinstance(value, str):
        value = _color(value)
    return value


def _same(a, b) -> bool:
    try:
        return bool(a == b) and type(a) is type(b) or (isinstance(a, (int, float)) and not isinstance(a, bool)
                                                        and not isinstance(b, bool) and a == b)
    except Exception:  # noqa: BLE001 - arrays and such
        return False


def _copy(v):
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def _literal(v) -> str:
    """repr that reads back with ast.literal_eval (tuples of floats short)."""
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, tuple):
        return "(" + ", ".join(_literal(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_literal(x)}" for k, x in v.items()) + "}"
    return repr(v)
