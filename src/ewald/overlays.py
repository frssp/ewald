"""Figure overlays shared by every renderer: text label, color legend, type legend, scale bar,
and overlay elements (built-in or ``@overlay`` plugins).

The layout is computed once in pixel coordinates (origin top-left) by
:func:`overlay_layout`; raster renderers draw it with PIL (:func:`draw_overlays`)
and the vector renderer draws the same items as vector text and shapes.

An :class:`OverlayElement` returns 2D items (:class:`Text`, :class:`Rect`, :class:`Polyline`,
:class:`Circle`, :class:`Image`) from ``layout(data, ctx)``; every renderer draws them, so a
plugin overlay appears in OpenGL, ray-traced and SVG/PDF figures alike.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------ 2D items (pixels, origin top-left)
@dataclass
class Text:
    """``text`` at (x, y); ``anchor`` = horizontal l/m/r + vertical t/m/b; ``size`` in px (None: the
    figure's font size); ``color`` None is the ink color (dark on light backgrounds, light on dark).
    ``halo`` (a color) draws an outline around the letters so the text reads on top of a picture."""

    x: float
    y: float
    text: str
    size: float | None = None
    color: object = None
    anchor: str = "lt"
    bold: bool = False
    halo: object = None


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    fill: object = None          # None: not filled
    outline: object = None       # None: no outline; "ink" for the ink color
    width: float = 1.0


@dataclass
class Polyline:
    points: object               # (n, 2) pixels
    color: object = None         # None: ink
    width: float = 1.0
    closed: bool = False
    fill: object = None          # fill color of a closed polyline (a polygon)


@dataclass
class Circle:
    x: float
    y: float
    r: float
    fill: object = None
    outline: object = None       # None: no outline; "ink" for the ink color
    width: float = 1.0


@dataclass
class Image:
    """An RGB(A) uint8 image with its top-left corner at (x, y), scaled to ``width`` x ``height`` px."""

    x: float
    y: float
    rgba: object
    width: float | None = None
    height: float | None = None


ITEMS = (Text, Rect, Polyline, Circle, Image)


@dataclass
class OverlayContext:
    """What an overlay may use besides the data: image ``size`` (w, h) in px, the ``camera`` dict
    (None when unknown), ``frame``, the figure's ``font_size``/``small`` sizes in px and ``ink``,
    the RGB text color that reads on the background."""

    size: tuple
    camera: dict | None = None
    frame: int = 0
    font_size: int = 12
    small: int = 10
    ink: tuple = (0.08, 0.08, 0.08)
    background: object = "white"
    extras: dict = field(default_factory=dict)

    def corner(self, corner: str, w: float, h: float, margin: float = 0.02) -> tuple[float, float]:
        """Top-left of a ``w`` x ``h`` box in a corner ('top-left', 'top-right', 'bottom-left',
        'bottom-right') with a margin of ``margin`` x the image height."""
        W, H = self.size
        m = margin * H
        x = W - m - w if corner.endswith("right") else m
        y = H - m - h if corner.startswith("bottom") else m
        return x, y


class OverlayElement:
    """Base class of overlays: ``layout(data, ctx)`` returns 2D items in pixels.

    Store every constructor argument as an attribute of the same name (as for modifiers and
    visual elements): :meth:`parameters`, sessions, the GUI form and the Claude tools read them.
    ``layout`` must not modify ``data``; ``enabled=False`` keeps an overlay without drawing it.
    """

    name: str = ""
    enabled: bool = True

    def applies(self, data) -> bool:
        return True

    def layout(self, data, ctx: OverlayContext) -> list:
        raise NotImplementedError

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


def camera_axes(camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit (right, up, forward) vectors of a camera dict; forward points into the screen."""
    pos, focal = np.asarray(camera["position"], float), np.asarray(camera["focal_point"], float)
    fwd = focal - pos
    fwd /= np.linalg.norm(fwd) or 1.0
    up = np.asarray(camera.get("up", (0, 0, 1)), float)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(fwd, [0.0, 1.0, 0.0] if abs(fwd[1]) < 0.9 else [1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    return right, np.cross(right, fwd), fwd


class CoordinateTripod(OverlayElement):
    """Axis directions as arrows in a corner that turn with the camera: Cartesian x, y, z or the
    cell vectors a, b, c.

    ``axes`` is "xyz" or "abc"; ``size`` is the arrow length as a fraction of the image height.
    """

    name = "Coordinate tripod"

    def __init__(self, axes: str = "xyz", corner: str = "bottom-right", size: float = 0.09):
        self.axes = axes
        self.corner = corner
        self.size = size

    def applies(self, data):
        return self.axes != "abc" or (data is not None and data.cell.is_valid)

    def layout(self, data, ctx):
        if ctx.camera is None:
            return []
        if self.axes == "abc":
            vecs = np.asarray(data.cell.matrix, float)
            labels = ("a", "b", "c")
        else:
            vecs, labels = np.eye(3), ("x", "y", "z")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        right, up, fwd = camera_axes(ctx.camera)
        L = self.size * ctx.size[1]
        pad = 1.5 * ctx.font_size
        ox, oy = ctx.corner(self.corner, 2 * (L + pad), 2 * (L + pad))
        ox, oy = ox + L + pad, oy + L + pad
        colors = ((0.85, 0.15, 0.15), (0.15, 0.6, 0.2), (0.15, 0.3, 0.9))
        out = []
        lw = max(1.5, ctx.size[1] / 300)
        for k in np.argsort(-(vecs @ fwd)):        # far arrows first
            d = np.array([vecs[k] @ right, -(vecs[k] @ up)])
            tip = np.array([ox, oy]) + L * d
            out.append(Polyline([[ox, oy], tip], colors[k], width=lw))
            n = np.linalg.norm(d)
            if n > 0.05:
                u = d / n
                head = 0.22 * L
                side = np.array([-u[1], u[0]]) * 0.45 * head
                base = tip - u * head
                out.append(Polyline([tip, base + side, base - side], colors[k], width=0, closed=True,
                                    fill=colors[k]))
            lab = tip + (d / n if n > 0.05 else np.array([0.7, -0.7])) * 0.55 * ctx.font_size
            out.append(Text(float(lab[0]), float(lab[1]), labels[k], ctx.small, colors[k], "mm", True))
        return out


class Attributes(OverlayElement):
    """Global attributes (temperature, energy, timestep, ...) as "name = value" lines in a corner.

    ``attributes`` is a comma-separated list; empty shows every numeric attribute (up to 8).
    """

    name = "Attributes"

    def __init__(self, attributes: str = "", corner: str = "top-right", format: str = "{:.4g}"):
        self.attributes = attributes
        self.corner = corner
        self.format = format

    def layout(self, data, ctx):
        attrs = getattr(data, "attributes", {}) if data is not None else {}
        wanted = [a.strip() for a in self.attributes.split(",") if a.strip()]
        if not wanted:
            wanted = [k for k, v in attrs.items() if isinstance(v, (int, float, np.integer, np.floating))
                      and not isinstance(v, bool)][:8]
        lines = []
        for key in wanted:
            if key not in attrs:
                continue
            v = attrs[key]
            text = self.format.format(v) if isinstance(v, (float, np.floating)) else str(v)
            lines.append(f"{key} = {text}")
        if not lines:
            return []
        fs = ctx.small
        width = 0.62 * fs * max(map(len, lines))
        x, y = ctx.corner(self.corner, width, 1.35 * fs * len(lines))
        right = self.corner.endswith("right")
        return [Text(x + width if right else x, y + 1.35 * fs * i, line, fs, None, "rt" if right else "lt")
                for i, line in enumerate(lines)]


class AtomLabels(OverlayElement):
    """A text label on every atom, e.g. its type and index. ``text`` is a format string with ``{type}``,
    ``{index}`` (0-based), ``{id}`` (the Particle Identifier, else index + 1) and any per-atom property
    whose name has no spaces (``{Charge:.2f}``); a bare 'type', 'index', 'id' or property name means that
    field. ``types`` limits the labels to those particle types; ``size`` is the font size in px (None: the
    figure's small size); ``color`` None is the ink color. Hidden atoms (Transparency 1) get no label, and
    at most ``max_labels`` atoms are labelled (the nearest to the camera)."""

    name = "Atom labels"

    def __init__(self, text: str = "{type}", types: list | None = None, size: float | None = None, color=None,
                 max_labels: int = 2000):
        self.text = text
        self.types = types
        self.size = size
        self.color = color
        self.max_labels = max_labels

    def applies(self, data):
        return data is not None and data.count > 0

    def layout(self, data, ctx):
        if ctx.camera is None:
            return []
        keep = atom_label_mask(data, self.types)
        idx = np.flatnonzero(keep)
        if not len(idx):
            return []
        from .vector import _project

        x, y, depth, _ = _project(np.asarray(data.positions, float)[idx], ctx.camera, ctx.size)
        w, h = ctx.size
        inside = (x >= 0) & (x <= w) & (y >= 0) & (y <= h) & (depth > 0)
        idx, x, y, depth = idx[inside], x[inside], y[inside], depth[inside]
        order = np.argsort(depth)[: int(self.max_labels)][::-1]   # the nearest ones, far first
        texts = atom_label_texts(data, idx[order], self.text)
        size = self.size or ctx.small
        halo = (1.0, 1.0, 1.0) if sum(ctx.ink) < 1.5 else (0.0, 0.0, 0.0)   # reads on top of the atoms
        return [Text(float(x[k]), float(y[k]), t, size, self.color, "mm", True, halo) for k, t in zip(order, texts)]


class MeasurementLabels(OverlayElement):
    """The values of the view's ``measurements`` (:mod:`ewald.measure`) next to their lines: a distance in
    Angstrom at its middle, an angle at its vertex, a dihedral at its middle bond. ``size`` in px (None: the
    figure's small size); ``color`` None is the measurement line color."""

    name = "Measurements"

    def __init__(self, measurements: list | None = None, size: float | None = None, color=None):
        self.measurements = measurements or []
        self.size = size
        self.color = color

    def applies(self, data):
        return data is not None and data.count > 0 and bool(self.measurements)

    def layout(self, data, ctx):
        from . import measure
        from .scene import MEASUREMENT_COLOR

        if ctx.camera is None:
            return []
        items = [m for m in measure.check(self.measurements) if max(m) < data.count]
        if not items:
            return []
        from .vector import _project

        at = np.array([measure.anchor(data, m) for m in items])
        x, y, depth, _ = _project(at, ctx.camera, ctx.size)
        halo = (1.0, 1.0, 1.0) if sum(ctx.ink) < 1.5 else (0.0, 0.0, 0.0)
        color = self.color or MEASUREMENT_COLOR
        size = self.size or ctx.small
        return [Text(float(x[k]), float(y[k]) - 0.6 * size, measure.text(data, m), size, color, "mb", True, halo)
                for k, m in enumerate(items) if depth[k] > 0]


def atom_label_mask(data, types=None) -> np.ndarray:
    """Which atoms get a label: visible ones (Transparency < 1) of ``types`` (all when None)."""
    keep = np.ones(data.count, bool)
    if "Transparency" in data:
        keep &= np.asarray(data["Transparency"], float) < 1
    if types:
        keep &= np.isin(np.asarray(data.types).astype(str), [str(t) for t in types])
    return keep


def atom_label_texts(data, indices, template: str = "{type}") -> list[str]:
    """The label of each atom in ``indices`` (see :class:`AtomLabels`)."""
    template = str(template)
    if "{" not in template:
        template = "{" + template + "}"
    types = np.asarray(data.types)
    ids = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else None
    fields = {name: np.asarray(data[name]) for name in data.particles
              if " " not in name and "{" + name in template and np.asarray(data[name]).ndim == 1}
    out = []
    for i in np.asarray(indices, np.int64):
        values = {k: v[i].item() if hasattr(v[i], "item") else v[i] for k, v in fields.items()}
        values.update(type=str(types[i]), index=int(i), id=int(ids[i]) if ids is not None else int(i) + 1)
        try:
            out.append(template.format(**values))
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"atom label {template!r}: {exc!r}; fields are type, index, id and per-atom "
                             "properties without spaces") from None
    return out


def label_params(labels) -> dict | None:
    """:class:`AtomLabels` parameters from the ``labels`` setting: None/False (none), True ('{type}'),
    a text template, or a dict with ``text``, ``types``, ``size``, ``color``, ``max_labels``."""
    if labels is None or labels is False:
        return None
    if labels is True:
        return {"text": "{type}"}
    if isinstance(labels, str):
        return {"text": labels}
    if isinstance(labels, dict):
        unknown = set(labels) - {"text", "types", "size", "color", "max_labels"}
        if unknown:
            raise KeyError(f"labels: unknown {sorted(unknown)}; use text, types, size, color, max_labels")
        return {"text": "{type}", **labels}
    raise TypeError(f"labels is True, a text like '{{type}}{{index}}' or a dict, not {labels!r}")


BUILTIN = (CoordinateTripod, Attributes, AtomLabels, MeasurementLabels)


def coerce_overlays(overlays) -> list[OverlayElement]:
    """``overlays`` as accepted by ``render()``: elements, registered names, or ``{"name": ...,
    "params": {...}, "enabled": ...}`` dicts (the session form)."""
    if not overlays:
        return []
    if isinstance(overlays, (OverlayElement, str, dict)):
        overlays = [overlays]
    out = []
    for v in overlays:
        if isinstance(v, OverlayElement):
            out.append(v)
            continue
        from .plugins import overlays as registry

        name, params = (v, {}) if isinstance(v, str) else (v["name"], dict(v.get("params") or {}))
        reg = registry()
        if name not in reg:
            raise KeyError(f"no overlay {name!r}; registered: {', '.join(reg) or 'none'}")
        el = reg[name].cls(**params)
        el.enabled = bool(v.get("enabled", True)) if isinstance(v, dict) else True
        out.append(el)
    return out


def overlay_to_dict(el: OverlayElement) -> dict:
    """``{"name", "params", "enabled"}``, the form sessions, the GUI and the Claude tools store."""
    from .plugins import overlay_name

    return {"name": overlay_name(type(el)), "params": el.parameters(), "enabled": bool(el.enabled)}


def overlay_items(data, ctx: OverlayContext, overlays) -> list:
    """The 2D items of every enabled overlay element that applies to ``data``."""
    items = []
    for el in coerce_overlays(overlays):
        if not el.enabled or not el.applies(data):
            continue
        out = el.layout(data, ctx) or []
        for item in out if isinstance(out, (list, tuple)) else [out]:
            if not isinstance(item, ITEMS):
                raise TypeError(f"overlay {el.title!r} returned {type(item).__name__}, not an overlay item")
            items.append(item)
    return items


def format_label(template: str | None, frame: int, data=None, time_per_frame: float | None = None) -> str:
    """Expand ``[Frame]``, ``[Timestep]``, ``[Time]`` and ``[attribute name]`` in a text label.

    ``[Time]`` is the ``Time`` attribute when the reader provides one, else
    ``frame * time_per_frame`` (1 per frame when not given).
    """
    if not template:
        return ""
    attrs = getattr(data, "attributes", {}) if data is not None else {}
    if "Time" in attrs and time_per_frame is None:
        time = attrs["Time"]
    else:
        time = frame * (1.0 if time_per_frame is None else time_per_frame)
    out = (template.replace("[Frame]", str(frame)).replace("[Timestep]", str(attrs.get("Timestep", frame)))
           .replace("[Time]", f"{float(time):g}"))
    for key, value in attrs.items():
        token = f"[{key}]"
        if token in out:
            out = out.replace(token, f"{value:g}" if isinstance(value, (float, np.floating)) else str(value))
    return out


def pixels_per_angstrom(camera: dict, size) -> float:
    """Image scale at the focal plane (exact for orthographic views)."""
    h = float(size[1])
    if camera.get("parallel_projection", True):
        return h / (2.0 * float(camera["parallel_scale"]))
    dist = float(np.linalg.norm(np.subtract(camera["focal_point"], camera["position"])))
    return h / (2.0 * dist * math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2))


def nice_length(target: float) -> float:
    """Largest 1, 2 or 5 x 10^k not above ``target`` (scale bar length)."""
    if target <= 0:
        return 1.0
    k = math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * 10 ** k <= target:
            return float(m * 10 ** k)
    return float(10 ** k)


def type_entries(data, type_colors: dict | None = None) -> list[tuple[str, np.ndarray]]:
    """(type name, RGB) for every particle type, in order of first appearance."""
    from .render import _rgb, type_color

    if data is None or not data.count:
        return []
    out = []
    for k, t in enumerate(dict.fromkeys(data.types.tolist())):
        c = (type_colors or {}).get(t)
        out.append((str(t), np.clip(_rgb(c) if c is not None else type_color(t, k), 0, 1)))
    return out


def is_dark(background, transparent: bool = False) -> bool:
    """Whether lines and text on ``background`` should be light (never on a transparent one)."""
    from .colormaps import to_rgb

    return not transparent and sum(to_rgb(background)) < 1.5


def ink_color(background, transparent: bool) -> tuple:
    return (0.92, 0.92, 0.93) if is_dark(background, transparent) else (0.08, 0.08, 0.08)


def cell_line_color(background, transparent: bool = False) -> str:
    """Default simulation-cell color: light gray on dark backgrounds, black otherwise."""
    return "#AEB4BE" if is_dark(background, transparent) else "black"


def overlay_layout(data, size, *, text=None, legend=True, type_legend=False, scale_bar=None, camera=None,
                   type_colors=None, overlays=None, frame=0, background="white", transparent=False) -> dict:
    """Positions (pixels, origin top-left) and contents of the requested overlays.

    ``scale_bar``: True picks a round length (about a fifth of the width), a number
    gives the length in Angstrom; needs ``camera``. ``overlays``: overlay elements (see
    :func:`coerce_overlays`), laid out under ``"items"``.
    """
    w, h = int(size[0]), int(size[1])
    fs = max(10, int(h * 0.03))
    out = {"font_size": fs, "small": int(fs * 0.8)}
    y = int(0.02 * h)
    if text:
        out["text"] = {"x": int(0.02 * w), "y": y, "text": str(text)}
        y += int(1.6 * fs)
    if type_legend:
        rows = []
        for name, rgb in type_entries(data, type_colors):
            rows.append({"x": int(0.02 * w), "y": y, "d": fs, "name": name, "rgb": rgb})
            y += int(1.35 * fs)
        if rows:
            out["types"] = rows
    if legend and data is not None and "ColorCoding.range" in data.attributes:
        lo, hi = data.attributes["ColorCoding.range"]
        x0, x1, y0, y1 = int(0.88 * w), int(0.91 * w), int(0.15 * h), int(0.85 * h)
        ticks = [(int(y1 - 1 - f * (y1 - y0 - 1)), f"{lo + f * (hi - lo):.3g}") for f in (0, 0.25, 0.5, 0.75, 1)]
        title = str(data.attributes.get("ColorCoding.property", ""))
        title_x = int(min(x0, w * 0.985 - 0.62 * int(fs * 0.8) * len(title)))  # keep long titles inside
        out["colorbar"] = {"box": (x0, y0, x1, y1), "cmap": data.attributes.get("ColorCoding.cmap", "viridis"),
                           "title": title, "title_x": title_x, "ticks": ticks,
                           "tick_len": int(0.006 * w) + 1, "label_x": x1 + int(0.01 * w)}
    if scale_bar and camera is not None:
        ppa = pixels_per_angstrom(camera, size)
        length = nice_length(0.2 * w / ppa) if scale_bar is True else float(scale_bar)
        px = length * ppa
        x0, yb = int(0.04 * w), int(0.94 * h)
        label = f"{length / 10:g} nm" if length >= 10 and length % 10 == 0 else f"{length:g} Å"
        out["scale_bar"] = {"x0": x0, "x1": x0 + px, "y": yb, "thickness": max(2, int(0.006 * h)),
                            "label": label, "length": length}
    if overlays:
        ctx = OverlayContext((w, h), camera, frame, fs, out["small"], ink_color(background, transparent), background)
        items = overlay_items(data, ctx, overlays)
        if items:
            out["items"] = items
    return out


def draw_overlays(rgba: np.ndarray, data, *, text=None, legend=True, type_legend=False, scale_bar=None,
                  camera=None, type_colors=None, background="white", transparent=False, overlays=None,
                  frame=0) -> np.ndarray:
    """Draw the overlays onto an (H, W, 3|4) uint8 image with PIL; returns a new RGBA array."""
    lay = overlay_layout(data, (rgba.shape[1], rgba.shape[0]), text=text, legend=legend, type_legend=type_legend,
                         scale_bar=scale_bar, camera=camera, type_colors=type_colors, overlays=overlays,
                         frame=frame, background=background, transparent=transparent)
    if len(lay) <= 2:
        return rgba if rgba.shape[-1] == 4 else np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    from PIL import Image, ImageDraw, ImageFont

    from .colormaps import get_colormap

    im = Image.fromarray(np.ascontiguousarray(rgba)).convert("RGBA")
    draw = ImageDraw.Draw(im)
    ink = tuple(int(255 * v) for v in ink_color(background, transparent)) + (255,)
    from matplotlib import font_manager

    path = font_manager.findfont("DejaVu Sans")  # bundled with matplotlib on every platform
    font = ImageFont.truetype(path, lay["font_size"])
    small = ImageFont.truetype(path, lay["small"])
    fs = lay["font_size"]
    if "text" in lay:
        t = lay["text"]
        draw.text((t["x"], t["y"]), t["text"], fill=ink, font=font)
    for row in lay.get("types", []):
        x, y, d = row["x"], row["y"], row["d"]
        fill = tuple(int(255 * v) for v in row["rgb"]) + (255,)
        draw.ellipse([x, y, x + d, y + d], fill=fill, outline=ink, width=max(1, d // 14))
        draw.text((x + int(1.5 * d), y + d // 2), row["name"], fill=ink, font=font, anchor="lm")
    if "colorbar" in lay:
        cb = lay["colorbar"]
        x0, y0, x1, y1 = cb["box"]
        cmap = get_colormap(cb["cmap"])
        for y in range(y0, y1):
            c = cmap(1 - (y - y0) / max(1, y1 - y0 - 1))
            draw.line([(x0, y), (x1, y)], fill=tuple(int(255 * v) for v in c[:3]) + (255,))
        draw.rectangle([x0, y0, x1, y1 - 1], outline=ink)
        tx = min(cb["title_x"], int(rgba.shape[1] * 0.985 - draw.textlength(cb["title"], font=small)))
        draw.text((tx, y0 - int(1.4 * fs)), cb["title"], fill=ink, font=small)
        for y, label in cb["ticks"]:
            draw.line([(x1, y), (x1 + cb["tick_len"], y)], fill=ink)
            draw.text((cb["label_x"], y), label, fill=ink, font=small, anchor="lm")
    if "scale_bar" in lay:
        sb = lay["scale_bar"]
        th = sb["thickness"]
        draw.rectangle([sb["x0"], sb["y"] - th, sb["x1"], sb["y"]], fill=ink)
        draw.text(((sb["x0"] + sb["x1"]) / 2, sb["y"] - th - int(0.3 * fs)), sb["label"], fill=ink, font=small,
                  anchor="mb")
    if "items" in lay:
        im = _draw_items_pil(im, lay["items"], ink, path, fs)
    return np.asarray(im)


def _rgba255(color, ink):
    from matplotlib.colors import to_rgba

    if color is None or (isinstance(color, str) and color == "ink"):
        return ink
    return tuple(int(round(255 * v)) for v in to_rgba(color))


_PIL_ANCHOR = {"t": "t", "m": "m", "b": "b"}


def _draw_items_pil(im, items, ink, font_path, font_size):
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    fonts = {}

    def font(size, bold):
        key = (int(round(size or font_size)), bold)
        if key not in fonts:
            from matplotlib import font_manager

            p = font_manager.findfont("DejaVu Sans:bold") if bold else font_path
            fonts[key] = ImageFont.truetype(p, max(1, key[0]))
        return fonts[key]

    draw = ImageDraw.Draw(im)
    for it in items:
        if isinstance(it, Image):
            src = np.asarray(it.rgba, np.uint8)
            src = src if src.shape[-1] == 4 else np.dstack([src, np.full(src.shape[:2], 255, np.uint8)])
            pic = PILImage.fromarray(np.ascontiguousarray(src), "RGBA")
            w = int(round(it.width or pic.width))
            h = int(round(it.height or pic.height * w / pic.width))
            pic = pic.resize((max(1, w), max(1, h)), PILImage.LANCZOS)
            im.alpha_composite(pic, (int(round(it.x)), int(round(it.y))))
            draw = ImageDraw.Draw(im)
        elif isinstance(it, Text):
            anchor = it.anchor[0] + _PIL_ANCHOR.get(it.anchor[1], "t")
            halo = {} if it.halo is None else {"stroke_width": max(1, round((it.size or font_size) / 8)),
                                                "stroke_fill": _rgba255(it.halo, ink)}
            draw.text((it.x, it.y), it.text, fill=_rgba255(it.color, ink), font=font(it.size, it.bold), anchor=anchor,
                      **halo)
        elif isinstance(it, Rect):
            draw.rectangle([it.x0, it.y0, it.x1, it.y1],
                           fill=None if it.fill is None else _rgba255(it.fill, ink),
                           outline=None if it.outline is None else _rgba255(it.outline, ink),
                           width=max(1, int(round(it.width))))
        elif isinstance(it, Circle):
            draw.ellipse([it.x - it.r, it.y - it.r, it.x + it.r, it.y + it.r],
                         fill=None if it.fill is None else _rgba255(it.fill, ink),
                         outline=None if it.outline is None else _rgba255(it.outline, ink),
                         width=max(1, int(round(it.width))))
        elif isinstance(it, Polyline):
            pts = [tuple(map(float, p)) for p in np.asarray(it.points, float)]
            if it.closed and it.fill is not None:
                draw.polygon(pts, fill=_rgba255(it.fill, ink))
            if it.width > 0 and len(pts) > 1:
                draw.line(pts + (pts[:1] if it.closed else []), fill=_rgba255(it.color, ink),
                          width=max(1, int(round(it.width))), joint="curve")
    return im


_MPL_HA = {"l": "left", "m": "center", "r": "right"}
_MPL_VA = {"t": "top", "m": "center", "b": "bottom"}


def draw_items_mpl(ax, items, ink, pt: float, font_size: float):
    """Draw overlay items on a matplotlib axes in pixel coordinates (y down); ``pt`` = points per px."""
    from matplotlib import patheffects
    from matplotlib.patches import Circle as MCircle
    from matplotlib.patches import Polygon, Rectangle

    def col(c):
        return ink if c is None or (isinstance(c, str) and c == "ink") else c

    for it in items:
        if isinstance(it, Image):
            src = np.asarray(it.rgba, np.uint8)
            w = it.width or src.shape[1]
            h = it.height or src.shape[0] * w / src.shape[1]
            xl, yl = ax.get_xlim(), ax.get_ylim()
            ax.imshow(src, extent=(it.x, it.x + w, it.y + h, it.y), aspect="auto", interpolation="lanczos",
                      zorder=3)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        elif isinstance(it, Text):
            ax.text(it.x, it.y, it.text, color=col(it.color), fontsize=(it.size or font_size) * pt,
                    ha=_MPL_HA.get(it.anchor[0], "left"), va=_MPL_VA.get(it.anchor[1], "top"),
                    fontweight="bold" if it.bold else "normal", zorder=4,
                    **({} if it.halo is None else {"path_effects": [patheffects.withStroke(
                        linewidth=(it.size or font_size) * pt / 4, foreground=col(it.halo))]}))
        elif isinstance(it, Rect):
            ax.add_patch(Rectangle((it.x0, it.y0), it.x1 - it.x0, it.y1 - it.y0,
                                   facecolor="none" if it.fill is None else col(it.fill),
                                   edgecolor="none" if it.outline is None else col(it.outline),
                                   linewidth=it.width * pt, zorder=3))
        elif isinstance(it, Circle):
            ax.add_patch(MCircle((it.x, it.y), it.r, facecolor="none" if it.fill is None else col(it.fill),
                                 edgecolor="none" if it.outline is None else col(it.outline),
                                 linewidth=it.width * pt, zorder=3))
        elif isinstance(it, Polyline):
            pts = np.asarray(it.points, float)
            if it.closed and it.fill is not None:
                ax.add_patch(Polygon(pts, closed=True, facecolor=col(it.fill), edgecolor="none", zorder=3))
            if it.width > 0 and len(pts) > 1:
                p = np.vstack([pts, pts[:1]]) if it.closed else pts
                ax.plot(p[:, 0], p[:, 1], color=col(it.color), linewidth=it.width * pt, solid_capstyle="round",
                        solid_joinstyle="round", zorder=3)
