"""Figure overlays shared by every renderer: text label, color legend, type legend, scale bar.

The layout is computed once in pixel coordinates (origin top-left) by
:func:`overlay_layout`; raster renderers draw it with PIL (:func:`draw_overlays`)
and the vector renderer draws the same items as vector text and shapes.
"""
from __future__ import annotations

import math

import numpy as np


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


def ink_color(background, transparent: bool) -> tuple:
    from .render import _rgb

    dark = not transparent and sum(_rgb(background)) < 1.5
    return (0.92, 0.92, 0.93) if dark else (0.08, 0.08, 0.08)


def overlay_layout(data, size, *, text=None, legend=True, type_legend=False, scale_bar=None, camera=None,
                   type_colors=None) -> dict:
    """Positions (pixels, origin top-left) and contents of the requested overlays.

    ``scale_bar``: True picks a round length (about a fifth of the width), a number
    gives the length in Angstrom; needs ``camera``.
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
    return out


def draw_overlays(rgba: np.ndarray, data, *, text=None, legend=True, type_legend=False, scale_bar=None,
                  camera=None, type_colors=None, background="white", transparent=False) -> np.ndarray:
    """Draw the overlays onto an (H, W, 3|4) uint8 image with PIL; returns a new RGBA array."""
    lay = overlay_layout(data, (rgba.shape[1], rgba.shape[0]), text=text, legend=legend, type_legend=type_legend,
                         scale_bar=scale_bar, camera=camera, type_colors=type_colors)
    if len(lay) <= 2:
        return rgba if rgba.shape[-1] == 4 else np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

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
        cmap = matplotlib.colormaps[cb["cmap"]]
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
    return np.asarray(im)
