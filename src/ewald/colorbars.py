"""Color bars as overlay items, drawn the same way in the viewport and in every picture.

:func:`colorbar_items` lays out the color bars of a view (``{"title", "cmap", "range"}`` dicts, see
:func:`ewald.overlays.color_legends`) as 2D overlay items (:mod:`ewald.overlays`): the gradient is a
:class:`~ewald.overlays.ColorRamp` (pixels in pictures, shapes in SVG/PDF), ticks and labels are lines
and text. Picture renderers draw the items
with the other overlays; the viewport draws them into a transparent image (:func:`items_rgba`) that
it pins to a corner of the 3D view, so both show the same bar.

Sizes derive from one ``unit``, the label font size in px: the viewport passes its UI font size
times the device pixel ratio, pictures pass their small overlay font size.
"""
from __future__ import annotations

import math

import numpy as np

from .overlays import ColorRamp, Image, Polyline, Rect, Text

STYLES = ("slim", "card", "horizontal", "classic")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values inside [lo, hi] (about ``n``); whole numbers only when both ends are whole and
    at least 2 apart (frames, timesteps)."""
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    from matplotlib.ticker import MaxNLocator

    integer = lo == int(lo) and hi == int(hi) and hi - lo >= 2
    loc = MaxNLocator(nbins=max(1, n), steps=[1, 2, 2.5, 5, 10], integer=integer, min_n_ticks=2)
    eps = 1e-9 * (hi - lo)
    ticks = [float(v) for v in loc.tick_values(lo, hi) if lo - eps <= v <= hi + eps]
    while len(ticks) > n + 1:      # 0, 1, ..., 7 for frames 0-7: every other one
        ticks = ticks[::2]
    return ticks or [lo, hi]


def tick_labels(ticks) -> list[str]:
    """One format for every tick of a bar: as many decimals as the step needs, scientific notation for
    very large or small values, a real minus sign."""
    ticks = [float(t) for t in ticks]
    big = max((abs(t) for t in ticks), default=0.0)
    step = min((b - a for a, b in zip(ticks, ticks[1:]) if b > a), default=big or 1.0)
    if big >= 1e6 or (0 < big < 1e-3):
        out = [f"{t:.2g}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e") for t in ticks]
    else:
        digits = max(0, -math.floor(math.log10(step) + 1e-9)) if step > 0 else 0
        if step > 0 and round(step * 10 ** digits) % 10 and abs(step * 10 ** digits - round(step * 10 ** digits)) > 1e-6:
            digits += 1                   # 2.5, 0.25, ...
        out = [f"{t:.{digits}f}" for t in ticks]
    return [s.replace("-", "−") if s.startswith("-") and s.strip("-0.") else s.lstrip("-") for s in out]


def _text_width(text: str, size: float, bold: bool = False) -> float:
    """Width of ``text`` in DejaVu Sans at ``size`` px (the font every renderer draws overlays with)."""
    return float(_font(size, bold).getlength(text)) if text else 0.0


_FONTS: dict = {}


def _font(size: float, bold: bool = False):
    from PIL import ImageFont

    key = (max(1, int(round(size))), bold)
    if key not in _FONTS:
        from matplotlib import font_manager

        path = font_manager.findfont("DejaVu Sans:bold" if bold else "DejaVu Sans")
        _FONTS[key] = ImageFont.truetype(path, key[0])
    return _FONTS[key]


def gradient_rgba(cmap, width: int, height: int, vertical: bool = True, radius: float = 0.0) -> np.ndarray:
    """The bar's colors as a (height, width, 4) uint8 image, high values at the top (or right), with
    corners rounded by ``radius`` px (anti-aliased)."""
    from .colormaps import get_colormap

    width, height = max(1, int(round(width))), max(1, int(round(height)))
    n = height if vertical else width
    f = (np.arange(n) + 0.5) / n
    rgb = np.asarray(get_colormap(cmap)(f[::-1] if vertical else f))[:, :3]
    img = np.empty((height, width, 4), np.uint8)
    img[..., :3] = np.round(255 * (rgb[:, None, :] if vertical else rgb[None, :, :]))
    img[..., 3] = np.round(255 * _round_mask(width, height, radius))
    return img


def _round_mask(w: int, h: int, r: float) -> np.ndarray:
    """Coverage (0-1) of a w x h rectangle with corners of radius ``r``, 4x4 supersampled."""
    r = min(float(r), w / 2, h / 2)
    if r <= 0.25:
        return np.ones((h, w))
    s = 4
    ys = (np.arange(h * s) + 0.5) / s
    xs = (np.arange(w * s) + 0.5) / s
    dx = np.maximum(np.maximum(r - xs, xs - (w - r)), 0)[None, :]
    dy = np.maximum(np.maximum(r - ys, ys - (h - r)), 0)[:, None]
    inside = (dx ** 2 + dy ** 2) <= r * r
    return inside.reshape(h, s, w, s).mean(axis=(1, 3))


def _surface(background, transparent: bool) -> tuple:
    """The card's fill: a translucent panel a little lighter (dark backgrounds) or whiter (light ones)."""
    from .overlays import is_dark

    return (0.13, 0.14, 0.16, 0.78) if is_dark(background, transparent) else (1.0, 1.0, 1.0, 0.82)


def colorbar_items(bars, size, *, unit: float, ink=(0.08, 0.08, 0.08), background="white",
                   transparent: bool = False, style: str = "slim", corner: str | None = None) -> list:
    """Overlay items of the color ``bars`` in an image of ``size`` (w, h) px; ``unit`` is the label font
    size in px. Several bars sit side by side (vertical styles) or stack upward (horizontal)."""
    if style not in STYLES:
        raise ValueError(f"color bar style {style!r}: one of {', '.join(STYLES)}")
    bars = [dict(b) for b in bars or ()]
    if not bars:
        return []
    W, H = float(size[0]), float(size[1])
    u = float(unit)
    ink = tuple(float(c) for c in ink[:3])
    faint = ink + (0.55,)
    margin = max(0.8 * u, 0.02 * H)
    blocks = [_block(b, u, H, ink, faint, style, background, transparent) for b in bars]
    out = []
    if style == "horizontal":
        corner = corner or "bottom-right"
        y = H - margin if corner.startswith("bottom") else margin
        for w, h, items in blocks:
            x = W - margin - w if corner.endswith("right") else margin
            top = y - h if corner.startswith("bottom") else y
            out += _shift(items, round(x), round(top))
            y = top - 0.9 * u if corner.startswith("bottom") else top + h + 0.9 * u
        return out
    corner = corner or ("top-right" if style == "card" else "right")
    x = W - margin
    for w, h, items in blocks:
        x -= w
        if corner == "right":
            top = (H - h) / 2
        elif corner.startswith("bottom"):
            top = H - margin - h
        else:
            top = margin
        out += _shift(items, round(x), round(top))
        x -= 1.2 * u
    return out


def _shift(items, dx, dy):
    out = []
    for it in items:
        if isinstance(it, (Text, Image)):
            it.x += dx
            it.y += dy
        elif isinstance(it, (Rect, ColorRamp)):
            it.x0, it.x1, it.y0, it.y1 = it.x0 + dx, it.x1 + dx, it.y0 + dy, it.y1 + dy
        elif isinstance(it, Polyline):
            it.points = (np.asarray(it.points, float) + [dx, dy]).tolist()
        out.append(it)
    return out


def _block(bar, u, H, ink, faint, style, background, transparent):
    """(width, height, items at the origin) of one bar."""
    lo, hi = (float(v) for v in bar["range"])
    title = str(bar.get("title", ""))
    ticks = nice_ticks(lo, hi, 5 if style != "horizontal" else 4)
    labels = tick_labels(ticks)
    span = hi - lo if hi > lo else 1.0
    lab = 0.92 * u
    if style == "horizontal":
        L, T = 11.0 * u, 0.62 * u
        title_h = 1.5 * u if title else 0.0
        items = [Text(0, 0, title, u, ink, "lt")] if title else []
        items.append(ColorRamp(0, title_h, L, title_h + T, bar["cmap"], False, T / 2))
        items.append(Rect(0, title_h, L, title_h + T, outline=ink + (0.18,), width=1, radius=T / 2))
        for t, s in zip(ticks, labels):
            x = (t - lo) / span * L
            items.append(Polyline([[x, title_h + T + 0.15 * u], [x, title_h + T + 0.45 * u]], faint, 1))
            items.append(Text(x, title_h + T + 0.6 * u, s, lab, ink, "mt"))
        # keep the end labels inside the block
        w_first, w_last = _text_width(labels[0], lab) / 2, _text_width(labels[-1], lab) / 2
        pad_l = max(0.0, w_first - (ticks[0] - lo) / span * L)
        pad_r = max(0.0, w_last - (hi - ticks[-1]) / span * L)
        width = max(pad_l + L + pad_r, _text_width(title, u))
        items = _shift(items, pad_l, 0)
        return width, title_h + T + 0.6 * u + 1.2 * lab, items
    classic = style == "classic"
    L = min(max(8.5 * u, 0.3 * H), 0.45 * H) if not classic else 0.42 * H
    T = 0.7 * u if not classic else 1.1 * u
    title_h = 1.7 * u if title else 0.4 * u
    label_w = max(_text_width(s, lab) for s in labels)
    gap = 0.5 * u
    tick_out = 0.35 * u
    radius = 0.0 if classic else T / 2
    items = []
    if title:
        items.append(Text(0, 0, title, u, ink, "lt", bold=classic))
    items.append(ColorRamp(0, title_h, T, title_h + L, bar["cmap"], True, radius))
    items.append(Rect(0, title_h, T, title_h + L, outline=ink + ((1.0,) if classic else (0.18,)),
                      width=1, radius=radius))
    for t, s in zip(ticks, labels):
        y = title_h + L - (t - lo) / span * L
        if classic:
            items.append(Polyline([[T - tick_out, y], [T, y]], ink, 1))
            items.append(Polyline([[0, y], [tick_out, y]], ink, 1))
        else:
            items.append(Polyline([[T + 0.15 * u, y], [T + 0.15 * u + tick_out, y]], faint, 1))
        items.append(Text(T + gap + (0.0 if classic else tick_out), y, s, lab, ink, "lm"))
    width = max(T + gap + tick_out + label_w, _text_width(title, u, classic))
    height = title_h + L + 0.5 * lab
    if style == "card":
        pad = 0.8 * u
        items = _shift(items, pad, pad)
        items.insert(0, Rect(0, 0, width + 2 * pad, height + 2 * pad - 0.3 * u, fill=_surface(background, transparent),
                             outline=ink + (0.12,), width=1, radius=0.55 * u))
        width, height = width + 2 * pad, height + 2 * pad - 0.3 * u
    return width, height, items


def items_bounds(items) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) around ``items`` (text by its estimated extent)."""
    xs, ys = [], []
    for it in items:
        if isinstance(it, Image):
            xs += [it.x, it.x + it.width]
            ys += [it.y, it.y + it.height]
        elif isinstance(it, (Rect, ColorRamp)):
            xs += [it.x0, it.x1]
            ys += [it.y0, it.y1]
        elif isinstance(it, Polyline):
            p = np.asarray(it.points, float)
            xs += list(p[:, 0])
            ys += list(p[:, 1])
        elif isinstance(it, Text):
            w, h = _text_width(it.text, it.size or 10), 1.3 * (it.size or 10)
            x0 = it.x - {"l": 0, "m": w / 2, "r": w}[it.anchor[0]]
            y0 = it.y - {"t": 0, "m": h / 2, "b": h}[it.anchor[1]]
            xs += [x0, x0 + w]
            ys += [y0, y0 + h]
    return min(xs), min(ys), max(xs), max(ys)


def items_rgba(items, ink) -> tuple[np.ndarray, float, float]:
    """``items`` drawn on a transparent image just big enough: (H, W, 4) uint8 and the image's
    top-left corner in the items' coordinates."""
    from PIL import Image as PILImage

    from .overlays import _draw_items_pil

    x0, y0, x1, y1 = items_bounds(items)
    x0, y0 = math.floor(x0) - 2, math.floor(y0) - 2
    w, h = int(math.ceil(x1 - x0)) + 3, int(math.ceil(y1 - y0)) + 3
    ink255 = tuple(int(round(255 * c)) for c in ink[:3])
    im = PILImage.new("RGBA", (w, h), ink255 + (0,))   # ink under transparent pixels: no dark fringes on text
    from matplotlib import font_manager

    _draw_items_pil(im, _shift([_copy(it) for it in items], -x0, -y0), ink255 + (255,),
                    font_manager.findfont("DejaVu Sans"), 12)
    return np.asarray(im), float(x0), float(y0)


def _copy(it):
    from dataclasses import replace

    return replace(it)


def default_style() -> str:
    import os

    return os.environ.get("EWALD_COLORBAR_STYLE", "slim")
