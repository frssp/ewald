"""One color parser for every color argument (view settings, steps, overlays, themes, GUI forms).

Accepted forms, everywhere a color is taken:

- a matplotlib or CSS name, any case: ``"red"``, ``"Crimson"``, ``"tab:blue"``, ``"xkcd:sky blue"``,
  ``"C0"`` (the plot color cycle);
- hex, with or without ``#``: ``"#f80"``, ``"#ff8800"``, ``"ff8800"``, ``"#ff880080"`` (with alpha);
  digits only (``"112233"``) need the ``#``;
- a grey level as a string: ``"0.5"``;
- CSS functions: ``"rgb(255, 128, 0)"``, ``"rgba(255, 128, 0, 0.5)"``, ``"rgb(100%, 50%, 0%)"``;
- three or four numbers as a tuple, list, array or comma-separated string: ``(1, 0.5, 0)`` in 0..1,
  or ``(255, 128, 0)`` in 0..255 (any of r, g, b above 1 means 0..255); a fourth number is alpha
  (0..1, or 0..255 together with 0..255 RGB);
- a 24-bit integer ``0xFF8800``.

Atoms, bonds and lines are drawn opaque: alpha is kept by :func:`to_rgba` but the view's
``transparency`` setting is what makes atoms see-through.
"""
from __future__ import annotations

import re

import numpy as np

FORMS = ("a name ('red', 'tab:blue'), hex ('#ff8800', '#f80', 'ff8800'), 'rgb(255, 128, 0)', "
         "a grey level '0.5', or (r, g, b[, a]) in 0..1 or 0..255")

_HEX = re.compile(r"[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}")
_CSS = re.compile(r"(rgba?)\s*\(\s*([^)]*)\)", re.IGNORECASE)


def _error(color, why=""):
    return ValueError(f"not a color: {color!r}{f' ({why})' if why else ''}; use {FORMS}")


def _numbers(values, color) -> tuple:
    try:
        v = [float(x) for x in values]
    except (TypeError, ValueError):
        raise _error(color) from None
    if len(v) not in (3, 4) or not all(np.isfinite(v)):
        raise _error(color, "three or four numbers")
    rgb, alpha = v[:3], (v[3] if len(v) == 4 else None)
    if min(v) < 0:
        raise _error(color, "negative component")
    scale = 255.0 if max(rgb) > 1 else 1.0
    if max(rgb) > 255:
        raise _error(color, "components are 0..1 or 0..255")
    rgb = [x / scale for x in rgb]
    if alpha is None:
        alpha = 1.0
    elif alpha > 1:
        if scale == 1 or alpha > 255:
            raise _error(color, "alpha is 0..1")
        alpha /= 255.0
    return (*rgb, alpha)


def _css(match, color) -> tuple:
    parts = [p.strip() for p in match.group(2).replace("/", ",").split(",") if p.strip()]
    if len(parts) != (4 if match.group(1).lower() == "rgba" else len(parts)) or len(parts) not in (3, 4):
        raise _error(color)
    values = []
    for i, p in enumerate(parts):
        try:
            if p.endswith("%"):
                values.append(float(p[:-1]) / 100 * (255 if i < 3 else 1))
            else:
                values.append(float(p))
        except ValueError:
            raise _error(color) from None
    rgb = [x / 255.0 for x in values[:3]]   # CSS rgb() is always 0..255
    if max(values[:3]) > 255 or min(values) < 0 or (len(values) == 4 and values[3] > 1):
        raise _error(color)
    return (*rgb, values[3] if len(values) == 4 else 1.0)


def to_rgba(color) -> tuple:
    """``(r, g, b, a)`` floats in 0..1 from any accepted form (see the module docstring); ValueError
    (naming the forms) for anything else."""
    from matplotlib.colors import to_rgba as _mpl

    if isinstance(color, bytes):
        color = color.decode()
    if isinstance(color, str):
        text = color.strip()
        if not text:
            raise _error(color, "empty")
        if (m := _CSS.fullmatch(text)) is not None:
            return _css(m, color)
        if "," in text:
            return _numbers(text.strip("()[] ").split(","), color)
        try:
            return tuple(float(x) for x in _mpl(text))
        except ValueError:
            if _HEX.fullmatch(text) and not text.isdigit():   # hex without '#' ('123' is no color)
                return tuple(float(x) for x in _mpl("#" + text))
            raise _error(color) from None
    if isinstance(color, (int, np.integer)) and not isinstance(color, (bool, np.bool_)):
        if not 0 <= int(color) <= 0xFFFFFF:
            raise _error(color, "a 24-bit 0xRRGGBB integer")
        c = int(color)
        return ((c >> 16 & 255) / 255.0, (c >> 8 & 255) / 255.0, (c & 255) / 255.0, 1.0)
    if color is None or isinstance(color, (bool, np.bool_, float)):
        raise _error(color)
    return _numbers(np.asarray(color, dtype=object).ravel().tolist() if np.ndim(color) else [color], color)


def to_rgb(color) -> np.ndarray:
    """RGB in 0..1 (a float array of 3) from any accepted form; alpha is dropped."""
    return np.array(to_rgba(color)[:3], float)


def to_hex(color, keep_alpha: bool = False) -> str:
    """``'#rrggbb'`` (``'#rrggbbaa'`` with ``keep_alpha`` and alpha below 1) for any accepted form."""
    r = to_rgba(color)
    digits = [r[0], r[1], r[2]] + ([r[3]] if keep_alpha and r[3] < 1 else [])
    return "#" + "".join(f"{int(round(255 * float(np.clip(x, 0, 1)))):02x}" for x in digits)


def is_color(color) -> bool:
    """True when :func:`to_rgba` reads ``color``."""
    try:
        to_rgba(color)
    except (ValueError, TypeError):
        return False
    return True


def normalize(color):
    """A color as it is stored (view settings, sessions): a string matplotlib reads as is stays, so a
    notebook keeps ``'red'`` or ``'#ff8800'``; other strings ('ff8800', 'rgb(...)', '255,128,0')
    become ``'#rrggbb'``; numbers become an ``(r, g, b)`` float tuple in 0..1. Raises ValueError."""
    rgba = to_rgba(color)
    if isinstance(color, str):
        from matplotlib.colors import is_color_like

        text = color.strip()
        return text if is_color_like(text) else to_hex(color)
    return tuple(round(float(x), 4) for x in rgba[:3])
