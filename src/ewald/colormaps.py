"""Color maps by name: every matplotlib colormap plus ``@colormap`` plugins.

Everything in ewald that maps numbers to colors (``ColorCoding``, ``GridSlice``,
``VectorVis``, the color legend of every renderer, the GUI's color map menus) goes through
:func:`get_colormap`, so a plugin color map works everywhere at once::

    from ewald.plugins import colormap

    @colormap(kind="diverging", description="blue-white-red for charge differences")
    def charge_diff():
        return ["#2166ac", "#f7f7f7", "#b2182b"]     # or an (n, 3|4) array, or a matplotlib Colormap

A registered name shadows a matplotlib one. Append ``_r`` to any name for the reversed map.
:func:`lightness` and :func:`is_perceptual` tell whether a map stays readable in grayscale
and for color-blind readers (CIELAB lightness changes monotonically along it).
"""
from __future__ import annotations

import numpy as np

from .colors import to_rgb  # noqa: E402,F401  (the one color parser; kept importable from here)


KINDS = ("sequential", "diverging", "cyclic", "categorical")

#: Offered first in menus (the GUI lists plugins after these); any matplotlib name works.
COMMON = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu_r", "bwr",
          "seismic", "Spectral", "jet", "rainbow", "hot", "Greys", "twilight", "hsv", "tab10"]


def get_colormap(name):
    """The matplotlib ``Colormap`` for a registered plugin name or a matplotlib name.

    A ``Colormap`` passes through unchanged; ``name + "_r"`` gives a plugin map reversed.
    """
    import matplotlib
    from matplotlib.colors import Colormap

    if isinstance(name, Colormap):
        return name
    from .plugins import colormaps

    reg = colormaps()
    if name in reg:
        return reg[name].colormap()
    if name.endswith("_r") and name[:-2] in reg:
        return reg[name[:-2]].colormap().reversed()
    try:
        return matplotlib.colormaps[name]
    except KeyError:
        raise KeyError(f"unknown color map {name!r}; use a matplotlib name or one of: "
                       f"{', '.join(reg) or 'no plugins installed'}") from None


def as_colormap(name: str, value):
    """A ``Colormap`` from what a ``@colormap`` function returns: a Colormap, or a list/array of
    colors (any form :mod:`ewald.colors` reads) spaced evenly from 0 to 1."""
    from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

    from .colors import to_rgba

    if isinstance(value, Colormap):
        return value.copy() if value.name == name else _renamed(value, name)
    colors = [to_rgba(c) for c in value] if not isinstance(value, np.ndarray) else np.asarray(value, float)
    if len(colors) < 2:
        raise ValueError(f"color map {name!r} needs at least two colors")
    if len(colors) >= 64:
        return ListedColormap(colors, name=name)
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def _renamed(cmap, name):
    out = cmap.copy()
    out.name = name
    return out


def names() -> list[str]:
    """Names for menus: :data:`COMMON` first, then the plugin color maps."""
    from .plugins import colormaps

    return COMMON + [n for n in colormaps() if n not in COMMON]


def lightness(name, n: int = 64) -> np.ndarray:
    """CIELAB L* (0 black .. 100 white) at ``n`` evenly spaced points of the map."""
    rgb = get_colormap(name)(np.linspace(0, 1, n))[:, :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])          # relative luminance (D65)
    return np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, y * 24389 / 27)


def is_perceptual(name, kind: str | None = None, tol: float = 1.0) -> bool:
    """Whether the map reads the same in grayscale: L* changes monotonically (sequential) or
    monotonically on each side of the middle (diverging). Cyclic and categorical maps never do."""
    if kind in ("cyclic", "categorical"):
        return False
    lum = lightness(name)
    d = np.diff(lum)
    if kind == "diverging":
        half = len(d) // 2
        return bool((np.all(d[:half] >= -tol) and np.all(d[half:] <= tol))
                    or (np.all(d[:half] <= tol) and np.all(d[half:] >= -tol)))
    return bool(np.all(d >= -tol) or np.all(d <= tol)) and abs(lum[-1] - lum[0]) > 20
