"""Themes: named bundles of element colors, background, cell color and radius preset.

``render(data, "fig.png", theme="VESTA")`` and the GUI's View › Theme menu apply one; explicit
arguments (``type_colors``, ``background``, ``radius_preset``, ...) still win over the theme.
Plugins add themes with ``@theme``::

    from ewald.plugins import theme
    from ewald.themes import Theme

    @theme(description="our group's figure style")
    def group_style():
        return Theme(colors={"Li": "#7f3fbf", "O": "#d62728"}, base="VESTA",
                     background="white", cell_color="#555555")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SCHEMES = ("Jmol", "CPK", "VESTA")


@lru_cache(maxsize=None)
def element_colors(scheme: str) -> dict:
    """Element symbol -> RGB (0..1) of a color scheme: 'Jmol' and 'CPK' (from ASE) or 'VESTA'."""
    if scheme in ("Jmol", "CPK"):
        from ase.data import chemical_symbols
        from ase.data.colors import cpk_colors, jmol_colors

        table = jmol_colors if scheme == "Jmol" else cpk_colors
        return {s: tuple(float(v) for v in table[z]) for z, s in enumerate(chemical_symbols) if 0 < z < len(table)}
    if scheme == "VESTA":
        data = json.loads((Path(__file__).parent / "tables" / "colors.json").read_text())
        return {k: tuple(v) for k, v in data["VESTA"].items()}
    raise KeyError(f"unknown color scheme {scheme!r}; use one of {', '.join(SCHEMES)}")


def muted(rgb, amount: float = 0.35):
    """``rgb`` moved ``amount`` of the way toward the gray of the same luminance."""
    rgb = np.asarray(rgb, float)
    gray = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
    return tuple(float(v) for v in np.clip(gray + (1 - amount) * (rgb - gray), 0, 1))


@dataclass
class Theme:
    """Colors per element (``colors`` on top of the ``base`` scheme, optionally ``mute``d toward
    gray), plus ``background``, ``cell_color`` and ``radius_preset``; None leaves a setting alone."""

    base: str = "Jmol"
    colors: dict = field(default_factory=dict)
    mute: float = 0.0
    background: str | None = None
    cell_color: str | None = None
    radius_preset: str | None = None
    description: str = ""

    def color_of(self, particle_type) -> tuple | None:
        """The theme's color for a particle type (by its element), or None for unknown types."""
        from matplotlib.colors import to_rgb

        from .radii import element_of

        t = str(particle_type)
        c = self.colors.get(t)
        if c is None:
            el = element_of(t)
            c = self.colors.get(el)
            if c is None:
                c = element_colors(self.base).get(el)
        if c is None:
            return None
        c = to_rgb(c)
        return muted(c, self.mute) if self.mute else tuple(float(v) for v in c)

    def type_colors(self, types) -> dict:
        """``type_colors`` for the given particle types (types the theme does not know are left out)."""
        out = {}
        for t in dict.fromkeys(str(t) for t in types):
            c = self.color_of(t)
            if c is not None:
                out[t] = c
        return out

    def render_kwargs(self, types=()) -> dict:
        """The ``render()`` keyword arguments this theme sets for particles of ``types``."""
        kw = {"type_colors": self.type_colors(types)} if len(types) else {}
        for key in ("background", "cell_color", "radius_preset"):
            if getattr(self, key) is not None:
                kw[key] = getattr(self, key)
        return kw


BUILTIN = {
    "Default": Theme(description="Jmol element colors and covalent radii (ewald's defaults)"),
    "VESTA": Theme(base="VESTA", background="white", cell_color="black",
                   description="VESTA's element colors on white"),
    "CPK": Theme(base="CPK", description="CPK element colors"),
    "Journal": Theme(mute=0.35, background="white", cell_color="#333333",
                     description="Jmol colors muted 35% toward gray, white background, dark gray cell"),
    "ASE": Theme(background="white", cell_color="black", radius_preset="covalent",
                 description="ASE GUI: Jmol colors and covalent radii on white (pair with the 'Simple (ASE)' style)"),
    "Dark slide": Theme(background="black", cell_color="#AEB4BE",
                        description="Jmol colors on black, light cell lines, for slides"),
}


def get_theme(theme) -> Theme:
    """A :class:`Theme` from a registered name (built-in or plugin) or a Theme instance."""
    if isinstance(theme, Theme):
        return theme
    from .plugins import themes

    reg = themes()
    if theme not in reg:
        raise KeyError(f"unknown theme {theme!r}; available: {', '.join(reg)}")
    return reg[theme].theme()


def apply_theme(theme, data, kwargs: dict) -> dict:
    """``kwargs`` with the theme's settings filled in where ``kwargs`` has no value (None counts as
    none); explicit ``type_colors`` entries win per type."""
    if theme is None:
        return kwargs
    t = get_theme(theme)
    types = data.types if data is not None and data.count else ()
    out = dict(kwargs)
    for key, value in t.render_kwargs(types).items():
        if key == "type_colors":
            out["type_colors"] = {**value, **(kwargs.get("type_colors") or {})}
        elif out.get(key) is None:
            out[key] = value
    return out
