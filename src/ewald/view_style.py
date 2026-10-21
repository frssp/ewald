"""Visual elements state (particles, bonds, cell): what OVITO calls visual elements.

Qt-free, so sessions and headless rendering use it too; ``ewald.gui.style`` re-exports it."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .view3d import DEFAULT_PRESET, PRESET_THEMES, STYLE_PRESETS  # noqa: F401 (public here too)

DEFAULT_BACKGROUND = "#14161A"
_background = [DEFAULT_BACKGROUND]


def default_background() -> str:
    """The background of a view that sets none: the UI theme's or Preferences' (the GUI sets it)."""
    return _background[0]


def set_default_background(color: str):
    _background[0] = color
_THEMED = ("type_colors", "background", "cell_color", "radius_preset", "theme")
# trajectory-line color map settings: the same names in ViewStyle, pipe.view and render()
TRAJECTORY_KEYS = ("trajectory_color_by", "trajectory_colormap", "trajectory_color_range", "trajectory_time_range",
                   "trajectory_trail", "trajectory_radius")


@dataclass
class ViewStyle:
    preset: str = DEFAULT_PRESET
    radius_scale: float = STYLE_PRESETS[DEFAULT_PRESET][0]
    atom_style: str = STYLE_PRESETS[DEFAULT_PRESET][2]
    type_colors: dict = field(default_factory=dict)   # type -> (r, g, b) in 0..1 or '#hex'
    type_radii: dict = field(default_factory=dict)    # type -> radius (A)
    radius_preset: str = "covalent"                   # ewald.radii.PRESETS; type_radii win over it
    oxidation_states: dict = field(default_factory=dict)  # type -> oxidation state for the ionic presets
    bonds_enabled: bool = STYLE_PRESETS[DEFAULT_PRESET][1]
    bond_pairs: list = field(default_factory=list)    # [(A, B, dmin, dmax)]
    bond_radius: float = 0.15
    show_cell: bool = True
    cell_width: float = 2.0
    background: str = DEFAULT_BACKGROUND
    bond_factor: float = 1.15                         # VESTA: max bond = factor * (r_A + r_B)
    visuals: list = field(default_factory=list)       # plugin visual elements: {name, params, enabled}
    theme: str = ""                                   # last applied ewald.themes name (its values are above)
    cell_color: str | None = None                     # None: picked from the background
    boundary: list = field(default_factory=list)      # VESTA boundary [lo, hi] (fractional); empty = off
    boundary_bonds: str = "half"                      # bonds leaving it: half / complete / none
    replicate: list = field(default_factory=list)     # [na, nb, nc] periodic copies drawn; empty = off
    replicate_cell: str = "super"                     # cell lines of a replicated view: super / unit / all
    vectors: object = None                            # per-atom vector properties drawn as arrows (render's vectors=)
    trajectory_width: float = 2.0                     # trajectory lines (pixels)
    trajectory_color_by: str | None = None            # color map by 'Time', 'Frame', 'Step Length', ...; None: atom colors
    trajectory_colormap: str = "viridis"
    trajectory_color_range: list | None = None        # [lo, hi] of the color map (either may be None)
    trajectory_time_range: list | None = None         # [t0, t1] Time window drawn
    trajectory_trail: object = None                   # True: up to the current frame; a number: that much Time before it
    trajectory_radius: float | None = None            # ray-traced tube radius (A); None: from the width
    show_atoms: bool = True                           # the Particles element (False: a stick model)
    cell_radius: float | None = None                  # ray-traced cell tube radius (A); None: automatic
    surfaces: dict = field(default_factory=dict)      # data.surfaces look by name: {visible, color, transparency, edges}
    type_transparency: dict = field(default_factory=dict)  # type -> 0..1 (1 hides the type)
    selection: object = True                          # highlight 'Selection': True (red), False or a color
    labels: object = None                             # atom labels: True, a template or {text, types, size, color}
    measurements: list = field(default_factory=list)  # [[i, j], [i, j, k], ...] distances / angles / dihedrals
    light: object = None                              # key light: [x, y, z] (camera frame) or {direction|azimuth/elevation, intensity}

    def apply_preset(self, name: str, types=()):
        """Switch to a style preset. 'Simple (ASE)' also applies the ASE theme (white background);
        leaving it restores the colors and background from before."""
        saved = getattr(self, "_before_preset_theme", None)
        if name not in PRESET_THEMES and saved is not None and self.theme == PRESET_THEMES.get(self.preset):
            for k, v in saved.items():
                setattr(self, k, v)
        self._before_preset_theme = None
        if name in PRESET_THEMES:
            if self.theme != PRESET_THEMES[name]:
                self._before_preset_theme = {k: getattr(self, k) for k in _THEMED}
            self.apply_theme(PRESET_THEMES[name], types)
        self.preset = name
        self.radius_scale, self.bonds_enabled, self.atom_style = STYLE_PRESETS[name]

    def scene_kwargs(self) -> dict:
        kw = {"radius_scale": self.radius_scale, "atom_style": self.atom_style, "show_cell": self.show_cell,
              "cell_width": self.cell_width, "bond_radius": self.bond_radius}
        if self.type_colors:
            kw["type_colors"] = dict(self.type_colors)
        if self.type_radii:
            kw["type_radii"] = dict(self.type_radii)
        kw.update(self.radius_kwargs(with_type_radii=False))
        if self.bonds_enabled and self.bond_pairs:
            kw["bond_pairs"] = [tuple(p) for p in self.bond_pairs]
        if self.visuals:
            kw["visuals"] = [dict(v) for v in self.visuals]
        if self.cell_color:
            kw["cell_color"] = self.cell_color
        if self.vectors:
            kw["vectors"] = self.vectors
        if self.trajectory_width != 2.0:
            kw["trajectory_width"] = self.trajectory_width
        for k in TRAJECTORY_KEYS:
            v = getattr(self, k)
            if v is not None and v != "viridis":
                kw[k] = list(v) if isinstance(v, (list, tuple)) else v
        if not self.show_atoms:
            kw["show_atoms"] = False
        if self.cell_radius:
            kw["cell_radius"] = self.cell_radius
        if self.surfaces:
            kw["surfaces"] = {k: dict(v) for k, v in self.surfaces.items()}
        if self.type_transparency:
            kw["type_transparency"] = dict(self.type_transparency)
        if self.selection is not False:
            from .view3d import SELECTION_COLOR

            kw["selection_color"] = SELECTION_COLOR if self.selection is True else self.selection
        if self.labels:
            kw["labels"] = dict(self.labels) if isinstance(self.labels, dict) else self.labels
        if self.measurements:
            kw["measurements"] = [list(m) for m in self.measurements]
        if self.light is not None:
            kw["light"] = dict(self.light) if isinstance(self.light, dict) else list(self.light)
        if self.boundary:
            kw["boundary"] = [list(map(float, p)) for p in self.boundary] if np.ndim(self.boundary) == 2 \
                else [float(x) for x in self.boundary]
            if self.boundary_bonds != "half":
                kw["boundary_bonds"] = self.boundary_bonds
        if self.replicate and tuple(self.replicate) != (1, 1, 1):
            kw["replicate"] = [int(x) for x in self.replicate]
            if self.replicate_cell != "super":
                kw["replicate_cell"] = self.replicate_cell
        return kw

    def apply_theme(self, name: str, types=()):
        """Copy a theme's colors (for ``types``), background, cell color and radius preset into the style."""
        from .themes import get_theme

        t = get_theme(name)
        self.theme = name
        self.type_colors = {k: tuple(v) for k, v in t.type_colors(types).items()}
        if t.background is not None:
            self.background = t.background
        self.cell_color = t.cell_color
        if t.radius_preset is not None:
            self.radius_preset = t.radius_preset

    def radius_kwargs(self, with_type_radii: bool = True) -> dict:
        """Keyword arguments of :func:`ewald.render.default_radii` for this style (defaults left out)."""
        kw = {"type_radii": dict(self.type_radii)} if with_type_radii and self.type_radii else {}
        if self.radius_preset != "covalent":
            kw["radius_preset"] = self.radius_preset
        if self.oxidation_states:
            kw["oxidation_states"] = dict(self.oxidation_states)
        return kw

    def to_dict(self) -> dict:
        return asdict(self)

    def ensure_bond_pairs(self, types):
        """Seed an empty bond table like VESTA: one row per type pair, max = bond_factor * sum of covalent
        radii. In compounds metal-metal pairs are left out (SrTiO3 gets Ti-O and Sr-O, not Sr-Ti)."""
        if self.bond_pairs:
            return
        from .bonds import covalent_pairs

        for a, b, lo, hi in covalent_pairs(np.unique(np.asarray(types)), self.bond_factor):
            self.bond_pairs.append([a, b, lo, round(float(hi), 2)])

    # ------------------------------------------------------------ notebook code (ewald.view3d.View)
    def apply_view(self, view, types=()):
        """Take the display settings of a pipeline's :class:`ewald.view3d.View` (a notebook's view cell,
        Claude's set_view); camera, frame and picture settings are the window's business."""
        from matplotlib.colors import to_hex

        def hexed(c):
            return to_hex(c) if isinstance(c, str) else tuple(c)

        self._before_preset_theme = None
        self.preset = view.style
        self.radius_scale, self.atom_style = float(view.radius_scale), view.atom_style
        theme = view.theme
        self.type_colors, self.cell_color, self.background = {}, None, default_background()
        self.radius_preset = "covalent"
        if theme:
            self.apply_theme(theme, types)
        else:
            self.theme = ""
        self.type_colors.update({t: hexed(c) for t, c in view.colors.items()})
        if "radius_preset" in view._values or not theme:
            self.radius_preset = view.radius_preset
        if view.background is not None:
            self.background = to_hex(view.background)
        if view.cell_color is not None:
            self.cell_color = hexed(view.cell_color)
        self.type_radii = {t: float(r) for t, r in view.radii.items()}
        self.oxidation_states = dict(view.oxidation_states)
        self._pref_filled = {}   # what Preferences filled is filled again on the next draw
        self._radius_cache = None
        bonds = view.bonds
        self.bonds_enabled = bool(bonds)
        self.bond_factor, self.bond_radius = float(view.bond_factor), float(view.bond_radius)
        if view.bond_pairs:
            self.bond_pairs = [list(p) for p in view.bond_pairs]
        elif isinstance(bonds, float):   # a plain cutoff: every type pair up to it
            uniq = sorted({str(t) for t in types})
            self.bond_pairs = [[a, b, 0.0, bonds] for k, a in enumerate(uniq) for b in uniq[k:]]
        else:
            self.bond_pairs = []
            if self.bonds_enabled and len(types):
                self.ensure_bond_pairs(types)
        self.show_cell, self.cell_width = bool(view.cell), float(view.cell_width)
        self.show_atoms, self.cell_radius = bool(view.atoms), view.cell_radius
        self.surfaces = {k: dict(v) for k, v in view.surfaces.items()}
        self.type_transparency = dict(view.transparency)
        self.selection = view.selection
        self.labels = view.labels
        self.light = view.light
        self.measurements = [list(m) for m in view.measurements]
        self.boundary = list(view.boundary or [])
        self.boundary_bonds = view.boundary_bonds
        self.replicate = list(view.replicate or [])
        self.replicate_cell = view.replicate_cell
        self.visuals = [{"name": v, "params": {}, "enabled": True} if isinstance(v, str) else dict(v)
                        for v in view.visuals]
        self.vectors = view.vectors
        self.trajectory_width = float(view.trajectory_width)
        for k in TRAJECTORY_KEYS:
            v = getattr(view, k)
            setattr(self, k, list(v) if isinstance(v, (list, tuple)) else v)

    def view_settings(self, types=()) -> dict:
        """The display settings of this style as :class:`ewald.view3d.View` settings (what the notebook's
        view cell says), leaving out what Preferences and the theme filled in by themselves."""
        from .view3d import STYLE_PRESETS as presets

        filled = getattr(self, "_pref_filled", {})
        theme = None
        if self.theme:
            from .themes import get_theme

            try:
                theme = get_theme(self.theme)
            except KeyError:
                theme = None
        theme_colors = {k: tuple(v) for k, v in theme.type_colors(types).items()} if theme is not None else {}

        def same_color(a, b):
            from .render import _rgb

            try:
                return np.allclose(_rgb(a), _rgb(b), atol=1e-3)
            except Exception:  # noqa: BLE001
                return False

        colors = {}
        for t, c in self.type_colors.items():
            if t in filled and filled[t][0] is not None and filled[t][0] == c:
                continue
            if t in theme_colors and same_color(c, theme_colors[t]):
                continue
            colors[t] = tuple(round(float(x), 4) for x in c) if not isinstance(c, str) else c
        radii = {t: float(r) for t, r in self.type_radii.items()
                 if not (t in filled and filled[t][1] is not None and filled[t][1] == r)}
        out = {"style": self.preset if self.preset in presets else DEFAULT_PRESET,
               "theme": self.theme or None, "atom_style": self.atom_style, "radius_scale": self.radius_scale,
               "colors": colors, "radii": radii, "oxidation_states": dict(self.oxidation_states),
               "bonds": bool(self.bonds_enabled), "bond_factor": self.bond_factor, "bond_radius": self.bond_radius,
               "cell": bool(self.show_cell), "cell_width": self.cell_width,
               "boundary": list(self.boundary) or None, "boundary_bonds": self.boundary_bonds,
               "replicate": [int(x) for x in self.replicate] if self.replicate and tuple(self.replicate) != (1, 1, 1)
               else None, "replicate_cell": self.replicate_cell,
               "visuals": [dict(v) for v in self.visuals], "vectors": self.vectors,
               "trajectory_width": self.trajectory_width, "atoms": bool(self.show_atoms),
               **{k: (list(getattr(self, k)) if isinstance(getattr(self, k), (list, tuple)) else getattr(self, k))
                  for k in TRAJECTORY_KEYS},
               "cell_radius": self.cell_radius, "surfaces": {k: dict(v) for k, v in self.surfaces.items()},
               "transparency": dict(self.type_transparency), "selection": self.selection, "labels": self.labels,
               "light": self.light, "measurements": [list(m) for m in self.measurements]}
        if theme is None or theme.radius_preset != self.radius_preset:
            out["radius_preset"] = self.radius_preset
        base = theme.background if theme is not None and theme.background is not None else default_background()
        out["background"] = None if self.background == base else self.background
        if self.cell_color is not None and (theme is None or theme.cell_color != self.cell_color):
            out["cell_color"] = self.cell_color
        if self.bond_pairs and len(types):
            from .bonds import covalent_pairs

            auto = [[a, b, lo, round(float(hi), 2)] for a, b, lo, hi in
                    covalent_pairs(np.unique(np.asarray(types)), self.bond_factor)]
            if [list(p) for p in self.bond_pairs] != auto:
                out["bond_pairs"] = [list(p) for p in self.bond_pairs]
        return out
