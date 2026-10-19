"""Visual elements state (particles, bonds, cell): what OVITO calls visual elements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

STYLE_PRESETS = {
    # name: (radius_scale, bonds_enabled, atom_style)
    "Ball & stick": (0.5, True, "auto"),
    "Space-filling": (1.0, False, "auto"),
    "Simple (ASE)": (1.1, False, "flat"),  # ASE GUI: covalent x 0.89, flat outlined discs
    "Points": (0.25, False, "impostor"),
}
# Presets that also apply an ewald.themes theme when picked (ASE's white background).
PRESET_THEMES = {"Simple (ASE)": "ASE"}
DEFAULT_PRESET = "Ball & stick"
# Not bonded to each other by default in compounds (metal-metal contacts would clutter oxides etc.)
_NONMETALS = frozenset("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())
_THEMED = ("type_colors", "background", "cell_color", "radius_preset", "theme")


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
    background: str = "#14161A"
    bond_factor: float = 1.15                         # VESTA: max bond = factor * (r_A + r_B)
    visuals: list = field(default_factory=list)       # plugin visual elements: {name, params, enabled}
    theme: str = ""                                   # last applied ewald.themes name (its values are above)
    cell_color: str | None = None                     # None: picked from the background

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
        return kw

    def apply_theme(self, name: str, types=()):
        """Copy a theme's colors (for ``types``), background, cell color and radius preset into the style."""
        from ..themes import get_theme

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
        from ase.data import atomic_numbers, covalent_radii

        uniq = sorted(map(str, np.unique(np.asarray(types))))
        metal = {t: t in atomic_numbers and t not in _NONMETALS for t in uniq}
        compound = not all(metal.values())  # SrTiO3: Ti-O and Sr-O, not Sr-Ti; a pure metal keeps its bonds
        for a_i, a in enumerate(uniq):
            for b in uniq[a_i:]:
                if compound and metal[a] and metal[b]:
                    continue
                ra = covalent_radii[atomic_numbers[a]] if a in atomic_numbers else 0.8
                rb = covalent_radii[atomic_numbers[b]] if b in atomic_numbers else 0.8
                self.bond_pairs.append([a, b, 0.0, round(float(self.bond_factor * (ra + rb)), 2)])
