"""Visual elements state (particles, bonds, cell): what OVITO calls visual elements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STYLE_PRESETS = {
    # name: (radius_scale, bonds_enabled, atom_style)
    "Space-filling": (1.0, False, "auto"),
    "Ball & stick": (0.4, True, "auto"),
    "Points": (0.25, False, "impostor"),
}


@dataclass
class ViewStyle:
    preset: str = "Space-filling"
    radius_scale: float = 1.0
    atom_style: str = "auto"
    type_colors: dict = field(default_factory=dict)   # type -> (r, g, b) in 0..1 or '#hex'
    type_radii: dict = field(default_factory=dict)    # type -> radius (A)
    radius_preset: str = "covalent"                   # ewald.radii.PRESETS; type_radii win over it
    oxidation_states: dict = field(default_factory=dict)  # type -> oxidation state for the ionic presets
    bonds_enabled: bool = False
    bond_pairs: list = field(default_factory=list)    # [(A, B, dmin, dmax)]
    bond_radius: float = 0.15
    show_cell: bool = True
    cell_width: float = 2.0
    background: str = "#14161A"
    bond_factor: float = 1.15                         # VESTA: max bond = factor * (r_A + r_B)
    visuals: list = field(default_factory=list)       # plugin visual elements: {name, params, enabled}

    def apply_preset(self, name: str):
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
        return kw

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
        """Seed the bond table like VESTA: one row per type pair, max = bond_factor * sum of covalent radii."""
        if self.bond_pairs:
            return
        from ase.data import atomic_numbers, covalent_radii

        uniq = sorted(set(types))
        for a_i, a in enumerate(uniq):
            for b in uniq[a_i:]:
                ra = covalent_radii[atomic_numbers[a]] if a in atomic_numbers else 0.8
                rb = covalent_radii[atomic_numbers[b]] if b in atomic_numbers else 0.8
                self.bond_pairs.append([a, b, 0.0, round(self.bond_factor * (ra + rb), 2)])
