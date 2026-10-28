"""Tabulated atomic radii as presets: covalent, atomic, van der Waals and Shannon ionic radii.

| preset | source |
|---|---|
| covalent | Cordero et al., Dalton Trans. 2832 (2008) (``ase.data.covalent_radii``) |
| atomic | Slater, J. Chem. Phys. 41, 3199 (1964), empirical |
| atomic (calc.) | Clementi, Raimondi, Reinhardt, J. Chem. Phys. 47, 1300 (1967), calculated |
| vdW | Alvarez, Dalton Trans. 42, 8617 (2013) (``ase.data.vdw_alvarez``) |
| vdW (Bondi) | Bondi, J. Phys. Chem. 68, 441 (1964), completed from WebElements (``ase.data.vdw_radii``) |
| ionic | Shannon, Acta Cryst. A32, 751 (1976), effective ionic radii |
| ionic (crystal) | Shannon (1976) crystal radii (ionic + 0.14 Angstrom for most ions) |

Slater, Clementi and Shannon values are bundled in ``tables/radii.json`` (extracted from pymatgen,
MIT; regenerate with ``tables/make_radii.py``). An element missing from a table falls back to the
covalent radius, and the fallback is reported.

Shannon radii depend on the oxidation state and the coordination number (CN). :func:`resolve` takes
the oxidation state of each type from, in order: the ``oxidation_states`` argument, a type name such
as ``"Fe3+"`` or ``"O2-"``, an integer "Oxidation State" or "Charge" property, a pymatgen
bond-valence analysis (small cells only), a charge-balanced guess
from the oxidation states most common in the ICSD, and finally the most common state alone. The CN is counted from the structure (oxidation
partners closer than 1.25 x the typical nearest cation-anion distance) unless ``coordination`` gives
it; without a structure it is VI. The nearest CN the table has is used. A type left without an
oxidation state, or whose ion is not in the table, falls back to the covalent radius.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PRESETS = {
    "covalent": "Cordero et al., Dalton Trans. 2832 (2008)",
    "atomic": "Slater, J. Chem. Phys. 41, 3199 (1964)",
    "atomic (calc.)": "Clementi, Raimondi, Reinhardt, J. Chem. Phys. 47, 1300 (1967)",
    "vdW": "Alvarez, Dalton Trans. 42, 8617 (2013)",
    "vdW (Bondi)": "Bondi, J. Phys. Chem. 68, 441 (1964); WebElements for the rest",
    "ionic": "Shannon, Acta Cryst. A32, 751 (1976), effective ionic radii",
    "ionic (crystal)": "Shannon, Acta Cryst. A32, 751 (1976), crystal radii",
}
_ALIASES = {"cordero": "covalent", "slater": "atomic", "empirical": "atomic", "atomic_calculated": "atomic (calc.)",
            "atomic-calc": "atomic (calc.)", "calculated": "atomic (calc.)", "clementi": "atomic (calc.)",
            "vdw": "vdW", "alvarez": "vdW", "bondi": "vdW (Bondi)", "vdw_bondi": "vdW (Bondi)",
            "shannon": "ionic", "crystal": "ionic (crystal)", "ionic_crystal": "ionic (crystal)"}
IONIC = ("ionic", "ionic (crystal)")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IIIPY": 3, "IV": 4, "IVPY": 4, "IVSQ": 4, "V": 5, "VI": 6, "VII": 7,
          "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIV": 14}
_SPIN_ORDER = ("High Spin", "", "Low Spin")
_ION = re.compile(r"^([A-Z][a-z]?)(\d*)([+-])$")
CN_FACTOR = 1.25
CN_CUTOFF = 4.5          # Angstrom; longest cation-anion contact considered
CN_MAX_ATOMS = 50_000    # above this the CN is not counted (pass ``coordination``)
BV_MAX_ATOMS = 500       # pymatgen's BVAnalyzer is slow; only for small cells
_WARNED: set = set()


def preset_name(name: str | None) -> str:
    """Canonical preset name; accepts the aliases 'slater', 'clementi', 'bondi', 'shannon', 'crystal', ..."""
    if name is None:
        return "covalent"
    if name in PRESETS:
        return name
    key = str(name).strip().lower()
    for canon in PRESETS:
        if canon.lower() == key:
            return canon
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown radius preset {name!r}; use one of {list(PRESETS)}")


@lru_cache(maxsize=1)
def _tables() -> dict:
    return json.loads(Path(__file__).with_name("tables").joinpath("radii.json").read_text(encoding="utf-8"))


def element_of(type_name) -> str | None:
    """Element symbol of a particle type name ('Cu', 'Fe3+', 'O2-'), or None."""
    from ase.data import atomic_numbers

    t = str(type_name)
    if t in atomic_numbers:
        return t
    m = _ION.match(t)
    return m.group(1) if m and m.group(1) in atomic_numbers else None


def atomic_number(type_name) -> int:
    """Atomic number of a particle type name ('Cu', 'cu', 'Fe3+'), 0 when it names no element."""
    from ase.data import atomic_numbers

    el = element_of(type_name) or element_of(str(type_name).capitalize())
    return atomic_numbers[el] if el else 0


def element_masses(types, default: float | None = 1.0) -> np.ndarray:
    """Standard atomic mass (amu) of each type name; ``default`` for names that are no element
    (NaN when ``default`` is None)."""
    from ase.data import atomic_masses

    names, inverse = np.unique(np.asarray([str(t) for t in types], dtype=object), return_inverse=True)
    fill = np.nan if default is None else default
    table = np.array([atomic_masses[z] if (z := atomic_number(t)) else fill for t in names], float)
    return table[inverse.reshape(-1)] if len(names) else np.zeros(0)


def parse_oxidation_states(text: str) -> dict:
    """``"Sr=2, Ti=+4, O=-2"`` (or ``Ti:4``, spaces) -> ``{"Sr": 2, "Ti": 4, "O": -2}``."""
    out = {}
    for item in re.split(r"[,;\s]+", text.strip()):
        if not item:
            continue
        m = re.fullmatch(r"([^=:]+)[=:]([+-]?\d+)([+-]?)", item)
        if not m:
            raise ValueError(f"cannot read {item!r}; write e.g. Ti=4, O=-2")
        n = int(m.group(2))
        out[m.group(1)] = -abs(n) if m.group(3) == "-" else n
    return out


def format_oxidation_states(states: dict) -> str:
    return ", ".join(f"{t}={int(o):+d}" for t, o in states.items())


def _oxidation_from_name(type_name) -> int | None:
    m = _ION.match(str(type_name))
    if not m:
        return None
    return int(m.group(2) or 1) * (1 if m.group(3) == "+" else -1)


def _table_radius(symbol: str, preset: str) -> float | None:
    from ase.data import atomic_numbers, covalent_radii

    z = atomic_numbers[symbol]
    if preset == "covalent":
        return float(covalent_radii[z])
    if preset in ("vdW", "vdW (Bondi)"):
        if preset == "vdW":
            from ase.data.vdw_alvarez import vdw_radii
        else:
            from ase.data import vdw_radii
        r = vdw_radii[z] if z < len(vdw_radii) else np.nan
        return None if np.isnan(r) else float(r)
    table = _tables()["atomic" if preset == "atomic" else "atomic_calculated"]
    return table.get(symbol)


def shannon_radius(symbol: str, oxidation: int, cn: int | None = None, kind: str = "ionic",
                   spin: str | None = None) -> tuple[float, int] | None:
    """Shannon radius of ``symbol`` in ``oxidation`` state at the tabulated CN nearest ``cn`` (VI if None).

    Returns ``(radius, cn_used)`` or None when the ion is not tabulated. ``kind`` is 'ionic' or
    'crystal'; ``spin`` 'high' (default when both exist) or 'low'.
    """
    cns = _tables()["shannon"].get(symbol, {}).get(str(int(oxidation)))
    if not cns:
        return None
    want = 6 if cn is None else int(cn)
    # plain roman numerals before the square-planar/pyramidal variants at the same CN
    best = min(cns, key=lambda k: (abs(_ROMAN[k] - want), _ROMAN[k] > want, k not in ("IV", "III")))
    spins = cns[best]
    order = ("Low Spin", "", "High Spin") if spin == "low" else _SPIN_ORDER
    entry = next(spins[s] for s in order if s in spins)
    return float(entry[0 if kind == "crystal" else 1]), _ROMAN[best]


def common_oxidation_state(symbol: str) -> int | None:
    """Most frequent nonzero oxidation state of ``symbol`` in the ICSD that has a Shannon radius."""
    occ = _tables()["oxidation_occurrence"].get(symbol, {})
    shannon = _tables()["shannon"].get(symbol, {})
    ranked = sorted(((n, int(o)) for o, n in occ.items() if int(o) != 0), reverse=True)
    for _, o in ranked:
        if str(o) in shannon:
            return o
    # not in the ICSD counts: the state with the most tabulated environments
    return int(max(shannon, key=lambda o: len(shannon[o]))) if shannon else None


def element_radius(symbol: str, preset: str = "covalent", oxidation: int | None = None,
                   cn: int | None = None) -> float | None:
    """Unscaled radius of an element under ``preset``, falling back to covalent; None for non-elements.

    Ionic presets without ``oxidation`` use the most common oxidation state (for tables that show
    one number per element, like Preferences > Elements).
    """
    sym = element_of(symbol)
    if sym is None:
        return None
    preset = preset_name(preset)
    if preset in IONIC:
        oxi = oxidation if oxidation is not None else common_oxidation_state(sym)
        hit = shannon_radius(sym, oxi, cn, "crystal" if preset == "ionic (crystal)" else "ionic") if oxi else None
        return hit[0] if hit else _table_radius(sym, "covalent")
    r = _table_radius(sym, preset)
    return r if r is not None else _table_radius(sym, "covalent")


@dataclass
class RadiusTable:
    """Per-type radii resolved for one frame, with where each value came from."""

    preset: str
    radii: dict = field(default_factory=dict)        # type -> radius (A), unscaled
    oxidation: dict = field(default_factory=dict)    # type -> (state, source); ionic presets only
    coordination: dict = field(default_factory=dict)  # type -> CN used
    fallback: dict = field(default_factory=dict)     # type -> reason it got the covalent radius

    def summary(self) -> str:
        parts = []
        for t, r in self.radii.items():
            s = f"{t} {r:.3f}"
            if t in self.oxidation and self.oxidation[t][0] is not None:
                oxi, src = self.oxidation[t]
                s += f" ({oxi:+d}, CN {self.coordination.get(t, '?')}, {src})"
            if t in self.fallback:
                s += f" (covalent: {self.fallback[t]})"
            parts.append(s)
        return f"{self.preset} radii: " + ", ".join(parts)


def resolve(data, preset: str = "covalent", oxidation_states: dict | None = None,
            coordination: dict | None = None) -> RadiusTable:
    """Radius of every particle type in ``data`` under ``preset`` (unscaled), see the module docstring."""
    preset = preset_name(preset)
    from .render import type_groups

    types = [str(t) for t in type_groups(data.types)[0]] if data.count else []
    out = RadiusTable(preset)
    elements = {t: element_of(t) for t in types}
    if preset not in IONIC:
        for t, sym in elements.items():
            if sym is None:
                continue
            r = _table_radius(sym, preset)
            if r is None:
                r = _table_radius(sym, "covalent")
                out.fallback[t] = f"no {preset} radius for {sym}"
            out.radii[t] = r
        return out

    out.oxidation = guess_oxidation_states(data, oxidation_states)
    cn = count_coordination(data, {t: o for t, (o, _) in out.oxidation.items()}, coordination)
    kind = "crystal" if preset == "ionic (crystal)" else "ionic"
    for t, sym in elements.items():
        if sym is None:
            continue
        oxi = out.oxidation.get(t, (None, ""))[0]
        hit = shannon_radius(sym, oxi, cn.get(t), kind) if oxi else None
        if hit:
            out.radii[t], out.coordination[t] = hit
        else:
            out.radii[t] = _table_radius(sym, "covalent")
            out.fallback[t] = (f"no Shannon radius for {sym}{abs(oxi)}{'+' if oxi > 0 else '-'}" if oxi
                               else "oxidation state unknown")
    if out.fallback and (msg := out.summary()) not in _WARNED:
        _WARNED.add(msg)  # once per distinct result, not once per frame
        log.warning("%s; pass oxidation_states={...} to set them", msg)
    return out


def guess_oxidation_states(data, given: dict | None = None) -> dict:
    """``{type: (oxidation_state or None, source)}``; see the module docstring for the order."""
    from .render import type_groups

    types = [str(t) for t in type_groups(data.types)[0]] if data.count else []
    out = {}
    given = {str(k): v for k, v in (given or {}).items()}
    for t in types:
        sym = element_of(t)
        if t in given or (sym in given and sym != t):
            out[t] = (int(given[t] if t in given else given[sym]), "given")
        elif (o := _oxidation_from_name(t)) is not None:
            out[t] = (o, "type name")
    tarr = np.asarray(data.types).astype(str)
    for prop in ("Oxidation State", "Charge"):
        if prop in data and len(out) < len(types):
            q = np.asarray(data[prop], float)
            for t in types:
                if t in out:
                    continue
                v = np.unique(q[tarr == t])
                if len(v) == 1 and v[0] != 0 and abs(v[0] - round(v[0])) < 1e-6:
                    out[t] = (int(round(v[0])), prop)
    missing = [t for t in types if t not in out and element_of(t)]
    if missing:
        guess, source = _bond_valence(data), "bond valence"
        if not guess or any(t not in guess for t in missing):
            guess, source = _charge_balance(data, {t: o for t, (o, _) in out.items()}), "charge balance"
        single = len({element_of(t) for t in types}) == 1 and not out  # a pure element: no ions
        for t in missing:
            if guess.get(t) is not None:
                out[t] = (guess[t], source)
            elif not single and (o := common_oxidation_state(element_of(t))) is not None:
                out[t] = (o, "most common")  # no charge-neutral assignment (e.g. mixed valence)
            else:
                out[t] = (None, "unknown")
    return out


def _bond_valence(data) -> dict:
    """Per-type oxidation states from pymatgen's BVAnalyzer, or {} (not installed, too big, no cell)."""
    if data.count > BV_MAX_ATOMS or not data.cell.is_valid or not np.all(data.cell.pbc):
        return {}
    try:
        from pymatgen.core import Lattice, Structure
        try:
            from pymatgen.core.bond_valence import BVAnalyzer      # pymatgen-core >= 2026
        except ImportError:
            from pymatgen.analysis.bond_valence import BVAnalyzer  # older pymatgen
    except ImportError:
        return {}
    types = np.asarray(data.types).astype(str)
    species = [element_of(t) for t in types]
    if any(s is None for s in species):
        return {}
    try:
        s = Structure(Lattice(data.cell.matrix), species, data.positions - data.cell.origin,
                      coords_are_cartesian=True)
        valences = BVAnalyzer().get_valences(s)
    except Exception as exc:  # BVAnalyzer raises ValueError when no charge-neutral assignment exists
        log.debug("bond-valence analysis failed: %s", exc)
        return {}
    out = {}
    for t in dict.fromkeys(types.tolist()):
        vals = np.asarray(valences)[types == t]
        u, n = np.unique(np.rint(vals).astype(int), return_counts=True)
        if u[np.argmax(n)] != 0:
            out[t] = int(u[np.argmax(n)])
    return out


def _charge_balance(data, known: dict) -> dict:
    """One nonzero oxidation state per type, charge neutral overall, maximizing ICSD frequency."""
    from itertools import product

    types = np.asarray(data.types).astype(str)
    uniq, counts = np.unique(types, return_counts=True)
    count = dict(zip(uniq.tolist(), counts.tolist()))
    free = [t for t in count if t not in known and element_of(t)]
    if not free or (len(count) == 1 and not known):  # a pure element has no ions to balance
        return {}
    fixed_charge = sum(count[t] * q for t, q in known.items() if q is not None and t in count)
    if len(free) + len(known) < len(count):
        return {}  # non-element types with unknown charge
    occ = _tables()["oxidation_occurrence"]
    choices = []
    for t in free:
        states = {int(o): n for o, n in occ.get(element_of(t), {}).items() if int(o) != 0}
        total = sum(states.values())
        top = sorted(states.items(), key=lambda kv: -kv[1])[:4]
        if not top:
            return {}
        choices.append([(o, np.log(n / total)) for o, n in top])
    best, best_score = None, -np.inf
    for combo in product(*choices):
        if fixed_charge + sum(count[t] * o for t, (o, _) in zip(free, combo)) == 0:
            score = sum(s for _, s in combo)
            if score > best_score:
                best, best_score = combo, score
    return {} if best is None else {t: o for t, (o, _) in zip(free, best)}


def count_coordination(data, oxidation: dict, given: dict | None = None) -> dict:
    """Most common number of counter-ions around each type (cations count anions and vice versa).

    A pair counts when it is shorter than ``CN_FACTOR`` x the median nearest distance between the
    two types, so mixed environments (O in SrTiO3: 2 Ti + 4 Sr) are counted right.
    """
    from .bonds import _half_pairs

    out = {str(k): int(v) for k, v in (given or {}).items()}
    if data.count == 0 or data.count > CN_MAX_ATOMS:
        return out
    types = np.asarray(data.types).astype(str)
    sign = np.array([np.sign(oxidation.get(t) or 0) for t in types])
    if not (np.any(sign > 0) and np.any(sign < 0)):
        return out
    i, j, _, dist = _half_pairs(data, CN_CUTOFF)
    opp = sign[i] * sign[j] < 0
    i, j, dist = i[opp], j[opp], dist[opp]
    if not len(i):
        return out
    # nearest counter-ion distance per atom and type pair, then the median over atoms per type pair
    ii, jj, dd = np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([dist, dist])
    ti, tj = types[ii], types[jj]
    ref = {}
    for a, b in {(a, b) for a, b in zip(ti.tolist(), tj.tolist())}:
        m = (ti == a) & (tj == b)
        order = np.lexsort((dd[m], ii[m]))
        first = np.r_[True, np.diff(ii[m][order]) != 0]
        ref[a, b] = float(np.median(dd[m][order][first]))
    cut = np.array([CN_FACTOR * min(ref[a, b], ref[b, a]) for a, b in zip(ti.tolist(), tj.tolist())])
    near = dd < cut
    per_atom = np.bincount(ii[near], minlength=data.count)
    for t in dict.fromkeys(types.tolist()):
        if t in out or sign[types == t][0] == 0:
            continue
        cn = per_atom[types == t]
        cn = cn[cn > 0]
        if len(cn):
            u, n = np.unique(cn, return_counts=True)
            out[t] = int(u[np.argmax(n)])
    return out
