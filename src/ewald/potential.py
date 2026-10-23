"""Electrostatic potential along a cell axis: planar and macroscopic averages, the work
function of a slab and the potential lineup across an interface (band offsets and Schottky
barriers).

The lineup is the bulk-plus-lineup method: each side's band edges are measured against
its own macroscopic-average potential in a BULK reference cell, and the interface cell
supplies the step between the two sides' plateaus (Van de Walle & Martin, PRB 35, 8154
(1987), https://doi.org/10.1103/PhysRevB.35.8154). The macroscopic average is the planar
average filtered with one box per material period (Baldereschi, Baroni & Resta, PRL 61, 734
(1988), https://doi.org/10.1103/PhysRevLett.61.734; two successive filters: Colombo, Resta &
Baroni, PRB 44, 5572 (1991), https://doi.org/10.1103/PhysRevB.44.5572, and Junquera et al.,
PRB 67, 155327 (2003), https://doi.org/10.1103/PhysRevB.67.155327).

The filters, plateau rules, junction classification and barrier formulas are ported from
sim-agent (frssp/sim_agent ``src/sim_agent/band_alignment.py``, MIT) so a calculation analysed
here and in sim-agent's ``finalize_workgraph`` gives the same numbers; keep the two in step.
Where sim-agent reads the side atoms and periods from the builder's metadata, this module
derives them from the structures (:func:`split_sides`, :func:`normal_period`).

Every energy here is in eV, every length in Angstrom; profiles run along the third cell
vector (c), sampled uniformly over ``[0, |c|)`` as VASP's LOCPOT grid is.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: A bulk gap below this is a metal (sim-agent ``METAL_GAP_THRESHOLD_EV``).
METAL_GAP_THRESHOLD_EV = 0.1
#: Largest potential change across a plateau still read as flat (sim-agent ``PLATEAU_DRIFT_TOL_EV``).
PLATEAU_DRIFT_TOL_EV = 0.05
#: Shortest plateau worth fitting a slope to.
MIN_PLATEAU_LENGTH_A = 1.0
#: A work function outside this window is a misread profile, not a surface (sim-agent ``surface_core``).
PHI_PLAUSIBLE_RANGE_EV = (0.5, 12.0)
#: An occupation above this is a filled state, as pymatgen's ``eigenvalue_band_properties``
#: (``occu_tol``) which sim-agent uses for the band edges.
OCCUPIED = 1e-8

SUBSTRATE, FILM = "substrate", "film"


# ----------------------------------------------------------------------- profiles
@dataclass
class PlanarProfile:
    """A planar average along c: ``values`` (eV) on a uniform grid over ``[0, length)``."""

    values: np.ndarray
    length: float
    source: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        self.length = float(self.length)

    @property
    def z(self) -> np.ndarray:
        """Grid positions (Angstrom)."""
        return np.arange(self.values.size) * self.length / max(self.values.size, 1)

    def macroscopic(self, windows: Sequence[float]) -> np.ndarray:
        return macroscopic_average(self.values, self.length, windows)

    def __len__(self) -> int:
        return self.values.size

    def __repr__(self) -> str:
        return f"<PlanarProfile {self.values.size} points over {self.length:.3f} A{', ' + self.source if self.source else ''}>"


def _c_length(data) -> float:
    return float(np.linalg.norm(np.asarray(data.cell.matrix)[2]))


def planar_average(data, grid: str = "potential", property: str = "Potential") -> PlanarProfile:
    """Planar average along c of a volumetric property (a LOCPOT's ``"Potential"``)."""
    g = data.grids[grid]
    values = np.asarray(g[property], dtype=float)
    if values.ndim > 3:
        values = values[..., 0]
    return PlanarProfile(values.mean(axis=(0, 1)), float(np.linalg.norm(g.domain.matrix[2])), grid)


def read_planar_json(path, length) -> PlanarProfile:
    """``locpot_planar_avg.json`` as sim-agent's LVHAR statics write it (``planar_avg_z`` and the
    grid; no lengths), on the c length of the calculation's cell: a number, or a frame."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    try:
        values = raw["planar_avg_z"]
    except (KeyError, TypeError):
        raise ValueError(f"{path}: no 'planar_avg_z' (not a sim-agent locpot_planar_avg.json)") from None
    if not values:
        raise ValueError(f"{path}: the planar average is empty")
    if not isinstance(length, (int, float)):
        length = _c_length(length)
    return PlanarProfile(values, length, str(path))


def box_filter(profile: Sequence[float], length: float, window: float) -> np.ndarray:
    """Periodic running mean of width ``window`` (Angstrom), exact: applied in reciprocal space,
    where a box of width w multiplies the component q = 2 pi m / L by sin(qw/2)/(qw/2), so a
    component whose period divides w is removed entirely."""
    f = np.asarray(profile, dtype=float)
    if f.size == 0:
        return f.copy()
    if window <= 0 or length <= 0:
        raise ValueError(f"window and length must be positive (got {window}, {length})")
    m = np.fft.fftfreq(f.size, d=1.0 / f.size)
    return np.real(np.fft.ifft(np.fft.fft(f) * np.sinc(m * window / length)))


def macroscopic_average(profile: Sequence[float], length: float, windows: Sequence[float]) -> np.ndarray:
    """One box filter per window, in sequence (the double average across an interface)."""
    out = np.asarray(profile, dtype=float)
    for w in windows:
        out = box_filter(out, length, float(w))
    return out


# ----------------------------------------------------------------------- geometry
def _planes(z_frac: np.ndarray, length: float, tol: float) -> list[np.ndarray]:
    """Atom indices grouped into planes along c (periodic), in increasing z."""
    order = np.argsort(z_frac)
    z = z_frac[order] * length
    if z.size == 0:
        return []
    groups, current = [], [order[0]]
    for i in range(1, z.size):
        if z[i] - z[i - 1] > tol:
            groups.append(np.array(current))
            current = []
        current.append(order[i])
    groups.append(np.array(current))
    if len(groups) > 1 and (z[0] + length) - z[-1] <= tol:   # the last plane wraps onto the first
        groups[0] = np.concatenate([groups.pop(), groups[0]])
    return groups


def _composition(types: Sequence[str]) -> dict[str, float]:
    names, counts = np.unique(np.asarray(types).astype(str), return_counts=True)
    return {n: c / counts.sum() for n, c in zip(names, counts)}


def _distance(a: dict, b: dict) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def split_sides(interface, substrate, film, plane_tol: float = 0.3) -> dict:
    """Which atoms of an interface cell (c along the normal) are substrate and which film.

    Each plane of atoms along c goes to the side whose bulk reference composition it is
    closer to; a plane equally close to both (the shared anion plane of GaAs/AlAs) goes to
    the side of its neighbours when they agree and stays ``ambiguous`` otherwise, then
    joins the substrate. The result must be two contiguous layers; interleaved sides raise.

    Returns ``{"substrate": indices, "film": indices, "ambiguous": indices}``.
    """
    length = _c_length(interface)
    z = np.mod(interface.cell.to_fractional(interface.positions - interface.cell.origin)[:, 2], 1.0)
    types = np.asarray(interface.types).astype(str)
    refs = {SUBSTRATE: _composition(substrate.types), FILM: _composition(film.types)}
    planes = _planes(z, length, plane_tol)
    labels = []
    for p in planes:
        comp = _composition(types[p])
        d = {side: _distance(comp, ref) for side, ref in refs.items()}
        labels.append(None if abs(d[SUBSTRATE] - d[FILM]) < 1e-9 else min(d, key=d.get))
    ambiguous = []
    n = len(labels)
    for i, lab in enumerate(labels):
        if lab is None:
            before, after = labels[(i - 1) % n], labels[(i + 1) % n]
            if before is not None and before == after:
                labels[i] = before
            else:
                ambiguous.extend(planes[i].tolist())
                labels[i] = SUBSTRATE
    out = {side: np.sort(np.concatenate([p for p, lab in zip(planes, labels) if lab == side] or
                                        [np.array([], dtype=int)])).astype(int) for side in (SUBSTRATE, FILM)}
    if not out[SUBSTRATE].size or not out[FILM].size:
        raise ValueError("every plane of the interface matches one bulk reference: it is not an interface "
                         "of these two materials along c")
    if layers_overlap(z[out[SUBSTRATE]], z[out[FILM]]):
        raise ValueError("substrate and film planes interleave along c: not a layered interface with c "
                         "along the normal")
    out["ambiguous"] = np.array(sorted(ambiguous), dtype=int)
    return out


def normal_period(bulk, symprec: float = 1e-3) -> float:
    """Repeat distance along the normal (the direction of a x b) of a bulk reference cell:
    the primitive cell's interplanar spacing d for that plane, which is the period sim-agent
    filters with. Falls back to the cell's own spacing when no primitive cell is found."""
    m = np.asarray(bulk.cell.matrix, dtype=float)
    normal = np.cross(m[0], m[1])
    normal /= np.linalg.norm(normal)
    own = float(abs(np.dot(m[2], normal)))
    try:
        import spglib

        err = getattr(spglib, "error", None)
        if err is not None and hasattr(err, "OLD_ERROR_HANDLING"):
            err.OLD_ERROR_HANDLING = False   # spglib >= 2.5: raise instead of returning None (as modifiers/symmetry.py)
        names = list(dict.fromkeys(np.asarray(bulk.types).astype(str)))
        numbers = [names.index(t) + 1 for t in np.asarray(bulk.types).astype(str)]
        frac = bulk.cell.to_fractional(bulk.positions - bulk.cell.origin)
        prim = spglib.find_primitive((m, frac % 1.0, numbers), symprec=symprec)
    except Exception:  # noqa: BLE001 - spglib missing or failing: the cell itself
        prim = None
    if prim is None:
        return own
    recip = np.linalg.inv(np.asarray(prim[0], dtype=float)).T   # rows: b_i with a_i . b_j = delta_ij
    best = None
    r = range(-6, 7)
    for h in r:
        for k in r:
            for l in r:
                if (h, k, l) == (0, 0, 0):
                    continue
                g = h * recip[0] + k * recip[1] + l * recip[2]
                gn = np.linalg.norm(g)
                if np.linalg.norm(np.cross(g / gn, normal)) < 1e-4 and (best is None or gn < best):
                    best = gn
    return own if best is None else float(1.0 / best)


def layer_extent(z_frac: Sequence[float]) -> tuple[float, float]:
    """``(start, width)``, fractional, of the arc of the periodic cell one side's atoms occupy:
    everything except the largest empty stretch, so a layer across z = 0 is handled too."""
    z = np.sort(np.mod(np.asarray(z_frac, dtype=float), 1.0))
    if z.size == 0:
        raise ValueError("a side of the interface has no atoms")
    if z.size == 1:
        return float(z[0]), 0.0
    gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))
    i = int(np.argmax(gaps))
    return float(z[(i + 1) % z.size]), float(1.0 - gaps[i])


def layers_overlap(substrate_z: Sequence[float], film_z: Sequence[float]) -> bool:
    """True when either side's atoms fall inside the other side's arc."""
    for own, other in ((substrate_z, film_z), (film_z, substrate_z)):
        start, width = layer_extent(own)
        if np.any(np.mod(np.asarray(other, dtype=float) - start, 1.0) < width - 1e-9):
            return True
    return False


# ----------------------------------------------------------------------- lineup
def _measure_plateau(macro: np.ndarray, length: float, start: float, width: float, margin: float) -> dict:
    n = macro.size
    u = np.mod(np.arange(n) / n - start, 1.0) * length
    lo, hi = margin, width * length - margin
    sel = (u >= lo) & (u <= hi)
    plateau_len = max(0.0, hi - lo)
    base = {"layer_thickness_A": round(width * length, 4), "plateau_A": [round(lo, 4), round(hi, 4)],
            "plateau_length_A": round(plateau_len, 4), "n_points": int(sel.sum())}
    if plateau_len < MIN_PLATEAU_LENGTH_A:
        return {**base, "ok": False, "why": "short"}
    if sel.sum() < 3:
        return {**base, "ok": False, "why": "coarse"}
    x, y = u[sel], macro[sel]
    slope = float(np.polyfit(x, y, 1)[0])
    return {**base, "ok": True, "mean_eV": float(y.mean()), "std_eV": float(y.std()),
            "slope_eV_per_A": slope, "drift_eV": abs(slope) * plateau_len}


def potential_lineup(planar: PlanarProfile, substrate_z_frac: Sequence[float], film_z_frac: Sequence[float],
                     substrate_period: float, film_period: float, margin: float | None = None,
                     drift_tol: float = PLATEAU_DRIFT_TOL_EV) -> dict:
    """Step in macroscopic-average potential from substrate to film in an interface cell.

    ``substrate_z_frac``/``film_z_frac`` are the fractional c coordinates of each side's atoms
    (:func:`split_sides`), the periods the two filter widths (:func:`normal_period`).
    ``margin`` (default half the summed windows, the reach of the double filter) is kept
    from each layer's outermost atoms. Returns ``ok`` and, when ok, ``delta_v_eV`` =
    V(film) - V(substrate); otherwise ``code`` (``NOT_LAYERED``, ``THIN_LAYER``,
    ``GRID_TOO_COARSE``, ``PLATEAU_NOT_FLAT``), a ``reason`` and ``delta_v_eV`` None. Plateau
    diagnostics (``plateaus[side]``) and the averaged curve (``macroscopic``) are always there.
    """
    profile, length = planar.values, planar.length
    windows = [float(substrate_period), float(film_period)]
    margin = 0.5 * sum(windows) if margin is None else float(margin)
    result: dict[str, Any] = {"ok": False, "code": None, "reason": "", "delta_v_eV": None,
                              "windows_A": [round(w, 4) for w in windows], "margin_A": round(margin, 4),
                              "drift_tol_eV": drift_tol, "plateaus": {}}
    if layers_overlap(substrate_z_frac, film_z_frac):
        result.update(code="NOT_LAYERED", reason="The substrate and film atoms interleave along c, so neither "
                      "side has a bulk-like region to read a potential from.")
        return result
    macro = macroscopic_average(profile, length, windows)
    result["macroscopic"] = macro
    for side, zs in ((SUBSTRATE, substrate_z_frac), (FILM, film_z_frac)):
        start, width = layer_extent(zs)
        result["plateaus"][side] = {"start_frac": start, **_measure_plateau(macro, length, start, width, margin)}
    plateaus = result["plateaus"]
    thin = [s for s, p in plateaus.items() if p.get("why") == "short"]
    if thin:
        need = 2 * margin + MIN_PLATEAU_LENGTH_A
        detail = ", ".join(f"{s} {plateaus[s]['layer_thickness_A']:.1f} A" for s in thin)
        result.update(code="THIN_LAYER", reason=f"No bulk-like region inside: {detail}. Each side needs at least "
                      f"{need:.1f} A between its outermost atoms (twice the {margin:.1f} A filter reach plus "
                      f"{MIN_PLATEAU_LENGTH_A:.1f} A to measure).")
        return result
    coarse = [s for s, p in plateaus.items() if not p["ok"]]
    if coarse:
        dz = length / max(len(profile), 1)
        result.update(code="GRID_TOO_COARSE", reason=f"The potential grid ({dz:.2f} A spacing) puts fewer than 3 "
                      f"points on the {', '.join(coarse)} plateau: a denser FFT grid along c is needed, not "
                      "thicker layers.")
        return result
    tilted = {s: p for s, p in plateaus.items() if p["drift_eV"] > drift_tol}
    if tilted:
        detail = ", ".join(f"{s} drifts {p['drift_eV']:.3f} eV over {p['plateau_length_A']:.1f} A"
                           for s, p in tilted.items())
        result.update(code="PLATEAU_NOT_FLAT", reason=f"The averaged potential is not flat inside the layers "
                      f"({detail}; tolerance {drift_tol:.3f} eV), so the step depends on where it is read and "
                      "no offset is given. Usual causes: inequivalent seams (a field across both layers) or "
                      "layers too thin to screen the seam.")
        return result
    result.update(ok=True, delta_v_eV=plateaus[FILM]["mean_eV"] - plateaus[SUBSTRATE]["mean_eV"])
    return result


# ----------------------------------------------------------------------- band edges and alignment
def bands_cross_fermi(eigenvalues, efermi: float, tol: float = 1e-4) -> bool:
    """True when a band has states both below and above E_F: a metal (pymatgen's
    ``BandStructure.is_metal`` criterion on raw eigenvalues, ``(nspin, nk, nb)`` or ``(nk, nb)``)."""
    e = np.asarray(eigenvalues, dtype=float)
    if e.ndim == 2:
        e = e[None]
    for channel in e:
        if channel.size and np.any((channel < efermi - tol).any(axis=0) & (channel > efermi + tol).any(axis=0)):
            return True
    return False


def bulk_edges(data, planar: PlanarProfile | None = None) -> dict:
    """``{"vbm", "cbm", "efermi", "is_metal"}`` (eV, absolute) of a bulk reference frame read from
    a vasprun.xml (its ``"Electronic Structure"`` attribute), and ``v_mean``, the mean of its
    planar-averaged potential, when ``planar`` is given."""
    es = data.attributes.get("Electronic Structure") if hasattr(data, "attributes") else data
    if es is None or not getattr(es, "has_bands", False):
        raise ValueError("no eigenvalues in this frame: open the reference's vasprun.xml")
    gap = es.band_gap(occupied=OCCUPIED)
    if gap is None:
        raise ValueError("the eigenvalues are all filled or all empty: no band edge")
    out = {"vbm": gap["vbm"], "cbm": gap["cbm"], "efermi": float(es.efermi),
           "is_metal": bands_cross_fermi(es.eigenvalues, float(es.efermi))}
    if planar is not None:
        out["v_mean"] = float(np.mean(planar.values))
    return out


def bulk_gap(bulk: dict) -> float:
    """Gap of a bulk reference (never negative; 0 for one whose bands cross E_F: a metal's edge
    spacing is a k-mesh artifact)."""
    if bulk.get("is_metal") is True:
        return 0.0
    return max(0.0, float(bulk["cbm"]) - float(bulk["vbm"]))


def classify_junction(substrate_gap: float, film_gap: float, threshold: float = METAL_GAP_THRESHOLD_EV) -> str:
    """``semiconductor/semiconductor``, ``metal/semiconductor``, ... (substrate first)."""
    def kind(g):
        return "metal" if g < threshold else "semiconductor"

    return f"{kind(substrate_gap)}/{kind(film_gap)}"


def _alignment_type(ev_a: float, ec_a: float, ev_b: float, ec_b: float) -> str:
    """Type I (straddling), II (staggered) or III (broken gap)."""
    if ev_a > ec_b or ev_b > ec_a:
        return "III"
    if (ev_a <= ev_b and ec_a >= ec_b) or (ev_b <= ev_a and ec_b >= ec_a):
        return "I"
    return "II"


def band_alignment(lineup: dict, substrate_bulk: dict, film_bulk: dict, interface_fermi: float | None,
                   gap_threshold: float = METAL_GAP_THRESHOLD_EV) -> dict:
    """Band edges of both sides on the interface's energy axis, and what they make: ``ΔE_v`` /
    ``ΔE_c`` (film - substrate) for two semiconductors, ``Φ_Bp`` = E_F - VBM and ``Φ_Bn`` =
    CBM - E_F for a metal on a semiconductor. ``substrate_bulk``/``film_bulk`` come from
    :func:`bulk_edges` (with ``v_mean``). Returns ``ok``, ``junction``, ``bulk_gaps_eV``,
    ``band_edges_on_interface_axis`` and the numbers, or ``code``/``reason``."""
    gaps = {SUBSTRATE: bulk_gap(substrate_bulk), FILM: bulk_gap(film_bulk)}
    junction = classify_junction(gaps[SUBSTRATE], gaps[FILM], gap_threshold)
    out: dict[str, Any] = {"ok": False, "junction": junction, "bulk_gaps_eV": gaps,
                           "gap_threshold_eV": gap_threshold, "code": None, "reason": ""}
    if junction == "metal/metal":
        out.update(code="NO_BAND_EDGE", reason=f"Both bulk references are metallic (gaps {gaps[SUBSTRATE]:.3f} / "
                   f"{gaps[FILM]:.3f} eV): there is no band edge to align.")
        return out
    if not lineup.get("ok"):
        out.update(code=lineup.get("code"), reason=lineup.get("reason", ""))
        return out
    edges = {}
    for side, bulk in ((SUBSTRATE, substrate_bulk), (FILM, film_bulk)):
        shift = lineup["plateaus"][side]["mean_eV"] - float(bulk["v_mean"])
        edges[side] = {"vbm_eV": float(bulk["vbm"]) + shift, "cbm_eV": float(bulk["cbm"]) + shift}
    out["band_edges_on_interface_axis"] = edges
    if junction == "semiconductor/semiconductor":
        sub, film = edges[SUBSTRATE], edges[FILM]
        out.update(ok=True, valence_band_offset_eV=film["vbm_eV"] - sub["vbm_eV"],
                   conduction_band_offset_eV=film["cbm_eV"] - sub["cbm_eV"],
                   alignment_type=_alignment_type(sub["vbm_eV"], sub["cbm_eV"], film["vbm_eV"], film["cbm_eV"]))
        return out
    if interface_fermi is None:
        out.update(code="FERMI_LEVEL_MISSING", reason=f"A {junction} junction needs the interface calculation's "
                   "Fermi energy for the Schottky barrier.")
        return out
    semi = SUBSTRATE if junction.endswith("metal") else FILM
    ef = float(interface_fermi)
    phi_p, phi_n = ef - edges[semi]["vbm_eV"], edges[semi]["cbm_eV"] - ef
    out.update(ok=True, semiconductor_side=semi, interface_fermi_eV=ef, schottky_barrier_p_eV=phi_p,
               schottky_barrier_n_eV=phi_n, fermi_level_in_gap=bool(phi_p >= 0 and phi_n >= 0))
    return out


def scissor_schottky(result: dict, reference_gap: float) -> dict:
    """Scissor-corrected n-type barrier, an APPROXIMATION shown beside the computed one:
    Φ_Bn* = E_g(ref) - Φ_Bp, the semiconductor's CBM moved rigidly while its VBM, the lineup
    and E_F stay as computed (sim-agent ``scissor_schottky``)."""
    if not result.get("ok") or "schottky_barrier_p_eV" not in result:
        return {"applied": False, "reason": (
            "a scissor correction applies to a completed metal/semiconductor result (Schottky barrier); "
            "band offsets of two semiconductors need one reference gap per side and are not corrected here.")}
    try:
        ref = float(reference_gap)
    except (TypeError, ValueError):
        return {"applied": False, "reason": f"reference gap {reference_gap!r} is not a number."}
    if not ref > 0:
        return {"applied": False, "reason": f"reference gap {ref} eV must be positive."}
    semi = result["semiconductor_side"]
    return {"applied": True, "reference_gap_eV": ref, "computed_gap_eV": float(result["bulk_gaps_eV"][semi]),
            "cbm_shift_eV": ref - float(result["bulk_gaps_eV"][semi]),
            "schottky_barrier_n_scissor_eV": ref - float(result["schottky_barrier_p_eV"]),
            "cbm_scissor_eV": float(result["band_edges_on_interface_axis"][semi]["vbm_eV"]) + ref,
            "fermi_above_computed_cbm": float(result["schottky_barrier_n_eV"]) < 0}


def energy_zero(result: dict) -> tuple[float, str]:
    """The energy zero of a lineup plot and its name: the interface E_F with a metal, the
    substrate VBM for two semiconductors, else the raw VASP scale."""
    ef = result.get("interface_fermi_eV")
    if ef is not None:
        return float(ef), "E - E_F (interface)"
    sub = (result.get("band_edges_on_interface_axis") or {}).get(SUBSTRATE)
    if sub and result["bulk_gaps_eV"].get(SUBSTRATE, 0) >= result.get("gap_threshold_eV", METAL_GAP_THRESHOLD_EV):
        return float(sub["vbm_eV"]), "E - VBM (substrate)"
    return 0.0, "potential (VASP scale)"


# ----------------------------------------------------------------------- work function
def work_function(planar: PlanarProfile, efermi: float) -> dict:
    """Φ = V_vacuum - E_F of a slab, with the vacuum level the maximum of the planar average
    (the vacuum plateau; sim-agent ``surface_core.extract_vacuum_level``). ``plausible`` is
    False outside :data:`PHI_PLAUSIBLE_RANGE_EV`, the signature of a misread profile."""
    i = int(np.argmax(planar.values))
    v = float(planar.values[i])
    phi = v - float(efermi)
    lo, hi = PHI_PLAUSIBLE_RANGE_EV
    return {"work_function_eV": phi, "vacuum_level_eV": v, "vacuum_z_A": float(planar.z[i]),
            "fermi_eV": float(efermi), "plausible": bool(lo <= phi <= hi)}


# ----------------------------------------------------------------------- as a table
@dataclass
class Level:
    """A horizontal energy level for a plot: ``energy`` on the table's energy axis, drawn over
    ``z`` (Angstrom, None = the whole width), ``style`` "solid", "dashed" or "dotted"."""

    name: str
    energy: float
    z: tuple | None = None
    style: str = "solid"
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "energy": self.energy, "z": None if self.z is None else list(self.z),
                "style": self.style, "note": self.note}


def lineup_table(planar: PlanarProfile, lineup: dict, alignment: dict, scissor: dict | None = None,
                 names: dict | None = None):
    """A ``Lineup`` table (z, planar and macroscopic average, on the alignment's energy zero) whose
    attributes carry what a plot draws over it: ``levels`` (E_F and the band edges of each side
    with a gap, over that side's plateau), ``regions`` (the plateaus) and ``energy_axis``,
    plus the numbers (``result``). ``names`` labels the sides (``{"film": "TiO2"}``)."""
    from .verbs import Table

    names = {SUBSTRATE: SUBSTRATE, FILM: FILM, **(names or {})}
    zero, axis = energy_zero(alignment)
    z = planar.z
    columns = {"z (A)": z, "Planar average": planar.values - zero}
    if lineup.get("macroscopic") is not None:
        columns["Macroscopic average"] = np.asarray(lineup["macroscopic"]) - zero
    regions, levels = [], []
    edges = alignment.get("band_edges_on_interface_axis") or {}
    threshold = alignment.get("gap_threshold_eV", METAL_GAP_THRESHOLD_EV)
    for side, p in (lineup.get("plateaus") or {}).items():
        if "plateau_A" not in p:
            continue
        lo = p["start_frac"] * planar.length + p["plateau_A"][0]
        span = (lo, lo + p["plateau_A"][1] - p["plateau_A"][0])
        regions.append({"name": f"{names[side]} plateau", "z": list(span), "side": side})
        if side in edges and alignment["bulk_gaps_eV"].get(side, 0) >= threshold:
            levels.append(Level(f"{names[side]} VBM", edges[side]["vbm_eV"] - zero, span,
                                note="bulk reference + lineup").as_dict())
            levels.append(Level(f"{names[side]} CBM", edges[side]["cbm_eV"] - zero, span).as_dict())
            if (scissor or {}).get("applied") and side == alignment.get("semiconductor_side"):
                levels.append(Level(f"{names[side]} CBM* (scissor, approx.)", scissor["cbm_scissor_eV"] - zero,
                                    span, "dotted", f"E_g {scissor['reference_gap_eV']} eV").as_dict())
    if alignment.get("interface_fermi_eV") is not None:
        levels.append(Level("E_F (interface)", alignment["interface_fermi_eV"] - zero, None, "dashed").as_dict())
    result = {k: v for k, v in alignment.items() if not isinstance(v, np.ndarray)}
    if scissor:
        result["scissor"] = scissor
    result["plateaus"] = {s: {k: v for k, v in p.items()} for s, p in (lineup.get("plateaus") or {}).items()}
    return Table("Lineup", columns, attributes={"energy_axis": axis, "energy_zero_eV": zero, "levels": levels,
                                               "regions": regions, "result": result})
