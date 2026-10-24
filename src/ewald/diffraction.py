"""Powder diffraction and static structure factors.

Three things share one structure-factor kernel here:

- :func:`powder_peaks`: X-ray or neutron powder reflections of a periodic cell: every reciprocal
  lattice point inside the limiting sphere, atomic form factors (or scattering lengths), the
  Lorentz-polarization factor. Same formalism and tables as pymatgen's ``XRDCalculator`` and
  ``NDCalculator`` (tested against them), so a supercell or an MD snapshot gives the pattern of
  that exact configuration, thermal diffuse scattering included.
- :func:`powder_profile`: the peaks broadened into a continuous pattern (pseudo-Voigt).
- :func:`structure_factor`: S(q) for liquids, glasses and MD, by the direct sum over the
  reciprocal lattice of the cell, by the Fourier transform of g(r), or by the Debye equation
  (finite clusters), optionally X-ray or neutron weighted (Faber-Ziman).

The amplitude sum ``F(hkl) = sum_j w_j exp(2 pi i hkl . r_j)`` is separable in h, k and l, so it
is computed as one matrix product per h (``(K x n) @ (n x L)``), written once against ``xp`` so
``device="cuda"`` runs it on the GPU. The cost grows with atoms x reflections (cell volume x
q_max^3): a few seconds for ~10k atoms at Cu K-alpha.

>>> from ewald.diffraction import powder_peaks
>>> peaks = powder_peaks(data.positions, data.types, data.cell, wavelength="CuKa")
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .backend import get_backend
from .data import Cell

# Characteristic wavelengths in Angstrom (pymatgen's table; K-alpha is the 2:1 weighted mean).
WAVELENGTHS = {
    "CuKa": 1.54184, "CuKa2": 1.54439, "CuKa1": 1.54056, "CuKb1": 1.39222,
    "MoKa": 0.71073, "MoKa2": 0.71359, "MoKa1": 0.70930, "MoKb1": 0.63229,
    "CrKa": 2.29100, "CrKa2": 2.29361, "CrKa1": 2.28970, "CrKb1": 2.08487,
    "FeKa": 1.93735, "FeKa2": 1.93998, "FeKa1": 1.93604, "FeKb1": 1.75661,
    "CoKa": 1.79026, "CoKa2": 1.79285, "CoKa1": 1.78896, "CoKb1": 1.63079,
    "AgKa": 0.560885, "AgKa2": 0.563813, "AgKa1": 0.559421, "AgKb1": 0.497082,
}
_TWO_THETA_TOL = 1e-5      # reflections closer than this (degrees) are one peak (as pymatgen)
_SCALED_INTENSITY_TOL = 1e-3   # peaks below this fraction of 100 are systematic absences


@lru_cache(maxsize=1)
def _tables() -> dict:
    return json.loads(Path(__file__).with_name("tables").joinpath("scattering.json").read_text(encoding="utf-8"))


def wavelength_of(value) -> float:
    """Wavelength in Angstrom from a number, a numeric string or a name like 'CuKa', 'MoKa1'."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    for name, lam in WAVELENGTHS.items():
        if name.lower() == text.lower().replace("-", "").replace("_", "").replace("alpha", "a").replace("beta", "b"):
            return lam
    raise ValueError(f"unknown wavelength {value!r}; give Angstrom or one of {', '.join(WAVELENGTHS)}")


def elements_of(types, elements=None) -> np.ndarray:
    """Element symbol per particle: the type name itself ('Cu', 'Fe3+'), or ``elements[type]``
    (e.g. ``{"1": "Si", "2": "O"}`` for numbered LAMMPS types)."""
    from .radii import element_of

    elements = {str(k): str(v) for k, v in (elements or {}).items()}
    names, codes = np.unique(np.asarray(types).astype(str), return_inverse=True)
    out = []
    for t in names:
        el = elements.get(t) or element_of(t)
        if el is None:
            raise ValueError(f"particle type {t!r} is not an element; pass elements={{{t!r}: 'Si', ...}}")
        out.append(el)
    return np.asarray(out, dtype=object)[codes].astype(str)


def xray_form_factor(element: str, s2) -> np.ndarray:
    """X-ray atomic form factor at ``s2 = (sin(theta)/lambda)^2`` (1/Angstrom^2)."""
    from ase.data import atomic_numbers

    try:
        coeff = np.asarray(_tables()["xray"][element], float)
    except KeyError:
        raise ValueError(f"no X-ray scattering factors for {element!r}") from None
    s2 = np.asarray(s2, float)
    return atomic_numbers[element] - 41.78214 * s2 * np.sum(
        coeff[:, 0] * np.exp(-coeff[:, 1] * s2[..., None]), axis=-1)


def neutron_length(element: str) -> float:
    """Bound coherent neutron scattering length in fm."""
    try:
        return float(_tables()["neutron"][element])
    except KeyError:
        raise ValueError(f"no neutron scattering length for {element!r}") from None


# --------------------------------------------------------------------------- shared kernel

def _hkl_bounds(cell: Cell, g_max: float) -> np.ndarray:
    """Largest |h|, |k|, |l| of a crystallographic reciprocal vector shorter than ``g_max``."""
    return np.ceil(g_max * np.linalg.norm(cell.matrix, axis=1) + 1e-9).astype(int)


def _amplitudes(frac, codes, weights, bounds, recip, xp, chunk: int = 0):
    """Complex amplitude ``F = sum_j w_t(|g|) exp(2 pi i hkl . r_j)`` on the half grid
    h = 0..H, k = -K..K, l = -L..L, plus ``|g|`` there. ``weights(t, g)`` gives the scattering
    weight of type code ``t`` at reciprocal length ``g`` (numpy, shape of ``g``)."""
    H, K, L = (int(b) for b in bounds)
    h = np.arange(0, H + 1)
    k = np.arange(-K, K + 1)
    l = np.arange(-L, L + 1)
    hkl = np.stack(np.meshgrid(h, k, l, indexing="ij"), -1).reshape(-1, 3).astype(float)
    g = np.linalg.norm(hkl @ recip, axis=1).reshape(len(h), len(k), len(l))
    F = xp.zeros(g.shape, dtype=xp.complex128)
    two_pi_i = 2j * np.pi
    for t in np.unique(codes):
        f = xp.asarray(frac[codes == t])
        ex = xp.exp(two_pi_i * f[:, 0:1] * xp.asarray(h, dtype=float)[None, :])   # (n, H+1)
        ey = xp.exp(two_pi_i * f[:, 1:2] * xp.asarray(k, dtype=float)[None, :])   # (n, 2K+1)
        ez = xp.exp(two_pi_i * f[:, 2:3] * xp.asarray(l, dtype=float)[None, :])   # (n, 2L+1)
        S = xp.empty(g.shape, dtype=xp.complex128)
        for a in range(len(h)):
            S[a] = (ex[:, a:a + 1] * ey).T @ ez
        F += xp.asarray(weights(t, g)) * S
    return F, g, (h, k, l)


def _half_mask(h, k, l):
    """One of each Friedel pair ``hkl``/``-hkl``: h > 0, or h = 0 and k > 0, or h = k = 0 and l > 0."""
    H, Kg, Lg = np.meshgrid(h, k, l, indexing="ij")
    return (H > 0) | ((H == 0) & (Kg > 0)) | ((H == 0) & (Kg == 0) & (Lg > 0))


# --------------------------------------------------------------------------- powder diffraction

def powder_peaks(positions, types, cell: Cell, wavelength="CuKa", radiation: str = "xray",
                 two_theta_range=(0.0, 90.0), debye_waller=None, elements=None, scaled: bool = True,
                 backend=None) -> dict:
    """Powder diffraction reflections of a periodic cell.

    ``radiation``: "xray" (form factors, Lorentz-polarization ``(1 + cos^2 2t) / (sin^2 t cos t)``)
    or "neutron" (coherent lengths, Lorentz ``1 / (sin^2 t cos t)``). ``debye_waller`` maps
    elements to B factors (Angstrom^2, ``exp(-B s^2)``). Reflections at the same 2-theta are one
    peak. Returns ``{'2theta', 'Intensity', 'd', 'h', 'k', 'l', 'Multiplicity'}`` sorted by
    2-theta, with the strongest peak at 100 when ``scaled``; ``h k l`` is one representative of the
    reflections in the peak, in the basis of the given cell (a primitive cell gives primitive
    indices), and ``Multiplicity`` how many there are (``hkl`` and ``-hkl`` counted).
    """
    be = get_backend(backend)
    xp = be.xp
    if not cell.is_valid:
        raise ValueError("powder diffraction needs a periodic cell")
    lam = wavelength_of(wavelength)
    radiation = str(radiation).lower()
    if radiation not in ("xray", "neutron"):
        raise ValueError("radiation must be 'xray' or 'neutron'")
    lo, hi = (0.0, 180.0) if two_theta_range is None else (float(two_theta_range[0]), float(two_theta_range[1]))
    g_min, g_max = (2 * np.sin(np.radians(t / 2)) / lam for t in (lo, min(hi, 180.0)))
    els = elements_of(types, elements)
    names, codes = np.unique(els, return_inverse=True)
    dw = {str(k): float(v) for k, v in (debye_waller or {}).items()}

    def weights(t, g):
        s2 = (g / 2) ** 2
        f = xray_form_factor(names[t], s2) if radiation == "xray" else np.full(g.shape, neutron_length(names[t]))
        return f * np.exp(-dw[names[t]] * s2) if dw.get(names[t]) else f

    recip = np.linalg.inv(cell.matrix).T
    frac = cell.to_fractional(np.asarray(positions, float))
    F, g, (h, k, l) = _amplitudes(frac, codes, weights, _hkl_bounds(cell, g_max), recip, xp)
    F = be.to_numpy(F)
    keep = _half_mask(h, k, l) & (g <= g_max) & (g > 0)
    if g_min > 0:
        keep &= g >= g_min
    idx = np.argwhere(keep)
    hkl = np.stack([h[idx[:, 0]], k[idx[:, 1]], l[idx[:, 2]]], 1)
    gv = g[keep]
    amp = F[keep]
    order = np.lexsort((-hkl[:, 2], -hkl[:, 1], -hkl[:, 0], gv))
    hkl, gv, amp = hkl[order], gv[order], amp[order]
    theta = np.arcsin(np.clip(lam * gv / 2, -1, 1))
    lorentz = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
    if radiation == "xray":
        lorentz = lorentz * (1 + np.cos(2 * theta) ** 2)
    intensity = 2 * (amp * amp.conjugate()).real * lorentz
    tt = np.degrees(2 * theta)
    keys, first, inverse, counts = np.unique(np.round(tt / _TWO_THETA_TOL).astype(np.int64),
                                             return_index=True, return_inverse=True, return_counts=True)
    summed = np.zeros(len(keys))
    np.add.at(summed, inverse, intensity)
    top = summed.max() if len(summed) else 0.0
    ok = summed / top * 100 > _SCALED_INTENSITY_TOL if top > 0 else np.zeros(len(summed), bool)
    first, summed, counts = first[ok], summed[ok], counts[ok]
    if scaled and len(summed):
        summed = summed / summed.max() * 100.0
    return {"2theta": tt[first], "Intensity": summed, "d": 1.0 / gv[first],
            "h": hkl[first, 0], "k": hkl[first, 1], "l": hkl[first, 2], "Multiplicity": 2 * counts}


def powder_profile(peaks: dict, two_theta=None, fwhm: float = 0.1, eta: float = 0.5,
                   scaled: bool = True) -> dict:
    """Continuous pattern: every peak a pseudo-Voigt of ``fwhm`` degrees and Lorentzian
    fraction ``eta`` (0 Gaussian, 1 Lorentzian), area proportional to its intensity. ``two_theta``
    is the grid (default 0.01 degree steps over the peaks' range). Returns
    ``{'2theta', 'Intensity'}``, maximum 100 when ``scaled``."""
    pos = np.asarray(peaks["2theta"], float)
    amp = np.asarray(peaks["Intensity"], float)
    if two_theta is None:
        lo, hi = (pos.min() - 5 * fwhm, pos.max() + 5 * fwhm) if len(pos) else (0.0, 90.0)
        two_theta = np.arange(max(0.0, lo), hi, 0.01)
    x = np.asarray(two_theta, float)
    w = max(float(fwhm), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    y = np.zeros(len(x))
    sigma = w / (2 * np.sqrt(2 * np.log(2)))
    gamma = w / 2
    for start in range(0, len(pos), 256):
        dx = x[:, None] - pos[None, start:start + 256]
        gauss = np.exp(-0.5 * (dx / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        lorentz = gamma / (np.pi * (dx ** 2 + gamma ** 2))
        y += ((1 - eta) * gauss + eta * lorentz) @ amp[start:start + 256]
    if scaled and y.max() > 0:
        y = y / y.max() * 100.0
    return {"2theta": x, "Intensity": y}


# --------------------------------------------------------------------------- S(q)

def _weights_per_type(names, weighting: str):
    """``w(t, q)`` for type codes, and whether it depends on q."""
    weighting = str(weighting).lower()
    if weighting in ("none", "", "unweighted"):
        return (lambda t, q: np.ones(np.shape(q))), False
    if weighting == "neutron":
        b = [neutron_length(n) for n in names]
        return (lambda t, q: np.full(np.shape(q), b[t])), False
    if weighting == "xray":
        return (lambda t, q: xray_form_factor(names[t], (np.asarray(q) / (4 * np.pi)) ** 2)), True
    raise ValueError("weighting must be 'none', 'xray' or 'neutron'")


def structure_factor(positions, cell: Cell, q_max: float = 10.0, bins: int | None = None,
                     q_min: float = 0.0, method: str = "auto", types=None, weighting: str = "none",
                     elements=None, cutoff: float = 0.0, lorch: bool = True, backend=None, **kwargs) -> dict:
    """Static structure factor S(q) (q = 4 pi sin(theta) / lambda, 1/Angstrom) on the centers of
    ``bins`` bins (default 200; ``number_of_bins`` is the old name) between ``q_min`` and ``q_max``.

    ``method``:

    - "direct": ``S(q) = |sum_j exp(i q . r_j)|^2 / N`` on every reciprocal lattice vector of the
      cell (q = 0 left out), averaged in each |q| bin (freud's ``StaticStructureFactorDirect``).
      Exact for the periodic configuration; bins below 2 pi / L are empty (NaN).
    - "rdf": ``S(q) = 1 + 4 pi rho int r^2 (g(r) - 1) sin(qr)/(qr) dr`` up to ``cutoff`` (default: a
      third of the smallest cell width; at most half), with the Lorch window when ``lorch``. Smooth, any q.
    - "debye": ``S(q) = 1 + (2/N) sum_{i<j} sin(q r_ij) / (q r_ij)`` over all pairs (minimum image
      in a periodic cell), from a 0.001 Angstrom pair-distance histogram (freud's
      ``StaticStructureFactorDebye``); for clusters and nanoparticles. O(N^2). In a periodic cell the
      minimum-image cut makes it ring below about 4 pi / L; use "direct" or "rdf" there.
    - "auto": "direct" in a periodic cell, else "debye".

    ``weighting`` "xray" or "neutron" weights atoms by form factor or coherent length (needs
    ``types``) and normalizes in the Faber-Ziman way, ``(|sum w e|^2 / N - (<w^2> - <w>^2)) / <w>^2``,
    so S tends to 1 at high q; "none" gives the number-number S(q). Returns ``{'q', 'S(q)'}``.
    """
    from .kernels import _bins

    bins = _bins(bins, kwargs, "number_of_bins")
    be = get_backend(backend)
    xp = be.xp
    pos = np.asarray(positions, float)
    n = len(pos)
    periodic = cell.is_valid and bool(np.all(cell.pbc))
    method = str(method).lower()
    if method == "auto":
        method = "direct" if periodic else "debye"
    edges = np.linspace(float(q_min), float(q_max), bins + 1)
    q = 0.5 * (edges[1:] + edges[:-1])
    if str(weighting).lower() in ("none", "", "unweighted"):
        names, codes = np.array(["X"]), np.zeros(n, int)
    elif types is None:
        raise ValueError("weighted S(q) needs particle types")
    else:
        names, codes = np.unique(elements_of(types, elements), return_inverse=True)
    w, q_dependent = _weights_per_type(names, weighting)
    conc = np.bincount(codes, minlength=len(names)) / max(n, 1)

    def moments(qv):
        ws = np.stack([w(t, qv) for t in range(len(names))])      # (types, q)
        mean = conc @ ws
        return ws, mean, conc @ ws ** 2

    if method == "direct":
        if not periodic:
            raise ValueError("method='direct' needs a cell periodic in all three directions")
        recip = np.linalg.inv(cell.matrix).T
        frac = cell.to_fractional(pos)
        g_max = float(q_max) / (2 * np.pi)
        F, g, (h, k, l) = _amplitudes(frac, codes, lambda t, gg: w(t, 2 * np.pi * gg), _hkl_bounds(cell, g_max),
                                      recip, xp)
        F = be.to_numpy(F)
        qq = 2 * np.pi * g
        keep = _half_mask(h, k, l) & (qq > 0) & (qq >= edges[0]) & (qq <= edges[-1])
        qv, intensity = qq[keep], (np.abs(F[keep]) ** 2) / max(n, 1)
        _, mean, mean2 = moments(qv)
        sq = (intensity - (mean2 - mean ** 2)) / mean ** 2
        b = np.clip(np.searchsorted(edges, qv, side="right") - 1, 0, len(q) - 1)
        tot = np.bincount(b, weights=sq, minlength=len(q))
        cnt = np.bincount(b, minlength=len(q))
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"q": q, "S(q)": np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)}

    if method == "rdf":
        from .kernels import rdf_from_pairs
        from .neighbors import find_pairs

        if not cell.is_valid:
            raise ValueError("method='rdf' needs a cell (for the density); use method='debye' for clusters")
        # a third of the cell keeps the cell-list neighbor search (half the width is allowed, but brute force)
        rmax = float(cutoff) if cutoff and cutoff > 0 else float(cell.perpendicular_widths().min()) / 3 * 0.999
        nb = max(200, int(rmax / 0.005))
        i, j, _, dist = find_pairs(pos, cell, rmax, be, half=True)
        tab = rdf_from_pairs(dist, n, cell.volume, rmax, nb, be, half=True, i=i, j=j,
                             types=np.asarray(names)[codes] if len(names) > 1 else None)
        r = tab["r"]
        dr = rmax / nb
        rho = n / cell.volume
        window = np.sinc(r / rmax) if lorch else np.ones_like(r)        # sin(pi r/R)/(pi r/R)
        kern = np.sinc(np.outer(q, r) / np.pi) * (4 * np.pi * rho * r ** 2 * window * dr)   # sin(qr)/(qr)
        if len(names) == 1:
            return {"q": q, "S(q)": 1.0 + kern @ (tab["g(r)"] - 1.0)}
        ws, mean, _ = moments(q)
        total = np.zeros(len(q))
        for a in range(len(names)):
            for c in range(a, len(names)):
                part = kern @ (tab[f"g(r) {names[a]}-{names[c]}"] - 1.0)
                total += (1 if a == c else 2) * conc[a] * conc[c] * ws[a] * ws[c] * part
        return {"q": q, "S(q)": 1.0 + total / mean ** 2}

    if method == "debye":
        dr = 0.001
        # pair-distance histogram per type pair, in chunks of rows
        nt = len(names)
        P = xp.asarray(pos)
        c = xp.asarray(codes)
        inv = xp.asarray(np.linalg.inv(cell.matrix)) if periodic else None
        mat = xp.asarray(cell.matrix) if periodic else None
        hists = {}
        chunk = max(1, 4_000_000 // max(n, 1))
        for s in range(0, n, chunk):
            d = P[s:s + chunk, None, :] - P[None, :, :]
            if periodic:
                f = d @ inv
                d = (f - xp.round(f)) @ mat
            dist = xp.sqrt((d ** 2).sum(-1))
            ii = xp.arange(s, min(s + chunk, n))[:, None]
            upper = xp.arange(n)[None, :] > ii
            dist_u = dist[upper]
            bins = (dist_u / dr + 0.5).astype(xp.int64)
            if nt == 1:
                pair = xp.zeros(bins.shape, dtype=xp.int64)
            else:
                ti = xp.broadcast_to(c[s:s + chunk, None], upper.shape)[upper]
                tj = xp.broadcast_to(c[None, :], upper.shape)[upper]
                pair = xp.minimum(ti, tj) * nt + xp.maximum(ti, tj)
            for key in np.unique(be.to_numpy(pair)):
                sel = bins[pair == key]
                hb = be.to_numpy(xp.bincount(sel))
                prev = hists.get(int(key))
                if prev is not None and len(prev) < len(hb):
                    prev = np.pad(prev, (0, len(hb) - len(prev)))
                elif prev is not None:
                    hb = np.pad(hb, (0, len(prev) - len(hb)))
                hists[int(key)] = hb if prev is None else prev + hb
        ws, mean, mean2 = moments(q)
        total = np.zeros(len(q))
        for key, hb in hists.items():
            a, b = divmod(key, nt)
            rb = np.arange(len(hb)) * dr
            nz = hb > 0
            sinc = np.sinc(np.outer(q, rb[nz]) / np.pi)
            total += 2 * ws[a] * ws[b] * (sinc @ hb[nz])
        intensity = (conc @ ws ** 2) + total / max(n, 1)       # (1/N) sum_ij w_i w_j sinc
        return {"q": q, "S(q)": (intensity - (mean2 - mean ** 2)) / mean ** 2}
    raise ValueError("method must be 'auto', 'direct', 'rdf' or 'debye'")


# --------------------------------------------------------------------------- experimental data

def load_pattern(path) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` from a measured pattern: the first two numeric columns of a text file (.xy, .xye,
    .dat, .txt, .csv, .chi; comma, semicolon, tab or space separated; header and comment lines
    skipped)."""
    import os
    import re

    p = os.path.expanduser(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"experimental pattern not found: {p}")
    rows = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [x for x in re.split(r"[,;\s]+", line.strip()) if x]
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError(f"no two-column numeric data in {p}")
    a = np.asarray(rows, float)
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a[:, 0], a[:, 1]


def compare_pattern(x, y_calc, x_exp, y_exp, scale: bool = True):
    """Interpolate a measured pattern onto ``x`` and fit it to ``y_calc``.

    Returns ``(y_exp_on_x, rwp)``: with ``scale`` the measured curve is scaled so its maximum
    inside the calculated range is the calculated maximum; ``rwp`` is the weighted profile R
    factor ``sqrt(sum w (y_o - c y_c - b)^2 / sum w y_o^2)`` with ``w = 1/y_o`` and the weighted
    least-squares scale ``c`` and flat background ``b`` (with ``scale=False``, as for S(q): ``c = 1,
    b = 0``). NaN when the ranges do not overlap."""
    x = np.asarray(x, float)
    yc = np.asarray(y_calc, float)
    inside = (x >= x_exp[0]) & (x <= x_exp[-1])
    ye = np.full(len(x), np.nan)
    ye[inside] = np.interp(x[inside], x_exp, y_exp)
    ok = inside & np.isfinite(yc)
    if scale and ok.any() and np.nanmax(ye[ok]) > 0:
        ye = ye * (np.nanmax(yc[ok]) / np.nanmax(ye[ok]))
    good = ok & (ye > 0)
    if not good.any():
        return ye, float("nan")
    wgt = 1.0 / ye[good]
    yo, ycg = ye[good], yc[good]
    if scale:
        A = np.stack([ycg, np.ones_like(ycg)], 1) * np.sqrt(wgt)[:, None]
        (c, b), *_ = np.linalg.lstsq(A, yo * np.sqrt(wgt), rcond=None)
        fit = c * ycg + b
    else:
        fit = ycg
    rwp = float(np.sqrt(np.sum(wgt * (yo - fit) ** 2) / np.sum(wgt * yo ** 2)))
    return ye, rwp
