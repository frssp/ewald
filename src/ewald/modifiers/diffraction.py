"""Powder diffraction patterns and static structure factors (see :mod:`ewald.diffraction`)."""
from __future__ import annotations

import os

import numpy as np

from .. import diffraction as _d
from ..pipeline import Modifier


def _stamp(path) -> tuple:
    try:
        st = os.stat(os.path.expanduser(str(path)))
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return ()


class _ExperimentalMixin:
    """Reads ``self.experimental`` once per file change."""

    def _experiment(self):
        path = os.path.expanduser(str(self.experimental))
        key = (path, _stamp(path))
        if self.__dict__.get("_exp_cache", (None,))[0] != key:
            self.__dict__["_exp_cache"] = (key, _d.load_pattern(path))
        return self.__dict__["_exp_cache"][1]


def _as_dict(value, what):
    if value in (None, "", {}):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):   # "Cu=0.5, O=0.8" or "1=Si 2=O"
        out = {}
        for item in value.replace(";", ",").replace(" ", ",").split(","):
            if item:
                k, _, v = item.partition("=")
                out[k.strip()] = v.strip()
        return out
    raise ValueError(f"{what} must be a dict like {{'Cu': ...}}")


class PowderDiffraction(_ExperimentalMixin, Modifier):
    """Powder X-ray or neutron diffraction pattern of the (periodic) cell.

    Every reciprocal lattice vector in the 2-theta range contributes, with atomic form factors
    (X-ray, Lorentz-polarization factor) or coherent scattering lengths (neutron, Lorentz
    factor), as pymatgen's XRDCalculator/NDCalculator. A supercell or MD snapshot gives the
    pattern of that configuration; ``TimeAveraging('table:diffraction')`` averages a trajectory.
    ``wavelength`` is in Angstrom or a name ('CuKa', 'CuKa1', 'MoKa', 'CoKa', 'AgKa', ...).
    The peaks are broadened by a pseudo-Voigt of ``fwhm`` degrees and Lorentzian fraction
    ``eta`` on a ``two_theta_step`` grid (degrees). ``debye_waller`` maps elements to B factors (Angstrom^2);
    ``elements`` maps type names to elements for numbered types (``{'1': 'Si', '2': 'O'}``).

    Writes ``tables['diffraction']`` ('2theta', 'Intensity', max 100) and
    ``tables['diffraction-peaks']`` ('2theta', 'Intensity', 'd', 'h', 'k', 'l', 'Multiplicity'),
    and the attribute ``PowderDiffraction.num_peaks``. ``experimental`` is a measured pattern
    (two-column text: 2-theta, intensity); it is added as the 'Experimental' column, scaled to the
    calculated maximum, and ``PowderDiffraction.Rwp`` compares the two.
    """

    choice_parameters = {"radiation": ("xray", "neutron")}
    file_parameters = ("experimental",)

    ALIASES = {"step": "two_theta_step"}

    def __init__(self, radiation: str = "xray", wavelength: str = "CuKa", two_theta_min: float = 5.0,
                 two_theta_max: float = 90.0, two_theta_step: float = 0.02, fwhm: float = 0.1, eta: float = 0.5,
                 debye_waller: dict | None = None, elements: dict | None = None, experimental: str = ""):
        self.radiation = radiation
        self.wavelength = wavelength
        self.two_theta_min = two_theta_min
        self.two_theta_max = two_theta_max
        self.two_theta_step = two_theta_step
        self.fwhm = fwhm
        self.eta = eta
        self.debye_waller = debye_waller
        self.elements = elements
        self.experimental = experimental

    def modify(self, data, ctx):
        lo, hi = sorted((float(self.two_theta_min), float(self.two_theta_max)))
        peaks = _d.powder_peaks(data.positions, data.types, data.cell, self.wavelength, self.radiation, (lo, hi),
                                _as_dict(self.debye_waller, "debye_waller"), _as_dict(self.elements, "elements"),
                                backend=ctx.backend)
        x = np.arange(lo, hi + 0.5 * float(self.two_theta_step), max(float(self.two_theta_step), 1e-4))
        table = _d.powder_profile(peaks, x, self.fwhm, self.eta)
        if self.experimental:
            ye, rwp = _d.compare_pattern(x, table["Intensity"], *self._experiment())
            table["Experimental"] = ye
            data.attributes["PowderDiffraction.Rwp"] = rwp
        data.tables["diffraction"] = table
        data.tables["diffraction-peaks"] = peaks
        data.attributes["PowderDiffraction.num_peaks"] = int(len(peaks["2theta"]))


class StructureFactor(_ExperimentalMixin, Modifier):
    """Static structure factor S(q) (q = 4 pi sin(theta) / lambda in 1/Angstrom), for liquids,
    glasses and MD.

    ``method``: "direct" (sum over the reciprocal lattice of the cell, exact for the periodic
    configuration, empty bins below 2 pi / L), "rdf" (Fourier transform of g(r) up to ``cutoff``,
    default a third of the smallest cell width, at most half, with the Lorch window when ``lorch``), "debye" (Debye
    equation over all pairs, for clusters and nanoparticles) or "auto" (direct in a periodic
    cell, else debye). ``weighting`` "xray" or "neutron" weights the atoms by form factor or
    coherent length with Faber-Ziman normalization, to compare with total-scattering data;
    "none" gives the number-number S(q). ``elements`` maps numbered types to elements.

    Writes ``tables['structure-factor']`` with 'q' and 'S(q)' (bin centers between ``q_min``
    and ``q_max``); ``TimeAveraging('table:structure-factor')`` averages a trajectory.
    ``experimental`` (two-column text: q, S(q)) adds the 'Experimental' column, interpolated
    but not rescaled, and ``StructureFactor.Rwp``.
    """

    choice_parameters = {"method": ("auto", "direct", "rdf", "debye"), "weighting": ("none", "xray", "neutron")}
    file_parameters = ("experimental",)

    ALIASES = {"number_of_bins": "bins"}

    def __init__(self, method: str = "auto", q_max: float = 10.0, q_min: float = 0.0, bins: int = 200,
                 weighting: str = "none", cutoff: float = 0.0, lorch: bool = True, elements: dict | None = None,
                 experimental: str = ""):
        self.method = method
        self.q_max = q_max
        self.q_min = q_min
        self.bins = bins
        self.weighting = weighting
        self.cutoff = cutoff
        self.lorch = lorch
        self.elements = elements
        self.experimental = experimental

    def modify(self, data, ctx):
        table = _d.structure_factor(data.positions, data.cell, self.q_max, self.bins, self.q_min,
                                    self.method, data.types, self.weighting, _as_dict(self.elements, "elements"),
                                    self.cutoff, self.lorch, backend=ctx.backend)
        if self.experimental:
            ye, rwp = _d.compare_pattern(table["q"], table["S(q)"], *self._experiment(), scale=False)
            table["Experimental"] = ye
            data.attributes["StructureFactor.Rwp"] = rwp
        data.tables["structure-factor"] = table
