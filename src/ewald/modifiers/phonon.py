"""Phonon mode animation (OVITO has none; like phonopy --modulation, VESTA, v_sim)."""
from __future__ import annotations

import numpy as np

from ..colors import to_rgb as _to_rgb
from ..pipeline import Modifier


class PhononMode(Modifier):
    """Animate one phonon mode read from phonopy band.yaml/mesh.yaml/qpoints.yaml or a VASP
    IBRION=5-8 OUTCAR: frame ``f`` of ``N`` shows phase ``2 pi f / N`` of the vibration, so
    the trajectory loops over one period.

    ``qpoint`` and ``band`` pick the mode (0-based; ``band=None``: the first optical branch).
    ``amplitude`` is the largest atomic displacement (A). ``supercell`` (na, nb, nc) repeats
    the primitive cell so a q != 0 mode shows its wave; None picks the smallest commensurate
    one (up to ``max_supercell`` per axis). With ``arrows`` each atom gets an arrow,
    ``arrow_width`` A thick, in ``arrow_color``; ``arrow_mode`` says what it shows:
    'displacement' (default) the displacement in this frame (property 'Phonon Displacement',
    the longest ``arrow_length`` A at full amplitude; it shrinks and flips as the atom swings
    back), 'velocity' the velocity in this frame ('Phonon Velocity', A/ps, longest
    ``arrow_length`` A as an atom passes its rest position), or 'fixed' the mode pattern
    ('Phonon Mode', the displacement at phase 0, the same in every frame, as in a still
    figure). 'Phonon Mode' is always set. Sets the attributes 'Phonon.Frequency' (THz),
    'Phonon.Wavenumber' (cm^-1), 'Phonon.QPoint', 'Phonon.Band' and 'Phonon.q'.
    """

    category = "Phonons"
    choice_parameters = {"arrow_mode": ("displacement", "velocity", "fixed")}

    def __init__(self, qpoint: int = 0, band: int | None = None, amplitude: float = 0.3, supercell=None,
                 max_supercell: int = 8, arrows: bool = True, arrow_length: float = 2.5, arrow_width: float = 0.2,
                 arrow_color=(0.85, 0.1, 0.1), arrow_mode: str = "displacement"):
        self.qpoint = qpoint
        self.band = band
        self.amplitude = amplitude
        self.supercell = supercell
        self.max_supercell = max_supercell
        self.arrows = arrows
        self.arrow_length = arrow_length
        self.arrow_width = arrow_width
        self.arrow_color = arrow_color
        self.arrow_mode = arrow_mode

    def modify(self, data, ctx):
        from ..phonon import THZ_TO_CM

        modes = data.attributes.get("Phonon Modes")
        # duck-typed: modes from an older ewald-phonopy plugin are another class with the same methods
        if not (hasattr(modes, "displacements") and hasattr(modes, "frequencies")):
            raise ValueError("no phonon modes in this data: open a phonopy band.yaml/mesh.yaml/qpoints.yaml "
                             "(with eigenvectors) or a VASP OUTCAR of an IBRION=5-8 run")
        band = modes.default_band() if self.band is None else int(self.band)
        q, b = modes.check(int(self.qpoint), band)
        sc = modes.commensurate_supercell(q, int(self.max_supercell)) if self.supercell is None \
            else tuple(int(n) for n in np.broadcast_to(self.supercell, (3,)))
        arrow_mode = str(self.arrow_mode).lower()
        if arrow_mode not in self.choice_parameters["arrow_mode"]:
            raise ValueError(f"arrow_mode must be one of {self.choice_parameters['arrow_mode']}, got {self.arrow_mode!r}")
        cell_data, u = modes.displacements(q, b, sc, float(self.amplitude))
        n = max(ctx.pipeline.num_frames, 1)
        theta = 2 * np.pi * (ctx.frame % n) / n
        f = modes.frequencies[q, b]
        omega = 2 * np.pi * abs(float(f)) or 1.0           # rad/ps; a zero mode keeps d/dtheta
        now = u * np.exp(-1j * theta)
        out = data.copy()
        out.particles = dict(cell_data.particles)
        out.positions = cell_data.positions + now.real
        out.particles["Phonon Mode"] = u.real.copy()
        out.particles["Phonon Displacement"] = now.real.copy()
        out.particles["Phonon Velocity"] = omega * now.imag  # d/dt Re(u e^{-i omega t})
        out.cell = cell_data.cell
        out.bonds = None
        out.attributes.update({"Phonon.Frequency": float(f), "Phonon.Wavenumber": float(f * THZ_TO_CM),
                               "Phonon.QPoint": q, "Phonon.Band": b,
                               "Phonon.q": tuple(float(x) for x in modes.qpoints[q]),
                               "Phonon.Supercell": tuple(int(x) for x in sc)})
        if self.arrows and self.amplitude:
            from .vectors import VectorVis

            name, peak = {"displacement": ("Phonon Displacement", 1.0), "velocity": ("Phonon Velocity", omega),
                          "fixed": ("Phonon Mode", 1.0)}[arrow_mode]
            scaling = float(self.arrow_length) / (abs(float(self.amplitude)) * peak)  # full swing = arrow_length
            style = VectorVis(name, scaling=scaling, width=float(self.arrow_width),
                              color=tuple(_to_rgb(self.arrow_color)), min_length=1e-4).style(out)
            out.vectors = {**out.vectors, name: style}
        return out
