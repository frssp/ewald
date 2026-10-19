"""Phonon mode animation (OVITO has none; like phonopy --modulation, VESTA, v_sim)."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier


class PhononMode(Modifier):
    """Animate one phonon mode read from phonopy band.yaml/mesh.yaml/qpoints.yaml or a VASP
    IBRION=5-8 OUTCAR: frame ``f`` of ``N`` shows phase ``2 pi f / N`` of the vibration, so
    the trajectory loops over one period.

    ``qpoint`` and ``band`` pick the mode (0-based; ``band=None``: the first optical branch).
    ``amplitude`` is the largest atomic displacement (A). ``supercell`` (na, nb, nc) repeats
    the primitive cell so a q != 0 mode shows its wave; None picks the smallest commensurate
    one (up to ``max_supercell`` per axis). With ``arrows`` the mode pattern (the property
    'Phonon Mode', the displacement at phase 0) is drawn as arrows, the longest
    ``arrow_length`` A, ``arrow_width`` A thick, in ``arrow_color``. Sets the attributes 'Phonon.Frequency' (THz),
    'Phonon.Wavenumber' (cm^-1), 'Phonon.QPoint', 'Phonon.Band' and 'Phonon.q'.
    """

    def __init__(self, qpoint: int = 0, band: int | None = None, amplitude: float = 0.3, supercell=None,
                 max_supercell: int = 8, arrows: bool = True, arrow_length: float = 2.5, arrow_width: float = 0.2,
                 arrow_color=(0.85, 0.1, 0.1)):
        self.qpoint = qpoint
        self.band = band
        self.amplitude = amplitude
        self.supercell = supercell
        self.max_supercell = max_supercell
        self.arrows = arrows
        self.arrow_length = arrow_length
        self.arrow_width = arrow_width
        self.arrow_color = arrow_color

    def modify(self, data, ctx):
        from ..phonon import THZ_TO_CM, PhononModes

        modes = data.attributes.get("Phonon Modes")
        if not isinstance(modes, PhononModes):
            raise ValueError("no phonon modes in this data: open a phonopy band.yaml/mesh.yaml/qpoints.yaml "
                             "(with eigenvectors) or a VASP OUTCAR of an IBRION=5-8 run")
        band = modes.default_band() if self.band is None else int(self.band)
        q, b = modes.check(int(self.qpoint), band)
        sc = modes.commensurate_supercell(q, int(self.max_supercell)) if self.supercell is None \
            else tuple(int(n) for n in np.broadcast_to(self.supercell, (3,)))
        cell_data, u = modes.displacements(q, b, sc, float(self.amplitude))
        n = max(ctx.pipeline.num_frames, 1)
        theta = 2 * np.pi * (ctx.frame % n) / n
        out = data.copy()
        out.particles = dict(cell_data.particles)
        out.positions = cell_data.positions + (u * np.exp(-1j * theta)).real
        out.particles["Phonon Mode"] = u.real.copy()
        out.cell = cell_data.cell
        out.bonds = None
        f = modes.frequencies[q, b]
        out.attributes.update({"Phonon.Frequency": float(f), "Phonon.Wavenumber": float(f * THZ_TO_CM),
                               "Phonon.QPoint": q, "Phonon.Band": b,
                               "Phonon.q": tuple(float(x) for x in modes.qpoints[q]),
                               "Phonon.Supercell": tuple(int(x) for x in sc)})
        if self.arrows and self.amplitude:
            from .vectors import VectorVis

            scaling = float(self.arrow_length) / abs(float(self.amplitude))  # longest pattern vector = amplitude
            style = VectorVis("Phonon Mode", scaling=scaling, width=float(self.arrow_width),
                              color=tuple(self.arrow_color), min_length=1e-4).style(out)
            out.vectors = {**out.vectors, "Phonon Mode": style}
        return out
