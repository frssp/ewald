"""Wigner-Seitz defect analysis (port of OVITO's WignerSeitzAnalysisModifier,
Copyright 2026 OVITO GmbH, used under the MIT option of its GPLv3-or-MIT license)."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from ._knn import PeriodicKDTree


class WignerSeitzAnalysis(Modifier):
    """Count atoms in the Wigner-Seitz cell of every site of a reference frame.

    Each atom is assigned to its closest reference site (minimum image in the
    reference cell).  Sites with occupancy 0 are vacancies; occupancy n > 1
    contributes n - 1 interstitials.

    By default the output is the *reference* configuration (sites) with an
    'Occupancy' property, as in OVITO.  ``output_current_config=True`` keeps the
    current atoms instead and gives each the 'Occupancy' of its site plus
    'Site Type', 'Site Index' and 'Site Identifier' (if the reference has ids).
    ``per_type_occupancies`` makes 'Occupancy' an (N, n_types) array, one
    column per particle type (sorted type names, listed in attribute
    'WignerSeitz.occupancy_types').  ``affine_mapping="to_reference"`` maps
    current positions into the reference cell first (for a deformed box).

    Attributes: 'WignerSeitz.vacancy_count', 'WignerSeitz.interstitial_count'.
    """

    def __init__(self, reference_frame: int = 0, output_current_config: bool = False,
                 per_type_occupancies: bool = False, affine_mapping: str | None = None,
                 frame_offset: int | None = None):
        if affine_mapping not in (None, "off", "to_reference"):
            raise ValueError("affine_mapping must be None/'off' or 'to_reference'")
        self.reference_frame = reference_frame
        self.frame_offset = frame_offset
        self.output_current_config = output_current_config
        self.per_type_occupancies = per_type_occupancies
        self.affine_mapping = affine_mapping

    def modify(self, data, ctx):
        frame = ctx.frame + self.frame_offset if self.frame_offset is not None else self.reference_frame
        ref = ctx.source_frame(frame)
        if ref.count == 0:
            raise ValueError("reference configuration for Wigner-Seitz analysis contains no sites")
        pos = data.positions
        if self.affine_mapping == "to_reference" and data.cell.is_valid and ref.cell.is_valid:
            pos = ref.cell.to_cartesian(data.cell.to_fractional(pos))
        site, _, _ = PeriodicKDTree(ref.positions, ref.cell).query(pos, 1)
        site = site[:, 0]

        if self.per_type_occupancies:
            names = sorted(set(data.types))
            col = np.searchsorted(np.array(names), data.types)
            occ = np.zeros((ref.count, len(names)), dtype=np.int32)
            np.add.at(occ, (site, col), 1)
            total = occ.sum(1)
        else:
            names = None
            occ = np.bincount(site, minlength=ref.count).astype(np.int32)
            total = occ
        vacancies = int(np.count_nonzero(total == 0))
        interstitials = int(np.sum(np.maximum(total - 1, 0)))

        if self.output_current_config:
            out = data
            out.particles["Occupancy"] = occ[site]
            out.particles["Site Type"] = ref.types[site]
            out.particles["Site Index"] = site.astype(np.int64)
            if "Particle Identifier" in ref:
                out.particles["Site Identifier"] = ref["Particle Identifier"][site]
        else:
            out = ref
            out.attributes.update({k: v for k, v in data.attributes.items() if k not in out.attributes})
            out.particles["Occupancy"] = occ
        out.attributes["WignerSeitz.vacancy_count"] = vacancies
        out.attributes["WignerSeitz.interstitial_count"] = interstitials
        if names is not None:
            out.attributes["WignerSeitz.occupancy_types"] = names
        return out
