"""Crystal analysis: elastic strain calculation, grain segmentation and fcc planar faults.

Elastic strain and grain segmentation are C++ ports of OVITO 3.16 (StructureAnalysis.cpp,
ElasticStrainEngine.cpp, GrainSegmentationEngine.cpp; Copyright 2026 OVITO GmbH, used under
the MIT option of its GPLv3-or-MIT dual license; see ewald/_csrc/LICENSE.OVITO-MIT.txt) and
reproduce OVITO's results. The planar-fault classification follows OVITO's
IdentifyFCCPlanarFaultsModifier (Python-based in OVITO 3.8+). Names, ids, colors and table
layouts match OVITO.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..bonds import Bonds, minimum_images
from ..colors import to_rgb as _to_rgb
from ..pipeline import Modifier
from ._knn import nearest_neighbors
from .structure import (BCC, CUBIC_DIAMOND, FCC, HCP, HEX_DIAMOND, OTHER, PTM_MAX_INPUT_NEIGHBORS, STRUCTURE_COLORS,
                        StructureIdentificationModifier, _native)

# ---------------------------------------------------------------------------
# Elastic strain
# ---------------------------------------------------------------------------

LATTICE_NAMES = ["OTHER", "FCC", "HCP", "BCC", "CUBIC_DIAMOND", "HEX_DIAMOND"]
LATTICE_COLORS = STRUCTURE_COLORS[[OTHER, FCC, HCP, BCC, CUBIC_DIAMOND, HEX_DIAMOND]]
_LATTICE_ALIASES = {"CUBICDIAMOND": "CUBIC_DIAMOND", "HEXAGONALDIAMOND": "HEX_DIAMOND",
                    "HEXAGONAL_DIAMOND": "HEX_DIAMOND", "DIAMOND": "CUBIC_DIAMOND"}


def _lattice_id(value) -> int:
    if isinstance(value, (int, np.integer)):
        lattice = int(value)
    else:
        name = str(value).upper().replace(" ", "_").replace("-", "_")
        name = _LATTICE_ALIASES.get(name.replace("_", ""), _LATTICE_ALIASES.get(name, name))
        if name not in LATTICE_NAMES:
            raise ValueError(f"unknown crystal structure {value!r}; use one of {LATTICE_NAMES[1:]}")
        lattice = LATTICE_NAMES.index(name)
    if not 1 <= lattice <= 5:
        raise ValueError(f"unknown crystal structure {value!r}; use one of {LATTICE_NAMES[1:]}")
    return lattice


def _cell_args(data):
    cell = data.cell
    if cell.is_valid:
        return np.asarray(cell.matrix, float), np.asarray(cell.origin, float), [bool(b) for b in cell.pbc]
    return np.eye(3), np.zeros(3), [False, False, False]


def _check_3d(data, name):
    cell = data.cell
    if cell.is_valid and np.count_nonzero(cell.pbc) < 3 and np.min(cell.perpendicular_widths()) < 1e-6:
        raise ValueError(f"{name} does not support 2D simulation cells")


class ElasticStrain(StructureIdentificationModifier):
    """Atomic-level elastic strain in a crystal (OVITO's Elastic strain calculation).

    Identifies the lattice (``input_crystal_structure``: 'FCC', 'HCP', 'BCC', 'CUBIC_DIAMOND'
    or 'HEX_DIAMOND') with an extended CNA, groups atoms into crystal clusters and compares each
    atom's neighbor vectors with the ideal lattice of constant ``lattice_constant`` (and
    ``axial_ratio`` c/a for the hexagonal lattices). Stacking faults (HCP atoms in FCC, and the
    reverse) are mapped back to their parent crystal.

    Writes 'Structure Type' (OTHER, FCC, HCP, BCC, CUBIC_DIAMOND, HEX_DIAMOND), 'Color',
    'Cluster', 'Volumetric Strain', 'Elastic Strain' (XX, YY, ZZ, XY, XZ, YZ; Euler-Almansi
    in the spatial frame by default, Green-Lagrange with ``push_strain_tensors_forward=False``)
    and optionally 'Elastic Deformation Gradient' (3x3). Non-crystalline atoms get zeros.
    """

    _prefix = "ElasticStrain"
    _num_types = 6
    _default_types = (FCC, HCP, BCC, CUBIC_DIAMOND, HEX_DIAMOND)
    _names = LATTICE_NAMES
    _colors = LATTICE_COLORS
    choice_parameters = {"input_crystal_structure": tuple(LATTICE_NAMES[1:])}

    def __init__(self, input_crystal_structure: str = "FCC", lattice_constant: float = 1.0,
                 axial_ratio: float = float(np.sqrt(8.0 / 3.0)), calculate_strain_tensors: bool = True,
                 calculate_deformation_gradients: bool = False, push_strain_tensors_forward: bool = True,
                 color_by_type: bool = True, only_selected: bool = False, num_threads: int = 0):
        super().__init__(None, color_by_type, only_selected)
        self.input_crystal_structure = (LATTICE_NAMES[input_crystal_structure]
                                        if isinstance(input_crystal_structure, (int, np.integer))
                                        else input_crystal_structure)
        _lattice_id(self.input_crystal_structure)
        self.lattice_constant = lattice_constant
        self.axial_ratio = axial_ratio
        self.calculate_strain_tensors = calculate_strain_tensors
        self.calculate_deformation_gradients = calculate_deformation_gradients
        self.push_strain_tensors_forward = push_strain_tensors_forward
        self.num_threads = num_threads

    def modify(self, data, ctx):
        native = _native()
        _check_3d(data, "ElasticStrain")
        lattice = _lattice_id(self.input_crystal_structure)
        subset = self._subset(data)
        pos = np.ascontiguousarray(data.positions if subset is None else data.positions[subset], float)
        n = len(pos)
        matrix, _, pbc = _cell_args(data)
        k = min({FCC: 12, 2: 12, 3: 14, 4: 16, 5: 16}[lattice] + 1, 16)
        if not any(pbc):
            k = min(k, max(n - 1, 0))
        if n == 0 or k == 0:
            res = {"structure": np.zeros(n, np.int32), "cluster": np.zeros(n, np.int64),
                   "volumetric_strain": np.zeros(n), "strain": np.zeros((n, 6)),
                   "deformation_gradient": np.zeros((n, 3, 3))}
        else:
            index, delta, dist = nearest_neighbors(pos, data.cell, k)
            count = np.isfinite(dist).sum(1).astype(np.int32)
            res = native.elastic_strain(pos, matrix, pbc, index, delta, count, lattice, float(self.lattice_constant),
                                        float(self.axial_ratio), bool(self.push_strain_tensors_forward),
                                        self.num_threads)

        def scatter(arr):
            if subset is None:
                return arr
            out = np.zeros((data.count,) + arr.shape[1:], arr.dtype)
            out[subset] = arr
            return out

        self._finish(data, scatter(res["structure"]))
        data.particles["Cluster"] = scatter(res["cluster"])
        data.particles["Volumetric Strain"] = scatter(res["volumetric_strain"])
        if self.calculate_strain_tensors:
            data.particles["Elastic Strain"] = scatter(res["strain"])
        if self.calculate_deformation_gradients:
            data.particles["Elastic Deformation Gradient"] = scatter(res["deformation_gradient"])


# ---------------------------------------------------------------------------
# Grain segmentation
# ---------------------------------------------------------------------------

_ALGORITHMS = {"auto": 0, "graphclusteringauto": 0, "graphclusteringautomatic": 0,
               "manual": 1, "graphclusteringmanual": 1,
               "mst": 2, "minimumspanningtree": 2}


def _ptm_input(data, who):
    if "Structure Type" not in data:
        raise ValueError(f"{who} requires Polyhedral Template Matching (PTM) output. Insert "
                         "PolyhedralTemplateMatching(output_orientation=True) before it.")
    if "Orientation" not in data or "Correspondences" not in data:
        raise ValueError(f"{who} requires lattice orientations. Turn on output_orientation of the "
                         "PolyhedralTemplateMatching modifier before it.")
    return (np.ascontiguousarray(data["Structure Type"], np.int32),
            np.ascontiguousarray(data["Correspondences"], np.int64))


class GrainSegmentation(Modifier):
    """Decompose a polycrystal into grains from PTM orientations (OVITO's Grain segmentation).

    Needs ``PolyhedralTemplateMatching(output_orientation=True)`` upstream. Neighboring
    crystalline atoms are joined by bonds weighted by their lattice disorientation, and
    clusters are merged in order of increasing distance: ``algorithm='auto'`` (graph
    clustering with an automatically chosen ``log`` merge threshold), 'manual' (graph
    clustering, threshold ``merging_threshold`` on the log merge distance) or 'mst' (minimum
    spanning tree, ``merging_threshold`` in degrees). ``handle_stacking_faults`` merges
    HCP stacking faults and twins into their parent FCC grain (and hexagonal into cubic
    diamond). Grains smaller than ``min_grain_size`` atoms dissolve; ``orphan_adoption``
    attaches the remaining non-crystalline atoms to the nearest grain.

    Writes 'Grain' (0 = no grain), 'Color' (``color_particles``), table 'grains' (Grain
    Identifier, Grain Size, Color, Structure Type, Orientation as x, y, z, w, Center of Mass,
    Radius, Radius of Gyration; largest grain first), tables 'grains-merge' and 'grains-log'
    (the merge sequence), attributes ``GrainSegmentation.grain_count`` and
    ``GrainSegmentation.auto_merge_threshold``, and with ``output_bonds`` the neighbor graph
    as bonds with a 'Disorientation' property (degrees).
    """

    choice_parameters = {"algorithm": ("auto", "manual", "mst")}

    def __init__(self, algorithm: str = "auto", merging_threshold: float = 0.0, min_grain_size: int = 100,
                 handle_stacking_faults: bool = True, orphan_adoption: bool = True, color_particles: bool = True,
                 output_bonds: bool = False):
        self.algorithm = algorithm
        self._algorithm_id()
        self.merging_threshold = merging_threshold
        self.min_grain_size = min_grain_size
        self.handle_stacking_faults = handle_stacking_faults
        self.orphan_adoption = orphan_adoption
        self.color_particles = color_particles
        self.output_bonds = output_bonds

    def _algorithm_id(self) -> int:
        a = self.algorithm
        if isinstance(a, (int, np.integer)) and 0 <= int(a) <= 2:
            return int(a)
        key = str(getattr(a, "name", a)).lower().replace("_", "").replace(" ", "").replace("-", "")
        if key not in _ALGORITHMS:
            raise ValueError(f"unknown grain segmentation algorithm {a!r}; use 'auto', 'manual' or 'mst'")
        return _ALGORITHMS[key]

    def modify(self, data, ctx):
        native = _native()
        _check_3d(data, "GrainSegmentation")
        structure, corr = _ptm_input(data, "GrainSegmentation")
        algorithm = self._algorithm_id()
        pos = np.ascontiguousarray(data.positions, float)
        n = len(pos)
        # OVITO stores PTM orientations in single precision; round the same way to reproduce it.
        orientation = np.ascontiguousarray(np.asarray(data["Orientation"], np.float32), float)
        matrix, origin, pbc = _cell_args(data)
        k = PTM_MAX_INPUT_NEIGHBORS if any(pbc) else min(PTM_MAX_INPUT_NEIGHBORS, max(n - 1, 0))
        if k > 0:
            index, delta, dist = nearest_neighbors(pos, data.cell, k)
        else:
            index, delta, dist = np.full((n, 0), -1), np.zeros((n, 0, 3)), np.zeros((n, 0))
        count = np.isfinite(dist).sum(1).astype(np.int32)
        masses = np.ascontiguousarray(data["Mass"], float) if "Mass" in data else None
        if masses is not None and not np.any(masses):
            masses = None
        res = native.grain_segmentation(pos, matrix, origin, pbc, structure, orientation, corr, index, delta, count,
                                        algorithm, bool(self.handle_stacking_faults), float(self.merging_threshold),
                                        int(self.min_grain_size), bool(self.orphan_adoption), masses,
                                        bool(self.output_bonds))
        grain = res["grain"]
        data.particles["Grain"] = grain
        colors = res["grain_color"].astype(float)
        if self.color_particles:
            palette = np.vstack([[0.8, 0.8, 0.8], colors]) if len(colors) else np.array([[0.8, 0.8, 0.8]])
            from .. import color_rule

            data.particles["Color"] = palette[grain]
            color_rule.record(data, {"kind": "categorical", "source": "the GrainSegmentation step (color_particles=True)",
                                     "property": "Grain", "note": "one color per grain (tables['grains'] 'Color'); "
                                     "gray #cccccc = no grain (Grain 0)"})
        ng = len(res["grain_size"])
        data.tables["grains"] = {"Grain Identifier": np.arange(1, ng + 1), "Grain Size": res["grain_size"],
                                 "Color": colors, "Structure Type": res["grain_structure"],
                                 "Orientation": res["grain_orientation"], "Center of Mass": res["grain_center"],
                                 "Radius": res["grain_radius"], "Radius of Gyration": res["grain_radius_of_gyration"]}
        if algorithm == 2:
            data.tables["grains-merge"] = {"Merge size": res["merge_size"],
                                           "Misorientation (degrees)": res["merge_distance"]}
        else:
            data.tables["grains-merge"] = {"Delta merge size": res["merge_size"],
                                           "Log merge distance": res["merge_distance"]}
            data.tables["grains-log"] = {"Log merge distance": res["log_merge_distance"],
                                         "Log geometric merge size": res["log_merge_size"]}
        if algorithm == 0:
            data.attributes["GrainSegmentation.auto_merge_threshold"] = float(res["suggested_threshold"])
        data.attributes["GrainSegmentation.grain_count"] = int(grain.max()) if n else 0
        if self.output_bonds:
            topo = res["bond_topology"]
            new = Bonds(topo, minimum_images(data, topo[:, 0], topo[:, 1]))
            new["Disorientation"] = res["bond_disorientation"]
            data.bonds = new if data.bonds is None else data.bonds.extend(new)


# ---------------------------------------------------------------------------
# fcc planar faults
# ---------------------------------------------------------------------------

PLANAR_FAULT_NAMES = ["Non-hcp", "Indeterminate hcp-like", "Intrinsic stacking fault", "Coherent twin boundary",
                      "Multilayer stacking fault"]
NONHCP, FAULT_OTHER, ISF, TWIN, MULTI = range(5)


@njit(cache=True)
def _classify_planar_faults(hcp, nbr, is_hcp, is_fcc, layer):
    """OVITO's two-pass classification. nbr: (n_hcp, 12) global indices in PTM template order;
    is_hcp/is_fcc: per-particle flags; layer: -1/0/+1 of each template neighbor."""
    n_hcp = len(hcp)
    hcp_pos = -np.ones(len(is_hcp), np.int64)
    for i in range(n_hcp):
        hcp_pos[hcp[i]] = i
    basal = np.nonzero(layer == 0)[0]
    out_of_plane = np.nonzero(layer != 0)[0]
    kind = np.zeros(n_hcp, np.int32)
    for i in range(n_hcp):
        n_basal = n_pos = n_neg = n_fcc_pos = n_fcc_neg = 0
        for j in range(12):
            nb = nbr[i, j]
            if is_hcp[nb]:
                if layer[j] == 0:
                    n_basal += 1
                else:
                    # Two hcp atoms are stacked (parallel basal planes) unless a common neighbor
                    # lies in the basal planes of both.
                    other = hcp_pos[nb]
                    stacked = True
                    for a in basal:
                        for b in basal:
                            if nbr[i, a] == nbr[other, b]:
                                stacked = False
                    if stacked:
                        if layer[j] == 1:
                            n_pos += 1
                        else:
                            n_neg += 1
            elif layer[j] != 0 and is_fcc[nb]:
                if layer[j] > 0:
                    n_fcc_pos += 1
                else:
                    n_fcc_neg += 1
        if (n_pos != 0) != (n_neg != 0):
            kind[i] = ISF
        elif n_basal >= 1 and n_pos == 0 and n_neg == 0 and n_fcc_pos != 0 and n_fcc_neg != 0:
            kind[i] = TWIN
        elif n_pos != 0 and n_neg != 0:
            kind[i] = MULTI
        else:
            kind[i] = FAULT_OTHER
    # Second pass, in place and in index order like OVITO.
    for i in range(n_hcp):
        if kind[i] == TWIN or kind[i] == FAULT_OTHER:
            n_isf = n_twin = 0
            for j in basal:
                other = hcp_pos[nbr[i, j]]
                if other >= 0:
                    if kind[other] == ISF:
                        n_isf += 1
                    elif kind[other] == TWIN:
                        n_twin += 1
            if n_isf != 0 and n_twin == 0:
                kind[i] = ISF
            elif n_isf == 0 and n_twin != 0:
                kind[i] = TWIN
        elif kind[i] == MULTI:
            for j in out_of_plane:
                other = hcp_pos[nbr[i, j]]
                if other >= 0 and kind[other] == ISF:
                    kind[other] = MULTI
    return kind


class IdentifyFCCPlanarFaults(Modifier):
    """Classify hcp-like atoms in fcc crystals by planar fault (OVITO's Identify fcc planar faults).

    Needs ``PolyhedralTemplateMatching(output_orientation=True)`` upstream (and
    ``output_interatomic_distance=True`` for the area estimate). Writes 'Planar Fault Type':
    0 non-hcp, 1 indeterminate hcp-like, 2 intrinsic stacking fault, 3 coherent twin boundary,
    4 multilayer stacking fault; recolors hcp atoms by fault type, and fills table
    'planar_faults' with the atom count, fraction and estimated area of each type.
    """

    def __init__(self, color_other=(1.0, 0.4, 0.4), color_isf=(0.9, 0.7, 0.2), color_twin=(0.4, 0.4, 1.0),
                 color_multi=(0.0, 1.0, 0.9)):
        self.color_other = color_other
        self.color_isf = color_isf
        self.color_twin = color_twin
        self.color_multi = color_multi

    def modify(self, data, ctx):
        native = _native()
        structure, corr = _ptm_input(data, "IdentifyFCCPlanarFaults")
        hcp = np.flatnonzero(structure == HCP)
        kind = np.zeros(data.count, np.int32)
        if len(hcp):
            pos = np.ascontiguousarray(data.positions, float)
            periodic = data.cell.is_valid and data.cell.pbc.any()
            k = PTM_MAX_INPUT_NEIGHBORS if periodic else min(PTM_MAX_INPUT_NEIGHBORS, max(len(pos) - 1, 0))
            index, delta, dist = nearest_neighbors(pos, data.cell, k)
            count = np.isfinite(dist).sum(1).astype(np.int32)
            nbr, _, template = native.ptm_ordered_neighbors(structure, corr, index, delta, count, 12)
            ideal = native.ptm_template(HCP, int(template[hcp[0]]))
            layer = np.sign(ideal[:, 2]).astype(np.int64)
            sub = nbr[hcp]
            if np.any(sub < 0):
                raise ValueError("an hcp atom has fewer than 12 neighbors")
            kind[hcp] = _classify_planar_faults(hcp.astype(np.int64), sub, structure == HCP, structure == FCC, layer)
        data.particles["Planar Fault Type"] = kind
        from .. import color_rule
        from ..render import default_colors

        rule = color_rule.rule(data)
        color = np.array(default_colors(data), float)
        for t, c, name in ((FAULT_OTHER, self.color_other, "other HCP"), (ISF, self.color_isf, "intrinsic stacking fault"),
                           (TWIN, self.color_twin, "twin boundary"), (MULTI, self.color_multi, "multilayer fault")):
            c = _to_rgb(c)
            color[kind == t] = c
            rule = color_rule.painted(rule, "the IdentifyFCCPlanarFaults step", f"Planar Fault Type == {t} ({name})",
                                      c, np.count_nonzero(kind == t))
        data.particles["Color"] = color
        color_rule.record(data, rule)
        counts = np.bincount(kind, minlength=5)[:5]
        counts[NONHCP] = 0
        table = {"Planar Fault Type": np.arange(5), "Atom Count": counts,
                 "Atom Fraction": counts / max(len(hcp), 1)}
        if "Interatomic Distance" in data:
            d2 = np.asarray(data["Interatomic Distance"], float) ** 2
            area = np.zeros(5)
            area[ISF] = np.sqrt(3) / 4 * d2[kind == ISF].sum()
            area[TWIN] = np.sqrt(3) / 2 * d2[kind == TWIN].sum()
            table["Estimated Area"] = area
        data.tables["planar_faults"] = table
        for t in (FAULT_OTHER, ISF, TWIN, MULTI):
            key = PLANAR_FAULT_NAMES[t].lower().replace(" ", "_").replace("-", "_")
            data.attributes[f"IdentifyFCCPlanarFaults.counts.{key}"] = int(counts[t])
