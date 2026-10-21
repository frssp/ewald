"""Structure identification: common neighbor analysis (CNA) and polyhedral template matching (PTM).

The algorithms follow OVITO 3.16 (CommonNeighborAnalysisModifier.cpp,
PolyhedralTemplateMatchingModifier.cpp; Copyright 2026 OVITO GmbH, used under
the MIT option of its GPLv3-or-MIT dual license; see ewald/_csrc/LICENSE.OVITO-MIT.txt).
Output names, structure type ids, colors and attribute names match OVITO so scripts and
results carry over.
"""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from ._knn import nearest_neighbors

# OVITO structure type ids (StructureIdentificationModifier / PTMAlgorithm).
OTHER, FCC, HCP, BCC, ICO, SC, CUBIC_DIAMOND, HEX_DIAMOND, GRAPHENE = range(9)
STRUCTURE_NAMES = ["OTHER", "FCC", "HCP", "BCC", "ICO", "SC", "CUBIC_DIAMOND", "HEX_DIAMOND", "GRAPHENE"]
STRUCTURE_COLORS = np.array([
    (0.95, 0.95, 0.95), (0.4, 1.0, 0.4), (1.0, 0.4, 0.4), (0.4, 0.4, 1.0), (0.95, 0.8, 0.2),
    (0.627451, 0.078431, 0.996078), (0.074510, 0.627451, 0.996078), (0.996078, 0.537255, 0.0),
    (0.627451, 0.470588, 0.996078)])
ORDERING_NAMES = ["NONE", "PURE", "L10", "L12_A", "L12_B", "B2", "ZINCBLENDE_WURTZITE", "BORON_NITRIDE"]


def _type_ids(names, table=STRUCTURE_NAMES) -> set[int]:
    ids = set()
    for n in names:
        if isinstance(n, (int, np.integer)):
            ids.add(int(n))
        else:
            ids.add(list(table).index(str(n).upper().replace(" ", "_")))
    return ids


class StructureIdentificationModifier(Modifier):
    """Common output handling: 'Structure Type', optional 'Color', count attributes.

    ``only_selected`` restricts the analysis to selected particles (the others
    are ignored as neighbors and get type OTHER), like OVITO.
    """

    _prefix = "StructureIdentification"
    _num_types = 5
    _default_types = (FCC, HCP, BCC, ICO)
    _names = STRUCTURE_NAMES
    _colors = STRUCTURE_COLORS

    def __init__(self, types=None, color_by_type: bool = True, only_selected: bool = False):
        self.types = tuple(self._default_types if types is None else sorted(_type_ids(types, self._names)))
        self.color_by_type = color_by_type
        self.only_selected = only_selected

    def _subset(self, data):
        if self.only_selected and "Selection" in data:
            return np.nonzero(data.selection)[0]
        return None

    def _finish(self, data, structure):
        data.particles["Structure Type"] = structure.astype(np.int32)
        counts = np.bincount(structure, minlength=self._num_types)
        for t in range(self._num_types):
            data.attributes[f"{self._prefix}.counts.{self._names[t]}"] = int(counts[t])
        data.tables["structures"] = {"Structure Type": np.arange(self._num_types),
                                     "Count": counts[: self._num_types]}
        if self.color_by_type:
            from .. import color_rule

            data.particles["Color"] = self._colors[structure]
            color_rule.record(data, color_rule.categorical(
                f"the {type(self).__name__} step (color_by_type=True)", "Structure Type",
                self._names[: self._num_types], self._colors[: self._num_types]))


# ---------------------------------------------------------------------------
# Common neighbor analysis
# ---------------------------------------------------------------------------

def _cna_signatures(xp, adjacency):
    """CNA indices for every (particle, neighbor) pair.

    ``adjacency`` is a (M, n, n) boolean bond matrix among the n nearest
    neighbors of M particles.  Returns (M, n) arrays: number of common neighbors,
    number of bonds among them, and the size (in bonds) of the largest connected
    bond cluster (OVITO's calcMaxChainLength).
    """
    m, n, _ = adjacency.shape
    common = adjacency                                         # [m, ni, c]: c is common to center and ni
    ncommon = common.sum(-1)
    bonds = common[:, :, :, None] & common[:, :, None, :] & adjacency[:, None, :, :]   # [m, ni, c1, c2]
    degree = bonds.sum(-1)                                     # [m, ni, c]
    nbonds = degree.sum(-1) // 2
    # Largest connected bond cluster: transitive closure of each common-neighbor
    # bond graph by repeated boolean matrix squaring, then bonds per component.
    reach = (bonds | xp.eye(n, dtype=bool)[None, None]).astype(xp.float32)
    for _ in range(int(np.ceil(np.log2(max(n, 2))))):
        reach = xp.minimum(reach @ reach, 1.0)
    cluster_bonds = (reach @ degree[..., None].astype(xp.float32))[..., 0]   # 2 x bonds in c's cluster
    maxchain = (cluster_bonds.max(-1) / 2).astype(xp.int64)
    return ncommon, nbonds, maxchain


def _classify_small(xp, ncommon, nbonds, maxchain, enabled):
    s421 = ((ncommon == 4) & (nbonds == 2) & (maxchain == 1)).sum(-1)
    s422 = ((ncommon == 4) & (nbonds == 2) & (maxchain == 2)).sum(-1)
    s555 = ((ncommon == 5) & (nbonds == 5) & (maxchain == 5)).sum(-1)
    out = xp.zeros(len(s421), dtype=xp.int32)
    if FCC in enabled:
        out = xp.where(s421 == 12, FCC, out)
    if HCP in enabled:
        out = xp.where((s421 == 6) & (s422 == 6), HCP, out)
    if ICO in enabled:
        out = xp.where(s555 == 12, ICO, out)
    return out


def _classify_large(xp, ncommon, nbonds, maxchain):
    s444 = ((ncommon == 4) & (nbonds == 4) & (maxchain == 4)).sum(-1)
    s666 = ((ncommon == 6) & (nbonds == 6) & (maxchain == 6)).sum(-1)
    return xp.where((s444 == 6) & (s666 == 8), BCC, OTHER).astype(xp.int32)


def _bond_matrix(xp, delta, cutoff_sq):
    d = delta[:, :, None, :] - delta[:, None, :, :]
    a = (d * d).sum(-1) <= cutoff_sq[:, None, None]
    n = delta.shape[1]
    return a & ~xp.eye(n, dtype=bool)[None]


class CommonNeighborAnalysis(StructureIdentificationModifier):
    """Common neighbor analysis (Honeycutt-Andersen; adaptive a-CNA of Stukowski 2012).

    ``mode``: ``"adaptive"`` (default, per-atom cutoff from the 12 or 14 nearest
    neighbors), ``"interval"`` (interval CNA of Larsen 2020, no parameters) or
    ``"fixed"`` (global ``cutoff``).  Identifies FCC, HCP, BCC and ICO.  Writes
    'Structure Type' (0 OTHER, 1 FCC, 2 HCP, 3 BCC, 4 ICO), 'Color', and
    attributes ``CommonNeighborAnalysis.counts.<TYPE>``.

    On the CPU the signatures are computed by the compiled kernel (a direct
    port of OVITO's code); with ``compute(device="cuda")`` the adaptive and
    fixed modes run as vectorized CuPy kernels instead.
    """

    _prefix = "CommonNeighborAnalysis"
    _num_types = 5
    _default_types = (FCC, HCP, BCC, ICO)

    def __init__(self, mode: str = "adaptive", cutoff: float = 3.2, types=None,
                 color_by_type: bool = True, only_selected: bool = False, chunk_size: int = 32768):
        super().__init__(types, color_by_type, only_selected)
        if mode not in ("adaptive", "interval", "fixed"):
            raise ValueError("mode must be 'adaptive', 'interval' or 'fixed'")
        self.mode = mode
        self.cutoff = cutoff
        self.chunk_size = chunk_size

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        subset = self._subset(data)
        pos = data.positions if subset is None else data.positions[subset]
        enabled = set(self.types)
        want_small = bool(enabled & {FCC, HCP, ICO})
        want_bcc = BCC in enabled
        k = 15 if self.mode == "fixed" else (14 if want_bcc else 12)
        result = np.zeros(len(pos), dtype=np.int32)
        native = None
        if be.name == "cpu" or self.mode == "interval":
            try:
                from .. import _native as native
            except ImportError as exc:
                if self.mode == "interval":
                    raise ImportError(NATIVE_MISSING) from exc
        if len(pos) > k or (len(pos) and data.cell.is_valid and data.cell.pbc.any()):
            _, delta, dist = nearest_neighbors(pos, data.cell, k)
            if native is not None:
                mask = sum(1 << t for t in enabled)
                mode = {"adaptive": 0, "interval": 1, "fixed": 2}[self.mode]
                count = np.isfinite(dist).sum(1).astype(np.int32)
                result = native.cna_identify(delta, count, mode, self.cutoff, mask, 0)
            for s in range(0, len(pos) if native is None else 0, self.chunk_size):
                e = min(len(pos), s + self.chunk_size)
                result[s:e] = self._chunk(be, xp, delta[s:e], dist[s:e], want_small, want_bcc, enabled)
        structure = np.zeros(data.count, dtype=np.int32)
        if subset is None:
            structure = result
        else:
            structure[subset] = result
        self._finish(data, structure)

    def _chunk(self, be, xp, delta, dist, want_small, want_bcc, enabled):
        delta = xp.asarray(delta)
        dist = xp.asarray(dist)
        m = len(dist)
        out = xp.zeros(m, dtype=xp.int32)
        if self.mode == "fixed":
            rc = self.cutoff
            nwithin = (dist <= rc).sum(-1)
            cut = xp.full(m, rc * rc)
            if want_small:
                sel = nwithin == 12
                if bool(sel.any()):
                    sig = _cna_signatures(xp, _bond_matrix(xp, delta[sel, :12], cut[sel]))
                    out[sel] = _classify_small(xp, *sig, enabled)
            if want_bcc:
                sel = nwithin == 14
                if bool(sel.any()):
                    sig = _cna_signatures(xp, _bond_matrix(xp, delta[sel, :14], cut[sel]))
                    out[sel] = _classify_large(xp, *sig)
            return be.to_numpy(out)

        finite = xp.isfinite(dist)
        if want_small:
            sel = finite[:, 11]
            idx = xp.nonzero(sel)[0]
            if idx.size:
                rc = dist[idx, :12].mean(-1) * (1.0 + np.sqrt(2.0)) * 0.5
                sig = _cna_signatures(xp, _bond_matrix(xp, delta[idx, :12], rc * rc))
                out[idx] = _classify_small(xp, *sig, enabled)
        if want_bcc:
            idx = xp.nonzero((out == OTHER) & finite[:, 13])[0]
            if idx.size:
                d = dist[idx, :14]
                rc = (d[:, :8].sum(-1) / np.sqrt(0.75) + d[:, 8:].sum(-1)) / 14 * 1.207
                sig = _cna_signatures(xp, _bond_matrix(xp, delta[idx, :14], rc * rc))
                out[idx] = _classify_large(xp, *sig)
        return be.to_numpy(out)


# ---------------------------------------------------------------------------
# Polyhedral template matching
# ---------------------------------------------------------------------------

_PTM_FLAGS = {FCC: 1 << 0, HCP: 1 << 1, BCC: 1 << 2, ICO: 1 << 3, SC: 1 << 4,
              CUBIC_DIAMOND: 1 << 5, HEX_DIAMOND: 1 << 6, GRAPHENE: 1 << 7}
PTM_MAX_INPUT_NEIGHBORS = 18


NATIVE_MISSING = (
    "this modifier needs the C++ extension ewald._native, which was not built because no C++ compiler "
    "was found at install time. Install one (Windows: Microsoft C++ Build Tools, workload 'Desktop "
    "development with C++'; macOS: xcode-select --install; Linux: g++), then reinstall ewald "
    "(pip install -e . from the source folder). See 'Without a C++ compiler' in docs/user/install.md.")


def _native():
    try:
        from .. import _native
    except ImportError as exc:  # pragma: no cover - depends on the build
        raise ImportError(NATIVE_MISSING) from exc
    return _native


class PolyhedralTemplateMatching(StructureIdentificationModifier):
    """Polyhedral template matching (Larsen, Schmidt, Schiotz 2016), using the PTM library.

    ``types`` defaults to FCC, HCP, BCC (OVITO's default); add 'ICO', 'SC',
    'CUBIC_DIAMOND', 'HEX_DIAMOND', 'GRAPHENE' as needed.  ``rmsd_cutoff`` = 0
    disables RMSD filtering (OVITO default 0.1).

    Writes 'Structure Type', 'RMSD', 'Color' and optionally 'Interatomic Distance',
    'Orientation' (quaternion x, y, z, w), 'Elastic Deformation Gradient' (3x3),
    'Correspondences', and 'Ordering Type' (needs 'Particle Type'), plus
    attributes ``PolyhedralTemplateMatching.counts.<TYPE>`` and table 'ptm-rmsd'.
    """

    _prefix = "PolyhedralTemplateMatching"
    _num_types = 9
    _default_types = (FCC, HCP, BCC)

    def __init__(self, types=None, rmsd_cutoff: float = 0.1, output_rmsd: bool = True,
                 output_interatomic_distance: bool = False, output_orientation: bool = False,
                 output_deformation_gradient: bool = False, output_ordering: bool = False,
                 color_by_type: bool = True, only_selected: bool = False, num_threads: int = 0):
        super().__init__(types, color_by_type, only_selected)
        self.rmsd_cutoff = rmsd_cutoff
        self.output_rmsd = output_rmsd
        self.output_interatomic_distance = output_interatomic_distance
        self.output_orientation = output_orientation
        self.output_deformation_gradient = output_deformation_gradient
        self.output_ordering = output_ordering
        self.num_threads = num_threads

    def modify(self, data, ctx):
        native = _native()
        if data.cell.is_valid and np.count_nonzero(data.cell.pbc) < 3 and np.min(
                data.cell.perpendicular_widths()) < 1e-6:
            raise ValueError("PTM does not support 2D cells")
        subset = self._subset(data)
        pos = data.positions if subset is None else data.positions[subset]
        n = len(pos)
        periodic = data.cell.is_valid and data.cell.pbc.any()
        k = PTM_MAX_INPUT_NEIGHBORS if periodic else min(PTM_MAX_INPUT_NEIGHBORS, max(n - 1, 0))
        flags = 0
        for t in self.types:
            flags |= _PTM_FLAGS.get(t, 0)
        if n == 0 or k == 0:
            res = {key: np.zeros((n,) + shp) for key, shp in
                   (("structure", ()), ("rmsd", ()), ("interatomic_distance", ()), ("orientation", (4,)),
                    ("ordering", ()), ("deformation_gradient", (3, 3)), ("correspondences", ()))}
        else:
            index, delta, dist = nearest_neighbors(pos, data.cell, k)
            count = np.isfinite(dist).sum(1).astype(np.int32)
            types = None
            if self.output_ordering:
                types = np.unique(data.types if subset is None else data.types[subset],
                                  return_inverse=True)[1].astype(np.int32) + 1
            res = native.ptm_identify(index, delta, count, types, flags,
                                      self.output_deformation_gradient, self.num_threads)
        structure = res["structure"].astype(np.int32)
        rmsd = res["rmsd"]
        # RMSD histogram before thresholding, as OVITO does.
        nbins = 100
        rmax = rmsd.max() if n else 0.0
        width = 1.01 * rmax / nbins if rmax > 0 else 1.0
        valid = structure != OTHER
        hist = np.bincount(np.minimum((rmsd[valid] / width).astype(int), nbins), minlength=nbins + 1)[:nbins]
        data.tables["ptm-rmsd"] = {"RMSD": (np.arange(nbins) + 0.5) * width, "Count": hist}
        if self.rmsd_cutoff > 0:
            structure = np.where(rmsd > self.rmsd_cutoff, OTHER, structure)

        def scatter(arr, fill=0):
            if subset is None:
                return arr
            out = np.full((data.count,) + arr.shape[1:], fill, dtype=arr.dtype)
            out[subset] = arr
            return out

        self._finish(data, scatter(structure))
        if self.output_rmsd:
            data.particles["RMSD"] = scatter(rmsd)
        if self.output_interatomic_distance:
            data.particles["Interatomic Distance"] = scatter(res["interatomic_distance"])
        if self.output_orientation:
            data.particles["Orientation"] = scatter(res["orientation"])
            data.particles["Correspondences"] = scatter(res["correspondences"])
        if self.output_deformation_gradient:
            data.particles["Elastic Deformation Gradient"] = scatter(res["deformation_gradient"])
        if self.output_ordering:
            ordering = scatter(res["ordering"].astype(np.int32))
            data.particles["Ordering Type"] = ordering
            oc = np.bincount(ordering, minlength=len(ORDERING_NAMES))
            for i, name in enumerate(ORDERING_NAMES):
                data.attributes[f"PolyhedralTemplateMatching.ordering_counts.{name}"] = int(oc[i])


# ---------------------------------------------------------------------------
# Diamond structure identification
# ---------------------------------------------------------------------------

DIAMOND_NAMES = ["OTHER", "CUBIC_DIAMOND", "CUBIC_DIAMOND_FIRST_NEIGHBOR", "CUBIC_DIAMOND_SECOND_NEIGHBOR",
                 "HEX_DIAMOND", "HEX_DIAMOND_FIRST_NEIGHBOR", "HEX_DIAMOND_SECOND_NEIGHBOR"]
DIAMOND_COLORS = np.array([(0.95, 0.95, 0.95), (0.074510, 0.627451, 0.996078), (0.0, 0.996078, 0.960784),
                           (0.494118, 0.996078, 0.709804), (0.996078, 0.537255, 0.0),
                           (0.996078, 0.862745, 0.0), (0.8, 0.898039, 0.317647)])


class IdentifyDiamond(StructureIdentificationModifier):
    """Cubic and hexagonal diamond lattice identification (Maras et al. 2016).

    A CNA on the 12 second-nearest neighbors (reached through the 4 nearest
    neighbors).  Also labels the first and second neighbors of diamond atoms,
    which mark surfaces and defects.  Structure type ids follow OVITO: 0 OTHER,
    1-3 cubic diamond (atom, 1st neighbor, 2nd neighbor), 4-6 hexagonal diamond.
    ``color_by_type`` (default) also writes OVITO's 'Color' per type: OTHER white, cubic
    diamond blue / cyan / green, hexagonal diamond orange / yellow / lime (the view's legend
    and ``ewald.color_rule.describe(data)`` list them).
    """

    _prefix = "IdentifyDiamond"
    _num_types = 7
    _default_types = tuple(range(1, 7))
    _names = DIAMOND_NAMES
    _colors = DIAMOND_COLORS

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        subset = self._subset(data)
        pos = data.positions if subset is None else data.positions[subset]
        n = len(pos)
        result = np.zeros(n, dtype=np.int32)
        if n:
            index, delta, _ = nearest_neighbors(pos, data.cell, 4)
            result = self._identify(be, xp, index, delta)
        structure = result if subset is None else np.zeros(data.count, np.int32)
        if subset is not None:
            structure[subset] = result
        self._finish(data, structure)

    def _identify(self, be, xp, index, delta):
        n = len(index)
        ok = (index >= 0).all(1)
        safe = np.where(index >= 0, index, 0)
        second = delta[:, :, None, :] + delta[safe]                      # (n, 4, 4, 3)
        nonzero = ~np.all(np.abs(second) <= 1e-2, axis=-1)                # OVITO's isZero(1e-2)
        ok &= (nonzero.sum(2) == 3).all(1)
        cand = np.nonzero(ok)[0]
        out = np.zeros(n, dtype=np.int32)
        if len(cand):
            sec = second[cand][nonzero[cand]].reshape(len(cand), 12, 3)
            v = xp.asarray(sec)
            rc = xp.sqrt((v * v).sum(-1)).mean(-1) * 1.2071068
            ncommon, nbonds, chain = _cna_signatures(xp, _bond_matrix(xp, v, rc * rc))
            sig_ok = ((ncommon == 4) & (nbonds == 2) & ((chain == 1) | (chain == 2))).all(1)
            n421 = (chain == 1).sum(1)
            t = xp.where(sig_ok & (n421 == 12), 1, xp.where(sig_ok & (n421 == 6), 4, 0))
            out[cand] = be.to_numpy(t).astype(np.int32)
        enabled = set(self.types)
        out[(out == 1) & (1 not in enabled)] = 0
        out[(out == 4) & (4 not in enabled)] = 0
        # Mark neighbors in particle order, as OVITO's sequential loops do.
        for src_types, offset in (((1, 4), 1), ((2, 5), 1)):
            src = np.nonzero(np.isin(out, src_types))[0]
            if not len(src):
                continue
            nb = index[src]
            label = np.repeat(out[src] + offset, 4)
            nb = nb.ravel()
            keep = nb >= 0
            nb, label = nb[keep], label[keep]
            first = np.unique(nb, return_index=True)
            targets, pos = first
            assign = out[targets] == 0
            out[targets[assign]] = label[pos[assign]]
        return out
