"""Centrosymmetry, Ackland-Jones bond-angle analysis, Chill+ and atomic strain.

Ported from OVITO 3.16 (CentroSymmetryModifier.cpp, AcklandJonesModifier.cpp,
ChillPlusModifier.cpp, AtomicStrainModifier.cpp; Copyright 2026 OVITO GmbH, used under the MIT option
of its GPLv3-or-MIT license; see ewald/_csrc/LICENSE.OVITO-MIT.txt).  The array code runs on the pipeline device.
"""
from __future__ import annotations

import numpy as np

from ..neighbors import find_pairs
from ..pipeline import Modifier
from ._knn import nearest_neighbors
from .structure import BCC, FCC, HCP, ICO, OTHER, StructureIdentificationModifier, _native


class CentroSymmetry(Modifier):
    """Centrosymmetry parameter (Kelchner et al. 1998) from ``num_neighbors`` neighbors.

    ``mode="conventional"`` sums the N/2 smallest |r_i + r_j|^2 over all
    neighbor pairs (OVITO's default); ``mode="matching"`` uses the minimum-weight
    perfect matching of Larsen (2020), which is better defined for distorted
    environments.  Writes 'Centrosymmetry' and the histogram table
    'csp-centrosymmetry'.
    """

    def __init__(self, num_neighbors: int = 12, mode: str = "conventional", only_selected: bool = False):
        if num_neighbors < 2 or num_neighbors > 32 or num_neighbors % 2:
            raise ValueError("num_neighbors must be an even number between 2 and 32")
        if mode not in ("conventional", "matching"):
            raise ValueError("mode must be 'conventional' or 'matching'")
        self.num_neighbors = num_neighbors
        self.mode = mode
        self.only_selected = only_selected

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        subset = np.nonzero(data.selection)[0] if self.only_selected and "Selection" in data else None
        pos = data.positions if subset is None else data.positions[subset]
        k = self.num_neighbors
        csp = np.zeros(len(pos))
        if len(pos):
            _, delta, dist = nearest_neighbors(pos, data.cell, k)
            count = np.isfinite(dist).sum(1).astype(np.int32)
            if self.mode == "matching":
                csp = _native().csp_matching(delta, count, 0)
            else:
                d = xp.asarray(np.where(np.isfinite(dist)[..., None], delta, np.nan))
                # |r_i + r_j|^2 of every neighbor pair, one pair (row) at a time over contiguous
                # (k, N) components: gathering an (N, k(k-1)/2, 3) array is ~8x slower
                comp = [xp.ascontiguousarray(d[..., c].T) for c in range(3)]
                iu, ju = np.triu_indices(k, 1)
                pair = xp.empty((len(iu), len(pos)))
                for p, (a, b) in enumerate(zip(iu, ju)):
                    sx, sy, sz = (c[a] + c[b] for c in comp)
                    pair[p] = sx * sx + sy * sy + sz * sz
                pair[xp.isnan(pair)] = np.inf  # missing neighbors sort last
                pair = xp.sort(pair, axis=0)
                ncount = xp.asarray(count)
                take = xp.arange(pair.shape[0])[:, None] < (ncount // 2)[None, :]
                csp = be.to_numpy(xp.where(take, pair, 0.0).sum(0))
        out = np.zeros(data.count)
        if subset is None:
            out = csp
        else:
            out[subset] = csp
        data.particles["Centrosymmetry"] = out
        nbins = 100
        vals = out if subset is None else out[subset]
        width = 1.01 * vals.max() / nbins if len(vals) and vals.max() > 0 else 1.0
        hist = np.bincount(np.minimum((vals / width).astype(int), nbins), minlength=nbins + 1)[:nbins]
        data.tables["csp-centrosymmetry"] = {"CSP": (np.arange(nbins) + 0.5) * width, "Count": hist}


class AcklandJones(StructureIdentificationModifier):
    """Bond-angle structure identification of Ackland and Jones (2006).

    Identifies FCC, HCP, BCC and ICO from the 14 nearest neighbors.  Writes
    'Structure Type', 'Color' and ``AcklandJones.counts.<TYPE>``.
    """

    _prefix = "AcklandJones"
    _num_types = 5
    _default_types = (FCC, HCP, BCC, ICO)

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        subset = self._subset(data)
        pos = data.positions if subset is None else data.positions[subset]
        result = np.zeros(len(pos), dtype=np.int32)
        if len(pos):
            _, delta, dist = nearest_neighbors(pos, data.cell, 14)
            result = be.to_numpy(self._classify(xp, xp.asarray(delta), xp.asarray(dist)))
        structure = result if subset is None else np.zeros(data.count, np.int32)
        if subset is not None:
            structure[subset] = result
        self._finish(data, structure)

    def _classify(self, xp, delta, dist):
        m = len(dist)
        d2 = dist * dist
        valid_n = xp.isfinite(dist)
        enough = valid_n[:, 5]
        r0_sq = xp.where(enough, d2[:, :6].mean(1) if m else xp.zeros(0), 1.0)
        # Neighbors up to the first one beyond 1.45 r0^2 (n0) and 1.55 r0^2 (n1), as in OVITO's loops.
        in0 = valid_n & (d2 <= 1.45 * r0_sq[:, None])
        n0 = xp.cumprod(in0, axis=1).sum(1)
        in1 = valid_n & (d2 < 1.55 * r0_sq[:, None])
        n1 = n0 + xp.cumprod(in1 | (xp.arange(14)[None, :] < n0[:, None]), axis=1).sum(1) - n0
        norm = xp.where(valid_n, dist, 1.0)
        # unit vectors as contiguous (14, M) components; then the bond-angle histogram one
        # neighbor pair at a time: (M, 91, 3) gathers and searchsorted cost several times more
        comp = [xp.ascontiguousarray((xp.where(valid_n, delta[..., c], 0.0) / norm).T) for c in range(3)]
        iu, ju = np.triu_indices(14, 1)
        inner = (-0.945, -0.915, -0.755, -0.195, 0.195, 0.245, 0.795)  # bin edges between -1 and 1
        atoms = xp.arange(m)
        codes = xp.empty((len(iu), m), dtype=xp.int64)  # bin * M + atom; bin 8 = pair not used
        for p, (a, b) in enumerate(zip(iu, ju)):
            cos = comp[0][a] * comp[0][b] + comp[1][a] * comp[1][b] + comp[2][a] * comp[2][b]
            k = xp.zeros(m, dtype=xp.int64)
            for e in inner:
                k += cos >= e
            k[b >= n0] = 8  # both j and k must be within the first n0
            codes[p] = k * m + atoms
        chi = xp.bincount(codes.ravel(), minlength=9 * m)[:8 * m].reshape(8, m).T.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_bcc = 0.35 * chi[:, 4] / (chi[:, 5] + chi[:, 6] - chi[:, 4])
        delta_cp = xp.abs(1 - chi[:, 6] / 24)
        delta_fcc = 0.61 * (xp.abs(chi[:, 0] + chi[:, 1] - 6) + chi[:, 2]) / 6
        delta_hcp = (xp.abs(chi[:, 0] - 3) + xp.abs(chi[:, 0] + chi[:, 1] + chi[:, 2] + chi[:, 3] - 9)) / 12
        delta_bcc = xp.where(chi[:, 0] == 7, 0.0, delta_bcc)
        delta_fcc = xp.where((chi[:, 0] != 7) & (chi[:, 0] == 6), 0.0, delta_fcc)
        delta_hcp = xp.where((chi[:, 0] != 7) & (chi[:, 0] != 6) & (chi[:, 0] <= 3), 0.0, delta_hcp)
        en = set(self.types)
        out = xp.full(m, OTHER, dtype=xp.int32)
        decided = ~enough | (chi[:, 7] > 0)
        ico = ~decided & (chi[:, 4] < 3)
        if ICO in en:
            out = xp.where(ico & (n1 >= 11) & (n1 <= 13), ICO, out)
        decided = decided | ico
        bcc = ~decided & (delta_bcc <= delta_cp)
        if BCC in en:
            out = xp.where(bcc & (n1 >= 11), BCC, out)
        decided = decided | bcc | (n1 > 12) | (n1 < 11)
        fcc = ~decided & (delta_fcc < delta_hcp)
        if FCC in en:
            out = xp.where(fcc, FCC, out)
        if HCP in en:
            out = xp.where(~decided & ~fcc, HCP, out)
        return out


CHILL_NAMES = ["OTHER", "HEXAGONAL_ICE", "CUBIC_ICE", "INTERFACIAL_ICE", "HYDRATE", "INTERFACIAL_HYDRATE"]
CHILL_COLORS = np.array([(0.95, 0.95, 0.95), (0.0, 0.9, 0.9), (1.0, 0.756863, 0.019608), (0.5, 0.12, 0.4),
                         (1.0, 0.3, 0.1), (0.1, 1.0, 0.1)])


def _q3m(xp, delta):
    """Unnormalized spherical harmonics Y_3^m (m = 0..3) of bond directions, shape (P, 4).

    Only |Y_3^m| per m matters for the Chill+ correlation, so m < 0 is folded
    into m > 0 (|Y_3^-m| = |Y_3^m|) by the caller.
    """
    r = xp.sqrt((delta * delta).sum(-1))
    x, y, z = (delta / r[:, None]).T
    w = x + 1j * y
    return xp.stack([0.25 * np.sqrt(7 / np.pi) * (5 * z * z - 3) * z + 0j,
                     -0.125 * np.sqrt(21 / np.pi) * w * (5 * z * z - 1),
                     0.25 * np.sqrt(105 / (2 * np.pi)) * w * w * z,
                     -0.125 * np.sqrt(35 / np.pi) * w * w * w], 1)


class ChillPlus(StructureIdentificationModifier):
    """Chill+ identification of water structures (Nguyen and Molinero 2015).

    Apply it to the oxygen atoms only (e.g. after ``SelectType`` with
    ``only_selected=True``, or on an oxygen-only frame).  Bond correlations
    c_ij of the l=3 bond-orientational order parameters of neighbors within
    ``cutoff`` (default 3.5 Å) classify each molecule with four neighbors as
    hexagonal ice, cubic ice, interfacial ice, hydrate or interfacial hydrate.
    Writes 'Structure Type', 'Color' and ``ChillPlus.counts.<TYPE>``.
    """

    _prefix = "ChillPlus"
    _num_types = 6
    _default_types = tuple(range(1, 6))
    _names = CHILL_NAMES
    _colors = CHILL_COLORS

    def __init__(self, cutoff: float = 3.5, color_by_type: bool = True, only_selected: bool = False):
        super().__init__(None, color_by_type, only_selected)
        self.cutoff = cutoff

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        subset = self._subset(data)
        pos = data.positions if subset is None else data.positions[subset]
        result = np.zeros(len(pos), dtype=np.int32)
        if len(pos):
            i, j, delta, _ = find_pairs(pos, data.cell, self.cutoff, be)
            result = be.to_numpy(self._classify(xp, len(pos), i, j, delta))
        structure = result if subset is None else np.zeros(data.count, np.int32)
        if subset is not None:
            structure[subset] = result
        self._finish(data, structure)

    @staticmethod
    def _classify(xp, n, i, j, delta):
        ylm = _q3m(xp, delta)
        qr, qi = xp.zeros((n, 4)), xp.zeros((n, 4))
        _scatter_add(xp, qr, i, ylm.real)
        _scatter_add(xp, qi, i, ylm.imag)
        q = qr + 1j * qi
        # sum over m = -3..3 of q_i(m) conj(q_j(m)); m and -m give conjugate products with equal real part
        weight = xp.asarray([1.0, 2.0, 2.0, 2.0])
        dot = ((q[i] * q[j].conj()).real * weight).sum(1)
        norm = ((q * q.conj()).real * weight).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = dot / (xp.sqrt(norm[i]) * xp.sqrt(norm[j]))
        ecl = xp.bincount(i, weights=((c > -0.35) & (c < 0.25)).astype(float), minlength=n)
        stg = xp.bincount(i, weights=(c < -0.8).astype(float), minlength=n)
        coord = xp.bincount(i, minlength=n)
        four = coord == 4
        out = xp.zeros(n, dtype=xp.int32)
        # Reverse priority order of OVITO's if/else chain.
        out = xp.where(four & (stg == 2), 3, out)
        out = xp.where(four & (stg == 3) & (ecl == 0), 3, out)
        out = xp.where(four & (stg == 3) & (ecl == 1), 1, out)
        out = xp.where(four & (stg == 4), 2, out)
        out = xp.where(four & (ecl == 3), 5, out)
        out = xp.where(four & (ecl == 4), 4, out)
        return out


class AtomicStrain(Modifier):
    """Atomic-level strain from a reference frame (Shimizu, Ogata, Li 2007; Falk-Langer D2min).

    For every particle the deformation gradient F is fitted to the change of its
    neighbor vectors within ``cutoff`` (in the reference configuration).  Writes
    'Shear Strain' (von Mises) and 'Volumetric Strain', plus on request
    'Deformation Gradient' (3x3), 'Strain Tensor' (XX, YY, ZZ, XY, XZ, YZ; Green-Lagrangian),
    'Nonaffine Squared Displacement' (D2min), 'Rotation' (quaternion x, y, z, w)
    and 'Stretch Tensor'.  Particles with too few neighbors are flagged in
    'Selection' (``select_invalid_particles``) and counted in
    'AtomicStrain.invalid_particle_count'.  Particles are matched to the
    reference by 'Particle Identifier' if present, otherwise by index.
    """

    def __init__(self, cutoff: float = 3.0, reference_frame: int = 0, frame_offset: int | None = None,
                 output_deformation_gradients: bool = False, output_strain_tensors: bool = False,
                 output_nonaffine_squared_displacements: bool = False, output_rotations: bool = False,
                 output_stretch_tensors: bool = False, select_invalid_particles: bool = True,
                 affine_mapping: str | None = None, minimum_image_convention: bool = True):
        if affine_mapping not in (None, "off", "to_reference", "to_current"):
            raise ValueError("affine_mapping must be None/'off', 'to_reference' or 'to_current'")
        self.cutoff = cutoff
        self.reference_frame = reference_frame
        self.frame_offset = frame_offset
        self.output_deformation_gradients = output_deformation_gradients
        self.output_strain_tensors = output_strain_tensors
        self.output_nonaffine_squared_displacements = output_nonaffine_squared_displacements
        self.output_rotations = output_rotations
        self.output_stretch_tensors = output_stretch_tensors
        self.select_invalid_particles = select_invalid_particles
        self.affine_mapping = affine_mapping
        self.minimum_image_convention = minimum_image_convention

    def modify(self, data, ctx):
        be, xp = ctx.backend, ctx.xp
        frame = ctx.frame + self.frame_offset if self.frame_offset is not None else self.reference_frame
        ref = ctx.source_frame(frame)
        cur_index = _match_to_reference(data, ref)          # reference particle -> current particle
        ref_pos = ref.positions
        Mref, Mcur = ref.cell.matrix, data.cell.matrix
        if not (ref.cell.is_valid and data.cell.is_valid):
            raise ValueError("atomic strain needs a simulation cell in both configurations")
        # Displacements in the reference cell frame (OVITO's convention).
        s_ref = ref.cell.to_fractional(ref_pos)
        s_cur = data.cell.to_fractional(data.positions[cur_index])
        ds = s_cur - s_ref
        if self.minimum_image_convention:
            ds[:, ref.cell.pbc] -= np.round(ds[:, ref.cell.pbc])
        disp = ds @ Mref
        ref_to_cur = np.linalg.inv(Mref) @ Mcur                 # row-vector map, reference -> current cell

        i, j, dref, _ = find_pairs(ref_pos, ref.cell, self.cutoff, be)
        dref = xp.asarray(dref)
        disp_x = xp.asarray(disp)
        dcur = dref + disp_x[j] - disp_x[i]
        tm = xp.asarray(ref_to_cur)
        if self.affine_mapping == "to_current":
            dref, dcur = dref @ tm, dcur @ tm
        elif self.affine_mapping != "to_reference":
            dcur = dcur @ tm
        n = ref.count
        V = xp.zeros((n, 3, 3))
        W = xp.zeros((n, 3, 3))
        _scatter_add(xp, V, i, dref[:, :, None] * dref[:, None, :])
        _scatter_add(xp, W, i, dcur[:, :, None] * dref[:, None, :])
        nnb = xp.bincount(i, minlength=n)
        sumsq = xp.bincount(i, weights=(dref * dref).sum(1), minlength=n)
        thr = sumsq * 1e-12
        detV = xp.linalg.det(V)
        detW = xp.linalg.det(W)
        valid = (nnb >= 3) & (xp.abs(detV) > thr) & (xp.abs(detW) > thr)
        Vs = xp.where(valid[:, None, None], V, xp.eye(3)[None])
        F = W @ xp.linalg.inv(Vs)
        F = xp.where(valid[:, None, None], F, 0.0)
        E = 0.5 * (xp.swapaxes(F, 1, 2) @ F - xp.eye(3)[None])
        E = xp.where(valid[:, None, None], E, 0.0)
        exx, eyy, ezz, exy, exz, eyz = E[:, 0, 0], E[:, 1, 1], E[:, 2, 2], E[:, 0, 1], E[:, 0, 2], E[:, 1, 2]
        shear = xp.sqrt(exy ** 2 + exz ** 2 + eyz ** 2 + ((exx - eyy) ** 2 + (exx - ezz) ** 2 + (eyy - ezz) ** 2) / 6)
        vol = (exx + eyy + ezz) / 3

        # Outputs are indexed like the current configuration.
        def to_cur(arr):
            arr = be.to_numpy(arr)
            out = np.zeros((data.count,) + arr.shape[1:], arr.dtype)
            out[cur_index] = arr
            return out

        data.particles["Shear Strain"] = to_cur(shear)
        data.particles["Volumetric Strain"] = to_cur(vol)
        if self.output_deformation_gradients:
            data.particles["Deformation Gradient"] = to_cur(F)
        if self.output_strain_tensors:
            data.particles["Strain Tensor"] = to_cur(xp.stack([exx, eyy, ezz, exy, exz, eyz], 1))
        if self.output_nonaffine_squared_displacements:
            res = (dref[:, None, :] * F[i]).sum(-1) - dcur
            d2 = xp.bincount(i, weights=(res * res).sum(1), minlength=n)
            data.particles["Nonaffine Squared Displacement"] = to_cur(xp.where(valid, d2, 0.0))
        if self.output_rotations or self.output_stretch_tensors:
            Fn = be.to_numpy(F)
            u, s, vt = np.linalg.svd(Fn)
            R = u @ vt
            U = np.swapaxes(vt, 1, 2) @ (s[:, :, None] * vt)
            ok = be.to_numpy(valid)
            if self.output_rotations:
                from scipy.spatial.transform import Rotation

                R = np.where((np.linalg.det(R) < 0)[:, None, None], -R, R)
                q = np.zeros((n, 4))
                if ok.any():
                    q[ok] = Rotation.from_matrix(R[ok]).as_quat()
                data.particles["Rotation"] = to_cur(q)
            if self.output_stretch_tensors:
                st = np.stack([U[:, 0, 0], U[:, 1, 1], U[:, 2, 2], U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]], 1)
                data.particles["Stretch Tensor"] = to_cur(np.where(ok[:, None], st, 0.0))
        invalid = ~be.to_numpy(valid)
        if self.select_invalid_particles:
            data.particles["Selection"] = to_cur(invalid)
        data.attributes["AtomicStrain.invalid_particle_count"] = int(invalid.sum())


def _scatter_add(xp, target, index, values):
    if xp is np:
        np.add.at(target, index, values)
    else:  # CuPy
        import cupyx

        cupyx.scatter_add(target, index, values)


def _match_to_reference(data, ref):
    """Index of the current particle for every reference particle."""
    if "Particle Identifier" in data and "Particle Identifier" in ref:
        cur_ids = data["Particle Identifier"]
        order = np.argsort(cur_ids)
        pos = np.clip(np.searchsorted(cur_ids[order], ref["Particle Identifier"]), 0, len(order) - 1)
        if np.any(cur_ids[order][pos] != ref["Particle Identifier"]) or len(cur_ids) != ref.count:
            raise ValueError("particle identifiers of the current and reference frames do not match")
        return order[pos]
    if data.count != ref.count:
        raise ValueError("particle count differs from the reference and there are no identifiers to match")
    return np.arange(data.count)
