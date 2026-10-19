// Native kernels for ewald: Polyhedral Template Matching and Voronoi analysis.
//
// The per-particle logic is ported from OVITO 3.16 (PTMAlgorithm.cpp,
// PolyhedralTemplateMatchingModifier.cpp, VoronoiAnalysisModifier.cpp):
//
//   Copyright 2026 OVITO GmbH, Germany
//   OVITO is free software; you can redistribute it and/or modify it either under the
//   terms of the GNU General Public License version 3 as published by the Free Software
//   Foundation (the "GPL") or, at your option, under the terms of the MIT License.
//   If you do not alter this notice, a recipient may use your version of this
//   file under either the GPL or the MIT License.
//
// This port is used under the MIT License (see LICENSE.OVITO-MIT.txt). The PTM
// library (ptm/, MIT, PM Larsen) and voro++ (voro++/, BSD, LBNL) are vendored unchanged.
//
// Neighbor search is done on the Python side: callers pass, for every particle,
// its nearest neighbors sorted by distance as (index, delta) arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "parallel.h"
#include "ptm/ptm_constants.h"
#include "ptm/ptm_functions.h"
#include "ptm/ptm_initialize_data.h"
#include "mwm_csp/mwm_csp.h"
#include "voro++/cell.hh"

namespace py = pybind11;
using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using Int32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

using ewald::parallel_for;

// ---------------------------------------------------------------------------
// Polyhedral Template Matching
// ---------------------------------------------------------------------------

struct NeighborLists {
    const int64_t* index;   // (N, K)
    const double* delta;    // (N, K, 3)
    const int32_t* count;   // (N)
    size_t K;
};

struct PtmNbrData {
    const NeighborLists* nbrs;
    const int32_t* types;   // may be null
    const uint64_t* cached;
};

// Port of get_neighbours() from OVITO's PTMAlgorithm.cpp.
static int get_neighbours(void* vdata, size_t, size_t atom_index, int num_requested, ptm_atomicenv_t* env)
{
    auto* data = static_cast<PtmNbrData*>(vdata);
    const NeighborLists& nl = *data->nbrs;
    int available = std::min<int>(nl.count[atom_index], PTM_MAX_INPUT_POINTS - 1);
    int numNeighbors = std::min(num_requested - 1, available);
    int dummy = 0;
    ptm_decode_correspondences(PTM_MATCH_FCC, data->cached[atom_index], env->correspondences, &dummy);
    env->atom_indices[0] = atom_index;
    env->points[0][0] = env->points[0][1] = env->points[0][2] = 0;
    const int64_t* idx = nl.index + atom_index * nl.K;
    const double* dl = nl.delta + atom_index * nl.K * 3;
    for(int i = 0; i < numNeighbors; i++) {
        int p = env->correspondences[i + 1] - 1;
        env->atom_indices[i + 1] = (size_t)idx[p];
        env->points[i + 1][0] = dl[3 * p + 0];
        env->points[i + 1][1] = dl[3 * p + 1];
        env->points[i + 1][2] = dl[3 * p + 2];
    }
    if(data->types) {
        env->numbers[0] = data->types[atom_index];
        for(int i = 0; i < numNeighbors; i++) {
            int p = env->correspondences[i + 1] - 1;
            env->numbers[i + 1] = data->types[idx[p]];
        }
    }
    else {
        for(int i = 0; i < numNeighbors + 1; i++) env->numbers[i] = 0;
    }
    env->num = numNeighbors + 1;
    return numNeighbors + 1;
}

static std::once_flag ptm_init_flag;

static py::dict ptm_identify(Int64Array nbr_index, DoubleArray nbr_delta, Int32Array nbr_count,
                             py::object types_obj, int32_t flags, bool calc_def_gradient, int num_threads)
{
    std::call_once(ptm_init_flag, []() {
        if(ptm_initialize_global() != PTM_NO_ERROR) throw std::runtime_error("PTM initialization failed");
    });
    if(nbr_index.ndim() != 2 || nbr_delta.ndim() != 3 || nbr_delta.shape(2) != 3)
        throw std::invalid_argument("nbr_index must be (N, K) and nbr_delta (N, K, 3)");
    const size_t N = nbr_index.shape(0), K = nbr_index.shape(1);
    if((size_t)nbr_delta.shape(0) != N || (size_t)nbr_delta.shape(1) != K || (size_t)nbr_count.size() != N)
        throw std::invalid_argument("neighbor array shapes do not match");
    Int32Array types;
    if(!types_obj.is_none()) {
        types = types_obj.cast<Int32Array>();
        if((size_t)types.size() != N) throw std::invalid_argument("types must have length N");
    }

    NeighborLists nl{nbr_index.data(), nbr_delta.data(), nbr_count.data(), K};
    std::vector<uint64_t> cached(N, 0);
    Int32Array structure(N), ordering(N);
    Int64Array correspondences(N);
    DoubleArray rmsd(N), distance(N), orientation({N, (size_t)4}), defgrad({N, (size_t)3, (size_t)3});
    int32_t* st = structure.mutable_data();
    int32_t* ot = ordering.mutable_data();
    int64_t* co = correspondences.mutable_data();
    double* rm = rmsd.mutable_data();
    double* di = distance.mutable_data();
    double* qu = orientation.mutable_data();
    double* fg = defgrad.mutable_data();
    const int32_t* typ = types_obj.is_none() ? nullptr : types.data();

    {
        py::gil_scoped_release release;
        // Pass 1: topological ordering of each particle's neighbors.
        parallel_for(N, num_threads, [&](size_t start, size_t end, int) {
            ptm_local_handle_t handle = ptm_initialize_local();
            double points[PTM_MAX_INPUT_POINTS - 1][3];
            for(size_t i = start; i < end; i++) {
                int n = std::min<int>(nl.count[i], PTM_MAX_INPUT_POINTS - 1);
                const double* dl = nl.delta + i * K * 3;
                for(int k = 0; k < n; k++)
                    for(int d = 0; d < 3; d++) points[k][d] = dl[3 * k + d];
                ptm_preorder_neighbours(handle, n, points, &cached[i]);
            }
            ptm_uninitialize_local(handle);
        });
        // Pass 2: template matching.
        PtmNbrData nbrdata{&nl, typ, cached.data()};
        parallel_for(N, num_threads, [&](size_t start, size_t end, int) {
            ptm_local_handle_t handle = ptm_initialize_local();
            ptm_result_t result;
            ptm_atomicenv_t env;
            for(size_t i = start; i < end; i++) {
                std::memset(&result, 0, sizeof(result));
                ptm_index(handle, i, get_neighbours, (void*)&nbrdata, flags, calc_def_gradient, &result, &env);
                int type = result.structure_type;
                double* q = qu + 4 * i;
                double* F = fg + 9 * i;
                if(type == PTM_MATCH_NONE) {
                    st[i] = 0; ot[i] = 0; rm[i] = 0; di[i] = 0; co[i] = 0;
                    q[0] = q[1] = q[2] = q[3] = 0;
                    for(int k = 0; k < 9; k++) F[k] = 0;
                    continue;
                }
                st[i] = type;  // PTM_MATCH_* numbering equals OVITO's StructureType enum.
                ot[i] = result.ordering_type;
                rm[i] = result.rmsd;
                di[i] = result.interatomic_distance;
                // OVITO stores the quaternion as (x, y, z, w); PTM returns (w, x, y, z).
                q[0] = result.orientation[1]; q[1] = result.orientation[2];
                q[2] = result.orientation[3]; q[3] = result.orientation[0];
                for(int k = 0; k < 9; k++) F[k] = calc_def_gradient ? result.F[k] : 0.0;
                co[i] = (int64_t)ptm_encode_correspondences(type, env.num, env.correspondences, result.template_index);
            }
            ptm_uninitialize_local(handle);
        });
    }
    py::dict out;
    out["structure"] = structure;
    out["rmsd"] = rmsd;
    out["interatomic_distance"] = distance;
    out["orientation"] = orientation;
    out["ordering"] = ordering;
    out["deformation_gradient"] = defgrad;
    out["correspondences"] = correspondences;
    return out;
}

// ---------------------------------------------------------------------------
// Voronoi analysis
// ---------------------------------------------------------------------------

static constexpr int FaceOrderStorageLimit = 32;

static py::dict voronoi_analysis(DoubleArray positions, DoubleArray cell_matrix, DoubleArray cell_origin,
                                 py::array_t<bool, py::array::c_style | py::array::forcecast> pbc,
                                 Int64Array center_index, Int64Array nbr_index, DoubleArray nbr_delta,
                                 Int32Array nbr_count, py::object radii_obj, double edge_threshold, double face_threshold,
                                 double relative_face_threshold, bool compute_bonds, int num_threads)
{
    const size_t N = positions.shape(0), K = nbr_index.shape(1);
    if(positions.ndim() != 2 || positions.shape(1) != 3) throw std::invalid_argument("positions must be (N, 3)");
    if((size_t)nbr_index.shape(0) != N || (size_t)nbr_delta.shape(0) != N || (size_t)nbr_count.size() != N
       || (size_t)center_index.size() != N)
        throw std::invalid_argument("neighbor array shapes do not match");
    // radii (if given) are indexed by global particle index, like center_index and nbr_index.
    DoubleArray radii;
    if(!radii_obj.is_none()) radii = radii_obj.cast<DoubleArray>();
    const double* pos = positions.data();
    const double* M = cell_matrix.data();     // rows = cell vectors a, b, c
    const double* org = cell_origin.data();
    const bool* hasPbc = pbc.data();
    const int64_t* cidx = center_index.data();
    const int64_t* nidx = nbr_index.data();
    const double* ndel = nbr_delta.data();
    const int32_t* ncnt = nbr_count.data();
    const double* rad = radii_obj.is_none() ? nullptr : radii.data();

    // Unit normals of the cell faces (pointing along +a*, +b*, +c*).
    double normals[3][3];
    for(int d = 0; d < 3; d++) {
        const double* u = M + 3 * ((d + 1) % 3);
        const double* v = M + 3 * ((d + 2) % 3);
        double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        const double* w = M + 3 * d;
        if(n[0] * w[0] + n[1] * w[1] + n[2] * w[2] < 0) for(double& c : n) c = -c;
        double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for(int k = 0; k < 3; k++) normals[d][k] = n[k] / len;
    }
    double corner1[3], corner2[3], boxDiameter = 0;
    for(int k = 0; k < 3; k++) {
        corner1[k] = org[k];
        corner2[k] = org[k] + M[k] + M[3 + k] + M[6 + k];
    }
    for(int d = 0; d < 3; d++) boxDiameter += M[3 * d] * M[3 * d] + M[3 * d + 1] * M[3 * d + 1] + M[3 * d + 2] * M[3 * d + 2];
    boxDiameter = std::sqrt(boxDiameter);
    const double sqEdgeThreshold = edge_threshold * edge_threshold * 4;

    DoubleArray volume(N), cavity(N), surface(N);
    Int32Array coordination(N), maxFaceOrderArr(N), status(N);
    Int32Array indices({N, (size_t)FaceOrderStorageLimit});
    double* vol = volume.mutable_data();
    double* cav = cavity.mutable_data();
    double* sur = surface.mutable_data();
    int32_t* coord = coordination.mutable_data();
    int32_t* mfo = maxFaceOrderArr.mutable_data();
    int32_t* stat = status.mutable_data();   // 0 = ok, 1 = needs more neighbors, 2 = skipped/failed
    int32_t* vind = indices.mutable_data();
    std::fill(vind, vind + N * FaceOrderStorageLimit, 0);
    std::vector<int64_t> bond_i, bond_j;
    std::vector<int32_t> bond_order;
    std::vector<double> bond_area;
    std::mutex bond_mutex;

    {
        py::gil_scoped_release release;
        parallel_for(N, num_threads, [&](size_t start, size_t end, int) {
            std::vector<int64_t> li, lj;
            std::vector<int32_t> lo;
            std::vector<double> la;
            for(size_t index = start; index < end; index++) {
                vol[index] = cav[index] = sur[index] = 0;
                coord[index] = mfo[index] = 0;
                stat[index] = 0;
                const double* p = pos + 3 * index;
                voro::voronoicell_neighbor v;
                v.init(-boxDiameter, boxDiameter, -boxDiameter, boxDiameter, -boxDiameter, boxDiameter);
                bool skipParticle = false;
                for(int dim = 0; dim < 3; dim++) {
                    if(!hasPbc[dim]) {
                        const double* n = normals[dim];
                        double r = 2 * (n[0] * (corner2[0] - p[0]) + n[1] * (corner2[1] - p[1]) + n[2] * (corner2[2] - p[2]));
                        if(r <= 0) skipParticle = true;
                        v.nplane(n[0] * r, n[1] * r, n[2] * r, r * r, -1);
                        r = 2 * (n[0] * (p[0] - corner1[0]) + n[1] * (p[1] - corner1[1]) + n[2] * (p[2] - corner1[2]));
                        if(r <= 0) skipParticle = true;
                        v.nplane(-n[0] * r, -n[1] * r, -n[2] * r, r * r, -1);
                    }
                }
                if(skipParticle) { stat[index] = 2; continue; }

                // Cut with neighbor planes in order of increasing distance, as
                // NearestNeighborFinder::visitNeighbors() does in OVITO.
                double mrs = v.max_radius_squared();
                int n = ncnt[index];
                bool exhausted = true;
                bool cellDeleted = false;
                for(int k = 0; k < n; k++) {
                    const double* d = ndel + (index * K + k) * 3;
                    double rs = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if(rs > mrs) { exhausted = false; break; }
                    int64_t j = nidx[index * K + k];
                    double rsq = rs;
                    if(rad) rsq += rad[cidx[index]] * rad[cidx[index]] - rad[j] * rad[j];
                    if(!v.nplane(d[0], d[1], d[2], rsq, (int)j)) { cellDeleted = true; break; }
                    mrs = v.max_radius_squared();
                }
                if(cellDeleted) { stat[index] = 2; continue; }
                if(exhausted && n < (int)K) exhausted = false;   // caller passed every neighbor there is
                if(exhausted) { stat[index] = 1; continue; }

                // Port of processCell() from VoronoiAnalysisModifier.cpp.
                vol[index] = v.volume();
                cav[index] = 0.5 * std::sqrt(v.max_radius_squared());
                double faceAreaThreshold = face_threshold;
                if(relative_face_threshold > 0)
                    faceAreaThreshold = std::max(v.surface_area() * relative_face_threshold, faceAreaThreshold);
                int localMaxFaceOrder = 0;
                int localVoronoiIndex[FaceOrderStorageLimit] = {0};
                int coordNumber = 0;
                double cellFaceArea = 0;
                for(int i = 1; i < v.p; i++) {
                    for(int jj = 0; jj < v.nu[i]; jj++) {
                        int k = v.ed[i][jj];
                        if(k < 0) continue;
                        int neighbor_id = v.ne[i][jj];
                        int faceOrder = 0;
                        double area = 0;
                        double dvec[3] = {v.pts[3 * k] - v.pts[3 * i], v.pts[3 * k + 1] - v.pts[3 * i + 1], v.pts[3 * k + 2] - v.pts[3 * i + 2]};
                        if(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2] >= sqEdgeThreshold) faceOrder++;
                        v.ed[i][jj] = -1 - k;
                        int l = v.cycle_up(v.ed[i][v.nu[i] + jj], k);
                        do {
                            int m = v.ed[k][l];
                            if(sqEdgeThreshold != 0) {
                                double u[3] = {v.pts[3 * m] - v.pts[3 * k], v.pts[3 * m + 1] - v.pts[3 * k + 1], v.pts[3 * m + 2] - v.pts[3 * k + 2]};
                                if(u[0] * u[0] + u[1] * u[1] + u[2] * u[2] >= sqEdgeThreshold) faceOrder++;
                            }
                            else faceOrder++;
                            double w[3] = {v.pts[3 * m] - v.pts[3 * i], v.pts[3 * m + 1] - v.pts[3 * i + 1], v.pts[3 * m + 2] - v.pts[3 * i + 2]};
                            double c[3] = {dvec[1] * w[2] - dvec[2] * w[1], dvec[2] * w[0] - dvec[0] * w[2], dvec[0] * w[1] - dvec[1] * w[0]};
                            area += std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) / 8;
                            dvec[0] = w[0]; dvec[1] = w[1]; dvec[2] = w[2];
                            v.ed[k][l] = -1 - m;
                            l = v.cycle_up(v.ed[k][v.nu[k] + l], m);
                            k = m;
                        } while(k != i);
                        cellFaceArea += area;
                        if((faceAreaThreshold == 0 || area > faceAreaThreshold) && faceOrder >= 3) {
                            coordNumber++;
                            if(faceOrder > localMaxFaceOrder) localMaxFaceOrder = faceOrder;
                            if(faceOrder - 1 < FaceOrderStorageLimit) localVoronoiIndex[faceOrder - 1]++;
                            if(compute_bonds && neighbor_id >= 0) {
                                li.push_back(cidx[index]); lj.push_back(neighbor_id);
                                lo.push_back(faceOrder); la.push_back(area);
                            }
                        }
                    }
                }
                coord[index] = coordNumber;
                mfo[index] = localMaxFaceOrder;
                sur[index] = cellFaceArea;
                std::copy(localVoronoiIndex, localVoronoiIndex + FaceOrderStorageLimit, vind + index * FaceOrderStorageLimit);
            }
            if(compute_bonds) {
                std::lock_guard<std::mutex> lock(bond_mutex);
                bond_i.insert(bond_i.end(), li.begin(), li.end());
                bond_j.insert(bond_j.end(), lj.begin(), lj.end());
                bond_order.insert(bond_order.end(), lo.begin(), lo.end());
                bond_area.insert(bond_area.end(), la.begin(), la.end());
            }
        });
    }
    py::dict out;
    out["volume"] = volume;
    out["cavity_radius"] = cavity;
    out["surface_area"] = surface;
    out["coordination"] = coordination;
    out["max_face_order"] = maxFaceOrderArr;
    out["indices"] = indices;
    out["status"] = status;
    if(compute_bonds) {
        out["face_i"] = Int64Array(bond_i.size(), bond_i.data());
        out["face_j"] = Int64Array(bond_j.size(), bond_j.data());
        out["face_order"] = Int32Array(bond_order.size(), bond_order.data());
        out["face_area"] = DoubleArray(bond_area.size(), bond_area.data());
    }
    return out;
}

void register_cna(py::module_& m);
void register_crystal(py::module_& m);

// ---------------------------------------------------------------------------
// Centrosymmetry parameter, minimum-weight matching mode (mwm_csp, MIT, PM Larsen)
// ---------------------------------------------------------------------------

static DoubleArray csp_matching(DoubleArray nbr_delta, Int32Array nbr_count, int num_threads)
{
    if(nbr_delta.ndim() != 3 || nbr_delta.shape(2) != 3) throw std::invalid_argument("nbr_delta must be (N, K, 3)");
    const size_t N = nbr_delta.shape(0), K = nbr_delta.shape(1);
    if(K > MWM_CSP_MAX_POINTS) throw std::invalid_argument("at most 32 neighbors are supported");
    DoubleArray out(N);
    double* csp = out.mutable_data();
    const double* delta = nbr_delta.data();
    const int32_t* count = nbr_count.data();
    {
        py::gil_scoped_release release;
        parallel_for(N, num_threads, [&](size_t start, size_t end, int) {
            double P[MWM_CSP_MAX_POINTS][3];
            for(size_t i = start; i < end; i++) {
                int n = std::min<int>(count[i], (int)K);
                for(int k = 0; k < n; k++)
                    for(int d = 0; d < 3; d++) P[k][d] = delta[(i * K + k) * 3 + d];
                csp[i] = calculate_mwm_csp(n, P);
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_native, m)
{
    register_cna(m);
    register_crystal(m);
    m.doc() = "Native CNA, PTM and Voronoi kernels ported from OVITO (MIT).";
    m.def("ptm_identify", &ptm_identify, py::arg("nbr_index"), py::arg("nbr_delta"), py::arg("nbr_count"),
          py::arg("types") = py::none(), py::arg("flags") = PTM_CHECK_DEFAULT,
          py::arg("calc_def_gradient") = false, py::arg("num_threads") = 0);
    m.def("voronoi_analysis", &voronoi_analysis, py::arg("positions"), py::arg("cell_matrix"), py::arg("cell_origin"),
          py::arg("pbc"), py::arg("center_index"), py::arg("nbr_index"), py::arg("nbr_delta"), py::arg("nbr_count"),
          py::arg("radii") = py::none(), py::arg("edge_threshold") = 0.0, py::arg("face_threshold") = 0.0,
          py::arg("relative_face_threshold") = 0.0, py::arg("compute_bonds") = false, py::arg("num_threads") = 0);
    m.def("csp_matching", &csp_matching, py::arg("nbr_delta"), py::arg("nbr_count"), py::arg("num_threads") = 0);
    m.attr("FACE_ORDER_STORAGE_LIMIT") = FaceOrderStorageLimit;
}
