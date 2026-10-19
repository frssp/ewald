// Crystal-analysis kernels: elastic strain calculation and grain segmentation.
//
// Ported from OVITO 3.16 (StructureAnalysis.cpp, ClusterGraph.cpp, Cluster.h,
// ElasticStrainEngine.cpp, GrainSegmentationEngine.cpp, NodePairSampling.cpp,
// ThresholdSelection.h, PTMNeighborFinder.cpp, DisjointSet.h):
//
//   Copyright 2026 OVITO GmbH, Germany
//   OVITO is free software; you can redistribute it and/or modify it either under the
//   terms of the GNU General Public License version 3 as published by the Free Software
//   Foundation (the "GPL") or, at your option, under the terms of the MIT License.
//   If you do not alter this notice, a recipient may use your version of this
//   file under either the GPL or the MIT License.
//
// This port is used under the MIT License (see LICENSE.OVITO-MIT.txt).
//
// Neighbor search is done on the Python side: callers pass, for every particle, its
// nearest neighbors sorted by distance as (index, delta) arrays. Floating-point types
// (double for lattice math, float for cluster transition matrices) and the order of
// operations follow OVITO so results agree to rounding.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cna_bonds.h"
#include "parallel.h"
#include "ptm/ptm_constants.h"
#include "ptm/ptm_functions.h"
#include "ptm/ptm_initialize_data.h"
#include "ptm/ptm_quat.h"

namespace py = pybind11;
using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using Int32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

using ewald::parallel_for;
using namespace ewald::cna;

namespace {

// ---------------------------------------------------------------------------
// Minimal vector/matrix types with OVITO's storage and arithmetic order
// ---------------------------------------------------------------------------

template <typename T> constexpr T epsilon_v = T(0);
template <> constexpr double epsilon_v<double> = 1e-12;
template <> constexpr float epsilon_v<float> = 1e-6f;
constexpr float CA_TRANSITION_MATRIX_EPSILON = 1e-4f;

template <typename T>
struct Vec3 {
    T v[3] = {0, 0, 0};
    Vec3() = default;
    Vec3(T x, T y, T z) : v{x, y, z} {}
    T& operator[](int i) { return v[i]; }
    T operator[](int i) const { return v[i]; }
    T x() const { return v[0]; }
    T y() const { return v[1]; }
    T z() const { return v[2]; }
    Vec3 operator+(const Vec3& o) const { return {v[0] + o.v[0], v[1] + o.v[1], v[2] + o.v[2]}; }
    Vec3 operator-(const Vec3& o) const { return {v[0] - o.v[0], v[1] - o.v[1], v[2] - o.v[2]}; }
    Vec3 operator-() const { return {-v[0], -v[1], -v[2]}; }
    Vec3& operator+=(const Vec3& o) { v[0] += o.v[0]; v[1] += o.v[1]; v[2] += o.v[2]; return *this; }
    Vec3& operator-=(const Vec3& o) { v[0] -= o.v[0]; v[1] -= o.v[1]; v[2] -= o.v[2]; return *this; }
    Vec3 operator*(T s) const { return {v[0] * s, v[1] * s, v[2] * s}; }
    Vec3 operator/(T s) const { return {v[0] / s, v[1] / s, v[2] / s}; }
    T dot(const Vec3& o) const { return v[0] * o.v[0] + v[1] * o.v[1] + v[2] * o.v[2]; }
    Vec3 cross(const Vec3& b) const {
        return {v[1] * b.v[2] - v[2] * b.v[1], v[2] * b.v[0] - v[0] * b.v[2], v[0] * b.v[1] - v[1] * b.v[0]};
    }
    T squaredLength() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }
    T length() const { return std::sqrt(squaredLength()); }
    Vec3 normalized() const { return *this / length(); }
    bool equals(const Vec3& o, T tol = epsilon_v<T>) const {
        return std::abs(o.v[0] - v[0]) <= tol && std::abs(o.v[1] - v[1]) <= tol && std::abs(o.v[2] - v[2]) <= tol;
    }
    bool isZero(T tol = epsilon_v<T>) const {
        return std::abs(v[0]) <= tol && std::abs(v[1]) <= tol && std::abs(v[2]) <= tol;
    }
};
template <typename T> Vec3<T> operator*(T s, const Vec3<T>& a) { return a * s; }
using Vector3 = Vec3<double>;

// 3x3 matrix stored as three columns, c[col][row], like OVITO's Matrix_3.
template <typename T>
struct Mat3 {
    Vec3<T> c[3];
    Mat3() = default;
    Mat3(T m11, T m12, T m13, T m21, T m22, T m23, T m31, T m32, T m33)
        : c{Vec3<T>(m11, m21, m31), Vec3<T>(m12, m22, m32), Vec3<T>(m13, m23, m33)} {}
    static Mat3 Zero() { return Mat3(0, 0, 0, 0, 0, 0, 0, 0, 0); }
    static Mat3 Identity() { return Mat3(1, 0, 0, 0, 1, 0, 0, 0, 1); }
    Vec3<T>& column(int i) { return c[i]; }
    const Vec3<T>& column(int i) const { return c[i]; }
    Vec3<T>& operator[](int i) { return c[i]; }
    const Vec3<T>& operator[](int i) const { return c[i]; }
    T operator()(int row, int col) const { return c[col][row]; }
    T& operator()(int row, int col) { return c[col][row]; }
    template <typename U> Mat3<U> toDataType() const {
        const Mat3& m = *this;
        return Mat3<U>((U)m(0, 0), (U)m(0, 1), (U)m(0, 2), (U)m(1, 0), (U)m(1, 1), (U)m(1, 2),
                       (U)m(2, 0), (U)m(2, 1), (U)m(2, 2));
    }
    T determinant() const {
        const Mat3& t = *this;
        return ((t[0][0] * t[1][1] - t[0][1] * t[1][0]) * (t[2][2])
                - (t[0][0] * t[1][2] - t[0][2] * t[1][0]) * (t[2][1])
                + (t[0][1] * t[1][2] - t[0][2] * t[1][1]) * (t[2][0]));
    }
    bool inverse(Mat3& result, T eps = epsilon_v<T>) const {
        const Mat3& t = *this;
        T det = determinant();
        if(std::abs(det) <= eps) return false;
        result = Mat3((t[1][1] * t[2][2] - t[1][2] * t[2][1]) / det,
                      (t[2][0] * t[1][2] - t[1][0] * t[2][2]) / det,
                      (t[1][0] * t[2][1] - t[1][1] * t[2][0]) / det,
                      (t[2][1] * t[0][2] - t[0][1] * t[2][2]) / det,
                      (t[0][0] * t[2][2] - t[2][0] * t[0][2]) / det,
                      (t[0][1] * t[2][0] - t[0][0] * t[2][1]) / det,
                      (t[0][1] * t[1][2] - t[1][1] * t[0][2]) / det,
                      (t[0][2] * t[1][0] - t[0][0] * t[1][2]) / det,
                      (t[0][0] * t[1][1] - t[1][0] * t[0][1]) / det);
        return true;
    }
    Mat3 inverse() const {
        if(determinant() == 0) throw std::runtime_error("Matrix3 cannot be inverted: determinant is zero.");
        Mat3 r;
        inverse(r, T(0));
        return r;
    }
    bool equals(const Mat3& m, T tol = epsilon_v<T>) const {
        for(int i = 0; i < 3; i++)
            if(!c[i].equals(m.c[i], tol)) return false;
        return true;
    }
    bool isOrthogonalMatrix(T eps = epsilon_v<T>) const {
        const Mat3& t = *this;
        return (std::abs(t[0][0] * t[1][0] + t[0][1] * t[1][1] + t[0][2] * t[1][2]) <= eps) &&
               (std::abs(t[0][0] * t[2][0] + t[0][1] * t[2][1] + t[0][2] * t[2][2]) <= eps) &&
               (std::abs(t[1][0] * t[2][0] + t[1][1] * t[2][1] + t[1][2] * t[2][2]) <= eps) &&
               (std::abs(t[0][0] * t[0][0] + t[0][1] * t[0][1] + t[0][2] * t[0][2] - T(1)) <= eps) &&
               (std::abs(t[1][0] * t[1][0] + t[1][1] * t[1][1] + t[1][2] * t[1][2] - T(1)) <= eps) &&
               (std::abs(t[2][0] * t[2][0] + t[2][1] * t[2][1] + t[2][2] * t[2][2] - T(1)) <= eps);
    }
};
template <typename T>
Vec3<T> operator*(const Mat3<T>& m, const Vec3<T>& v) {
    return {m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
            m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
            m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]};
}
template <typename T>
Mat3<T> operator*(const Mat3<T>& a, const Mat3<T>& b) {
    return Mat3<T>(a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0),
                   a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1),
                   a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2),
                   a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0),
                   a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1),
                   a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2),
                   a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0),
                   a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1),
                   a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2));
}
using Matrix3 = Mat3<double>;
using Matrix3F = Mat3<float>;

// Simulation cell: cell vectors as matrix columns, periodic flags and the reciprocal matrix.
struct SimCell {
    Matrix3 m = Matrix3::Identity();
    Matrix3 inv = Matrix3::Identity();
    Vector3 origin;
    bool pbc[3] = {false, false, false};

    SimCell(const DoubleArray& cell_vectors, py::object pbc_flags) {
        auto c = cell_vectors.unchecked<2>();
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++) m(j, i) = c(i, j);
        auto flags = pbc_flags.cast<std::vector<bool>>();
        for(int i = 0; i < 3; i++) pbc[i] = flags[i];
        if(pbc[0] || pbc[1] || pbc[2]) {
            if(!m.inverse(inv)) throw std::runtime_error("simulation cell is singular");
        }
    }
    bool hasPbc(int d) const { return pbc[d]; }
    double prodrow(const Vector3& v, int d) const { return inv[0][d] * v[0] + inv[1][d] * v[1] + inv[2][d] * v[2]; }
    Vector3 wrapVector(const Vector3& v) const {
        Vector3 vout = v;
        for(int dim = 0; dim < 3; dim++) {
            if(pbc[dim]) {
                if(double s = std::floor(prodrow(v, dim) + 0.5)) vout -= s * m.column(dim);
            }
        }
        return vout;
    }
};

// k nearest neighbors of every particle (index, delta, count), sorted by distance.
struct Neighbors {
    const int64_t* index;
    const double* delta;
    const int32_t* count;
    size_t K;
    Neighbors(const Int64Array& idx, const DoubleArray& dl, const Int32Array& cnt)
        : index(idx.data()), delta(dl.data()), count(cnt.data()), K(idx.ndim() == 2 ? idx.shape(1) : 0) {}
    int num(size_t i) const { return std::min<int>(count[i], (int)K); }
    int64_t idx(size_t i, int n) const { return index[i * K + n]; }
    Vector3 vec(size_t i, int n) const {
        const double* d = delta + (i * K + n) * 3;
        return {d[0], d[1], d[2]};
    }
    double distSq(size_t i, int n) const { return vec(i, n).squaredLength(); }
};

[[noreturn]] void cellTooSmallError(int dim) {
    static const char* axes[3] = {"X", "Y", "Z"};
    throw std::runtime_error(std::string("Simulation box is too short along cell vector ") + std::to_string(dim + 1) +
                             " (" + axes[dim] + ") to perform analysis. Please extend it first using the "
                             "'Replicate' modifier.");
}

// ---------------------------------------------------------------------------
// Cluster graph (ClusterGraph.cpp / Cluster.h), reduced to what the structure analysis needs
// ---------------------------------------------------------------------------

struct Cluster;

struct ClusterTransition {
    Cluster* cluster1 = nullptr;
    Cluster* cluster2 = nullptr;
    Matrix3F tm;
    ClusterTransition* reverse = nullptr;
    ClusterTransition* next = nullptr;
    int distance = 0;
    int area = 0;
    bool isSelfTransition() const { return reverse == this; }
};

struct Cluster {
    int id;
    int structure;
    int atomCount = 0;
    ClusterTransition* transitions = nullptr;
    int rank = 0;
    Matrix3F orientation = Matrix3F::Identity();
    int symmetryTransformation = 0;
    ClusterTransition* parentTransition = nullptr;

    Cluster(int _id, int _structure) : id(_id), structure(_structure) {}

    void insertTransition(ClusterTransition* newTransition) {
        ClusterTransition* appendAfter = nullptr;
        for(ClusterTransition* t = transitions; t != nullptr && t->distance < newTransition->distance; t = t->next)
            appendAfter = t;
        if(appendAfter) {
            newTransition->next = appendAfter->next;
            appendAfter->next = newTransition;
        }
        else {
            newTransition->next = transitions;
            transitions = newTransition;
        }
    }
    ClusterTransition* findTransition(Cluster* clusterB) const {
        for(ClusterTransition* t = transitions; t != nullptr; t = t->next)
            if(t->cluster2 == clusterB) return t;
        return nullptr;
    }
};

struct ClusterGraph {
    std::deque<Cluster> clusterPool;
    std::deque<ClusterTransition> transitionPool;
    std::vector<Cluster*> clusters;
    std::vector<ClusterTransition*> clusterTransitions;

    ClusterGraph() { createCluster(0, 0); }

    Cluster* createCluster(int structureType, int id = -1) {
        if(id < 0) id = (int)clusters.size();
        clusterPool.emplace_back(id, structureType);
        clusters.push_back(&clusterPool.back());
        return clusters.back();
    }
    Cluster* findCluster(int64_t id) const { return clusters[id]; }

    ClusterTransition* createSelfTransition(Cluster* cluster) {
        if(cluster->transitions != nullptr && cluster->transitions->isSelfTransition()) return cluster->transitions;
        transitionPool.emplace_back();
        ClusterTransition* t = &transitionPool.back();
        t->cluster1 = t->cluster2 = cluster;
        t->tm = Matrix3F::Identity();
        t->reverse = t;
        t->distance = 0;
        t->next = cluster->transitions;
        t->area = 0;
        cluster->transitions = t;
        return t;
    }

    ClusterTransition* createClusterTransition(Cluster* clusterA, Cluster* clusterB, const Matrix3F& tm, int distance = 1) {
        if(clusterA == clusterB && tm.equals(Matrix3F::Identity(), CA_TRANSITION_MATRIX_EPSILON))
            return createSelfTransition(clusterA);
        for(ClusterTransition* t = clusterA->transitions; t != nullptr; t = t->next) {
            if(t->cluster2 == clusterB && t->tm.equals(tm, CA_TRANSITION_MATRIX_EPSILON)) return t;
        }
        transitionPool.emplace_back();
        ClusterTransition* tAB = &transitionPool.back();
        transitionPool.emplace_back();
        ClusterTransition* tBA = &transitionPool.back();
        tAB->cluster1 = clusterA;
        tAB->cluster2 = clusterB;
        tBA->cluster1 = clusterB;
        tBA->cluster2 = clusterA;
        tAB->tm = tm;
        tBA->tm = tm.inverse();
        tAB->reverse = tBA;
        tBA->reverse = tAB;
        tAB->distance = distance;
        tBA->distance = distance;
        tAB->area = 0;
        tBA->area = 0;
        clusterA->insertTransition(tAB);
        clusterB->insertTransition(tBA);
        clusterTransitions.push_back(tAB);
        return tAB;
    }

    ClusterTransition* concatenateClusterTransitions(ClusterTransition* tAB, ClusterTransition* tBC) {
        if(tBC->isSelfTransition()) return tAB;
        if(tAB->isSelfTransition()) return tBC;
        if(tAB->reverse == tBC) return createSelfTransition(tAB->cluster1);
        return createClusterTransition(tAB->cluster1, tBC->cluster2, tBC->tm * tAB->tm, tAB->distance + tBC->distance);
    }
};

// ---------------------------------------------------------------------------
// Structure analysis (StructureAnalysis.cpp)
// ---------------------------------------------------------------------------

enum LatticeStructureType { LATTICE_OTHER = 0, LATTICE_FCC, LATTICE_HCP, LATTICE_BCC, LATTICE_CUBIC_DIAMOND,
                            LATTICE_HEX_DIAMOND, NUM_LATTICE_TYPES };
enum CoordinationStructureType { COORD_OTHER = 0, COORD_FCC, COORD_HCP, COORD_BCC, COORD_CUBIC_DIAMOND,
                                 COORD_HEX_DIAMOND, NUM_COORD_TYPES };
constexpr int SA_MAX_NEIGHBORS = 16;

struct CoordinationStructure {
    int numNeighbors = 0;
    std::vector<Vector3> latticeVectors;
    NeighborBondArray neighborArray;
    int cnaSignatures[SA_MAX_NEIGHBORS] = {};
    int commonNeighbors[SA_MAX_NEIGHBORS][2] = {};
};

struct SymmetryPermutation {
    Matrix3 transformation;
    std::array<int, SA_MAX_NEIGHBORS> permutation;
    std::vector<int> product;
    std::vector<int> inverseProduct;
};

struct LatticeStructure {
    const CoordinationStructure* coordStructure = nullptr;
    std::vector<Vector3> latticeVectors;
    Matrix3 primitiveCell = Matrix3::Zero();
    Matrix3 primitiveCellInverse = Matrix3::Zero();
    int maxNeighbors = 0;
    std::vector<SymmetryPermutation> permutations;
};

CoordinationStructure coordinationStructures[NUM_COORD_TYPES];
LatticeStructure latticeStructures[NUM_LATTICE_TYPES];

template <typename iterator>
void bitmapSort(iterator begin, iterator end, int max)
{
    int bitarray = 0;
    for(iterator pin = begin; pin != end; ++pin) bitarray |= 1 << (*pin);
    iterator pout = begin;
    for(int i = max - 1; i >= 0; i--)
        if(bitarray & (1 << i)) *pout++ = i;
}

void initializeListOfStructures()
{
    using VecType = Vector3;
    using MatType = Matrix3;
    using FloatType = double;
    using std::sqrt;
    coordinationStructures[COORD_OTHER].numNeighbors = 0;
    latticeStructures[LATTICE_OTHER].coordStructure = &coordinationStructures[COORD_OTHER];
    latticeStructures[LATTICE_OTHER].maxNeighbors = 0;

    // FCC
    VecType fccVec[12] = {
        VecType(0.5, 0.5, 0.0), VecType(0.0, 0.5, 0.5), VecType(0.5, 0.0, 0.5),
        VecType(-0.5, -0.5, 0.0), VecType(0.0, -0.5, -0.5), VecType(-0.5, 0.0, -0.5),
        VecType(-0.5, 0.5, 0.0), VecType(0.0, -0.5, 0.5), VecType(-0.5, 0.0, 0.5),
        VecType(0.5, -0.5, 0.0), VecType(0.0, 0.5, -0.5), VecType(0.5, 0.0, -0.5)};
    coordinationStructures[COORD_FCC].numNeighbors = 12;
    for(int ni1 = 0; ni1 < 12; ni1++) {
        coordinationStructures[COORD_FCC].neighborArray.setNeighborBond(ni1, ni1, false);
        for(int ni2 = ni1 + 1; ni2 < 12; ni2++) {
            bool bonded = (fccVec[ni1] - fccVec[ni2]).length() < (std::sqrt(0.5f) + 1.0) * 0.5;
            coordinationStructures[COORD_FCC].neighborArray.setNeighborBond(ni1, ni2, bonded);
        }
        coordinationStructures[COORD_FCC].cnaSignatures[ni1] = 0;
    }
    coordinationStructures[COORD_FCC].latticeVectors.assign(std::begin(fccVec), std::end(fccVec));
    latticeStructures[LATTICE_FCC].latticeVectors.assign(std::begin(fccVec), std::end(fccVec));
    latticeStructures[LATTICE_FCC].coordStructure = &coordinationStructures[COORD_FCC];
    latticeStructures[LATTICE_FCC].primitiveCell.column(0) = Vector3(0.5, 0.5, 0.0);
    latticeStructures[LATTICE_FCC].primitiveCell.column(1) = Vector3(0.0, 0.5, 0.5);
    latticeStructures[LATTICE_FCC].primitiveCell.column(2) = Vector3(0.5, 0.0, 0.5);
    latticeStructures[LATTICE_FCC].maxNeighbors = 12;

    // HCP
    VecType hcpVec[18] = {
        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
        VecType(-sqrt(2.0) / 2.0, 0.0, 0.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(0.0, -sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
        VecType(sqrt(2.0) / 2.0, 0.0, 0.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
        VecType(0.0, -sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),

        VecType(0.0, sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
        VecType(0.0, sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0)};
    coordinationStructures[COORD_HCP].numNeighbors = 12;
    for(int ni1 = 0; ni1 < 12; ni1++) {
        coordinationStructures[COORD_HCP].neighborArray.setNeighborBond(ni1, ni1, false);
        for(int ni2 = ni1 + 1; ni2 < 12; ni2++) {
            bool bonded = (hcpVec[ni1] - hcpVec[ni2]).length() < (std::sqrt(0.5) + 1.0) * 0.5;
            coordinationStructures[COORD_HCP].neighborArray.setNeighborBond(ni1, ni2, bonded);
        }
        coordinationStructures[COORD_HCP].cnaSignatures[ni1] = (hcpVec[ni1].z() == 0) ? 1 : 0;
    }
    coordinationStructures[COORD_HCP].latticeVectors.assign(std::begin(hcpVec), std::begin(hcpVec) + 12);
    latticeStructures[LATTICE_HCP].latticeVectors.assign(std::begin(hcpVec), std::end(hcpVec));
    latticeStructures[LATTICE_HCP].coordStructure = &coordinationStructures[COORD_HCP];
    latticeStructures[LATTICE_HCP].primitiveCell.column(0) = Vector3(sqrt(0.5) / 2, -sqrt(6.0) / 4, 0.0);
    latticeStructures[LATTICE_HCP].primitiveCell.column(1) = Vector3(sqrt(0.5) / 2, sqrt(6.0) / 4, 0.0);
    latticeStructures[LATTICE_HCP].primitiveCell.column(2) = Vector3(0.0, 0.0, sqrt(8.0 / 6.0));
    latticeStructures[LATTICE_HCP].maxNeighbors = 12;

    // BCC
    VecType bccVec[14] = {
        VecType(0.5, 0.5, 0.5), VecType(-0.5, 0.5, 0.5), VecType(0.5, 0.5, -0.5), VecType(-0.5, -0.5, 0.5),
        VecType(0.5, -0.5, 0.5), VecType(-0.5, 0.5, -0.5), VecType(-0.5, -0.5, -0.5), VecType(0.5, -0.5, -0.5),
        VecType(1.0, 0.0, 0.0), VecType(-1.0, 0.0, 0.0), VecType(0.0, 1.0, 0.0), VecType(0.0, -1.0, 0.0),
        VecType(0.0, 0.0, 1.0), VecType(0.0, 0.0, -1.0)};
    coordinationStructures[COORD_BCC].numNeighbors = 14;
    for(int ni1 = 0; ni1 < 14; ni1++) {
        coordinationStructures[COORD_BCC].neighborArray.setNeighborBond(ni1, ni1, false);
        for(int ni2 = ni1 + 1; ni2 < 14; ni2++) {
            bool bonded = (bccVec[ni1] - bccVec[ni2]).length() < (FloatType(1) + sqrt(FloatType(2))) * FloatType(0.5);
            coordinationStructures[COORD_BCC].neighborArray.setNeighborBond(ni1, ni2, bonded);
        }
        coordinationStructures[COORD_BCC].cnaSignatures[ni1] = (ni1 < 8) ? 0 : 1;
    }
    coordinationStructures[COORD_BCC].latticeVectors.assign(std::begin(bccVec), std::end(bccVec));
    latticeStructures[LATTICE_BCC].latticeVectors.assign(std::begin(bccVec), std::end(bccVec));
    latticeStructures[LATTICE_BCC].coordStructure = &coordinationStructures[COORD_BCC];
    latticeStructures[LATTICE_BCC].primitiveCell.column(0) = Vector3(1.0, 0.0, 0.0);
    latticeStructures[LATTICE_BCC].primitiveCell.column(1) = Vector3(0.0, 1.0, 0.0);
    latticeStructures[LATTICE_BCC].primitiveCell.column(2) = Vector3(0.5, 0.5, 0.5);
    latticeStructures[LATTICE_BCC].maxNeighbors = 14;

    // Cubic diamond
    VecType diamondCubicVec[] = {
        VecType(0.25, 0.25, 0.25), VecType(0.25, -0.25, -0.25), VecType(-0.25, -0.25, 0.25),
        VecType(-0.25, 0.25, -0.25),

        VecType(0, -0.5, 0.5), VecType(0.5, 0.5, 0), VecType(-0.5, 0, 0.5), VecType(-0.5, 0.5, 0),
        VecType(0, 0.5, 0.5), VecType(0.5, -0.5, 0), VecType(0.5, 0, 0.5), VecType(0.5, 0, -0.5),
        VecType(-0.5, -0.5, 0), VecType(0, -0.5, -0.5), VecType(0, 0.5, -0.5), VecType(-0.5, 0, -0.5),

        VecType(0.25, -0.25, 0.25), VecType(0.25, 0.25, -0.25), VecType(-0.25, 0.25, 0.25),
        VecType(-0.25, -0.25, -0.25)};
    coordinationStructures[COORD_CUBIC_DIAMOND].numNeighbors = 16;
    for(int ni1 = 0; ni1 < 16; ni1++) {
        coordinationStructures[COORD_CUBIC_DIAMOND].neighborArray.setNeighborBond(ni1, ni1, false);
        FloatType cutoff = (ni1 < 4) ? (sqrt(3.0) * 0.25 + sqrt(0.5)) / 2 : (1.0 + sqrt(0.5)) / 2;
        for(int ni2 = ni1 + 1; ni2 < 4; ni2++)
            coordinationStructures[COORD_CUBIC_DIAMOND].neighborArray.setNeighborBond(ni1, ni2, false);
        for(int ni2 = std::max(ni1 + 1, 4); ni2 < 16; ni2++) {
            bool bonded = (diamondCubicVec[ni1] - diamondCubicVec[ni2]).length() < cutoff;
            coordinationStructures[COORD_CUBIC_DIAMOND].neighborArray.setNeighborBond(ni1, ni2, bonded);
        }
        coordinationStructures[COORD_CUBIC_DIAMOND].cnaSignatures[ni1] = (ni1 < 4) ? 0 : 1;
    }
    coordinationStructures[COORD_CUBIC_DIAMOND].latticeVectors.assign(std::begin(diamondCubicVec),
                                                                      std::begin(diamondCubicVec) + 16);
    latticeStructures[LATTICE_CUBIC_DIAMOND].latticeVectors.assign(std::begin(diamondCubicVec), std::end(diamondCubicVec));
    latticeStructures[LATTICE_CUBIC_DIAMOND].coordStructure = &coordinationStructures[COORD_CUBIC_DIAMOND];
    latticeStructures[LATTICE_CUBIC_DIAMOND].primitiveCell.column(0) = Vector3(0.5, 0.5, 0.0);
    latticeStructures[LATTICE_CUBIC_DIAMOND].primitiveCell.column(1) = Vector3(0.0, 0.5, 0.5);
    latticeStructures[LATTICE_CUBIC_DIAMOND].primitiveCell.column(2) = Vector3(0.5, 0.0, 0.5);
    latticeStructures[LATTICE_CUBIC_DIAMOND].maxNeighbors = 16;

    // Hexagonal diamond
    VecType diamondHexVec[] = {
        VecType(-sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
        VecType(0, -sqrt(3.0 / 2.0) / 3, -sqrt(3.0) / 12),
        VecType(sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
        VecType(0, 0, sqrt(3.0) / 4),

        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
        VecType(-sqrt(2.0) / 2.0, 0.0, 0.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 4.0, 0.0),
        VecType(sqrt(2.0) / 2.0, 0.0, 0.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 4.0, 0.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(0.0, -sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
        VecType(0.0, -sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),

        VecType(-sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
        VecType(0, -sqrt(3.0 / 2.0) / 3, sqrt(3.0) / 12),
        VecType(sqrt(2.0) / 4, sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
        VecType(0, 0, -sqrt(3.0) / 4),

        VecType(-sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),
        VecType(0, sqrt(3.0 / 2.0) / 3, -sqrt(3.0) / 12),
        VecType(sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, -sqrt(3.0) / 12),

        VecType(-sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),
        VecType(0, sqrt(3.0 / 2.0) / 3, sqrt(3.0) / 12),
        VecType(sqrt(2.0) / 4, -sqrt(3.0 / 2.0) / 6, sqrt(3.0) / 12),

        VecType(0.0, sqrt(6.0) / 6.0, sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0),
        VecType(0.0, sqrt(6.0) / 6.0, -sqrt(3.0) / 3.0),
        VecType(sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, -sqrt(3.0) / 3.0),
        VecType(-sqrt(2.0) / 4.0, -sqrt(6.0) / 12.0, sqrt(3.0) / 3.0)};
    coordinationStructures[COORD_HEX_DIAMOND].numNeighbors = 16;
    for(int ni1 = 0; ni1 < 16; ni1++) {
        coordinationStructures[COORD_HEX_DIAMOND].neighborArray.setNeighborBond(ni1, ni1, false);
        FloatType cutoff = (ni1 < 4) ? (sqrt(3.0) * 0.25 + sqrt(0.5)) / 2 : (1.0 + sqrt(0.5)) / 2;
        for(int ni2 = ni1 + 1; ni2 < 4; ni2++)
            coordinationStructures[COORD_HEX_DIAMOND].neighborArray.setNeighborBond(ni1, ni2, false);
        for(int ni2 = std::max(ni1 + 1, 4); ni2 < 16; ni2++) {
            bool bonded = (diamondHexVec[ni1] - diamondHexVec[ni2]).length() < cutoff;
            coordinationStructures[COORD_HEX_DIAMOND].neighborArray.setNeighborBond(ni1, ni2, bonded);
        }
        coordinationStructures[COORD_HEX_DIAMOND].cnaSignatures[ni1] =
            (ni1 < 4) ? 0 : ((diamondHexVec[ni1].z() == 0) ? 2 : 1);
    }
    coordinationStructures[COORD_HEX_DIAMOND].latticeVectors.assign(std::begin(diamondHexVec), std::begin(diamondHexVec) + 16);
    latticeStructures[LATTICE_HEX_DIAMOND].latticeVectors.assign(std::begin(diamondHexVec), std::end(diamondHexVec));
    latticeStructures[LATTICE_HEX_DIAMOND].coordStructure = &coordinationStructures[COORD_HEX_DIAMOND];
    latticeStructures[LATTICE_HEX_DIAMOND].primitiveCell.column(0) = Vector3(sqrt(0.5) / 2, -sqrt(6.0) / 4, 0.0);
    latticeStructures[LATTICE_HEX_DIAMOND].primitiveCell.column(1) = Vector3(sqrt(0.5) / 2, sqrt(6.0) / 4, 0.0);
    latticeStructures[LATTICE_HEX_DIAMOND].primitiveCell.column(2) = Vector3(0.0, 0.0, sqrt(8.0 / 6.0));
    latticeStructures[LATTICE_HEX_DIAMOND].maxNeighbors = 16;

    for(auto coordStruct = std::begin(coordinationStructures); coordStruct != std::end(coordinationStructures); ++coordStruct) {
        // Find two non-coplanar common neighbors for every neighbor bond.
        for(int neighIndex = 0; neighIndex < coordStruct->numNeighbors; neighIndex++) {
            MatType tm;
            tm.column(0) = coordStruct->latticeVectors[neighIndex];
            bool found = false;
            for(int i1 = 0; i1 < coordStruct->numNeighbors && !found; i1++) {
                if(!coordStruct->neighborArray.neighborBond(neighIndex, i1)) continue;
                tm.column(1) = coordStruct->latticeVectors[i1];
                for(int i2 = i1 + 1; i2 < coordStruct->numNeighbors; i2++) {
                    if(!coordStruct->neighborArray.neighborBond(neighIndex, i2)) continue;
                    tm.column(2) = coordStruct->latticeVectors[i2];
                    if(std::abs(tm.determinant()) > epsilon_v<double>) {
                        coordStruct->commonNeighbors[neighIndex][0] = i1;
                        coordStruct->commonNeighbors[neighIndex][1] = i2;
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    // Generate symmetry information.
    for(auto latticeStruct = std::begin(latticeStructures); latticeStruct != std::end(latticeStructures); ++latticeStruct) {
        if(latticeStruct->latticeVectors.empty()) continue;

        latticeStruct->primitiveCellInverse = latticeStruct->primitiveCell.inverse();
        const CoordinationStructure& coordStruct = *latticeStruct->coordStructure;

        // Find three non-coplanar neighbor vectors.
        int nindices[3];
        MatType tm1;
        int n = 0;
        for(int i = 0; i < coordStruct.numNeighbors && n < 3; i++) {
            tm1.column(n) = coordStruct.latticeVectors[i];
            if(n == 1) {
                if(tm1.column(0).cross(tm1.column(1)).squaredLength() <= epsilon_v<double>) continue;
            }
            else if(n == 2) {
                if(std::abs(tm1.determinant()) <= epsilon_v<double>) continue;
            }
            nindices[n++] = i;
        }
        MatType tm1inverse = tm1.inverse();

        // Find symmetry permutations.
        std::vector<int> permutation(latticeStruct->latticeVectors.size());
        std::vector<int> lastPermutation(latticeStruct->latticeVectors.size(), -1);
        std::iota(permutation.begin(), permutation.end(), 0);
        SymmetryPermutation symmetryPermutation;
        do {
            int changedFrom = std::mismatch(permutation.begin(), permutation.end(), lastPermutation.begin()).first - permutation.begin();
            std::copy(permutation.begin(), permutation.end(), lastPermutation.begin());
            if(changedFrom <= nindices[2]) {
                MatType tm2;
                tm2.column(0) = latticeStruct->latticeVectors[permutation[nindices[0]]];
                tm2.column(1) = latticeStruct->latticeVectors[permutation[nindices[1]]];
                tm2.column(2) = latticeStruct->latticeVectors[permutation[nindices[2]]];
                symmetryPermutation.transformation = tm2 * tm1inverse;
                if(!symmetryPermutation.transformation.isOrthogonalMatrix()) {
                    bitmapSort(permutation.begin() + nindices[2] + 1, permutation.end(), (int)permutation.size());
                    continue;
                }
                changedFrom = 0;
            }
            int sortFrom = nindices[2];
            int invalidFrom;
            for(invalidFrom = changedFrom; invalidFrom < coordStruct.numNeighbors; invalidFrom++) {
                VecType v = symmetryPermutation.transformation * coordStruct.latticeVectors[invalidFrom];
                if(!v.equals(latticeStruct->latticeVectors[permutation[invalidFrom]])) break;
            }
            if(invalidFrom == coordStruct.numNeighbors) {
                std::copy(permutation.begin(), permutation.begin() + coordStruct.numNeighbors, symmetryPermutation.permutation.begin());
                latticeStruct->permutations.push_back(symmetryPermutation);
            }
            else {
                sortFrom = invalidFrom;
            }
            bitmapSort(permutation.begin() + sortFrom + 1, permutation.end(), (int)permutation.size());
        } while(std::next_permutation(permutation.begin(), permutation.end()));

        // Determine products of symmetry transformations.
        for(size_t s1 = 0; s1 < latticeStruct->permutations.size(); s1++) {
            for(size_t s2 = 0; s2 < latticeStruct->permutations.size(); s2++) {
                MatType product = latticeStruct->permutations[s2].transformation * latticeStruct->permutations[s1].transformation;
                for(size_t i = 0; i < latticeStruct->permutations.size(); i++) {
                    if(latticeStruct->permutations[i].transformation.equals(product)) {
                        latticeStruct->permutations[s1].product.push_back((int)i);
                        break;
                    }
                }
                for(size_t i = 0; i < latticeStruct->permutations.size(); i++) {
                    if(latticeStruct->permutations[i].transformation.equals(product)) {
                        latticeStruct->permutations[s1].inverseProduct.push_back((int)i);
                        break;
                    }
                }
            }
        }
    }
}

void ensureStructuresInitialized()
{
    static std::once_flag flag;
    std::call_once(flag, initializeListOfStructures);
}

class StructureAnalysis
{
public:
    StructureAnalysis(const Neighbors& nbrs, const double* positions, size_t count, const SimCell& cell,
                      LatticeStructureType inputCrystalType, std::vector<Matrix3> preferredCrystalOrientations,
                      int numThreads)
        : _nbrs(nbrs), _positions(positions), _count(count), _cell(cell), _inputCrystalType(inputCrystalType),
          _structureTypes(count, LATTICE_OTHER), _atomClusters(count, 0), _atomSymmetryPermutations(count, 0),
          _preferredCrystalOrientations(std::move(preferredCrystalOrientations)), _numThreads(numThreads)
    {
        ensureStructuresInitialized();
        _neighborListsSize = latticeStructures[inputCrystalType].maxNeighbors;
        _neighborLists.assign(count * _neighborListsSize, -1);
    }

    const std::vector<int32_t>& structureTypes() const { return _structureTypes; }
    const std::vector<int64_t>& atomClusters() const { return _atomClusters; }
    ClusterGraph& clusterGraph() { return _clusterGraph; }
    Cluster* atomCluster(size_t atomIndex) const { return _clusterGraph.findCluster(_atomClusters[atomIndex]); }

    int numberOfNeighbors(size_t atomIndex) const {
        const int* neighborList = &_neighborLists[atomIndex * _neighborListsSize];
        size_t count = 0;
        while(count < _neighborListsSize && neighborList[count] != -1) count++;
        return (int)count;
    }
    int getNeighbor(size_t centralAtomIndex, int neighborListIndex) const {
        return _neighborLists[centralAtomIndex * _neighborListsSize + neighborListIndex];
    }
    void setNeighbor(size_t centralAtomIndex, int neighborListIndex, int neighborAtomIndex) {
        _neighborLists[centralAtomIndex * _neighborListsSize + neighborListIndex] = neighborAtomIndex;
    }
    int findNeighbor(size_t centralAtomIndex, int neighborAtomIndex) const {
        const int* neighborList = &_neighborLists[centralAtomIndex * _neighborListsSize];
        for(size_t index = 0; index < _neighborListsSize && neighborList[index] != -1; index++)
            if(neighborList[index] == neighborAtomIndex) return (int)index;
        return -1;
    }
    const Vector3& neighborLatticeVector(size_t centralAtomIndex, int neighborIndex) const {
        int structureType = _structureTypes[centralAtomIndex];
        const LatticeStructure& latticeStructure = latticeStructures[structureType];
        int symmetryPermutationIndex = _atomSymmetryPermutations[centralAtomIndex];
        const auto& permutation = latticeStructure.permutations[symmetryPermutationIndex].permutation;
        return latticeStructure.latticeVectors[permutation[neighborIndex]];
    }
    Vector3 position(size_t i) const { return {_positions[3 * i], _positions[3 * i + 1], _positions[3 * i + 2]}; }

    void identifyStructures()
    {
        parallel_for(_count, _numThreads, [this](size_t start, size_t end, int) {
            for(size_t particleIndex = start; particleIndex < end; particleIndex++)
                determineLocalStructure(particleIndex);
        });
    }

    bool determineLocalStructure(size_t particleIndex)
    {
        int numNeighbors = _nbrs.num(particleIndex);
        int neighborIndices[SA_MAX_NEIGHBORS];
        Vector3 neighborVectors[SA_MAX_NEIGHBORS];

        int nn;
        if(_inputCrystalType == LATTICE_FCC || _inputCrystalType == LATTICE_HCP) nn = 12;
        else if(_inputCrystalType == LATTICE_BCC) nn = 14;
        else if(_inputCrystalType == LATTICE_CUBIC_DIAMOND || _inputCrystalType == LATTICE_HEX_DIAMOND) nn = 16;
        else return false;

        if(numNeighbors < nn) return false;

        double localScaling = 0;
        double localCutoff = 0;
        NeighborBondArray neighborArray;

        if(_inputCrystalType != LATTICE_CUBIC_DIAMOND && _inputCrystalType != LATTICE_HEX_DIAMOND) {
            if(_inputCrystalType == LATTICE_FCC || _inputCrystalType == LATTICE_HCP) {
                for(int n = 0; n < 12; n++) localScaling += std::sqrt(_nbrs.distSq(particleIndex, n));
                localScaling /= 12;
                localCutoff = localScaling * (1.0 + std::sqrt(2.0)) * 0.5;
            }
            else if(_inputCrystalType == LATTICE_BCC) {
                for(int n = 0; n < 8; n++) localScaling += std::sqrt(_nbrs.distSq(particleIndex, n));
                localScaling /= 8;
                localCutoff = localScaling / (std::sqrt(3.0) / 2.0) * 0.5 * (1.0 + std::sqrt(2.0));
            }
            double localCutoffSquared = localCutoff * localCutoff;

            if(numNeighbors > nn && _nbrs.distSq(particleIndex, nn) <= localCutoffSquared) return false;

            for(int ni1 = 0; ni1 < nn; ni1++) {
                neighborIndices[ni1] = (int)_nbrs.idx(particleIndex, ni1);
                neighborVectors[ni1] = _nbrs.vec(particleIndex, ni1);
                neighborArray.setNeighborBond(ni1, ni1, false);
                for(int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2, (_nbrs.vec(particleIndex, ni1) - _nbrs.vec(particleIndex, ni2)).squaredLength() <= localCutoffSquared);
            }
        }
        else {
            // Generate list of second nearest neighbors.
            int outputIndex = 4;
            for(int i = 0; i < 4; i++) {
                const Vector3 v0 = _nbrs.vec(particleIndex, i);
                neighborVectors[i] = v0;
                neighborIndices[i] = (int)_nbrs.idx(particleIndex, i);
                size_t j0 = (size_t)neighborIndices[i];
                if(_nbrs.num(j0) < 4) return false;
                for(int j = 0; j < 4; j++) {
                    Vector3 v = v0 + _nbrs.vec(j0, j);
                    if((size_t)_nbrs.idx(j0, j) == particleIndex && v.isZero()) continue;
                    if(outputIndex == 16) return false;
                    neighborIndices[outputIndex] = (int)_nbrs.idx(j0, j);
                    neighborVectors[outputIndex] = v;
                    neighborArray.setNeighborBond(i, outputIndex, true);
                    outputIndex++;
                }
                if(outputIndex != i * 3 + 7) return false;
            }

            localScaling = 0;
            for(int n = 4; n < 16; n++) localScaling += neighborVectors[n].length();
            localScaling /= 12;
            localCutoff = localScaling * double(1.2071068);
            double localCutoffSquared = localCutoff * localCutoff;

            for(int ni1 = 4; ni1 < nn; ni1++) {
                for(int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2, (neighborVectors[ni1] - neighborVectors[ni2]).squaredLength() <= localCutoffSquared);
            }
        }

        CoordinationStructureType coordinationType;
        int cnaSignatures[SA_MAX_NEIGHBORS];
        if(_inputCrystalType == LATTICE_FCC || _inputCrystalType == LATTICE_HCP) {
            int n421 = 0;
            int n422 = 0;
            for(int ni = 0; ni < nn; ni++) {
                unsigned int commonNeighbors;
                int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
                if(numCommonNeighbors != 4) break;
                CNAPairBond neighborBonds[SA_MAX_NEIGHBORS * SA_MAX_NEIGHBORS];
                int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, nn, neighborBonds);
                if(numNeighborBonds != 2) break;
                int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
                if(maxChainLength == 1) {
                    n421++;
                    cnaSignatures[ni] = 0;
                }
                else if(maxChainLength == 2) {
                    n422++;
                    cnaSignatures[ni] = 1;
                }
                else break;
            }
            if(n421 == 12) coordinationType = COORD_FCC;
            else if(n421 == 6 && n422 == 6) coordinationType = COORD_HCP;
            else return false;
        }
        else if(_inputCrystalType == LATTICE_BCC) {
            int n444 = 0;
            int n666 = 0;
            for(int ni = 0; ni < nn; ni++) {
                unsigned int commonNeighbors;
                int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
                if(numCommonNeighbors != 4 && numCommonNeighbors != 6) break;
                CNAPairBond neighborBonds[SA_MAX_NEIGHBORS * SA_MAX_NEIGHBORS];
                int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, 14, neighborBonds);
                if(numNeighborBonds != 4 && numNeighborBonds != 6) break;
                int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
                if(numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4) {
                    n444++;
                    cnaSignatures[ni] = 1;
                }
                else if(numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6) {
                    n666++;
                    cnaSignatures[ni] = 0;
                }
                else break;
            }
            if(n666 == 8 && n444 == 6) coordinationType = COORD_BCC;
            else return false;
        }
        else {
            for(int ni = 0; ni < 4; ni++) {
                cnaSignatures[ni] = 0;
                unsigned int commonNeighbors;
                if(findCommonNeighbors(neighborArray, ni, commonNeighbors) != 3) return false;
            }
            int n543 = 0;
            int n544 = 0;
            for(int ni = 4; ni < nn; ni++) {
                unsigned int commonNeighbors;
                int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
                if(numCommonNeighbors != 5) break;
                CNAPairBond neighborBonds[SA_MAX_NEIGHBORS * SA_MAX_NEIGHBORS];
                int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, nn, neighborBonds);
                if(numNeighborBonds != 4) break;
                int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
                if(maxChainLength == 3) {
                    n543++;
                    cnaSignatures[ni] = 1;
                }
                else if(maxChainLength == 4) {
                    n544++;
                    cnaSignatures[ni] = 2;
                }
                else break;
            }
            if(n543 == 12) coordinationType = COORD_CUBIC_DIAMOND;
            else if(n543 == 6 && n544 == 6) coordinationType = COORD_HEX_DIAMOND;
            else return false;
        }

        // Find the first neighbor permutation that matches the template.
        int neighborMapping[SA_MAX_NEIGHBORS];
        int previousMapping[SA_MAX_NEIGHBORS];
        for(int n = 0; n < nn; n++) {
            neighborMapping[n] = n;
            previousMapping[n] = -1;
        }
        const CoordinationStructure& coordStructure = coordinationStructures[coordinationType];
        for(;;) {
            int ni1 = 0;
            while(neighborMapping[ni1] == previousMapping[ni1]) ni1++;
            for(; ni1 < nn; ni1++) {
                int atomNeighborIndex1 = neighborMapping[ni1];
                previousMapping[ni1] = atomNeighborIndex1;
                if(cnaSignatures[atomNeighborIndex1] != coordStructure.cnaSignatures[ni1]) break;
                int ni2;
                for(ni2 = 0; ni2 < ni1; ni2++) {
                    int atomNeighborIndex2 = neighborMapping[ni2];
                    if(neighborArray.neighborBond(atomNeighborIndex1, atomNeighborIndex2) != coordStructure.neighborArray.neighborBond(ni1, ni2))
                        break;
                }
                if(ni2 != ni1) break;
            }
            if(ni1 == nn) {
                _structureTypes[particleIndex] = coordinationType;
                for(int i = 0; i < nn; i++) {
                    const Vector3& neighborVector = neighborVectors[neighborMapping[i]];
                    for(int dim = 0; dim < 3; dim++) {
                        if(_cell.hasPbc(dim)) {
                            if(std::abs(_cell.prodrow(neighborVector, dim)) >= 0.5 + epsilon_v<double>)
                                cellTooSmallError(dim);
                        }
                    }
                    setNeighbor(particleIndex, i, neighborIndices[neighborMapping[i]]);
                }
                return true;
            }
            bitmapSort(neighborMapping + ni1 + 1, neighborMapping + nn, nn);
            if(!std::next_permutation(neighborMapping, neighborMapping + nn)) return false;
        }
    }

    void buildClusters()
    {
        std::deque<int> atomsToVisit;
        for(size_t seedAtomIndex = 0; seedAtomIndex < _count; seedAtomIndex++) {
            if(_atomClusters[seedAtomIndex] != 0) continue;
            int coordStructureType = _structureTypes[seedAtomIndex];
            if(coordStructureType == COORD_OTHER) continue;

            int latticeStructureType = coordStructureType;
            Cluster* cluster = _clusterGraph.createCluster(latticeStructureType);
            cluster->atomCount = 1;
            _atomClusters[seedAtomIndex] = cluster->id;
            _atomSymmetryPermutations[seedAtomIndex] = 0;
            const CoordinationStructure& coordStructure = coordinationStructures[coordStructureType];
            const LatticeStructure& latticeStructure = latticeStructures[latticeStructureType];

            Matrix3 orientationV = Matrix3::Zero();
            Matrix3 orientationW = Matrix3::Zero();

            atomsToVisit.push_back((int)seedAtomIndex);
            do {
                int currentAtomIndex = atomsToVisit.front();
                atomsToVisit.pop_front();

                int symmetryPermutationIndex = _atomSymmetryPermutations[currentAtomIndex];
                const auto& permutation = latticeStructure.permutations[symmetryPermutationIndex].permutation;

                for(int neighborIndex = 0; neighborIndex < coordStructure.numNeighbors; neighborIndex++) {
                    int neighborAtomIndex = getNeighbor(currentAtomIndex, neighborIndex);

                    const Vector3& latticeVector = latticeStructure.latticeVectors[permutation[neighborIndex]];
                    Vector3 spatialVector = _cell.wrapVector(position(neighborAtomIndex) - position(currentAtomIndex));
                    for(int i = 0; i < 3; i++) {
                        for(int j = 0; j < 3; j++) {
                            orientationV(i, j) += (double)(latticeVector[j] * latticeVector[i]);
                            orientationW(i, j) += (double)(latticeVector[j] * spatialVector[i]);
                        }
                    }

                    if(_atomClusters[neighborAtomIndex] != 0) continue;
                    if(_structureTypes[neighborAtomIndex] != coordStructureType) continue;

                    Matrix3 tm1, tm2;
                    bool properOverlap = true;
                    for(int i = 0; i < 3; i++) {
                        int atomIndex;
                        if(i != 2) {
                            atomIndex = getNeighbor(currentAtomIndex, coordStructure.commonNeighbors[neighborIndex][i]);
                            tm1.column(i) = latticeStructure.latticeVectors[permutation[coordStructure.commonNeighbors[neighborIndex][i]]] -
                                            latticeStructure.latticeVectors[permutation[neighborIndex]];
                        }
                        else {
                            atomIndex = currentAtomIndex;
                            tm1.column(i) = -latticeStructure.latticeVectors[permutation[neighborIndex]];
                        }
                        int j = findNeighbor(neighborAtomIndex, atomIndex);
                        if(j == -1) {
                            properOverlap = false;
                            break;
                        }
                        tm2.column(i) = latticeStructure.latticeVectors[j];
                    }
                    if(!properOverlap) continue;

                    Matrix3 tm2inverse;
                    if(!tm2.inverse(tm2inverse)) continue;
                    Matrix3 transition = tm1 * tm2inverse;

                    for(size_t i = 0; i < latticeStructure.permutations.size(); i++) {
                        if(transition.equals(latticeStructure.permutations[i].transformation, CA_TRANSITION_MATRIX_EPSILON)) {
                            _atomClusters[neighborAtomIndex] = cluster->id;
                            cluster->atomCount++;
                            _atomSymmetryPermutations[neighborAtomIndex] = (int)i;
                            atomsToVisit.push_back(neighborAtomIndex);
                            break;
                        }
                    }
                }
            } while(!atomsToVisit.empty());

            cluster->orientation = (orientationW * orientationV.inverse()).toDataType<float>();

            if(latticeStructureType == _inputCrystalType && !_preferredCrystalOrientations.empty()) {
                // Pick the symmetry variant closest to one of the preferred crystal orientations.
                double smallestDeviation = std::numeric_limits<double>::max();
                const Matrix3F oldOrientation = cluster->orientation;
                for(size_t symmetryPermutationIndex = 0; symmetryPermutationIndex < latticeStructure.permutations.size(); symmetryPermutationIndex++) {
                    const Matrix3& symmetryTMatrix = latticeStructure.permutations[symmetryPermutationIndex].transformation;
                    Matrix3F newOrientation = oldOrientation * symmetryTMatrix.inverse().toDataType<float>();
                    auto scaling = std::pow(std::abs(newOrientation.determinant()), 1.0f / 3.0f);
                    for(const Matrix3& preferredOrientation : _preferredCrystalOrientations) {
                        double deviation = 0;
                        for(int i = 0; i < 3; i++)
                            for(int j = 0; j < 3; j++)
                                deviation += std::abs(newOrientation(i, j) / scaling - preferredOrientation(i, j));
                        if(deviation < smallestDeviation) {
                            smallestDeviation = deviation;
                            cluster->symmetryTransformation = (int)symmetryPermutationIndex;
                            cluster->orientation = newOrientation;
                        }
                    }
                }
            }
        }

        // Reorient atoms to align clusters with the global coordinate system.
        for(size_t atomIndex = 0; atomIndex < _count; atomIndex++) {
            int64_t clusterId = _atomClusters[atomIndex];
            if(clusterId == 0) continue;
            Cluster* cluster = _clusterGraph.findCluster(clusterId);
            if(cluster->symmetryTransformation == 0) continue;
            const LatticeStructure& latticeStructure = latticeStructures[cluster->structure];
            int oldSymmetryPermutation = _atomSymmetryPermutations[atomIndex];
            _atomSymmetryPermutations[atomIndex] =
                latticeStructure.permutations[oldSymmetryPermutation].inverseProduct[cluster->symmetryTransformation];
        }
    }

    void connectClusters()
    {
        for(size_t atomIndex = 0; atomIndex < _count; atomIndex++) {
            int64_t clusterId = _atomClusters[atomIndex];
            if(clusterId == 0) continue;
            Cluster* cluster1 = _clusterGraph.findCluster(clusterId);

            int structureType = _structureTypes[atomIndex];
            const LatticeStructure& latticeStructure = latticeStructures[structureType];
            const CoordinationStructure& coordStructure = coordinationStructures[structureType];
            int symmetryPermutationIndex = _atomSymmetryPermutations[atomIndex];
            const auto& permutation = latticeStructure.permutations[symmetryPermutationIndex].permutation;

            for(int ni = 0; ni < coordStructure.numNeighbors; ni++) {
                int neighbor = getNeighbor(atomIndex, ni);

                int64_t neighborClusterId = _atomClusters[neighbor];
                if(neighborClusterId == 0 || neighborClusterId == clusterId) {
                    if(neighborClusterId == 0) {
                        int otherNeighborListCount = numberOfNeighbors(neighbor);
                        if(otherNeighborListCount < (int)_neighborListsSize)
                            setNeighbor(neighbor, otherNeighborListCount, (int)atomIndex);
                    }
                    continue;
                }
                Cluster* cluster2 = _clusterGraph.findCluster(neighborClusterId);

                if(ClusterTransition* t = cluster1->findTransition(cluster2)) {
                    t->area++;
                    t->reverse->area++;
                    continue;
                }

                Matrix3 tm1, tm2;
                bool properOverlap = true;
                for(int i = 0; i < 3; i++) {
                    int ai;
                    if(i != 2) {
                        ai = getNeighbor(atomIndex, coordStructure.commonNeighbors[ni][i]);
                        tm1.column(i) = latticeStructure.latticeVectors[permutation[coordStructure.commonNeighbors[ni][i]]] -
                                        latticeStructure.latticeVectors[permutation[ni]];
                    }
                    else {
                        ai = (int)atomIndex;
                        tm1.column(i) = -latticeStructure.latticeVectors[permutation[ni]];
                    }
                    int j = findNeighbor(neighbor, ai);
                    if(j == -1) {
                        properOverlap = false;
                        break;
                    }
                    int neighborStructureType = _structureTypes[neighbor];
                    const LatticeStructure& neighborLatticeStructure = latticeStructures[neighborStructureType];
                    int neighborSymmetryPermutationIndex = _atomSymmetryPermutations[neighbor];
                    const auto& neighborPermutation = neighborLatticeStructure.permutations[neighborSymmetryPermutationIndex].permutation;
                    tm2.column(i) = neighborLatticeStructure.latticeVectors[neighborPermutation[j]];
                }
                if(!properOverlap) continue;

                Matrix3 tm1inverse;
                if(!tm1.inverse(tm1inverse)) continue;
                Matrix3 transition = tm2 * tm1inverse;

                if(transition.isOrthogonalMatrix()) {
                    ClusterTransition* t = _clusterGraph.createClusterTransition(cluster1, cluster2, transition.toDataType<float>());
                    t->area++;
                    t->reverse->area++;
                }
            }
        }
    }

    void formSuperClusters()
    {
        size_t oldTransitionCount = _clusterGraph.clusterTransitions.size();

        for(size_t ci = 0; ci < _clusterGraph.clusters.size(); ci++) {
            Cluster* cluster = _clusterGraph.clusters[ci];
            cluster->rank = 0;
            if(cluster->id == 0) continue;
            if(cluster->structure != _inputCrystalType) {
                for(ClusterTransition* t1 = cluster->transitions; t1 != nullptr; t1 = t1->next) {
                    if(t1->cluster2->structure == _inputCrystalType) {
                        for(ClusterTransition* t2 = t1->next; t2 != nullptr; t2 = t2->next) {
                            if(t2->cluster2->structure == _inputCrystalType && t2->cluster2 != t1->cluster2 && t2->distance == 1) {
                                const LatticeStructure& latticeStructure = latticeStructures[t2->cluster2->structure];
                                auto misorientation = t2->tm * t1->reverse->tm;
                                for(const SymmetryPermutation& symElement : latticeStructure.permutations) {
                                    if(symElement.transformation.equals(misorientation.toDataType<double>(), CA_TRANSITION_MATRIX_EPSILON)) {
                                        _clusterGraph.createClusterTransition(t1->cluster2, t2->cluster2, misorientation, 2);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        size_t newTransitionCount = _clusterGraph.clusterTransitions.size();

        auto getParentCluster = [this](Cluster* c) {
            if(c->parentTransition == nullptr) return c;
            ClusterTransition* newParentTransition = c->parentTransition;
            Cluster* parent = newParentTransition->cluster2;
            while(parent->parentTransition != nullptr) {
                newParentTransition = _clusterGraph.concatenateClusterTransitions(newParentTransition, parent->parentTransition);
                parent = parent->parentTransition->cluster2;
            }
            c->parentTransition = newParentTransition;
            return parent;
        };

        for(size_t index = oldTransitionCount; index < newTransitionCount; index++) {
            ClusterTransition* t = _clusterGraph.clusterTransitions[index];
            Cluster* parentCluster1 = getParentCluster(t->cluster1);
            Cluster* parentCluster2 = getParentCluster(t->cluster2);
            if(parentCluster1 == parentCluster2) continue;

            ClusterTransition* parentTransition = t;
            if(parentCluster2 != t->cluster2)
                parentTransition = _clusterGraph.concatenateClusterTransitions(parentTransition, t->cluster2->parentTransition);
            if(parentCluster1 != t->cluster1)
                parentTransition = _clusterGraph.concatenateClusterTransitions(t->cluster1->parentTransition->reverse, parentTransition);

            if(parentCluster1->rank > parentCluster2->rank) {
                parentCluster2->parentTransition = parentTransition->reverse;
            }
            else {
                parentCluster1->parentTransition = parentTransition;
                if(parentCluster1->rank == parentCluster2->rank) parentCluster2->rank++;
            }
        }

        for(size_t ci = 0; ci < _clusterGraph.clusters.size(); ci++) {
            Cluster* cluster = _clusterGraph.clusters[ci];
            Cluster* parentCluster = getParentCluster(cluster);
            if(parentCluster != cluster) parentCluster->atomCount += cluster->atomCount;
        }
    }

private:
    const Neighbors& _nbrs;
    const double* _positions;
    size_t _count;
    const SimCell& _cell;
    LatticeStructureType _inputCrystalType;
    std::vector<int32_t> _structureTypes;
    std::vector<int64_t> _atomClusters;
    std::vector<int> _atomSymmetryPermutations;
    std::vector<int> _neighborLists;
    size_t _neighborListsSize = 0;
    ClusterGraph _clusterGraph;
    std::vector<Matrix3> _preferredCrystalOrientations;
    int _numThreads;
};

// ---------------------------------------------------------------------------
// Elastic strain (ElasticStrainEngine.cpp)
// ---------------------------------------------------------------------------

py::dict elastic_strain(const DoubleArray& positions, const DoubleArray& cell_vectors, py::object pbc,
                        const Int64Array& nbr_index, const DoubleArray& nbr_delta, const Int32Array& nbr_count,
                        int crystal_structure, double lattice_constant, double ca_ratio, bool push_forward,
                        int num_threads)
{
    if(crystal_structure < LATTICE_FCC || crystal_structure > LATTICE_HEX_DIAMOND)
        throw std::invalid_argument("invalid input crystal structure");
    size_t n = positions.shape(0);
    SimCell cell(cell_vectors, pbc);
    Neighbors nbrs(nbr_index, nbr_delta, nbr_count);

    double latticeConstant = lattice_constant;
    double axialScaling;
    if(crystal_structure == LATTICE_FCC || crystal_structure == LATTICE_BCC || crystal_structure == LATTICE_CUBIC_DIAMOND) {
        axialScaling = 1;
    }
    else {
        latticeConstant *= std::sqrt(2.0);
        axialScaling = ca_ratio / std::sqrt(8.0 / 3.0);
    }
    std::vector<Matrix3> preferred;
    if(crystal_structure == LATTICE_FCC || crystal_structure == LATTICE_BCC || crystal_structure == LATTICE_CUBIC_DIAMOND)
        preferred.push_back(Matrix3::Identity());

    py::array_t<double> volumetric(n);
    py::array_t<double> strain({(py::ssize_t)n, (py::ssize_t)6});
    py::array_t<double> defgrad({(py::ssize_t)n, (py::ssize_t)3, (py::ssize_t)3});
    py::array_t<int32_t> structure(n);
    py::array_t<int64_t> clusters(n);
    {
        py::gil_scoped_release release;
        StructureAnalysis sa(nbrs, positions.data(), n, cell, (LatticeStructureType)crystal_structure, preferred, num_threads);
        sa.identifyStructures();
        sa.buildClusters();
        sa.connectClusters();
        sa.formSuperClusters();

        double* vol = volumetric.mutable_data();
        double* eps = strain.mutable_data();
        double* F = defgrad.mutable_data();
        parallel_for(n, num_threads, [&](size_t start, size_t end, int) {
            for(size_t particleIndex = start; particleIndex < end; particleIndex++) {
                Cluster* localCluster = sa.atomCluster(particleIndex);
                if(localCluster->id != 0) {
                    Matrix3 idealUnitCellTM(latticeConstant, 0, 0, 0, latticeConstant, 0, 0, 0, latticeConstant * axialScaling);
                    Cluster* parentCluster = nullptr;
                    if(localCluster->parentTransition != nullptr) {
                        parentCluster = localCluster->parentTransition->cluster2;
                        idealUnitCellTM = idealUnitCellTM * localCluster->parentTransition->tm.toDataType<double>();
                    }
                    else if(localCluster->structure == crystal_structure) {
                        parentCluster = localCluster;
                    }
                    if(parentCluster != nullptr) {
                        Matrix3 orientationV = Matrix3::Zero();
                        Matrix3 orientationW = Matrix3::Zero();
                        int numneigh = sa.numberOfNeighbors(particleIndex);
                        for(int nb = 0; nb < numneigh; nb++) {
                            int neighborAtomIndex = sa.getNeighbor(particleIndex, nb);
                            Vector3 latticeVector = idealUnitCellTM * sa.neighborLatticeVector(particleIndex, nb);
                            Vector3 spatialVector = cell.wrapVector(sa.position(neighborAtomIndex) - sa.position(particleIndex));
                            for(int i = 0; i < 3; i++) {
                                for(int j = 0; j < 3; j++) {
                                    orientationV(i, j) += (double)(latticeVector[j] * latticeVector[i]);
                                    orientationW(i, j) += (double)(latticeVector[j] * spatialVector[i]);
                                }
                            }
                        }
                        Matrix3 elasticF = orientationW * orientationV.inverse();
                        for(int i = 0; i < 3; i++)
                            for(int j = 0; j < 3; j++) F[particleIndex * 9 + i * 3 + j] = elasticF(i, j);

                        // Product_AtA as a symmetric tensor (XX, YY, ZZ, XY, XZ, YZ).
                        auto AtA = [](const Matrix3& A, double S[3][3]) {
                            for(int i = 0; i < 3; i++) {
                                for(int j = 0; j <= i; j++) {
                                    double b = 0;
                                    for(int k = 0; k < 3; k++) b += A(k, i) * A(k, j);
                                    S[i][j] = b;
                                }
                            }
                        };
                        double S[3][3];
                        double e[6];
                        if(!push_forward) {
                            AtA(elasticF, S);
                            e[0] = (S[0][0] - 1) * 0.5;
                            e[1] = (S[1][1] - 1) * 0.5;
                            e[2] = (S[2][2] - 1) * 0.5;
                            e[3] = S[1][0] * 0.5;
                            e[4] = S[2][0] * 0.5;
                            e[5] = S[2][1] * 0.5;
                        }
                        else {
                            Matrix3 inverseF;
                            if(!elasticF.inverse(inverseF))
                                throw std::runtime_error("Cannot compute strain tensor in spatial reference frame, because the "
                                                         "elastic deformation gradient at atom index " +
                                                         std::to_string(particleIndex + 1) + " is singular.");
                            AtA(inverseF, S);
                            e[0] = (1 - S[0][0]) * 0.5;
                            e[1] = (1 - S[1][1]) * 0.5;
                            e[2] = (1 - S[2][2]) * 0.5;
                            e[3] = -S[1][0] * 0.5;
                            e[4] = -S[2][0] * 0.5;
                            e[5] = -S[2][1] * 0.5;
                        }
                        for(int k = 0; k < 6; k++) eps[particleIndex * 6 + k] = e[k];
                        vol[particleIndex] = (e[0] + e[1] + e[2]) / 3.0;
                        continue;
                    }
                }
                vol[particleIndex] = 0;
                for(int k = 0; k < 6; k++) eps[particleIndex * 6 + k] = 0;
                for(int k = 0; k < 9; k++) F[particleIndex * 9 + k] = 0;
            }
        });
        std::copy(sa.structureTypes().begin(), sa.structureTypes().end(), structure.mutable_data());
        std::copy(sa.atomClusters().begin(), sa.atomClusters().end(), clusters.mutable_data());
    }
    py::dict res;
    res["structure"] = structure;
    res["cluster"] = clusters;
    res["volumetric_strain"] = volumetric;
    res["strain"] = strain;
    res["deformation_gradient"] = defgrad;
    return res;
}

// ---------------------------------------------------------------------------
// PTM neighbor lists in template order (PTMNeighborFinder.cpp)
// ---------------------------------------------------------------------------

// OVITO structure type ids, as written by PolyhedralTemplateMatching.
enum PtmStructure { S_OTHER = 0, S_FCC, S_HCP, S_BCC, S_ICO, S_SC, S_CUBIC_DIAMOND, S_HEX_DIAMOND, S_GRAPHENE,
                    NUM_PTM_STRUCTURES };

int ovitoToPtmType(int type)
{
    static const int map[NUM_PTM_STRUCTURES] = {PTM_MATCH_NONE, PTM_MATCH_FCC, PTM_MATCH_HCP, PTM_MATCH_BCC, PTM_MATCH_ICO,
                                                PTM_MATCH_SC, PTM_MATCH_DCUB, PTM_MATCH_DHEX, PTM_MATCH_GRAPHENE};
    return (type >= 0 && type < NUM_PTM_STRUCTURES) ? map[type] : PTM_MATCH_NONE;
}

struct PtmNeighborList {
    int count = 0;
    int templateIndex = 0;
    size_t index[PTM_MAX_INPUT_POINTS];
    Vector3 delta[PTM_MAX_INPUT_POINTS];
};

// The neighbors of a particle in the order of its PTM template ('k nearest' order for OTHER).
void ptmNeighbors(const Neighbors& nbrs, const int32_t* structures, const int64_t* correspondences, size_t particleIndex,
                  PtmNeighborList& out)
{
    int ptm_type = ovitoToPtmType(structures[particleIndex]);
    int8_t corr[PTM_MAX_INPUT_POINTS];
    size_t atom_indices[PTM_MAX_INPUT_POINTS];
    double points[PTM_MAX_INPUT_POINTS][3];
    int numNeighbors = std::min(nbrs.num(particleIndex), PTM_MAX_INPUT_POINTS - 1);
    out.templateIndex = 0;
    int num_inner = ptm_num_nbrs[ptm_type];
    int num_outer = 0;
    if(ptm_type == PTM_MATCH_NONE) {
        for(int i = 0; i < PTM_MAX_INPUT_POINTS; i++) corr[i] = (int8_t)i;
        num_inner = numNeighbors;
    }
    else {
        numNeighbors = ptm_num_nbrs[ptm_type];
        ptm_decode_correspondences(ptm_type, (uint64_t)correspondences[particleIndex], corr, &out.templateIndex);
    }
    if(ptm_type == PTM_MATCH_DCUB || ptm_type == PTM_MATCH_DHEX) {
        num_inner = 4;
        num_outer = 3;
    }
    else if(ptm_type == PTM_MATCH_GRAPHENE) {
        num_inner = 3;
        num_outer = 2;
    }
    auto fill = [&](size_t center, int offset, int num, const double* delta) {
        int available = std::min(nbrs.num(center), PTM_MAX_INPUT_POINTS - 1);
        if(available < num) return false;
        if(offset == 0) {
            atom_indices[0] = center;
            points[0][0] = points[0][1] = points[0][2] = 0;
        }
        for(int i = 0; i < num; i++) {
            int p = corr[i + 1 + offset] - 1;
            Vector3 d = nbrs.vec(center, p);
            atom_indices[i + 1 + offset] = (size_t)nbrs.idx(center, p);
            points[i + 1 + offset][0] = d[0] + delta[0];
            points[i + 1 + offset][1] = d[1] + delta[1];
            points[i + 1 + offset][2] = d[2] + delta[2];
        }
        return true;
    };
    double zero[3] = {0, 0, 0};
    if(!fill(particleIndex, 0, num_inner, zero)) {
        out.count = 0;
        return;
    }
    if(num_outer != 0) {
        for(int i = 0; i < num_inner; i++) {
            if(!fill(atom_indices[1 + i], num_inner + i * num_outer, num_outer, points[i + 1])) {
                out.count = 0;
                return;
            }
        }
    }
    out.count = numNeighbors;
    for(int i = 0; i < numNeighbors; i++) {
        out.index[i] = atom_indices[i + 1];
        out.delta[i] = Vector3(points[i + 1][0], points[i + 1][1], points[i + 1][2]);
    }
}

void initializePtm()
{
    if(ptm_initialize_global() != PTM_NO_ERROR) throw std::runtime_error("PTM initialization failed");
}

// ---------------------------------------------------------------------------
// Grain segmentation (GrainSegmentationEngine.cpp, NodePairSampling.cpp, ThresholdSelection.h)
// ---------------------------------------------------------------------------

struct Quat {
    double x = 0, y = 0, z = 0, w = 1;
    double norm() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    Quat normalized() const {
        double c = norm();
        return {x / c, y / c, z / c, w / c};
    }
};

constexpr double radiansToDegrees(double r) { return r * (180.0 / 3.14159265358979323846); }

double ptmDisorientation(int structureTypeA, int structureTypeB, const Quat& qa, const Quat& qb)
{
    double disorientation = std::numeric_limits<double>::max();
    if(structureTypeA != structureTypeB) return disorientation;
    double orientA[4] = {qa.w, qa.x, qa.y, qa.z};
    double orientB[4] = {qb.w, qb.x, qb.y, qb.z};
    int s = structureTypeA;
    if(s == S_SC || s == S_FCC || s == S_BCC || s == S_CUBIC_DIAMOND)
        disorientation = ptm::quat_disorientation_cubic(orientA, orientB);
    else if(s == S_HCP || s == S_HEX_DIAMOND || s == S_GRAPHENE)
        disorientation = ptm::quat_disorientation_hcp_conventional(orientA, orientB);
    else
        return disorientation;
    return radiansToDegrees(disorientation);
}

double ptmInterfacialDisorientation(int structureTypeA, const Quat& qa, const Quat& qb, Quat& output)
{
    double orientA[4] = {qa.w, qa.x, qa.y, qa.z};
    double orientB[4] = {qb.w, qb.x, qb.y, qb.z};
    double disorientation;
    if(structureTypeA == S_FCC || structureTypeA == S_CUBIC_DIAMOND)
        disorientation = ptm::quat_disorientation_hexagonal_to_cubic(orientA, orientB);
    else
        disorientation = ptm::quat_disorientation_cubic_to_hexagonal(orientA, orientB);
    output.w = orientB[0];
    output.x = orientB[1];
    output.y = orientB[2];
    output.z = orientB[3];
    return radiansToDegrees(disorientation);
}

class DisjointSet
{
public:
    explicit DisjointSet(size_t n) : parents(n), sizes(n) { clear(); }
    void clear() {
        std::iota(parents.begin(), parents.end(), (size_t)0);
        std::fill(sizes.begin(), sizes.end(), 1);
    }
    size_t find(size_t index) {
        size_t x = parents[index];
        while(x != parents[x]) {
            parents[x] = parents[parents[x]];
            x = parents[x];
        }
        parents[index] = x;
        return x;
    }
    size_t merge(size_t index1, size_t index2) {
        size_t parentA = find(index1);
        size_t parentB = find(index2);
        if(parentA == parentB) return parentA;
        if(sizes[parentA] < sizes[parentB]) {
            parents[parentA] = parentB;
            sizes[parentB] += sizes[parentA];
            return parentB;
        }
        parents[parentB] = parentA;
        sizes[parentA] += sizes[parentB];
        return parentA;
    }
    size_t nodesize(size_t index) const { return sizes[index]; }

private:
    std::vector<size_t> parents;
    std::vector<size_t> sizes;
};

// Weighted graph for node pair sampling. OVITO keeps each node's edges in a red-black tree
// keyed by the opposite node; a std::map gives the same ordering and lookups.
class Graph
{
public:
    std::vector<double> wnode;
    std::vector<std::map<size_t, double>> adj;
    std::set<size_t> active_nodes;

    explicit Graph(size_t num_nodes) : wnode(num_nodes, 0), adj(num_nodes) {}
    size_t num_nodes() const { return active_nodes.size(); }
    size_t next_node() const { return *active_nodes.begin(); }

    std::pair<double, size_t> nearest_neighbor(size_t a) const {
        double dmin = std::numeric_limits<double>::max();
        size_t vmin = std::numeric_limits<size_t>::max();
        for(const auto& [v, weight] : adj[a]) {
            if(v == a) throw std::runtime_error("Graph has self loops");
            double d = wnode[v] / weight;
            if(d < dmin) {
                dmin = d;
                vmin = v;
            }
            else if(d == dmin) {
                vmin = std::min(vmin, v);
            }
        }
        return {dmin * wnode[a], vmin};
    }
    void add_edge(size_t u, size_t v, double w) {
        for(size_t index : {u, v}) {
            if(adj[index].empty()) active_nodes.insert(index);
            wnode[index] += w;
        }
        adj[u].emplace(v, w);
        adj[v].emplace(u, w);
    }
    void remove_node(size_t u) { active_nodes.erase(u); }
    size_t contract_edge(size_t a, size_t b) {
        if(adj[b].size() > adj[a].size()) std::swap(a, b);
        adj[b].erase(a);
        adj[a].erase(b);
        for(const auto& [v, w] : adj[b]) {
            adj[v].erase(b);
            auto it = adj[a].find(v);
            if(it != adj[a].end()) {
                it->second += w;
                adj[v][a] += w;
            }
            else {
                adj[a].emplace(v, w);
                adj[v].emplace(a, w);
            }
        }
        std::map<size_t, double>().swap(adj[b]);
        remove_node(b);
        wnode[a] += wnode[b];
        return a;
    }
};

struct NeighborBond {
    size_t a;
    size_t b;
    double disorientation;
    double length;
};

struct DendrogramNode {
    DendrogramNode(size_t _a, size_t _b, double _distance, double _disorientation, size_t _size, Quat _orientation)
        : a(_a), b(_b), distance(_distance), disorientation(_disorientation), size(_size), orientation(_orientation) {}
    size_t a = 0;
    size_t b = 0;
    double distance = std::numeric_limits<double>::lowest();
    double disorientation = std::numeric_limits<double>::lowest();
    size_t size = 0;
    double merge_size = 0;
    Quat orientation;
};

enum MergeAlgorithm { GraphClusteringAutomatic = 0, GraphClusteringManual = 1, MinimumSpanningTree = 2 };

class InterfaceHandler
{
public:
    explicit InterfaceHandler(const std::vector<int32_t>& structures) {
        int structureCounts[NUM_PTM_STRUCTURES] = {0};
        for(auto s : structures) structureCounts[s]++;
        parent_fcc = structureCounts[S_FCC] >= structureCounts[S_HCP];
        parent_dcub = structureCounts[S_CUBIC_DIAMOND] >= structureCounts[S_HEX_DIAMOND];
        for(int i = 0; i < NUM_PTM_STRUCTURES; i++) target[i] = i;
        if(parent_fcc) target[S_HCP] = S_FCC;
        else target[S_FCC] = S_HCP;
        if(parent_dcub) target[S_HEX_DIAMOND] = S_CUBIC_DIAMOND;
        else target[S_CUBIC_DIAMOND] = S_HEX_DIAMOND;
    }
    int parent_phase(int defectStructureType) const { return target[defectStructureType]; }
    bool reorder_bond(NeighborBond& bond, const std::vector<int32_t>& adjusted) const {
        auto a = bond.a;
        auto b = bond.b;
        auto sa = adjusted[a];
        auto sb = adjusted[b];
        bool flipped = false;
        if(sa == S_FCC && sb == S_HCP) flipped |= !parent_fcc;
        else if(sa == S_HCP && sb == S_FCC) flipped |= parent_fcc;
        else if(sa == S_CUBIC_DIAMOND && sb == S_HEX_DIAMOND) flipped |= !parent_dcub;
        else if(sa == S_HEX_DIAMOND && sb == S_CUBIC_DIAMOND) flipped |= parent_dcub;
        else return false;
        if(flipped) std::swap(a, b);
        bond.a = a;
        bond.b = b;
        return true;
    }

private:
    bool parent_fcc;
    bool parent_dcub;
    int target[NUM_PTM_STRUCTURES];
};

class GrainSegmentation
{
public:
    static constexpr int MAX_DISORDERED_NEIGHBORS = 8;
    static constexpr double misorientationThreshold = 4.0;
    static constexpr size_t minPlotSize = 20;

    GrainSegmentation(const Neighbors& nbrs, const double* positions, size_t n, const SimCell& cell,
                      const int32_t* structures, const double* orientations, const int64_t* correspondences,
                      MergeAlgorithm algorithm, bool handleBoundaries)
        : _nbrs(nbrs), _positions(positions), _numParticles(n), _cell(cell), _structures(structures),
          _correspondences(correspondences), _algorithmType(algorithm), _handleBoundaries(handleBoundaries)
    {
        _orientations.resize(n);
        for(size_t i = 0; i < n; i++)
            _orientations[i] = {orientations[4 * i], orientations[4 * i + 1], orientations[4 * i + 2], orientations[4 * i + 3]};
    }

    void perform() {
        createNeighborBonds();
        rotateInterfaceAtoms();
        computeDisorientationAngles();
        determineMergeSequence();
    }

    bool isCrystallineBond(const NeighborBond& bond) const {
        auto a = _adjustedStructureTypes[bond.a];
        auto b = _adjustedStructureTypes[bond.b];
        if(a == S_OTHER || b == S_OTHER) return false;
        if(a == b) return true;
        if(!_handleBoundaries) return false;
        if(a == S_FCC && b == S_HCP) return true;
        if(a == S_HCP && b == S_FCC) return true;
        if(a == S_CUBIC_DIAMOND && b == S_HEX_DIAMOND) return true;
        if(a == S_HEX_DIAMOND && b == S_CUBIC_DIAMOND) return true;
        return false;
    }

    void createNeighborBonds() {
        PtmNeighborList list;
        for(size_t index = 0; index < _numParticles; index++) {
            ptmNeighbors(_nbrs, _structures, _correspondences, index, list);
            int numNeighbors = list.count;
            if(_structures[index] == S_OTHER) numNeighbors = std::min(numNeighbors, (int)MAX_DISORDERED_NEIGHBORS);
            for(int j = 0; j < numNeighbors; j++) {
                size_t neighborIndex = list.index[j];
                double length = std::sqrt(list.delta[j].squaredLength());
                if(index < neighborIndex)
                    _neighborBonds.push_back({index, neighborIndex, std::numeric_limits<double>::infinity(), length});
                for(int dim = 0; dim < 3; dim++) {
                    if(_cell.hasPbc(dim) && std::abs(_cell.prodrow(list.delta[j], dim)) >= 0.5 + epsilon_v<double>)
                        cellTooSmallError(dim);
                }
            }
        }
    }

    bool interface_cubic_hex(NeighborBond& bond, const InterfaceHandler& handler, Quat& output) {
        bond.disorientation = std::numeric_limits<double>::infinity();
        if(!handler.reorder_bond(bond, _adjustedStructureTypes)) return false;
        bond.disorientation = ptmInterfacialDisorientation(_adjustedStructureTypes[bond.a], _adjustedOrientations[bond.a],
                                                           _adjustedOrientations[bond.b], output);
        return bond.disorientation < misorientationThreshold;
    }

    void rotateInterfaceAtoms() {
        _adjustedStructureTypes.assign(_structures, _structures + _numParticles);
        _adjustedOrientations = _orientations;
        if(!_handleBoundaries) return;

        struct Compare {
            bool operator()(const NeighborBond& a, const NeighborBond& b) const { return a.disorientation > b.disorientation; }
        };
        std::priority_queue<NeighborBond, std::vector<NeighborBond>, Compare> pq;
        Quat rotated;
        InterfaceHandler handler(_adjustedStructureTypes);
        for(auto bond : _neighborBonds) {
            if(interface_cubic_hex(bond, handler, rotated)) pq.push({bond.a, bond.b, bond.disorientation, 0});
        }
        PtmNeighborList list;
        while(!pq.empty()) {
            auto bond = pq.top();
            pq.pop();
            if(!interface_cubic_hex(bond, handler, rotated)) continue;

            size_t index = bond.b;
            _adjustedStructureTypes[index] = handler.parent_phase(_adjustedStructureTypes[index]);
            _adjustedOrientations[index] = rotated;

            ptmNeighbors(_nbrs, _structures, _correspondences, index, list);
            for(int j = 0; j < list.count; j++) {
                bond.a = index;
                bond.b = list.index[j];
                if(interface_cubic_hex(bond, handler, rotated)) pq.push({bond.a, bond.b, bond.disorientation, 0});
            }
        }
    }

    void computeDisorientationAngles() {
        for(NeighborBond& bond : _neighborBonds)
            bond.disorientation = ptmDisorientation(_adjustedStructureTypes[bond.a], _adjustedStructureTypes[bond.b],
                                                    _adjustedOrientations[bond.a], _adjustedOrientations[bond.b]);
        std::sort(_neighborBonds.begin(), _neighborBonds.end(),
                  [](const NeighborBond& a, const NeighborBond& b) { return a.disorientation < b.disorientation; });
    }

    // Disorientation between two clusters; also accumulates qb into qa (weighted by the norms).
    static double calculate_disorientation(int structureType, Quat& qa, const Quat& qb) {
        double qa_norm = qa.norm();
        double qb_norm = qb.norm();
        double qtarget[4] = {qa.w / qa_norm, qa.x / qa_norm, qa.y / qa_norm, qa.z / qa_norm};
        double q[4] = {qb.w / qb_norm, qb.x / qb_norm, qb.y / qb_norm, qb.z / qb_norm};
        if(structureType == S_OTHER) return std::numeric_limits<double>::max();
        int type = ovitoToPtmType(structureType);
        if(structureType == S_ICO) type = 0;
        double disorientation = ptm_map_and_calculate_disorientation(type, qtarget, q);
        qa.w += q[0] * qb_norm;
        qa.x += q[1] * qb_norm;
        qa.y += q[2] * qb_norm;
        qa.z += q[3] * qb_norm;
        return disorientation;
    }

    static double calculateGraphWeight(double disorientation) {
        if(disorientation < 1e-5) disorientation = 0;
        return std::exp(-double(1) / 3 * disorientation * disorientation);
    }

    void minimum_spanning_tree_clustering(std::vector<Quat>& qsum, DisjointSet& uf) {
        for(const NeighborBond& edge : _neighborBonds) {
            if(edge.disorientation < misorientationThreshold) {
                size_t pa = uf.find(edge.a);
                size_t pb = uf.find(edge.b);
                if(pa != pb && isCrystallineBond(edge)) {
                    size_t parent = uf.merge(pa, pb);
                    size_t child = (parent == pa) ? pb : pa;
                    double disorientation = calculate_disorientation(_adjustedStructureTypes[parent], qsum[parent], qsum[child]);
                    _dendrogram.emplace_back(parent, child, edge.disorientation, disorientation, 1, qsum[parent]);
                }
            }
        }
    }

    void node_pair_sampling_clustering(Graph& graph, std::vector<Quat>& qsum) {
        double totalWeight = 1;
        std::vector<size_t> chain;
        while(graph.num_nodes()) {
            size_t node = graph.next_node();
            chain.push_back(node);
            while(!chain.empty()) {
                size_t a = chain.back();
                chain.pop_back();
                auto [d, b] = graph.nearest_neighbor(a);
                if(b == std::numeric_limits<size_t>::max()) {
                    graph.remove_node(a);
                }
                else if(!chain.empty()) {
                    size_t c = chain.back();
                    chain.pop_back();
                    if(b == c) {
                        size_t parent = graph.contract_edge(a, b);
                        size_t child = (parent == a) ? b : a;
                        double disorientation = calculate_disorientation(_adjustedStructureTypes[parent], qsum[parent], qsum[child]);
                        _dendrogram.emplace_back(parent, child, d / totalWeight, disorientation, 1, qsum[parent]);
                    }
                    else {
                        chain.push_back(c);
                        chain.push_back(a);
                        chain.push_back(b);
                    }
                }
                else {
                    chain.push_back(a);
                    chain.push_back(b);
                }
            }
        }
    }

    void determineMergeSequence() {
        bool graphClustering = _algorithmType == GraphClusteringAutomatic || _algorithmType == GraphClusteringManual;
        Graph graph(graphClustering ? _numParticles : 0);
        if(graphClustering) {
            for(const auto& edge : _neighborBonds) {
                if(isCrystallineBond(edge) && edge.disorientation < misorientationThreshold)
                    graph.add_edge(edge.a, edge.b, calculateGraphWeight(edge.disorientation));
            }
        }

        std::vector<Quat> qsum(_adjustedOrientations);
        DisjointSet uf(_numParticles);
        _dendrogram.clear();
        if(graphClustering) node_pair_sampling_clustering(graph, qsum);
        else minimum_spanning_tree_clustering(qsum, uf);

        std::sort(_dendrogram.begin(), _dendrogram.end(),
                  [](const DendrogramNode& a, const DendrogramNode& b) { return a.distance < b.distance; });

        uf.clear();
        for(DendrogramNode& node : _dendrogram) {
            size_t sa = uf.nodesize(uf.find(node.a));
            size_t sb = uf.nodesize(uf.find(node.b));
            size_t dsize = std::min(sa, sb);
            node.merge_size = 2. / (1. / sa + 1. / sb);
            uf.merge(node.a, node.b);
            node.size = dsize;
        }

        for(const DendrogramNode& node : _dendrogram) {
            if(node.size >= minPlotSize) {
                mergeDistance.push_back(graphClustering ? std::log(node.distance) : node.distance);
                mergeSize.push_back((double)node.size);
            }
        }

        if(graphClustering) {
            // ThresholdSelection::Regressor
            std::vector<double> weights, xs, ys;
            double gradient = 0, intercept = 0, mean_absolute_deviation = 0;
            std::vector<double> residuals;
            if(!_dendrogram.empty()) {
                for(const auto& node : _dendrogram) {
                    weights.push_back(node.merge_size);
                    xs.push_back(std::log(node.merge_size));
                    ys.push_back(std::log(node.distance));
                }
                residuals = least_absolute_deviations(weights, xs, ys, gradient, intercept);
                std::vector<double> sorted(residuals);
                mean_absolute_deviation = calculate_median(sorted);
            }
            double threshold = 0;
            for(const auto& node : _dendrogram) {
                double x = std::log(node.merge_size);
                double y = std::log(node.distance);
                double prediction = x * gradient + intercept;
                double residual = y - prediction;
                if(residual < 1.5 * mean_absolute_deviation) threshold = std::max(threshold, y);
            }
            suggestedMergingThreshold = threshold;
            for(size_t i = 0; i < residuals.size(); i++) {
                if(ys[i] > 0) {
                    logMergeSize.push_back(xs[i]);
                    logMergeDistance.push_back(ys[i]);
                }
            }
        }
    }

    static double calculate_median(std::vector<double>& data) {
        size_t n = data.size();
        std::sort(data.begin(), data.end());
        double median = data[n / 2];
        if(n % 2 == 0) {
            median += data[n / 2 - 1];
            median /= 2;
        }
        return median;
    }

    static void weighted_linear_regression(std::vector<double>& weights, std::vector<double>& xs, std::vector<double>& ys,
                                           double& gradient, double& intercept) {
        double wsum = 0;
        for(auto w : weights) wsum += w;
        for(size_t i = 0; i < weights.size(); i++) weights[i] /= wsum;
        double xmean = 0, ymean = 0;
        for(size_t i = 0; i < weights.size(); i++) {
            xmean += weights[i] * xs[i];
            ymean += weights[i] * ys[i];
        }
        double sum_xx = 0, sum_xy = 0;
        for(size_t i = 0; i < weights.size(); i++) {
            sum_xx += weights[i] * (xs[i] - xmean) * (xs[i] - xmean);
            sum_xy += weights[i] * (xs[i] - xmean) * (ys[i] - ymean);
        }
        gradient = sum_xy / sum_xx;
        intercept = ymean - gradient * xmean;
    }

    static std::vector<double> least_absolute_deviations(std::vector<double>& weights, std::vector<double>& xs,
                                                         std::vector<double>& ys, double& gradient, double& intercept) {
        std::vector<double> residuals(weights.size());
        std::vector<double> w(weights);
        for(int it = 0; it < 100; it++) {
            weighted_linear_regression(w, xs, ys, gradient, intercept);
            for(size_t i = 0; i < xs.size(); i++) {
                double prediction = gradient * xs[i] + intercept;
                double r = std::abs(ys[i] - prediction);
                residuals[i] = r;
                w[i] = weights[i] / std::max(1E-4, r);
            }
        }
        return residuals;
    }

    // Second stage: merge up to the threshold, build the grain list, adopt orphans, grain geometry.
    void segment(double mergingThreshold, bool adoptOrphanAtoms, size_t minGrainAtomCount, const double* masses) {
        if(_algorithmType == GraphClusteringAutomatic) mergingThreshold = suggestedMergingThreshold;
        if(_algorithmType == MinimumSpanningTree) mergingThreshold = std::log(mergingThreshold);

        std::vector<Quat> meanOrientation(_orientations);
        DisjointSet uf(_numParticles);
        for(const auto& node : _dendrogram) {
            if(std::log(node.distance) > mergingThreshold) break;
            uf.merge(node.a, node.b);
            size_t parent = uf.find(node.a);
            meanOrientation[parent] = node.orientation;
        }

        std::vector<size_t> clusterRemapping(_numParticles);
        numClusters = 1;
        std::vector<int> clusterStructureTypes;
        std::vector<Quat> clusterOrientations;
        for(size_t i = 0; i < _numParticles; i++) {
            if(uf.find(i) == i) {
                if(uf.nodesize(i) < minGrainAtomCount || _structures[i] == S_OTHER) {
                    clusterRemapping[i] = 0;
                }
                else {
                    clusterRemapping[i] = numClusters;
                    numClusters++;
                    clusterStructureTypes.push_back(_structures[i]);
                    clusterOrientations.push_back(meanOrientation[i].normalized());
                }
            }
        }

        size_t ngrains = numClusters - 1;
        grainSizes.assign(ngrains, 0);
        grainStructureTypes = clusterStructureTypes;
        grainOrientations = clusterOrientations;

        // Random grain colors with a fixed seed (std::default_random_engine of libstdc++ and
        // boost::random::uniform_real_distribution, as in OVITO's Linux builds).
        std::minstd_rand0 rng(1);
        auto uniform = [&rng]() {
            for(;;) {
                double numerator = static_cast<double>(rng() - std::minstd_rand0::min());
                double divisor = static_cast<double>(std::minstd_rand0::max() - std::minstd_rand0::min()) + 1;
                double result = numerator / divisor * (1.0 - 0.0) + 0.0;
                if(result < 1.0) return result;
            }
        };
        grainColors.resize(ngrains);
        for(size_t g = 0; g < ngrains; g++) {
            // Arguments of OVITO's ColorG::fromHSV(...) call, evaluated right to left by GCC.
            float v = 1.0f - static_cast<float>(uniform()) * 0.5f;
            float s = 1.0f - static_cast<float>(uniform()) * 0.8f;
            float h = static_cast<float>(uniform());
            grainColors[g] = fromHSV(h, s, v);
        }

        for(size_t i = 0; i < _numParticles; i++) clusterRemapping[i] = clusterRemapping[uf.find(i)];
        atomClusters.assign(_numParticles, 0);
        for(size_t i = 0; i < _numParticles; i++) {
            size_t gid = clusterRemapping[i];
            atomClusters[i] = (int64_t)gid;
            if(gid != 0) grainSizes[gid - 1]++;
        }

        if(numClusters > 1) {
            std::vector<size_t> mapping(ngrains);
            std::iota(mapping.begin(), mapping.end(), size_t(0));
            std::sort(mapping.begin(), mapping.end(), [this](size_t a, size_t b) { return grainSizes[a] > grainSizes[b]; });
            auto reorder = [&mapping](auto& v) {
                auto copy = v;
                for(size_t i = 0; i < mapping.size(); i++) v[i] = copy[mapping[i]];
            };
            reorder(grainSizes);
            reorder(grainStructureTypes);
            reorder(grainOrientations);
            std::vector<size_t> inverseMapping(numClusters);
            inverseMapping[0] = 0;
            for(size_t i = 1; i < numClusters; i++) inverseMapping[mapping[i - 1] + 1] = i;
            for(auto& id : atomClusters) id = (int64_t)inverseMapping[id];
            if(adoptOrphanAtoms) mergeOrphanAtoms();
        }
        computeGrainGeometry(masses);
    }

    static std::array<float, 3> fromHSV(float hue, float saturation, float value) {
        if(saturation == 0) return {value, value, value};
        if(hue >= 1.0f || hue < 0.0f) hue = 0;
        hue *= 6.0f;
        int i = (int)std::floor(hue);
        float f = hue - (float)i;
        float p = value * (1.0f - saturation);
        float q = value * (1.0f - (saturation * f));
        float t = value * (1.0f - (saturation * (1.0f - f)));
        switch(i) {
            case 0: return {value, t, p};
            case 1: return {q, value, p};
            case 2: return {p, value, t};
            case 3: return {p, q, value};
            case 4: return {t, p, value};
            case 5: return {value, p, q};
            default: return {value, value, value};
        }
    }

    void mergeOrphanAtoms() {
        std::vector<NeighborBond> noncrystallineBonds;
        for(auto nb : _neighborBonds) {
            if(atomClusters[nb.a] == 0 || atomClusters[nb.b] == 0) {
                noncrystallineBonds.push_back(nb);
                std::swap(nb.a, nb.b);
                noncrystallineBonds.push_back(nb);
            }
        }
        auto byA = [](const NeighborBond& a, const NeighborBond& b) { return a.a < b.a; };
        std::sort(noncrystallineBonds.begin(), noncrystallineBonds.end(), byA);

        struct PQNode {
            int64_t cluster;
            size_t particleIndex;
            double length;
        };
        struct PQCompareLength {
            bool operator()(const PQNode& a, const PQNode& b) const { return a.length > b.length; }
        };
        std::priority_queue<PQNode, std::vector<PQNode>, PQCompareLength> pq;
        for(const auto& bond : _neighborBonds) {
            auto clusterA = atomClusters[bond.a];
            auto clusterB = atomClusters[bond.b];
            if(clusterA != 0 && clusterB == 0) pq.push({clusterA, bond.b, bond.length});
            else if(clusterA == 0 && clusterB != 0) pq.push({clusterB, bond.a, bond.length});
        }
        while(!pq.empty()) {
            auto node = pq.top();
            pq.pop();
            if(atomClusters[node.particleIndex] != 0) continue;
            atomClusters[node.particleIndex] = node.cluster;
            grainSizes[node.cluster - 1]++;
            auto range = std::equal_range(noncrystallineBonds.begin(), noncrystallineBonds.end(),
                                          NeighborBond{node.particleIndex, 0, 0, 0}, byA);
            for(auto it = range.first; it != range.second; ++it) {
                auto neighborIndex = it->b;
                if(atomClusters[neighborIndex] != 0) continue;
                pq.push({node.cluster, neighborIndex, node.length + it->length});
            }
        }
    }

    // Union-find that also tracks each atom's periodic image relative to the root of its tree.
    struct PeriodicImageUnionFind {
        std::vector<size_t> parents;
        std::vector<std::array<int, 3>> images;
        std::vector<uint8_t> ranks;
        explicit PeriodicImageUnionFind(size_t n) : parents(n), images(n, {0, 0, 0}), ranks(n, 0) {
            std::iota(parents.begin(), parents.end(), size_t(0));
        }
        std::pair<size_t, std::array<int, 3>> find(size_t index) {
            size_t root = index;
            std::array<int, 3> image = {0, 0, 0};
            while(parents[root] != root) {
                for(int d = 0; d < 3; d++) image[d] += images[root][d];
                root = parents[root];
            }
            size_t current = index;
            std::array<int, 3> currentImage = image;
            while(parents[current] != root) {
                size_t next = parents[current];
                std::array<int, 3> nextImage;
                for(int d = 0; d < 3; d++) nextImage[d] = currentImage[d] - images[current][d];
                parents[current] = root;
                images[current] = currentImage;
                current = next;
                currentImage = nextImage;
            }
            return {root, image};
        }
        void merge(size_t a, size_t b, const std::array<int, 3>& image) {
            auto [rootA, imageA] = find(a);
            auto [rootB, imageB] = find(b);
            if(rootA == rootB) return;
            if(ranks[rootA] < ranks[rootB]) {
                parents[rootA] = rootB;
                for(int d = 0; d < 3; d++) images[rootA][d] = imageB[d] - imageA[d] - image[d];
            }
            else {
                parents[rootB] = rootA;
                for(int d = 0; d < 3; d++) images[rootB][d] = imageA[d] - imageB[d] + image[d];
                if(ranks[rootA] == ranks[rootB]) ranks[rootA]++;
            }
        }
    };

    Vector3 position(size_t i) const { return {_positions[3 * i], _positions[3 * i + 1], _positions[3 * i + 2]}; }

    void computeGrainGeometry(const double* masses) {
        size_t ngrains = numClusters - 1;
        grainCenters.assign(ngrains, Vector3());
        grainRadii.assign(ngrains, 0);
        grainRadiiOfGyration.assign(ngrains, 0);
        if(numClusters <= 1) return;

        PeriodicImageUnionFind imageMapping(_numParticles);
        for(const NeighborBond& bond : _neighborBonds) {
            if(atomClusters[bond.a] == 0 || atomClusters[bond.a] != atomClusters[bond.b]) continue;
            Vector3 delta = position(bond.b) - position(bond.a);
            std::array<int, 3> image = {0, 0, 0};
            for(int dim = 0; dim < 3; dim++)
                if(_cell.hasPbc(dim)) image[dim] = -static_cast<int32_t>(std::floor(_cell.prodrow(delta, dim) + 0.5));
            imageMapping.merge(bond.a, bond.b, image);
        }
        auto unwrapShift = [this](const std::array<int, 3>& image) {
            return _cell.m * Vector3(image[0], image[1], image[2]);
        };

        std::vector<Vector3> weightedSums(ngrains), sums(ngrains);
        std::vector<double> totalWeights(ngrains, 0);
        std::vector<size_t> counts(ngrains, 0);
        for(size_t i = 0; i < _numParticles; i++) {
            int64_t grain = atomClusters[i];
            if(grain == 0) continue;
            auto image = imageMapping.find(i).second;
            Vector3 delta = position(i);
            if(image[0] || image[1] || image[2]) delta += unwrapShift(image);
            double weight = masses ? masses[i] : 1.0;
            weightedSums[grain - 1] += weight * delta;
            totalWeights[grain - 1] += weight;
            sums[grain - 1] += delta;
            counts[grain - 1]++;
        }
        std::vector<Vector3> unwrappedCenters(ngrains);
        for(size_t g = 0; g < ngrains; g++) {
            if(totalWeights[g] > 0) unwrappedCenters[g] = weightedSums[g] / totalWeights[g];
            else if(counts[g] != 0) unwrappedCenters[g] = sums[g] / (double)counts[g];
            grainCenters[g] = counts[g] != 0 ? wrapPoint(unwrappedCenters[g]) : Vector3();
        }
        std::vector<double> weightedSq(ngrains, 0), sq(ngrains, 0), maxSq(ngrains, 0);
        for(size_t i = 0; i < _numParticles; i++) {
            int64_t grain = atomClusters[i];
            if(grain == 0) continue;
            auto image = imageMapping.find(i).second;
            Vector3 delta = position(i) - unwrappedCenters[grain - 1];
            if(image[0] || image[1] || image[2]) delta += unwrapShift(image);
            double weight = masses ? masses[i] : 1.0;
            weightedSq[grain - 1] += weight * delta.squaredLength();
            sq[grain - 1] += delta.squaredLength();
            maxSq[grain - 1] = std::max(maxSq[grain - 1], delta.squaredLength());
        }
        for(size_t g = 0; g < ngrains; g++) {
            grainRadii[g] = std::sqrt(maxSq[g]);
            if(totalWeights[g] > 0) grainRadiiOfGyration[g] = std::sqrt(weightedSq[g] / totalWeights[g]);
            else if(counts[g] != 0) grainRadiiOfGyration[g] = std::sqrt(sq[g] / double(counts[g]));
        }
    }

    Vector3 wrapPoint(const Vector3& p) const {
        Vector3 pout = p;
        Vector3 rel = p - _cell.origin;
        for(int dim = 0; dim < 3; dim++) {
            if(_cell.hasPbc(dim)) {
                if(double s = std::floor(_cell.prodrow(rel, dim))) pout -= s * _cell.m.column(dim);
            }
        }
        return pout;
    }

    const std::vector<NeighborBond>& neighborBonds() const { return _neighborBonds; }

    // Outputs.
    double suggestedMergingThreshold = 0;
    std::vector<double> mergeDistance, mergeSize, logMergeDistance, logMergeSize;
    size_t numClusters = 1;
    std::vector<int64_t> atomClusters;
    std::vector<int64_t> grainSizes;
    std::vector<int> grainStructureTypes;
    std::vector<Quat> grainOrientations;
    std::vector<std::array<float, 3>> grainColors;
    std::vector<Vector3> grainCenters;
    std::vector<double> grainRadii, grainRadiiOfGyration;

private:
    const Neighbors& _nbrs;
    const double* _positions;
    size_t _numParticles;
    const SimCell& _cell;
    const int32_t* _structures;
    const int64_t* _correspondences;
    std::vector<Quat> _orientations;
    MergeAlgorithm _algorithmType;
    bool _handleBoundaries;
    std::vector<int32_t> _adjustedStructureTypes;
    std::vector<Quat> _adjustedOrientations;
    std::vector<NeighborBond> _neighborBonds;
    std::vector<DendrogramNode> _dendrogram;
};

template <typename T>
py::array_t<T> toArray(const std::vector<T>& v)
{
    py::array_t<T> a(v.size());
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

py::dict grain_segmentation(const DoubleArray& positions, const DoubleArray& cell_vectors, const DoubleArray& cell_origin,
                            py::object pbc, const Int32Array& structure, const DoubleArray& orientation,
                            const Int64Array& correspondences, const Int64Array& nbr_index, const DoubleArray& nbr_delta,
                            const Int32Array& nbr_count, int algorithm, bool handle_coherent_interfaces,
                            double merging_threshold, int64_t min_grain_size, bool adopt_orphans,
                            std::optional<DoubleArray> masses, bool output_bonds)
{
    if(algorithm < 0 || algorithm > 2) throw std::invalid_argument("invalid merge algorithm");
    initializePtm();
    size_t n = positions.shape(0);
    SimCell cell(cell_vectors, pbc);
    auto o = cell_origin.unchecked<1>();
    cell.origin = Vector3(o(0), o(1), o(2));
    Neighbors nbrs(nbr_index, nbr_delta, nbr_count);
    std::unique_ptr<GrainSegmentation> gs;
    {
        py::gil_scoped_release release;
        gs = std::make_unique<GrainSegmentation>(nbrs, positions.data(), n, cell, structure.data(), orientation.data(),
                                                 correspondences.data(), (MergeAlgorithm)algorithm, handle_coherent_interfaces);
        gs->perform();
        gs->segment(merging_threshold, adopt_orphans, (size_t)std::max<int64_t>(min_grain_size, 0),
                    masses ? masses->data() : nullptr);
    }
    size_t ng = gs->numClusters - 1;
    py::array_t<float> colors({(py::ssize_t)ng, (py::ssize_t)3});
    py::array_t<double> orient({(py::ssize_t)ng, (py::ssize_t)4});
    py::array_t<double> centers({(py::ssize_t)ng, (py::ssize_t)3});
    for(size_t g = 0; g < ng; g++) {
        for(int k = 0; k < 3; k++) colors.mutable_at(g, k) = gs->grainColors[g][k];
        const Quat& q = gs->grainOrientations[g];
        orient.mutable_at(g, 0) = q.x;
        orient.mutable_at(g, 1) = q.y;
        orient.mutable_at(g, 2) = q.z;
        orient.mutable_at(g, 3) = q.w;
        for(int k = 0; k < 3; k++) centers.mutable_at(g, k) = gs->grainCenters[g][k];
    }
    py::dict res;
    res["grain"] = toArray(gs->atomClusters);
    res["grain_size"] = toArray(gs->grainSizes);
    res["grain_structure"] = toArray(std::vector<int32_t>(gs->grainStructureTypes.begin(), gs->grainStructureTypes.end()));
    res["grain_color"] = colors;
    res["grain_orientation"] = orient;
    res["grain_center"] = centers;
    res["grain_radius"] = toArray(gs->grainRadii);
    res["grain_radius_of_gyration"] = toArray(gs->grainRadiiOfGyration);
    res["merge_distance"] = toArray(gs->mergeDistance);
    res["merge_size"] = toArray(gs->mergeSize);
    res["log_merge_distance"] = toArray(gs->logMergeDistance);
    res["log_merge_size"] = toArray(gs->logMergeSize);
    res["suggested_threshold"] = gs->suggestedMergingThreshold;
    if(output_bonds) {
        std::vector<int64_t> topo;
        std::vector<double> dis;
        for(const auto& edge : gs->neighborBonds()) {
            if(gs->isCrystallineBond(edge)) {
                topo.push_back((int64_t)edge.a);
                topo.push_back((int64_t)edge.b);
                dis.push_back(edge.disorientation);
            }
        }
        py::array_t<int64_t> t({(py::ssize_t)dis.size(), (py::ssize_t)2});
        std::copy(topo.begin(), topo.end(), t.mutable_data());
        res["bond_topology"] = t;
        res["bond_disorientation"] = toArray(dis);
    }
    return res;
}

// Neighbors of every particle in PTM template order, for analyses built on PTM output
// (the counterpart of OVITO's PTMNeighborFinder). Missing entries are -1.
py::tuple ptm_ordered_neighbors(const Int32Array& structure, const Int64Array& correspondences,
                                const Int64Array& nbr_index, const DoubleArray& nbr_delta, const Int32Array& nbr_count,
                                int max_neighbors)
{
    initializePtm();
    size_t n = structure.shape(0);
    Neighbors nbrs(nbr_index, nbr_delta, nbr_count);
    py::array_t<int64_t> index({(py::ssize_t)n, (py::ssize_t)max_neighbors});
    py::array_t<double> delta({(py::ssize_t)n, (py::ssize_t)max_neighbors, (py::ssize_t)3});
    py::array_t<int32_t> templ(n);
    int64_t* pi = index.mutable_data();
    double* pd = delta.mutable_data();
    int32_t* pt = templ.mutable_data();
    {
        py::gil_scoped_release release;
        PtmNeighborList list;
        for(size_t i = 0; i < n; i++) {
            ptmNeighbors(nbrs, structure.data(), correspondences.data(), i, list);
            pt[i] = list.templateIndex;
            for(int j = 0; j < max_neighbors; j++) {
                bool ok = j < list.count;
                pi[i * max_neighbors + j] = ok ? (int64_t)list.index[j] : -1;
                for(int k = 0; k < 3; k++) pd[(i * max_neighbors + j) * 3 + k] = ok ? list.delta[j][k] : 0.0;
            }
        }
    }
    return py::make_tuple(index, delta, templ);
}

// Ideal neighbor vectors of a PTM template (OVITO structure type id, template index).
py::array_t<double> ptm_template(int structure_type, int template_index)
{
    int t = ovitoToPtmType(structure_type);
    if(t == PTM_MATCH_NONE) throw std::invalid_argument("no template for this structure type");
    const ptm::refdata_t* ref = ptm::refdata[t];
    int num = ref->num_nbrs;
    py::array_t<double> out({(py::ssize_t)num, (py::ssize_t)3});
    for(int i = 0; i < num; i++)
        for(int k = 0; k < 3; k++) out.mutable_at(i, k) = ref->points[template_index][i + 1][k];
    return out;
}

}  // namespace

void register_crystal(py::module_& m)
{
    m.def("grain_segmentation", &grain_segmentation, py::arg("positions"), py::arg("cell_vectors"), py::arg("cell_origin"),
          py::arg("pbc"), py::arg("structure"), py::arg("orientation"), py::arg("correspondences"), py::arg("nbr_index"),
          py::arg("nbr_delta"), py::arg("nbr_count"), py::arg("algorithm"), py::arg("handle_coherent_interfaces"),
          py::arg("merging_threshold"), py::arg("min_grain_size"), py::arg("adopt_orphans"), py::arg("masses"),
          py::arg("output_bonds"));
    m.def("ptm_ordered_neighbors", &ptm_ordered_neighbors, py::arg("structure"), py::arg("correspondences"),
          py::arg("nbr_index"), py::arg("nbr_delta"), py::arg("nbr_count"), py::arg("max_neighbors"));
    m.def("ptm_template", &ptm_template, py::arg("structure_type"), py::arg("template_index"));
    m.def("elastic_strain", &elastic_strain, py::arg("positions"), py::arg("cell_vectors"), py::arg("pbc"),
          py::arg("nbr_index"), py::arg("nbr_delta"), py::arg("nbr_count"), py::arg("crystal_structure"),
          py::arg("lattice_constant"), py::arg("ca_ratio"), py::arg("push_forward"), py::arg("num_threads") = 0);
}
