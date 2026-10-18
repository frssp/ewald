// Common neighbor analysis kernels (adaptive, interval and fixed-cutoff modes).
//
// Ported from OVITO 3.16 CommonNeighborAnalysisModifier.cpp:
//
//   Copyright 2026 OVITO GmbH, Germany
//   OVITO is free software; you can redistribute it and/or modify it either under the
//   terms of the GNU General Public License version 3 as published by the Free Software
//   Foundation (the "GPL") or, at your option, under the terms of the MIT License.
//   If you do not alter this notice, a recipient may use your version of this
//   file under either the GPL or the MIT License.
//
// This port is used under the MIT License (see LICENSE.OVITO-MIT.txt).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "parallel.h"

namespace py = pybind11;

namespace {

enum StructureType { OTHER = 0, FCC, HCP, BCC, ICO };
constexpr int MAX_NEIGHBORS = 14;
using CNAPairBond = unsigned int;

struct Vec3 {
    double x, y, z;
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    double squaredLength() const { return x * x + y * y + z * z; }
};

struct NeighborBondArray {
    unsigned int neighborArray[32];
    NeighborBondArray() { std::memset(neighborArray, 0, sizeof(neighborArray)); }
    void setNeighborBond(int a, int b, bool bonded) {
        if(bonded) { neighborArray[a] |= (1u << b); neighborArray[b] |= (1u << a); }
        else { neighborArray[a] &= ~(1u << b); neighborArray[b] &= ~(1u << a); }
    }
};

inline int popcount(unsigned int v) { int c = 0; for(; v; c++) v &= v - 1; return c; }
inline int ctz(unsigned int v) { int c = 0; while(!(v & 1u)) { v >>= 1; c++; } return c; }

int findCommonNeighbors(const NeighborBondArray& na, int neighborIndex, unsigned int& commonNeighbors)
{
    commonNeighbors = na.neighborArray[neighborIndex];
    return popcount(commonNeighbors);
}

int findNeighborBonds(const NeighborBondArray& na, unsigned int commonNeighbors, int numNeighbors, CNAPairBond* neighborBonds)
{
    int numBonds = 0;
    unsigned int nib[32];
    int nibn = 0;
    unsigned int ni1b = 1;
    for(int ni1 = 0; ni1 < numNeighbors; ni1++, ni1b <<= 1) {
        if(commonNeighbors & ni1b) {
            unsigned int b = commonNeighbors & na.neighborArray[ni1];
            for(int n = 0; n < nibn; n++)
                if(b & nib[n]) neighborBonds[numBonds++] = ni1b | nib[n];
            nib[nibn++] = ni1b;
        }
    }
    return numBonds;
}

int getAdjacentBonds(unsigned int atom, CNAPairBond* bondsToProcess, int& numBonds, unsigned int& atomsToProcess, unsigned int& atomsProcessed)
{
    int adjacentBonds = 0;
    for(int b = numBonds - 1; b >= 0; b--) {
        if(atom & *bondsToProcess) {
            ++adjacentBonds;
            atomsToProcess |= *bondsToProcess & (~atomsProcessed);
            std::memmove(bondsToProcess, bondsToProcess + 1, sizeof(CNAPairBond) * b);
            numBonds--;
        }
        else ++bondsToProcess;
    }
    return adjacentBonds;
}

int calcMaxChainLength(CNAPairBond* neighborBonds, int numBonds)
{
    int maxChainLength = 0;
    while(numBonds) {
        numBonds--;
        unsigned int atomsToProcess = neighborBonds[numBonds];
        unsigned int atomsProcessed = 0;
        int clusterSize = 1;
        do {
            unsigned int nextAtom = 1u << ctz(atomsToProcess);
            atomsProcessed |= nextAtom;
            atomsToProcess &= ~nextAtom;
            clusterSize += getAdjacentBonds(nextAtom, neighborBonds, numBonds, atomsToProcess, atomsProcessed);
        } while(atomsToProcess);
        maxChainLength = std::max(clusterSize, maxChainLength);
    }
    return maxChainLength;
}

StructureType analyzeSmallSignature(const NeighborBondArray& na)
{
    constexpr int nn = 12;
    int n421 = 0, n422 = 0, n555 = 0;
    for(int ni = 0; ni < nn; ni++) {
        unsigned int commonNeighbors;
        int numCommonNeighbors = findCommonNeighbors(na, ni, commonNeighbors);
        if(numCommonNeighbors != 4 && numCommonNeighbors != 5) break;
        CNAPairBond neighborBonds[MAX_NEIGHBORS * MAX_NEIGHBORS];
        int numNeighborBonds = findNeighborBonds(na, commonNeighbors, nn, neighborBonds);
        if(numNeighborBonds != 2 && numNeighborBonds != 5) break;
        int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
        if(numCommonNeighbors == 4 && numNeighborBonds == 2) {
            if(maxChainLength == 1) n421++;
            else if(maxChainLength == 2) n422++;
            else break;
        }
        else if(numCommonNeighbors == 5 && numNeighborBonds == 5 && maxChainLength == 5) n555++;
        else break;
    }
    if(n421 == 12) return FCC;
    if(n421 == 6 && n422 == 6) return HCP;
    if(n555 == 12) return ICO;
    return OTHER;
}

StructureType analyzeLargeSignature(const NeighborBondArray& na)
{
    constexpr int nn = 14;
    int n444 = 0, n666 = 0;
    for(int ni = 0; ni < nn; ni++) {
        unsigned int commonNeighbors;
        int numCommonNeighbors = findCommonNeighbors(na, ni, commonNeighbors);
        if(numCommonNeighbors != 4 && numCommonNeighbors != 6) break;
        CNAPairBond neighborBonds[MAX_NEIGHBORS * MAX_NEIGHBORS];
        int numNeighborBonds = findNeighborBonds(na, commonNeighbors, nn, neighborBonds);
        if(numNeighborBonds != 4 && numNeighborBonds != 6) break;
        int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
        if(numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4) n444++;
        else if(numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6) n666++;
        else break;
    }
    if(n666 == 8 && n444 == 6) return BCC;
    return OTHER;
}

struct Enabled {
    bool fcc, hcp, bcc, ico;
    bool small() const { return fcc || hcp || ico; }
    bool allows(StructureType t) const {
        switch(t) { case FCC: return fcc; case HCP: return hcp; case BCC: return bcc; case ICO: return ico; default: return true; }
    }
};

// OVITO's analyzeSmallSignature() tests all three small types; filter disabled ones afterwards.
StructureType filterType(StructureType t, const Enabled& en) { return en.allows(t) ? t : OTHER; }

StructureType determineAdaptive(const Vec3* v, const double* d2, int numNeighbors, const Enabled& en)
{
    if(en.small()) {
        constexpr int nn = 12;
        if(numNeighbors < nn) return OTHER;
        double localScaling = 0;
        for(int n = 0; n < nn; n++) localScaling += std::sqrt(d2[n]);
        double localCutoff = localScaling / nn * (1.0 + std::sqrt(2.0)) * 0.5;
        double localCutoffSquared = localCutoff * localCutoff;
        NeighborBondArray na;
        for(int a = 0; a < nn; a++)
            for(int b = a + 1; b < nn; b++)
                na.setNeighborBond(a, b, (v[a] - v[b]).squaredLength() <= localCutoffSquared);
        auto type = filterType(analyzeSmallSignature(na), en);
        if(type != OTHER) return type;
    }
    if(en.bcc) {
        constexpr int nn = 14;
        if(numNeighbors < nn) return OTHER;
        double localScaling = 0;
        for(int n = 0; n < 8; n++) localScaling += std::sqrt(d2[n] / (3.0 / 4.0));
        for(int n = 8; n < 14; n++) localScaling += std::sqrt(d2[n]);
        double localCutoff = localScaling / nn * 1.207;
        double localCutoffSquared = localCutoff * localCutoff;
        NeighborBondArray na;
        for(int a = 0; a < nn; a++)
            for(int b = a + 1; b < nn; b++)
                na.setNeighborBond(a, b, (v[a] - v[b]).squaredLength() <= localCutoffSquared);
        auto type = analyzeLargeSignature(na);
        if(type != OTHER) return type;
    }
    return OTHER;
}

enum GraphEdgeType { NONE = 0, SHORT = 1, LONG = 2 };

struct GraphEdge {
    GraphEdge(int i_, int j_, double l, int t) : i(i_), j(j_), length(l), edgeType(t) {}
    int i = 0, j = 0;
    double length = 0;
    int edgeType = 0;
    GraphEdge* nextShort = nullptr;
    GraphEdge* nextLong = nullptr;
};

struct EdgeIterator {
    EdgeIterator(int nn, const Vec3* nv, double shortThreshold, double longThreshold) {
        if(nn < 12) shortThreshold = 0;
        if(nn < 14) longThreshold = 0;
        GraphEdge shortEnd(-1, -1, std::numeric_limits<double>::infinity(), SHORT);
        GraphEdge longEnd(-1, -1, std::numeric_limits<double>::infinity(), LONG);
        for(int i = 0; i < nn; i++) {
            for(int j = i + 1; j < nn; j++) {
                double length = std::sqrt((nv[i] - nv[j]).squaredLength());
                int edgeType = NONE;
                if(i < 12 && j < 12 && length < shortThreshold) edgeType |= SHORT;
                if(length < longThreshold) edgeType |= LONG;
                if(edgeType == NONE) {
                    if(length < longEnd.length) longEnd = GraphEdge(i, j, length, LONG);
                    else if(length < shortEnd.length) shortEnd = GraphEdge(i, j, length, SHORT);
                }
                else edges.push_back(GraphEdge(i, j, length, edgeType));
            }
        }
        std::sort(edges.begin(), edges.end(), [](const GraphEdge& a, const GraphEdge& b) { return a.length < b.length; });
        if(shortEnd.i != -1) edges.push_back(shortEnd);
        if(longEnd.i != -1) edges.push_back(longEnd);
        for(int i = (int)edges.size() - 1; i >= 0; i--) {
            if(edges[i].edgeType & SHORT) { edges[i].nextShort = nextShort; nextShort = &edges[i]; }
            if(edges[i].edgeType & LONG) { edges[i].nextLong = nextLong; nextLong = &edges[i]; }
        }
    }
    std::vector<GraphEdge> edges;
    GraphEdge* nextLong = nullptr;
    GraphEdge* nextShort = nullptr;
};

StructureType determineInterval(const Vec3* v, const double* d2, int numNeighbors, const Enabled& en)
{
    bool analyzeShort = numNeighbors >= 12 && en.small();
    bool analyzeLong = numNeighbors >= 14 && en.bcc;
    if(analyzeLong) numNeighbors = 14;
    else if(analyzeShort) numNeighbors = 12;
    else return OTHER;
    double neighborLengths[MAX_NEIGHBORS];
    for(int i = 0; i < numNeighbors; i++) neighborLengths[i] = std::sqrt(d2[i]);
    const double x = 2.0 / 3.0;
    const double fraction = ((1 - x) * 1 + x * std::sqrt(2.0));
    double shortLengthThreshold = 0, longLengthThreshold = 0;
    if(analyzeShort) {
        double s = 0;
        for(int n = 0; n < 12; n++) s += neighborLengths[n];
        shortLengthThreshold = fraction * s / 12;
    }
    if(analyzeLong) {
        double s = 0;
        for(int n = 0; n < 8; n++) s += neighborLengths[n] / std::sqrt(3.0 / 4.0);
        for(int n = 8; n < 14; n++) s += neighborLengths[n];
        longLengthThreshold = fraction * s / 14;
    }
    double bestIntervalWidth = 0;
    StructureType bestType = OTHER;
    EdgeIterator it(numNeighbors, v, shortLengthThreshold, longLengthThreshold);
    if(analyzeShort) {
        constexpr int nn = 12;
        int n4 = 0, n5 = 0;
        int coordinations[nn] = {0};
        NeighborBondArray na;
        GraphEdge* edge = it.nextShort;
        GraphEdge* next = edge != nullptr ? edge->nextShort : nullptr;
        while(next != nullptr) {
            coordinations[edge->i]++;
            coordinations[edge->j]++;
            na.setNeighborBond(edge->i, edge->j, true);
            if(coordinations[edge->i] == 4) n4++;
            if(coordinations[edge->i] == 5) { n4--; n5++; }
            if(coordinations[edge->i] > 5) break;
            if(coordinations[edge->j] == 4) n4++;
            if(coordinations[edge->j] == 5) { n4--; n5++; }
            if(coordinations[edge->j] > 5) break;
            if(n4 == nn || n5 == nn) {
                auto type = filterType(analyzeSmallSignature(na), en);
                if(type != OTHER) {
                    double intervalWidth = next->length - edge->length;
                    if(intervalWidth > bestIntervalWidth) { bestIntervalWidth = intervalWidth; bestType = type; }
                }
            }
            edge = next;
            next = next->nextShort;
        }
    }
    if(analyzeLong) {
        constexpr int nn = 14;
        int n4 = 0, n6 = 0;
        int coordinations[nn] = {0};
        NeighborBondArray na;
        GraphEdge* edge = it.nextLong;
        GraphEdge* next = edge != nullptr ? edge->nextLong : nullptr;
        while(next != nullptr) {
            coordinations[edge->i]++;
            coordinations[edge->j]++;
            na.setNeighborBond(edge->i, edge->j, true);
            if(coordinations[edge->i] == 4) n4++;
            if(coordinations[edge->i] == 5) n4--;
            if(coordinations[edge->i] == 6) n6++;
            if(coordinations[edge->i] > 6) break;
            if(coordinations[edge->j] == 4) n4++;
            if(coordinations[edge->j] == 5) n4--;
            if(coordinations[edge->j] == 6) n6++;
            if(coordinations[edge->j] > 6) break;
            if(n4 == 6 && n6 == 8) {
                auto type = analyzeLargeSignature(na);
                if(type != OTHER) {
                    double intervalWidth = next->length - edge->length;
                    if(intervalWidth > bestIntervalWidth) { bestIntervalWidth = intervalWidth; bestType = type; }
                }
            }
            edge = next;
            next = next->nextLong;
        }
    }
    return bestType;
}

StructureType determineFixed(const Vec3* v, const double* d2, int available, double cutoff, const Enabled& en)
{
    double cutoffSq = cutoff * cutoff;
    int numNeighbors = 0;
    while(numNeighbors < available && d2[numNeighbors] <= cutoffSq) numNeighbors++;
    if(numNeighbors > MAX_NEIGHBORS) return OTHER;
    if(numNeighbors != 12 && numNeighbors != 14) return OTHER;
    NeighborBondArray na;
    for(int a = 0; a < numNeighbors; a++)
        for(int b = a + 1; b < numNeighbors; b++)
            na.setNeighborBond(a, b, (v[a] - v[b]).squaredLength() <= cutoffSq);
    StructureType type = numNeighbors == 12 ? analyzeSmallSignature(na) : analyzeLargeSignature(na);
    return filterType(type, en);
}

}  // namespace

// mode: 0 = adaptive, 1 = interval, 2 = fixed cutoff.  nbr_delta (N, K, 3) sorted by distance;
// for the fixed mode K must exceed 14 so that over-coordinated particles are recognized.
py::array_t<int32_t> cna_identify(py::array_t<double, py::array::c_style | py::array::forcecast> nbr_delta,
                                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> nbr_count,
                                  int mode, double cutoff, int enabled_mask, int num_threads)
{
    if(nbr_delta.ndim() != 3 || nbr_delta.shape(2) != 3) throw std::invalid_argument("nbr_delta must be (N, K, 3)");
    const size_t N = nbr_delta.shape(0), K = nbr_delta.shape(1);
    if((size_t)nbr_count.size() != N) throw std::invalid_argument("nbr_count must have length N");
    Enabled en{bool(enabled_mask & (1 << FCC)), bool(enabled_mask & (1 << HCP)),
               bool(enabled_mask & (1 << BCC)), bool(enabled_mask & (1 << ICO))};
    py::array_t<int32_t> out(N);
    int32_t* st = out.mutable_data();
    const double* delta = nbr_delta.data();
    const int32_t* count = nbr_count.data();
    {
        py::gil_scoped_release release;
        ewald::parallel_for(N, num_threads, [&](size_t start, size_t end, int) {
            Vec3 v[MAX_NEIGHBORS + 1];
            double d2[MAX_NEIGHBORS + 1];
            for(size_t i = start; i < end; i++) {
                int n = std::min<int>({count[i], (int)K, MAX_NEIGHBORS + 1});
                for(int k = 0; k < n; k++) {
                    const double* d = delta + (i * K + k) * 3;
                    v[k] = {d[0], d[1], d[2]};
                    d2[k] = v[k].squaredLength();
                }
                StructureType t;
                if(mode == 0) t = determineAdaptive(v, d2, n, en);
                else if(mode == 1) t = determineInterval(v, d2, n, en);
                else t = determineFixed(v, d2, n, cutoff, en);
                st[i] = t;
            }
        });
    }
    return out;
}

void register_cna(py::module_& m)
{
    m.def("cna_identify", &cna_identify, py::arg("nbr_delta"), py::arg("nbr_count"), py::arg("mode") = 0,
          py::arg("cutoff") = 3.2, py::arg("enabled_mask") = 0x1e, py::arg("num_threads") = 0);
}
