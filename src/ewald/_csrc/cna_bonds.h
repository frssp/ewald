// Bit-array helpers of OVITO's common neighbor analysis, shared by cna.cpp and crystal.cpp.
//
// Ported from OVITO 3.16 CommonNeighborAnalysisModifier.h/.cpp:
//
//   Copyright 2026 OVITO GmbH, Germany
//   OVITO is free software; you can redistribute it and/or modify it either under the
//   terms of the GNU General Public License version 3 as published by the Free Software
//   Foundation (the "GPL") or, at your option, under the terms of the MIT License.
//   If you do not alter this notice, a recipient may use your version of this
//   file under either the GPL or the MIT License.
//
// This port is used under the MIT License (see LICENSE.OVITO-MIT.txt).
#pragma once

#include <algorithm>
#include <cstring>

namespace ewald::cna {

using CNAPairBond = unsigned int;

struct NeighborBondArray {
    unsigned int neighborArray[32];
    NeighborBondArray() { std::memset(neighborArray, 0, sizeof(neighborArray)); }
    bool neighborBond(int a, int b) const { return (neighborArray[a] & (1u << b)) != 0; }
    void setNeighborBond(int a, int b, bool bonded) {
        if(bonded) { neighborArray[a] |= (1u << b); neighborArray[b] |= (1u << a); }
        else { neighborArray[a] &= ~(1u << b); neighborArray[b] &= ~(1u << a); }
    }
};

inline int popcount(unsigned int v) { int c = 0; for(; v; c++) v &= v - 1; return c; }
inline int ctz(unsigned int v) { int c = 0; while(!(v & 1u)) { v >>= 1; c++; } return c; }

inline int findCommonNeighbors(const NeighborBondArray& na, int neighborIndex, unsigned int& commonNeighbors)
{
    commonNeighbors = na.neighborArray[neighborIndex];
    return popcount(commonNeighbors);
}

inline int findNeighborBonds(const NeighborBondArray& na, unsigned int commonNeighbors, int numNeighbors, CNAPairBond* neighborBonds)
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

inline int getAdjacentBonds(unsigned int atom, CNAPairBond* bondsToProcess, int& numBonds, unsigned int& atomsToProcess, unsigned int& atomsProcessed)
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

inline int calcMaxChainLength(CNAPairBond* neighborBonds, int numBonds)
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

}  // namespace ewald::cna
