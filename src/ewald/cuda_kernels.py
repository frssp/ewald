"""CUDA C kernels launched through CuPy ``RawKernel``.

The neighbor kernel is the same two-pass cell-list algorithm as the numba CPU
kernel in :mod:`ewald.neighbors`: one thread per (cell-sorted) atom, pass 1
counts neighbors, pass 2 writes them at prefix-sum offsets and sorts each row
by j. ``PAIRS_SOURCE`` is plain C++ apart from ``__global__`` and the thread
index, so the tests compile it with a host C++ compiler and run it on the CPU.
"""
from __future__ import annotations

import numpy as np

PAIRS_SOURCE = r"""
extern "C" __global__
void ewald_pairs(const double* fs, const long long* bs, const long long* starts,
                 const long long* order, const double* mat, const int* pbc,
                 const long long* nbins, const double cutoff2, const int half,
                 const long long n, long long* offsets, long long* out_i,
                 long long* out_j, double* out_d, const int fill)
{
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    const long long i = order[k];
    const long long nb0 = nbins[0], nb1 = nbins[1], nb2 = nbins[2];
    const long long cx = bs[3 * k], cy = bs[3 * k + 1], cz = bs[3 * k + 2];
    const double fx = fs[3 * k], fy = fs[3 * k + 1], fz = fs[3 * k + 2];
    long long pos = fill ? offsets[k] : 0;
    const long long first = pos;
    for (int ox = -1; ox <= 1; ++ox) {
        long long nx = cx + ox;
        if (pbc[0]) nx = ((nx % nb0) + nb0) % nb0;
        else if (nx < 0 || nx >= nb0) continue;
        for (int oy = -1; oy <= 1; ++oy) {
            long long ny = cy + oy;
            if (pbc[1]) ny = ((ny % nb1) + nb1) % nb1;
            else if (ny < 0 || ny >= nb1) continue;
            for (int oz = -1; oz <= 1; ++oz) {
                long long nz = cz + oz;
                if (pbc[2]) nz = ((nz % nb2) + nb2) % nb2;
                else if (nz < 0 || nz >= nb2) continue;
                const long long c = (nx * nb1 + ny) * nb2 + nz;
                for (long long s = starts[c]; s < starts[c + 1]; ++s) {
                    const long long j = order[s];
                    if (j == i || (half && j < i)) continue;
                    double f0 = fs[3 * s] - fx, f1 = fs[3 * s + 1] - fy, f2 = fs[3 * s + 2] - fz;
                    if (pbc[0]) f0 -= rint(f0);
                    if (pbc[1]) f1 -= rint(f1);
                    if (pbc[2]) f2 -= rint(f2);
                    const double d0 = f0 * mat[0] + f1 * mat[3] + f2 * mat[6];
                    const double d1 = f0 * mat[1] + f1 * mat[4] + f2 * mat[7];
                    const double d2 = f0 * mat[2] + f1 * mat[5] + f2 * mat[8];
                    if (d0 * d0 + d1 * d1 + d2 * d2 < cutoff2) {
                        if (fill) {
                            out_i[pos] = i;
                            out_j[pos] = j;
                            out_d[3 * pos] = d0;
                            out_d[3 * pos + 1] = d1;
                            out_d[3 * pos + 2] = d2;
                        }
                        ++pos;
                    }
                }
            }
        }
    }
    if (!fill) {
        offsets[k] = pos;
        return;
    }
    for (long long a = first + 1; a < pos; ++a) {  // insertion sort of the row by j
        const long long jj = out_j[a];
        const double t0 = out_d[3 * a], t1 = out_d[3 * a + 1], t2 = out_d[3 * a + 2];
        long long b = a - 1;
        while (b >= first && out_j[b] > jj) {
            out_j[b + 1] = out_j[b];
            out_d[3 * (b + 1)] = out_d[3 * b];
            out_d[3 * (b + 1) + 1] = out_d[3 * b + 1];
            out_d[3 * (b + 1) + 2] = out_d[3 * b + 2];
            --b;
        }
        out_j[b + 1] = jj;
        out_d[3 * (b + 1)] = t0;
        out_d[3 * (b + 1) + 1] = t1;
        out_d[3 * (b + 1) + 2] = t2;
    }
}
"""

_kernels = {}


def _pairs_kernel(cp):
    if "pairs" not in _kernels:
        _kernels["pairs"] = cp.RawKernel(PAIRS_SOURCE, "ewald_pairs")
    return _kernels["pairs"]


def cell_list_pairs(cp, fs, bs, starts, order, matrix, pbc, nbins, cutoff, half, threads=128):
    """Run the two-pass kernel. ``fs``/``bs`` are cell-sorted fractional
    positions and bin coordinates, ``order`` maps sorted -> original index.
    Returns ``(i, j, delta)`` as arrays of module ``cp``.
    """
    n = int(len(fs))
    fs = cp.ascontiguousarray(cp.asarray(fs, dtype=cp.float64))
    bs = cp.ascontiguousarray(cp.asarray(bs, dtype=cp.int64))
    starts = cp.asarray(starts, dtype=cp.int64)
    order = cp.asarray(order, dtype=cp.int64)
    mat = cp.asarray(np.ascontiguousarray(matrix, dtype=np.float64).ravel())
    pbc_ = cp.asarray(np.asarray(pbc, dtype=np.int32))
    nb = cp.asarray(np.asarray(nbins, dtype=np.int64))
    kernel = _pairs_kernel(cp)
    grid = ((n + threads - 1) // threads,)
    head = (fs, bs, starts, order, mat, pbc_, nb, np.float64(cutoff * cutoff),
            np.int32(bool(half)), np.int64(n))
    counts = cp.zeros(n, dtype=cp.int64)
    empty_i = cp.zeros(1, dtype=cp.int64)
    empty_d = cp.zeros(3, dtype=cp.float64)
    kernel(grid, (threads,), head + (counts, empty_i, empty_i, empty_d, np.int32(0)))
    # Rows are laid out in original atom order: prefix sum over i.
    row_len = cp.empty(n, dtype=cp.int64)
    row_len[order] = counts
    row_start = cp.zeros(n + 1, dtype=cp.int64)
    row_start[1:] = cp.cumsum(row_len)
    total = int(row_start[-1])
    out_i = cp.empty(max(total, 1), dtype=cp.int64)
    out_j = cp.empty(max(total, 1), dtype=cp.int64)
    out_d = cp.empty((max(total, 1), 3), dtype=cp.float64)
    offsets = cp.ascontiguousarray(row_start[order])
    kernel(grid, (threads,), head + (offsets, out_i, out_j, out_d, np.int32(1)))
    return out_i[:total], out_j[:total], out_d[:total]
