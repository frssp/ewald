"""The ray tracer on a GPU: :func:`ewald.raytrace.raytrace` with ``device="cuda"``, ``"opencl"``
or ``"auto"``.

``RAYTRACE_BODY`` is a line-by-line port of the numba kernels in :mod:`ewald.raytrace`
(``_render``, ``_shade``, ``_trace``, the grid DDA, the BVH walk and the hit tests) to C, one
thread per pixel. One source serves two APIs: with :data:`CUDA_PRELUDE` it is CUDA C, launched
through CuPy ``RawKernel`` on NVIDIA GPUs; with :data:`OPENCL_PRELUDE` it is OpenCL C, launched
through pyopencl on Intel, AMD, NVIDIA and Apple GPUs. The acceleration structures are still
built on the CPU and uploaded.

``REAL`` is the floating-point type of the geometry and the shading: ``float`` on a GPU
(consumer GPUs run double precision at 1/32 to 1/64 of the float rate, Intel Arc not at all),
``double`` in the tests, where the source must give the CPU tracer's image
(tests/test_gpu_raytrace.py: the CUDA source compiled by the host C++ compiler behind a shim, the
OpenCL source on pocl's CPU device). ``WIDE`` holds the random numbers, the R2
ambient-occlusion sequence and the sums of a pixel: ``double``, the CPU tracer's type, so a
pixel takes the same rays on every device, or ``float`` on a device without double precision.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

MAX_STACK = 64     # BVH traversal stack per thread; deeper trees fall back to the CPU tracer
THREADS = 128

CUDA_PRELUDE = r"""
#ifndef REAL
#define REAL float
#endif
#ifndef WIDE
#define WIDE double
#endif
#define DEVFN __device__ static inline
#define KERNEL extern "C" __global__
#define GLOBAL
typedef long long I64;
typedef unsigned long long U64;
typedef int IX;
#define THREAD_ID ((I64)blockDim.x * blockIdx.x + threadIdx.x)
"""

# REAL and WIDE come as -D options; float builds add -cl-single-precision-constant, so the
# double literals of the body are float literals on a device without double precision
OPENCL_PRELUDE = r"""
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#pragma OPENCL FP_CONTRACT OFF
#define DEVFN static inline
#define KERNEL __kernel
#define GLOBAL __global
typedef long I64;
typedef ulong U64;
typedef int IX;
#define THREAD_ID ((I64)get_global_id(0))
"""

RAYTRACE_BODY = r"""
#define KIND_SHIFT 29
#define NO_SHADOW ((IX)1 << 28)
#define INDEX_MASK (NO_SHADOW - 1)
#define MAX_STACK %(max_stack)d

DEVFN REAL rmin(REAL a, REAL b) { return a < b ? a : b; }
DEVFN REAL rmax(REAL a, REAL b) { return a > b ? a : b; }

DEVFN WIDE rand01(U64* state)
{
    U64 x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (WIDE)((x * (U64)2685821657736338717) >> 11) * (1.0 / 9007199254740992.0);
}

DEVFN REAL hit_sphere(const REAL* o, const REAL* d, GLOBAL const REAL* sph, IX i)
{
    GLOBAL const REAL* s = sph + 4 * i;
    REAL ox = o[0] - s[0], oy = o[1] - s[1], oz = o[2] - s[2];
    REAL r = s[3];
    REAL b = ox * d[0] + oy * d[1] + oz * d[2];
    REAL cc = ox * ox + oy * oy + oz * oz - r * r;
    REAL disc = b * b - cc;
    if (disc < 0) return -1;
    REAL q = sqrt(disc);
    REAL t = -b - q;
    if (t > (REAL)1e-6) return t;
    t = -b + q;
    return t > (REAL)1e-6 ? t : -1;
}

DEVFN REAL hit_cylinder(const REAL* o, const REAL* d, GLOBAL const REAL* cyl, IX i)
{
    GLOBAL const REAL* c = cyl + 9 * i;
    REAL ax = c[3] - c[0], ay = c[4] - c[1], az = c[5] - c[2];
    REAL r = c[6];
    REAL L = sqrt(ax * ax + ay * ay + az * az);
    if (L < (REAL)1e-12) return -1;
    ax /= L; ay /= L; az /= L;
    REAL ox = o[0] - c[0], oy = o[1] - c[1], oz = o[2] - c[2];
    REAL dd = d[0] * ax + d[1] * ay + d[2] * az;
    REAL od = ox * ax + oy * ay + oz * az;
    REAL px = d[0] - dd * ax, py = d[1] - dd * ay, pz = d[2] - dd * az;
    REAL qx = ox - od * ax, qy = oy - od * ay, qz = oz - od * az;
    REAL best = -1;
    if (fabs(dd) > (REAL)1e-12) {  /* flat caps (c[7], c[8]): the planes h = 0 and h = L inside the rim */
        for (int e = 0; e < 2; ++e) {
            if (c[7 + e] > (REAL)0.5) {
                REAL t = ((e ? L : 0) - od) / dd;
                if (t > (REAL)1e-6 && (best < 0 || t < best)) {
                    REAL wx = qx + t * px, wy = qy + t * py, wz = qz + t * pz;
                    if (wx * wx + wy * wy + wz * wz <= r * r) best = t;
                }
            }
        }
    }
    REAL A = px * px + py * py + pz * pz;
    if (A < (REAL)1e-14) return best;
    REAL B = px * qx + py * qy + pz * qz;
    REAL C = qx * qx + qy * qy + qz * qz - r * r;
    REAL disc = B * B - A * C;
    if (disc < 0) return best;
    REAL s = sqrt(disc);
    REAL ts[2] = {(-B - s) / A, (-B + s) / A};
    for (int k = 0; k < 2; ++k) {
        REAL t = ts[k];
        if (t > (REAL)1e-6) {
            REAL h = od + t * dd;
            if (0 <= h && h <= L) return (best < 0 || t < best) ? t : best;
        }
    }
    return best;
}

DEVFN REAL hit_triangle(const REAL* o, const REAL* d, GLOBAL const REAL* tri, IX i)
{
    GLOBAL const REAL* T = tri + 10 * i;
    REAL e1x = T[3] - T[0], e1y = T[4] - T[1], e1z = T[5] - T[2];
    REAL e2x = T[6] - T[0], e2y = T[7] - T[1], e2z = T[8] - T[2];
    REAL px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
    REAL det = e1x * px + e1y * py + e1z * pz;
    if (fabs(det) < (REAL)1e-14) return -1;
    REAL inv = 1 / det;
    REAL tx = o[0] - T[0], ty = o[1] - T[1], tz = o[2] - T[2];
    REAL u = (tx * px + ty * py + tz * pz) * inv;
    if (u < 0 || u > 1) return -1;
    REAL qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
    REAL v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    if (v < 0 || u + v > 1) return -1;
    REAL t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    return t > (REAL)1e-6 ? t : -1;
}

DEVFN REAL hit_code(const REAL* o, const REAL* d, IX code, int any_hit,
                                       GLOBAL const REAL* sph, GLOBAL const REAL* cyl, GLOBAL const REAL* tri)
{
    IX kind = code >> KIND_SHIFT;
    IX i = code & INDEX_MASK;
    if (kind == 0) return hit_sphere(o, d, sph, i);
    if (kind == 1) return hit_cylinder(o, d, cyl, i);
    if (any_hit && tri[10 * i + 9] < (REAL)0.999) return -1;
    return hit_triangle(o, d, tri, i);
}

DEVFN REAL inv_dir(REAL x)
{
    if (fabs(x) < (REAL)1e-30) return (REAL)1e30;
    return 1 / x;
}

DEVFN REAL hit_box(GLOBAL const REAL* box, IX k, REAL ox, REAL oy, REAL oz,
                                      REAL ix, REAL iy, REAL iz, REAL tmax)
{
    GLOBAL const REAL* b = box + 8 * k;
    REAL ta = (b[0] - ox) * ix, tb = (b[3] - ox) * ix;
    REAL tn = rmin(ta, tb), tf = rmax(ta, tb);
    ta = (b[1] - oy) * iy; tb = (b[4] - oy) * iy;
    tn = rmax(tn, rmin(ta, tb)); tf = rmin(tf, rmax(ta, tb));
    ta = (b[2] - oz) * iz; tb = (b[5] - oz) * iz;
    tn = rmax(tn, rmin(ta, tb)); tf = rmin(tf, rmax(ta, tb));
    tn = rmax(tn, (REAL)0);
    return (tn <= tf && tn < tmax) ? tn : -1;
}

DEVFN void dda_axis(REAL o, REAL d, REAL lo, REAL size, IX dim, REAL t0,
                                       IX* c, IX* step, REAL* next, REAL* delta)
{
    IX cc = (IX)((o + d * (t0 + (REAL)1e-9) - lo) / size);
    cc = cc < 0 ? 0 : (cc > dim - 1 ? dim - 1 : cc);
    *c = cc;
    if (d > 0) { *step = 1; *next = (lo + (cc + 1) * size - o) / d; *delta = size / d; }
    else if (d < 0) { *step = -1; *next = (lo + cc * size - o) / d; *delta = -size / d; }
    else { *step = 0; *next = (REAL)1e30; *delta = (REAL)1e30; }
}

DEVFN IX trace_grid(const REAL* o, const REAL* d, REAL tmax, GLOBAL const REAL* grid,
                                       GLOBAL const IX* starts, GLOBAL const IX* items, IX n_items,
                                       GLOBAL const REAL* sph, GLOBAL const REAL* cyl, GLOBAL const REAL* tri, int any_hit, REAL* t_out)
{
    if (n_items == 0) return -1;
    REAL t0 = 0, t1 = tmax;
    for (int k = 0; k < 3; ++k) {
        REAL hi = grid[k] + grid[3 + k] * grid[6 + k];
        if (fabs(d[k]) < (REAL)1e-12) {
            if (o[k] < grid[k] || o[k] > hi) return -1;
        } else {
            REAL ta = (grid[k] - o[k]) / d[k];
            REAL tb = (hi - o[k]) / d[k];
            if (ta > tb) { REAL tmp = ta; ta = tb; tb = tmp; }
            t0 = rmax(t0, ta);
            t1 = rmin(t1, tb);
        }
    }
    if (t0 > t1) return -1;
    IX dx = (IX)grid[6], dy = (IX)grid[7], dz = (IX)grid[8];
    IX cx, sx, cy, sy, cz, sz;
    REAL nx, ex, ny, ey, nz, ez;
    dda_axis(o[0], d[0], grid[0], grid[3], dx, t0, &cx, &sx, &nx, &ex);
    dda_axis(o[1], d[1], grid[1], grid[4], dy, t0, &cy, &sy, &ny, &ey);
    dda_axis(o[2], d[2], grid[2], grid[5], dz, t0, &cz, &sz, &nz, &ez);
    REAL best_t = (REAL)1e30;
    IX best = -1;
    // a ray crosses at most dx + dy + dz cells; the bound also ends a ray made NaN by float round-off
    for (IX steps = dx + dy + dz + 3; steps > 0; --steps) {
        IX c = (cx * dy + cy) * dz + cz;
        for (IX s = starts[c]; s < starts[c + 1]; ++s) {
            IX code = items[s];
            if (any_hit && (code & NO_SHADOW)) continue;
            REAL t = hit_code(o, d, code, any_hit, sph, cyl, tri);
            if (t > 0 && t < best_t && t < tmax) {
                best_t = t; best = code;
                if (any_hit) { *t_out = best_t; return best; }
            }
        }
        REAL tcell = rmin(nx, rmin(ny, nz));
        if (best >= 0 && best_t <= tcell) { *t_out = best_t; return best; }
        if (tcell > t1) break;
        if (ny < nx && ny <= nz) {
            cy += sy;
            if (cy < 0 || cy >= dy) break;
            ny += ey;
        } else if (nz < nx && nz < ny) {
            cz += sz;
            if (cz < 0 || cz >= dz) break;
            nz += ez;
        } else {
            cx += sx;
            if (cx < 0 || cx >= dx) break;
            nx += ex;
        }
    }
    if (best >= 0) { *t_out = best_t; return best; }
    return -1;
}

DEVFN IX trace_bvh(const REAL* o, const REAL* d, REAL tmax, GLOBAL const REAL* box, IX n_nodes,
                                      GLOBAL const IX* items, GLOBAL const REAL* sph, GLOBAL const REAL* cyl, GLOBAL const REAL* tri,
                                      int any_hit, REAL* t_out)
{
    if (n_nodes == 0) return -1;
    IX stack[MAX_STACK];
    REAL stack_t[MAX_STACK];
    REAL ox = o[0], oy = o[1], oz = o[2];
    REAL ix = inv_dir(d[0]), iy = inv_dir(d[1]), iz = inv_dir(d[2]);
    REAL best_t = tmax;
    IX best = -1;
    REAL t_root = hit_box(box, 0, ox, oy, oz, ix, iy, iz, best_t);
    if (t_root < 0) return -1;
    stack[0] = 0; stack_t[0] = t_root;
    int sp = 1;
    while (sp > 0) {
        --sp;
        if (stack_t[sp] >= best_t) continue;
        IX node = stack[sp];
        IX n = (IX)box[8 * node + 7];
        if (n > 0) {
            IX first = (IX)box[8 * node + 6];
            for (IX s = first; s < first + n; ++s) {
                IX code = items[s];
                if (any_hit && (code & NO_SHADOW)) continue;
                REAL t = hit_code(o, d, code, any_hit, sph, cyl, tri);
                if (t > 0 && t < best_t) {
                    best_t = t; best = code;
                    if (any_hit) { *t_out = best_t; return best; }
                }
            }
        } else {
            IX left = (IX)box[8 * node + 6];
            REAL tl = hit_box(box, left, ox, oy, oz, ix, iy, iz, best_t);
            REAL tr = hit_box(box, left + 1, ox, oy, oz, ix, iy, iz, best_t);
            if (tl >= 0 && tr >= 0) {
                if (tl <= tr) {
                    stack[sp] = left + 1; stack_t[sp] = tr;
                    stack[sp + 1] = left; stack_t[sp + 1] = tl;
                } else {
                    stack[sp] = left; stack_t[sp] = tl;
                    stack[sp + 1] = left + 1; stack_t[sp + 1] = tr;
                }
                sp += 2;
            } else if (tl >= 0) {
                stack[sp] = left; stack_t[sp] = tl; sp += 1;
            } else if (tr >= 0) {
                stack[sp] = left + 1; stack_t[sp] = tr; sp += 1;
            }
        }
    }
    if (best >= 0) { *t_out = best_t; return best; }
    return -1;
}

typedef struct {
    GLOBAL const REAL* grid; GLOBAL const IX* starts; GLOBAL const IX* items; IX n_items;
    GLOBAL const REAL* nodes; GLOBAL const IX* bvh_items; IX n_nodes;
    GLOBAL const REAL* sph; GLOBAL const REAL* cyl; GLOBAL const REAL* tri;
} Accel;

DEVFN IX trace(const REAL* o, const REAL* d, REAL tmax, const Accel* a, int any_hit, REAL* t_out)
{
    REAL tg = -1;
    IX cg = -1;
    if (a->n_items) {
        cg = trace_grid(o, d, tmax, a->grid, a->starts, a->items, a->n_items, a->sph, a->cyl, a->tri, any_hit, &tg);
        if (cg >= 0) {
            if (any_hit) { *t_out = tg; return cg; }
            tmax = tg;
        }
    }
    if (a->n_nodes) {
        REAL tb = -1;
        IX cb = trace_bvh(o, d, tmax, a->nodes, a->n_nodes, a->bvh_items, a->sph, a->cyl, a->tri, any_hit, &tb);
        if (cb >= 0) { *t_out = tb; return cb; }
    }
    *t_out = tg;
    return cg;
}

DEVFN void shade(const REAL* p, const REAL* n, const REAL* d, REAL cr, REAL cg, REAL cb,
                             const REAL* light, REAL ambient, REAL diffuse, REAL specular, REAL shininess,
                             int shadows, IX ao_samples, REAL ao_dist, const Accel* a, WIDE* aos,
                             REAL* out)
{
    REAL ndl = n[0] * light[0] + n[1] * light[1] + n[2] * light[2];
    REAL lit = 0, spec = 0, ts;
    if (ndl > 0) {
        REAL vis = 1;
        if (shadows && trace(p, light, (REAL)1e30, a, 1, &ts) >= 0) vis = 0;
        lit = ndl * vis;
        REAL hx = light[0] - d[0], hy = light[1] - d[1], hz = light[2] - d[2];
        REAL hn = sqrt(hx * hx + hy * hy + hz * hz);
        REAL ndh = (n[0] * hx + n[1] * hy + n[2] * hz) / hn;
        if (ndh > 0) spec = vis * pow(ndh, shininess);
    }
    REAL ao = 1;
    if (ao_samples > 0) {
        REAL tx, ty, tz;
        if (fabs(n[0]) > (REAL)0.9) { tx = 0; ty = 1; tz = 0; } else { tx = 1; ty = 0; tz = 0; }
        REAL bx = n[1] * tz - n[2] * ty, by = n[2] * tx - n[0] * tz, bz = n[0] * ty - n[1] * tx;
        REAL bn = sqrt(bx * bx + by * by + bz * bz);
        bx /= bn; by /= bn; bz /= bn;
        REAL cx = n[1] * bz - n[2] * by, cy = n[2] * bx - n[0] * bz, cz = n[0] * by - n[1] * bx;
        IX hits = 0;
        REAL od[3];
        for (IX s = 0; s < ao_samples; ++s) {
            WIDE g = aos[2];
            aos[2] += 1.0;
            WIDE r1 = fmod(aos[0] + g * 0.7548776662466927, 1.0);
            WIDE r2 = fmod(aos[1] + g * 0.5698402909980532, 1.0);
            REAL phi = (REAL)(2 * 3.141592653589793 * r1);
            REAL sr = (REAL)sqrt(r2);
            REAL x = cos(phi) * sr, y = sin(phi) * sr, z = (REAL)sqrt(r2 < 1.0 ? 1.0 - r2 : 0.0);
            od[0] = x * bx + y * cx + z * n[0];
            od[1] = x * by + y * cy + z * n[1];
            od[2] = x * bz + y * cz + z * n[2];
            if (trace(p, od, ao_dist, a, 1, &ts) >= 0) hits += 1;
        }
        ao = 1 - (REAL)hits / (REAL)ao_samples;
    }
    REAL sh = ambient * ao + diffuse * lit;
    out[0] = rmin((REAL)1, cr * sh + specular * spec);
    out[1] = rmin((REAL)1, cg * sh + specular * spec);
    out[2] = rmin((REAL)1, cb * sh + specular * spec);
}

KERNEL void ewald_raytrace(GLOBAL const REAL* grid, GLOBAL const IX* starts, GLOBAL const IX* items, const I64 n_items,
                    GLOBAL const REAL* nodes, GLOBAL const IX* bvh_items, const I64 n_nodes,
                    GLOBAL const REAL* sph, GLOBAL const REAL* cyl, GLOBAL const REAL* tri,
                    GLOBAL const REAL* sph_rgb, GLOBAL const REAL* cyl_rgb, GLOBAL const REAL* tri_n, GLOBAL const REAL* tri_vrgb,
                    GLOBAL const WIDE* P, GLOBAL const I64* I, GLOBAL const I64* rows, GLOBAL const unsigned char* active,
                    GLOBAL WIDE* acc, GLOBAL WIDE* sq, GLOBAL I64* nsamp, GLOBAL U64* rng, GLOBAL WIDE* ao_next)
{
    const IX width = I[0], height = I[1], ortho = I[2], s1 = I[3], samples = I[4];
    const IX ao_samples = I[5], shadows = I[6], n_rows = I[8], keep = I[9];
    const I64 seed = I[7];
    IX k = (IX)THREAD_ID;
    if (k >= n_rows * width) return;
    const IX py = rows[k / width], px = k %% width;
    const IX pix = py * width + px;
    if (!active[pix]) return;
    Accel a = {grid, starts, items, (IX)n_items, nodes, bvh_items, (IX)n_nodes, sph, cyl, tri};
    REAL eye[3], fwd[3], right[3], up[3], light[3];
    for (int q = 0; q < 3; ++q) {
        eye[q] = (REAL)P[q]; fwd[q] = (REAL)P[3 + q]; right[q] = (REAL)P[6 + q]; up[q] = (REAL)P[9 + q];
        light[q] = (REAL)P[14 + q];
    }
    const WIDE half_w = P[12], half_h = P[13];
    const REAL ambient = (REAL)P[17], diffuse = (REAL)P[18], specular = (REAL)P[19], shininess = (REAL)P[20];
    const REAL bg[4] = {(REAL)P[21], (REAL)P[22], (REAL)P[23], (REAL)P[24]};
    const REAL ao_dist = (REAL)P[25];

    U64 state = (U64)seed * (U64)0x9E3779B97F4A7C15 + (U64)(pix + 1);
    WIDE aos[3];
    aos[0] = rand01(&state); aos[1] = rand01(&state); aos[2] = 0.0;
    const WIDE jox = rand01(&state), joy = rand01(&state);
    IX s0 = keep ? (IX)nsamp[pix] : 0;
    if (s0 > 0) { state = rng[pix]; aos[2] = ao_next[pix]; }
    WIDE acc_r = acc[4 * pix], acc_g = acc[4 * pix + 1], acc_b = acc[4 * pix + 2], acc_a = acc[4 * pix + 3];
    WIDE acc_q = keep ? sq[pix] : 0.0;
    REAL o[3], d[3], n[3], p[3], sh[3];
    for (IX s = s0; s < s1; ++s) {
        WIDE jx = samples > 1 ? jox + s * 0.7548776662466927 : 0.5;   // R2 sub-pixel positions, as on the CPU
        WIDE jy = samples > 1 ? joy + s * 0.5698402909980532 : 0.5;
        jx -= floor(jx); jy -= floor(jy);
        WIDE u = ((px + jx) / width * 2.0 - 1.0) * half_w;
        WIDE v = (1.0 - (py + jy) / height * 2.0) * half_h;
        for (int q = 0; q < 3; ++q) {
            if (ortho) { o[q] = (REAL)(eye[q] + u * right[q] + v * up[q]); d[q] = fwd[q]; }
            else { o[q] = eye[q]; d[q] = (REAL)(fwd[q] + u * right[q] + v * up[q]); }
        }
        REAL nd = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int q = 0; q < 3; ++q) d[q] /= nd;
        WIDE vr = 0, vg = 0, vb = 0, va = 0;
        WIDE trans = 1.0;
        for (int layer = 0; layer < 16; ++layer) {
            REAL t;
            IX code = trace(o, d, (REAL)1e30, &a, 0, &t);
            if (code < 0) {
                vr += trans * bg[0]; vg += trans * bg[1]; vb += trans * bg[2]; va += trans * bg[3];
                break;
            }
            IX kind = code >> KIND_SHIFT;
            IX i = code & INDEX_MASK;
            for (int q = 0; q < 3; ++q) p[q] = o[q] + t * d[q];
            REAL alpha = 1, cr, cg, cb;
            if (kind == 0) {
                for (int q = 0; q < 3; ++q) n[q] = (p[q] - sph[4 * i + q]) / sph[4 * i + 3];
                cr = sph_rgb[3 * i]; cg = sph_rgb[3 * i + 1]; cb = sph_rgb[3 * i + 2];
            } else if (kind == 1) {
                GLOBAL const REAL* c = cyl + 9 * i;
                REAL ax = c[3] - c[0], ay = c[4] - c[1], az = c[5] - c[2];
                REAL L2 = ax * ax + ay * ay + az * az;
                REAL h = ((p[0] - c[0]) * ax + (p[1] - c[1]) * ay + (p[2] - c[2]) * az) / L2;
                n[0] = p[0] - (c[0] + h * ax); n[1] = p[1] - (c[1] + h * ay); n[2] = p[2] - (c[2] + h * az);
                REAL nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                /* a flat cap: the hit is closer to the end's plane than to the side */
                REAL plane = (h < (REAL)0.5 ? fabs(h) : fabs(1 - h)) * sqrt(L2);
                if ((h < (REAL)0.5 ? c[7] : c[8]) > (REAL)0.5 && plane < c[6] - nn) {
                    REAL sg = (h < (REAL)0.5 ? -1 : 1) / sqrt(L2);
                    n[0] = sg * ax; n[1] = sg * ay; n[2] = sg * az;
                } else {
                    for (int q = 0; q < 3; ++q) n[q] /= nn;
                }
                cr = cyl_rgb[3 * i]; cg = cyl_rgb[3 * i + 1]; cb = cyl_rgb[3 * i + 2];
            } else {
                GLOBAL const REAL* T = tri + 10 * i;
                REAL e1x = T[3] - T[0], e1y = T[4] - T[1], e1z = T[5] - T[2];
                REAL e2x = T[6] - T[0], e2y = T[7] - T[1], e2z = T[8] - T[2];
                REAL qx = p[0] - T[0], qy = p[1] - T[1], qz = p[2] - T[2];
                REAL d11 = e1x * e1x + e1y * e1y + e1z * e1z, d12 = e1x * e2x + e1y * e2y + e1z * e2z;
                REAL d22 = e2x * e2x + e2y * e2y + e2z * e2z;
                REAL dq1 = qx * e1x + qy * e1y + qz * e1z, dq2 = qx * e2x + qy * e2y + qz * e2z;
                REAL den = d11 * d22 - d12 * d12;
                REAL wb = den > 0 ? (d22 * dq1 - d12 * dq2) / den : 0;
                REAL wc = den > 0 ? (d11 * dq2 - d12 * dq1) / den : 0;
                REAL wa = 1 - wb - wc;
                GLOBAL const REAL* N = tri_n + 9 * i;
                for (int q = 0; q < 3; ++q) n[q] = wa * N[q] + wb * N[3 + q] + wc * N[6 + q];
                REAL nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (nn < (REAL)1e-12) {
                    n[0] = e1y * e2z - e1z * e2y; n[1] = e1z * e2x - e1x * e2z; n[2] = e1x * e2y - e1y * e2x;
                    nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                }
                for (int q = 0; q < 3; ++q) n[q] /= nn;
                GLOBAL const REAL* V = tri_vrgb + 9 * i;
                cr = wa * V[0] + wb * V[3] + wc * V[6];
                cg = wa * V[1] + wb * V[4] + wc * V[7];
                cb = wa * V[2] + wb * V[5] + wc * V[8];
                alpha = T[9];
            }
            if (n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0)
                for (int q = 0; q < 3; ++q) n[q] = -n[q];
            REAL eps = (REAL)1e-4 * (1 + fabs(t));
            for (int q = 0; q < 3; ++q) p[q] += n[q] * eps;
            shade(p, n, d, cr, cg, cb, light, ambient, diffuse, specular, shininess, (int)shadows, ao_samples,
                  ao_dist, &a, aos, sh);
            WIDE w = trans * alpha;
            vr += w * sh[0]; vg += w * sh[1]; vb += w * sh[2]; va += w;
            trans *= 1.0 - alpha;
            if (trans < 1e-3) break;
            for (int q = 0; q < 3; ++q) o[q] = p[q] - 2 * n[q] * eps;
        }
        acc_r += vr; acc_g += vg; acc_b += vb; acc_a += va;
        acc_q += vr * vr + vg * vg + vb * vb + va * va;
    }
    acc[4 * pix] = acc_r; acc[4 * pix + 1] = acc_g; acc[4 * pix + 2] = acc_b; acc[4 * pix + 3] = acc_a;
    if (keep) { sq[pix] = acc_q; nsamp[pix] = s1; rng[pix] = state; ao_next[pix] = aos[2]; }
}
"""


def cuda_source(real: str = "float") -> str:
    """The CUDA C source in ``real`` precision (``WIDE`` stays double)."""
    return f"#define REAL {real}\n" + CUDA_PRELUDE + RAYTRACE_BODY % {"max_stack": MAX_STACK}


def opencl_source() -> str:
    """The OpenCL C source; ``REAL`` and ``WIDE`` come from the build options."""
    return OPENCL_PRELUDE + RAYTRACE_BODY % {"max_stack": MAX_STACK}


GPU_KIND_SHIFT, GPU_NO_SHADOW = 29, 1 << 28    # the kernel's 32-bit item codes


def gpu_codes(codes):
    """The CPU tracer's 64-bit item codes (``kind << 40 | NO_SHADOW | index``) as the kernel's
    32-bit ones (``kind << 29 | 1 << 28 | index``): GPUs emulate 64-bit integers (Apple, Intel
    Arc), and the grid and BVH walks do little else."""
    from .raytrace import INDEX_MASK, NO_SHADOW

    codes = np.asarray(codes, np.int64)
    out = (codes >> 40 << GPU_KIND_SHIFT) | np.where(codes & NO_SHADOW, GPU_NO_SHADOW, 0) | (codes & INDEX_MASK)
    return out.astype(np.int32)


def fits(scene, accel) -> bool:
    """Whether the kernel's 32-bit codes and indices, and the BVH node fields stored as float,
    hold this scene."""
    n = max(len(scene.sph_rgb), len(scene.cyl_rgb), len(scene.tri_vrgb))
    return n < GPU_NO_SHADOW and len(accel.items) < 2 ** 31 and len(accel.bvh_items) < 2 ** 24 \
        and len(accel.nodes) < 2 ** 24


def _scene_arrays(scene, accel, dtype):
    """The 14 scene arguments of ``ewald_raytrace`` as host arrays (and two counts)."""
    def arr(a, dt=dtype):
        a = np.ascontiguousarray(np.asarray(a, dtype=dt).ravel())
        return a if len(a) else np.zeros(1, dt)     # no zero-size device arrays

    return (arr(accel.grid), arr(accel.starts, np.int32), arr(gpu_codes(accel.items), np.int32),
            np.int64(len(accel.items)), arr(accel.nodes), arr(gpu_codes(accel.bvh_items), np.int32),
            np.int64(len(accel.nodes)), arr(accel.sph), arr(accel.cyl), arr(accel.tri),
            arr(scene.sph_rgb), arr(scene.cyl_rgb), arr(scene.tri_n), arr(scene.tri_vrgb))


# Rays one kernel launch may trace. Windows resets a GPU whose kernel runs for more than about
# 2 s (TDR) and macOS stops it as well, so a picture is traced in launches of a few rows that
# stay well below that even on an integrated GPU.
RAYS_PER_LAUNCH = 1 << 23


def _launches(ints, rows):
    """``(ints, rows, threads)`` for each kernel launch of one pass over ``rows``."""
    ints = np.asarray(ints, np.int64)
    width, s1, ao, shadows = (int(ints[i]) for i in (0, 3, 5, 6))
    per_row = width * max(1, s1) * (1 + ao + shadows)
    step = max(1, RAYS_PER_LAUNCH // max(1, per_row))
    rows = np.ascontiguousarray(rows, np.int64)
    for i in range(0, len(rows), step):
        chunk = rows[i:i + step]
        part = ints.copy()
        part[8] = len(chunk)
        yield part, chunk, len(chunk) * width


# --- CUDA (CuPy) ----------------------------------------------------------------------------

_cuda_kernels = {}


def _kernel(cp, real: str):
    if real not in _cuda_kernels:
        _cuda_kernels[real] = cp.RawKernel(cuda_source(real), "ewald_raytrace")
    return _cuda_kernels[real]


class CudaTracer:
    """The device side of :func:`ewald.raytrace.raytrace` on a CuPy backend. ``acc``, ``sq``,
    ``nsamp``, ``rng`` and ``ao_next`` live on the device between passes."""

    name = "CUDA"

    def __init__(self, backend, scene, accel, real: str = "float"):
        self.backend, self.real, self.cp = backend, real, backend.xp
        dt = np.float32 if real == "float" else np.float64
        self.args = tuple(a if isinstance(a, np.int64) else self.cp.asarray(a)
                          for a in _scene_arrays(scene, accel, dt))

    def upload(self, a):
        return self.cp.asarray(np.ascontiguousarray(a))

    def download(self, a):
        return self.backend.to_numpy(a)

    def trace(self, params, ints, rows, active, acc, sq, nsamp, rng, ao_next):
        """One pass of :func:`ewald.raytrace._render`: every ``active`` pixel of ``rows`` is
        brought to ``ints[3]`` samples; the state arrays are updated in place."""
        cp = self.cp
        dev_params = cp.asarray(np.ascontiguousarray(params, np.float64))
        for part, chunk, n in _launches(ints, rows):
            args = self.args + (dev_params, cp.asarray(part), cp.asarray(chunk), active, acc, sq, nsamp, rng,
                                ao_next)
            _kernel(cp, self.real)(((n + THREADS - 1) // THREADS,), (THREADS,), args)
            self.backend.synchronize()   # one launch at a time; progress follows the work done


# --- OpenCL (pyopencl) ----------------------------------------------------------------------

OPENCL_FP64 = None   # None: double sums where the device has them; tests set False (as on Intel Arc)
OPENCL_MISSING = "device='opencl' needs pyopencl: pip install 'ewald[opencl]' (or pip install pyopencl)"


def opencl_devices(gpu_only: bool = False) -> list:
    """The OpenCL devices on this machine, GPUs first; [] without pyopencl or a driver."""
    try:
        import pyopencl as cl

        devices = [d for p in cl.get_platforms() for d in p.get_devices()]
    except Exception:
        return []
    gpus = [d for d in devices if d.type & cl.device_type.GPU]
    return gpus if gpu_only else gpus + [d for d in devices if d not in gpus]


_opencl_programs = {}   # (device, REAL, WIDE) -> (context, queue, kernel): building takes seconds


def _opencl_kernel(device, real: str, wide: str):
    import pyopencl as cl

    key = (device.int_ptr, real, wide)
    if key not in _opencl_programs:
        ctx = cl.Context([device])
        options = [f"-DREAL={real}", f"-DWIDE={wide}"]
        if wide == "float":
            options.append("-cl-single-precision-constant")
        program = cl.Program(ctx, opencl_source()).build(options=options)
        _opencl_programs[key] = (ctx, cl.CommandQueue(ctx), program.ewald_raytrace)
    return _opencl_programs[key]


class OpenCLTracer:
    """The device side of :func:`ewald.raytrace.raytrace` on an OpenCL device. On a device
    without double precision (Intel Arc and most integrated GPUs) the sums and the random
    numbers are ``float`` as well; the host arrays stay float64."""

    name = "OpenCL"

    def __init__(self, device, scene, accel, real: str = "float"):
        import pyopencl as cl
        import pyopencl.array as cla

        fp64 = bool(device.double_fp_config) and OPENCL_FP64 is not False
        if real == "double" and not device.double_fp_config:
            raise RuntimeError(f"{device.name} has no double precision")
        self.cl, self.cla, self.device = cl, cla, device
        self.wide = np.float64 if fp64 else np.float32
        self.ctx, self.queue, self.kernel = _opencl_kernel(device, real, "double" if fp64 else "float")
        dt = np.float32 if real == "float" else np.float64
        self.args = tuple(a if isinstance(a, np.int64) else self._to_device(a)
                          for a in _scene_arrays(scene, accel, dt))

    def _to_device(self, a):
        return self.cla.to_device(self.queue, np.ascontiguousarray(a))

    def upload(self, a):
        a = np.asarray(a)
        return self._to_device(a.astype(self.wide) if a.dtype == np.float64 else a)

    def download(self, a):
        a = a.get()
        return a.astype(np.float64) if a.dtype.kind == "f" else a

    def trace(self, params, ints, rows, active, acc, sq, nsamp, rng, ao_next):
        """As :meth:`CudaTracer.trace`."""
        dev_params = self._to_device(np.asarray(params, self.wide))
        for part, chunk, n in _launches(ints, rows):
            extra = (dev_params, self._to_device(part), self._to_device(chunk), active, acc, sq, nsamp, rng, ao_next)
            args = [a if isinstance(a, np.int64) else a.data for a in self.args + extra]
            self.kernel(self.queue, (n,), None, *args)
            self.queue.finish()      # one launch at a time; progress follows the work done


# --- choosing a device ----------------------------------------------------------------------

def tracer(scene, accel, device, real: str = "float"):
    """A GPU tracer for ``device``, or None for the CPU kernels. ``"cuda"``/``"gpu"`` (or a CuPy
    :class:`ewald.backend.Backend`) is the CUDA tracer, ``"opencl"`` the first OpenCL GPU (any
    OpenCL device if there is no GPU), ``"auto"`` CUDA if CuPy finds a GPU, else an OpenCL GPU
    (not on macOS, where Apple's OpenCL is slower than the CPU), else the CPU. A BVH deeper than the kernel's fixed traversal stack stays on the CPU."""
    if device is None or (isinstance(device, str) and device.lower() == "cpu"):
        return None
    from .backend import get_backend, gpu_available

    name = device.lower() if isinstance(device, str) else None
    if name == "opencl":
        devices = opencl_devices()
        if not devices:
            try:
                import pyopencl  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(OPENCL_MISSING) from exc
            raise RuntimeError("pyopencl is installed but no OpenCL device was found (install the GPU driver)")
        make = lambda: OpenCLTracer(devices[0], scene, accel, real)   # noqa: E731
    elif name == "auto" and not gpu_available():
        # Apple's OpenCL (deprecated, over Metal) traced 40x slower than the CPU on an M2 Air
        devices = [] if sys.platform == "darwin" else opencl_devices(gpu_only=True)
        if not devices:
            return None
        make = lambda: OpenCLTracer(devices[0], scene, accel, real)   # noqa: E731
    else:
        backend = get_backend(device)
        if not backend.raw_kernels:
            return None
        make = lambda: CudaTracer(backend, scene, accel, real)   # noqa: E731
    if not fits(scene, accel):
        logging.getLogger(__name__).warning("scene too large for the GPU tracer's 32-bit indices; tracing on the CPU")
        return None
    if accel.depth + 2 > MAX_STACK:
        logging.getLogger(__name__).warning("BVH depth %d exceeds the GPU tracer's stack; tracing on the CPU",
                                            accel.depth)
        return None
    return make()
