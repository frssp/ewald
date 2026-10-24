"""The ray tracer on an NVIDIA GPU: :func:`ewald.raytrace.raytrace` with ``device="cuda"``.

``RAYTRACE_SOURCE`` is a line-by-line port of the numba kernels in :mod:`ewald.raytrace`
(``_render``, ``_shade``, ``_trace``, the grid DDA, the BVH walk and the hit tests) to CUDA C,
one thread per pixel, launched through CuPy ``RawKernel``. The acceleration structures are still
built on the CPU and uploaded. ``REAL`` is the floating-point type of the geometry and the
shading: ``float`` on the GPU (consumer GPUs run double precision at 1/32 to 1/64 of the float
rate), ``double`` in the tests, where the source is compiled with the host C++ compiler behind a
shim and must give the CPU tracer's image (tests/test_cuda_raytrace.py). The random streams and
the R2 ambient-occlusion sequence are the CPU tracer's, in 64-bit integers and doubles, so a
pixel takes the same rays on both devices.
"""
from __future__ import annotations

import numpy as np

MAX_STACK = 64     # BVH traversal stack per thread; deeper trees fall back to the CPU tracer

RAYTRACE_SOURCE = r"""
#ifndef REAL
#define REAL float
#endif
#define NO_SHADOW (1LL << 39)
#define INDEX_MASK (NO_SHADOW - 1)
#define MAX_STACK %(max_stack)d

__device__ static inline REAL rmin(REAL a, REAL b) { return a < b ? a : b; }
__device__ static inline REAL rmax(REAL a, REAL b) { return a > b ? a : b; }

__device__ static inline double rand01(unsigned long long* state)
{
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (double)((x * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ static inline REAL hit_sphere(const REAL* o, const REAL* d, const REAL* sph, long long i)
{
    const REAL* s = sph + 4 * i;
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

__device__ static inline REAL hit_cylinder(const REAL* o, const REAL* d, const REAL* cyl, long long i)
{
    const REAL* c = cyl + 7 * i;
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
    REAL A = px * px + py * py + pz * pz;
    if (A < (REAL)1e-14) return -1;
    REAL B = px * qx + py * qy + pz * qz;
    REAL C = qx * qx + qy * qy + qz * qz - r * r;
    REAL disc = B * B - A * C;
    if (disc < 0) return -1;
    REAL s = sqrt(disc);
    REAL ts[2] = {(-B - s) / A, (-B + s) / A};
    for (int k = 0; k < 2; ++k) {
        REAL t = ts[k];
        if (t > (REAL)1e-6) {
            REAL h = od + t * dd;
            if (0 <= h && h <= L) return t;
        }
    }
    return -1;
}

__device__ static inline REAL hit_triangle(const REAL* o, const REAL* d, const REAL* tri, long long i)
{
    const REAL* T = tri + 10 * i;
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

__device__ static inline REAL hit_code(const REAL* o, const REAL* d, long long code, int any_hit,
                                       const REAL* sph, const REAL* cyl, const REAL* tri)
{
    long long kind = code >> 40;
    long long i = code & INDEX_MASK;
    if (kind == 0) return hit_sphere(o, d, sph, i);
    if (kind == 1) return hit_cylinder(o, d, cyl, i);
    if (any_hit && tri[10 * i + 9] < (REAL)0.999) return -1;
    return hit_triangle(o, d, tri, i);
}

__device__ static inline REAL inv_dir(REAL x)
{
    if (fabs(x) < (REAL)1e-30) return (REAL)1e30;
    return 1 / x;
}

__device__ static inline REAL hit_box(const REAL* box, long long k, REAL ox, REAL oy, REAL oz,
                                      REAL ix, REAL iy, REAL iz, REAL tmax)
{
    const REAL* b = box + 8 * k;
    REAL ta = (b[0] - ox) * ix, tb = (b[3] - ox) * ix;
    REAL tn = rmin(ta, tb), tf = rmax(ta, tb);
    ta = (b[1] - oy) * iy; tb = (b[4] - oy) * iy;
    tn = rmax(tn, rmin(ta, tb)); tf = rmin(tf, rmax(ta, tb));
    ta = (b[2] - oz) * iz; tb = (b[5] - oz) * iz;
    tn = rmax(tn, rmin(ta, tb)); tf = rmin(tf, rmax(ta, tb));
    tn = rmax(tn, (REAL)0);
    return (tn <= tf && tn < tmax) ? tn : -1;
}

__device__ static inline void dda_axis(REAL o, REAL d, REAL lo, REAL size, long long dim, REAL t0,
                                       long long* c, long long* step, REAL* next, REAL* delta)
{
    long long cc = (long long)((o + d * (t0 + (REAL)1e-9) - lo) / size);
    cc = cc < 0 ? 0 : (cc > dim - 1 ? dim - 1 : cc);
    *c = cc;
    if (d > 0) { *step = 1; *next = (lo + (cc + 1) * size - o) / d; *delta = size / d; }
    else if (d < 0) { *step = -1; *next = (lo + cc * size - o) / d; *delta = -size / d; }
    else { *step = 0; *next = (REAL)1e30; *delta = (REAL)1e30; }
}

__device__ static long long trace_grid(const REAL* o, const REAL* d, REAL tmax, const REAL* grid,
                                       const long long* starts, const long long* items, long long n_items,
                                       const REAL* sph, const REAL* cyl, const REAL* tri, int any_hit, REAL* t_out)
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
    long long dx = (long long)grid[6], dy = (long long)grid[7], dz = (long long)grid[8];
    long long cx, sx, cy, sy, cz, sz;
    REAL nx, ex, ny, ey, nz, ez;
    dda_axis(o[0], d[0], grid[0], grid[3], dx, t0, &cx, &sx, &nx, &ex);
    dda_axis(o[1], d[1], grid[1], grid[4], dy, t0, &cy, &sy, &ny, &ey);
    dda_axis(o[2], d[2], grid[2], grid[5], dz, t0, &cz, &sz, &nz, &ez);
    REAL best_t = (REAL)1e30;
    long long best = -1;
    while (true) {
        long long c = (cx * dy + cy) * dz + cz;
        for (long long s = starts[c]; s < starts[c + 1]; ++s) {
            long long code = items[s];
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

__device__ static long long trace_bvh(const REAL* o, const REAL* d, REAL tmax, const REAL* box, long long n_nodes,
                                      const long long* items, const REAL* sph, const REAL* cyl, const REAL* tri,
                                      int any_hit, REAL* t_out)
{
    if (n_nodes == 0) return -1;
    long long stack[MAX_STACK];
    REAL stack_t[MAX_STACK];
    REAL ox = o[0], oy = o[1], oz = o[2];
    REAL ix = inv_dir(d[0]), iy = inv_dir(d[1]), iz = inv_dir(d[2]);
    REAL best_t = tmax;
    long long best = -1;
    REAL t_root = hit_box(box, 0, ox, oy, oz, ix, iy, iz, best_t);
    if (t_root < 0) return -1;
    stack[0] = 0; stack_t[0] = t_root;
    int sp = 1;
    while (sp > 0) {
        --sp;
        if (stack_t[sp] >= best_t) continue;
        long long node = stack[sp];
        long long n = (long long)box[8 * node + 7];
        if (n > 0) {
            long long first = (long long)box[8 * node + 6];
            for (long long s = first; s < first + n; ++s) {
                long long code = items[s];
                if (any_hit && (code & NO_SHADOW)) continue;
                REAL t = hit_code(o, d, code, any_hit, sph, cyl, tri);
                if (t > 0 && t < best_t) {
                    best_t = t; best = code;
                    if (any_hit) { *t_out = best_t; return best; }
                }
            }
        } else {
            long long left = (long long)box[8 * node + 6];
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

struct Accel {
    const REAL* grid; const long long* starts; const long long* items; long long n_items;
    const REAL* nodes; const long long* bvh_items; long long n_nodes;
    const REAL* sph; const REAL* cyl; const REAL* tri;
};

__device__ static long long trace(const REAL* o, const REAL* d, REAL tmax, const Accel& a, int any_hit, REAL* t_out)
{
    REAL tg = -1;
    long long cg = -1;
    if (a.n_items) {
        cg = trace_grid(o, d, tmax, a.grid, a.starts, a.items, a.n_items, a.sph, a.cyl, a.tri, any_hit, &tg);
        if (cg >= 0) {
            if (any_hit) { *t_out = tg; return cg; }
            tmax = tg;
        }
    }
    if (a.n_nodes) {
        REAL tb = -1;
        long long cb = trace_bvh(o, d, tmax, a.nodes, a.n_nodes, a.bvh_items, a.sph, a.cyl, a.tri, any_hit, &tb);
        if (cb >= 0) { *t_out = tb; return cb; }
    }
    *t_out = tg;
    return cg;
}

__device__ static void shade(const REAL* p, const REAL* n, const REAL* d, REAL cr, REAL cg, REAL cb,
                             const REAL* light, REAL ambient, REAL diffuse, REAL specular, REAL shininess,
                             int shadows, long long ao_samples, REAL ao_dist, const Accel& a, double* aos,
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
        long long hits = 0;
        REAL od[3];
        for (long long s = 0; s < ao_samples; ++s) {
            double g = aos[2];
            aos[2] += 1.0;
            double r1 = fmod(aos[0] + g * 0.7548776662466927, 1.0);
            double r2 = fmod(aos[1] + g * 0.5698402909980532, 1.0);
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

extern "C" __global__
void ewald_raytrace(const REAL* grid, const long long* starts, const long long* items, const long long n_items,
                    const REAL* nodes, const long long* bvh_items, const long long n_nodes,
                    const REAL* sph, const REAL* cyl, const REAL* tri,
                    const REAL* sph_rgb, const REAL* cyl_rgb, const REAL* tri_n, const REAL* tri_vrgb,
                    const double* P, const long long* I, const long long* rows, const unsigned char* active,
                    double* acc, double* sq, long long* nsamp, unsigned long long* rng, double* ao_next)
{
    const long long width = I[0], height = I[1], ortho = I[2], s1 = I[3], samples = I[4];
    const long long ao_samples = I[5], shadows = I[6], seed = I[7], n_rows = I[8], keep = I[9];
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n_rows * width) return;
    const long long py = rows[k / width], px = k %% width;
    const long long pix = py * width + px;
    if (!active[pix]) return;
    Accel a = {grid, starts, items, n_items, nodes, bvh_items, n_nodes, sph, cyl, tri};
    REAL eye[3], fwd[3], right[3], up[3], light[3];
    for (int q = 0; q < 3; ++q) {
        eye[q] = (REAL)P[q]; fwd[q] = (REAL)P[3 + q]; right[q] = (REAL)P[6 + q]; up[q] = (REAL)P[9 + q];
        light[q] = (REAL)P[14 + q];
    }
    const double half_w = P[12], half_h = P[13];
    const REAL ambient = (REAL)P[17], diffuse = (REAL)P[18], specular = (REAL)P[19], shininess = (REAL)P[20];
    const REAL bg[4] = {(REAL)P[21], (REAL)P[22], (REAL)P[23], (REAL)P[24]};
    const REAL ao_dist = (REAL)P[25];

    unsigned long long state = (unsigned long long)seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)(pix + 1);
    double aos[3];
    aos[0] = rand01(&state); aos[1] = rand01(&state); aos[2] = 0.0;
    long long s0 = keep ? nsamp[pix] : 0;
    if (s0 > 0) { state = rng[pix]; aos[2] = ao_next[pix]; }
    double acc_r = acc[4 * pix], acc_g = acc[4 * pix + 1], acc_b = acc[4 * pix + 2], acc_a = acc[4 * pix + 3];
    double acc_q = keep ? sq[pix] : 0.0;
    REAL o[3], d[3], n[3], p[3], sh[3];
    for (long long s = s0; s < s1; ++s) {
        double jx = samples > 1 ? rand01(&state) : 0.5;
        double jy = samples > 1 ? rand01(&state) : 0.5;
        double u = ((px + jx) / width * 2.0 - 1.0) * half_w;
        double v = (1.0 - (py + jy) / height * 2.0) * half_h;
        for (int q = 0; q < 3; ++q) {
            if (ortho) { o[q] = (REAL)(eye[q] + u * right[q] + v * up[q]); d[q] = fwd[q]; }
            else { o[q] = eye[q]; d[q] = (REAL)(fwd[q] + u * right[q] + v * up[q]); }
        }
        REAL nd = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int q = 0; q < 3; ++q) d[q] /= nd;
        double vr = 0, vg = 0, vb = 0, va = 0;
        double trans = 1.0;
        for (int layer = 0; layer < 16; ++layer) {
            REAL t;
            long long code = trace(o, d, (REAL)1e30, a, 0, &t);
            if (code < 0) {
                vr += trans * bg[0]; vg += trans * bg[1]; vb += trans * bg[2]; va += trans * bg[3];
                break;
            }
            long long kind = code >> 40;
            long long i = code & INDEX_MASK;
            for (int q = 0; q < 3; ++q) p[q] = o[q] + t * d[q];
            REAL alpha = 1, cr, cg, cb;
            if (kind == 0) {
                for (int q = 0; q < 3; ++q) n[q] = (p[q] - sph[4 * i + q]) / sph[4 * i + 3];
                cr = sph_rgb[3 * i]; cg = sph_rgb[3 * i + 1]; cb = sph_rgb[3 * i + 2];
            } else if (kind == 1) {
                const REAL* c = cyl + 7 * i;
                REAL ax = c[3] - c[0], ay = c[4] - c[1], az = c[5] - c[2];
                REAL L2 = ax * ax + ay * ay + az * az;
                REAL h = ((p[0] - c[0]) * ax + (p[1] - c[1]) * ay + (p[2] - c[2]) * az) / L2;
                n[0] = p[0] - (c[0] + h * ax); n[1] = p[1] - (c[1] + h * ay); n[2] = p[2] - (c[2] + h * az);
                REAL nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                for (int q = 0; q < 3; ++q) n[q] /= nn;
                cr = cyl_rgb[3 * i]; cg = cyl_rgb[3 * i + 1]; cb = cyl_rgb[3 * i + 2];
            } else {
                const REAL* T = tri + 10 * i;
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
                const REAL* N = tri_n + 9 * i;
                for (int q = 0; q < 3; ++q) n[q] = wa * N[q] + wb * N[3 + q] + wc * N[6 + q];
                REAL nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (nn < (REAL)1e-12) {
                    n[0] = e1y * e2z - e1z * e2y; n[1] = e1z * e2x - e1x * e2z; n[2] = e1x * e2y - e1y * e2x;
                    nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                }
                for (int q = 0; q < 3; ++q) n[q] /= nn;
                const REAL* V = tri_vrgb + 9 * i;
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
                  ao_dist, a, aos, sh);
            double w = trans * alpha;
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
""" % {"max_stack": MAX_STACK}

THREADS = 128
_kernels = {}


def _kernel(cp, real: str):
    if real not in _kernels:
        _kernels[real] = cp.RawKernel(f"#define REAL {real}\n" + RAYTRACE_SOURCE, "ewald_raytrace")
    return _kernels[real]


class DeviceScene:
    """The acceleration structure and the primitives of one :class:`ewald.raytrace.Scene` on the
    device, in ``real`` precision (the item codes stay 64-bit)."""

    def __init__(self, cp, scene, accel, real: str = "float"):
        dt = np.float32 if real == "float" else np.float64

        def up(a, dtype=dt):
            a = np.ascontiguousarray(np.asarray(a, dtype=dtype).ravel())
            return cp.asarray(a if len(a) else np.zeros(1, dtype))   # no zero-size device arrays

        self.args = (up(accel.grid), up(accel.starts, np.int64), up(accel.items, np.int64), np.int64(len(accel.items)),
                     up(accel.nodes), up(accel.bvh_items, np.int64), np.int64(len(accel.nodes)),
                     up(accel.sph), up(accel.cyl), up(accel.tri),
                     up(scene.sph_rgb), up(scene.cyl_rgb), up(scene.tri_n), up(scene.tri_vrgb))


def trace_rows(cp, dscene, real, params, ints, rows, active, acc, sq, nsamp, rng, ao_next):
    """One pass of :func:`ewald.raytrace._render` on the device: every ``active`` pixel of
    ``rows`` is brought to ``ints[3]`` samples. ``acc``/``sq``/``nsamp``/``rng``/``ao_next`` are
    device arrays, updated in place."""
    ints = np.asarray(ints, np.int64).copy()
    ints[8] = len(rows)
    n = len(rows) * int(ints[0])
    if n == 0:
        return
    args = dscene.args + (cp.asarray(np.ascontiguousarray(params, np.float64)), cp.asarray(ints),
                          cp.asarray(np.ascontiguousarray(rows, np.int64)), active, acc, sq, nsamp, rng, ao_next)
    _kernel(cp, real)(((n + THREADS - 1) // THREADS,), (THREADS,), args)
