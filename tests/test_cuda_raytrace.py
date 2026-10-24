"""Run the CUDA ray tracer on the CPU.

The kernel source (``ewald.raytrace_cuda.RAYTRACE_SOURCE``) is compiled with the host C++
compiler behind a shim (``__global__``/``__device__`` removed, a serial loop standing in for the
thread grid) and CuPy's ``RawKernel`` is replaced by a ctypes call, as in test_cuda_kernel.py.
In double precision the port must give the numba tracer's image; in float, the precision the
GPU uses, it must stay within what the eye cannot see.
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

import ewald.raytrace as rt
import ewald.raytrace_cuda as rc
from ewald.backend import Backend

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

SHIM = r"""
#include <math.h>
struct D3 { long long x; };
static D3 blockIdx, threadIdx, blockDim;
#define __global__
#define __device__
%s
extern "C" void host_launch(long long nblocks, long long nthreads, void** a)
{
    blockDim.x = nthreads;
    for (long long b = 0; b < nblocks; ++b)
        for (long long t = 0; t < nthreads; ++t) {
            blockIdx.x = b; threadIdx.x = t;
            ewald_raytrace((const REAL*)a[0], (const long long*)a[1], (const long long*)a[2], *(long long*)a[3],
                           (const REAL*)a[4], (const long long*)a[5], *(long long*)a[6],
                           (const REAL*)a[7], (const REAL*)a[8], (const REAL*)a[9],
                           (const REAL*)a[10], (const REAL*)a[11], (const REAL*)a[12], (const REAL*)a[13],
                           (const double*)a[14], (const long long*)a[15], (const long long*)a[16],
                           (const unsigned char*)a[17], (double*)a[18], (double*)a[19], (long long*)a[20],
                           (unsigned long long*)a[21], (double*)a[22]);
        }
}
"""


class _NumpyAsCuPy:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture(scope="module")
def host_kernels(tmp_path_factory):
    d = tmp_path_factory.mktemp("cuda_raytrace_on_cpu")
    libs = {}
    for real in ("double", "float"):
        src = d / f"rt_{real}.cpp"
        src.write_text(f"#define REAL {real}\n" + SHIM % rc.RAYTRACE_SOURCE, encoding="utf-8")
        so = d / f"rt_{real}.so"
        # no FMA contraction: numba does not fuse either, and the double image must match it
        subprocess.run(["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.host_launch.restype = None
        libs[real] = lib
    return libs


@pytest.fixture(params=["double", "float"])
def fake_gpu(request, host_kernels, monkeypatch):
    real = request.param
    launches = []

    def kernel(cp, which):
        assert which == real

        def call(grid, block, args):
            assert len(args) == 23
            launches.append((grid, block))
            keep = []
            for x in args:
                if isinstance(x, np.ndarray):
                    assert x.flags.c_contiguous
                    keep.append(x)
                elif isinstance(x, np.int64):
                    keep.append(np.array([x], np.int64))
                else:
                    raise TypeError(f"untyped kernel argument {x!r}")  # CuPy would reject it too
            ptrs = (ctypes.c_void_p * len(keep))(*[k.ctypes.data for k in keep])
            host_kernels[real].host_launch(ctypes.c_longlong(grid[0]), ctypes.c_longlong(block[0]), ptrs)
        return call

    monkeypatch.setattr(rc, "_kernel", kernel)
    monkeypatch.setattr(rt, "GPU_REAL", real)
    be = Backend("fake-cuda", _NumpyAsCuPy(), raw_kernels=True)
    be.synchronize = lambda: None
    be.launches = launches
    be.real = real
    return be


CAM = {"position": (6.0, 4.0, 9.0), "focal_point": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0),
       "parallel_projection": True, "parallel_scale": 2.6}


def busy_scene():
    """Every primitive kind and path: spheres, bonds, opaque and see-through triangles in the grid,
    a fine surface mesh in the BVH, shadow-free cell lines."""
    sc = rt.Scene()
    rng = np.random.default_rng(3)
    c = rng.uniform(-1.5, 1.5, (12, 3))
    sc.add_spheres(c, rng.uniform(0.25, 0.5, 12), rng.uniform(0.2, 1.0, (12, 3)))
    sc.add_cylinders(c[:6], c[6:], 0.12, (0.7, 0.7, 0.7))
    sc.add_cylinders([[-2, -2, -2]], [[2, -2, -2]], 0.03, (0, 0, 0), cast_shadows=False)
    sc.add_triangles([[-2.5, -1.8, -2.5]], [[2.5, -1.8, -2.5]], [[0, -1.8, 2.5]], (0.9, 0.9, 0.9))
    sc.add_triangles([[-1, 1, 1]], [[1, 1, 1]], [[0, 2, 1]], (0.2, 0.4, 1.0), alpha=0.4)
    # a fine sphere mesh, enough triangles for the BVH
    th, ph = np.meshgrid(np.linspace(0, np.pi, 26), np.linspace(0, 2 * np.pi, 46), indexing="ij")
    pts = 0.6 * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1) + [1.6, 1.4, 0.0]
    a, b, cc, d = pts[:-1, :-1], pts[1:, :-1], pts[1:, 1:], pts[:-1, 1:]
    tri_a = np.concatenate([a.reshape(-1, 3), a.reshape(-1, 3)])
    tri_b = np.concatenate([b.reshape(-1, 3), cc.reshape(-1, 3)])
    tri_c = np.concatenate([cc.reshape(-1, 3), d.reshape(-1, 3)])
    sc.add_triangles(tri_a, tri_b, tri_c, (1.0, 0.6, 0.1), surface=True)
    return sc


def test_scene_uses_grid_and_bvh():
    accel = rt.Accel(busy_scene())
    assert len(accel.items) and len(accel.nodes) and accel.depth + 2 <= rc.MAX_STACK


@pytest.mark.parametrize("ortho", [True, False])
def test_gpu_image_matches_cpu(fake_gpu, ortho):
    sc = busy_scene()
    cam = dict(CAM, parallel_projection=ortho, view_angle=40.0)
    kw = dict(samples=4, ao_samples=4)
    cpu = rt.raytrace(sc, cam, (48, 36), **kw)
    gpu = rt.raytrace(sc, cam, (48, 36), device=fake_gpu, **kw)
    assert fake_gpu.launches and fake_gpu.launches[0][1] == (rc.THREADS,)
    diff = np.abs(gpu - cpu).max(-1)
    if fake_gpu.real == "double":   # the same rays: a pixel may differ only on a knife edge
        assert (diff < 1e-9).mean() > 0.999 and diff.max() < 0.3
    else:   # float geometry and shading: invisible differences, a few edge pixels apart
        assert np.median(diff) < 2e-3 and (diff < 0.02).mean() > 0.99


def test_gpu_adaptive_and_progress(fake_gpu):
    sc = busy_scene()
    seen = []
    kw = dict(samples=16, ao_samples=4, noise_threshold=0.01, min_samples=4)
    cpu = rt.raytrace(sc, CAM, (40, 30), **kw)
    gpu = rt.raytrace(sc, CAM, (40, 30), device=fake_gpu, progress=lambda d, t: seen.append(d), **kw)
    assert len(fake_gpu.launches) > rt.PROGRESS_STEPS        # batches of several passes
    assert seen[-1] == 30 and seen == sorted(seen)
    tol = 1e-9 if fake_gpu.real == "double" else 0.02
    assert (np.abs(gpu - cpu).max(-1) < tol).mean() > 0.98
    with pytest.raises(rt.RenderCancelled):
        rt.raytrace(sc, CAM, (40, 30), device=fake_gpu, progress=lambda d, t: False, **kw)


def test_cpu_device_and_deep_bvh_stay_on_cpu(fake_gpu, monkeypatch):
    sc = busy_scene()
    rt.raytrace(sc, CAM, (16, 12), samples=1, ao_samples=0, device="cpu")
    assert not fake_gpu.launches
    monkeypatch.setattr(rc, "MAX_STACK", 2)
    rt.raytrace(sc, CAM, (16, 12), samples=1, ao_samples=0, device=fake_gpu)
    assert not fake_gpu.launches


def test_render_passes_device_to_the_ray_tracer(fake_gpu):
    from ewald import DataCollection
    from ewald.data import Cell
    from ewald.render import render

    data = DataCollection([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], ["Cu", "O"], Cell(np.eye(3) * 4))
    img = render(data, None, size=(40, 30), renderer="raytrace", quality="draft", device=fake_gpu)
    assert fake_gpu.launches and img.ndim == 3
