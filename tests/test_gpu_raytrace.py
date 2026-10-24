"""Run the GPU ray tracers without a GPU.

CUDA: the kernel source (``ewald.raytrace_gpu.cuda_source``) is compiled with the host C++
compiler behind a shim (``__global__``/``__device__`` removed, a serial loop standing in for the
thread grid) and CuPy's ``RawKernel`` is replaced by a ctypes call, as in test_cuda_kernel.py.
OpenCL: the same body with the OpenCL prelude runs through pyopencl on pocl's CPU device, when
both are installed (``pip install "pyopencl[pocl]"``).
In double precision the port must give the numba tracer's image; in float, the precision a GPU
uses, it must stay within what the eye cannot see, also with float sums (Intel Arc).
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

import ewald.raytrace as rt
import ewald.raytrace_gpu as rc
from ewald.backend import Backend

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
            ewald_raytrace((const REAL*)a[0], (const int*)a[1], (const int*)a[2], *(long long*)a[3],
                           (const REAL*)a[4], (const int*)a[5], *(long long*)a[6],
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
        src.write_text(SHIM % rc.cuda_source(real), encoding="utf-8")
        so = d / f"rt_{real}.so"
        # no FMA contraction: numba does not fuse either, and the double image must match it
        subprocess.run(["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.host_launch.restype = None
        libs[real] = lib
    return libs


def _cuda_on_cpu(request, host_kernels, monkeypatch):
    real = request.param.split("-")[1]
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


def _opencl_on_pocl(request, monkeypatch):
    pytest.importorskip("pyopencl")
    devices = [d for d in rc.opencl_devices() if d.double_fp_config]
    if not devices:
        pytest.skip("needs an OpenCL device with double precision (pocl)")
    real, fp64 = {"double": ("double", None), "float": ("float", None), "float-sums": ("float", False)}[
        request.param.split("-", 1)[1]]
    launches = []
    trace = rc.OpenCLTracer.trace

    def counted(self, *args):
        launches.append((None, None))
        trace(self, *args)

    monkeypatch.setattr(rc, "opencl_devices", lambda gpu_only=False: devices)
    monkeypatch.setattr(rc, "OPENCL_FP64", fp64)
    monkeypatch.setattr(rc.OpenCLTracer, "trace", counted)
    monkeypatch.setattr(rt, "GPU_REAL", real)

    class Device(str):
        pass

    dev = Device("opencl")
    dev.launches, dev.real = launches, real
    return dev


@pytest.fixture(params=["cuda-double", "cuda-float", "opencl-double", "opencl-float", "opencl-float-sums"])
def fake_gpu(request, monkeypatch):
    """A GPU device for ``raytrace(device=...)``: the CUDA source on the host compiler, or the
    OpenCL source on pocl; ``.launches`` records the kernel calls, ``.real`` the precision."""
    if request.param.startswith("cuda"):
        if shutil.which("g++") is None:
            pytest.skip("needs g++")
        return _cuda_on_cpu(request, request.getfixturevalue("host_kernels"), monkeypatch)
    return _opencl_on_pocl(request, monkeypatch)


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
    sc.add_cylinders([[-2.0, 1.2, -1.0]], [[-1.4, 1.6, -0.1]], 0.25, (0.3, 0.5, 0.9), flat_caps=(True, True))
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
    assert fake_gpu.launches and fake_gpu.launches[0][1] in ((rc.THREADS,), None)
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


def test_a_pass_is_split_into_short_launches(fake_gpu, monkeypatch):
    """Windows and macOS stop a GPU kernel that runs for seconds, so a pass goes out in launches
    of a few rows; the picture does not depend on how it is split."""
    sc = busy_scene()
    kw = dict(samples=4, ao_samples=4)
    whole = rt.raytrace(sc, CAM, (40, 30), device=fake_gpu, **kw)
    monkeypatch.setattr(rc, "RAYS_PER_LAUNCH", 40 * 4 * 6 * 7)    # 7 rows a launch
    parts = list(rc._launches([40, 30, 1, 4, 4, 4, 1, 0, 0, 0], np.arange(30)))
    assert [len(c) for _, c, _ in parts] == [7, 7, 7, 7, 2]
    assert all(p[8] == len(c) and n == 40 * len(c) for p, c, n in parts)
    split = rt.raytrace(sc, CAM, (40, 30), device=fake_gpu, **kw)
    assert np.array_equal(whole, split)


@pytest.mark.parametrize("platform, picks", [("win32", True), ("linux", True), ("darwin", False)])
def test_auto_takes_an_opencl_gpu_except_on_macos(monkeypatch, platform, picks):
    monkeypatch.setattr(rc, "opencl_devices", lambda gpu_only=False: ["gpu"])
    monkeypatch.setattr(rc, "OpenCLTracer", lambda device, scene, accel, real: ("opencl", device))
    monkeypatch.setattr("ewald.backend.gpu_available", lambda: False)
    monkeypatch.setattr(rc.sys, "platform", platform)
    sc = busy_scene()
    got = rc.tracer(sc, rt.Accel(sc), "auto")
    assert got == (("opencl", "gpu") if picks else None)
    assert rc.tracer(sc, rt.Accel(sc), "opencl") == ("opencl", "gpu")   # asked for by name: always


def test_gpu_codes_are_32_bit_and_keep_kind_shadow_and_index():
    codes = np.array([0 << 40 | 5, 1 << 40 | rt.NO_SHADOW | 7, 2 << 40 | (rc.GPU_NO_SHADOW - 1)], np.int64)
    g = rc.gpu_codes(codes)
    assert g.dtype == np.int32
    assert list(g >> rc.GPU_KIND_SHIFT) == [0, 1, 2]
    assert list((g & rc.GPU_NO_SHADOW) != 0) == [False, True, False]
    assert list(g & (rc.GPU_NO_SHADOW - 1)) == [5, 7, rc.GPU_NO_SHADOW - 1]


def test_scene_too_large_for_32_bit_codes_stays_on_cpu(fake_gpu, monkeypatch):
    monkeypatch.setattr(rc, "GPU_NO_SHADOW", 4)     # pretend: at most 4 primitives of a kind
    rt.raytrace(busy_scene(), CAM, (16, 12), samples=1, ao_samples=0, device=fake_gpu)
    assert not fake_gpu.launches
