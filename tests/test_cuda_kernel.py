"""Run the CUDA C neighbor kernel on the CPU.

The kernel source is compiled with the host C++ compiler behind a tiny shim
(``__global__`` removed, a serial loop standing in for the thread grid), and
CuPy's ``RawKernel`` is replaced by a ctypes wrapper. Everything else, from
dispatch in ``find_pairs`` to the prefix sums in ``cell_list_pairs``, runs as it
does on a GPU, with numpy standing in for CuPy.
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

import ewald.cuda_kernels as ck
from ewald.backend import Backend
from ewald.data import Cell
from ewald.neighbors import find_pairs

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

SHIM = r"""
#include <math.h>
struct D3 { long long x; };
static D3 blockIdx, threadIdx, blockDim;
#define __global__
%s
extern "C" void host_launch(long long nblocks, long long nthreads,
    const double* fs, const long long* bs, const long long* starts, const long long* order,
    const double* mat, const int* pbc, const long long* nbins, const double cutoff2,
    const int half, const long long n, long long* offsets, long long* out_i,
    long long* out_j, double* out_d, const int fill)
{
    blockDim.x = nthreads;
    for (long long b = 0; b < nblocks; ++b)
        for (long long t = 0; t < nthreads; ++t) {
            blockIdx.x = b; threadIdx.x = t;
            ewald_pairs(fs, bs, starts, order, mat, pbc, nbins, cutoff2, half, n,
                        offsets, out_i, out_j, out_d, fill);
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
def host_kernel(tmp_path_factory):
    d = tmp_path_factory.mktemp("cuda_on_cpu")
    (d / "k.cpp").write_text(SHIM % ck.PAIRS_SOURCE)
    subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-o", str(d / "k.so"), str(d / "k.cpp")],
                   check=True)
    lib = ctypes.CDLL(str(d / "k.so"))
    lib.host_launch.restype = None
    return lib


@pytest.fixture
def fake_cuda(host_kernel, monkeypatch):
    launches = []

    def call(grid, block, args):
        assert len(args) == 15
        launches.append((grid, block))
        conv = []
        for a in args:
            if isinstance(a, np.ndarray):
                assert a.flags.c_contiguous
                conv.append(ctypes.c_void_p(a.ctypes.data))
            elif isinstance(a, np.float64):
                conv.append(ctypes.c_double(a))
            elif isinstance(a, np.int32):
                conv.append(ctypes.c_int(a))
            elif isinstance(a, np.int64):
                conv.append(ctypes.c_longlong(a))
            else:
                raise TypeError(f"untyped kernel argument {a!r}")  # CuPy would reject it too
        host_kernel.host_launch(ctypes.c_longlong(grid[0]), ctypes.c_longlong(block[0]), *conv)

    monkeypatch.setattr(ck, "_pairs_kernel", lambda cp: call)
    be = Backend("fake-cuda", _NumpyAsCuPy(), raw_kernels=True)
    be.launches = launches
    return be


TRICLINIC = np.array([[14.0, 0, 0], [4.0, 13.0, 0], [-3.0, 2.5, 15.0]])


@pytest.mark.parametrize("pbc", [(True, True, True), (False, True, False), (False, False, False)])
@pytest.mark.parametrize("half", [False, True])
def test_cuda_kernel_matches_brute_force(rng, fake_cuda, pbc, half):
    cell = Cell(TRICLINIC, np.zeros(3), pbc)
    pos = rng.random((1000, 3)) @ TRICLINIC
    ref = [np.asarray(a) for a in find_pairs(pos, cell, 3.1, method="brute", half=half)]
    order = np.lexsort((ref[1], ref[0]))
    ref = [a[order] for a in ref]
    out = [np.asarray(a) for a in find_pairs(pos, cell, 3.1, backend=fake_cuda, half=half)]
    assert len(fake_cuda.launches) == 2           # auto picked the kernel: count + fill
    assert fake_cuda.launches[0][0] == (8,)       # ceil(1000 / 128) blocks
    np.testing.assert_array_equal(out[0], ref[0])
    np.testing.assert_array_equal(out[1], ref[1])
    np.testing.assert_allclose(out[2], ref[2], atol=1e-12)
    np.testing.assert_allclose(out[3], ref[3], atol=1e-12)


def test_cuda_kernel_no_pairs(fake_cuda):
    cell = Cell(np.diag([30.0] * 3))
    pos = np.array([[x, y, z] for x in range(0, 30, 6) for y in range(0, 30, 6)
                    for z in range(0, 30, 6)], dtype=float)
    i, j, d, dist = find_pairs(pos, cell, 2.0, backend=fake_cuda, method="cuda")
    assert len(i) == len(j) == len(d) == len(dist) == 0


def test_cuda_method_needs_cuda_backend(rng):
    with pytest.raises(ValueError, match="CUDA backend"):
        find_pairs(rng.random((100, 3)) * 20, Cell(np.diag([20.0] * 3)), 3.0, method="cuda")
