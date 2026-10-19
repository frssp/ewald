"""Open / frame-switch / memory timings on synthetic large trajectories.

    python benchmarks/large_files.py                       # 2M-atom dump x 4 frames, 256-atom XDATCAR x 20k
    python benchmarks/large_files.py --atoms 5000000 --frames 3 --xdatcar-frames 50000
    python benchmarks/large_files.py --keep /tmp/bench     # keep the generated files for reuse

Files are generated (never committed): a LAMMPS ``dump custom`` with shuffled ids
(as written by MPI runs) and a VASP XDATCAR. Each stage reports wall time and the
growth of peak RSS, so the numbers show where opening and scrubbing a big file go.
"""
from __future__ import annotations

import argparse
import os
import resource
import tempfile
import time

import numpy as np


def peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if os.uname().sysname != "Darwin" else kb / 2**20


def write_dump(path, natoms, nframes, rng):
    box = (natoms / 0.0847) ** (1 / 3)  # Cu number density
    with open(path, "wb") as fh:
        for step in range(nframes):
            ids = rng.permutation(natoms) + 1
            pos = rng.random((natoms, 3)) * box
            types = rng.integers(1, 3, natoms)
            fh.write(f"ITEM: TIMESTEP\n{step * 100}\nITEM: NUMBER OF ATOMS\n{natoms}\n"
                     f"ITEM: BOX BOUNDS pp pp pp\n0 {box}\n0 {box}\n0 {box}\n"
                     "ITEM: ATOMS id type x y z c_pe\n".encode())
            chunk = 500_000
            for s in range(0, natoms, chunk):
                e = min(s + chunk, natoms)
                pe = -3.5 + 0.1 * rng.standard_normal(e - s)
                rows = np.column_stack([ids[s:e], types[s:e], pos[s:e], pe])
                np.savetxt(fh, rows, fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6g"])


def write_xdatcar(path, natoms, nframes, rng):
    counts = [natoms // 2, natoms - natoms // 2]
    a = (natoms / 0.05) ** (1 / 3)
    with open(path, "wb") as fh:
        fh.write(f"bench\n1.0\n{a} 0 0\n0 {a} 0\n0 0 {a}\nSi O\n{counts[0]} {counts[1]}\n".encode())
        base = rng.random((natoms, 3))
        for k in range(nframes):
            fh.write(f"Direct configuration= {k + 1:5d}\n".encode())
            frac = (base + 0.01 * rng.standard_normal((natoms, 3))) % 1.0
            np.savetxt(fh, frac, fmt="%.8f")


def stage(label, fn):
    rss0 = peak_rss_mb()
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    print(f"  {label:<38s} {dt:8.3f} s   peak RSS {peak_rss_mb():7.0f} MB (+{peak_rss_mb() - rss0:.0f})")
    return out


def bench(path, kind):
    import ewald

    size = os.path.getsize(path) / 2**20
    print(f"{kind}: {path} ({size:.0f} MB)")
    pipe = stage("import_file (index + checks)", lambda: ewald.import_file(path))
    n = pipe.num_frames
    data = stage("compute(0) first frame", lambda: pipe.compute(0))
    print(f"  -> {data.count} atoms, {n} frames, properties {list(data.particles)}")
    stage("compute(0) again (cached)", lambda: pipe.compute(0))
    stage(f"compute({n - 1}) last frame", lambda: pipe.compute(n - 1))
    frames = np.random.default_rng(1).integers(0, n, 5)
    t = time.perf_counter()
    for f in frames:
        pipe._cache.clear()
        pipe.compute(int(f))
    print(f"  {'random frame switch (uncached, mean)':<38s} {(time.perf_counter() - t) / len(frames):8.3f} s")
    if kind == "xdatcar":
        t = time.perf_counter()
        for f in range(min(n, 200)):
            pipe.compute(f)
        print(f"  {'sequential playback (per frame)':<38s} {(time.perf_counter() - t) / min(n, 200) * 1e3:8.2f} ms")
    cached = sum(sum(a.nbytes for a in d.particles.values()) for d in pipe._cache.values()) / 2**20
    print(f"  pipeline cache: {len(pipe._cache)} frames, {cached:.0f} MB of arrays")
    return data


def bench_render(data):
    try:
        import pyvista as pv
    except ImportError:
        return
    from ewald.render import build_scene

    print("render: build_scene + first draw (offscreen)")
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    stage("build_scene (colors, radii, impostor)", lambda: build_scene(pl, data))
    stage("first render", lambda: pl.render())
    stage("rebuild for next frame", lambda: (build_scene(pl, data), pl.render()))
    pl.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atoms", type=int, default=2_000_000)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--xdatcar-atoms", type=int, default=256)
    ap.add_argument("--xdatcar-frames", type=int, default=20_000)
    ap.add_argument("--keep", help="directory for the generated files (reused if present)")
    ap.add_argument("--render", action="store_true", help="also time VTK scene building (needs a display)")
    args = ap.parse_args()

    tmp = None if args.keep else tempfile.TemporaryDirectory()
    root = args.keep or tmp.name
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    dump = os.path.join(root, f"bench_{args.atoms}x{args.frames}.lammpstrj")
    xdat = os.path.join(root, f"XDATCAR_{args.xdatcar_atoms}x{args.xdatcar_frames}")
    if not os.path.exists(dump):
        t = time.perf_counter()
        write_dump(dump, args.atoms, args.frames, rng)
        print(f"generated {dump} in {time.perf_counter() - t:.0f} s")
    if not os.path.exists(xdat):
        write_xdatcar(xdat, args.xdatcar_atoms, args.xdatcar_frames, rng)
    print(f"baseline peak RSS {peak_rss_mb():.0f} MB\n")
    data = bench(dump, "lammps dump")
    print()
    bench(xdat, "xdatcar")
    if args.render:
        print()
        bench_render(data)
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
