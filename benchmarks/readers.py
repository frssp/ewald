"""ewald's native readers against ASE and other trajectory libraries, on generated files.

    python benchmarks/readers.py                    # every library that is installed
    python benchmarks/readers.py --libs ewald ase   # a subset
    python benchmarks/readers.py --keep /tmp/bench  # keep the generated files (~750 MB) for reuse

Files: LAMMPS dump 2,000 atoms x 2,000 frames and 1M atoms x 5 frames, extxyz 1,000 atoms x
2,000 frames with forces, XDATCAR 256 atoms x 20,000 frames. Each library runs in its own
process. "open" is the time until the frame count is known (ASE has no index, so that is a full
parse); "random" is the mean over up to 5 distinct uncached frames from the back half of the
file; "full" reads every frame in order; "RSS" is the peak growth after imports. Results for
2026-10-19 are in docs/developer/readers.md.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

FILES = {  # name: (kind, libraries that read it)
    "long.lammpstrj": ("dump", ["ewald", "ovito", "chemfiles", "mdanalysis", "ase"]),
    "big.lammpstrj": ("dump", ["ewald", "ovito", "mdanalysis", "ase"]),  # chemfiles 0.10.4 segfaults
    "long.extxyz": ("xyz", ["ewald", "ovito", "chemfiles", "mdanalysis", "oxyz", "extxyz", "ase"]),
    "XDATCAR": ("xdatcar", ["ewald", "ovito", "ase"]),
}
MODULES = {"ewald": "ewald", "ase": "ase", "ovito": "ovito", "chemfiles": "chemfiles",
           "mdanalysis": "MDAnalysis", "oxyz": "oxyz", "extxyz": "extxyz"}


def generate(root, rng):
    sys.path.insert(0, os.path.dirname(__file__))
    from large_files import write_dump, write_xdatcar

    def make(name, fn, *args):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            fn(path, *args, rng)

    make("long.lammpstrj", write_dump, 2_000, 2_000)
    make("big.lammpstrj", write_dump, 1_000_000, 5)
    make("long.extxyz", write_extxyz, 1_000, 2_000)
    make("XDATCAR", write_xdatcar, 256, 20_000)


def write_extxyz(path, natoms, nframes, rng):
    a = (natoms / 0.085) ** (1 / 3)
    species = np.where(np.arange(natoms) % 2, "Cu", "Au")
    with open(path, "w") as fh:
        for _ in range(nframes):
            fh.write(f'{natoms}\nLattice="{a} 0 0 0 {a} 0 0 0 {a}" Properties=species:S:1:pos:R:3:forces:R:3 '
                     f'energy={-3.5 * natoms + rng.random():.6f} pbc="T T T"\n')
            values = np.column_stack([rng.random((natoms, 3)) * a, rng.standard_normal((natoms, 3))])
            np.savetxt(fh, np.column_stack([species, np.char.mod("%.8f", values)]), fmt="%s")


def opener(lib, kind, path):
    """(open() -> (handle, nframes), read(handle, i) -> positions, full(handle) or None)."""
    if lib == "ewald":
        from ewald.io import open_reader

        def open_():
            r = open_reader(path)
            return r, len(r)
        return open_, lambda h, i: h.read(i).particles["Position"], None
    if lib == "ase":
        import ase.io

        fmt = {"dump": "lammps-dump-text", "xyz": "extxyz", "xdatcar": "vasp-xdatcar"}[kind]
        full = lambda h: len(ase.io.read(path, index=":", format=fmt))  # noqa: E731
        return (lambda: (None, full(None)), lambda h, i: ase.io.read(path, index=i, format=fmt).positions, full)
    if lib == "ovito":
        from ovito.io import import_file

        def open_():
            p = import_file(path, multiple_frames=True)
            return p, p.source.num_frames
        return open_, lambda h, i: np.asarray(h.compute(i).particles.positions), None
    if lib == "chemfiles":
        import chemfiles

        def open_():
            t = chemfiles.Trajectory(path, "r", {"dump": "LAMMPS", "xyz": "XYZ"}[kind])
            return t, t.nsteps
        return open_, lambda h, i: np.array(h.read_step(i).positions), None
    if lib == "mdanalysis":
        import MDAnalysis

        def open_():
            fmt = {"dump": "LAMMPSDUMP", "xyz": "XYZ"}[kind]
            u = MDAnalysis.Universe(path, format=fmt, topology_format=fmt)
            return u, len(u.trajectory)

        def full(h):
            return sum(1 for _ in h.trajectory)
        return open_, lambda h, i: h.trajectory[i].positions.copy(), full
    if lib == "oxyz":
        import oxyz

        return (lambda: (None, oxyz.scan(path).n_frames), lambda h, i: oxyz.read(path, index=i).positions,
                lambda h: len(oxyz.read(path)))
    if lib == "extxyz":
        import extxyz

        return (lambda: (None, sum(1 for _ in extxyz.iread_dicts(path))),
                lambda h, i: extxyz.read_dicts(path, index=i).arrays["pos"], lambda h: len(extxyz.read_dicts(path)))
    raise ValueError(lib)


def peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if sys.platform != "darwin" else kb / 2**20


def worker(lib, kind, path):
    open_, read, full = opener(lib, kind, path)
    base = peak_rss_mb()
    t = time.perf_counter()
    handle, n = open_()
    out = {"open": time.perf_counter() - t}
    read(handle, 0)
    back = np.arange(max(n // 2, 1), n)
    picks = np.random.default_rng(1).choice(back, min(5, len(back)), replace=False) if len(back) else [0]
    t = time.perf_counter()
    for i in picks:
        read(handle, int(i))
    out["random"] = (time.perf_counter() - t) / len(picks)
    t = time.perf_counter()
    if full:
        full(handle)
    else:
        for i in range(n):
            read(handle, i)
    out["full"] = time.perf_counter() - t
    out["rss"] = peak_rss_mb() - base
    print(json.dumps(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(*sys.argv[2:5])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libs", nargs="+", default=list(MODULES), choices=list(MODULES))
    ap.add_argument("--keep", help="directory for the generated files (reused if present)")
    args = ap.parse_args()
    import importlib.util

    libs = [lib for lib in args.libs if importlib.util.find_spec(MODULES[lib])]
    tmp = None if args.keep else tempfile.TemporaryDirectory()
    root = args.keep or tmp.name
    os.makedirs(root, exist_ok=True)
    generate(root, np.random.default_rng(0))
    for name, (kind, readers) in FILES.items():
        path = os.path.join(root, name)
        print(f"\n{name} ({os.path.getsize(path) / 2**20:.0f} MB)")
        print(f"  {'library':<11s} {'open':>9s} {'random':>10s} {'full':>9s} {'RSS':>8s}")
        for lib in (r for r in readers if r in libs):
            proc = subprocess.run([sys.executable, __file__, "--worker", lib, kind, path],
                                  capture_output=True, text=True)
            if proc.returncode:
                print(f"  {lib:<11s} failed (exit {proc.returncode}) {proc.stderr.strip()[-120:]}")
                continue
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            print(f"  {lib:<11s} {r['open']:8.2f}s {r['random'] * 1e3:8.1f}ms {r['full']:8.2f}s {r['rss']:6.0f}MB")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
