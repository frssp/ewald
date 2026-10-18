"""Neighbor list / RDF / MSD timings on CPU and GPU.

    python benchmarks/neighbors.py                 # CPU, plus CUDA if available
    python benchmarks/neighbors.py --sizes 20 40 60 --cutoff 3.0 --repeat 5

Each size n builds an fcc Cu crystal of 4*n^3 atoms (n=40 -> 256k atoms).
Times are the best of ``--repeat`` runs after one warm-up run (numba compile,
CUDA kernel compile and memory pools are excluded).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from ase.build import bulk

from ewald import DataCollection, gpu_available
from ewald.backend import get_backend
from ewald.kernels import msd_fft, rdf
from ewald.neighbors import HAVE_NUMBA, find_pairs


def best_time(fn, backend, repeat):
    fn()
    backend.synchronize()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        backend.synchronize()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--cutoff", type=float, default=3.0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--devices", nargs="+", default=None, help="cpu and/or cuda (default: all available)")
    ap.add_argument("--msd-frames", type=int, default=200)
    args = ap.parse_args()
    devices = args.devices or (["cpu", "cuda"] if gpu_available() else ["cpu"])
    methods = {"cpu": (["numba"] if HAVE_NUMBA else []) + ["xp"], "cuda": ["cuda", "xp"]}

    print(f"{'atoms':>9} {'device':>6} {'task':>18} {'time [s]':>10}")
    for size in args.sizes:
        atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((size,) * 3)
        atoms.rattle(0.05, seed=1)
        data = DataCollection.from_ase(atoms)
        n = data.count
        for dev in devices:
            be = get_backend(dev)
            for m in methods[dev]:
                for half in (False, True):
                    t = best_time(lambda: find_pairs(data.positions, data.cell, args.cutoff, be,
                                                     half=half, method=m), be, args.repeat)
                    print(f"{n:>9} {dev:>6} {('pairs ' + m + (' half' if half else '')):>18} {t:>10.3f}")
            t = best_time(lambda: rdf(data.positions, data.cell, args.cutoff, 200, data.types, be),
                          be, args.repeat)
            print(f"{n:>9} {dev:>6} {'rdf (auto)':>18} {t:>10.3f}")
            traj = np.random.default_rng(0).normal(size=(args.msd_frames, min(n, 50_000), 3)).cumsum(0)
            xt = be.asarray(traj)
            t = best_time(lambda: msd_fft(xt, be), be, args.repeat)
            print(f"{traj.shape[1]:>9} {dev:>6} {f'msd {args.msd_frames} frames':>18} {t:>10.3f}")


if __name__ == "__main__":
    main()
