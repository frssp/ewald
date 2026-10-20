"""Analysis-modifier and ray-tracer timings.

    python benchmarks/analysis.py                      # 108k-atom fcc Cu, every modifier, ray tracer
    python benchmarks/analysis.py --size 40 --repeat 3 # 256k atoms
    python benchmarks/analysis.py --skip-render        # modifiers only

Modifiers run on a thermally displaced fcc Cu crystal of 4*size^3 atoms (size=30 -> 108k atoms).
Times are the best of ``--repeat`` runs after one warm-up run, so numba compile and cache loading
are excluded. The ray tracer renders the SrTiO3 example with TiO6 polyhedra at each quality preset.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ewald import DataCollection, Pipeline, StaticSource
from ewald import modifiers as M
from ewald.data import Cell

MODIFIERS = {
    "CoordinationAnalysis": lambda: M.CoordinationAnalysis(3.0),
    "CreateBonds": lambda: M.CreateBonds(3.0),
    "ClusterAnalysis": lambda: M.ClusterAnalysis(3.0),
    "CommonNeighborAnalysis": lambda: M.CommonNeighborAnalysis(),
    "PolyhedralTemplateMatching": lambda: M.PolyhedralTemplateMatching(),
    "CentroSymmetry": lambda: M.CentroSymmetry(),
    "VoronoiAnalysis": lambda: M.VoronoiAnalysis(),
    "SteinhardtOrderParameters": lambda: M.SteinhardtOrderParameters(),
    "AcklandJones": lambda: M.AcklandJones(),
}


def fcc_cu(size: int, seed: int = 0) -> DataCollection:
    a = 3.615
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    grid = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
    pos = ((grid + basis).reshape(-1, 3) * a)
    pos = (pos + 0.08 * np.random.default_rng(seed).standard_normal(pos.shape)) % (size * a)
    d = DataCollection(cell=Cell(np.eye(3) * size * a))
    d.particles["Position"] = pos
    d.particles["Particle Type"] = np.full(len(pos), "Cu")
    return d


def best(fn, repeat: int) -> float:
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def bench_modifiers(size: int, repeat: int):
    data = fcc_cu(size)
    print(f"fcc Cu, {data.count:,} atoms, best of {repeat}")
    for name, make in MODIFIERS.items():
        def run():
            Pipeline(StaticSource([data]), [make()]).compute(0)
        print(f"  {name:28s} {best(run, repeat) * 1000:9.1f} ms", flush=True)


def bench_raytrace(repeat: int):
    from ewald import examples, load, render

    pipe = load(examples.path("POSCAR.SrTiO3"))
    pipe.add(M.CreateBonds(mode="pairwise", pairs={("Ti", "O"): 2.2}))
    pipe.add(M.CoordinationPolyhedra(center_types=["Ti"], transparency=0.25))
    data = pipe.compute(0)
    print(f"ray tracer, SrTiO3 + TiO6 polyhedra, 1600x1200, best of {repeat}")
    for quality in ("draft", "standard", "publication"):
        t = best(lambda: render.render(data, None, renderer="raytrace", quality=quality), repeat)
        print(f"  {quality:28s} {t:9.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=30, help="fcc cells per edge (4*size^3 atoms)")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    bench_modifiers(args.size, args.repeat)
    if not args.skip_render:
        bench_raytrace(1)


if __name__ == "__main__":
    main()
