"""Analysis-modifier and ray-tracer timings.

    python benchmarks/analysis.py                      # 108k-atom fcc Cu, every modifier, ray tracer
    python benchmarks/analysis.py --size 40 --repeat 3 # 256k atoms
    python benchmarks/analysis.py --skip-render        # modifiers only
    python benchmarks/analysis.py --skip-modifiers     # ray tracer only
    python benchmarks/analysis.py --skip-modifiers --device cuda   # ray tracer on the GPU

Modifiers run on a thermally displaced fcc Cu crystal of 4*size^3 atoms (size=30 -> 108k atoms).
Times are the best of ``--repeat`` runs after one warm-up run, so numba compile and cache loading
are excluded. The ray tracer renders the SrTiO3 example with TiO6 polyhedra at each quality preset,
then times the trace alone (scene built once) on scenes that stress the acceleration structure:
a large isosurface (a gyroid, ~0.5 M triangles), the 108k-atom Cu crystal, and the 108k atoms of
a nanoparticle-sized slab in a mostly empty view.
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


def bench_raytrace(repeat: int, device=None):
    from ewald import examples, load, render

    pipe = load(examples.path("POSCAR.SrTiO3"))
    pipe.add(M.CreateBonds(mode="pairwise", pairs={("Ti", "O"): 2.2}))
    pipe.add(M.CoordinationPolyhedra(center_types=["Ti"], transparency=0.25))
    data = pipe.compute(0)
    print(f"ray tracer, SrTiO3 + TiO6 polyhedra, 1600x1200, best of {repeat}")
    for quality in ("draft", "standard", "publication"):
        t = best(lambda: render.render(data, None, renderer="raytrace", quality=quality, device=device), repeat)
        print(f"  {quality:28s} {t:9.2f} s", flush=True)


def gyroid(n: int = 96, period: float = 6.0):
    from ewald.data import VoxelGrid

    L = 3 * period
    cell = Cell(np.eye(3) * L)
    x = np.arange(n) / n * 2 * np.pi * 3
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    g = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    d = DataCollection([[L / 2] * 3], ["Si"], cell)
    d.grids["charge-density"] = VoxelGrid(cell.copy(), {"Charge Density": g})
    return d


def bench_trace_scenes(size: int, repeat: int, device=None):
    """Trace time only: the scene and camera are built once, so this is the ray tracer itself."""
    from ewald import render
    from ewald.raytrace import raytrace, scene_from_data

    cases = {}
    iso = Pipeline(StaticSource([gyroid()]), [M.CreateIsosurface(0.0)]).compute(0)
    cases["isosurface (gyroid)"] = (iso, {})
    cu = fcc_cu(size)
    cases[f"fcc Cu {cu.count // 1000}k atoms"] = (cu, {})
    slab = fcc_cu(size)
    keep = slab.positions[:, 2] < 0.15 * slab.cell.matrix[2, 2]
    slab = DataCollection(slab.positions[keep], np.asarray(slab.particles["Particle Type"])[keep], slab.cell)
    cases[f"Cu slab {slab.count // 1000}k atoms, side view"] = (slab, {"direction": "x"})
    px = (800, 600)
    print(f"ray tracer, trace only, {px[0]}x{px[1]}, publication preset, best of {repeat}")
    q = render.quality_settings("publication")
    for name, (data, kw) in cases.items():
        cam = render.default_camera(data, px, kw.get("direction", "iso"), True, 1.0, 1.0, {})
        scene = scene_from_data(data, show_cell=False)
        n = len(scene.sph_r) + len(scene.cyl_r) + len(scene.tri_alpha)
        t = best(lambda: raytrace(scene, cam, px, samples=q["samples"], ao_samples=q["ao_samples"],
                                  shadows=q["shadows"], device=device), repeat)
        print(f"  {name:34s} {n:>9,} prims {t:9.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=30, help="fcc cells per edge (4*size^3 atoms)")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-modifiers", action="store_true")
    ap.add_argument("--device", default=None, help="ray tracer device: cpu (default), cuda or auto")
    args = ap.parse_args()
    if not args.skip_modifiers:
        bench_modifiers(args.size, args.repeat)
    if not args.skip_render:
        bench_raytrace(1, args.device)
        bench_trace_scenes(args.size, 1, args.device)


if __name__ == "__main__":
    main()
