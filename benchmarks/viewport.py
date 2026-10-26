"""Interactive 3D view timings for large structures: rotation fps, frame switches, memory.

    python benchmarks/viewport.py                                  # 108k and 1M atoms, bonds on and off
    python benchmarks/viewport.py --atoms 108000 1000000 4000000 --bonds on off -o viewport.json
    xvfb-run -a python benchmarks/viewport.py --atoms 108000       # Linux without a display (llvmpipe)

Opens the GUI's own 3D view (``ewald.gui.viewport.Viewport``) in a window and draws a jiggled FCC
Cu crystal (Ball & stick, bonds up to 2.9 A, 12 per atom), so every machine draws the same scene.
For each size and bond setting it reports:

- **build**: ``set_data`` of the first frame (bond search, scene, VTK actors) and its first render
  (the upload to the GPU);
- **rotate**: frame times of full-detail renders while the camera turns (what the view shows when
  the mouse stops), and of the coarse level of detail the view switches to while dragging;
- **frame switch**: ``set_data`` of other frames of the same atoms plus their render, as when
  playing or scrubbing a trajectory, after the bond search the main window's compute worker
  does off the GUI thread (timed apart; file reading is not included, see docs/developer/readers.md);
- the OpenGL driver, the process memory, and how many cylinders (half-bonds) the GPU holds.

Each render waits for the GPU to finish, so the numbers are real frame times. ``-o`` writes them
as JSON. Container numbers (Mesa llvmpipe) only show the CPU side; run it on a real GPU for fps.
Settings go to a temporary config folder, so your preferences are not used or changed. On a laptop
with two GPUs, ``ewald gpu`` says which one Python uses (``ewald gpu --prefer``, docs/user/gpu.md).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

CUTOFF = 2.9   # Cu-Cu bonds: the 12 nearest neighbors (2.56 A), not the next shell (3.62 A)
A_CU = 3.615


def fcc_cu(n_target: int, rng, jiggle: float = 0.05):
    """A periodic FCC Cu block of about ``n_target`` atoms with thermal-like noise."""
    from ewald.data import Cell, DataCollection

    n = max(1, round((n_target / 4) ** (1 / 3)))
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = ((g[:, None, :] + basis[None]) * A_CU).reshape(-1, 3)
    base = DataCollection(pos, ["Cu"] * len(pos), Cell(np.eye(3) * n * A_CU, pbc=(True,) * 3))

    def frame():
        d = base.copy()
        d.particles["Position"] = pos + jiggle * rng.standard_normal(pos.shape)
        return d

    return frame


def memory_mb():
    """(current, peak) process memory in MB, or (None, None)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return info.rss / 2**20, getattr(info, "peak_wset", info.rss) / 2**20
    except ImportError:
        pass
    try:
        import resource
    except ImportError:  # Windows without psutil
        return None, None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = kb / 2**20 if sys.platform == "darwin" else kb / 1024
    return None, peak


def stats(times):
    t = np.asarray(times, float)
    return {"frames": len(t), "mean_ms": 1e3 * t.mean(), "p95_ms": 1e3 * np.percentile(t, 95),
            "fps": 1.0 / t.mean()}


def cylinders(plotter) -> int:
    """Cylinders the view draws: points of the stick mapper inputs (one per half-bond)."""
    n = 0
    for name, actor in plotter.actors.items():
        if str(name).startswith("bonds") and not str(name).endswith(":caps"):
            data = actor.GetMapper().GetInput()
            n += data.GetNumberOfPoints() if data is not None else 0
    return n


def run_case(qapp, view, make_frame, bonds: bool, rotate: int, switches: int) -> dict:
    from ewald.bonds import find_bonds

    style = view.style
    style.apply_preset("Ball & stick")
    style.bonds_enabled = bonds
    style.bond_pairs = [["Cu", "Cu", 0.0, CUTOFF]] if bonds else []
    p = view.plotter
    rw = p.render_window

    def draw():
        t = time.perf_counter()
        rw.Render()
        rw.WaitForCompletion()
        qapp.processEvents()
        return time.perf_counter() - t

    out = {"bonds": bonds}
    first = make_frame()
    out["atoms"] = first.count
    view.reset_view()
    t = time.perf_counter()
    view.set_data(first)
    out["build_s"] = time.perf_counter() - t
    out["first_render_s"] = draw()
    out["cylinders"] = cylinders(p)

    view.detail_mode = "full"
    times = []
    for _ in range(rotate):
        p.camera.Azimuth(360.0 / rotate)
        times.append(draw())
    out["rotate_full"] = stats(times)
    view.detail_mode = "coarse"
    view._go_coarse_for_motion()
    times = []
    for _ in range(rotate):
        p.camera.Azimuth(360.0 / rotate)
        times.append(draw())
    out["rotate_coarse"] = stats(times)
    view.refine()
    view.detail_mode = "full"

    if bonds:
        extra = make_frame()
        t = time.perf_counter()
        find_bonds(extra, pairs=style.scene_kwargs()["bond_pairs"])
        out["bond_search_s"] = time.perf_counter() - t
    sets, renders, worker = [], [], []
    pairs = style.scene_kwargs().get("bond_pairs")
    for _ in range(switches):
        data = make_frame()
        if pairs:  # what the main window's compute worker does after reading a frame (off the GUI thread)
            from ewald.render import scene_bonds

            t = time.perf_counter()
            scene_bonds(data, None, pairs)
            worker.append(time.perf_counter() - t)
        t = time.perf_counter()
        view.set_data(data, interactive=True)
        sets.append(time.perf_counter() - t)
        renders.append(draw())
    out["switch_set_data_s"] = float(np.mean(sets))
    out["switch_render_s"] = float(np.mean(renders))
    out["switch_total_s"] = out["switch_set_data_s"] + out["switch_render_s"]
    out["switch_worker_s"] = float(np.mean(worker)) if worker else 0.0
    out["memory_mb"], out["peak_memory_mb"] = memory_mb()
    return out


def report(r: dict):
    b = "bonds" if r["bonds"] else "atoms only"
    print(f"{r['atoms']:>10,} atoms, {b}: {r['cylinders']:,} cylinders")
    print(f"  build {r['build_s']:7.2f} s   first render {r['first_render_s']:7.2f} s"
          + (f"   bond search {r['bond_search_s']:.2f} s" if "bond_search_s" in r else ""))
    for key, label in (("rotate_full", "rotate (full detail)"), ("rotate_coarse", "rotate (while dragging)")):
        s = r[key]
        print(f"  {label:<24s} {s['fps']:7.1f} fps   mean {s['mean_ms']:7.1f} ms   p95 {s['p95_ms']:7.1f} ms")
    print(f"  frame switch {r['switch_total_s']:7.3f} s on the GUI thread (set_data {r['switch_set_data_s']:.3f} s"
          f" + render {r['switch_render_s']:.3f} s)"
          + (f", {r['switch_worker_s']:.3f} s before it on the worker" if r.get("switch_worker_s") else ""))
    if r["peak_memory_mb"] is not None:
        print(f"  peak memory {r['peak_memory_mb']:.0f} MB")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atoms", type=int, nargs="+", default=[108_000, 1_000_000])
    ap.add_argument("--bonds", nargs="+", choices=["on", "off"], default=["off", "on"])
    ap.add_argument("--size", default="1280x800", help="window size in pixels (default 1280x800)")
    ap.add_argument("--rotate", type=int, default=36, help="rendered frames per rotation (default 36)")
    ap.add_argument("--switches", type=int, default=3, help="frame switches per case (default 3)")
    ap.add_argument("--max-cylinders", type=float, default=8e6,
                    help="skip bond cases above this many half-bonds (default 8e6, about 4.6 GB of GPU memory "
                         "with VTK's stick mapper)")
    ap.add_argument("-o", "--output", help="also write the results to this JSON file")
    args = ap.parse_args()

    os.environ.setdefault("EWALD_CONFIG_DIR", tempfile.mkdtemp(prefix="ewald-bench-"))
    os.environ.setdefault("QT_API", "pyside6")
    from ewald.gui.viewport import Viewport
    from ewald.gui import app as appmod
    from ewald.view_style import ViewStyle

    qapp = appmod._app()
    w, h = (int(v) for v in args.size.lower().split("x"))
    view = Viewport(ViewStyle())
    view.resize(w, h)
    view.show()
    view.ensure_view()
    qapp.processEvents()
    view.plotter.render_window.Render()
    gl = view.opengl_info()
    try:
        from ewald import gpu_select

        gpus = gpu_select.list_gpus()
    except Exception:  # noqa: BLE001 - only informative
        gpus = []
    import ewald

    meta = {"ewald": getattr(ewald, "__version__", ""), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "opengl": gl, "gpus": gpus, "window": [w, h]}
    print(f"OpenGL: {gl.get('renderer')} ({gl.get('version')}){'  [software]' if gl.get('software') else ''}")
    if gpus:
        print("GPUs:", ", ".join(gpus))
    rng = np.random.default_rng(0)
    results = []
    for n in args.atoms:
        make_frame = fcc_cu(n, rng)
        for b in args.bonds:
            bonds = b == "on"
            if bonds and 2 * 6 * n > args.max_cylinders:
                print(f"{n:>10,} atoms, bonds: skipped (about {12 * n:,} half-bonds > --max-cylinders)")
                results.append({"atoms": n, "bonds": True, "skipped": "max-cylinders"})
                continue
            r = run_case(qapp, view, make_frame, bonds, args.rotate, args.switches)
            report(r)
            results.append(r)
            view.set_data(None)
            qapp.processEvents()
    view.close()
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
