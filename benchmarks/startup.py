"""GUI startup timeline: where the seconds between launch and a usable window go.

    python benchmarks/startup.py              # Linux without a display: xvfb-run -a python benchmarks/startup.py
    python benchmarks/startup.py --cold-jit   # also with an empty numba cache (fresh install, after an update)
    python benchmarks/startup.py --profile    # also: which calls the import and ``MainWindow()`` spend it in

Prints when ``import ewald.gui.app`` finishes, when ``MainWindow()`` returns, when the 3D view
and the Python console are built, every stretch longer than 100 ms in which the window could
not respond to input, and the peak memory. Runs on Windows, macOS and Linux; settings go to a
temporary config folder, so your own preferences and layout are not used or changed.
``--profile`` runs the import and ``MainWindow()``, then the time from ``show()`` until the 3D
view is built, under cProfile (which makes them slower) and
prints the calls with the most cumulative and own time (on Python 3.12+ it also sees the
threads started meanwhile, such as the plugin catalog download, which do not block the window).
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import os
import subprocess
import sys
import tempfile
import time

T0 = time.perf_counter()


def peak_rss_mb():
    try:
        import resource
    except ImportError:  # Windows
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process().memory_info().peak_wset / 2**20
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 2**20 if sys.platform == "darwin" else kb / 1024


def timeline(idle_ms: int, profile: bool = False):
    marks = []
    prof = cProfile.Profile() if profile else None
    if prof is not None:
        prof.enable()

    def mark(name):
        marks.append((time.perf_counter() - T0, name))

    from ewald.gui import app as appmod

    mark("import ewald.gui.app")
    from PySide6 import QtCore

    qapp = appmod._app()
    win = appmod.MainWindow()
    mark("MainWindow()")
    view_prof = None
    if prof is not None:
        prof.disable()
        view_prof = cProfile.Profile()
        view_prof.enable()

    def view_built():
        mark("3D view built")
        if view_prof is not None:
            view_prof.disable()

    win.show()
    win.viewport.viewBuilt.connect(view_built)
    if win.viewport.view_built:
        view_built()
    last, stalls = [time.perf_counter()], []

    def tick():
        now = time.perf_counter()
        if now - last[0] > 0.1:
            stalls.append((last[0] - T0, now - last[0]))
        last[0] = now
        if getattr(win.console, "built", True) and not any(n == "console built" for _, n in marks):
            mark("console built")

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    QtCore.QTimer.singleShot(idle_ms, qapp.quit)
    qapp.exec()
    for t, name in marks:
        print(f"{t:7.2f} s  {name}")
    print("  window busy (> 100 ms):", ", ".join(f"{a:.2f} s for {b:.2f} s" for a, b in stalls) or "none")
    rss = peak_rss_mb()
    print(f"  peak memory: {rss:.0f} MB" if rss is not None else "  peak memory: (pip install psutil)")
    for p, what in ((prof, "import + MainWindow()"), (view_prof, "show() until the 3D view is built")):
        if p is None:
            continue
        stats = pstats.Stats(p, stream=sys.stdout)
        print(f"\n{what} under cProfile, EWALD's own functions by cumulative time:")
        stats.sort_stats("cumulative").print_stats(r"[\\/]ewald[\\/]", 30)
        print("every function by own time:")
        stats.sort_stats("tottime").print_stats(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cold-jit", action="store_true", help="also run once with an empty numba cache")
    ap.add_argument("--idle", type=int, default=6000, help="ms to keep the window open (default 6000)")
    ap.add_argument("--profile", action="store_true", help="profile the import and MainWindow()")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return timeline(args.idle, args.profile)
    runs = [("warm numba cache", {})]
    with tempfile.TemporaryDirectory() as tmp:
        if args.cold_jit:
            runs.append(("cold numba cache", {"NUMBA_CACHE_DIR": os.path.join(tmp, "numba")}))
        for title, extra in runs:
            env = dict(os.environ, EWALD_CONFIG_DIR=os.path.join(tmp, "config"), **extra)
            print(f"== {title}")
            sys.stdout.flush()
            cmd = [sys.executable, os.path.abspath(__file__), "--child", "--idle", str(args.idle)]
            subprocess.run(cmd + (["--profile"] if args.profile else []), env=env, check=False)


if __name__ == "__main__":
    main()
