"""GUI resize timing: what maximizing and restoring the main window costs.

    python benchmarks/resize.py                     # Linux without a display: xvfb-run -a python benchmarks/resize.py
    python benchmarks/resize.py --notebook examples/notebooks/cu_void.py

Opens the main window with a tutorial notebook (run, so its plots and the 3D view are drawn),
then maximizes and restores it a few times. For each step it prints when the window was settled
(the last stretch of more than 30 ms in which it could not respond to input ended), the longest
such stretch, and how many 3D renders it took and how long they were. Runs on Windows, macOS
and Linux; settings go to a temporary config folder, so your own preferences and layout are not
used or changed.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notebook", default=os.path.join(HERE, os.pardir, "examples", "notebooks", "cu_void.py"))
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    os.environ.setdefault("EWALD_CONFIG_DIR", tempfile.mkdtemp(prefix="ewald-resize-"))

    from ewald import settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow, _app
    from PySide6 import QtCore

    app = _app()
    notebook_panel.CONFIRM_UNSAVED = False
    notebook_panel.CELL_BUDGET = None
    w = MainWindow(settings=settings.Settings())
    w.resize(1100, 700)
    w.show()
    view = w.panel.open_notebook(os.path.abspath(args.notebook), trusted=True)
    view.run_all()

    renders = []
    beat = [time.perf_counter()]
    gaps = []

    def tick():
        now = time.perf_counter()
        if now - beat[0] > 0.030:
            gaps.append((beat[0], now))
        beat[0] = now

    heart = QtCore.QTimer(interval=5)
    heart.timeout.connect(tick)
    heart.start()

    def settle(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents(QtCore.QEventLoop.AllEvents, 5)
            tick()

    settle(3.0)  # the 3D view and the plots are built after the first paint
    vp = w.viewport
    if vp.view_built:
        rw = vp.plotter.render_window
        start = []
        rw.AddObserver("StartEvent", lambda *_: start.append(time.perf_counter()))
        rw.AddObserver("EndEvent", lambda *_: renders.append(time.perf_counter() - start[-1]))
    normal = w.geometry()

    def maximize():
        w.showMaximized()
        app.processEvents()
        if w.size() == normal.size():  # no window manager (xvfb): take the screen by hand
            w.setGeometry(w.screen().availableGeometry())

    def restore():
        w.showNormal()
        app.processEvents()
        if w.size() != normal.size():
            w.setGeometry(normal)

    print(f"{'step':<10}{'settled':>10}{'longest':>10}{'renders':>9}{'render ms':>12}")
    for _ in range(args.repeat):
        for name, step in (("maximize", maximize), ("restore", restore)):
            gaps.clear()
            renders.clear()
            t0 = beat[0] = time.perf_counter()
            step()
            settle(1.5)
            mine = [(a, b) for a, b in gaps if b >= t0]
            settled = (max(b for _, b in mine) - t0) if mine else 0.0
            longest = max((b - a for a, b in mine), default=0.0)
            ms = ", ".join(f"{r * 1000:.0f}" for r in renders) or "-"
            print(f"{name:<10}{settled * 1000:>8.0f} ms{longest * 1000:>7.0f} ms{len(renders):>9}{ms:>12}")
    gl = vp.opengl_info() if vp.view_built else {}
    print("OpenGL:", gl.get("renderer", "?"), gl.get("version", ""))
    heart.stop()
    w.close()


if __name__ == "__main__":
    sys.exit(main())
