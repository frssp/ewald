"""GUI resize timing: what maximizing and restoring the main window costs.

    python benchmarks/resize.py                     # Linux without a display: xvfb-run -a python benchmarks/resize.py
    python benchmarks/resize.py --notebook examples/notebooks/cu_void.py
    python benchmarks/resize.py --profile          # also: where the first maximize spends its time
    python benchmarks/resize.py --gl-format core   # with another OpenGL default format for the window
    python benchmarks/resize.py --prewarm          # first grow the window to the screen size, unseen

Opens the main window with a tutorial notebook (run, so its plots and the 3D view are drawn),
then maximizes and restores it a few times. For each step it prints when the window was settled
(the last stretch of more than 30 ms in which it could not respond to input ended), the longest
such stretch, and how many 3D renders it took and how long they were. Runs on Windows, macOS
and Linux; settings go to a temporary config folder, so your own preferences and layout are not
used or changed. ``--profile`` samples every thread's Python stack during the first maximize
and prints the most frequent ones (time inside Qt or VTK shows as the Python line that called it).
``--gl-format`` sets the process OpenGL default format before the window exists: ``app`` keeps
what ``ewald gui`` does, ``vtk`` is VTK's format (3.2 core, alpha, 8-bit depth), ``core`` is
3.2 core without alpha and with a 24-bit depth buffer. ``--prewarm`` tries a startup warm-up:
once the 3D view is built it makes the window transparent, gives it the whole screen, puts it
back and prints how long that took, so you can see whether the cost of the first maximize moves
there (``--prewarm max`` uses showMaximized/showNormal instead of setting the geometry).
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


class Sampler:
    """Every ``interval`` seconds, the innermost frames of every thread's Python stack."""

    def __init__(self, interval=0.02, depth=6):
        self.interval, self.depth = interval, depth
        self.counts = collections.Counter()
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resize-sampler", daemon=True)

    def _run(self):
        me = threading.get_ident()
        names = lambda: {t.ident: t.name for t in threading.enumerate()}
        while not self._stop.wait(self.interval):
            self.samples += 1
            tn = names()
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                stack = traceback.extract_stack(frame)[-self.depth:]
                key = (tn.get(ident, str(ident)),
                       tuple(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in stack))
                self.counts[key] += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def report(self, top=12):
        print(f"\nstacks during the first maximize ({self.samples} samples, {self.interval * 1000:.0f} ms apart):")
        for (thread, stack), n in self.counts.most_common(top):
            print(f"\n{n:4d} x  [{thread}]")
            for line in stack:
                print(f"         {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notebook", default=os.path.join(HERE, os.pardir, "examples", "notebooks", "cu_void.py"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="sample the stacks during the first maximize")
    ap.add_argument("--gl-format", choices=("app", "vtk", "core"), default="app")
    ap.add_argument("--prewarm", nargs="?", const="size", choices=("size", "max"),
                    help="grow the window to the screen once, unseen, before timing")
    args = ap.parse_args()
    os.environ.setdefault("EWALD_CONFIG_DIR", tempfile.mkdtemp(prefix="ewald-resize-"))

    from ewald import settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow, _app
    from PySide6 import QtCore

    if args.gl_format != "app":
        from PySide6 import QtGui

        from ewald.gui.application import vtk_surface_format

        fmt = vtk_surface_format()
        if args.gl_format == "core":
            fmt.setAlphaBufferSize(0)
            fmt.setDepthBufferSize(24)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
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
    if vp.view_built and getattr(vp, "uses_vtk", False):  # the legacy-gl view has no VTK render window
        rw = vp.plotter.render_window
        start = []
        rw.AddObserver("StartEvent", lambda *_: start.append(time.perf_counter()))
        rw.AddObserver("EndEvent", lambda *_: renders.append(time.perf_counter() - start[-1]))
    normal = w.geometry()
    if args.prewarm:
        t0 = time.perf_counter()
        w.setWindowOpacity(0.0)
        if args.prewarm == "max":
            w.showMaximized()
        else:
            w.setGeometry(w.screen().availableGeometry())
        app.processEvents()
        w.repaint()
        app.processEvents()
        w.showNormal()
        w.setGeometry(normal)
        app.processEvents()
        w.setWindowOpacity(1.0)
        warm = time.perf_counter() - t0
        settle(1.5)
        print(f"prewarm ({args.prewarm}): {warm * 1000:.0f} ms")

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
    sampler = None
    for i in range(args.repeat):
        for name, step in (("maximize", maximize), ("restore", restore)):
            gaps.clear()
            renders.clear()
            t0 = beat[0] = time.perf_counter()
            if args.profile and i == 0 and name == "maximize":
                with Sampler() as sampler:
                    step()
                    settle(1.5)
            else:
                step()
                settle(1.5)
            mine = [(a, b) for a, b in gaps if b >= t0]
            settled = (max(b for _, b in mine) - t0) if mine else 0.0
            longest = max((b - a for a, b in mine), default=0.0)
            ms = ", ".join(f"{r * 1000:.0f}" for r in renders) or "-"
            print(f"{name:<10}{settled * 1000:>8.0f} ms{longest * 1000:>7.0f} ms{len(renders):>9}{ms:>12}")
    gl = vp.opengl_info() if vp.view_built else {}
    print("OpenGL:", gl.get("renderer", "?"), gl.get("version", ""), "| default format:", args.gl_format)
    if sampler is not None:
        sampler.report()
    heart.stop()
    w.close()


if __name__ == "__main__":
    sys.exit(main())
