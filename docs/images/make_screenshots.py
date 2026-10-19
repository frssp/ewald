"""Regenerate the GUI screenshots of the manual from the tutorial notebooks.

    xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_screenshots.py [name ...]

Each shot runs in its own process (a fresh window and config folder), opens a notebook from
examples/notebooks/, runs it, and saves a 256-color PNG here. Names: notebook, structure,
trajectory, minimal, plugins (default: all).
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.join(HERE, "..", "..", "examples", "notebooks")

# name: (output file, notebook, cell to scroll to, extra setup)
SHOTS = {
    "notebook": ("notebook.png", "cu_void.py", 4, None),
    "structure": ("tutorial-structure.png", "srtio3_octahedra.py", 6, None),
    "trajectory": ("tutorial-trajectory.png", "cu_md_trajectory.py", 7, "frame"),
    "minimal": ("minimal-mode.png", "cu_void.py", None, "minimal"),
    "plugins": ("plugins.png", "cu_void.py", None, "plugins"),
}


def shoot(name):
    from PIL import Image
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ewald import settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow

    notebook_panel.CONFIRM_UNSAVED = False
    out, notebook, scroll_to, extra = SHOTS[name]

    def settle(n=30):
        for _ in range(n):
            app.processEvents()
            QtCore.QThread.msleep(40)

    w = MainWindow(settings=settings.Settings())
    w.resize(1600, 980)
    w.show()
    settle()
    w.open_notebook(os.path.join(NOTEBOOKS, notebook), run=True)
    w.bottom_dock.hide()
    w.resizeDocks([w.notebook_dock], [540], QtCore.Qt.Horizontal)
    settle(60)
    if scroll_to is not None:
        view = w.panel.view
        view.scroll.ensureWidgetVisible(view.widget_for(view.notebook.cells[scroll_to]), 0, 0)
    if extra == "frame":
        w.set_frame(w.pipeline.num_frames - 1)
    elif extra == "minimal":
        w.set_minimal(True)
    elif extra == "plugins":
        w.plugins_dock.show()
        w.plugins_dock.raise_()
        w.resizeDocks([w.plugins_dock, w.notebook_dock], [330, 460], QtCore.Qt.Horizontal)
    settle()
    w.statusBar().clearMessage()
    settle(5)
    raw = os.path.join(tempfile.mkdtemp(), out)
    w.grab().save(raw)
    image = Image.open(raw).convert("RGB")
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        os.path.join(HERE, out), optimize=True)
    print("wrote", os.path.join(HERE, out))
    w.close()


if __name__ == "__main__":
    if os.environ.get("_EWALD_SHOT"):
        shoot(os.environ["_EWALD_SHOT"])
        sys.exit(0)
    for name in sys.argv[1:] or list(SHOTS):
        work = tempfile.mkdtemp()        # notebooks write their figures into the working folder
        env = dict(os.environ, _EWALD_SHOT=name, EWALD_CONFIG_DIR=tempfile.mkdtemp(), MPLBACKEND="Agg",
                   EWALD_PLUGIN_INDEX="")   # only the bundled example plugins, no network
        subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, cwd=work, check=True)
