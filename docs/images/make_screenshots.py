"""Regenerate the GUI screenshots of the manual from the tutorial notebooks.

    xvfb-run -a -s "-screen 0 1920x1200x24" python docs/images/make_screenshots.py [name ...]

Each shot runs in its own process (a fresh window and config folder), opens a notebook from
examples/notebooks/, runs it, and saves a 256-color PNG here. Names: notebook, structure,
trajectory, minimal, plugins, plugin-panel, plugin-settings, ui-theme (default: all); the
plugin- shots install examples/plugins/sim_agent_panel.py first, ui-theme installs
examples/plugins/ui_themes.py and uses its Studio theme.
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
    "structure": ("tutorial-structure.png", "srtio3_octahedra.py", 7, None),
    "trajectory": ("tutorial-trajectory.png", "cu_md_trajectory.py", 7, "frame"),
    "minimal": ("minimal-mode.png", "cu_void.py", None, "minimal"),
    "plugins": ("plugins.png", "cu_void.py", None, "plugins"),
    "plugin-panel": ("plugin_panel.png", "cu_void.py", None, "plugin-panel"),
    "plugin-settings": ("plugin_settings_page.png", "cu_void.py", None, "plugin-settings"),
    "ui-theme": ("ui_theme_studio.png", "cu_void.py", 4, "ui-theme"),
}
EXAMPLE_PLUGIN = os.path.join(HERE, "..", "..", "examples", "plugins", "sim_agent_panel.py")
THEME_PLUGIN = os.path.join(HERE, "..", "..", "examples", "plugins", "ui_themes.py")


def shoot(name):
    from PIL import Image
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ewald import settings
    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow

    notebook_panel.CONFIRM_UNSAVED = False
    out, notebook, scroll_to, extra = SHOTS[name]
    if extra and (extra.startswith("plugin-") or extra == "ui-theme"):
        import shutil

        from ewald import plugins

        plugins.plugin_dir().mkdir(parents=True, exist_ok=True)
        shutil.copy(THEME_PLUGIN if extra == "ui-theme" else EXAMPLE_PLUGIN, plugins.plugin_dir())

    def settle(n=30):
        for _ in range(n):
            app.processEvents()
            QtCore.QThread.msleep(40)

    s = settings.Settings()
    if extra == "ui-theme":
        s["general.ui_theme"] = "Studio"
    w = MainWindow(settings=s)
    w.resize(1600, 980)
    w.show()
    settle()
    w.open_notebook(os.path.join(NOTEBOOKS, notebook), run=True)
    w.workbench.show(w.display)  # the View tab below the Structure view
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
    elif extra == "plugin-panel":
        dock = w.show_plugin_panel("sim_agent")
        w.resizeDocks([dock, w.notebook_dock], [320, 480], QtCore.Qt.Horizontal)
        for task in ("relax this cell with PBE", "phonons at Gamma"):
            dock.widget().findChild(QtWidgets.QLineEdit).setText(task)
            dock.widget().findChild(QtWidgets.QLineEdit).returnPressed.emit()
        w.show_plugin_panel("agent_jobs")      # the plugin's tab under the 3D view
    target = w
    if extra == "plugin-settings":
        from ewald.gui.preferences import PreferencesDialog

        target = PreferencesDialog(w.settings, w, page="Sim agent")
        target.show()
    settle()
    w.statusBar().clearMessage()
    settle(5)
    raw = os.path.join(tempfile.mkdtemp(), out)
    target.grab().save(raw)
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
