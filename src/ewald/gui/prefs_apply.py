"""Applying stored preferences to the main window: element defaults, window settings, layout and the startup script."""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..settings import DEVICES, Settings
from .pref_widgets import _hex
from .style import PRESET_THEMES, set_default_background


# ---------------------------------------------------------------- applying to the main window
def apply_element_defaults(style, data, s: Settings):
    """Fill per-type colors/radii from the settings without touching what the user set in the session.

    Returns the :class:`ewald.radii.RadiusTable` used for the radii (None when nothing was filled).
    """
    from ..radii import IONIC, element_of

    # drop what we filled last time (unless the user has since edited it) so scheme changes take effect
    for t, (c, r) in getattr(style, "_pref_filled", {}).items():
        if c is not None and style.type_colors.get(t) == c:
            style.type_colors.pop(t, None)
        if r is not None and style.type_radii.get(t) == r:
            style.type_radii.pop(t, None)
    filled = {}
    style._pref_filled = filled
    overrides = s["elements.overrides"]
    fill_colors = not s.is_default("elements.color_scheme") or any("color" in o for o in overrides.values())
    # ionic radii depend on the structure: resolve them once per composition rather than on every draw
    fill_radii = (not s.is_default("elements.radius_scale") or any("radius" in o for o in overrides.values())
                  or style.radius_preset in IONIC)
    if not (fill_colors or fill_radii):
        return None  # built-in defaults already match; keep generated scripts free of redundant type_colors
    table = None
    if fill_radii:
        uniq, counts = np.unique(np.asarray(data.types).astype(str), return_counts=True)
        key = (style.radius_preset, tuple(sorted(style.oxidation_states.items())), tuple(uniq), tuple(counts),
               s["elements.radius_scale"], repr(sorted(overrides.items())))
        cache = getattr(style, "_radius_cache", None)
        if cache is None or cache[0] != key:
            cache = style._radius_cache = (key, *s.type_radii(data, style.radius_preset, style.oxidation_states))
        radii, table = cache[1], cache[2]
    for t in dict.fromkeys(str(t) for t in data.types):
        sym = element_of(t)
        if sym is None:
            continue
        c = r = None
        if fill_colors and t not in style.type_colors:
            c = style.type_colors[t] = _hex(s.element_color(sym))
        if fill_radii and t not in style.type_radii and t in radii:
            r = style.type_radii[t] = radii[t]
        if c is not None or r is not None:
            filled[t] = (c, r)
    return table


def apply_viewport_settings(vp, s: Settings):
    """Preferences › 3D view on one viewport (the main one and each view tab's)."""
    vp.rotation_step = s["viewport.rotation_step"]
    vp.hover_info = s["viewport.hover_info"]
    vp.hover_delay = s["viewport.hover_delay"]
    if not vp.hover_info:
        vp.hover_leave()
    vp.sphere_drawing = s["viewport.sphere_drawing"]
    vp.sphere_resolution = s["viewport.sphere_resolution"]
    vp.set_anti_aliasing(s["viewport.anti_aliasing"])
    vp.mesh_limit = s["viewport.mesh_limit"]
    vp.bond_limit = s["bonds.auto_limit"]
    vp.arrow_resolution = s["viewport.arrow_resolution"]
    vp.detail_mode = s["viewport.interactive_detail"]
    vp.target_fps = s["viewport.target_fps"]
    vp.interactive_atoms = s["viewport.interactive_atoms"]
    vp.refine_delay = s["viewport.refine_delay"]
    vp.show_overlay = s["viewport.show_overlay"]
    vp.initial_view = s["viewport.initial_view"]
    vp.set_axes_visible(s["viewport.show_axes"])
    vp.ortho_button.setChecked(s["viewport.projection"] == "ortho")


def apply_settings(win, s: Settings, startup: bool = False):
    """Push settings into a MainWindow: viewport, style, render page, device, font."""
    from . import theme

    vp, style, page = win.viewport, win.style, win.render_page
    win.settings = s
    win.timeline.show_info = s["viewport.show_overlay"]
    if win.data is not None:
        win.timeline.set_info(win.data.attributes)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        # one style sheet, set only when it changes: each set re-polishes every widget of every window
        theme.apply(app, s["general.ui_theme"],
                    font_size=None if s.is_default("general.font_size") else s["general.font_size"])
    from .console import restyle

    restyle(win.console)  # the console draws its own colors (background, syntax, selection)
    tokens = dict(theme.TOKENS)
    if getattr(win, "_ui_tokens", tokens) != tokens:  # open figures keep the colors they were drawn in
        from .plots import PlotsPanel

        for panel in win.findChildren(PlotsPanel):  # Plots tabs and notebook view cells
            panel.restyle()
        for highlighter in win.findChildren(QtGui.QSyntaxHighlighter):  # notebook code cells
            highlighter.rehighlight()
    win._ui_tokens = tokens
    apply_plot_style(win, s)
    watch_plot_style(win)

    win.set_button_style(s["general.toolbar_style"])
    from .icons import refresh

    refresh(win)  # header, timeline and cell icons follow the UI theme
    if hasattr(win, "activity_buttons"):
        win.refresh_activity_icons()
        win.arrange_activity_bar()   # Preferences › General › Hidden activity bar buttons
    # a background from the UI theme or Preferences is not written into the notebook's view
    background = theme.TOKENS["viewport"] if s["viewport.follow_theme"] else s["viewport.background"]
    set_default_background(background)
    if style.preset not in PRESET_THEMES:  # those presets keep their theme's background
        style.background = background
    style.bond_radius = s["bonds.radius"]
    if startup or style.preset != s["viewport.style_preset"]:
        style.apply_preset(s["viewport.style_preset"], win.data.types if win.data is not None else ())
    for v in [vp, *(t.viewport for t in getattr(win, "view_tabs", {}).values())]:
        apply_viewport_settings(v, s)
    if win.pipeline is not None:
        win.pipeline.cache_memory = s["performance.cache_memory"] * 2**20
    win.live_reload.configure(s["general.live_reload"], s["general.live_reload_interval"])
    win.live_reload_action.setChecked(s["general.live_reload"])
    if hasattr(win, "_render_frame_visibility"):
        win._render_frame_visibility()

    page.set_renderer(s["render.renderer"])
    page.samples.setValue(s["render.samples"])
    page.ao_samples.setValue(s["render.ao_samples"])
    page.noise.setValue(s["render.noise_threshold"])
    page.shadows.setChecked(s["render.shadows"])
    page.quality.setCurrentText(s["render.quality"])  # after the Custom values: a preset overwrites them
    page._quality_changed(s["render.quality"])
    page.movie_quality.setCurrentText(s["render.movie_quality"])
    page.dpi.setValue(s["render.dpi"])
    page.preset.setCurrentText(s["render.size_preset"])
    page.background.setCurrentText(s["render.background"])
    page.type_legend.setChecked(s["render.type_legend"])
    page.scale_bar.setChecked(s["render.scale_bar"])
    page.movie_format.blockSignals(True)
    page.movie_format.setCurrentText(s["render.movie_format"])
    page.movie_format.blockSignals(False)
    if s["render.output_dir"] and startup:
        page.output.setText(os.path.join(s["render.output_dir"], os.path.basename(page.output.text() or "render.png")))

    if s["compute.threads"]:
        try:
            import numba

            numba.set_num_threads(min(s["compute.threads"], numba.config.NUMBA_NUM_THREADS))
        except Exception as exc:
            win.message(f"Could not set CPU threads: {exc}", error=True)
    if not (startup and win.device is not None):  # a device given on the command line wins at startup
        dev = s["compute.device"]
        probe = dict((k, p) for k, _label, p in DEVICES).get(dev)
        ok, detail = probe() if probe else (False, "unknown device")
        if not ok:
            win.message(f"Device {dev!r} unavailable ({detail}); using CPU", error=True)
            dev = "cpu"
        win.set_device(dev)
    from .code_editor import OutputText

    lines = (s["python.output_scroll_after"], s["python.output_box_lines"])
    if (OutputText.scroll_after, OutputText.box_lines) != lines:   # long cell outputs scroll in a box
        OutputText.scroll_after, OutputText.box_lines = lines
        for text in win.findChildren(OutputText):
            text._fit()
    from .notebook_panel import NotebookView

    NotebookView.follow_running = s["python.follow_running"]
    for p in filter(None, (x.strip() for x in s["python.extra_paths"].split(";"))):
        if p not in sys.path:
            sys.path.append(p)

    if getattr(style, "bond_factor", None) != s["bonds.length_factor"]:
        style.bond_factor = s["bonds.length_factor"]
        style.bond_pairs.clear()  # re-seeded from the new factor on the next draw
    if startup or getattr(style, "_pref_preset", None) != s["elements.radius_source"]:
        style.radius_preset = style._pref_preset = s["elements.radius_source"]
    if win.data is not None:
        if style.bonds_enabled:
            style.ensure_bond_pairs(win.data.types)
        apply_element_defaults(style, win.data, s)
    win._style_changed(write=False)


def save_layout(win, s: Settings):
    if s["general.restore_layout"]:
        state = {"geometry": bytes(win.saveGeometry().toBase64()).decode(),
                 "state": bytes(win.saveState().toBase64()).decode(),
                 "workbench": win.workbench.state(),
                 "plugin_panels": win.open_plugin_panels(),
                 "view_folded": sorted(win.display.folded)}
        try:
            layout_path(s).parent.mkdir(parents=True, exist_ok=True)
            layout_path(s).write_text(json.dumps(state), encoding="utf-8", newline="\n")
        except OSError:
            pass


def restore_layout(win, s: Settings):
    p = layout_path(s)
    if not s["general.restore_layout"] or not p.exists():
        return
    state = None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
        win.restoreGeometry(QtCore.QByteArray(base64.b64decode(state["geometry"])))
        win.restoreState(QtCore.QByteArray(base64.b64decode(state["state"])))
    except Exception:
        pass
    if isinstance(state, dict) and isinstance(state.get("workbench"), dict):
        win.workbench.restore_state(state["workbench"])  # tab groups; older files keep the default
    if isinstance(state, dict) and isinstance(state.get("view_folded"), list):
        for title in state["view_folded"]:       # View tab cards folded when the window closed
            if isinstance(title, str):
                win.display.set_folded(title, True)
    if isinstance(state, dict) and isinstance(state.get("plugin_panels"), list):
        win._saved_plugin_docks = set(state["plugin_panels"])
        for name in state["plugin_panels"]:      # plugin side bars that were open, where they were
            if name in getattr(win, "plugin_panels", {}) and win.plugin_panels[name]["plugin"].area == "side":
                win.show_plugin_panel(name)


def layout_path(s: Settings):
    return s.path.with_name("layout.json")


def run_startup_script(win, s: Settings):
    path = s["python.startup_script"]
    if not path:
        return
    shell = getattr(win.console, "shell", None)
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as exc:
        win.message(f"Startup script {path}: {exc}", error=True)
        return
    try:
        if hasattr(shell, "run_cell"):
            shell.run_cell(src, store_history=False)
        elif shell is not None:
            exec(compile(src, path, "exec"), shell.locals)
        win.message(f"Ran startup script {path}")
    except Exception as exc:
        win.message(f"Startup script {path}: {exc}", error=True)


def apply_plot_style(win, s: Settings | None = None):
    """The matplotlib style of every plot in the window: ``python.plot_style`` for the plots
    cells and the console draw, the user's style file (:func:`ewald.plotstyle.read_style`) on
    top of those and of the Plots panel and view cells. Returns the file's problems."""
    from .. import plotstyle
    from ..settings import plot_style
    from . import plots, theme
    from .console import set_plot_style

    s = s if s is not None else win.settings
    user, problems = plotstyle.read_style()
    if user != plots.USER_RC:
        plots.USER_RC.clear()
        plots.USER_RC.update(user)
        for panel in win.findChildren(plots.PlotsPanel):
            if panel.canvas_built:
                panel.restyle()
    user_plots = (*plot_style(s["python.plot_style"], dict(theme.TOKENS)), user or None)
    set_plot_style(win.console, *user_plots)
    if getattr(win, "panel", None) is not None and hasattr(win.panel, "set_plot_style"):
        win.panel.set_plot_style(*user_plots)
    win.plot_style_problems = problems
    return problems


def watch_plot_style(win):
    """Re-read the style file when it changes (an editor opened by Preferences › Plots › Open as
    text). The folder is watched too: editors that save by replacing the file drop the file watch."""
    from .. import plotstyle

    path = plotstyle.user_style_path()
    watcher = getattr(win, "_plot_style_watcher", None)
    if watcher is None:
        watcher = win._plot_style_watcher = QtCore.QFileSystemWatcher(win)
        timer = win._plot_style_timer = QtCore.QTimer(win, singleShot=True, interval=200)   # one reload per save
        timer.timeout.connect(lambda: (apply_plot_style(win), watch_plot_style(win)))
        watcher.fileChanged.connect(lambda _: timer.start())
        watcher.directoryChanged.connect(lambda _: timer.start())
    wanted = [str(p) for p in (path.parent.parent, path.parent, path) if p.exists()]   # config dir: plots/ made later
    missing = [p for p in wanted if p not in watcher.files() + watcher.directories()]
    if missing:
        watcher.addPaths(missing)
