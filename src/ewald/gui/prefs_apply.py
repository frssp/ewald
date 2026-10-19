"""Applying stored preferences to the main window: element defaults, window settings, layout and the startup script."""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..settings import DEVICES, Settings
from .pref_widgets import _hex
from .style import PRESET_THEMES


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


def apply_settings(win, s: Settings, startup: bool = False):
    """Push settings into a MainWindow: viewport, style, render page, device, font."""
    from . import theme

    vp, style, page = win.viewport, win.style, win.render_page
    win.settings = s
    app = QtWidgets.QApplication.instance()
    if app is not None:
        # one style sheet, set only when it changes: each set re-polishes every widget of every window
        theme.apply(app, s["general.ui_theme"], font_size=s["general.font_size"])

    win.set_button_style(s["general.toolbar_style"])
    if style.preset not in PRESET_THEMES:  # those presets keep their theme's background
        style.background = s["viewport.background"]
    style.bond_radius = s["bonds.radius"]
    if startup or style.preset != s["viewport.style_preset"]:
        style.apply_preset(s["viewport.style_preset"], win.data.types if win.data is not None else ())
    vp.rotation_step = s["viewport.rotation_step"]
    vp.hover_info = s["viewport.hover_info"]
    vp.hover_delay = s["viewport.hover_delay"]
    if not vp.hover_info:
        vp.hover_leave()
    vp.sphere_drawing = s["viewport.sphere_drawing"]
    vp.sphere_resolution = s["viewport.sphere_resolution"]
    vp.set_anti_aliasing(s["viewport.anti_aliasing"])
    vp.mesh_limit = s["viewport.mesh_limit"]
    vp.arrow_resolution = s["viewport.arrow_resolution"]
    vp.detail_mode = s["viewport.interactive_detail"]
    vp.target_fps = s["viewport.target_fps"]
    vp.interactive_atoms = s["viewport.interactive_atoms"]
    vp.refine_delay = s["viewport.refine_delay"]
    if win.pipeline is not None:
        win.pipeline.cache_memory = s["performance.cache_memory"] * 2**20
    win.live_reload.configure(s["general.live_reload"], s["general.live_reload_interval"])
    win.live_reload_action.setChecked(s["general.live_reload"])
    vp.show_overlay = s["viewport.show_overlay"]
    vp.initial_view = s["viewport.initial_view"]
    vp.set_axes_visible(s["viewport.show_axes"])
    vp.ortho_button.setChecked(s["viewport.projection"] == "ortho")

    page.set_renderer(s["render.renderer"])
    page.samples.setValue(s["render.samples"])
    page.ao_samples.setValue(s["render.ao_samples"])
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
    if startup and win.device is None:
        win.device = s["compute.device"]
        probe = dict((k, p) for k, _label, p in DEVICES).get(win.device)
        ok, detail = probe() if probe else (False, "unknown device")
        if not ok:
            win.message(f"Default device {win.device!r} unavailable ({detail}); using CPU", error=True)
            win.device = "cpu"
        label = {"cpu": "CPU", "cuda": "CUDA", "auto": "auto"}.get(win.device, "CPU")
        win.device_combo.blockSignals(True)
        win.device_combo.setCurrentText(label)
        win.device_combo.blockSignals(False)
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
                 "state": bytes(win.saveState().toBase64()).decode()}
        try:
            layout_path(s).parent.mkdir(parents=True, exist_ok=True)
            layout_path(s).write_text(json.dumps(state))
        except OSError:
            pass


def restore_layout(win, s: Settings):
    p = layout_path(s)
    if not s["general.restore_layout"] or not p.exists():
        return
    try:
        state = json.loads(p.read_text())
        win.restoreGeometry(QtCore.QByteArray(base64.b64decode(state["geometry"])))
        win.restoreState(QtCore.QByteArray(base64.b64decode(state["state"])))
    except Exception:
        pass


def layout_path(s: Settings):
    return s.path.with_name("layout.json")


def run_startup_script(win, s: Settings):
    path = s["python.startup_script"]
    if not path:
        return
    shell = getattr(win.console, "shell", None)
    try:
        src = open(path).read()
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
