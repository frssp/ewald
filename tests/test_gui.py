import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
# the main-window test needs a real X server (OpenGL widgets); everything else runs offscreen
os.environ.setdefault("QT_QPA_PLATFORM", "xcb" if os.environ.get("DISPLAY") else "offscreen")
from PySide6 import QtCore, QtWidgets  # noqa: E402

from ewald import Pipeline, StaticSource  # noqa: E402
from ewald import modifiers as mods  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ewald.gui import notebook_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    notebook_panel.CONFIRM_UNSAVED = False  # no modal "unsaved changes" questions in tests
    yield app
    # Destroy every window (VTK render windows, in-process kernels) while Qt is still alive;
    # leaving them to interpreter shutdown can segfault after all tests have passed.
    import gc

    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    gc.collect()


@pytest.fixture
def pipe(cu_fcc):
    return Pipeline(StaticSource([cu_fcc, cu_fcc]))


def test_pipeline_panel_add_toggle_reorder(app, pipe):
    from PySide6 import QtCore

    from ewald.gui.pipeline_panel import PipelinePanel

    panel = PipelinePanel()
    panel.set_pipeline(pipe)
    changes = []
    panel.changed.connect(lambda: changes.append(1))
    panel.add_modifier(mods.CoordinationAnalysis(3.0))
    panel.add_modifier(mods.ColorCoding("Coordination"))
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ColorCoding"]
    items = panel.modifier_items()
    assert [it.text(1) for it in items] == ["ColorCoding", "CoordinationAnalysis"]  # newest on top, like OVITO
    panel.tree.setCurrentItem(items[0])
    panel._move(1)
    assert [m.name for m in pipe.modifiers] == ["ColorCoding", "CoordinationAnalysis"]
    panel.modifier_items()[1].setCheckState(0, QtCore.Qt.Unchecked)
    assert pipe.modifiers[0].enabled is False
    pipe.compute(0, errors="skip")
    panel.update_status()
    assert panel.modifier_items()[0].text(2).endswith("ms")
    assert len(changes) >= 3


def test_bonds_visual_element_and_pairs(app, cu_fcc):
    from ewald.gui.style import ViewStyle
    from ewald.render import bond_list

    style = ViewStyle()
    style.apply_preset("Ball & stick")
    style.ensure_bond_pairs(cu_fcc.types)
    assert style.bonds_enabled and style.bond_pairs[0][:2] == ["Cu", "Cu"]
    i, j, d = bond_list(cu_fcc, pairs=style.bond_pairs)
    assert len(i) == 12 * cu_fcc.count  # both directions, 12 neighbours in fcc
    none = bond_list(cu_fcc, pairs=[("Cu", "Zr", 0.0, 3.0)])
    assert len(none[0]) == 0


def test_style_presets_default_ase_and_bond_table():
    from ewald.gui.style import DEFAULT_PRESET, ViewStyle

    style = ViewStyle()
    assert style.preset == DEFAULT_PRESET == "Ball & stick" and style.bonds_enabled
    style.type_colors = {"Cu": (1.0, 0.0, 0.0)}
    style.apply_preset("Simple (ASE)", ["Cu"])
    assert style.atom_style == "flat" and style.background == "white" and style.theme == "ASE"
    assert style.type_colors["Cu"] != (1.0, 0.0, 0.0)  # Jmol colors from the theme
    style.apply_preset("Space-filling")  # leaving the ASE look restores what was there
    assert style.background == "#14161A" and style.type_colors == {"Cu": (1.0, 0.0, 0.0)} and style.theme == ""

    style.ensure_bond_pairs(["Sr", "Ti", "O", "O"])  # compound: no metal-metal rows
    assert {tuple(sorted(r[:2])) for r in style.bond_pairs} == {("O", "O"), ("O", "Sr"), ("O", "Ti")}
    metal = ViewStyle()
    metal.ensure_bond_pairs(["Cu", "Zr"])  # a pure metal keeps its bonds
    assert len(metal.bond_pairs) == 3


def test_command_palette_fuzzy():
    from ewald.gui.palette import fuzzy_score

    assert fuzzy_score("coord", "Coordination analysis / RDF") > fuzzy_score("coord", "Color coding")
    assert fuzzy_score("xyz", "Slice") is None


def test_app_icon_ships_and_loads(app):
    from pathlib import Path

    from ewald.gui.icons import APP_ICON_DIR, APP_ICON_SIZES, app_icon

    sizes = {s.width() for s in app_icon().availableSizes()}
    assert set(APP_ICON_SIZES) <= sizes
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["ewald"]
    shipped = {p for pattern in data for p in (root / "src" / "ewald").glob(pattern)}
    assert {*APP_ICON_DIR.glob("ewald*.*")} <= shipped


def test_vesta_elements_ini(tmp_path):
    from ewald.gui.pipeline_panel import read_vesta_elements

    p = tmp_path / "elements.ini"
    p.write_text("  29         Cu  1.28  1.40  0.77  0.13333  0.48235  0.80000\n")
    colors, radii = read_vesta_elements(p)
    assert radii["Cu"] == 1.28 and abs(colors["Cu"][1] - 0.48235) < 1e-9


def test_parameter_form_roundtrip(app):
    from ewald.gui.params import ParameterForm

    mod = mods.Slice(normal=(0, 0, 1), distance=1.0)
    form = ParameterForm(mod)
    form.editors["distance"][0].setValue(2.5)
    form.editors["normal"][0].setText("(1, 0, 0)")
    form.editors["inverse"][0].setChecked(True)
    form.apply()
    assert mod.distance == 2.5 and mod.normal == (1, 0, 0) and mod.inverse is True


def test_multiline_parameter_and_fes_map(app):
    from ewald.gui.params import ParameterForm
    from ewald.gui.plots import PlotsPanel, _grid

    class Script(mods.Modifier):  # like the ewald-plumed plugin's Plumed: its input is a plumed.dat
        multiline_parameters = ("input",)

        def __init__(self, input: str = "d: DISTANCE ATOMS=1,2"):
            super().__init__()
            self.input = input

    mod = Script()
    form = ParameterForm(mod)
    editor = form.editors["input"][0]
    assert isinstance(editor, QtWidgets.QPlainTextEdit)     # a multi-line text, not one line
    editor.setPlainText("d: DISTANCE ATOMS=1,2\nt: TORSION ATOMS=1,2,3,4\n")
    form.apply()
    assert mod.input.count("\n") == 2
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(-1, 1, 5), indexing="ij")
    table = {"d": x.ravel(), "t": y.ravel(), "Free Energy": (x + y).ravel()}
    names, (ux, uy, z) = _grid(table)
    assert z.shape == (5, 4) and z[0, 3] == pytest.approx(1 - 1)
    assert _grid({"r": [0, 1], "g": [1, 2]}) is None
    from ewald import DataCollection

    d = DataCollection(np.zeros((1, 3)), np.array(["C"]))
    d.tables["fes"] = table
    panel = PlotsPanel()
    panel.set_data(d)
    assert panel.figure.axes[0].get_ylabel() == "t"


def test_python_script_modifier_and_error_skip(pipe):
    script = mods.PythonScript("def modify(frame, data):\n    data['F'] = np.full(data.count, frame)\n")
    bad = mods.ExpressionSelection("NoSuchProperty > 1")
    pipe.modifiers.extend([script, bad])
    d = pipe.compute(1, errors="skip")
    assert np.all(d["F"] == 1)
    assert bad.status and "NoSuchProperty" in bad.status
    with pytest.raises(NameError):
        pipe.compute(1)


def test_inspector_filter(app, cu_fcc):
    from ewald.gui.inspector import DataInspector

    insp = DataInspector()
    insp.set_data(cu_fcc)
    insp.filter.setText("Position.Z < 1")
    insp._apply_filter()
    assert insp.model.rowCount() == int((cu_fcc.positions[:, 2] < 1).sum())


def test_timeline(app):
    from ewald.gui.timeline import Timeline

    tl = Timeline()
    seen = []
    tl.frameChanged.connect(seen.append)
    tl.set_num_frames(5)
    tl._set(3)
    tl._advance()
    tl._advance()
    assert seen == [3, 4, 0]


def test_diffraction_form_and_plots(app, tmp_path):
    from ase.build import bulk

    from ewald import DataCollection
    from ewald.backend import CPU
    from ewald.gui.params import ParameterForm
    from ewald.gui.plots import PlotsPanel
    from ewald.pipeline import ComputeContext

    mod = mods.PowderDiffraction()
    form = ParameterForm(mod)
    radiation = form.editors["radiation"][0]
    assert isinstance(radiation, QtWidgets.QComboBox) and radiation.count() == 2
    radiation.setCurrentText("neutron")
    exp = tmp_path / "measured.xy"
    exp.write_text("20 1\n40 5\n60 2\n")
    form.editors["experimental"][0].setText(str(exp))   # line edit next to a "..." file button
    form.apply()
    assert mod.radiation == "neutron" and mod.experimental == str(exp)
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615))
    out = d.copy()
    mod.modify(out, ComputeContext(0, None, CPU))
    panel = PlotsPanel()
    panel.set_data(out)
    panel.table.setCurrentText("diffraction")
    ax = panel.figure.axes[0]
    assert [l.get_label() for l in ax.lines] == ["Intensity", "Experimental"]
    assert ax.lines[1].get_marker() == "."    # the measured pattern as points
    panel.table.setCurrentText("diffraction-peaks")
    ax = panel.figure.axes[0]
    assert len(ax.collections) == 1 and not ax.lines   # reflections as sticks


def test_plots_thermo_series_marker_follows_frame(app, lammps_dir):
    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(lammps_dir / "thermo" / "dump.lammpstrj")
    panel = PlotsPanel()
    panel.set_data(pipe.compute(0))
    assert panel.table.currentText() == "Thermo"
    assert panel.column.currentText() == "Temp"  # one column at a time for a time series
    ax = panel.figure.axes[0]
    assert ax.get_xlabel() == "Step" and len(ax.lines) == 2  # Temp + frame marker
    panel.column.setCurrentText("Volume")  # only printed in run 2: NaNs dropped
    np.testing.assert_array_equal(panel.figure.axes[0].lines[0].get_xdata(), [20, 25, 30, 35, 40])
    ax = panel.figure.axes[0]
    panel.set_data(pipe.compute(3))
    assert panel.figure.axes[0] is ax  # frame change moves the marker, no full redraw
    assert list(panel._marker.get_xdata()) == [30, 30]


def test_timeline_frame_selection_playback(app):
    from ewald.gui.timeline import Timeline

    tl = Timeline()
    tl.set_num_frames(10)
    seen, cleared = [], []
    tl.frameChanged.connect(seen.append)
    tl.selectionChanged.connect(cleared.append)
    tl.set_selection([7, 2, 2, 5, 42])
    assert tl.selection == [2, 5, 7] and tl.sel.text() == "3 selected ✕"
    tl._set(3)
    tl._advance(); tl._advance(); tl._advance()
    assert seen == [3, 5, 7, 2]  # playback steps through the selected frames, looping
    tl.grab()  # paints the selection ticks
    tl.sel.click()
    assert tl.selection == [] and cleared == [[]] and tl.sel.isHidden()


def test_plots_drag_selects_atoms_and_shows_viewport_selection(app, cu_fcc):
    from ewald.gui.plots import PlotsPanel

    pipe = Pipeline(StaticSource([cu_fcc]), [mods.CoordinationAnalysis(3.0), mods.CreateBonds(3.0),
                                             mods.BondLengthDistribution()])
    d = pipe.compute(0)
    panel = PlotsPanel()
    panel.pipeline = pipe
    panel.show()  # the highlight is only computed while the tab is visible
    got, exprs = [], []
    panel.atomsSelected.connect(got.append)
    panel.expressionRequested.connect(exprs.append)
    panel.set_data(d)
    panel.hist.setCurrentText("Coordination")
    assert panel.link.kind == "property" and panel.keep.isVisibleTo(panel) and not panel.keep.isEnabled()
    panel.select(11.5, 12.5)
    assert len(got[-1]) == d.count and panel.link_label.text().startswith(f"{d.count} particles")
    panel._keep()
    assert exprs == ["(Coordination >= 11.5) & (Coordination <= 12.5)"]
    panel.set_data(d)  # new frame: the range is re-applied
    assert len(got) == 2
    panel.set_highlight([0, 1, 2])
    bars = [a for a in panel._overlays if hasattr(a, "get_height") and round(a.get_alpha(), 2) == 0.85]
    assert sum(b.get_height() for b in bars) == 3 and panel._ax2 is None  # counts: same axis
    # the g(r) plot: pairs at the first-neighbour distance
    panel.hist.setCurrentText("")
    panel.table.setCurrentText("coordination-rdf")
    assert panel.link.kind == "pairs" and panel._range is None  # a new plot drops the old range
    panel.select(2.4, 2.7)
    assert len(got[-1]) == d.count
    assert panel._ax2 is not None and panel._ax2.get_visible()  # g(r): selection on a second axis
    panel.select(0.5, 1.0)
    assert len(got[-1]) == 0
    panel.clear_selection()
    assert panel._range is None and len(got[-1]) == 0 and not panel.clear.isEnabled()


def test_plots_time_series_selects_and_jumps_to_frames(app, lammps_dir):
    from matplotlib.backend_bases import MouseEvent

    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(lammps_dir / "thermo" / "dump.lammpstrj")
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.pipeline = pipe
    frames, jumps = [], []
    panel.framesSelected.connect(frames.append)
    panel.frameRequested.connect(jumps.append)
    panel.set_data(pipe.compute(0))
    assert panel.link.kind == "frames" and not panel.keep.isVisibleTo(panel)
    panel.select(15, 32)
    assert frames == [[2, 3]] and "2 frames selected" in panel.link_label.text()
    marks = [ln for ln in panel._overlays if hasattr(ln, "get_xdata") and ln.get_marker() == "o"]
    assert list(marks[0].get_xdata()) == [20, 30]
    panel.set_frame_selection([0, 4])  # from the timeline: marks only
    assert frames == [[2, 3]]
    # mouse: a click jumps, a drag selects
    panel.canvas.draw()
    ax = panel._ax

    def event(kind, x, y):
        px, py = ax.transData.transform((x, y))
        return MouseEvent(kind, panel.canvas, px, py, button=1)

    for kind in ("button_press_event", "button_release_event"):
        panel.canvas.callbacks.process(kind, event(kind, 21, ax.get_ylim()[0] + 1))
    assert jumps == [2]
    panel.canvas.callbacks.process("button_press_event", event("button_press_event", 5, ax.get_ylim()[0] + 1))
    panel.canvas.callbacks.process("motion_notify_event", event("motion_notify_event", 22, ax.get_ylim()[0] + 1))
    panel.canvas.callbacks.process("button_release_event", event("button_release_event", 22, ax.get_ylim()[0] + 1))
    assert frames[-1] == [1, 2]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_links_plots_and_viewport(app, cu_fcc, lammps_dir):
    from ewald import import_file
    from ewald.gui.app import MainWindow

    w = MainWindow(Pipeline(StaticSource([cu_fcc, cu_fcc]), [mods.CoordinationAnalysis(3.0)]))
    assert w.plots.data is None  # a hidden tab is filled when it is opened
    w.bottom.setCurrentWidget(w.plots)
    w.plots.table.setCurrentText("coordination-rdf")
    w.plots.select(2.4, 2.7)
    assert len(w.viewport.picks) == cu_fcc.count and w._ask_context()[2] == w.viewport.picks
    assert f"{cu_fcc.count} atoms picked" in w.pick_label.text()
    w.timeline._set(1)  # the plot range follows the frame
    assert len(w.viewport.picks) == cu_fcc.count
    w.viewport.set_tool("pick")
    w.viewport.add_pick(3)  # a click in the view replaces the plot range ...
    assert w.plots._range is None and w.plots.highlight == [3]  # ... and shows on the plot
    w.plots.table.setCurrentText("")
    w.plots.hist.setCurrentText("Coordination")
    w.plots.select(11, 13)
    w.plots._keep()
    assert w.pipeline.modifiers[-1].name == "ExpressionSelection" and w.viewport.picks == []
    assert len(w.plots.highlight) == cu_fcc.count  # the Selection property is drawn on the plot
    w.close()
    w = MainWindow(import_file(lammps_dir / "thermo" / "dump.lammpstrj"))
    w.bottom.setCurrentWidget(w.plots)
    w.plots.select(15, 32)
    assert w.timeline.selection == [2, 3] and w.frame == 2
    w.plots.frameRequested.emit(4)
    assert w.frame == 4
    w.timeline.sel.click()
    assert w.plots.frame_selection == []
    w.close()


def test_plots_neb_profile_marks_image_and_jumps(app, vasp_dir):
    from matplotlib.backend_bases import MouseEvent

    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(vasp_dir / "neb")
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.pipeline = pipe
    panel.set_data(pipe.compute(1))
    assert panel.table.currentText() == "NEB" and panel.column.currentText() == "Energy"
    ax = panel.figure.axes[0]
    assert ax.get_xlabel().startswith("Reaction coordinate") and "Ea →" in ax.texts[0].get_text()
    spline, points = ax.lines[:2]
    assert len(spline.get_xdata()) == 200 and len(points.get_xdata()) == 4  # image 04 has no energy
    d = pipe.compute(2)
    panel.set_data(d)
    assert panel.figure.axes[0] is ax  # frame change moves the marker only
    assert list(panel._marker.get_xdata()) == [d.attributes["NEB.Reaction Coordinate"]]
    assert panel.link.kind == "frames"  # linked by the table's Frame column
    jumps = []
    panel.frameRequested.connect(jumps.append)
    panel.canvas.draw()
    ax = panel._ax
    x3 = d.tables["NEB"]["Reaction Coordinate"][3]
    px, py = ax.transData.transform((x3, 0.4))
    for kind in ("button_press_event", "button_release_event"):
        panel.canvas.callbacks.process(kind, MouseEvent(kind, panel.canvas, px, py, button=1))
    assert jumps == [3]
    panel.set_data(pipe.compute(4))
    assert not panel._marker.get_visible()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_viewport_context_menu_toggles_frame_overlay(app, vasp_dir):
    from ewald import import_file
    from ewald.gui.app import MainWindow

    w = MainWindow(import_file(vasp_dir / "neb"))  # NEB sources start with OverlayFrames on
    w.timeline._set(2)
    n_real = w.pipeline.source.compute(2).count
    assert w.data.count > n_real
    def entries(menu):
        return {a.text(): a for a in menu.actions() if a.text()}

    menu = entries(w.viewport_menu())
    assert "NEB energy profile" in menu and "Add isosurface" not in menu and "Clear picks" not in menu
    overlay, transparency, moving = (menu[k] for k in ("Overlay other frames (see-through)", "Overlay transparency",
                                                       "Overlay only atoms that move"))
    assert overlay.isChecked() and transparency.menu().isEnabled() and moving.isChecked()
    overlay.setChecked(False)  # toggled -> off
    assert w.data.count == n_real and not w._overlay().enabled
    w.set_overlay(True, transparency=0.5)
    ghosts = w.data["Overlay Frame"] >= 0
    assert np.allclose(w.data["Transparency"][ghosts], 0.5)
    menu = entries(w.viewport_menu())
    assert [a.text() for a in menu["Overlay transparency"].menu().actions() if a.isChecked()] == ["50 %"]
    menu["Overlay only atoms that move"].setChecked(False)  # every atom, moving or not
    assert w.data.count == 5 * n_real
    menu["NEB energy profile"].trigger()
    assert w.bottom.currentWidget() is w.plots and w.plots.table.currentText() == "NEB"
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_viewport_context_menu_follows_data_and_picks(app, cu_fcc, tmp_path):
    from ewald.gui.app import MainWindow

    def entries(menu):
        return {a.text(): a for a in menu.actions() if a.text()}

    w = MainWindow()
    assert "Load file…" in entries(w.viewport_menu())  # nothing loaded yet
    w.set_pipeline(Pipeline(StaticSource([cu_fcc])))
    menu = entries(w.viewport_menu())
    assert "Load file…" not in menu and "Play" not in menu and "Clear picks" not in menu  # one frame, no picks
    assert "View along" in menu and "Ask Claude about this structure…" in menu
    w.viewport.set_tool("pick")
    w.viewport.add_pick(3)
    w.viewport.add_pick(5, extend=True)
    menu = entries(w.viewport_menu())
    assert "2 atoms picked" in menu and not menu["2 atoms picked"].isEnabled()
    menu["Delete picked atoms"].trigger()
    assert w.data.count == cu_fcc.count - 2 and w.viewport.picks == []
    assert [m.name for m in w.pipeline.modifiers] == ["ExpressionSelection", "DeleteSelected"]
    show = entries(entries(w.viewport_menu())["Show"].menu())
    show["Bonds"].setChecked(True)
    assert w.style.bonds_enabled
    w.open_example("CHGCAR.Si")
    entries(w.viewport_menu())["Add isosurface"].trigger()
    assert w.data.surfaces and "Edit isosurface" in entries(w.viewport_menu())
    w.close()  # the phonon entries are tested with the ewald-phonopy plugin


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_smoke(app, pipe, tmp_path):
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.panel.add_modifier(mods.CoordinationAnalysis(3.0))
    w.timeline._set(1)
    assert w.data.count == pipe.compute(1).count and "Coordination" in w.data
    w.viewport.view_along("c*")
    w.viewport.rotate("azimuth", 15)
    w.viewport.set_tool("angle")
    w.viewport._picks = [0, 1, 2]
    assert "°" in w.viewport.describe_picks()
    w.render_page.output.setText(str(tmp_path / "x.png"))
    w.render_page.preset.setCurrentText("Custom (px)")
    w.render_page.width.setValue(320)
    w.render_page.height.setValue(240)
    w.render_from_page()
    assert os.path.getsize(tmp_path / "x.png") > 1000
    code = w.render_page.code.toPlainText()
    assert "CoordinationAnalysis(cutoff=3.0" in code and "camera=" in code
    w.close()


def test_preferences_claude_page_saves_key_outside_settings(app, monkeypatch):
    from ewald import settings
    from ewald.gui.preferences import PreferencesDialog
    from ewald.llm import credentials

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_keyring", lambda: None)   # file fallback, like a headless box
    checked = []
    monkeypatch.setattr(credentials, "check", lambda key, model: checked.append((key, model)) or (True, "Connected"))
    s = settings.load()
    dlg = PreferencesDialog(s)
    assert "Not set" in dlg.key_status.text()
    assert dlg.key_edit.echoMode() == dlg.key_edit.EchoMode.Password
    dlg.key_edit.setText("sk-ant-gui-key-0123456789")
    dlg.set_value("llm.model", "claude-sonnet-5-5")
    dlg.test_api_key(wait=True)
    assert checked == [("sk-ant-gui-key-0123456789", "claude-sonnet-5-5")]
    assert dlg.key_result.text() == "Connected"
    assert credentials.get_api_key() == (None, None)          # nothing saved before Apply
    dlg.apply()
    assert credentials.get_api_key() == ("sk-ant-gui-key-0123456789", "file")
    assert dlg.key_edit.text() == "" and "sk-ant-…6789" in dlg.key_status.text()
    assert "sk-ant" not in s.path.read_text() and settings.load()["llm.model"] == "claude-sonnet-5-5"
    dlg._search("api key")
    visible = {dlg.pages.item(i).text() for i in range(dlg.pages.count()) if not dlg.pages.item(i).isHidden()}
    assert visible == {"Claude"}
    dlg._search("")
    dlg.remove_api_key()
    assert credentials.get_api_key()[0] is not None            # removal waits for Apply too
    dlg.apply()
    assert credentials.get_api_key() == (None, None)
    dlg.close()


def test_ask_panel_agent_uses_claude_preferences(app, monkeypatch):
    from ewald import settings
    from ewald.gui import ask_panel

    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw))
    s = settings.load()
    s["llm.model"], s["llm.effort"] = "claude-haiku-5-5", "low"
    ask_panel.agent_from_settings(lambda: s)(None, None)
    assert made == [{"model": "claude-haiku-5-5", "effort": "low"}]


def test_preferences_dialog_edit_search_restore(app, tmp_path):
    from ewald import settings
    from ewald.gui.preferences import ElementTable, PreferencesDialog

    s = settings.load()
    dlg = PreferencesDialog(s)
    assert [dlg.pages.item(i).text() for i in range(dlg.pages.count())] == list(settings.PAGES)
    dlg.set_value("viewport.rotation_step", 45.0)
    dlg.set_value("compute.device", "cpu")
    assert s.is_default("viewport.rotation_step")           # untouched until Apply
    applied = []
    dlg.applied.connect(applied.append)
    dlg.apply()
    assert applied and s["viewport.rotation_step"] == 45.0
    assert settings.load()["compute.device"] == "cpu"        # saved to disk
    dlg._search("radius")
    visible = {dlg.pages.item(i).text() for i in range(dlg.pages.count()) if not dlg.pages.item(i).isHidden()}
    assert visible == {"Elements"}
    dlg._search("")
    dlg.pages.setCurrentRow(1)                                # Viewports
    dlg.restore_page()
    assert dlg.work.is_default("viewport.rotation_step") and not dlg.work.is_default("compute.device")

    ini = tmp_path / "elements.ini"
    ini.write_text("  29  Cu  1.28  1.40  0.73  0.13333  0.38039  0.66667\n")
    table = dlg.editors["elements.overrides"][0]
    assert isinstance(table, ElementTable)
    table.import_vesta(str(ini))
    assert dlg.work["elements.overrides"]["Cu"] == {"color": "#2261aa", "radius": 1.28}


def test_element_defaults_fill_only_unset_types(app, cu_fcc):
    from ewald import settings
    from ewald.gui.preferences import apply_element_defaults
    from ewald.gui.style import ViewStyle

    style = ViewStyle()
    s = settings.Settings()
    apply_element_defaults(style, cu_fcc, s)
    assert style.type_colors == {}                            # defaults: nothing redundant
    s["elements.overrides"] = {"Cu": {"color": "#c8794a"}}
    apply_element_defaults(style, cu_fcc, s)
    assert style.type_colors["Cu"] == "#c8794a"
    s["elements.overrides"] = {}
    s["elements.color_scheme"] = "CPK"
    apply_element_defaults(style, cu_fcc, s)            # previous fill replaced
    assert style.type_colors["Cu"] != "#c8794a"
    style.type_colors["Cu"] = "#000000"                       # session edit wins
    apply_element_defaults(style, cu_fcc, s)
    assert style.type_colors["Cu"] == "#000000"


def test_ionic_radius_preset_fills_type_radii(app):
    from ase.spacegroup import crystal

    from ewald import settings
    from ewald.data import DataCollection
    from ewald.gui.preferences import apply_element_defaults
    from ewald.gui.style import ViewStyle

    sto = DataCollection.from_ase(crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                                          spacegroup=221, cellpar=[3.905] * 3 + [90] * 3))
    style, s = ViewStyle(), settings.Settings()
    style.radius_preset = "vdW"
    assert apply_element_defaults(style, sto, s) is None and style.type_radii == {}   # preset goes via kwargs
    assert style.scene_kwargs()["radius_preset"] == "vdW"
    style.radius_preset = "ionic"
    table = apply_element_defaults(style, sto, s)
    assert style.type_radii["Ti"] == pytest.approx(0.8 * 0.605) and table.coordination["Sr"] == 12
    style.oxidation_states = {"Ti": 3}
    apply_element_defaults(style, sto, s)                     # refilled for the new oxidation state
    assert style.type_radii["Ti"] == pytest.approx(0.8 * 0.67)
    assert style.scene_kwargs()["oxidation_states"] == {"Ti": 3}


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_applies_settings(app, pipe):
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.gui.preferences import apply_settings

    s = settings.Settings()
    s["viewport.background"] = "#ffffff"
    s["viewport.style_preset"] = "Ball & stick"
    s["viewport.rotation_step"] = 30.0
    s["render.renderer"] = "opengl"
    s["elements.overrides"] = {"Cu": {"color": "#c8794a"}}
    w = MainWindow(pipe, settings=s)
    assert w.device == "auto" and w.style.background == "#ffffff" and w.style.bonds_enabled
    assert w.viewport.rotation_step == 30.0 and not w.render_page.raytrace
    assert w.style.type_colors["Cu"] == "#c8794a"
    assert any(a.text() == "&Preferences…" for a in w.menuBar().actions()[1].menu().actions())
    s["bonds.length_factor"] = 1.0
    s["viewport.show_axes"] = False
    apply_settings(w, s)
    assert w.style.bond_pairs[0][3] == round(1.0 * 2 * 1.32, 2)
    w.close()
    w.deleteLater()
    assert (s.path.parent / "layout.json").exists()


def test_preferences_layout_covers_every_option_once():
    from ewald.gui.preferences import LAYOUT
    from ewald.settings import OPTIONS, PAGES

    keys = [k for name in PAGES for _title, ks in LAYOUT[name][1] for k in ks]
    assert sorted(keys) == sorted(o.key for o in OPTIONS)
    for name in PAGES:
        for _title, ks in LAYOUT[name][1]:
            assert all(__import__("ewald.settings").settings.OPTION[k].page == name for k in ks)


def test_remote_file_dialog_browse_and_open(app, server):
    from ewald.gui import remote_dialog
    from ewald.gui.remote_dialog import RemoteFileDialog

    dlg = RemoteFileDialog(host="hpc")
    assert "hpc" in [dlg.host.itemText(i) for i in range(dlg.host.count())]
    assert dlg.browse("")  # home
    assert dlg.cwd == "/home/alice"
    assert not dlg.browse("/nowhere") and dlg.status.text().startswith(("FileNotFoundError", "OSError"))
    assert dlg.browse("/data")
    names = [dlg.list.topLevelItem(i).text(0) for i in range(dlg.list.topLevelItemCount())]
    assert "XDATCAR" in names and "dump.custom.lammpstrj" in names
    item = dlg.list.findItems("XDATCAR", QtCore.Qt.MatchExactly)[0]
    dlg.list.setCurrentItem(item)
    dlg._accept()
    assert dlg.url == "hpc:/data/XDATCAR" and dlg.result() == QtWidgets.QDialog.Accepted
    assert remote_dialog.load_recent()[0] == "hpc:/data/XDATCAR"
    assert RemoteFileDialog().recent.count() == 2

    from ewald import import_file

    assert import_file(dlg.url).num_frames > 1


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_session_save_load_and_publication_output(app, pipe, tmp_path):
    from ewald.gui.app import MainWindow
    from ewald.render import camera_state

    w = MainWindow(pipe)
    w.panel.add_modifier(mods.ColorCoding("Position.Z", cmap="plasma"))
    w.style.apply_preset("Ball & stick")
    w.viewport.view_along("a")
    page = w.render_page
    page.set_renderer("vector")
    assert page.output.text().endswith(".svg")
    page.preset.setCurrentText("Custom (px)")
    page.width.setValue(200)
    page.height.setValue(150)
    page.type_legend.setChecked(True)
    page.scale_bar.setChecked(True)
    page.text.setText("frame [Frame]")
    page.output.setText(str(tmp_path / "fig.svg"))
    w.render_from_page()
    assert "frame 0" in (tmp_path / "fig.svg").read_text()
    assert not page.preview.pixmap().isNull()
    page.movie_format.setCurrentText("GIF")
    assert page.is_movie and page.output.text().endswith(".gif")
    assert "render_animation(pipe" in page.code.toPlainText()
    assert w.export_movie(page.output.text()) and os.path.getsize(page.output.text()) > 100
    cam = camera_state(w.viewport.plotter)
    path = w.save_session(str(tmp_path / "s.ewald"))

    w2 = MainWindow()
    w2.load_session(path)
    assert [m.name for m in w2.pipeline.modifiers] == ["ColorCoding"]
    assert w2.style.preset == "Ball & stick" and w2.style.radius_scale == w.style.radius_scale
    assert w2.render_page.renderer_name == "vector" and w2.render_page.type_legend.isChecked()
    assert w2.render_page.text.text() == "frame [Frame]" and w2.render_page.width.value() == 200
    np.testing.assert_allclose(camera_state(w2.viewport.plotter)["position"], cam["position"], rtol=1e-6)
    w.close()
    w2.close()


def test_render_page_quality_presets(app):
    from ewald.gui.render_page import RenderPage
    from ewald.render import QUALITY_PRESETS

    class Win:  # the page only needs these for code generation
        pipeline = None
        frame = 0
        device = None
        data = None

    page = RenderPage.__new__(RenderPage)
    page.update_code = lambda *a: None
    RenderPage.__init__(page, Win())
    assert page.quality.currentText() == "Publication"
    assert page.samples.value() == QUALITY_PRESETS["publication"]["samples"]
    assert page.render_kwargs()["quality"] == "publication" and "samples" not in page.render_kwargs()
    page.quality.setCurrentText("Draft")
    assert page.samples.value() == 1 and not page.shadows.isChecked()
    page.samples.setValue(5)
    assert page.quality.currentText() == "Custom" and page.render_kwargs()["samples"] == 5
    page.movie_quality.setCurrentText("Standard")
    assert page.render_kwargs(movie=True) == {**page.render_kwargs(movie=True), "quality": "standard"}
    st = page.state()
    page.quality.setCurrentText("Publication")
    page.set_state(st)
    assert page.quality.currentText() == "Custom" and page.samples.value() == 5


def test_help_menu_and_late_registered_modifier(app, pipe, monkeypatch):
    from ewald import settings
    from ewald.gui import pipeline_panel
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    s["general.docs_url"] = "https://example.org/ewald"
    w = MainWindow(pipe, settings=s)
    menus = {a.text().replace("&", ""): a.menu() for a in w.menuBar().actions()}
    labels = [a.text() for a in menus["Help"].actions()]
    assert "Writing plugins" in labels and "Open plugin folder" in labels
    assert w.help_url("plugins/index.md") == "https://example.org/ewald/plugins/"

    # registered after the window exists (startup script, plugin): appears when the menus open
    monkeypatch.setitem(pipeline_panel.MODIFIER_MENU, "Plugins", {"Late wrap": mods.WrapAtoms})
    for menu in (w.panel.add_menu, w.step_menu):
        menu.aboutToShow.emit()
        groups = {a.text(): a.menu() for a in menu.actions()}
        assert [a.text() for a in groups["Plugins"].actions()] == ["Late wrap"]
    w.close()
    w.deleteLater()


def test_bond_tool_and_menu(app, srtio3_pipe):
    from ewald.gui.app import MainWindow
    from ewald.gui.pipeline_panel import MODIFIER_MENU

    assert "Coordination polyhedra" in MODIFIER_MENU["Bonds & polyhedra"]
    w = MainWindow(srtio3_pipe)
    w.recompute()
    n = w.data.bonds.count
    w.toggle_bond(1, 2)                                  # Ti-O bond exists: delete it
    assert isinstance(srtio3_pipe.modifiers[-1], mods.EditBonds) and w.data.bonds.count == n - 1
    w.toggle_bond(1, 2)                                  # and add it back
    assert w.data.bonds.count == n and len(srtio3_pipe.modifiers) == 3
    assert "coordination-polyhedra" not in w.data.surfaces
    srtio3_pipe.modifiers.append(mods.CoordinationPolyhedra(["Ti"]))
    w.recompute()
    assert any(name.startswith("surface:") for name in w.viewport.plotter.actors)
    w.close()


@pytest.fixture
def srtio3_pipe():
    from ase.spacegroup import crystal

    from ewald import DataCollection

    atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                    cellpar=[3.905] * 3 + [90] * 3)
    return Pipeline(StaticSource(DataCollection.from_ase(atoms)),
                    [mods.CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]), mods.Replicate((2, 2, 2))])


def test_viewport_draws_and_clears_surfaces(app):
    from ewald.data import Cell, DataCollection, VoxelGrid
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    cell = Cell(np.diag([5.0, 5.0, 5.0]))
    f = np.stack(np.meshgrid(*[np.arange(12) / 12] * 3, indexing="ij"), -1)
    data = DataCollection([[2.5, 2.5, 2.5]], ["Si"], cell)
    data.grids["g"] = VoxelGrid(cell.copy(), {"Charge Density": np.exp(-(((f - 0.5) * 5) ** 2).sum(-1))})
    pipe = Pipeline(StaticSource(data), [mods.CreateIsosurface(0.3, transparency=0.5), mods.GridSlice()])
    v = Viewport(ViewStyle())
    v.set_data(pipe.compute())
    assert {"surface:isosurface", "surface:slice"} <= set(v.plotter.actors)
    v.set_data(data)
    assert not any(str(k).startswith("surface:") for k in v.plotter.actors)
    v.close()


class _FakeAgent:
    def __init__(self, pipeline, on_event):
        self.toolset = type("TS", (), {"pipeline": pipeline, "frame": 0, "selection": None})()
        self.on_event = on_event
        self.calls = []

    def ask_about_selection(self, question, indices, frame=None):
        self.calls.append(("selection", question, list(indices), frame))
        self.on_event("tool_call", {"name": "describe_selection", "input": {}})
        self.toolset.pipeline.modifiers.append(mods.CommonNeighborAnalysis())
        return f"{len(indices)} atoms: **fcc** neighbors"

    def ask(self, question):
        self.calls.append(("frame", question))
        return "the whole frame"


def _wait(app, panel):
    import time

    t = time.time()
    while panel.busy() and time.time() - t < 10:
        app.processEvents()
    app.processEvents()


def test_ask_panel_selection_and_frame(app, pipe):
    from ewald.gui.ask_panel import AskPanel

    ctx = {"indices": [3, 1]}
    agents = []
    panel = AskPanel(lambda: (pipe, 1, ctx["indices"]),
                     agent_factory=lambda p, ev: agents.append(_FakeAgent(p, ev)) or agents[-1])
    panel.update_selection()
    assert panel.selection_label.text() == "2 atoms selected"
    changed = []
    panel.pipelineChanged.connect(lambda: changed.append(1))
    panel.input.setPlainText("What are these?")
    panel.ask()
    _wait(app, panel)
    assert agents[0].calls == [("selection", "What are these?", [3, 1], 1)]
    text = panel.transcript.toPlainText()
    assert "What are these?" in text and "2 atoms: fcc neighbors" in text
    assert changed == [1] and panel.ask_button.isEnabled() and panel.input.toPlainText() == ""

    ctx["indices"] = []
    panel.ask("And the frame?")
    _wait(app, panel)
    assert agents[0].calls[-1] == ("frame", "And the frame?") and len(agents) == 1   # same conversation
    panel.reset()
    assert panel.transcript.toPlainText() == ""


def test_ask_panel_without_claude(app, pipe):
    from ewald.gui.ask_panel import AskPanel

    def missing(p, ev):
        raise ModuleNotFoundError("No module named 'anthropic'")

    panel = AskPanel(lambda: (pipe, 0, []), agent_factory=missing)
    panel.ask("hello")
    assert "not available" in panel.transcript.toPlainText() and "pip install ewald[llm]" in panel.transcript.toPlainText()
    assert not panel.busy()


def test_main_window_ask_uses_picks_and_selection(app, pipe):
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.viewport.set_tool("pick")
    w.viewport.add_pick(5)
    w.viewport.add_pick(7, extend=True)
    w.viewport.add_pick(9, extend=True)
    w.viewport.add_pick(7, extend=True)           # Shift+click again removes it
    assert w._ask_context() == (pipe, 0, [5, 9])
    assert "2 atoms picked" in w.pick_label.text() and w.ask_panel.selection_label.text() == "2 atoms selected"
    w.viewport.clear_picks()
    pipe.modifiers.append(mods.ExpressionSelection("Position.Z < 1"))
    w.recompute()
    assert len(w._ask_context()[2]) == int((w.data.positions[:, 2] < 1).sum()) > 0
    w.show()
    app.processEvents()
    shown = lambda wid: not wid.visibleRegion().isEmpty()  # noqa: E731
    assert shown(w.display) and not shown(w.ask_dock)    # tabbed with the display panel, which is in front
    w.ask_claude()
    app.processEvents()
    assert shown(w.ask_dock) and not shown(w.display)
    w.close()


def test_viewport_draws_vector_arrows(app):
    from ewald.data import Cell, DataCollection
    from ewald.gui.params import ParameterForm
    from ewald.gui.pipeline_panel import MODIFIER_MENU
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    menu = MODIFIER_MENU["Vectors (arrows)"]
    assert all(isinstance(f(), mods.VectorVis) for f in menu.values())
    data = DataCollection([[1, 1, 1], [3, 3, 3]], ["Fe", "Fe"], Cell(np.eye(3) * 4))
    data.particles["Force"] = np.array([[0, 0, 1.0], [1.0, 0, 0]])
    mod = mods.VectorVis()
    form = ParameterForm(mod)
    form.editors["alignment"][0].setCurrentText("center")  # choices, not free text
    form.apply()
    assert mod.alignment == "center"
    v = Viewport(ViewStyle())
    v.arrow_resolution = 5
    v.set_data(Pipeline(StaticSource(data), [mod]).compute())
    actor = v.plotter.actors["vectors:Force"]
    assert actor.GetMapper().GetInput().GetNumberOfPoints() == 2 * (6 * 5 + 2)
    v.set_data(data)
    assert not any(str(k).startswith("vectors:") for k in v.plotter.actors)
    v.close()
@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_viewport_impostors_smooth_edges_and_opengl_check(app, pipe):
    from ewald import settings
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    w = MainWindow(pipe, settings=s)
    w.show()
    app.processEvents()
    vp = w.viewport
    assert "PointGaussian" in vp.plotter.actors["atoms"].GetMapper().GetClassName()  # default: impostors
    assert vp.anti_aliasing and vp.plotter.renderer.GetUseFXAA()
    gl = w.check_opengl()
    assert gl["renderer"] and "OpenGL" in w.hardware_summary()
    if gl["software"]:  # xvfb: Mesa llvmpipe
        assert "software OpenGL" in w.statusBar().currentMessage()
    s["viewport.sphere_drawing"], s["viewport.anti_aliasing"] = "mesh", False
    from ewald.gui import preferences as prefs

    prefs.apply_settings(w, s)
    vp.redraw()
    assert "PointGaussian" not in vp.plotter.actors["atoms"].GetMapper().GetClassName()
    assert not vp.plotter.renderer.GetUseFXAA()
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_interactive_detail_coarse_then_refine(app, pipe):
    from ewald import settings
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    s["viewport.interactive_detail"] = "coarse"
    s["performance.cache_memory"] = 64
    w = MainWindow(pipe, settings=s)
    vp, n = w.viewport, pipe.compute(0).count

    def impostors(name="atoms"):  # point-sprite actor -> its atom count, sphere mesh -> None
        # (the full scene is impostors too, by default: Sphere drawing = impostor)
        mapper = vp.plotter.actors[name].GetMapper()
        if "PointGaussian" not in mapper.GetClassName():
            return None
        source = mapper.GetInputAlgorithm()
        source.Update()
        return source.GetOutputDataObject(0).GetNumberOfPoints()

    assert pipe.cache_memory == 64 * 2**20
    vp.interactive_atoms = 100  # below the Preferences minimum, to subsample the 256-atom test cell
    assert not vp.coarse and impostors() == n
    # scrubbing: requests made while a frame is computed collapse into the latest one
    w.timeline.slider.setSliderDown(True)
    calls = []
    real = w.recompute
    w.recompute = lambda interactive=False: (calls.append(interactive), real(interactive))
    w.timeline._set(1)
    w.timeline._set(0)
    w.timeline._set(1)
    app.processEvents()
    assert calls.count(True) == 1 and calls[-1] and w.frame == 1 and vp.coarse
    assert 0 < impostors() <= 100 < n  # every k-th atom, as impostors
    vp.refine()
    assert not vp.coarse and impostors() == n
    # rotating: atoms swap for a subsampled impostor actor until the camera rests
    vp._go_coarse_for_motion()
    assert vp.coarse and not vp.plotter.actors["atoms"].GetVisibility()
    assert 0 < impostors("atoms_lod") <= 100
    vp.refine()
    assert "atoms_lod" not in vp.plotter.actors and vp.plotter.actors["atoms"].GetVisibility()
    # 'full' never goes coarse; 'auto' only when a full frame misses the target rate
    vp.detail_mode = "full"
    assert not vp.needs_coarse(scrub=True)
    vp.detail_mode, vp._render_time = "auto", 1.0
    assert vp.needs_coarse() and not (setattr(vp, "_render_time", 0.0) or vp.needs_coarse())
    w.timeline.slider.setSliderDown(False)
    w.close()
    w.deleteLater()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_feature_inventory_gui_section_is_current(tmp_path):
    # in a fresh process: other tests register plugin modifiers in the (global) Pipeline menu
    import subprocess
    import sys
    from pathlib import Path

    from ewald import inventory

    env = dict(os.environ, EWALD_CONFIG_DIR=str(tmp_path))
    code = "from ewald import inventory; print(*inventory.live_gui_lines(), sep=chr(10))"
    live = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                          timeout=300).stdout
    saved = inventory.saved_gui_lines((Path(__file__).parents[1] / "docs" / "features.md").read_text())
    assert saved == live.rstrip("\n").split("\n"), (
        "the GUI section of docs/features.md is stale; regenerate with: "
        "xvfb-run -a python -m ewald inventory -o docs/features.md --gui")


def test_plugin_visual_element_in_panel_and_viewport(app, monkeypatch):
    import shutil
    from pathlib import Path

    from ewald import plugins
    from ewald.data import Cell, DataCollection
    from ewald.gui.pipeline_panel import PipelinePanel
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parents[1] / "examples" / "plugins" / "cell_axes.py", folder)
    data = DataCollection([[1, 1, 1], [3, 3, 3]], ["Fe", "Fe"], Cell(np.eye(3) * 4))
    style = ViewStyle()
    panel = PipelinePanel(style)
    panel.set_pipeline(Pipeline(StaticSource(data)))
    panel._fill_visual_menu()
    assert [a.text() for a in panel.visual_menu.actions()] == ["Cell axes", "Selection halo"]
    redraws = []
    panel.styleChanged.connect(lambda: redraws.append(1))
    panel.add_visual("Cell axes")
    assert style.visuals == [{"name": "Cell axes", "params": {"length": 2.0, "width": 0.25}, "enabled": True}]
    form = panel._form  # the new element is selected and its form is open
    form.editors["length"][0].setValue(3.0)
    panel.scroll.widget().findChild(QtWidgets.QPushButton).click()
    assert style.visuals[0]["params"]["length"] == 3.0 and redraws
    v = Viewport(style)
    v.set_data(data)
    assert "visual:cell-axes" in v.plotter.actors
    item = [panel.tree.topLevelItem(k) for k in range(panel.tree.topLevelItemCount())
            if panel.tree.topLevelItem(k).text(1) == "Cell axes"][0]
    item.setCheckState(0, QtCore.Qt.Unchecked)
    assert style.visuals[0]["enabled"] is False
    v.redraw()
    assert "visual:cell-axes" not in v.plotter.actors
    panel.tree.setCurrentItem(item)
    panel._remove()
    assert style.visuals == []
    v.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_theme_menu_and_render_page_overlays(app, pipe, tmp_path):
    from ewald.gui.app import MainWindow
    from ewald.gui.params import ParameterForm

    w = MainWindow(pipe)
    w._fill_theme_menu()
    names = [a.text() for a in w.theme_menu.actions()]
    assert names[:2] == ["Default", "VESTA"] and "Dark slide" in names
    [a for a in w.theme_menu.actions() if a.text() == "Dark slide"][0].trigger()
    assert w.style.theme == "Dark slide" and w.style.background == "black" and "Cu" in w.style.type_colors
    w._fill_theme_menu()
    assert [a.text() for a in w.theme_menu.actions() if a.isChecked()] == ["Dark slide"]

    page = w.render_page
    page._fill_overlay_menu()
    assert [a.text() for a in page.overlay_menu.actions()][:2] == ["Coordinate tripod", "Attributes"]
    page.add_overlay("Coordinate tripod")
    form = page._overlay_form
    assert isinstance(form, ParameterForm) and isinstance(form.editors["axes"][0], QtWidgets.QComboBox)
    form.editors["axes"][0].setCurrentText("abc")
    page.overlay_form_holder.itemAt(1).widget().click()          # Apply overlay
    assert page.overlays == [{"name": "Coordinate tripod",
                              "params": {"axes": "abc", "corner": "bottom-right", "size": 0.09}, "enabled": True}]
    assert page.render_kwargs()["overlays"] == page.overlays and "Coordinate tripod" in page.code.toPlainText()
    page.overlay_list.item(0).setCheckState(QtCore.Qt.Unchecked)
    assert page.overlays[0]["enabled"] is False
    page.overlay_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.set_renderer("vector")
    page.preset.setCurrentText("Custom (px)")
    page.width.setValue(200)
    page.height.setValue(150)
    page.output.setText(str(tmp_path / "fig.svg"))
    w.render_from_page()
    svg = (tmp_path / "fig.svg").read_text()
    assert all(f"<!-- {c} -->" in svg for c in "abc")             # tripod labels drawn as text
    path = w.save_session(str(tmp_path / "s.ewald"))
    w2 = MainWindow()
    w2.load_session(path)
    assert w2.render_page.overlays == page.overlays and w2.render_page.overlay_list.count() == 1
    assert w2.style.theme == "Dark slide" and w2.style.cell_color == "#AEB4BE"
    page.remove_overlay(0)
    assert page.overlays == [] and page.overlay_list.count() == 0
    w.close()
    w2.close()


def test_colormap_combo_lists_plugins(app, monkeypatch):
    from ewald import plugins
    from ewald.gui.params import ParameterForm

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    plugins.register_colormap("two", ["navy", "gold"])
    form = ParameterForm(mods.ColorCoding("Position.Z", cmap="two"))
    combo = form.editors["cmap"][0]
    items = [combo.itemText(k) for k in range(combo.count())]
    assert items[0] == "viridis" and items[-1] == "two" and combo.currentText() == "two"
    assert not combo.itemIcon(0).isNull()
    assert "grayscale" in combo.itemData(0, QtCore.Qt.ToolTipRole)


def test_dos_and_band_plots_show_selected_atoms(app):
    from pathlib import Path

    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    data = import_file(Path(__file__).parent / "data" / "vasp" / "electronic" / "vasprun.xml").compute(0)
    es = data.attributes["Electronic Structure"]
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.set_data(data)
    assert panel.table.currentText() == "DOS" and panel.link is None
    ax = panel.figure.axes[0]
    assert ax.get_xlabel() == "E − E_F (eV)" and [t.get_text() for t in ax.get_legend().get_texts()] == \
        ["Total", "Ga", "As"]
    assert "Select atoms" in panel.link_label.text() and not panel._overlays
    panel.set_highlight([1])                                    # As selected in the viewport
    assert panel.link_label.text() == "PDOS of 1 selected atom"
    up = next(a for a in panel._overlays if hasattr(a, "get_ydata"))
    np.testing.assert_allclose(up.get_ydata(), es.pdos_of([1])[0])
    panel.column.setCurrentText("As p (up)")                    # one orbital: the selection follows it
    assert panel.link_label.text() == "p PDOS of 1 selected atom"
    up = next(a for a in panel._overlays if hasattr(a, "get_ydata"))
    np.testing.assert_allclose(up.get_ydata(), es.pdos_of([1], "p")[0])
    panel.table.setCurrentText("Bands")
    ax = panel.figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Γ", "X", "W|K", "Γ"]
    assert panel.link_label.text().startswith("Fat bands: weight of 1 selected atom")
    sizes = np.concatenate([a.get_sizes() for a in panel._overlays])
    assert sizes.max() <= 60 and len(sizes) > 100
    panel.set_highlight([])
    assert not panel._overlays and "Select atoms" in panel.link_label.text()


def test_open_example_menu(app):
    from ewald import examples
    from ewald.gui.app import MainWindow

    w = MainWindow()
    menus = {a.text().replace("&", ""): a.menu() for a in w.menuBar().actions()}
    sub = {a.text().replace("&", ""): a.menu() for a in menus["File"].actions() if a.menu()}["Open example"]
    acts = sub.actions()
    assert len(acts) == len(examples.EXAMPLES) and acts[0].toolTip() == examples.EXAMPLES[0].description
    acts[0].trigger()
    assert w.pipeline is not None and w.data.count == 40
    assert "POSCAR.SrTiO3" in w.windowTitle()
    w.close()
    w.deleteLater()


@pytest.fixture
def cu_path(tmp_path):
    from ase.build import bulk
    from ase.io import write

    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    del atoms[[0]]
    path = tmp_path / "cu.extxyz"
    write(path, atoms)
    return str(path)


def test_notebook_layout_tabs_steps_views_and_files(app, cu_path, tmp_path, monkeypatch):
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook

    w = MainWindow(settings=settings.Settings())
    menus = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert "Notebook" in menus and "Insert" in menus and "Pipeline" not in menus
    tabs = w.panel.tabs
    assert tabs.count() == 1 and tabs.tabText(0) == "Notebook 1"
    w.open_any(cu_path, ask_types=False)                      # the empty notebook makes way for the file
    assert tabs.count() == 1 and tabs.tabText(0) == "cu" and w.pipeline is w.panel.pipeline
    w.panel.add_modifier(mods.CoordinationAnalysis(cutoff=3.0))
    w.recompute()
    view = w.panel.view
    step = view.notebook.cells[-1]
    assert step.is_step and w.pipeline.modifiers == [step.modifier]
    assert "Coordination" in view.widget_for(step).chips.text()

    cell = view.add_view(table="coordination-rdf")             # a view cell: a live, linked plot
    w.recompute()
    plot = view.widget_for(cell).plot_panel
    assert plot in w.plot_panels() and plot.table.currentText() == "coordination-rdf"
    plot.hist.setCurrentText("Coordination")                   # switching the plot rewrites the cell
    assert cell.source.endswith("show(pipe, histogram='Coordination')")
    picked = []
    w.viewport.set_picks = picked.append                       # dragging on the plot selects atoms
    plot.atomsSelected.emit([1, 2])
    assert picked == [[1, 2]]

    path = tmp_path / "cu_nb.py"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    assert w.save_notebook() == str(path) and tabs.tabText(0) == "cu_nb"
    again = Notebook.load(path)
    assert [c.is_step for c in again.cells].count(True) == 1 and again.cells[-1].view == {
        "source": "pipe", "histogram": "Coordination"}

    w.panel.new_notebook()
    assert tabs.count() == 2 and tabs.currentIndex() == 1
    w.panel.close_tab(1)
    w.settings["python.run_notebooks"] = "always"
    w.open_notebook(str(path))                                 # already open: back to its tab
    assert tabs.count() == 1
    w.panel.close_tab(0)                                       # the last tab leaves an empty notebook
    assert tabs.count() == 1 and w.panel.notebook.pipeline is None
    w.open_notebook(str(path))
    assert tabs.tabText(0) == "cu_nb" and [m.name for m in w.panel.pipeline.modifiers] == ["CoordinationAnalysis"]
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_ewd_notebook_saves_state_and_opens_restricted(app, cu_path, tmp_path, monkeypatch):
    from ewald import settings
    from ewald.gui.app import MainWindow

    w = MainWindow(settings=settings.Settings())
    w.open_any(cu_path, ask_types=False)
    w.panel.add_modifier(mods.CoordinationAnalysis(cutoff=3.0))
    w.panel.view.add_view(histogram="Coordination")
    w.panel.view.run_all()
    w.style.radius_scale = 0.8
    path = tmp_path / "cu.ewd"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "cu"),
                                                                                   "EWALD notebook with results and data (*.ewd)"))
    assert w.save_notebook() == str(path)                     # .ewd is the default format
    w.close()

    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)
    w2 = MainWindow(settings=settings.Settings())
    view = w2.open_notebook(str(path))                        # asks, the answer is No: restricted
    assert asked and view.notebook.restricted and w2.style.radius_scale == 0.8
    hist = next(wd for wd in view.widgets if wd.cell.is_view)
    assert any(o.kind == "image" for o in hist.outputs.outputs)   # the saved figure, nothing ran
    view.run_all()                                            # asks again; No keeps it restricted
    assert view.notebook.restricted and len(asked) == 2
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    view.run_all()
    assert not view.notebook.restricted and [m.name for m in w2.pipeline.modifiers] == ["CoordinationAnalysis"]
    w2.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_open_turns_bonds_off_above_limit(app):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    s["bonds.auto_limit"] = 50
    w = MainWindow(settings=s)
    assert w.style.preset == "Ball & stick" and w.style.bonds_enabled
    w.open_file(str(examples.path("Pt_nanoparticle.xyz")))  # more than 50 atoms
    assert w.data.count > 50 and not w.style.bonds_enabled
    w.open_file(str(examples.path("POSCAR.SrTiO3")))  # small again: bonds come back, with a fresh table
    assert w.data.count <= 50 and w.style.bonds_enabled
    assert {tuple(sorted(r[:2])) for r in w.style.bond_pairs} == {("O", "O"), ("O", "Sr"), ("O", "Ti")}
    w.viewport.style_combo.setCurrentText("Simple (ASE)")
    assert w.style.atom_style == "flat" and "bonds" not in w.viewport.plotter.actors
    w.close()
    w.deleteLater()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_toolbar_button_styles(app):
    from PySide6 import QtCore
    from ewald.gui.app import MainWindow

    w = MainWindow()
    analyze, orbit = w.ws_buttons[0], w.viewport.tool_buttons["orbit"]
    assert w.settings["general.toolbar_style"] == "Icons and text"
    assert analyze.toolButtonStyle() == QtCore.Qt.ToolButtonTextBesideIcon and not analyze.icon().isNull()
    assert orbit.toolButtonStyle() == QtCore.Qt.ToolButtonIconOnly and orbit.text() == "Orbit"
    w.set_button_style("Text")
    assert analyze.icon().isNull() and orbit.toolButtonStyle() == QtCore.Qt.ToolButtonTextOnly
    assert app.testAttribute(QtCore.Qt.AA_DontShowIconsInMenus)
    w.set_button_style("Icons")
    assert analyze.toolButtonStyle() == QtCore.Qt.ToolButtonIconOnly
    assert not app.testAttribute(QtCore.Qt.AA_DontShowIconsInMenus)
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_viewport_renders_on_paint_and_falls_back(app, cu_fcc):
    from ewald.gui.app import MainWindow

    w = MainWindow(Pipeline(StaticSource([cu_fcc])))
    w.show()
    app.processEvents()
    vp = w.viewport
    assert vp._paint_driven and not vp.plotter.render_timer.isActive()  # no idle re-rendering
    renders = []
    tag = vp.plotter.render_window.AddObserver("StartEvent", lambda *_: renders.append(1))
    vp._paint_driven = False  # a pyvistaqt without the paint flag: render directly
    vp.render()
    vp.zoom_fit()
    assert len(renders) >= 2
    vp.plotter.render_window.RemoveObserver(tag)
    w.close()


def test_viewport_hover_shows_atom_type_index_and_coordinates(app):
    from ewald.data import Cell, DataCollection
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    data = DataCollection([[2.0, 2.0, 2.0], [6.0, 6.0, 6.0], [6.0, 6.0, 2.0]], ["O", "Ti", "Sr"],
                          Cell(np.diag([8.0, 8.0, 8.0])))
    v = Viewport(ViewStyle())
    v.set_data(data)
    v.view_along("c")
    cam, w, h = v.plotter.camera, 400, 300

    def screen(p):  # world -> window pixel (y from the top), the inverse of camera_ray
        m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
        m = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        c = m @ [*p, 1]
        return (c[0] / c[3] + 1) * w / 2, (1 - c[1] / c[3]) * h / 2

    # viewed along c, Ti (z=3) sits in front of Sr (z=1) at the same x, y: the front one wins
    assert v.hover_index(*screen(data.positions[1]), w, h) in (1, 2)
    front = 1 if np.dot(data.positions[1] - data.positions[2], np.asarray(cam.position) - cam.focal_point) > 0 else 2
    assert v.hover_index(*screen(data.positions[1]), w, h) == front
    assert v.hover_index(*screen(data.positions[0]), w, h) == 0
    assert v.hover_index(*screen([2.0, 6.0, 4.0]), w, h) is None  # empty space between atoms

    seen = []
    v.hovered.connect(seen.append)
    v.hover_move(*screen(data.positions[0]), w, h)
    v._hover_fire()  # what the rest timer does
    assert seen[-1].startswith("O  index 0") and "2.0000  2.0000  2.0000 Å" in seen[-1]
    assert "frac 0.2500  0.2500  0.2500" in seen[-1]
    v._on_drag_start(None, "LeftButtonPressEvent")  # rotating clears it and stops hover
    assert seen[-1] == ""
    v.hover_move(*screen(data.positions[0]), w, h)
    assert not v._hover_timer.isActive()
    v._on_drag_end(None, "LeftButtonReleaseEvent")
    v.hover_info = False  # Preferences › Viewports › Hover
    v.hover_move(*screen(data.positions[0]), w, h)
    assert not v._hover_timer.isActive()
    v.close()


def test_main_window_legacy_gl_viewport(app, pipe, tmp_path, monkeypatch):
    """The OpenGL 1.x viewport (for ssh -X) draws, picks, measures and keeps the camera in sessions."""
    import time

    from ewald.gui.app import MainWindow
    from ewald.gui.legacy_viewport import LegacyViewport
    from ewald.render import camera_state

    monkeypatch.setenv("EWALD_VIEWPORT", "legacy-gl")
    w = MainWindow(pipe)
    vp = w.viewport
    assert isinstance(vp, LegacyViewport)
    w.resize(900, 700)
    w.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    assert vp.opengl_info()["backend"] == "legacy-gl" and vp.opengl_info()["version"]
    img = vp.view.grabFramebuffer()
    W, H = vp.view.width(), vp.view.height()
    bg = img.pixelColor(2, 2)
    assert img.pixelColor(W // 2, H // 2) != bg  # atoms drawn in the middle
    assert "Frame 0" in vp.caption.text()
    vp.set_tool("distance")
    vp.click(W / 2, H / 2, W, H)
    vp.click(W / 2 + 25, H / 2, W, H)
    assert len(vp.picks) == 2 and "Å" in w.pick_label.text()
    vp.click(2, 2, W, H)  # empty background: nothing picked
    assert len(vp.picks) == 2
    vp.hover_info = True
    assert vp.hover_index(W / 2, H / 2, W, H) is not None  # the hover readout shares the camera
    assert vp.hover_index(2, 2, W, H) is None
    vp.view_along("c")
    vp.rotate("azimuth", 15)
    w.set_frame(1)
    app.processEvents()
    cam = camera_state(vp.plotter)
    path = w.save_session(str(tmp_path / "s.ewald"))
    w2 = MainWindow()
    w2.load_session(path)
    np.testing.assert_allclose(camera_state(w2.viewport.plotter)["position"], cam["position"], rtol=1e-6)
    w.close()
    w2.close()


def test_plugins_side_bar_installs_adds_a_step_and_removes(app, cu_path, tmp_path, monkeypatch):
    from ewald import settings
    from ewald.gui import plugins_panel
    from ewald.gui.app import MainWindow

    src = tmp_path / "bar_shift.py"
    src.write_text('"""Shift along z."""\nfrom ewald.pipeline import Modifier\nfrom ewald.plugins import modifier\n\n\n'
                   '@modifier(category="Test")\nclass BarShift(Modifier):\n    def __init__(self, dz: float = 1.0):\n'
                   '        self.dz = dz\n\n    def modify(self, data, ctx):\n        return None\n')
    index = tmp_path / "index.json"
    index.write_text('{"plugins": [{"name": "Bar shift", "kinds": ["modifier"], "source": "bar_shift.py"}]}')
    s = settings.Settings()
    s["python.plugin_catalogs"] = str(index)
    monkeypatch.setattr(plugins_panel.QtWidgets.QMessageBox, "question",
                        lambda *a, **k: plugins_panel.QtWidgets.QMessageBox.Yes)
    w = MainWindow(settings=s)
    assert not w.plugins_dock.isVisible() and "Plugins" in w.activity_buttons
    panel = w.plugins_panel
    panel.wait()
    entry = next(e for e in panel.entries if e.name == "Bar shift")
    panel.install(entry)
    item = next(i for i in panel.items if i.path.endswith("bar_shift.py"))
    panel.show_details(item)
    assert not panel.use_button.isHidden()
    w.open_any(cu_path, ask_types=False)
    panel._use()
    assert type(w.panel.view.notebook.cells[-1].modifier).__name__ == "BarShift"
    panel.remove(item)
    assert not any(i.path.endswith("bar_shift.py") for i in panel.items)
    panel.search.setText("@installed bar")
    assert panel.list_layout.count() > 0
    w.set_minimal(True)
    assert w.menuBar().isHidden() and w.minimal
    w.set_minimal(False)
    w.close()


def test_plugins_side_bar_offers_updates_and_greys_out_incompatible_plugins(app, tmp_path):
    import json

    from ewald import plugin_catalog as pc
    from ewald import settings
    from ewald.gui import plugins_panel

    def plugin(folder, version):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "up_shift.py").write_text(
            f"from ewald.pipeline import Modifier\nfrom ewald.plugins import modifier\n__ewald__ = {{'version': '{version}'}}\n\n"
            "@modifier(category='Test')\nclass UpShift(Modifier):\n    def modify(self, data, ctx):\n        return None\n")
        return str(tmp_path / folder / "up_shift.py")

    pc.install_file(plugin("old", "0.1"))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"format": 1, "plugins": [
        {"id": "up_shift", "name": "Up shift", "version": "0.3", "source": plugin("new", "0.3"), "kinds": ["modifier"]},
        {"id": "future", "name": "Future plugin", "version": "1.0", "requires_ewald": ">=99",
         "source": "future.py", "kinds": ["reader"]}]}))
    s = settings.Settings()
    s["python.plugin_catalogs"] = f"{index}; {tmp_path / 'not-yet.json'}"
    panel = plugins_panel.PluginsPanel(settings=s)
    panel.wait()
    rows = {r.item.name: r for r in panel.findChildren(plugins_panel.PluginRow)}
    buttons = {b.text(): b for r in rows.values() for b in r.findChildren(plugins_panel.QtWidgets.QPushButton)}
    assert "Up shift" not in rows and "Update to 0.3" in buttons          # matched by id, not listed twice
    assert not buttons["Install"].isEnabled() and rows["Future plugin"].property("incompatible")
    texts = [w.text() for w in panel.findChildren(plugins_panel.QtWidgets.QLabel)]
    assert any("needs ewald >=99" in t for t in texts)
    assert any(t.startswith("No plugin index at") and "not-yet.json" in t for t in texts)   # muted, not an error
    buttons["Update to 0.3"].click()
    assert next(i for i in panel.items if i.id == "up_shift").version == "0.3"
    app.sendPostedEvents(None, plugins_panel.QtCore.QEvent.DeferredDelete)    # the old rows go
    assert not any(b.text().startswith("Update") for b in panel.findChildren(plugins_panel.QtWidgets.QPushButton))
    panel.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_follows_a_growing_dump(app, lammps_dir, tmp_path):
    from ewald import import_file
    from ewald.gui.app import MainWindow

    blob = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    offsets = import_file(lammps_dir / "dump.custom.lammpstrj").source.readers[0]._offsets
    path = tmp_path / "dump.lammpstrj"
    path.write_bytes(blob[:offsets[2]])
    w = MainWindow(import_file(path))
    assert w.live_reload.timer.isActive() and w.live_reload_action.isChecked()
    assert w.timeline.slider.maximum() == 1
    w.live_reload.timer.stop()  # only the explicit polls below look at the file (a timer look
    # racing the writes made this test fail under load)
    w.timeline._set(1)  # on the last frame: follow the newest one
    w.timeline.set_selection([0, 1])
    path.write_bytes(blob)
    w.live_reload.poll(wait=True)
    n = len(offsets)
    assert w.timeline.slider.maximum() == n - 1 and w.frame == n - 1
    assert w.timeline.selection == [0, 1]  # kept while the file grows
    assert f"frame {n - 1}/{n - 1}" in w.status.text()
    w.timeline._set(0)  # looking at an early frame: stay there
    path.write_bytes(blob + blob[offsets[-1]:])
    w.live_reload.poll(wait=True)
    assert w.timeline.slider.maximum() == n and w.frame == 0
    w.live_reload.timer.start()
    w.live_reload_action.trigger()  # File › Live reload off
    assert not w.live_reload.timer.isActive()
    w.close()


def test_cell_panel_boundary_range(app):
    from ewald.gui.pipeline_panel import PipelinePanel
    from ewald.gui.style import ViewStyle

    style = ViewStyle()
    panel = PipelinePanel(style)
    seen = []
    panel.styleChanged.connect(lambda: seen.append(1))
    box = panel._boundary_editor()
    assert not box.isChecked() and "boundary" not in style.scene_kwargs()
    box.setChecked(True)
    assert style.boundary == [0.0, 1.0] and seen
    lo, hi = box.findChildren(QtWidgets.QDoubleSpinBox)
    lo.setValue(-0.1)
    hi.setValue(1.1)
    box.findChild(QtWidgets.QComboBox).setCurrentText("complete")
    kw = style.scene_kwargs()
    assert kw["boundary"] == pytest.approx([-0.1, 1.1]) and kw["boundary_bonds"] == "complete"
    box.setChecked(False)
    assert "boundary" not in style.scene_kwargs()


def test_viewport_boundary_images_hover_as_real_atom(app):
    from ewald.data import Cell, DataCollection
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    data = DataCollection([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], ["Ti", "Sr"], Cell(np.diag([4.0, 4.0, 4.0])))
    style = ViewStyle(boundary=[0.0, 1.0], bonds_enabled=False)
    v = Viewport(style)
    v.set_data(data)
    v.view_along("c")
    assert v._boundary_atoms[0].shape == (9, 3)  # Ti on every corner, Sr in the middle
    cam, w, h = v.plotter.camera, 400, 300

    def screen(p):
        m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
        m = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        c = m @ [*p, 1]
        return (c[0] / c[3] + 1) * w / 2, (1 - c[1] / c[3]) * h / 2

    assert v.hover_index(*screen([4.0, 4.0, 4.0]), w, h) == 0  # the corner image is Ti 0
    assert v.hover_index(*screen([2.0, 2.0, 2.0]), w, h) == 1
