import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
# the main-window test needs a real X server (OpenGL widgets); everything else runs offscreen
os.environ.setdefault("QT_QPA_PLATFORM", "xcb" if os.environ.get("DISPLAY") else "offscreen")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from ewald import Pipeline, StaticSource  # noqa: E402
from ewald import modifiers as mods  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ewald.gui import compute, notebook_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    notebook_panel.CONFIRM_UNSAVED = False  # no modal "unsaved changes" questions in tests
    compute.BUDGET = None  # every frame is shown before recompute() returns; background tests set it
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
    assert tl.selection == [2, 5, 7] and tl.sel.text() == "3 selected"
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
    w.workbench.show(w.plots)
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
    w.workbench.show(w.plots)
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
    assert w.workbench.is_shown(w.plots) and w.plots.table.currentText() == "NEB"
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
    assert "View along" in menu and "Ask AI about this structure…" in menu
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


def test_preferences_ai_page_saves_key_outside_settings(app, monkeypatch):
    from ewald import settings
    from ewald.gui.preferences import PreferencesDialog
    from ewald.llm import credentials

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_keyring", lambda: None)   # file fallback, like a headless box
    checked = []
    monkeypatch.setattr(credentials, "check",
                        lambda key, model, **kw: checked.append((key, model, kw["provider"].name)) or (True, "Connected"))
    s = settings.load()
    dlg = PreferencesDialog(s)
    assert "Not set" in dlg.key_status.text()
    assert dlg.key_edit.echoMode() == dlg.key_edit.EchoMode.Password
    dlg.key_edit.setText("sk-ant-gui-key-0123456789")
    dlg.set_value("llm.model", "claude-sonnet-5-5")
    dlg.test_api_key(wait=True)
    assert checked == [("sk-ant-gui-key-0123456789", "claude-sonnet-5-5", "anthropic")]
    assert dlg.key_result.text() == "Connected"
    assert credentials.get_api_key() == (None, None)          # nothing saved before Apply
    dlg.apply()
    assert credentials.get_api_key() == ("sk-ant-gui-key-0123456789", "file")
    assert dlg.key_edit.text() == "" and "sk-ant-…6789" in dlg.key_status.text()
    assert "sk-ant" not in s.path.read_text() and settings.load()["llm.model"] == "claude-sonnet-5-5"
    dlg._search("api key")
    visible = {dlg.pages.item(i).text() for i in range(dlg.pages.count()) if not dlg.pages.item(i).isHidden()}
    assert visible == {"AI"}
    dlg._search("")
    dlg.remove_api_key()
    assert credentials.get_api_key()[0] is not None            # removal waits for Apply too
    dlg.apply()
    assert credentials.get_api_key() == (None, None)
    # another provider: its own key field, model suggestions and server address
    model = dlg.editors["llm.model"][0]
    assert "Anthropic" in dlg.key_label.text() and model.itemText(0) == "claude-opus-5-5"
    dlg.set_value("llm.provider", "openai-compatible")
    assert dlg.work["llm.provider"] == "openai-compatible" and dlg.work["llm.model"] == ""
    assert "OpenAI-compatible" in dlg.key_label.text() and "optional" in dlg.key_label.text()
    assert model.itemText(0) == "llama3.1" and "llama3.1" in model.lineEdit().placeholderText()
    assert "11434" in dlg.editors["llm.base_url"][0].placeholderText()
    dlg.set_value("llm.model", "qwen3:8b")                     # any id the server knows
    dlg.apply()
    saved = settings.load()
    assert saved["llm.provider"] == "openai-compatible" and saved["llm.model"] == "qwen3:8b"
    dlg.close()


def test_ask_panel_agent_uses_ai_preferences(app, monkeypatch):
    from ewald import settings
    from ewald.gui import ask_panel

    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw))
    s = settings.load()
    s["llm.model"], s["llm.effort"] = "claude-haiku-5-5", "low"
    ask_panel.agent_from_settings(lambda: s)(None, None)
    assert made == [{"provider": "anthropic", "model": "claude-haiku-5-5", "base_url": None, "effort": "low"}]
    s["llm.provider"], s["llm.model"], s["llm.base_url"] = "openai-compatible", "", "http://localhost:1234/v1"
    factory = ask_panel.agent_from_settings(lambda: s)
    factory(None, None)
    assert made[-1] == {"provider": "openai-compatible", "model": None, "base_url": "http://localhost:1234/v1",
                        "effort": "low"}
    assert factory.assistant() == "AI"
    s["llm.provider"] = "gemini"
    assert factory.assistant() == "Gemini"


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
    s["viewport.follow_theme"] = False
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
    assert "Writing plugins" in labels and "Open plugin folder" not in labels  # in Insert › Plugin step
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
    text = panel.text()
    assert "What are these?" in text and "2 atoms: **fcc** neighbors" in text
    user, claude = panel.messages
    assert user.role == "user" and claude.role == "assistant"
    assert [claude.steps.itemAt(i).widget().text() for i in range(claude.steps.count())] == ["✓ Used describe_selection"]
    assert changed == [1] and panel.ask_button.isEnabled() and panel.input.toPlainText() == ""

    ctx["indices"] = []
    panel.ask("And the frame?")
    _wait(app, panel)
    assert agents[0].calls[-1] == ("frame", "And the frame?") and len(agents) == 1   # same conversation
    panel.reset()
    assert panel.text() == "" and not panel.welcome.isHidden()


def test_ask_panel_without_claude(app, pipe):
    from ewald.gui.ask_panel import AskPanel

    def missing(p, ev):
        raise ModuleNotFoundError("No module named 'anthropic'")

    panel = AskPanel(lambda: (pipe, 0, []), agent_factory=missing)
    panel.ask("hello")
    assert "not available" in panel.text() and "pip install ewald[llm]" in panel.text()
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
    assert shown(w.display) and not shown(w.ask_dock)    # the AI chat side bar starts hidden
    w.ask_claude()
    app.processEvents()
    assert shown(w.ask_dock) and shown(w.display)         # left of the notebook, not over Display
    assert w.dockWidgetArea(w.ask_dock) == QtCore.Qt.LeftDockWidgetArea
    app.processEvents()
    a, n = w.ask_dock.geometry(), w.notebook_dock.geometry()
    assert a.center().x() < n.center().x() and a.top() == n.top()   # left of it, not stacked
    w.close()


def test_ai_chat_side_bar_sends_notebook_and_inserts_cells(app, pipe):
    from ewald.gui import ask_panel
    from ewald.gui.app import MainWindow

    agents = []

    class NotebookAgent(_FakeAgent):
        def ask(self, question):
            self.calls.append(("frame", question))
            return "Try this:\n\n```python\npipe.add(CoordinationAnalysis(cutoff=3.0))\n```\n\nand ```x``` inline."

    w = MainWindow(pipe)
    w.ask_panel.agent_factory = lambda p, ev: agents.append(NotebookAgent(p, ev)) or agents[-1]
    act = w.activity_buttons["AI chat"]
    assert w.ask_dock.isHidden()
    act.trigger()                                          # the AI button in the activity bar
    assert not w.ask_dock.isHidden()
    w.ask_panel.ask("Add coordination")
    _wait(app, w.ask_panel)
    first = agents[0].calls[0][1]
    assert first.startswith("The notebook now reads:") and "pipe" in first and first.endswith("Add coordination")
    assert w.ask_panel.snippets == ["pipe.add(CoordinationAnalysis(cutoff=3.0))\n"]
    answer = w.ask_panel.messages[1]
    assert [b.insert_button is not None for b in answer.code_blocks] == [True]
    w.ask_panel.ask("Again")                               # unchanged notebook: not sent twice
    _wait(app, w.ask_panel)
    assert agents[0].calls[1][1] == "Again"

    view = w.panel.view
    n = len(view.notebook.cells)
    answer.code_blocks[0].insert_button.click()            # + Insert as cell under the code block
    assert len(view.notebook.cells) == n + 1
    assert any(c.source.strip() == "pipe.add(CoordinationAnalysis(cutoff=3.0))" for c in view.notebook.cells)
    assert not w.notebook_dock.isHidden()
    w.ask_panel.ask("And now?")                            # the notebook changed: sent again
    _wait(app, w.ask_panel)
    assert agents[0].calls[2][1].startswith("The notebook now reads:") and "cutoff=3.0" in agents[0].calls[2][1]
    assert ask_panel.code_blocks("a\n```py\nx = 1\n```\n```bash\nls\n```") == ["x = 1\n"]
    assert ask_panel.split_answer("a\n```bash\nls\n```\nb") == [("text", "", "a"), ("code", "bash", "ls\n"),
                                                                ("text", "", "b")]
    w.ask_panel.reset()
    w.ask_panel.starters[0].click()                        # a starter sends its question
    _wait(app, w.ask_panel)
    assert w.ask_panel.messages[0].text == ask_panel.STARTERS[0]
    w.close()


def test_ai_chat_tells_claude_about_the_notebook(app, monkeypatch):
    from ewald import settings
    from ewald.gui import ask_panel

    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw))
    s = settings.Settings()
    ask_panel.agent_from_settings(lambda: s, notebook=True)(None, None)
    ask_panel.agent_from_settings(lambda: s)(None, None)
    assert "Insert as notebook cell" in made[0]["system"] and "system" not in made[1]


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


def _mouse(panel, kind, x, y, button=1, key=None, step=0, dblclick=False):
    """Send a matplotlib mouse event at data point (x, y) of the panel's plot."""
    from matplotlib.backend_bases import MouseEvent

    px, py = panel._ax.transData.transform((x, y))
    ev = MouseEvent(kind, panel.canvas, px, py, button=button, key=key, step=step, dblclick=dblclick)
    panel.canvas.callbacks.process(kind, ev)


def test_plots_zoom_pan_reset_and_hover(app, lammps_dir):
    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(lammps_dir / "thermo" / "dump.lammpstrj")
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.pipeline = pipe
    views = []
    panel.viewChanged.connect(lambda x, y: views.append((x, y)))
    panel.set_data(pipe.compute(0))
    panel.canvas.draw()
    ax = panel._ax
    x0, x1 = full = ax.get_xlim()
    y = sum(ax.get_ylim()) / 2
    _mouse(panel, "scroll_event", 20, y, button="up", step=1)           # wheel up: zoom in around x = 20
    lo, hi = ax.get_xlim()
    assert hi - lo == pytest.approx((x1 - x0) / 1.25) and lo < 20 < hi
    assert views[-1][0] == (lo, hi) and views[-1][1] is None
    panel.set_data(pipe.compute(2))                                     # the zoom stays while frames change
    assert panel._ax.get_xlim() == (lo, hi)
    panel._drawn = None
    panel.set_data(pipe.compute(3))                                     # also through a full redraw
    assert panel._ax.get_xlim() == (lo, hi)
    ax = panel._ax
    panel.canvas.draw()
    _mouse(panel, "button_press_event", 20, y, button=3)                # right drag pans
    _mouse(panel, "motion_notify_event", 25, y, button=3)
    _mouse(panel, "button_release_event", 25, y, button=3)
    assert ax.get_xlim() == pytest.approx((lo - 5, hi - 5), abs=0.1)  # whole pixels
    assert panel._range is None and panel.frame_selection == []         # a pan selects nothing
    _mouse(panel, "motion_notify_event", 21, y, button=None)            # hover: values and the frame
    assert panel.readout.text().endswith("frame 2: 273.4")
    _mouse(panel, "button_press_event", 21, y, dblclick=True)           # double-click: full range
    _mouse(panel, "button_release_event", 21, y)
    assert panel._ax.get_xlim() == pytest.approx(full) and views[-1] == (None, None)
    panel.set_view(ylim=(0, 1), emit=False)
    assert panel.view_limits() == (None, (0.0, 1.0))
    panel.table.setCurrentText("")                                      # another plot: full range again
    panel.hist.setCurrentText(next(k for k in pipe.compute(0).particles if k != "Position"))
    assert panel.view_limits() == (None, None)


def test_plots_click_on_two_cv_plot_goes_to_nearest_frame(app, cu_fcc):
    from ewald.gui.plots import PlotsPanel

    def cvs(frame, data):
        data.tables["colvar"] = {"Frame": np.arange(3), "d1": np.array([0.1, 0.5, 0.9]),
                                 "d2": np.array([0.2, 0.6, 0.8])}
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        data.tables["fes"] = {"d1": x.ravel(), "d2": y.ravel(), "Free Energy": (x + y).ravel()}

    pipe = Pipeline(StaticSource([cu_fcc] * 3), [mods.PythonModifier(cvs)])
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.pipeline = pipe
    panel.set_data(pipe.compute(0))
    panel.table.setCurrentText("fes")
    assert panel.link.kind == "frames2d"
    panel.canvas.draw()
    jumps = []
    panel.frameRequested.connect(jumps.append)
    for x, y in ((0.52, 0.58), (0.3, 0.95)):                            # near frame 1; far from every frame
        _mouse(panel, "button_press_event", x, y)
        _mouse(panel, "button_release_event", x, y)
    assert jumps == [1]
    _mouse(panel, "motion_notify_event", 0.88, 0.8, button=None)
    assert "frame 2" in panel.readout.text()


def test_dos_and_band_plots_show_selected_atoms(app):
    from pathlib import Path

    from ewald import import_file
    from ewald.gui.plots import E_FERMI, PlotsPanel

    data = import_file(Path(__file__).parent / "data" / "vasp" / "electronic" / "vasprun.xml").compute(0)
    es = data.attributes["Electronic Structure"]
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.set_data(data)
    assert panel.table.currentText() == "DOS" and panel.link is None
    ax = panel.figure.axes[0]
    assert ax.get_xlabel() == E_FERMI and [t.get_text() for t in ax.get_legend().get_texts()] == \
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
    # one energy window for DOS and bands: typed, dragged or zoomed with the wheel
    assert panel.e_max.isVisibleTo(panel)
    panel.e_max.setValue(2.0)
    assert panel._ax.get_ylim()[1] == 2.0 and panel.view_limits()[1][1] == 2.0
    panel.table.setCurrentText("DOS")
    assert panel._ax.get_xlim()[1] == 2.0 and panel.e_max.value() == 2.0
    panel.canvas.draw()
    _mouse(panel, "button_press_event", -1.0, 0.5)                      # left drag on the DOS: zoom to it
    _mouse(panel, "motion_notify_event", 1.0, 0.5)
    _mouse(panel, "button_release_event", 1.0, 0.5)
    assert panel._ax.get_xlim() == pytest.approx((-1.0, 1.0), abs=0.02) and panel.e_min.value() == pytest.approx(-1.0, abs=0.02)
    panel.table.setCurrentText("Bands")
    assert panel._ax.get_ylim() == pytest.approx((-1.0, 1.0), abs=0.02)
    panel.canvas.draw()
    _mouse(panel, "scroll_event", panel._ax.get_xlim()[0] + 0.01, 0.0, button="down", step=-1)  # zoom out E
    assert panel._ax.get_ylim() == pytest.approx((-1.25, 1.25), abs=0.03)
    _mouse(panel, "motion_notify_event", 0.01, 0.5, button=None)
    assert panel.readout.text().startswith("E − E_F = 0.4") or panel.readout.text().startswith("E − E_F = 0.5")
    panel.reset_view()
    assert panel.view_limits() == (None, None) and not panel.figure.axes[0].get_ylim()[1] > 6.0


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
    plugins_button = w.activity_buttons["Plugins"]
    w._plugin_updates(2)                                       # a count badge while plugins have updates
    assert plugins_button.property("updates") == 2 and "2 updates available" in plugins_button.toolTip()
    w._plugin_updates(0)
    assert plugins_button.toolTip() == "Show or hide the Plugins side bar"
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
    plot.set_view(xlim=(10.5, 12.25))                          # zooming writes the range into the cell
    assert cell.source.endswith("show(pipe, histogram='Coordination', xlim=(10.5, 12.25))")
    assert plot._ax.get_xlim() == (10.5, 12.25)
    plot.reset_view()
    assert cell.source.endswith("show(pipe, histogram='Coordination')")

    path = tmp_path / "cu_nb.py"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    assert w.save_notebook() == str(path) and tabs.tabText(0) == "cu_nb"
    again = Notebook.load(path)
    assert [c.is_step for c in again.cells].count(True) == 1 and again.cells[-1].view == {
        "source": "pipe", "histogram": "Coordination"}

    first = w.render_page.quality.currentText()
    w.panel.new_notebook()
    assert tabs.count() == 2 and tabs.currentIndex() == 1
    other = "Draft" if first != "Draft" else "Standard"
    w.render_page.quality.setCurrentText(other)               # render settings belong to the notebook
    tabs.setCurrentIndex(0)
    assert w.render_page.quality.currentText() == first
    tabs.setCurrentIndex(1)
    assert w.render_page.quality.currentText() == other
    w.panel.close_tab(1)
    w.settings["python.run_notebooks"] = "always"
    w.open_notebook(str(path))                                 # already open: back to its tab
    assert tabs.count() == 1
    w.panel.close_tab(0)                                       # the last tab leaves an empty notebook
    assert tabs.count() == 1 and w.panel.notebook.pipeline is None
    w.open_notebook(str(path))
    assert tabs.tabText(0) == "cu_nb" and [m.name for m in w.panel.pipeline.modifiers] == ["CoordinationAnalysis"]
    act = next(a for a in w.analysis_menu.actions() if a.text() == "Common neighbor analysis")
    act.trigger()                                              # Insert › Analysis: a verb cell, run at once
    cell = w.panel.view.current
    assert cell.source.startswith("from ewald import cna\n") and cell.state == "ok"
    assert cell.outputs[-1].kind == "image" and isinstance(cell.outputs[-1].data, bytes)
    assert [m.name for m in w.panel.pipeline.modifiers] == ["CoordinationAnalysis", "CommonNeighborAnalysis"]

    view = w.panel.view
    assert w.snapshot_code().startswith("from ewald.views import snapshot\nsnapshot(pipe, frame=0,\n         camera={")
    w._insert_snapshot()                                       # Insert › 3D snapshot: a picture of the view
    snap = view.current
    assert snap.state == "ok" and snap.outputs[-1].kind == "image"
    from PySide6 import QtGui

    mime = QtCore.QMimeData()                                  # drag the cell by its grip to the top
    mime.setData("application/x-ewald-cell", snap.id.encode())
    drop = QtGui.QDropEvent(QtCore.QPointF(20, -1), QtCore.Qt.MoveAction, mime, QtCore.Qt.LeftButton,
                            QtCore.Qt.NoModifier)
    assert view.eventFilter(view.container, drop)              # Qt delivers synthetic drops only during a drag
    app.processEvents()
    assert view.notebook.cells[0] is snap and view.widgets[0].cell is snap
    view.move_cell(snap, len(view.notebook.cells))             # and back to the end
    assert view.notebook.cells[-1] is snap
    assert [m.name for m in w.panel.pipeline.modifiers] == ["CoordinationAnalysis", "CommonNeighborAnalysis"]
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
    assert not view.banner.isHidden() and "Restricted" in view.banner_text.text()
    view.run_all()                                            # asks again; No keeps it restricted
    view.run(view.notebook.cells[0])                          # so does running one cell
    assert view.notebook.restricted and len(asked) == 3 and view.notebook.cells[0].state != "error"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    w2.trust_action.trigger()                                 # Notebook › Trust notebook…
    assert not view.notebook.restricted and view.banner.isHidden()
    view.run_all()
    assert [m.name for m in w2.pipeline.modifiers] == ["CoordinationAnalysis"]
    w2.close()

    w3 = MainWindow(settings=settings.Settings())             # Yes when opening: trusted, the bar goes away
    view = w3.open_notebook(str(path))
    assert not view.notebook.restricted and view.banner.isHidden()
    assert view.notebook.cells[0].state == "ok" and [m.name for m in w3.pipeline.modifiers] == ["CoordinationAnalysis"]
    w3.close()

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.No)
    w4 = MainWindow(settings=settings.Settings())             # No when opening, then the bar's Trust button
    view = w4.open_notebook(str(path))
    w4.show()
    app.processEvents()
    assert view.notebook.restricted and view.banner.isVisible()
    buttons = {b.text(): b for b in view.banner.findChildren(QtWidgets.QPushButton)}
    buttons["Trust and run all"].click()
    app.processEvents()
    assert not view.notebook.restricted and not view.banner.isVisible()
    assert view.notebook.cells[0].state == "ok" and [m.name for m in w4.pipeline.modifiers] == ["CoordinationAnalysis"]
    w4.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_ewd_offers_to_install_the_pip_packages_it_needs(app, cu_path, tmp_path, monkeypatch):
    from ewald import notebook_io, plugins, settings
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook
    from test_notebook_io import add_package_plugin

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(cu_path)!r})\n")
    notebook_io.save_ewd(nb, tmp_path / "p.ewd", figures=False)
    add_package_plugin(tmp_path / "p.ewd")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.No)
    w = MainWindow(settings=settings.Settings())
    view = w.open_notebook(str(tmp_path / "p.ewd"))
    assert not view.packages_bar.isHidden() and "ewald-cp2k==0.2" in view.packages_text.text()
    ran = []
    monkeypatch.setattr(w.plugins_panel, "_run_pip", lambda action, req: ran.append((action, req)))
    view.packages_button.click()
    assert ran == [("install", ["ewald-cp2k==0.2"])] and not w.plugins_dock.isHidden()
    monkeypatch.setattr(plugins, "modifiers", lambda: {"Cp2kCharges": object()})   # pip installed it
    w.plugins_panel.changed.emit()
    assert view.packages_bar.isHidden()
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_notebook_tab_runs_its_cells_in_another_kernel(app, cu_path, tmp_path, monkeypatch):
    pytest.importorskip("jupyter_client")
    import sys

    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.kernel import Kernel
    from ewald.notebook import Notebook

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\nfrom ewald.modifiers import *\npipe = load({str(cu_path)!r})\n\n"
                          "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n")
    nb.save(tmp_path / "k.ipynb")
    s = settings.Settings()
    s["python.run_notebooks"] = "never"
    w = MainWindow(settings=s)
    view = w.open_notebook(str(tmp_path / "k.ipynb"))
    assert view.kernel_button.text() == "This app"
    view.fill_kernel_menu(view.kernel_menu)
    labels = [a.text() for a in view.kernel_menu.actions()]
    assert labels[0] == "This app's Python" and labels[-1] == "Other Python…"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (sys.executable, ""))
    view.kernel_menu.actions()[-1].trigger()                      # Other Python…
    k = view.notebook.kernel
    try:
        assert k is not None and view.kernel_button.text() == "This Python"
        view.run_all()
        assert view.running and not view.interrupt_button.isHidden()  # on a thread: the window keeps responding
        assert view.wait()
        assert view.interrupt_button.isHidden() and view.run_all_button.isEnabled()
        assert k.alive and [c.state for c in view.notebook.cells] == ["ok", "ok"]
        assert w.pipeline is view.notebook.pipeline and "Coordination" in w.pipeline.compute(0).particles
        assert "1 step" in view.status.text()
        view.notebook.save(tmp_path / "k2.ipynb")
        (tmp_path / "k3.ipynb").write_bytes((tmp_path / "k2.ipynb").read_bytes())
        reopened = w.open_notebook(str(tmp_path / "k3.ipynb"))      # the kernel choice is kept
        assert reopened.notebook.kernel is not None and not reopened.notebook.kernel.alive
        assert reopened.kernel_button.text() == "This Python"
        w.panel.tabs.setCurrentWidget(view)
        view.add_cell("code")
        sleeper = view.notebook.cells[-1]
        sleeper.source = "import time\ntime.sleep(60)"
        view.run(sleeper)
        import time

        time.sleep(1.0)
        view.interrupt()                                           # ■ Interrupt
        assert view.wait(30)
        assert sleeper.state == "error" and "KeyboardInterrupt" in sleeper.outputs[-1].data
    finally:
        import ewald.gui.notebook_panel as npn

        monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
        w.close()
    assert not k.alive                                             # closing the window stops it


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
    snap = vp.snapshot().toImage()  # what a tab drag shows in place of the GL window
    gl = vp._gl_widget().mapTo(vp, vp._gl_widget().rect().center())
    assert snap.pixelColor(gl.x(), gl.y()) != snap.pixelColor(gl.x() - W // 2 + 2, gl.y() - H // 2 + 2)
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


def test_background_compute_keeps_the_window_responsive_and_cancels(app, cu_fcc, monkeypatch):
    import threading
    import time

    from ewald.gui import compute
    from ewald.gui.app import MainWindow

    gate, runs = threading.Event(), []

    class Slow(mods.Modifier):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale

        def modify(self, data, ctx):
            runs.append((ctx.frame, self.scale))
            while not gate.is_set():  # a long analysis that checks for cancellation
                ctx.check_cancel()
                time.sleep(0.005)
            out = data.copy()
            out.attributes["Scale"] = self.scale
            return out

    gate.set()
    slow = Slow()
    w = MainWindow(Pipeline(StaticSource([cu_fcc] * 3), [slow]))
    assert w.data.attributes["Scale"] == 1.0  # opening a file still shows frame 0 right away
    monkeypatch.setattr(compute, "BUDGET", 0.01)
    gate.clear()
    runs.clear()
    w.timeline._set(1)
    assert w.data.attributes["Scale"] == 1.0  # still the previous result
    assert w.frame == 1 and w.runner.busy and "Computing frame 1" in w.status.text()
    slow.scale = 2.0
    w.recompute()  # a parameter change while frame 1 is still running cancels it
    gate.set()
    assert w.runner.wait(5)
    app.processEvents()
    assert not w.runner.busy and w.data.attributes["Scale"] == 2.0
    assert runs[0] == (1, 1.0) and runs[-1] == (1, 2.0)
    assert "frame 1/2" in w.status.text()
    w.settings["performance.background_compute"] = False
    gate.clear()
    threading.Timer(0.1, gate.set).start()
    w.timeline._set(2)  # synchronous when turned off
    assert w.data.attributes["Scale"] == 2.0 and w.frame == 2 and not w.runner.busy
    w.close()


def test_notebook_undo_redo_from_the_edit_menu(app, cu_path):
    from ewald import settings
    from ewald.gui.app import MainWindow

    w = MainWindow(settings=settings.Settings())
    w.open_any(cu_path, ask_types=False)
    view = w.panel.view
    assert not view.history.can_undo  # opening a file is not an edit
    w.panel.add_modifier(mods.CoordinationAnalysis(cutoff=3.0))
    step = view.notebook.cells[-1]
    form = view.widget_for(step)
    for cutoff in (2.0, 2.6):  # what the form does on each change, twice: one entry
        step.modifier.cutoff = cutoff
        view.notebook.update_step(step)
        form.edited.emit(step)
    assert w.data["Coordination"].max() == 12
    w._update_undo_actions()
    assert w.undo_action.text() == "&Undo Edit CoordinationAnalysis" and not w.redo_action.isEnabled()
    w.undo()
    assert w.pipeline.modifiers[0].cutoff == 3.0 and w.data["Coordination"].max() == 12
    w.undo()
    assert w.pipeline.modifiers == [] and "Coordination" not in w.data
    w.undo()
    assert w.statusBar().currentMessage() == "Nothing to undo"
    w.redo()
    w.redo()
    assert w.pipeline.modifiers[0].cutoff == 2.6
    w.pipeline.modifiers.append(mods.ExpressionSelection("Coordination < 12"))  # e.g. from the console
    w.panel.refresh()
    w._update_undo_actions()
    assert w.undo_action.text() == "&Undo Change pipeline"
    w.undo()
    assert [m.name for m in w.pipeline.modifiers] == ["CoordinationAnalysis"]
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
    counts = []
    panel.updatesChanged.connect(counts.append)
    panel.wait()
    assert counts == [1] and [(i.id, e.version) for i, e in panel.updates()] == [("up_shift", "0.3")]
    panel.search.setText("@updates")                                     # only what has an update
    app.sendPostedEvents(None, plugins_panel.QtCore.QEvent.DeferredDelete)
    assert [r.item.id for r in panel.findChildren(plugins_panel.PluginRow)] == ["up_shift"]
    panel.search.clear()
    app.sendPostedEvents(None, plugins_panel.QtCore.QEvent.DeferredDelete)
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
    assert counts == [1, 0]                                              # the activity bar badge goes away
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


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_reports_an_unfinished_calculation(app, vasp_dir, tmp_path):
    import warnings

    from ewald import import_file
    from ewald.gui.app import MainWindow

    xml = (vasp_dir / "vasprun.xml").read_text()
    path = tmp_path / "vasprun.xml"
    path.write_text(xml[: xml.rindex("<calculation>") + 20])  # VASP running step 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = MainWindow(import_file(path))
    w.live_reload.timer.stop()
    log = w.log.toPlainText()
    assert "WARNING Calculation incomplete: vasprun.xml: 1 ionic step(s) read" in log
    assert "DOS and bands not written yet" in w.statusBar().currentMessage()
    path.write_text(xml)
    w.live_reload.poll(wait=True)
    assert w.timeline.slider.maximum() == 1
    assert "Calculation finished" in w.log.toPlainText()
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


def test_panels_and_cells_fold(app, cu_path, tmp_path):
    from ewald.gui.app import MainWindow
    from ewald.gui.collapsible import STRIP, DockTitleBar
    from ewald.notebook import Notebook

    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    bar = w.notebook_dock.titleBarWidget()
    assert isinstance(bar, DockTitleBar)
    bar.fold.click()
    app.processEvents()
    assert bar.collapsed and w.notebook_dock.width() == STRIP and not w.panel.isVisible()
    bar.toggle()
    app.processEvents()
    assert not bar.collapsed and w.notebook_dock.widget() is w.panel and w.panel.isVisible()
    assert w.notebook_dock.width() > 4 * STRIP
    w.workbench.set_panel_collapsed(True)                   # the panel folds to its tab bar
    app.processEvents()
    assert w.workbench.panel_host.maximumHeight() < 60 and not w.console.isVisible()
    w.workbench.set_panel_collapsed(False)
    w.remotes_dock.show()
    w.tabifyDockWidget(w.display.parent(), w.remotes_dock)
    app.processEvents()
    side = w.remotes_dock.titleBarWidget()                  # tabbed with Display: they fold together
    side.set_collapsed(True)
    assert all(d.maximumWidth() == STRIP for d in [w.remotes_dock] + w.tabifiedDockWidgets(w.remotes_dock))
    assert w.tabPosition(QtCore.Qt.RightDockWidgetArea) == QtWidgets.QTabWidget.East   # tabs turn sideways
    side.set_collapsed(False)
    assert w.tabPosition(QtCore.Qt.RightDockWidgetArea) == QtWidgets.QTabWidget.South

    view = w.panel.view
    view.add_cell("code", "print('hi')\nx = 1")
    cw = view.widgets[-1]
    cw.fold_button.click()
    assert cw.cell.collapsed and cw.content.isHidden() and "print('hi')" in cw.folded_label.text()
    nb_path = tmp_path / "folded.ipynb"
    view.notebook.save(str(nb_path))
    assert Notebook.load(str(nb_path)).cells[-1].collapsed               # Jupyter's source_hidden
    cw.set_collapsed(False)
    assert not cw.content.isHidden()
    w.close()


def test_workbench_drag_plot_over_3d_view(app, pipe):
    """Drag the Plots tab onto the 3D view: it opens large there, the 3D view waits behind its
    tab (parked, no scene builds) and comes back where its slot is; splits, extra plot views,
    the keyboard actions and the saved layout."""
    import time

    from PySide6 import QtCore, QtGui, QtTest

    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.gui.preferences import save_layout

    def settle():
        for _ in range(10):
            app.processEvents()
            time.sleep(0.01)

    s = settings.Settings()
    w = MainWindow(pipe, settings=s)
    w.resize(1400, 900)
    w.show()
    settle()
    wb, vp = w.workbench, w.viewport
    assert wb.titles("editor") == ["3D view"] and wb.titles("panel") == ["Python console", "Data inspector", "Plots", "Log"]
    slot = wb.slot("viewport")
    assert vp.parent() is wb and vp.geometry() == QtCore.QRect(slot.mapTo(wb, QtCore.QPoint()), slot.size())

    bar = wb.tab_group(wb.group_of(w.plots)).bar          # a real mouse drag of the Plots tab
    start = bar.tabRect(2).center()
    QtTest.QTest.mousePress(bar, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
    QtTest.QTest.mouseMove(bar, start + QtCore.QPoint(40, 0))
    assert wb._drag is not None and not vp.isVisible()    # a snapshot stands in for the GL widget
    over = bar.mapFromGlobal(slot.mapToGlobal(slot.rect().center()))
    QtTest.QTest.mouseMove(bar, over)
    assert wb._drag["target"][1] == "center" and wb.marks.rect_ is not None
    QtTest.QTest.mouseRelease(bar, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, over)
    settle()
    assert wb.titles("editor") == ["3D view", "Plots"] and wb.is_shown(w.plots) and w.plots.data is not None
    assert vp.parked and not vp.isVisible() and wb.can_close("plot:1")
    built = vp._build_time
    w.set_frame(1)
    settle()
    assert vp._parked_stale and vp.data is w.data           # the frame waits for the 3D view
    wb.show(vp)
    settle()
    assert not vp.parked and not vp._parked_stale and vp.isVisible() and vp._build_time != built

    g = wb.split(w.plots, "right")                         # side by side
    settle()
    assert wb.is_shown(vp) and wb.is_shown(w.plots) and wb.layout.area_of(g) == "editor"
    assert abs(vp.width() - w.plots.width()) < 0.3 * vp.width()
    edge = slot.mapToGlobal(QtCore.QPoint(slot.width() - 3, slot.height() // 2))
    wb.begin_drag("log")
    assert wb.target_at(edge)[1:] == ("right", None)
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_Escape)        # Esc cancels
    assert wb._drag is None and vp.isVisible() and wb.titles("panel")[-1] == "Log"

    p2 = w.new_plot_view("plot:1")                          # a second plot next to the first
    assert wb.group_of(p2) is g and "plot:2" in w.extra_plots and wb.can_close(p2)
    menu = QtWidgets.QMenu()
    wb.fill_view_menu(menu, "plot:2")
    assert [a.text() for a in menu.actions() if a.text()] == [
        "Split right", "Split down", "Move to panel", "Maximize group", "New plot view", "Close"]
    acts = {a.text(): a for a in wb.actions()}
    assert acts["Move view down"].shortcut() == QtGui.QKeySequence("Ctrl+Alt+Down")
    wb.active = g
    acts["Maximize group"].trigger()
    settle()
    assert wb.is_shown(p2) and not wb.is_shown(vp) and vp.parked
    assert not wb.panel_host.isVisible()                   # the panel makes room too
    acts["Maximize group"].trigger()
    assert wb.panel_host.isVisible()
    acts["Move view down"].trigger()                       # from the editor area into the panel
    settle()
    assert wb.layout.area_of(wb.group_of(p2)) == "panel" and not vp.parked

    save_layout(w, s)
    w.close()
    w2 = MainWindow(pipe, settings=s)                       # the layout comes back, plot:2 included
    assert wb.layout.to_json() == w2.workbench.layout.to_json() and "plot:2" in w2.extra_plots
    w2.workbench.close_view("plot:2")
    assert "plot:2" not in w2.extra_plots and "plot:2" not in w2.workbench.specs
    w2.workbench.reset()
    assert w2.workbench.titles("editor") == ["3D view"]
    assert w2.workbench.titles("panel") == ["Python console", "Data inspector", "Plots", "Log"]
    w2.close()


def test_plot_export_is_a_print_figure_in_any_theme(app, vasp_dir, tmp_path):
    from PIL import Image

    from ewald import import_file, plotstyle
    from ewald.gui import theme
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(vasp_dir / "neb")
    theme.use("graphite")
    try:
        panel = PlotsPanel()
        panel.resize(600, 400)
        panel.pipeline = pipe
        panel.set_data(pipe.compute(1))
        screen = panel.figure
        panel.export(tmp_path / "neb.png", "2-column", quality="Draft")
        panel.export(tmp_path / "neb.svg")
        assert panel.figure is screen and theme.TOKENS["bg"] == theme.PALETTES["graphite"]["bg"]
        assert panel.figure.axes[0].get_facecolor()[:3] != (1.0, 1.0, 1.0)   # the canvas keeps the theme
    finally:
        theme.use(theme.DEFAULT)
    img = Image.open(tmp_path / "neb.png").convert("RGB")
    assert img.getpixel((2, 2)) == (255, 255, 255)
    assert abs(img.width / plotstyle.DPI["draft"] - 7.2) < 0.5     # 2-column wide (tight box)
    svg = (tmp_path / "neb.svg").read_text()
    assert "Reaction coordinate" in svg and "#ffffff" in svg.lower()


def test_theme_switch_restyles_console_plots_and_code(app, vasp_dir):
    from PySide6 import QtGui

    from ewald import import_file, settings
    from ewald.gui import prefs_apply, theme
    from ewald.gui.app import MainWindow
    from ewald.gui.console import pygments_style
    from ewald.gui.notebook_panel import PythonHighlighter

    assert theme.DEFAULT == "graphite" and settings.Settings()["general.ui_theme"] == "graphite"
    s = settings.Settings()
    s["general.ui_theme"] = "paper"
    w = MainWindow(settings=s)
    pipe = import_file(vasp_dir / "neb")
    w.plots.pipeline = pipe
    w.plots.set_data(pipe.compute(1))
    doc = QtGui.QTextDocument()
    hl = PythonHighlighter(doc)
    doc.setPlainText("import numpy")
    try:
        for name in theme.PALETTES:
            s["general.ui_theme"] = name
            prefs_apply.apply_settings(w, s)
            t = theme.PALETTES[name]
            con = w.console.widget()
            if hasattr(con, "_highlighter"):
                assert t["bg"] in con.style_sheet and t["accent"] in con.style_sheet   # background, selection
                style = con._highlighter._style
                assert style.background_color == t["bg"] and t["data"] in str(style.styles.values())
            from matplotlib.colors import to_hex

            assert to_hex(w.plots.figure.get_facecolor()).upper() == t["bg"]
            hl.rehighlight()
            formats = doc.firstBlock().layout().formats()
            assert formats[0].format.foreground().color().name().upper() == t["accent"]   # "import" in the accent
            icon = w.activity_buttons["Notebook"].icon()
            assert not icon.isNull()
            assert w.style.background == t["viewport"]               # the 3D view follows the theme
            assert w.style.view_settings().get("background") is None   # ... without writing it into the view
    finally:
        theme.use(theme.DEFAULT)
        w.close()
    assert pygments_style(theme.PALETTES["paper"]).background_color == theme.PALETTES["paper"]["bg"]


def test_notebook_cells_sit_at_the_top(app):
    from ewald.gui.app import MainWindow

    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    for _ in range(3):
        app.processEvents()
    QtCore.QThread.msleep(100)  # let wrapped editors settle their height for the final width
    for _ in range(5):
        app.processEvents()
    view = w.panel.view
    for cell in view.widgets:                               # spare height goes below the cells
        assert cell.height() <= cell.sizeHint().height() + 2
    code = next(c for c in view.widgets if c.cell.kind == "code")
    w.resizeDocks([w.notebook_dock], [300], QtCore.Qt.Horizontal)   # narrow: lines wrap
    for _ in range(3):
        app.processEvents()
    ed = code.editor
    lines = sum(round(ed.blockBoundingRect(b).height() / ed.fontMetrics().lineSpacing())
                for b in (ed.document().findBlockByNumber(i) for i in range(ed.document().blockCount())))
    assert ed.viewport().height() >= lines * ed.fontMetrics().lineSpacing()   # no clipped last line
    w.close()


def test_panel_headers_line_up_and_buttons_use_icons(app, pipe):
    """Dock title bars and tab-group heads share one height and inset, and no button is labelled
    with a symbol glyph (their size and baseline depend on the platform's fallback font)."""
    import unicodedata

    from ewald.gui import header
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.resize(1300, 820)
    w.show()
    app.processEvents()
    bars = [d.titleBarWidget() for d in w.findChildren(QtWidgets.QDockWidget) if d.isVisible()]
    nb_bar = w.notebook_dock.titleBarWidget()
    tabs = w.panel.tabs
    assert tabs.row.parent() is nb_bar and not nb_bar.title.isVisible()  # the tab row is the title bar
    heads = bars + [tabs.tabBar()]
    heads += [g.head for g in w.workbench.findChildren(QtWidgets.QWidget) if hasattr(g, "head") and g.isVisible()]
    assert len(heads) >= 3
    assert {h.height() for h in heads} <= {header.height(), header.height() + 1}
    tops = {h.mapTo(w, QtCore.QPoint(0, 0)).y() for h in heads if h.mapTo(w, QtCore.QPoint(0, 0)).y() < 200}
    assert len(tops) == 1  # notebook, 3D view and Display headers start on one line
    for b in w.findChildren(QtWidgets.QAbstractButton):
        symbols = [c for c in b.text() if unicodedata.category(c) == "So" or 0x2190 <= ord(c) <= 0x2BFF]
        assert not symbols, (b.objectName(), b.toolTip(), b.text())
    bar = tabs.tabBar()
    empty = QtCore.QPointF(bar.width() - 5, bar.height() / 2)  # right of the last tab
    assert bar.tabAt(empty.toPoint()) < 0
    press = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, empty, bar.mapToGlobal(empty),
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    bar.mousePressEvent(press)
    assert not press.isAccepted()  # left to the dock title bar, which starts the panel drag
    w.panel.new_notebook()
    first, second = tabs.widget(0), tabs.widget(1)
    bar.moveTab(1, 0)  # reordering tabs reorders the pages with them
    assert tabs.widget(0) is second and tabs.currentWidget() is tabs.widget(bar.currentIndex())
    tabs.setCurrentWidget(first)
    assert tabs.currentWidget() is first and tabs.indexOf(first) == 1
    nb_bar.fold.click()  # folding leaves the strip with the name
    app.processEvents()
    assert nb_bar.collapsed and nb_bar.vtitle.isVisible() and not tabs.row.isVisible()
    nb_bar.toggle()
    app.processEvents()
    assert not nb_bar.collapsed and tabs.row.isVisible() and not nb_bar.vtitle.isVisible()
    tl = w.timeline
    tl.play.setChecked(True)
    assert tl.play.property("ewald_icon") == "pause"
    tl.play.setChecked(False)
    w.set_minimal(True)
    assert w.minimal_exit.isVisible() and w.minimal_note.isVisible()
    w.menuBar().show()  # what a tap on Alt does; the View menu entry must leave minimal mode
    view_menu = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View")
    next(a for a in view_menu.actions() if a.text().startswith("Minimal mode")).trigger()
    assert not w.minimal and not w.minimal_exit.isVisible() and w.statusBar().isVisible()
    w.close()


def test_viewport_tools_pick_and_measure_with_real_clicks(app, pipe):
    """Pick, Distance and Angle react to real mouse clicks in the VTK view (pyvista hands button
    release observers the interactor style, which has no event position; the picks must not
    depend on it) and show their result in the status bar over any status message."""
    import time

    from PySide6 import QtTest

    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.resize(1100, 800)
    w.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    vp = w.viewport
    view = vp.plotter.interactor
    W, H = view.width(), view.height()
    points = {}
    for y in range(10, H - 10, 6):
        for x in range(10, W - 10, 6):
            i = vp.hover_index(x, y, W, H)
            if i is not None and i not in points:
                points[i] = QtCore.QPoint(x, y)
    atoms = list(points)[:3]
    assert len(atoms) == 3

    def click(i, modifier=QtCore.Qt.NoModifier):
        QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton, modifier, points[i])
        app.processEvents()

    w.message("a status message that would hide the pick result")
    vp.set_tool("pick")
    click(atoms[0])
    assert vp.picks == [atoms[0]] and f"#{atoms[0]}" in w.pick_label.text()
    assert w.statusBar().currentMessage() == ""
    click(atoms[1], QtCore.Qt.ShiftModifier)  # Shift+click adds
    assert vp.picks == atoms[:2]
    vp.set_tool("distance")
    click(atoms[0])
    click(atoms[1])
    assert vp.picks == atoms[:2] and "Å" in w.pick_label.text() and "d(" in w.pick_label.text()
    vp.set_tool("angle")
    for i in atoms:
        click(i)
    assert vp.picks == atoms and "°" in w.pick_label.text()
    w.close()


def test_menu_audit_fixes(app, cu_fcc):
    """Insert › Analysis greys out trajectory analyses for one frame; the palette lists each step
    once (Insert › Step), skips greyed-out ones and has the menus filled on opening; Window ›
    Command palette carries Ctrl+K; one entry each for Ask Claude, Run all and the plugin file."""
    from PySide6 import QtGui

    from ewald.gui.app import MainWindow
    from ewald.gui.palette import menu_entries

    w = MainWindow(Pipeline(StaticSource([cu_fcc])))
    w.analysis_menu.aboutToShow.emit()
    state = {a.data(): a.isEnabled() for a in w.analysis_menu.actions()}
    assert state["msd"] is False and state["vacf"] is False and state["rdf"] is True
    labels = [label for label, path, _ in menu_entries(w.menuBar()) if path.startswith("Insert › Step")]
    assert labels and len(labels) == len(set(labels))
    acts = [a for a in w.menuBar().findChildren(QtGui.QAction) if a.text()]
    texts = [a.text().replace("&", "") for a in acts]
    pal = next(a for a in acts if a.text() == "Command palette")
    assert pal.shortcut() == QtGui.QKeySequence("Ctrl+K")
    assert texts.count("AI chat") == 1 and "Ask AI about selection…" not in texts
    assert next(a for a in acts if a.text() == "AI chat").shortcut() == QtGui.QKeySequence("Ctrl+J")
    assert "Restart and run all" not in texts and "Load plugin file…" not in texts
    assert "Dark background" not in [a.text() for a in w.menuBar().actions()[2].menu().actions()]
    w.step_menu.aboutToShow.emit()                       # one frame of Cu: no trajectory steps
    steps = {a.text(): a for sub in w.step_menu.actions() if sub.menu() for a in sub.menu().actions()}
    assert not steps["Mean squared displacement / diffusion"].isEnabled()
    assert "trajectory" in steps["Mean squared displacement / diffusion"].toolTip()
    assert not steps["Isosurface"].isEnabled() and steps["Common neighbor analysis"].isEnabled()
    w.close()


def test_steps_follow_the_data(cu_fcc):
    """Insert › Step: why a step cannot run on the data, and what goes in before it."""
    from ewald.gui.pipeline_panel import prepare_step, step_unavailable

    pipe = Pipeline(StaticSource([cu_fcc, cu_fcc]))
    assert step_unavailable(mods.MeanSquaredDisplacement(), cu_fcc, pipe) is None
    assert "single frame" in step_unavailable(mods.MeanSquaredDisplacement(), cu_fcc, Pipeline(StaticSource(cu_fcc)))
    assert "grid" in step_unavailable(mods.CreateIsosurface(), cu_fcc, pipe)
    assert "vector" in step_unavailable(mods.VectorVis(), cu_fcc, pipe)
    assert "orientations" in step_unavailable(mods.GrainSegmentation(), cu_fcc, pipe)
    assert "Selection" in step_unavailable(mods.FreezeProperty(), cu_fcc, pipe)
    added = prepare_step(mods.BondAngleDistribution(), cu_fcc)
    assert [type(m).__name__ for m in added] == ["CreateBonds", "BondAngleDistribution"]
    srtio3 = cu_fcc.copy()
    types = np.array(["Sr"] * srtio3.count, dtype=object)
    types[: srtio3.count // 2] = "O"
    types[:4] = "Ti"
    srtio3.particles["Particle Type"] = types
    poly = prepare_step(mods.CoordinationPolyhedra(), srtio3)[0]
    assert poly.center_types == ["Ti"]                   # the smaller cation of the compound
    assert step_unavailable(mods.CoordinationPolyhedra(), srtio3, pipe) is None
    assert "center" in step_unavailable(mods.CoordinationPolyhedra(), cu_fcc, pipe)


def test_layout_moves_grey_out_when_they_do_nothing(app, pipe):
    """View › Layout: a view alone in its group cannot split off itself; the menu says so."""
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    wb = w.workbench
    wb.active = wb.group_of(w.viewport)
    wb.update_actions()
    acts = {a.text(): a for a in wb.actions()}
    assert not acts["Split right"].isEnabled() and "alone" in acts["Split right"].toolTip()
    assert acts["Move view down"].isEnabled()           # into the panel
    wb.enable_actions()
    assert acts["Split right"].isEnabled()
    groups = len(wb.layout.groups())
    acts["Split right"].trigger()                        # the shortcut path checks too: no change
    assert len(wb.layout.groups()) == groups
    w.close()
