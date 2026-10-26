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
    notebook_panel.CELL_BUDGET = None  # cells finish before run() returns; background tests set it
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


@pytest.fixture(autouse=True)
def _close_new_windows(app):
    """Destroy the windows a test leaves open. Every live widget is restyled on a theme switch
    and repolished on show, so leftovers from earlier tests made later GUI tests minutes slower."""
    before = set(app.topLevelWidgets())
    yield
    for w in app.topLevelWidgets():
        if w not in before:
            w.close()
            w.deleteLater()
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.fixture
def pipe(cu_fcc):
    return Pipeline(StaticSource([cu_fcc, cu_fcc]))


def _dispose(widget):
    """Delete a parentless test widget now, in the GUI thread. Left to the garbage collector, its
    signal-lambda cycles may be collected in a compute thread, and PySide crashes deleting it there."""
    import gc

    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    gc.collect()


def test_classic_layout_setting_is_retired(tmp_path, caplog):
    """The OVITO-like classic layout is gone: an old settings file asking for it loads quietly."""
    import json

    from ewald import settings

    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"general.layout": "classic", "general.startup_workspace": "Analyze"}))
    with caplog.at_level("WARNING"):
        s = settings.load(p)
    assert not caplog.records and "general.layout" not in settings.OPTION
    assert s["general.startup_workspace"] == "Notebook" and s.changed() == {}


def test_view_tab_surfaces_labels_transparency(app, pipe):
    """The View tab's Surfaces card restyles polyhedra by name, and the Atoms and Types cards set atoms
    on/off, labels, the selection highlight and per-type transparency; all go into the view settings."""
    from ewald.gui.pipeline_panel import DisplayPanel

    pipe.modifiers.append(mods.CoordinationPolyhedra(center_types=["Cu"]))
    panel = DisplayPanel()
    panel.set_pipeline(pipe)
    panel.set_data(pipe.compute(0))
    card = panel.sections["Surfaces"]
    shown = [c for c in card.findChildren(QtWidgets.QCheckBox) if c.text() == "coordination-polyhedra"]
    assert shown and shown[0].isChecked()
    shown[0].setChecked(False)
    card.findChild(QtWidgets.QDoubleSpinBox).setValue(0.6)
    assert panel.style.surfaces == {"coordination-polyhedra": {"visible": False, "transparency": 0.6}}
    atoms = panel.sections["Atoms"]
    [c for c in atoms.findChildren(QtWidgets.QCheckBox) if c.text() == "Draw atoms"][0].setChecked(False)
    labels = panel.sections["Labels"]
    [c for c in labels.findChildren(QtWidgets.QCheckBox) if c.text() == "Highlight selection"][0].setChecked(False)
    field = [e for e in labels.findChildren(QtWidgets.QLineEdit) if "{type}{index}" in e.placeholderText()][0]
    field.setText("{type}{index}")
    field.editingFinished.emit()
    panel.set_type_transparency("Cu", 0.5)
    st = panel.style
    assert (st.show_atoms, st.selection, st.labels, st.type_transparency) == (False, False, "{type}{index}", {"Cu": 0.5})
    settings = st.view_settings(["Cu"])
    assert settings["atoms"] is False and settings["labels"] == "{type}{index}" and settings["transparency"] == {"Cu": 0.5}
    assert settings["surfaces"]["coordination-polyhedra"]["visible"] is False
    lw = panel.light_widgets
    assert not lw["box"].isChecked()
    lw["box"].setChecked(True)
    lw["azimuth"].setValue(60)
    lw["intensity"].setValue(1.5)
    assert st.light == {"azimuth": 60.0, "elevation": 29.0, "intensity": 1.5}   # default light: 29° up
    assert st.view_settings(["Cu"])["light"] == st.light and st.scene_kwargs()["light"] == st.light
    lw["box"].setChecked(False)
    assert st.light is None and "light" not in st.scene_kwargs()
    _dispose(panel)


def test_view_tab_opens_no_window_and_keeps_its_cards_over_neb_frames(app, vasp_dir):
    """A NEB band with trajectory lines played on Windows flashed a window at every frame: the View tab
    was rebuilt (the OverlayFrames copies change the atom count) and a checked group showed its rows
    before they had a parent. Neither happens now."""
    from ewald import import_file
    from ewald.gui.pipeline_panel import DisplayPanel

    pipe = import_file(vasp_dir / "neb")
    pipe.add(mods.GenerateTrajectoryLines())
    shown = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QtCore.QEvent.Show and isinstance(obj, QtWidgets.QWidget) and obj.isWindow():
                shown.append(obj.objectName())
            return False

    panel = DisplayPanel()
    panel.style.trajectory_color_by = "Time"
    panel.set_pipeline(pipe)
    spy = Spy()
    app.installEventFilter(spy)
    try:
        panel.set_data(pipe.compute(0))
        page = panel.scroll.widget()
        for k in (1, 2, 3, 4):
            panel.set_data(pipe.compute(k))
        assert panel.scroll.widget() is page                 # frames of one band keep the cards
        panel.refresh()                                      # a rebuild with checked groups
        app.processEvents()
    finally:
        app.removeEventFilter(spy)
    assert shown == [], shown
    _dispose(panel)


def test_view_tab_trajectory_color_map(app, pipe):
    """The Elements card colors trajectory lines with a color map over a Time window or a trail; the values
    go into the style (and so the view cell) and the viewport draws them with a color bar."""
    from ewald.gui.pipeline_panel import DisplayPanel
    from ewald.render import build_scene

    pipe.modifiers.append(mods.GenerateTrajectoryLines(only_selected=False))
    panel = DisplayPanel()
    panel.set_pipeline(pipe)
    data = pipe.compute(1)
    panel.set_data(data)
    tw = panel.trajectory_widgets
    assert not tw["group"].isChecked() and tw["color_by"].findText("Path Length") >= 0
    tw["group"].setChecked(True)
    tw["colormap"].setCurrentText("magma")
    tw["window"][0].setText("0")
    tw["window"][0].editingFinished.emit()
    tw["trail"].setText("all")
    tw["trail"].editingFinished.emit()
    st = panel.style
    assert (st.trajectory_color_by, st.trajectory_colormap, st.trajectory_time_range, st.trajectory_trail) == \
        ("Time", "magma", [0.0, None], True)
    settings = st.view_settings(["Cu"])
    assert settings["trajectory_color_by"] == "Time" and settings["trajectory_trail"] is True
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=(100, 80))
    scene = build_scene(p, data, **st.scene_kwargs())
    assert scene.legends[0]["cmap"] == "magma" and scene["trajectories"].primitive.point_rgb is not None
    p.close()
    tw["group"].setChecked(False)
    assert st.trajectory_color_by is None and "trajectory_color_by" not in st.scene_kwargs()
    tw["draw"].setCurrentText("Tube")
    tw["radius"].setValue(0.12)
    assert st.trajectory_style == "tube" and st.trajectory_radius == 0.12
    assert st.view_settings(["Cu"])["trajectory_style"] == "tube"
    tw["radius"].setValue(0)
    assert st.trajectory_radius is None
    _dispose(panel)


def test_view_tab_cards_and_notebook_marks(app, pipe):
    """The View tab shows the look as cards, marks what the notebook's view sets, and an edit there
    changes the style without rebuilding the card being edited."""
    from ewald import view
    from ewald.gui.pipeline_panel import DisplayPanel

    panel = DisplayPanel()
    panel.set_pipeline(pipe)
    panel.set_data(pipe.compute(0))
    assert list(panel.sections) == ["Atoms", "Colors", "Types", "Bonds", "Surfaces", "Cell", "Labels", "Camera", "Elements"]
    marked = lambda: {lab.text() if isinstance(lab, QtWidgets.QLabel) else lab.title() for lab, _ in panel._marks if lab.property("set")} - {""}
    assert not marked()
    view(pipe, radius_scale=0.4, boundary=[-0.1, 1.1], direction="z")
    panel.set_data(pipe.compute(0), refresh=True)
    assert {"Radius scale", "Boundary (VESTA)", "Direction", "in notebook"} <= marked()
    card = panel.sections["Atoms"]
    edited = []
    panel.styleChanged.connect(lambda: (edited.append(1), panel.refresh()))   # as the window does
    card.findChild(QtWidgets.QDoubleSpinBox).setValue(0.7)
    assert panel.style.radius_scale == 0.7 and edited and panel.sections["Atoms"] is card
    page = panel.scroll.widget()
    page.resize(260, page.height())                         # narrow: the cards stack
    page.layout().setGeometry(QtCore.QRect(0, 0, 260, page.layout().heightForWidth(260)))
    xs = {c.x() for c in panel.sections.values()}
    assert len(xs) == 1 and all(c.width() <= 260 for c in panel.sections.values())
    asked = []
    panel.modifierRequested.connect(asked.append)
    panel.add_modifier(mods.CoordinationAnalysis(3.0))  # a step goes to the notebook, not into the pipeline
    assert [type(m).__name__ for m in asked] == ["CoordinationAnalysis"] and pipe.modifiers == []
    _dispose(panel)


def test_bonds_card_rule_pbc_column(app):
    """The Bonds card shows the bond rules with a PBC choice per rule (on, off, + atoms); a change
    writes the rule's fifth item, rules left at 'on' stay four items."""
    from ase.spacegroup import crystal

    from ewald import DataCollection
    from ewald.gui.pipeline_panel import DisplayPanel

    atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                    cellpar=[3.905] * 3 + [90] * 3)
    pipe = Pipeline(StaticSource(DataCollection.from_ase(atoms)))
    panel = DisplayPanel()
    panel.set_pipeline(pipe)
    panel.style.bond_pairs = [["Sr", "O", 0.0, 2.9], ["Ti", "O", 0.0, 2.2, "complete"]]
    panel.set_data(pipe.compute(0))
    table = panel.sections["Bonds"].findChild(QtWidgets.QTableWidget)
    assert table.columnCount() == 5 and table.horizontalHeaderItem(4).text() == "PBC"
    assert table.cellWidget(0, 4).currentText() == "on" and table.cellWidget(1, 4).currentText() == "+ atoms"
    table.cellWidget(0, 4).setCurrentIndex(1)
    assert panel.style.bond_pairs[0] == ["Sr", "O", 0.0, 2.9, False]
    assert panel.style.bond_pairs[1] == ["Ti", "O", 0.0, 2.2, "complete"]
    _dispose(panel)


def test_view_tab_columns_fold_and_aligned_labels(app, pipe):
    """The View tab lays its cards out in equal-width columns, every form label has one width (fields
    line up across cards), and a folded card stays folded across rebuilds."""
    from ewald.gui.pipeline_panel import DisplayPanel

    panel = DisplayPanel()
    panel.set_pipeline(pipe)
    panel.set_data(pipe.compute(0))
    page = panel.scroll.widget()
    lay = page.layout()
    width = 3 * lay._column + 2 * lay._gap + 20 + 40
    page.resize(width, page.height())
    lay.setGeometry(QtCore.QRect(0, 0, width, lay.heightForWidth(width)))
    cards = list(panel.sections.values())
    assert lay.columns(width) == 3 and len({c.x() for c in cards}) == 3 and len({c.width() for c in cards}) == 1
    labels = [lab for lab, _ in panel._marks if isinstance(lab, QtWidgets.QLabel) and lab.objectName() == "viewLabel"
              and not lab.text().startswith("Rules (")]
    assert len({lab.width() if lab.isVisible() else lab.maximumWidth() for lab in labels}) == 1
    panel.set_folded("Bonds", True)
    assert not panel.sections["Bonds"].content.isVisibleTo(panel.sections["Bonds"])
    panel.refresh()
    assert panel.sections["Bonds"].folded and not panel.sections["Bonds"].content.isVisibleTo(panel.sections["Bonds"])
    panel.set_folded("Bonds", False)
    assert panel.sections["Bonds"].content.isVisibleTo(panel.sections["Bonds"])
    _dispose(panel)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_view_cards_fold_all_and_stay_folded_after_restart(app, pipe):
    """View › Fold all View cards (or right-click a card) folds every card; the folded cards
    come back folded in the next window (layout.json)."""
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.gui.preferences import save_layout

    s = settings.Settings()
    w = MainWindow(pipe, settings=s)
    acts = {a.text(): a for a in w.menuBar().findChildren(QtGui.QAction)}
    acts["Fold all View cards"].trigger()
    assert all(c.folded for c in w.display.sections.values())
    acts["Unfold all View cards"].trigger()
    assert not any(c.folded for c in w.display.sections.values())
    assert [a.text() for a in w.display.actions()] == ["Fold all cards", "Unfold all cards"]
    w.display.set_folded("Bonds", True)
    w.display.set_folded("Camera", True)
    save_layout(w, s)
    w.close()
    w2 = MainWindow(pipe, settings=s)
    assert {t for t, c in w2.display.sections.items() if c.folded} == {"Bonds", "Camera"}
    w2.close()


def test_inspector_source_tab(app, pipe):
    from ewald.gui.inspector import DataInspector

    ins = DataInspector()
    ins.set_pipeline(pipe)
    ins.set_data(pipe.compute(0))
    text = ins.source.toPlainText()
    assert "Source     StaticSource" in text and "Frames     2" in text and "Position" in text
    _dispose(ins)


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


def test_parameter_form_color_fields_take_any_color_form(app):
    from ewald.gui.params import ColorEdit, ParameterForm

    mod = mods.AssignColor(color=(0.3, 0.3, 1.0))
    form = ParameterForm(mod)
    edit = form.editors["color"][0]
    assert isinstance(edit, ColorEdit) and edit.line.text() == "(0.3, 0.3, 1.0)"
    for text, value in (("rgb(255, 128, 0)", "rgb(255, 128, 0)"), ("tab:blue", "tab:blue"),
                        ("(255, 128, 0)", (255, 128, 0)), ("#f80", "#f80")):
        edit.line.setText(text)
        form.apply()
        assert mod.color == value
    edit.line.setText("notacolor")                    # marked, and the step keeps its color
    assert edit.line.property("invalid") is True
    form.apply()
    assert mod.color == "#f80"
    assert not isinstance(ParameterForm(mods.VectorVis()).editors["color_mapping_property"][0], ColorEdit)


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
    assert panel.table.currentText() == "Steps"
    assert panel.column.currentText() == "Temp"  # one column at a time for a time series
    ax = panel.figure.axes[0]
    assert ax.get_xlabel() == "Step" and len(ax.lines) == 2  # Temp + frame marker
    panel.column.setCurrentText("Volume")  # only printed in run 2: NaNs dropped
    np.testing.assert_array_equal(panel.figure.axes[0].lines[0].get_xdata(), [20, 25, 30, 35, 40])
    ax = panel.figure.axes[0]
    panel.set_data(pipe.compute(3))
    assert panel.figure.axes[0] is ax  # frame change moves the marker, no full redraw
    assert list(panel._marker.get_xdata()) == [30, 30]


def test_plots_keep_a_frame_independent_plot_during_playback(app, cu_fcc):
    from ewald.gui.plots import PlotsPanel

    msd = {"Time": np.arange(5.0), "MSD": np.arange(5.0) ** 2}
    panel = PlotsPanel()
    d0 = cu_fcc.copy()
    d0.tables["msd"] = msd
    panel.set_data(d0)
    ax = panel.figure.axes[0]
    d1 = cu_fcc.copy()
    d1.tables["msd"] = dict(msd)          # a trajectory-wide result: the same arrays on every frame
    panel.set_data(d1)
    assert panel.figure.axes[0] is ax     # no fig.clear()
    d2 = cu_fcc.copy()
    d2.tables["msd"] = {k: v.copy() for k, v in msd.items()}   # recomputed (kernel), same values
    panel.set_data(d2)
    assert panel.figure.axes[0] is ax
    d3 = cu_fcc.copy()
    d3.tables["msd"] = {"Time": msd["Time"], "MSD": msd["MSD"] + 1}   # per-frame data: new line data
    panel.set_data(d3)
    assert panel.figure.axes[0] is ax and list(ax.lines[0].get_ydata()) == list(msd["MSD"] + 1)
    _dispose(panel)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_playback_leaves_frame_independent_cells_alone(app, monkeypatch):
    from ewald import settings
    from ewald.gui import cell_widget
    from ewald.gui.app import MainWindow

    w = MainWindow(settings=settings.Settings())
    nb = os.path.join(os.path.dirname(__file__), "..", "examples", "notebooks", "cu_md_trajectory.py")
    view = w.panel.open_notebook(nb, trusted=True)
    view.run_all()
    app.processEvents()
    rebuilt, cleared = [], []
    fit = cell_widget.OutputArea._fit_images
    monkeypatch.setattr(cell_widget.OutputArea, "_fit_images", lambda self: (rebuilt.append(self), fit(self)))
    panels = {p.figure: p for p in w.plot_panels()}
    import matplotlib.figure

    clear = matplotlib.figure.Figure.clear
    monkeypatch.setattr(matplotlib.figure.Figure, "clear",
                        lambda self, *a, **k: (cleared.append(panels.get(self)), clear(self, *a, **k))[1])
    w.timeline.play.setChecked(True)
    for f in (1, 2, 3):
        w.timeline._set(f)
        app.processEvents()
    w.timeline.play.setChecked(False)
    app.processEvents()
    assert w.frame >= 3                   # (the play timer may have moved on by one)
    assert rebuilt == []                  # no cell's outputs (the printed "10 frames") rebuilt per frame
    # the MSD plot is left alone, the per-frame histogram updates its bars in place: no figure clears
    assert [p for p in cleared if p is not None] == []
    hist = next(p for p in panels.values() if p.hist.currentText() == "Displacement Magnitude")
    counts, _ = np.histogram(np.asarray(hist.data["Displacement Magnitude"]), bins=len(hist._ax.patches))
    assert [r.get_height() for r in hist._ax.patches] == list(counts)   # the frame on screen
    w.close()


LEVELS = {"energy_axis": "E - E_F",
          "levels": [{"name": "E_F", "energy": 0.0, "z": None, "style": "dashed"},
                     {"name": "VBM", "energy": -1.0, "z": [12.0, 18.0], "style": "solid"}],
          "regions": [{"name": "film plateau", "z": [12.0, 18.0]}],
          "gaps": [{"name": "Φ = 1.00 eV", "lower": -1.0, "upper": 0.0}]}


def test_plots_draw_an_analysis_tables_levels_and_named_rows(app):
    """A table carrying levels, plateaus and barriers (an analysis result in a frame's tables)
    draws them in the Plots panel as under a cell, in the theme's colors; one value per named
    row draws as bars."""
    import matplotlib.colors

    from ewald import DataCollection
    from ewald.gui.plots import PlotsPanel
    from ewald.gui.theme import TOKENS
    from ewald.verbs import Table

    z = np.linspace(0, 20, 50)
    d = DataCollection(np.zeros((1, 3)), np.array(["Si"]))
    d.tables["Lineup"] = Table("Lineup", {"z (A)": z, "Planar average": np.sin(z)}, attributes=LEVELS)
    panel = PlotsPanel()
    panel.set_data(d.copy())
    ax = panel.figure.axes[0]
    texts = {t.get_text(): t for t in ax.texts}
    assert {"E_F", "VBM", "film plateau", "Φ = 1.00 eV"} <= set(texts)
    assert ax.get_ylabel() == "E - E_F (eV)" and len(ax.patches) == 1     # the plateau shaded
    ef = next(line for line in ax.lines if line.get_linestyle() == "--")
    assert matplotlib.colors.to_hex(ef.get_color()) == TOKENS["text"].lower()   # not black on a dark theme
    moved = dict(LEVELS, levels=[dict(LEVELS["levels"][0], energy=0.5)])
    d2 = d.copy()
    d2.tables["Lineup"] = Table("Lineup", dict(d.tables["Lineup"]), attributes=moved)   # same curve
    panel.set_data(d2)
    ax = panel.figure.axes[0]
    assert next(t for t in ax.texts if t.get_text() == "E_F").xy[1] == 0.5 and "VBM" not in {
        t.get_text() for t in ax.texts}
    d.tables["Work functions"] = Table("Work functions", {"Surface #": np.arange(3),
                                                          "Work function (eV)": [4.6, 4.9, 4.4]},
                                       labels=["111", "100", "110"])
    panel.set_data(d)
    panel.table.setCurrentText("Work functions")
    ax = panel.figure.axes[0]
    assert len(ax.patches) == 3 and [t.get_text() for t in ax.get_xticklabels()] == ["111", "100", "110"]
    assert ax.get_ylim()[0] > 0                    # 4.4 and 4.9 eV do not look alike
    _dispose(panel)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_view_cell_of_an_analysis_table_is_a_live_plot_with_its_levels(app, tmp_path):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    path = tmp_path / "analysis.py"
    path.write_text(
        "# %%\nimport numpy as np\nfrom ewald import load\nfrom ewald.verbs import Table\n"
        f"from ewald.views import show\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n"
        "z = np.linspace(0, 20, 50)\n"
        f"lineup = Table('Lineup', {{'z (A)': z, 'Planar average': np.sin(z)}}, attributes={LEVELS!r})\n\n"
        "# %%\nshow(lineup)\n", encoding="utf-8")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    cell = view.widgets[1]
    plot = cell.plot_panel
    assert list(plot.data.tables) == ["Lineup"] and plot.table.currentText() == "Lineup"   # not pipe's tables
    assert not cell.plot_slot.docked                                    # not linked: under the cell
    assert {"E_F", "VBM", "film plateau"} <= {t.get_text() for t in plot.figure.axes[0].texts}
    w.timeline._set(3)                                                  # another frame of pipe
    app.processEvents()
    assert list(cell.plot_panel.data.tables) == ["Lineup"]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_profile_plot_aligns_the_3d_view_selects_slabs_and_marks_positions(app, tmp_path):
    """A profile along c shown on its own (an analysis table, ``attributes["axis"]``) links to the
    structure in the 3D view: Align 3D view turns it side-on with the plot's range, zooming either
    moves both, turning the view away lets go; a dragged range picks that slab, hovering puts a
    plane there and the picked atoms show as ticks."""
    from ewald import settings
    from ewald.gui.app import MainWindow

    data = os.path.join(os.path.dirname(__file__), "data", "interface_tin_mgo_100.extxyz")
    path = tmp_path / "profile.py"
    path.write_text(
        "# %%\nimport numpy as np\nfrom ewald import load\nfrom ewald.verbs import Table\n"
        f"from ewald.views import show\npipe = load({data!r})\n"
        "z = np.linspace(0, 29.35, 200)\n"
        "profile = Table('Profile', {'z (A)': z, 'V': np.cos(z)}, attributes={'axis': 'c'})\n\n"
        "# %%\nshow(profile, panel=False)\n", encoding="utf-8")
    w = MainWindow(settings=settings.Settings())
    w.resize(1200, 800)
    w.show()
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    for _ in range(20):
        app.processEvents()
    plot = view.widgets[1].plot_panel
    assert plot.link.kind == "space" and plot.space_data() is w.data
    assert plot.sync.isVisibleTo(plot) and plot.sync.isEnabled()

    plot.set_view(xlim=(8.0, 18.0))
    plot.sync.setChecked(True)                                 # Align 3D view
    app.processEvents()
    vp = w.viewport
    assert w._aligned_plot is plot
    assert vp.axis_view_range(w.data.cell, "c") == pytest.approx((8.0, 18.0))
    plot.set_view(xlim=(10.0, 14.0))                           # the plot zooms: the view follows
    app.processEvents()
    assert vp.axis_view_range(w.data.cell, "c") == pytest.approx((10.0, 14.0))
    vp.plotter.camera.parallel_scale *= 2                      # the view zooms out: the plot follows
    vp._camera_check()
    assert plot.view_limits()[0] == pytest.approx((8.0, 16.0))
    vp.rotate("azimuth", 30)                                   # turned away: no longer aligned
    vp._camera_check()
    assert not plot.sync.isChecked() and w._aligned_plot is None

    plot.select(10.0, 14.0)                                    # a slab: those atoms picked
    z = w.data.cell.to_fractional(w.data.positions)[:, 2] % 1 * 29.35
    assert sorted(vp.picks) == sorted(np.nonzero((z >= 10) & (z <= 14))[0].tolist())
    slab = vp.plotter.actors["_axis_slab"].GetMapper().GetInput().points   # and drawn as a slab
    assert sorted(set(np.round(slab[:, 2], 6))) == pytest.approx([10.0, 14.0])
    w._on_pick("")                                             # the picks come back as ticks on the plot
    assert len(plot._overlays) >= len(vp.picks)
    plot._hover_axis(12.0)                                     # the cursor at z = 12: a plane there
    assert "_axis_plane" in vp.plotter.actors
    np.testing.assert_allclose(vp.plotter.actors["_axis_plane"].GetMapper().GetInput().points[:, 2], 12.0)
    vp.style.replicate = [2, 3, 1]                             # across the copies drawn side by side
    plot._hover_axis(12.5)
    pts = vp.plotter.actors["_axis_plane"].GetMapper().GetInput().points
    np.testing.assert_allclose(np.ptp(pts, axis=0)[:2], [2 * 2.998, 3 * 2.998], atol=0.01)
    plot._hover_axis(None)
    assert "_axis_plane" not in vp.plotter.actors
    plot.clear_selection()                                     # the range dropped: the slab goes
    assert "_axis_slab" not in vp.plotter.actors
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_show_anywhere_in_a_cell_is_a_live_plot(app, tmp_path):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    path = tmp_path / "live.py"
    path.write_text(
        "# %%\nimport numpy as np\nfrom ewald import all_frames, load\nfrom ewald.modifiers import *\n"
        f"from ewald.views import show\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
        "# %%\n@all_frames\ndef height(frames, data):\n"
        "    z = [d['Position'][:, 2].mean() for d in frames]\n"
        "    data.tables['height'] = {'Frame': np.arange(len(z)), 'Mean z': np.array(z)}\n"
        "pipe.add(height)\n\n"
        "# %%\npipe.add(CalculateDisplacements(reference_frame=0))\nprint('two plots')\n"
        "show(pipe, table='height')\nshow(pipe, histogram='Displacement Magnitude')\n")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    cell = view.widgets[2]
    height, hist = cell.panels()                       # live plots, not PNGs, though the cell does more
    assert height.table.currentText() == "height" and hist.hist.currentText() == "Displacement Magnitude"
    assert height in w.plot_panels() and hist in w.plot_panels()   # linked like the Plots tab
    ax = height.figure.axes[0]
    w.timeline._set(3)
    app.processEvents()
    assert hist.data.attributes["SourceFrame"] == 3   # follows the frame
    assert height.figure.axes[0] is ax                 # all_frames result: computed once, not redrawn
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_deleting_the_window_with_a_docked_plot_in_a_split(app, tmp_path, monkeypatch):
    """Qt deletes the workbench (its splitters) before the notebook cells; the cell's docked plot
    then leaves the workbench, which must not touch the deleted splitters (seen on Windows:
    'Internal C++ object (QSplitter) already deleted' from Workbench._capture)."""
    import sys

    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc[1]))
    path = tmp_path / "docked.py"
    path.write_text(
        "# %%\nfrom ewald import load\nfrom ewald.modifiers import *\nfrom ewald.views import show\n"
        f"pipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
        "# %%\npipe.add(CalculateDisplacements(reference_frame=0))\n"
        "show(pipe, histogram='Displacement Magnitude')\n", encoding="utf-8")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    (docked,) = w.docked_plots()
    w.workbench.split(docked, "right")              # a QSplitter in the workbench
    app.processEvents()
    w.close()
    w.deleteLater()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert errors == []


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_linked_plots_open_in_the_plots_panel_others_stay_under_the_cell(app, tmp_path):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    path = tmp_path / "placed.py"
    path.write_text(
        "# %%\nimport matplotlib.pyplot as plt\nfrom ewald import load\nfrom ewald.modifiers import *\n"
        f"from ewald.views import show\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
        "# %%\npipe.add(CalculateDisplacements(reference_frame=0))\n"
        "show(pipe, histogram='Displacement Magnitude')\n"
        "show(pipe, histogram='Displacement Magnitude', panel=False)\n\n"
        "# %%\nplt.plot([1, 2, 3])\nplt.show()\n\n"
        "# %%\nshow(pipe, histogram='Displacement Magnitude')\n")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    cell, mpl_cell, view_cell = view.widgets[1], view.widgets[2], view.widgets[3]
    linked, inline = cell.outputs.slots
    assert linked.docked and not inline.docked                  # per frame: the Plots panel; panel=False: here
    assert linked.card.isVisibleTo(cell) and not inline.card.isVisibleTo(cell)
    assert inline.panel.isVisibleTo(cell) and linked.panel.parent() is not linked
    assert view_cell.plot_slot.docked and view_cell.plot_slot.card.png is not None   # thumbnail of its run
    assert mpl_cell.outputs.slots == [] and mpl_cell.outputs._images                # any other figure: an image
    assert sorted(w.docked_plots()) == sorted([linked.view_id, view_cell.plot_slot.view_id])
    assert w.workbench.widget(linked.view_id) is linked.panel and linked.panel in w.plot_panels()
    assert w.workbench.group_of(linked.view_id) is w.workbench.group_of("plot:1")   # next to "Plots"
    w.timeline._set(3)
    app.processEvents()
    assert linked.panel.data.attributes["SourceFrame"] == 3      # still follows the frame there

    cell._place_output(0, False)                                 # right-click › Show under the cell
    assert "show(pipe, histogram='Displacement Magnitude', panel=False)\nshow(pipe, histogram=" in cell.cell.source
    assert not linked.docked and linked.panel.parent() is linked and linked.view_id not in w.docked_plots()
    cell._place_output(1, True)                                  # ... › Show in Plots panel
    assert cell.cell.source.endswith("show(pipe, histogram='Displacement Magnitude', panel=True)")
    assert inline.docked and inline.view_id in w.docked_plots()
    view_cell._place_view(False)                                 # a view cell writes it into its call
    assert view_cell.cell.source.endswith("show(pipe, histogram='Displacement Magnitude', panel=False)")
    assert not view_cell.plot_slot.docked and view_cell.plot_slot.view_id not in w.docked_plots()

    w.workbench.close_view(inline.view_id)                       # closing the tab keeps the plot
    assert inline.view_id not in w.docked_plots() and inline.panel.parent() is None and inline.card.isVisibleTo(cell)
    inline.card.opened.emit()                                    # the thumbnail opens it again
    assert w.docked_plots() == [inline.view_id] and w.workbench.is_shown(inline.view_id)

    other = w.panel.new_notebook()                               # another notebook: its plots only
    app.processEvents()
    assert w.docked_plots() == []
    w.panel.tabs.setCurrentWidget(view)
    app.processEvents()
    assert w.docked_plots() == [inline.view_id]
    view.run(cell.cell)                                          # run again: the new plot takes the tab
    app.processEvents()
    new = view.widgets[1].outputs.slots[1]
    assert new is not inline and new.view_id == inline.view_id and w.workbench.widget(new.view_id) is new.panel
    w.panel.close_tab(w.panel.tabs.indexOf(view))
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert w.docked_plots() == [] and w._plot_docks == {} and w.panel.view is other
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_each_frame_figures_follow_the_frame_in_the_plots_panel(app, tmp_path):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    path = tmp_path / "figures.py"
    path.write_text(
        "# %%\nimport matplotlib.pyplot as plt\nfrom ewald import load, all_frames\n"
        f"pipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
        "# %% z histogram [each frame]\nplt.hist(data['Position'][:, 2], bins=20)\n\n"
        "# %%\n@all_frames\ndef mean_z(frames, data):\n    plt.plot([d['Position'][:, 2].mean() for d in frames])\n"
        "pipe.add(mean_z)\n")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    each = view.widgets[1]._figure_slots["Figure: z_histogram"]
    once = view.widgets[2]._figure_slots["Figure: mean_z"]
    assert each.docked and each.card.isVisibleTo(view.widgets[1]) and each.title == "z_histogram"
    assert w.docked_plots() == [each.view_id] and w.workbench.widget(each.view_id) is each.panel
    assert not once.docked and once.panel.isVisibleTo(view.widgets[2])    # all frames: under the cell
    png, kept = each.panel.png, once.panel.png
    w.timeline._set(4)
    app.processEvents()
    assert each.panel.png is not png and each.card.png is each.panel.png   # the frame on screen
    assert once.panel.png is kept                                           # drawn once
    once.placeRequested.emit(True)                                          # right-click › Plots panel
    assert once.docked and once.view_id in w.docked_plots()
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_cell_runs_once_each_frame_or_all_frames(app, tmp_path):
    from ewald import examples, settings
    from ewald.gui.app import MainWindow

    path = tmp_path / "frames.py"
    path.write_text(f"# %%\nfrom ewald import load\npipe = load({str(examples.path('Cu_md.extxyz'))!r})\n\n"
                    "# %% height [each frame]\ndata.attributes['z'] = data['Position'][:, 2].mean()\n\n"
                    "# %%\ndata.attributes['w'] = 1 / 0\n")
    w = MainWindow(settings=settings.Settings())
    view = w.panel.open_notebook(str(path), trusted=True)
    view.run_all()
    app.processEvents()
    assert view.widgets[1].frames_badge.text() == "Each frame"
    assert w.data.attributes["z"] == pytest.approx(w.data["Position"][:, 2].mean())
    view._cell_action(view.notebook.cells[2], "frames:each")      # the cell's menu › Runs › Each frame
    app.processEvents()
    assert view.widgets[2].frames_badge.text() == "Each frame"
    assert [m.name for m in w.pipeline.modifiers] == ["height", "each_frame_cell"]
    assert [o.kind for o in view.widgets[2].outputs.outputs] == ["error"]   # the step's error at this frame
    view._cell_action(view.notebook.cells[2], "frames:")          # back to once: the step goes
    app.processEvents()
    assert [m.name for m in w.pipeline.modifiers] == ["height"] and not hasattr(view.widgets[2], "frames_badge")
    view.undo()
    app.processEvents()
    assert view.notebook.cells[2].frames == "each"
    w.close()


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
def test_opening_one_neb_image_offers_the_whole_band(app, vasp_dir, monkeypatch):
    from ewald.gui.app import MainWindow

    w = MainWindow()
    asked = []

    def answer(reply):
        def question(*args, **kwargs):
            asked.append(args[2])
            return reply
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)

    answer(QtWidgets.QMessageBox.No)
    w.open_any(str(vasp_dir / "neb" / "02" / "OUTCAR"))
    assert asked[0].startswith("OUTCAR is one image of the NEB run")
    assert w.pipeline.num_frames == 1  # just that image
    answer(QtWidgets.QMessageBox.Yes)
    w.open_any(str(vasp_dir / "neb" / "02" / "OUTCAR"))
    assert w.pipeline.num_frames == 5 and any(type(m).__name__ == "NebEnergyProfile" for m in w.pipeline.modifiers)
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_dropping_a_neb_folder_opens_the_band(app, vasp_dir):
    from ewald.gui.app import MainWindow

    w = MainWindow()
    assert w.acceptDrops()
    w.open_dropped([str(vasp_dir / "neb")])
    assert w.pipeline.num_frames == 5
    w.close()


def test_join_files_dialog_orders_files_and_picks_join(app):
    from ewald.gui.join_dialog import JoinFilesDialog

    paths = [os.path.join("data", "run2", "dump.lammpstrj"), os.path.join("data", "run10", "dump.lammpstrj"),
             os.path.join("data", "run1", "dump.lammpstrj")]
    d = JoinFilesDialog(paths)
    assert [d.list.item(i).text() for i in range(3)] == [os.path.join(f"run{n}", "dump.lammpstrj") for n in (2, 10, 1)]
    assert d.list.item(0).toolTip() == paths[0] and d.join() == "timestep"
    d.sort_natural()
    assert d.paths() == [paths[2], paths[0], paths[1]]
    d.list.setCurrentRow(2)
    d.move(-1)
    assert d.paths() == [paths[2], paths[1], paths[0]] and d.list.currentRow() == 1
    d.move(5)   # past the end: nothing moves
    d.list.setCurrentRow(0)
    d.move(-1)
    assert d.paths() == [paths[2], paths[1], paths[0]]
    d.reverse()
    assert d.paths() == [paths[0], paths[1], paths[2]]
    d.join_buttons["concat"].setChecked(True)
    assert d.join() == "concat" and "nothing left out" in d.hint.text()
    assert JoinFilesDialog(paths, join="concat").join() == "concat"
    assert list(d.join_buttons) == ["timestep", "restart", "files", "concat"]
    d.join_buttons["restart"].setChecked(True)
    assert d.join() == "restart" and "step 0" in d.hint.text()
    d.close()


def test_join_dialog_names_files_from_any_platform():
    import ntpath
    import posixpath

    from ewald.gui.join_dialog import display_names

    assert display_names(["/a/run1/dump", "/a/run2/dump"], posixpath) == ["run1/dump", "run2/dump"]
    assert display_names(["C:\\md\\run1\\XDATCAR", "c:\\md\\run2\\XDATCAR"], ntpath) == \
        ["run1\\XDATCAR", "run2\\XDATCAR"]
    across = ["C:\\md\\XDATCAR", "D:\\md\\XDATCAR"]   # no common folder across drives
    assert display_names(across, ntpath) == across
    assert display_names(["/x/dump.200", "/x/dump.1000"], posixpath) == ["dump.200", "dump.1000"]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_open_several_files_as_one_trajectory(app, tmp_path, monkeypatch):
    """File › Open with several data files: one trajectory in natural order, as load([...])."""
    from ewald.data import Cell, DataCollection
    from ewald.gui.app import MainWindow
    from ewald.io import write_lammps_dump

    def frame(step):
        data = DataCollection(np.random.default_rng(step).random((4, 3)) * 5, cell=Cell(np.eye(3) * 5))
        data.particles["Particle Identifier"] = np.arange(1, 5)
        data.particles["Particle Type"] = np.array(["1"] * 4)
        data.attributes["Timestep"] = step
        return data

    write_lammps_dump(tmp_path / "dump.200", [frame(200), frame(300)])
    write_lammps_dump(tmp_path / "dump.1000", [frame(1000), frame(1100)])
    nb_path = tmp_path / "other.py"
    nb_path.write_text("# %%\nx = 1\n")
    chosen = [str(tmp_path / "dump.1000"), str(tmp_path / "dump.200")]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *a, **k: (chosen, ""))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("", False))
    asked = []

    def ask_join(parent, paths, join="timestep"):   # the dialog, accepted as shown
        asked.append(list(paths))
        return list(paths), join

    monkeypatch.setattr("ewald.gui.join_dialog.ask_join", ask_join)
    w = MainWindow()
    w.open_dialog()
    assert asked == [[str(tmp_path / "dump.200"), str(tmp_path / "dump.1000")]]   # shown in natural order
    pipe = w.pipeline
    assert [pipe.compute(i).attributes["Timestep"] for i in range(pipe.num_frames)] == [200, 300, 1000, 1100]
    assert "dump.200 + 1 more" in w.windowTitle()
    code = w.panel.notebook.to_py()
    assert "dump.200" in code and "dump.1000" in code and code.index("dump.200") < code.index("dump.1000")
    assert "join=" not in code

    # reordered in the dialog with every frame kept: load([...], join='concat') in that order
    monkeypatch.setattr("ewald.gui.join_dialog.ask_join", lambda parent, paths, join="timestep": (paths[::-1], "concat"))
    w.open_dialog()
    pipe = w.pipeline
    assert [pipe.compute(i).attributes["Timestep"] for i in range(pipe.num_frames)] == [1000, 1100, 200, 300]
    code = w.panel.notebook.to_py()
    assert "join='concat'" in code and code.index("dump.1000") < code.index("dump.200")
    monkeypatch.setattr("ewald.gui.join_dialog.ask_join", lambda *a, **k: None)   # cancelled: nothing opens
    w.open_dialog()
    assert w.pipeline is pipe

    monkeypatch.setattr(w, "_ask_trust", lambda path: False)
    chosen[:] = [str(nb_path), str(tmp_path / "dump.200")]   # a notebook opens in a tab, the data file as usual
    w.open_dialog()
    assert any(str(v.notebook.path).endswith("other.py") for v in w.panel.views()) and w.pipeline.num_frames == 2
    w.close()


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
    assert "Open file…" in entries(w.viewport_menu())  # nothing loaded yet
    w.set_pipeline(Pipeline(StaticSource([cu_fcc])))
    menu = entries(w.viewport_menu())
    assert "Open file…" not in menu and "Play" not in menu and "Clear picks" not in menu  # one frame, no picks
    assert "Look along" in menu and "Ask AI about this structure…" in menu
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
    w.close()  # the phonon entries: test_main_window_plays_phonon_mode


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
    w.render_from_page()                    # the ray tracer: on a worker thread, a spinner meanwhile
    button = w.render_page.render_button    # ... and Render turns into Cancel
    assert w.rendering and button.isEnabled() and button.text() == "Cancel"
    assert not w.render_page.busy_label.isHidden() and w.render_page.busy_label.text().startswith("Ray tracing…")
    w.wait_render()
    assert not w.rendering and button.isEnabled() and button.text() == "Render" and w.render_page.busy_label.isHidden()
    assert os.path.getsize(tmp_path / "x.png") > 1000
    w.render_page.output.setText(str(tmp_path / "cancelled.png"))
    w.render_from_page()
    button.click()                          # Cancel: stops after the rows being traced, no file, no error box
    assert w.render_page.busy_label.text() == "Cancelling…" and not button.isEnabled()
    w.wait_render()
    assert not (tmp_path / "cancelled.png").exists() and "cancelled" in w.statusBar().currentMessage()
    assert button.text() == "Render" and button.isEnabled()
    code = w.render_page.code.toPlainText()
    assert "CoordinationAnalysis(cutoff=3.0" in code and "camera=" in code
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_large_vector_render_asks_first(app, pipe, tmp_path, monkeypatch):
    """A vector figure of many shapes asks before drawing: PNG instead, vector anyway or cancel."""
    from ewald import vector
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(vector, "LARGE_VECTOR_SECONDS", 0.0)  # every scene counts as large
    w = MainWindow(pipe)
    page = w.render_page
    page.set_renderer("vector")
    page.preset.setCurrentText("Custom (px)")
    page.width.setValue(160)
    page.height.setValue(120)
    answers, asked = ["cancel", "vector", "raster"], []

    def ask(big):
        asked.append(big.estimate)
        a = answers.pop(0)
        return None if a == "cancel" else a

    monkeypatch.setattr(w, "ask_large_vector", ask)
    page.output.setText(str(tmp_path / "fig.svg"))
    w.render_from_page()
    assert not (tmp_path / "fig.svg").exists() and "cancelled" in w.statusBar().currentMessage()
    assert asked[0]["atoms"] == w.data.count and asked[0]["format"] == "svg"
    w.render_from_page()                                   # draw it anyway
    assert (tmp_path / "fig.svg").read_bytes()[:200].find(b"<svg") >= 0 and len(asked) == 2
    w.render_from_page()                                   # PNG instead: the page follows
    assert (tmp_path / "fig.png").stat().st_size > 500 and len(asked) == 3
    assert page.renderer_name in ("opengl", "raytrace") and page.output.text().endswith("fig.png")

    from PySide6 import QtCore, QtWidgets

    def press(name):  # the real dialog: its buttons map to the answers
        def click():
            box = next(b for b in QtWidgets.QApplication.topLevelWidgets() if b.objectName() == "LargeVectorDialog"
                       and b.isVisible())
            next(b for b in box.buttons() if b.text().startswith(name)).click()
        QtCore.QTimer.singleShot(50, click)

    monkeypatch.undo()
    big = vector.LargeVectorError({"atoms": 9000, "sticks": 100000, "faces": 0, "shapes": 109000,
                                   "seconds": 52.0, "bytes": 6.2e7, "format": "svg", "large": True})
    for name, want in (("Render PNG", "raster"), ("Render SVG", "vector"), ("Cancel", None)):
        press(name)
        assert w.ask_large_vector(big) == want
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


def test_preferences_ai_connection_rows_and_saved_connections(app, monkeypatch):
    from ewald import settings
    from ewald.gui.preferences import PreferencesDialog
    from ewald.llm import credentials

    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    s = settings.load()
    dlg = PreferencesDialog(s)
    rows = {k: dlg.editors[k][2] for k in ("llm.route_prefix", "llm.key_header", "llm.headers", "llm.effort")}
    assert rows["llm.route_prefix"].isHidden() and rows["llm.key_header"].isHidden()   # Claude: neither row
    assert not rows["llm.effort"].isHidden()
    assert "llm.user_id" not in dlg.editors and "llm.system_name" not in dlg.editors
    dlg.set_value("llm.provider", "openai-compatible")
    assert not rows["llm.route_prefix"].isHidden() and not rows["llm.headers"].isHidden()
    assert not rows["llm.key_header"].isHidden() and rows["llm.effort"].isHidden()
    dlg.set_value("llm.model", "gpt-oss-120b")
    dlg.set_value("llm.key_header", "x-ticket")
    dlg.set_value("llm.base_url", "http://gw.example/v1")
    dlg.set_value("llm.route_prefix", "openai/")
    dlg.set_value("llm.headers", "X-Team: md")
    dlg.save_profile("hub")
    dlg.set_value("llm.base_url", "")
    dlg.set_value("llm.model", "qwen3:8b")
    dlg.set_value("llm.route_prefix", "")
    dlg.save_profile("local")
    dlg.load_profile("hub")
    assert (dlg.work["llm.provider"], dlg.work["llm.model"], dlg.work["llm.headers"]) == ("openai-compatible",
                                                                                          "gpt-oss-120b", "X-Team: md")
    dlg.apply()
    saved = settings.load()
    assert set(saved["llm.profiles"]) == {"hub", "local"} and saved["llm.provider"] == "openai-compatible"
    assert saved["llm.profiles"]["hub"]["key_header"] == "x-ticket"
    box = dlg.editors["llm.profiles"][0]
    box.remove("local")
    assert set(dlg.work["llm.profiles"]) == {"hub"}
    dlg.close()


def test_ask_panel_agent_uses_ai_preferences(app, monkeypatch):
    from ewald import settings
    from ewald.gui import ask_panel

    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw))
    s = settings.load()
    s["llm.model"], s["llm.effort"] = "claude-haiku-5-5", "low"
    ask_panel.agent_from_settings(lambda: s)(None, None)
    kw = made[-1]
    assert (kw["provider"].name, kw["model"], kw["effort"]) == ("anthropic", "claude-haiku-5-5", "low")
    s["llm.provider"], s["llm.model"], s["llm.base_url"] = "openai-compatible", "", "http://localhost:1234/v1"
    s["llm.headers"] = "X-Team: md"
    factory = ask_panel.agent_from_settings(lambda: s)
    factory(None, None)
    kw = made[-1]
    assert (kw["provider"].name, kw["model"], kw["provider"].base_url) == ("openai-compatible", None,
                                                                          "http://localhost:1234/v1")
    assert kw["provider"].client_options["default_headers"] == {"X-Team": "md"}
    assert factory.assistant() == "AI"
    s["llm.provider"] = "gemini"
    assert factory.assistant() == "Gemini"


def test_ask_panel_new_agent_when_ai_preferences_change(app, pipe, monkeypatch):
    """Changing the headers, key, address or model in Preferences › AI applies to the next question,
    without restarting: the agent holds a client made from the old ones, so a new one is made and
    gets the conversation so far."""
    from ewald import settings
    from ewald.gui import ask_panel
    from ewald.llm import credentials

    s = settings.load()
    s["llm.provider"], s["llm.base_url"] = "openai-compatible", "https://gateway.example.org/v1"
    s["llm.key_header"], s["llm.headers"] = "x-ticket", "X-Team: md"
    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw) or _FakeAgent(p, ev))
    factory = ask_panel.agent_from_settings(lambda: s)
    panel = ask_panel.AskPanel(lambda: (pipe, 0, []), agent_factory=factory)
    seen = factory.connection()
    assert factory.connection() == seen

    panel.ask("first")
    _wait(app, panel)
    panel.ask("same connection")
    _wait(app, panel)
    assert len(made) == 1
    assert made[0]["provider"].client_options["default_headers"] == {"X-Team": "md"}

    s["llm.headers"] = "X-Team: md; X-Site: lab"               # Preferences › AI › Apply
    panel.ask("after the header change")
    _wait(app, panel)
    assert len(made) == 2
    assert made[1]["provider"].client_options["default_headers"] == {"X-Team": "md", "X-Site": "lab"}
    assert len([m for m in panel.messages if m.role == "user"]) == 3
    assert "first" in panel._conv().agent.calls[0][1]          # the new agent got the conversation

    for key, value in [("llm.key_header", "api-key"), ("llm.base_url", "https://other.example.org/v1"),
                       ("llm.model", "local-model"), ("llm.timeout", 30.0), ("llm.stream", False)]:
        before = factory.connection()
        s[key] = value
        assert factory.connection() != before, key
    before = factory.connection()
    monkeypatch.setattr(credentials.keychain, "set_secret", lambda *a, **k: "keychain")
    credentials.save_api_key("sk-test-new-key-0000000000", provider="openai-compatible")
    assert factory.connection() != before                      # a new key needs a new client too
    panel.ask("after the key change")
    _wait(app, panel)
    assert len(made) == 3


def test_ai_chat_reads_only_allowed_folders_and_asks_for_hosts(app, monkeypatch, tmp_path):
    import types

    from ewald import settings
    from ewald.gui import ask_panel

    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: types.SimpleNamespace(toolset=types.SimpleNamespace()))
    s = settings.load()
    s["llm.read_folders"] = str(tmp_path)
    asked = []
    agent = ask_panel.agent_from_settings(lambda: s, confirm_host=lambda h: asked.append(h) or False)(None, None)
    access = agent.toolset.read_access
    assert access.folders == [os.path.realpath(str(tmp_path))]
    with pytest.raises(PermissionError):
        access.check("cluster:/scratch/XDATCAR")
    assert asked == ["cluster"]
    s["llm.confirm_hosts"] = False
    agent = ask_panel.agent_from_settings(lambda: s, confirm_host=lambda h: False)(None, None)
    agent.toolset.read_access.check("cluster:/scratch/XDATCAR")


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_window_asks_before_the_ai_connects_to_a_host(app, monkeypatch):
    from PySide6 import QtWidgets

    from ewald.gui.app import MainWindow

    w = MainWindow()
    try:
        answers = iter([QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No])
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: next(answers))
        bridge = w.ask_panel.notebook_access(w.panel.notebook)   # each tab's chat has its own bridge
        assert bridge.confirm_host("cluster") is True
        assert bridge.confirm_host("other") is False
    finally:
        w.close()


def test_preferences_dialog_edit_search_restore(app, tmp_path):
    from ewald import settings
    from ewald.gui.preferences import ElementTable, PreferencesDialog

    s = settings.load()
    dlg = PreferencesDialog(s)
    assert [dlg.pages.item(i).text() for i in range(dlg.pages.count())] == list(settings.PAGES)
    dlg.set_value("viewport.rotation_step", 45.0)
    dlg.set_value("compute.device", "auto")
    assert s.is_default("viewport.rotation_step")           # untouched until Apply
    applied = []
    dlg.applied.connect(applied.append)
    dlg.apply()
    assert applied and s["viewport.rotation_step"] == 45.0
    assert settings.load()["compute.device"] == "auto"        # saved to disk
    dlg._search("radius")
    visible = {dlg.pages.item(i).text() for i in range(dlg.pages.count()) if not dlg.pages.item(i).isHidden()}
    assert visible == {"Atoms & bonds"}
    dlg._search("")
    dlg.show_page("3D view")
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
    assert w.device == "cpu" and w.style.background == "#ffffff" and w.style.bonds_enabled
    assert w.viewport.rotation_step == 30.0 and not w.render_page.raytrace
    assert w.style.type_colors["Cu"] == "#c8794a"
    assert any(a.text() == "Pr&eferences…" for a in w.menuBar().actions()[1].menu().actions())
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


def test_preferences_codes_page_is_drawn_from_the_registry(app, tmp_path):
    """One section per simulation code (VASP, LAMMPS, a plugin's), each with its own options."""
    from ewald import plugins, settings
    from ewald.codes import SimulationCode
    from ewald.gui.preferences import PreferencesDialog

    plugins.register_simulation_code(SimulationCode(
        "toycode", "Toy code", files=("*.toy",),
        options=[settings.Option("codes.toycode.flavour", "plain", "Flavour", choices=("plain", "spicy"))]))
    try:
        dlg = PreferencesDialog(settings.Settings({}, tmp_path / "s.json"), page="Codes")
        titles = [c[0].text() for c in dlg.code_cards]
        assert titles[:2] == ["VASP", "LAMMPS"] and "TOY CODE" in titles
        assert {"codes.lammps.type_prompt", "codes.toycode.flavour"} <= set(dlg.editors)
        dlg.set_value("codes.toycode.flavour", "spicy")
        dlg.apply()
        assert settings.load(tmp_path / "s.json")["codes.toycode.flavour"] == "spicy"
        dlg.search.setText("lammpstrj")
        visible = [c[0].text() for c in dlg.code_cards if not c[1].isHidden()]
        assert visible == ["LAMMPS"]
        dlg.close()
    finally:
        plugins._registry.codes.pop("toycode", None)


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
    w.wait_render()
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
    assert "Writing plugins" in labels and "Open plugin folder" not in labels  # in Insert › Step › Plugins
    assert w.help_url("plugins/index.md") == "https://example.org/ewald/plugins/"

    # registered after the window exists (startup script, plugin): appears when the menus open
    monkeypatch.setitem(pipeline_panel.MODIFIER_MENU, "Plugins", {"Late wrap": mods.WrapAtoms})
    for menu in (w.panel.add_menu, w.step_menu):
        menu.aboutToShow.emit()
        groups = {a.text(): a.menu() for a in menu.actions()}
        assert [a.text() for a in groups["Plugins"].actions()][0] == "Late wrap"
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
    assert [chip.text() for chip in claude._chips] == ["✓ Used describe_selection"]
    assert changed == [1] and panel.ask_button.isEnabled() and panel.input.toPlainText() == ""

    ctx["indices"] = []
    panel.ask("And the frame?")
    _wait(app, panel)
    assert agents[0].calls[-1] == ("frame", "And the frame?") and len(agents) == 1   # same conversation
    panel.reset()
    assert panel.text() == "" and not panel.welcome.isHidden()


def test_ask_panel_sends_the_color_rule(app, pipe):
    """What the 3D view shows (how the atoms are colored) goes with a question when it changed."""
    from ewald.gui.ask_panel import AskPanel

    shown = {"text": "Atoms are colored by 'Structure Type' (from the CNA step): 1 FCC = #66ff66."}
    panel = AskPanel(lambda: (pipe, 0, []), agent_factory=lambda p, ev: _FakeAgent(p, ev),
                     view=lambda: shown["text"])
    first = panel._notebook_prompt("What do the colors mean?")
    assert first.startswith("The 3D view now shows: Atoms are colored by 'Structure Type'")
    assert panel._notebook_prompt("again") == "again"            # unchanged: not sent twice
    shown["text"] = "Atoms have their particle type colors: Cu #c88033."
    assert "particle type colors" in panel._notebook_prompt("and now?")


def test_display_panel_colors_card(app, pipe):
    """The View tab's Colors card: the rule drawn (one row per category) and color_by / map / range."""
    from ewald.gui.pipeline_panel import DisplayPanel
    from ewald.gui.style import ViewStyle

    pipe.add(mods.CommonNeighborAnalysis())
    style = ViewStyle()
    panel = DisplayPanel(style)
    panel.set_pipeline(pipe)
    panel.set_data(pipe.compute(0), refresh=True)
    w = panel.color_widgets
    assert "Structure Type" in w["rule"].text() and "CommonNeighborAnalysis" in w["rule"].text()
    rows = [w["legend"].item(i, 1).text() for i in range(w["legend"].rowCount())]
    assert rows == ["FCC"] and w["legend"].item(0, 2).text() == f"{pipe.compute(0).count:,}"
    assert not w["colormap"].isEnabled()
    seen = []
    panel.styleChanged.connect(lambda: seen.append(1))
    w["color_by"].setCurrentText("Particle type")
    assert style.color_by == "type" and seen and "Particle type colors" in w["rule"].text()
    w["color_by"].setCurrentText("Position.Z")
    assert style.color_by == "Position.Z" and w["colormap"].isEnabled()
    w["colormap"].setCurrentText("magma")
    w["range"][1].setText("4")
    w["range"][1].editingFinished.emit()
    assert style.scene_kwargs()["color_range"] == [None, 4.0] and "magma from" in w["rule"].text()
    w["color_by"].setCurrentText("Steps")
    assert style.color_by is None and "color_by" not in style.scene_kwargs()
    _dispose(panel)


def test_ask_panel_without_claude(app, pipe):
    from ewald.gui.ask_panel import AskPanel

    def missing(p, ev):
        raise ModuleNotFoundError("No module named 'anthropic'")

    panel = AskPanel(lambda: (pipe, 0, []), agent_factory=missing)
    panel.ask("hello")
    assert "not available" in panel.text() and "Preferences › AI" in panel.text()
    assert not panel.busy()


def test_step_error_is_logged_once_not_every_frame(app, pipe):
    from ewald import Modifier
    from ewald.gui.app import MainWindow

    class NeedsPackage(Modifier):
        def modify(self, data, ctx):
            raise ImportError("NeedsPackage needs spglib")

    pipe.modifiers.append(NeedsPackage())
    w = MainWindow(pipe)
    for frame in (0, 1, 0, 1):
        w.frame = frame
        w.recompute(wait=True)
    assert w.log.toPlainText().count("NeedsPackage needs spglib") == 1


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
    assert shown(w.panel) and not shown(w.ask_dock)      # the AI chat side bar starts hidden
    w.ask_claude()
    app.processEvents()
    assert shown(w.ask_dock) and shown(w.panel)           # left of the notebook, not over it
    assert w.dockWidgetArea(w.ask_dock) == QtCore.Qt.LeftDockWidgetArea
    app.processEvents()
    a, n = w.ask_dock.geometry(), w.notebook_dock.geometry()
    assert a.center().x() < n.center().x() and a.top() == n.top()   # left of it, not stacked
    w.close()


def test_side_bars_open_together_stand_in_columns_left_of_the_notebook(app):
    """A second (third) side bar gets a column between the open ones and the notebook; splitting
    the notebook stacked it and the notebook under the first side bar, the panels overlapping."""
    from ewald.gui.app import MainWindow

    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    app.processEvents()
    for dock in (w.plugins_dock, w.remotes_dock, w.ask_dock):
        w._show_side_bar(dock)
        app.processEvents()
    docks = [w.plugins_dock, w.remotes_dock, w.ask_dock, w.notebook_dock]
    rects = [d.geometry() for d in docks]
    assert all(not d.isFloating() and w.dockWidgetArea(d) == QtCore.Qt.LeftDockWidgetArea for d in docks)
    assert len({r.top() for r in rects}) == 1 and len({r.height() for r in rects}) == 1   # one row
    assert all(a.right() < b.left() for a, b in zip(rects, rects[1:]))                   # left to right
    w.remotes_dock.hide()
    w._show_side_bar(w.remotes_dock)                       # shown again: back in a column, not stacked
    app.processEvents()
    assert w.remotes_dock.geometry().top() == w.notebook_dock.geometry().top()
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
    w.settings["llm.code_cells"] = "button"                # Preferences › AI › Code in answers
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

    w.settings["llm.code_cells"] = "insert"                # the default: code goes into the notebook at once
    assert view.undo() == "Insert cell"                    # (the button's cell would take the answer's code)
    n = len(view.notebook.cells)
    w.ask_panel.ask("Plot sin(x)")
    _wait(app, w.ask_panel)
    assert len(view.notebook.cells) == n + 1 and view.notebook.cells[-1].state == "new"   # inserted, not run
    assert view.notebook.cells[-1].source.strip() == "pipe.add(CoordinationAnalysis(cutoff=3.0))"
    assert w.ask_panel.messages[-1].code_blocks[0].insert_button.text() == "✓ Inserted"
    assert view.undo() == "Insert cell" and len(view.notebook.cells) == n
    w.settings["llm.run_cells"] = True                     # Let the AI run cells: the inserted code runs too
    w.ask_panel.ask("Plot sin(x)")
    _wait(app, w.ask_panel)
    for _ in range(400):
        app.processEvents()
        if view.notebook.cells[-1].state not in ("new", "running") and not view.running:
            break
        time.sleep(0.01)
    assert len(view.notebook.cells) == n + 1 and view.notebook.cells[-1].state == "ok"
    w.ask_panel.ask("Plot sin(x)")                         # the same step again: it goes into that cell
    _wait(app, w.ask_panel)
    for _ in range(400):
        app.processEvents()
        if not view.running:
            break
        time.sleep(0.01)
    assert len(view.notebook.cells) == n + 1 and view.notebook.cells[-1].state == "ok"
    w.close()


def test_ai_chat_edits_notebook_cells_from_its_thread(app, pipe):
    from ewald.gui.app import MainWindow
    from ewald.llm import Toolset

    results = {}

    class CellAgent:
        def __init__(self, pipeline, on_event):
            self.toolset = Toolset(pipeline)
            self.on_event = on_event

        def ask(self, question):   # runs on the chat's worker thread, like the real agent loop
            for name, args in self.script:
                results.setdefault(name, []).append(self.toolset.call(name, args))
            return "done"

    w = MainWindow(pipe)
    view = w.panel.view
    agent = CellAgent(pipe, None)
    w.ask_panel.agent_factory = lambda p, ev: agent
    agent.script = [("insert_cell", {"source": "x = 41 + 1"}), ("list_cells", {}), ("run_cell", {"index": -1})]
    w.ask_panel.ask("Add a cell")
    _wait(app, w.ask_panel)
    cell = view.notebook.cells[-1]
    assert cell.source == "x = 41 + 1" and cell.state == "new" and view.widget_for(cell) is not None
    assert not results["insert_cell"][0].is_error
    assert [c["first_line"] for c in results["list_cells"][0].content["cells"]][-1] == "x = 41 + 1"
    assert results["run_cell"][0].is_error and "not offered" in results["run_cell"][0].content
    assert "run_cell" not in [d["name"] for d in agent.toolset.definitions()]
    assert view.history.undo_label == "AI: insert cell"

    w.settings["llm.run_cells"] = True                    # Preferences › AI › Let the AI run cells
    agent.script = [("edit_cell", {"index": -1, "old_text": "41", "new_text": "40"}), ("run_cell", {"index": -1})]
    w.ask_panel.ask("Fix and run it")
    _wait(app, w.ask_panel)
    run = results["run_cell"][-1]
    assert not run.is_error and run.content["state"] == "ok" and view.notebook.namespace["x"] == 41
    assert view.notebook.cells[-1].source == "x = 40 + 1"
    assert view.undo() == "AI: edit cell"                   # each AI edit is one undo step (Ctrl+Z)
    assert view.notebook.cells[-1].source == "x = 41 + 1"
    w.close()


def test_ai_chat_tells_claude_about_the_notebook(app, monkeypatch):
    from ewald import settings
    from ewald.gui import ask_panel

    made = []
    monkeypatch.setattr(ask_panel, "_default_agent", lambda p, ev, **kw: made.append(kw))
    s = settings.Settings()
    ask_panel.agent_from_settings(lambda: s, notebook=True)(None, None)
    ask_panel.agent_from_settings(lambda: s)(None, None)
    assert "in place of the one cell that does the same job" in made[0]["system"] and "system" not in made[1]
    s["llm.code_cells"] = "button"
    ask_panel.agent_from_settings(lambda: s, notebook=True)(None, None)
    assert "Insert as notebook cell" in made[2]["system"]
    assert ask_panel.RUN_CELLS_NOTE[False] in made[0]["system"] and "(not run)" in made[0]["system"]
    s["llm.code_cells"] = "insert"
    s["llm.run_cells"] = True                              # Preferences › AI › Let the AI run cells
    factory = ask_panel.agent_from_settings(lambda: s, notebook=True)
    factory(None, None)
    assert ask_panel.RUN_CELLS_NOTE[True] in made[3]["system"] and "(and run)" in made[3]["system"]
    s["llm.run_cells"] = False                             # read again before every question
    assert ask_panel.RUN_CELLS_NOTE[False] in factory.system()


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
def test_playback_keeps_bonds_and_one_level_of_detail(app, pipe):
    """MD playback in Ball & stick flickered: frames near the frame-time budget went coarse (no bonds)
    and were refined (bonds) between frames. Now a coarse frame keeps its bonds unless the atoms are
    subsampled, playback keeps the detail its first frame got and refines only when it stops."""
    from ewald import settings
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    w = MainWindow(pipe, settings=s)
    vp = w.viewport
    assert vp.style.bonds_enabled and "bonds" in vp.plotter.actors
    vp.detail_mode, vp._build_time = "auto", 1.0           # a full frame misses the budget: coarse
    w.timeline.play.setChecked(True)
    assert vp.playing
    w.timeline.timer.stop()                                  # step by hand
    w.timeline._set(1)
    app.processEvents()
    assert w.frame == 1 and vp.coarse and "bonds" in vp.plotter.actors   # coarse, bonds kept
    assert not vp._refine_timer.isActive()                   # no refine between frames
    vp._build_time = 0.0                                     # would be full now, but playback keeps coarse
    w.timeline._set(0)
    app.processEvents()
    assert w.frame == 0 and vp.coarse and "bonds" in vp.plotter.actors
    w.timeline.play.setChecked(False)
    assert not vp.playing and vp._refine_timer.isActive()   # stopping refines the last frame
    vp.refine()
    assert not vp.coarse and "bonds" in vp.plotter.actors
    # rotating keeps bonds too while the atoms are not subsampled
    vp._render_time = 1.0
    vp._go_coarse_for_motion()
    assert vp.coarse and vp.plotter.actors["bonds"].GetVisibility()
    vp.refine()
    # subsampled atoms drop their bonds (half of them would dangle)
    vp.interactive_atoms = 100
    vp.redraw(coarse=True)
    assert "bonds" not in vp.plotter.actors
    vp.refine()
    assert "bonds" in vp.plotter.actors
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
    from ewald.gui.pipeline_panel import DisplayPanel
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parents[1] / "examples" / "plugins" / "cell_axes.py", folder)
    data = DataCollection([[1, 1, 1], [3, 3, 3]], ["Fe", "Fe"], Cell(np.eye(3) * 4))
    style = ViewStyle()
    panel = DisplayPanel(style)
    panel.set_pipeline(Pipeline(StaticSource(data)))
    panel._fill_visual_menu()
    assert [a.text() for a in panel.visual_menu.actions()] == ["Cell axes", "Selection halo"]
    redraws = []
    panel.styleChanged.connect(lambda: redraws.append(1))
    panel.add_visual("Cell axes")
    assert style.visuals == [{"name": "Cell axes", "params": {"length": 2.0, "width": 0.25}, "enabled": True}]
    form = panel._form  # the new element's form is open in the Elements card
    form.editors["length"][0].setValue(3.0)
    [b for b in panel.sections["Elements"].findChildren(QtWidgets.QPushButton) if b.text().startswith("Apply")][0].click()
    assert style.visuals[0]["params"]["length"] == 3.0 and redraws
    v = Viewport(style)
    v.set_data(data)
    assert "visual:cell-axes" in v.plotter.actors
    switch = [c for c in panel.sections["Elements"].findChildren(QtWidgets.QCheckBox) if c.text() == "Cell axes"][0]
    switch.setChecked(False)
    assert style.visuals[0]["enabled"] is False
    v.redraw()
    assert "visual:cell-axes" not in v.plotter.actors
    panel.remove_visual(0)
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
    w.wait_render()
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


def test_plots_update_per_frame_data_in_place(app):
    from ewald.gui.plots import PlotsPanel

    rng = np.random.default_rng(0)

    def frame(i, ints=False):
        from ewald import DataCollection
        from ewald.data import Cell

        d = DataCollection(rng.random((500, 3)) * 10, ["Cu"] * 500, Cell(np.eye(3) * 10))
        d.particles["E"] = rng.normal(-3 + 0.5 * i, 0.2 + 0.1 * i, 500)
        d.particles["N"] = rng.integers(10, 13, 500)
        r = np.linspace(0.05, 8, 100)
        d.tables["rdf"] = {"r": r, "g(r)": 1 + np.sin(3 * r + i) * np.exp(-r / (2 + i))}
        return d

    a, b = frame(0), frame(1)
    for choice in (("hist", "E"), ("hist", "N"), ("table", "rdf")):
        live, fresh = PlotsPanel(), PlotsPanel()
        for p in (live, fresh):
            p.resize(600, 400)
            p.set_data(a if p is live else b)
            getattr(p, choice[0]).setCurrentText(choice[1])
        axes = live._ax
        live.set_data(b)
        assert live._ax is axes, choice   # the same axes: updated, not rebuilt
        for p in (live, fresh):
            p.canvas.draw()
        np.testing.assert_allclose(live._ax.get_xlim(), fresh._ax.get_xlim())
        np.testing.assert_allclose(live._ax.get_ylim(), fresh._ax.get_ylim())
        if choice[0] == "hist":
            heights = [[r.get_height() for r in p._ax.patches] for p in (live, fresh)]
            np.testing.assert_allclose(*heights)
            np.testing.assert_allclose(live._edges, fresh._edges)
        else:
            np.testing.assert_allclose(live._ax.lines[0].get_ydata(), fresh._ax.lines[0].get_ydata())
    live.table.setCurrentText("")
    live.hist.setCurrentText("N")
    c = frame(2)
    c.particles["N"] = np.full(500, 7)   # other integer values: other bars, a full redraw
    axes = live._ax
    live.set_data(c)
    assert live._ax is not axes and [r.get_x() + 0.5 * r.get_width() for r in live._ax.patches] == [7]


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
    assert plugins_button.toolTip() == "Plugins (Ctrl+Shift+X)"
    tabs = w.panel.tabs
    assert tabs.count() == 0 and w.panel.view is None                    # a new window: the welcome page
    assert "_empty" in w.viewport.plotter.actors and not w.viewport._viewbar.isEnabled()   # and a hint in the view
    assert tabs.body.currentWidget() is w.panel.welcome
    w.open_any(cu_path, ask_types=False)
    assert "_empty" not in w.viewport.plotter.actors and w.viewport._viewbar.isEnabled()
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
    w.panel.close_tab(0)                                       # the last tab: the welcome page, no new notebook
    assert tabs.count() == 0 and w.panel.view is None and tabs.body.currentWidget() is w.panel.welcome
    assert w.pipeline is None and w.data is None                   # the 3D view, plots and inspector are cleared
    assert "_empty" in w.viewport.plotter.actors and not w.viewport._rail.isEnabled()
    from ewald.gui import welcome
    assert welcome.load_recent()[0] == str(path)                    # listed under Recent
    w.panel.welcome.refresh()
    recent = w.panel.welcome.findChildren(QtWidgets.QToolButton, "welcomeRecent")
    assert recent and recent[0].toolTip() == str(path)
    w._update_notebook_menu()
    assert not any(a.isEnabled() for a in w.notebook_menu.actions())     # Notebook › ... greys out
    w.panel.welcome.newRequested.emit()                               # New notebook on the welcome page
    assert tabs.count() == 1 and tabs.body.currentWidget() is tabs.stack
    w.panel.close_tab(0)
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
    from ewald.trust import _trust_store, is_trusted_file

    assert is_trusted_file(path)                              # saved here: it would open trusted
    _trust_store().unlink()                                   # as if someone else had sent it

    def always():
        s = settings.Settings()
        s["python.run_notebooks"] = "always"                  # trusted files run; others ask
        return s

    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)
    w2 = MainWindow(settings=always())
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
    assert not view.notebook.restricted and "Not run yet" in view.banner_text.text()
    view.run_all()
    assert view.banner.isHidden()
    assert [m.name for m in w2.pipeline.modifiers] == ["CoordinationAnalysis"]
    w2.close()

    assert is_trusted_file(path)                              # trusting it remembers it
    _trust_store().unlink()
    w3 = MainWindow(settings=always())             # Yes when opening: trusted, the bar goes away
    view = w3.open_notebook(str(path))
    assert not view.notebook.restricted and view.banner.isHidden()
    assert view.notebook.cells[0].state == "ok" and [m.name for m in w3.pipeline.modifiers] == ["CoordinationAnalysis"]
    w3.close()
    asked.clear()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: asked.append(a[2]))
    w3b = MainWindow(settings=always())            # trusted at w3's open: runs without asking
    view = w3b.open_notebook(str(path))
    assert not asked and not view.notebook.restricted and view.notebook.cells[0].state == "ok"
    w3b.close()
    _trust_store().unlink()

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.No)
    w4 = MainWindow(settings=always())             # No when opening, then the bar's Trust button
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
def test_opening_a_notebook_does_not_run_it_by_default(app, cu_path, tmp_path, monkeypatch):
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook
    from ewald.trust import is_trusted_file

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\nfrom ewald.modifiers import *\npipe = load({str(cu_path)!r})\n\n"
                          "# %%\npipe.add(CoordinationAnalysis(cutoff=3.0))\n")
    path = tmp_path / "mine.ipynb"
    nb.save(path)
    assert is_trusted_file(path)                              # saved here: trusted
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)
    s = settings.Settings()
    assert s["python.run_notebooks"] == "never"
    w = MainWindow(settings=s)
    w.show()
    view = w.open_notebook(str(path))                         # nothing asked, nothing run
    app.processEvents()
    assert not asked and not view.notebook.restricted and view.awaiting_run
    assert all(c.state == "new" for c in view.notebook.cells) and w.pipeline is None
    assert view.banner.isVisible() and "Not run yet" in view.banner_text.text()
    buttons = {b.text(): b for b in view.banner.findChildren(QtWidgets.QPushButton)}
    assert buttons["Trust"].isHidden()
    buttons["Run all"].click()                                # the bar's Run all
    assert view.wait()
    app.processEvents()
    assert not view.awaiting_run and not view.banner.isVisible()
    assert [m.name for m in w.pipeline.modifiers] == ["CoordinationAnalysis"]
    w.close()

    s["python.run_notebooks"] = "ask"                         # asks even for a trusted file
    w = MainWindow(settings=s)
    view = w.open_notebook(str(path))
    assert asked and "Run the cells" in asked[-1] and view.awaiting_run and not view.notebook.restricted
    w.close()


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
    import ewald.gui.notebook_panel as npn

    monkeypatch.setattr(npn, "CELL_BUDGET", 0.1)
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


def test_in_process_cells_run_off_the_gui_thread_and_interrupt(app, cu_path, tmp_path, monkeypatch):
    """A long cell in this app's Python runs on a worker thread: the window keeps responding,
    pyplot draws off screen, and Interrupt stops the cell (Run all stops there too)."""
    import threading
    import time

    import ewald.gui.notebook_panel as npn
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook

    nb = Notebook.from_py(f"# %%\nfrom ewald import load\npipe = load({str(cu_path)!r})\n\n"
                          "# %%\nimport threading, time\nimport matplotlib.pyplot as plt\n"
                          "where = threading.current_thread().name\nplt.plot([0, 1])\nplt.show()\n"
                          "while True:\n    time.sleep(0.01)\n\n"
                          "# %%\nafter = 1\n")
    nb.save(tmp_path / "long.ipynb")
    s = settings.Settings()
    s["python.run_notebooks"] = "never"
    w = MainWindow(settings=s)
    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
    try:
        view = w.open_notebook(str(tmp_path / "long.ipynb"))
        assert view.notebook.kernel is None
        monkeypatch.setattr(npn, "CELL_BUDGET", 0.1)
        t0 = time.monotonic()
        view.run_all()
        assert time.monotonic() - t0 < 5                       # run_all returned while the cell loops
        assert view.running and not view.interrupt_button.isHidden()
        assert not view.wait(0.3)                              # still running, events keep flowing
        view.interrupt()                                       # ■ Interrupt
        assert view.wait(30)
        loop, after = view.notebook.cells[1], view.notebook.cells[2]
        errors = [o.data for o in loop.outputs if o.kind == "error"]
        assert loop.state == "error" and errors and "CellInterrupted" in errors[0]
        assert any(o.kind == "image" for o in loop.outputs)    # pyplot drew off screen (Agg)
        assert after.state != "ok" and "after" not in view.notebook.namespace   # Run all stopped
        assert view.notebook.namespace["where"] != threading.main_thread().name
        assert view.interrupt_button.isHidden() and view.run_all_button.isEnabled()
        assert not view.notebook.interrupt()                   # nothing running now
        view.run(after)                                        # quick cells still finish in run()
        assert after.state == "ok" and view.notebook.namespace["after"] == 1
    finally:
        w.close()


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
    assert w.style.atom_style == "flat" and w.style.bonds_enabled and "bonds" in w.viewport.plotter.actors
    w.open_file(str(examples.path("Pt_nanoparticle.xyz")))  # a style picked for a big system keeps bonds off
    w.viewport.style_combo.setCurrentText("Ball & stick")
    w.viewport.style_combo.setCurrentText("Simple (ASE)")
    assert w.style.atom_style == "flat" and not w.style.bonds_enabled
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


def test_right_click_menu_and_right_drag_leave_the_view_out_of_zoom(app, monkeypatch):
    """After a right-click menu (or a right-drag zoom) every mouse move kept zooming: pyvista
    observes right releases on the interactor style, so VTK skipped the style's own EndDolly."""
    import sys

    from PySide6 import QtTest

    from ewald.data import Cell, DataCollection
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    v = Viewport(ViewStyle())
    v.resize(400, 300)
    v.show()
    v.ensure_view()
    v.set_data(DataCollection([[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]], ["O", "Ti"], Cell(np.diag([8.0] * 3))))
    app.processEvents()
    menus = []
    v.contextMenuRequested.connect(lambda: menus.append(1))
    w, style, p = v.plotter.interactor, v.plotter.iren.style, QtCore.QPoint(200, 150)
    try:
        QtTest.QTest.mousePress(w, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, p)
        assert style.GetState() == 4  # VTKIS_DOLLY while the button is down
        QtTest.QTest.mouseRelease(w, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, p)
        assert menus == [1] and style.GetState() == 0  # menu shown, zoom over
        QtTest.QTest.mousePress(w, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, p)
        QtGui.QCursor.setPos(w.mapToGlobal(p + QtCore.QPoint(0, 40)))  # the click test reads the cursor
        QtTest.QTest.mouseMove(w, p + QtCore.QPoint(0, 40))
        QtTest.QTest.mouseRelease(w, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, p + QtCore.QPoint(0, 40))
        assert menus == [1] and style.GetState() == 0  # a zoom drag, no menu
        QtGui.QCursor.setPos(w.mapToGlobal(p))
        # macOS: Control+click (Qt's MetaModifier there) opens the menu instead of picking
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                            staticmethod(lambda: QtCore.Qt.MetaModifier))
        QtTest.QTest.mouseClick(w, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, p)
        assert menus == [1, 1] and style.GetState() == 0 and not v.picks
    finally:
        _dispose(v)


def test_control_click_is_the_context_click_only_on_macos():
    from ewald.gui.viewport import control_click

    meta, ctrl = QtCore.Qt.MetaModifier, QtCore.Qt.ControlModifier
    assert control_click(meta, "darwin") and not control_click(ctrl, "darwin")
    assert not control_click(meta, "win32") and not control_click(ctrl, "win32")
    assert not control_click(meta, "linux") and not control_click(QtCore.Qt.NoModifier, "darwin")


def test_legacy_gl_view_right_click_and_macos_control_click_open_the_menu(monkeypatch):
    """The OpenGL 1.x view: a right click (or Control+click on macOS) asks for the menu, a right
    drag only zooms."""
    import sys
    from types import SimpleNamespace

    from ewald.gui.legacy_viewport import GLView

    menus, zooms = [], []
    viewport = SimpleNamespace(
        contextMenuRequested=SimpleNamespace(emit=lambda: menus.append(1)), hover_leave=lambda: None,
        zoom=zooms.append, click=lambda *a, **k: menus.append("pick"),
        plotter=SimpleNamespace(camera=None))
    view = SimpleNamespace(viewport=viewport, width=lambda: 400, height=lambda: 300, update=lambda: None)

    def event(kind, button, pos, mods=QtCore.Qt.NoModifier):
        return QtGui.QMouseEvent(kind, QtCore.QPointF(*pos), QtCore.QPointF(*pos), button, button, mods)

    def click(button, mods=QtCore.Qt.NoModifier, end=(10, 10)):
        GLView.mousePressEvent(view, event(QtCore.QEvent.MouseButtonPress, button, (10, 10), mods))
        if end != (10, 10):
            GLView.mouseMoveEvent(view, event(QtCore.QEvent.MouseMove, button, end, mods))
        GLView.mouseReleaseEvent(view, event(QtCore.QEvent.MouseButtonRelease, button, end, mods))

    click(QtCore.Qt.RightButton)
    assert menus == [1] and view._press is None
    click(QtCore.Qt.RightButton, end=(10, 60))  # a zoom drag, no menu
    assert menus == [1] and zooms
    monkeypatch.setattr(sys, "platform", "darwin")
    click(QtCore.Qt.LeftButton, QtCore.Qt.MetaModifier)
    assert menus == [1, 1]
    monkeypatch.setattr(sys, "platform", "win32")
    click(QtCore.Qt.LeftButton, QtCore.Qt.MetaModifier)
    assert menus == [1, 1, "pick"]


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
    assert not vp.caption.isVisibleTo(vp)   # the frame's timestep is in the timeline now (D63)
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
    panel.wait()
    fresh = [r for r in panel.findChildren(plugins_panel.PluginRow) if r.property("fresh")]
    assert [r.item.path.endswith("bar_shift.py") for r in fresh] == [True]     # scrolled to and marked
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


def test_plugins_from_file_sends_archives_to_pip_and_py_files_to_the_plugin_folder(app, tmp_path, monkeypatch):
    from ewald import settings
    from ewald.gui import plugins_panel

    panel = plugins_panel.PluginsPanel(settings=settings.Settings())
    panel.wait()
    (tmp_path / "one.py").write_text("x = 1\n")
    pip_calls, installs = [], []
    monkeypatch.setattr(panel, "_run_pip", lambda action, req: pip_calls.append((action, req)))
    monkeypatch.setattr(panel, "install", lambda entry, update=False: installs.append(entry))
    for name in ("repo-main.zip", "x-0.1-py3-none-any.whl", "x-0.1.tar.gz", "one.py"):
        monkeypatch.setattr(plugins_panel.QtWidgets.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, _n=name, **k: (str(tmp_path / _n), "")))
        panel.install_from_file()
    assert [r for _, r in pip_calls] == [str(tmp_path / n) for n in ("repo-main.zip", "x-0.1-py3-none-any.whl", "x-0.1.tar.gz")]
    assert len(installs) == 1 and installs[0].source.endswith("one.py")


def test_plugins_side_bar_stays_narrow_while_pip_runs(app, tmp_path):
    """A long pip requirement in the busy line, or a long plugin name, must not widen the side bar."""
    import json

    from ewald import settings
    from ewald.gui import plugins_panel

    index = tmp_path / "index.json"
    index.write_text(json.dumps({"format": 1, "plugins": [
        {"id": "long", "name": "A plugin with a very long name that goes on and on " * 3, "version": "1.0",
         "kinds": ["reader", "modifier", "visual", "overlay", "colormap", "theme", "tool"],
         "source": "pip:git+https://github.com/frssp/ewald-plugins@" + "0" * 40 + "#subdirectory=packages/ewald-long"}]}))
    s = settings.Settings()
    s["python.plugin_catalogs"] = str(index)
    panel = plugins_panel.PluginsPanel(settings=s)
    panel.wait()
    panel.resize(300, 600)
    panel.show()
    app.processEvents()
    narrow = panel.minimumSizeHint().width()
    assert narrow < 400
    req = "git+https://github.com/frssp/ewald-plugins@" + "0" * 40 + "#subdirectory=packages/ewald-long"
    panel._set_busy("long", "Installing…", f"Installing {req} with pip…")
    app.processEvents()
    assert panel.minimumSizeHint().width() <= narrow
    assert panel.busy_label.text().startswith("Installing git+") and req in panel.busy_label.label.toolTip()
    assert "…" in plugins_panel.QtWidgets.QLabel.text(panel.busy_label.label)    # what is drawn is shortened
    panel._set_busy("long")
    panel.close()


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
    assert panel.loading                                                 # a spinner, not "Nothing more to install"
    labels = [b.text() for b in panel.list_widget.findChildren(plugins_panel.BusyLabel)]
    assert labels == ["Loading plugin catalogs…"]
    panel.wait()
    assert not panel.loading
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
    buttons["Update to 0.3"].click()                                     # runs in the background
    app.sendPostedEvents(None, plugins_panel.QtCore.QEvent.DeferredDelete)
    spinning = [r for r in panel.findChildren(plugins_panel.PluginRow) if r.busy is not None]
    assert [(r.item.id, r.busy.text()) for r in spinning] == [("up_shift", "Updating…")]    # a spinner, no buttons
    assert not any(b.text().startswith("Update") for r in spinning for b in r.findChildren(plugins_panel.QtWidgets.QPushButton))
    assert not panel.busy_bar.isHidden() and "Updating Up shift" in panel.busy_label.text()
    panel.wait()
    assert not panel.busy and panel.busy_bar.isHidden()
    assert next(i for i in panel.items if i.id == "up_shift").version == "0.3"
    app.sendPostedEvents(None, plugins_panel.QtCore.QEvent.DeferredDelete)    # the old rows go
    assert not any(b.text().startswith("Update") for b in panel.findChildren(plugins_panel.QtWidgets.QPushButton))
    assert counts == [1, 0]                                              # the activity bar badge goes away
    panel.close()


_OUTLIVE_SCRIPT = r"""
import threading, time
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
from ewald import plugin_catalog as pc
from ewald.gui import live_reload, plugins_panel, workers

def flush():
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

# the Plugins panel goes while its catalog loader still reads the indexes
gate = threading.Event()
real = pc.catalog
pc.catalog = lambda urls, *a, **k: (gate.wait(5), real(urls))[1]
panel = plugins_panel.PluginsPanel()
panel.close(); panel.deleteLater(); flush()
gate.set(); panel = None

# a live reload goes while its probe still asks the source for frames
class Slow:
    def refresh(self):
        gate2.wait(5)
        return 0
class Pipe:
    source, num_frames = Slow(), 1
    def invalidate(self, first): pass
gate2 = threading.Event()
got = []
gone, kept = live_reload.LiveReload(), live_reload.LiveReload()
gone.set_pipeline(Pipe()); kept.set_pipeline(Pipe())
kept.grew.connect(lambda first, n: got.append(n))
gone.poll(); kept.poll()
gone.deleteLater(); flush()
gate2.set()
end = time.monotonic() + 10
while (workers.running() or not got) and time.monotonic() < end:
    QtCore.QThreadPool.globalInstance().waitForDone(50)
    flush()
assert got == [1], got               # the survivor still hears back, the deleted one is skipped
assert workers.running() == 0        # the finished loader was released
print("ok")
"""


def test_worker_threads_outlive_the_widgets_that_started_them(tmp_path):
    """Closing a panel while its background thread runs must not abort the process (a running
    QThread destroyed with its parent) or write into the freed object (a pool thread emitting on a
    deleted LiveReload): both crashed the GUI suite at random (issue #145)."""
    import subprocess
    import sys

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", EWALD_CONFIG_DIR=str(tmp_path), EWALD_PLUGIN_INDEX="")
    r = subprocess.run([sys.executable, "-c", _OUTLIVE_SCRIPT], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="viewport needs an X display with OpenGL (xvfb-run)")
def test_deleted_viewport_forgets_its_gl_surface(app, pipe):
    """pyvistaqt keeps a wrapper of the GL widget's surface, which Qt deletes with the widget. A
    QtInteractor object that outlives its widget must not keep it: PySide hands that stale wrapper
    out for whatever Qt allocates next at the address, and calls through it corrupt the heap
    (issue #145: random segfaults in menu building and widget teardown)."""
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    vp = Viewport(ViewStyle())
    vp.resize(120, 90)
    vp.show()
    vp.set_data(pipe.compute(0))
    app.processEvents()
    vp.ensure_view()                             # built after the first paint; realized on the next
    app.processEvents()
    plotter = vp.plotter                         # outlives the widget, like a console reference
    assert plotter._surface is not None          # realized: pyvistaqt holds the surface
    plotter.close()                              # what MainWindow.closeEvent does first
    vp.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert plotter._surface is None and plotter._ctx is None


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
    last = blob[offsets[-1]:]   # one more frame, at a later step (the same step again is a restart)
    head, _, rest = last.partition(b"ITEM: TIMESTEP\n")
    step, _, rest = rest.partition(b"\n")
    path.write_bytes(blob + head + b"ITEM: TIMESTEP\n" + str(int(step) + 1000).encode() + b"\n" + rest)
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
    from ewald.gui.pipeline_panel import DisplayPanel
    from ewald.gui.style import ViewStyle

    style = ViewStyle()
    panel = DisplayPanel(style)
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


def test_cell_panel_replicate(app):
    from ewald.gui.pipeline_panel import DisplayPanel
    from ewald.gui.style import ViewStyle

    style = ViewStyle()
    panel = DisplayPanel(style)
    box = panel._replicate_editor()
    assert not box.isChecked() and "replicate" not in style.scene_kwargs()
    box.setChecked(True)
    assert style.replicate == [2, 2, 2] and style.scene_kwargs()["replicate"] == [2, 2, 2]
    box.findChildren(QtWidgets.QSpinBox)[2].setValue(1)
    box.findChild(QtWidgets.QComboBox).setCurrentText("all")
    kw = style.scene_kwargs()
    assert kw["replicate"] == [2, 2, 1] and kw["replicate_cell"] == "all"
    assert style.view_settings()["replicate"] == [2, 2, 1]
    box.setChecked(False)
    assert "replicate" not in style.scene_kwargs() and style.view_settings()["replicate"] is None


def test_replicate_menu_writes_view_and_keeps_data(app, srtio3_pipe):
    from ewald.gui.app import MainWindow

    w = MainWindow(srtio3_pipe)
    n = w.data.count
    menu = w.fill_replicate_menu(QtWidgets.QMenu())
    labels = [a.text() for a in menu.actions()]
    assert "2 × 2 × 2" in labels and "Custom…" in labels
    next(a for a in menu.actions() if a.text() == "3 × 3 × 1").trigger()
    assert w.style.replicate == [3, 3, 1] and srtio3_pipe.view.replicate == [3, 3, 1]
    assert w.data.count == n                                  # display only
    assert w.viewport._boundary_atoms[0].shape[0] == 9 * n    # every copy hovers as its real atom
    assert w.viewport._boundary_atoms[1].max() == n - 1
    w.set_replicate(1)
    assert w.style.replicate == [] and srtio3_pipe.view.replicate is None
    assert any(a.text() == "Replicate cell" for a in w.viewport_menu().actions())
    w.close()


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
    w.addDockWidget(QtCore.Qt.RightDockWidgetArea, w.plugins_dock)
    w.addDockWidget(QtCore.Qt.RightDockWidgetArea, w.remotes_dock)
    w.plugins_dock.show()
    w.remotes_dock.show()
    w.tabifyDockWidget(w.plugins_dock, w.remotes_dock)
    app.processEvents()
    side = w.remotes_dock.titleBarWidget()                  # tabbed with Plugins: they fold together
    side.set_collapsed(True)
    assert all(d.maximumWidth() == STRIP for d in [w.remotes_dock] + w.tabifiedDockWidgets(w.remotes_dock))
    assert w.tabPosition(QtCore.Qt.RightDockWidgetArea) == QtWidgets.QTabWidget.East   # tabs turn sideways
    side.set_collapsed(False)
    assert w.tabPosition(QtCore.Qt.RightDockWidgetArea) == QtWidgets.QTabWidget.South

    view = w.panel.new_notebook()
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
    assert wb.titles("editor") == ["Structure"] and wb.titles("panel") == ["Python console", "View", "Data inspector", "Plots", "Log"]
    slot = wb.slot("viewport")
    assert vp.parent() is wb and vp.geometry() == QtCore.QRect(slot.mapTo(wb, QtCore.QPoint()), slot.size())

    bar = wb.tab_group(wb.group_of(w.plots)).bar          # a real mouse drag of the Plots tab
    start = bar.tabRect(3).center()
    QtTest.QTest.mousePress(bar, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
    QtTest.QTest.mouseMove(bar, start + QtCore.QPoint(40, 0))
    assert wb._drag is not None and not vp.isVisible()    # a snapshot stands in for the GL widget
    over = bar.mapFromGlobal(slot.mapToGlobal(slot.rect().center()))
    QtTest.QTest.mouseMove(bar, over)
    assert wb._drag["target"][1] == "center" and wb.marks.rect_ is not None
    QtTest.QTest.mouseRelease(bar, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, over)
    settle()
    assert wb.titles("editor") == ["Structure", "Plots"] and wb.is_shown(w.plots) and w.plots.data is not None
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
    assert [a.text().split("\t")[0] for a in menu.actions() if a.text()] == [
        "Split right", "Split down", "Move to panel", "Maximize group", "New plot tab", "Close"]
    acts = {a.text(): a for a in wb.actions()}
    assert acts["Move tab down"].shortcut() == QtGui.QKeySequence("Ctrl+Alt+Down")
    wb.active = g
    acts["Maximize group"].trigger()
    settle()
    assert wb.is_shown(p2) and not wb.is_shown(vp) and vp.parked
    assert not wb.panel_host.isVisible()                   # the panel makes room too
    acts["Maximize group"].trigger()
    assert wb.panel_host.isVisible()
    acts["Move tab down"].trigger()                        # from the editor area into the panel
    settle()
    assert wb.layout.area_of(wb.group_of(p2)) == "panel" and not vp.parked

    save_layout(w, s)
    w.close()
    w2 = MainWindow(pipe, settings=s)                       # the layout comes back, plot:2 included
    assert wb.layout.to_json() == w2.workbench.layout.to_json() and "plot:2" in w2.extra_plots
    w2.workbench.close_view("plot:2")
    assert "plot:2" not in w2.extra_plots and "plot:2" not in w2.workbench.specs
    w2.workbench.reset()
    assert w2.workbench.titles("editor") == ["Structure"]
    assert w2.workbench.titles("panel") == ["Python console", "View", "Data inspector", "Plots", "Log"]
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
    from ewald.gui.code_editor import PythonHighlighter

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
    w.panel.new_notebook()
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


def test_notebook_cell_controls_add_bar_and_running_mark(app):
    """Grip and fold show on hover or on the current cell, the add-cell buttons wrap instead of
    clipping at a narrow width, and a cell shows ``*`` while it runs."""
    from ewald.gui.app import MainWindow

    w = MainWindow()
    w.panel.new_notebook()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    view = w.panel.view
    cell = view.add_cell("code", "x = 1")
    cw = view.widget_for(cell)
    other = next(c for c in view.widgets if c is not cw and c.cell.kind == "code")
    other.set_current(False)
    assert other.drag_handle.isHidden() and other.fold_button.isHidden()
    app.sendEvent(other, QtGui.QEnterEvent(QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPointF(5, 5)))
    assert not other.drag_handle.isHidden() and not other.fold_button.isHidden()
    app.sendEvent(other, QtCore.QEvent(QtCore.QEvent.Leave))
    assert other.drag_handle.isHidden()

    bar = view.add_bar
    buttons = bar.findChildren(QtWidgets.QPushButton)
    for width in (900, 240):
        bar.resize(width, bar.layout().heightForWidth(width))
        bar.layout().setGeometry(QtCore.QRect(0, 0, width, bar.height()))
        assert all(b.geometry().right() <= width and b.width() >= b.sizeHint().width() for b in buttons)
    assert len({b.y() for b in buttons}) > 1                 # wrapped onto a second row

    cw.set_running()
    assert cw.counter.text() == "*" and cw.property("state") == "running"
    view.run(cell)
    cw = view.widget_for(cell)                               # a run may rebuild the cell widgets
    assert cw.counter.text() == str(cell.count) and cw.property("state") == "ok"

    printed = view.add_cell("code", "print('hi')")
    view.run(printed)
    assert not view.widget_for(printed).outputs.isHidden()
    view.select(printed)
    next(a for a in w.menuBar().findChildren(QtGui.QAction) if a.text() == "C&lear cell output").trigger()
    assert not printed.outputs and printed.count is None and view.widget_for(printed).counter.text() == ""
    assert view.widget_for(printed).outputs.isHidden() and cell.count is not None
    next(a for a in w.menuBar().findChildren(QtGui.QAction) if a.text() == "Clear all output&s").trigger()
    assert all(not c.outputs and c.count is None for c in view.notebook.cells)
    w.close()


def test_command_mode_cell_keys_like_jupyter(app, pipe):
    """Command mode (Esc): Shift+Down selects cells, C / X / V copy, cut and paste them, DD
    deletes and Z brings back, Shift+M merges, J / K move, Ctrl+Enter runs the selection."""
    from PySide6 import QtTest

    from ewald.gui import notebook_panel
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    view = w.panel.view
    nb = view.notebook
    a = view.add_cell("code", "a = 1")
    b = view.add_cell("code", "b = 2")
    c = view.add_cell("code", "c = 3")
    key = lambda cell, k, mods=QtCore.Qt.NoModifier: QtTest.QTest.keyClick(view.widget_for(cell), k, mods)  # noqa: E731
    sources = lambda: [x.source for x in nb.cells][-3:]  # noqa: E731

    view.select(a)
    key(a, QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier)
    assert view.current is b and view.selected_cells == [a, b]
    assert view.widget_for(a).property("selected") and not view.widget_for(c).property("selected")
    key(b, QtCore.Qt.Key_C)
    assert [e[2] for e in notebook_panel.CELL_CLIPBOARD] == ["a = 1", "b = 2"]
    assert QtWidgets.QApplication.clipboard().text() == "a = 1\n\nb = 2\n"
    view.select(c)
    key(c, QtCore.Qt.Key_V)
    assert sources() == ["a = 1", "b = 2", "c = 3", "a = 1", "b = 2"][-3:] and len(nb.cells) >= 5
    assert [x.source for x in view.selected_cells] == ["a = 1", "b = 2"]   # the pasted cells
    assert view.history.undo_label == "Paste cells"

    view.select(c)
    key(c, QtCore.Qt.Key_D)
    key(c, QtCore.Qt.Key_D)                                   # DD
    assert c not in nb.cells and view.history.undo_label == "Delete cell"
    n = len(nb.cells)
    key(view.current, QtCore.Qt.Key_Z)
    assert len(nb.cells) == n + 1 and view.current.source == "c = 3"
    c = view.current

    key(c, QtCore.Qt.Key_K)                                   # K: the cell above
    assert view.current is b and view.anchor is b
    key(b, QtCore.Qt.Key_X)                                   # X: cut
    assert b not in nb.cells and notebook_panel.CELL_CLIPBOARD[0][2] == "b = 2"
    view.select(a)
    key(a, QtCore.Qt.Key_V, QtCore.Qt.ShiftModifier)          # Shift+V: paste above
    i = nb.cells.index(a)
    assert nb.cells[i - 1].source == "b = 2"

    view.select(a)
    key(a, QtCore.Qt.Key_M, QtCore.Qt.ShiftModifier)          # Shift+M: merge with the cell below
    assert view.current.source == "a = 1\n\nc = 3" and view.history.undo_label == "Merge cells"

    first = view.current
    view.select(nb.cells[nb.cells.index(first) - 1])
    view.select(first, extend=True)                           # two cells selected
    key(first, QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)
    assert all(x.state == "ok" and x.count for x in view.selected_cells)
    assert nb.namespace["c"] == 3 and nb.namespace["b"] == 2

    view.select(first)                                        # the ⋯ menu's Move up moves the selection
    view.select(nb.cells[nb.cells.index(first) - 1], extend=True)
    block = view.selected_cells
    k = nb.cells.index(block[0])
    view._cell_action(block[0], "up")
    assert nb.cells[k - 1:k + 1] == block
    w.close()


def test_cell_type_switches_and_add_bar_like_jupyter(app, pipe):
    """+ Code, + Text, + Step (with Plugins), + View under the cells; Esc then M / Y turns a cell
    into text / code, A / B insert; the menu and the cell's ⋯ menu do the same; Ask AI about this cell."""
    from PySide6 import QtTest

    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    view = w.panel.view
    texts = [b.text() for b in view.add_bar.findChildren(QtWidgets.QPushButton)]
    assert texts == ["+ Code", "+ Text", "+ Step", "+ Plot"]
    view.step_menu.aboutToShow.emit()
    plugins = next(a for a in view.step_menu.actions() if a.text() == "Plugins").menu()
    assert any(a.text() == "Load plugin file…" for a in plugins.actions())

    cell = view.add_cell("code", "# Title")
    cw = view.widget_for(cell)
    QtTest.QTest.keyClick(cw.editor, QtCore.Qt.Key_Escape)
    assert view.current is cell                              # focus moved to the cell (command mode)
    QtTest.QTest.keyClick(cw, QtCore.Qt.Key_M)
    assert cell.kind == "markdown" and cell.source == "# Title"
    cw = view.widget_for(cell)
    assert not cw.view.isHidden() and cw.editor.isHidden()
    QtTest.QTest.keyClick(cw, QtCore.Qt.Key_Y)
    assert cell.kind == "code" and cell.state == "new"
    n = len(view.notebook.cells)
    QtTest.QTest.keyClick(view.widget_for(cell), QtCore.Qt.Key_B)
    i = view.notebook.cells.index(cell)
    assert len(view.notebook.cells) == n + 1 and view.current is view.notebook.cells[i + 1]
    view.select(cell)
    QtTest.QTest.keyClick(view.widget_for(cell), QtCore.Qt.Key_A)
    assert view.notebook.cells.index(cell) == i + 1 and view.current is view.notebook.cells[i]

    view.select(cell)
    acts = {a.text(): a for a in w.menuBar().findChildren(QtGui.QAction)}
    acts["Change cell to &text"].trigger()
    assert cell.kind == "markdown"
    acts["Change cell to c&ode"].trigger()
    assert cell.kind == "code"
    assert view.history.undo_label == "Change cell type"

    cw = view.widget_for(cell)
    menu = cw.findChild(QtWidgets.QToolButton, "cellMenu").menu()
    names = [a.text() for a in menu.actions()]
    assert "Ask AI about this cell" in names and names.index("Cell type") < names.index("Ask AI about this cell")
    assert names[-1] == "Delete\tD, D"
    next(a for a in menu.actions() if a.text() == "Ask AI about this cell").trigger()
    app.processEvents()
    assert w.ask_dock.isVisible() and "# Title" in w.ask_panel.input.toPlainText()
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
    next(a for a in view_menu.actions() if a.text().replace("&", "").startswith("Minimal mode")).trigger()
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


@pytest.mark.parametrize("backend", ["vtk", "legacy-gl"])
def test_double_click_on_empty_space_or_esc_clears_picks(app, pipe, monkeypatch, backend):
    """After picking or measuring, a real double click on empty space (any tool, orbit included)
    or Esc clears the picks and the pick readout, in the VTK view and in the legacy GL view;
    a double click on an atom keeps them."""
    import time

    from PySide6 import QtTest

    from ewald.gui.app import MainWindow

    if backend == "legacy-gl":
        monkeypatch.setenv("EWALD_VIEWPORT", "legacy-gl")
    w = MainWindow(pipe)
    w.resize(1100, 800)
    w.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    vp = w.viewport
    assert getattr(vp, "backend", "vtk") == backend
    view = vp.view if backend == "legacy-gl" else vp.plotter.interactor
    W, H = view.width(), view.height()
    points, empty = {}, None
    for y in range(10, H - 10, 6):
        for x in range(10, W - 10, 6):
            i = vp.pick_at(x, y) if backend == "legacy-gl" else vp.hover_index(x, y, W, H)
            if i is None:
                empty = empty or QtCore.QPoint(x, y)
            elif i not in points:
                points[i] = QtCore.QPoint(x, y)
    atoms = list(points)[:2]
    assert len(atoms) == 2 and empty is not None

    def click(point, double=False):
        # a real double click: press, release, double-click press, release
        QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
        if double:
            QtTest.QTest.mouseDClick(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
            QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
        app.processEvents()

    vp.set_tool("distance")
    click(points[atoms[0]])
    click(points[atoms[1]])
    assert vp.picks == atoms and "d(" in w.pick_label.text()
    click(empty)                                   # one click on empty space keeps them
    assert vp.picks == atoms
    time.sleep(QtWidgets.QApplication.doubleClickInterval() / 1000 + 0.05)
    click(points[atoms[1]], double=True)           # a double click on an atom keeps them
    assert atoms[1] in vp.picks
    vp.set_picks(atoms)
    click(empty, double=True)
    assert vp.picks == [] and w.pick_label.text() == ""
    vp.set_tool("orbit")                           # also while rotating, e.g. after a plot selection
    vp.set_picks(atoms)
    click(empty, double=True)
    assert vp.picks == []
    vp.set_picks(atoms)
    if backend == "legacy-gl":
        view.requestActivate()
    else:
        w.activateWindow()
        view.setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Escape)
    app.processEvents()
    assert vp.picks == []
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
    pal = next(a for a in acts if a.text() == "Command &palette")
    assert pal.shortcut() == QtGui.QKeySequence("Ctrl+K")
    assert texts.count("AI chat") == 1 and "Ask AI about selection…" not in texts
    assert next(a for a in acts if a.text() == "&AI chat").shortcut() == QtGui.QKeySequence("Ctrl+J")
    assert "Restart and run all" not in texts and "Load plugin file…" not in texts
    assert "Dark background" not in [a.text() for a in w.menuBar().actions()[2].menu().actions()]
    assert "Structure theme" in texts and "Theme" not in texts
    w.step_menu.aboutToShow.emit()                       # one frame of Cu: no trajectory steps
    steps = {a.text(): a for sub in w.step_menu.actions() if sub.menu() for a in sub.menu().actions()}
    assert not steps["Mean squared displacement / diffusion"].isEnabled()
    assert "trajectory" in steps["Mean squared displacement / diffusion"].toolTip()
    assert not steps["Isosurface"].isEnabled() and steps["Common neighbor analysis"].isEnabled()
    w.close()


def test_menu_ux_review(app, cu_fcc):
    """2026-10-23 menu review: Insert › Step › Plugins and the palette work before + Step was ever
    opened; Tools ticks the viewport tool in use; View › Orthographic follows the view; Window
    groups its entries; Notebook runs and interrupts the current cell; the palette shows shortcuts."""
    from PySide6 import QtGui

    from ewald.gui.app import MainWindow
    from ewald.gui.palette import menu_entries

    w = MainWindow(Pipeline(StaticSource([cu_fcc])))
    w.panel.new_notebook()
    w.step_menu.aboutToShow.emit()                # raised AttributeError before + Step was opened
    plug = next(a.menu() for a in w.step_menu.actions() if a.text() == "Plugins")
    assert "Load plugin file…" in [a.text() for a in plug.actions()]
    w.viewport.set_tool("distance")
    w.tools_menu.aboutToShow.emit()
    ticked = [a.data() for a in w.tools_menu.actions() if a.isChecked()]
    assert ticked == ["distance"]
    next(a for a in w.tools_menu.actions() if a.data() == "pick").trigger()
    assert w.viewport.tool == "pick"
    w.menuBar().actions()[2].menu().aboutToShow.emit()   # View: the tick follows the view
    assert w.ortho_action.isChecked() is w.viewport.ortho is True
    w.ortho_action.setChecked(False)
    assert w.viewport.ortho is False
    w.ortho_action.setChecked(True)
    assert w.viewport.ortho is True
    window = [a.text() for a in w.window_menu.actions()]
    assert window[2] == "" and window.index("&AI chat") < window.index("&Structure")
    notebook = [a.text().replace("&", "") for a in w.notebook_menu.actions()]
    assert "Run current cell" in notebook and "Interrupt" in notebook
    entries = {label: where for label, where, _ in menu_entries(w.menuBar())}
    assert entries["Command palette"].endswith(QtGui.QKeySequence("Ctrl+K").toString(QtGui.QKeySequence.NativeText))
    w.close()


def test_menu_reorganization(app, cu_fcc, tmp_path, monkeypatch):
    """The menu bar after the 2026-10-23 review (decisions D-menu): File › Open recent / Open from /
    Export, cell editing in Edit, Look along and Show in View, Quick analysis and Plot in Insert,
    side bars ticked in Window (Remotes with Ctrl+Alt+R), Help › Keyboard shortcuts, the
    Preferences pages, shortcuts written the macOS way and the palette's recent commands first."""
    from PySide6 import QtGui

    from ewald.gui import palette, welcome
    from ewald.gui.app import MainWindow
    from ewald.gui.palette import CommandPalette, menu_entries

    data = tmp_path / "POSCAR"
    data.write_text("x", encoding="utf-8")
    welcome.add_recent(str(data))
    w = MainWindow(Pipeline(StaticSource([cu_fcc])))
    menus = {a.text().replace("&", ""): a.menu() for a in w.menuBar().actions()}
    assert list(menus) == ["File", "Edit", "View", "Notebook", "Insert", "Tools", "Window", "Help"]

    def labels(menu):
        return [a.text().replace("&", "") for a in menu.actions() if a.text()]

    def sub(menu, title):
        return next(a.menu() for a in menu.actions() if a.menu() is not None and a.text().replace("&", "") == title)

    file_ = labels(menus["File"])
    assert {"Open recent", "Open from", "Export", "Import session (.ewald)…"} <= set(file_)
    assert "Remote computers (SSH profiles)…" not in file_ and "Session (.ewald)" not in file_
    new = next(a for a in menus["File"].actions() if a.text() == "&New notebook")
    assert QtGui.QKeySequence("Ctrl+N") in new.shortcuts() and QtGui.QKeySequence("Ctrl+T") in new.shortcuts()
    recent = sub(menus["File"], "Open recent")
    recent.aboutToShow.emit()
    assert labels(recent)[0] == "POSCAR" and "Clear recent files" in labels(recent)
    assert labels(sub(menus["File"], "Open from")) == ["Remote file (SSH)…", "Location…", "NEB folder…",
                                                       "Workflow folder…"]
    assert labels(sub(menus["File"], "Export"))[:3] == ["Image…", "Data…", "Movie…"]
    edit = labels(menus["Edit"])
    assert {"Cut cells", "Copy cells", "Paste cells below", "Delete cells", "Merge cells"} <= set(edit)
    assert "Plugin settings" not in edit
    assert not {"Cut cells", "Delete cells"} & set(labels(menus["Notebook"]))
    view = labels(menus["View"])
    assert view[:2] == ["Look along", "Orthographic"] and "Structure theme" in view and "Full screen" in view
    assert "View along a" not in view and "Fold all View cards" in labels(sub(menus["View"], "Layout"))
    assert labels(sub(menus["View"], "Show"))[:2] == ["Bonds", "Simulation cell"]
    insert = labels(menus["Insert"])
    assert {"Plot", "Quick analysis", "Cell above", "Cell below"} <= set(insert)
    assert "View" not in insert and "Analysis" not in insert and "Plugin step" not in insert
    assert "Command palette" in labels(menus["Tools"]) and "Command palette" not in labels(menus["Window"])
    window = labels(menus["Window"])
    assert window.index("Notebook") < window.index("AI chat") < window.index("Remotes") < window.index("Plugins")
    remotes = w.remotes_action
    assert remotes.shortcut() == QtGui.QKeySequence("Ctrl+Alt+R")
    remotes.trigger()
    menus["Window"].aboutToShow.emit()
    assert remotes.isChecked() and not w.remotes_dock.isHidden()
    remotes.trigger()
    assert w.remotes_dock.isHidden()
    help_ = labels(menus["Help"])
    assert help_[0] == "Tutorials" and "Keyboard shortcuts" in help_
    dlg = w.show_shortcuts()
    rows = dlg.rows()
    assert any(command == "Command palette" for _where, command, _keys in rows)
    assert any(where.startswith("Command mode") and command.startswith("Cut") for where, command, _keys in rows)
    dlg._filter("palette")
    assert dlg.tree.topLevelItem(0).isHidden() or "palette" in dlg.tree.topLevelItem(0).text(1).lower()
    dlg.close()
    w.analysis_menu.aboutToShow.emit()
    assert "table" in next(a for a in w.analysis_menu.actions() if a.data() == "rdf").toolTip()

    # shortcuts named in tooltips follow the platform
    monkeypatch.setattr(palette.sys, "platform", "darwin")
    monkeypatch.setattr(palette, "native_keys", lambda k: "<" + k + ">")
    assert palette.native_text("Run all (Ctrl+Shift+Enter)") == "Run all (<Ctrl+Shift+Enter>)"
    monkeypatch.setattr(palette.sys, "platform", "win32")
    assert palette.native_text("Run all (Ctrl+Shift+Enter)") == "Run all (Ctrl+Shift+Enter)"
    monkeypatch.undo()

    # the palette lists the commands run last first when nothing is typed
    monkeypatch.setattr(palette, "RECENT", [])
    entries = menu_entries(w.menuBar())
    pal = CommandPalette(entries, w)
    pal.edit.setText("zoom to fit")
    ran = []
    monkeypatch.setattr(pal, "entries", [(l, k, (lambda l=l: ran.append(l))) for l, k, _ in entries])
    pal._run()
    pal = CommandPalette(entries, w)
    assert pal.list.item(0).text().startswith("Zoom to fit") and palette.RECENT == ["Zoom to fit"]
    w.close()


def test_search_box_centered_and_step_button(app, cu_fcc, monkeypatch):
    """The search box sits in the middle of the window and the palette opens right below it; the
    notebook bar's + Step lists the steps to search and adds the picked one after the current
    cell; in minimal mode the Menus button keeps the bottom strip to itself."""
    from ewald.gui import palette
    from ewald.gui.app import MainWindow

    assert palette.fuzzy_score("msd", "Mean squared displacement / diffusion") > \
        palette.fuzzy_score("msd", "Combine datasets (add particles from a file)")
    w = MainWindow(Pipeline(StaticSource([cu_fcc])))
    w.resize(1400, 900)
    w.show()
    w.panel.new_notebook()
    app.processEvents()
    box = w.command_center.box
    center = box.mapTo(w, QtCore.QPoint(box.width() // 2, 0)).x()
    assert abs(center - w.width() // 2) <= 2 and box.width() == palette.CommandCenter.WIDTH
    opened = []
    monkeypatch.setattr(palette.CommandPalette, "exec", lambda self: opened.append(self))
    w.command_palette()
    pal = opened.pop()
    top_left = box.mapToGlobal(QtCore.QPoint(0, box.height()))
    assert pal.y() == top_left.y() + 2 and abs(pal.x() + pal.width() // 2 - (top_left.x() + box.width() // 2)) <= 1
    w.resize(700, 900)    # little room: the box moves right of the buttons, never over them
    app.processEvents()
    assert box.mapTo(w, QtCore.QPoint(0, 0)).x() >= w.command_center.mapTo(w, QtCore.QPoint(0, 0)).x()

    view = w.panel.view
    current = view.current
    view.step_button.click()
    pal = opened.pop()
    labels = [label for label, _kind, _fn in pal.entries]
    assert "Common neighbor analysis" in labels and "Mean squared displacement / diffusion" not in labels  # one frame
    assert pal.edit.placeholderText() == "Search steps…"
    pal.edit.setText("common neighbor")
    pal._run()
    assert "CommonNeighborAnalysis" in view.current.source
    index = view.notebook.cells.index(view.current)
    assert current is None or view.notebook.cells.index(current) == index - 1

    w.set_minimal(True)
    app.processEvents()
    assert w.contentsMargins().bottom() >= w.minimal_exit.height()
    w.set_minimal(False)
    assert w.contentsMargins().bottom() == 0
    w.close()


def test_preferences_pages_after_menu_review(app):
    """Preferences: 3D view, Atoms & bonds, Plots, Notebooks and Plugins pages; old page names
    still open the right page; the AI page starts with the provider."""
    from ewald import settings
    from ewald.gui.preferences import LAYOUT, PreferencesDialog

    assert settings.PAGES == ("General", "Codes", "3D view", "Atoms & bonds", "Rendering", "Plots", "Notebooks",
                              "Compute", "Python", "AI", "Plugins")
    assert settings.OPTION["python.plot_style"].page == "Plots" == settings.OPTION["plots.size"].page
    assert settings.OPTION["python.run_notebooks"].page == "Notebooks"
    assert settings.OPTION["python.plugin_catalogs"].page == "Plugins"
    assert settings.OPTION["viewport.target_fps"].page == "3D view"
    assert [title for title, _keys in LAYOUT["AI"][1]][:2] == ["Provider", "Connection"]
    dlg = PreferencesDialog(settings.Settings(), page="Viewports")
    assert dlg.pages.currentItem().text() == "3D view"
    dlg.show_page("Elements")
    assert dlg.pages.currentItem().text() == "Atoms & bonds"
    dlg.close()


def test_side_bar_and_tab_menus_after_menu_review(app, tmp_path):
    """The Plugins side bar puts its install choices in a menu; a notebook tab can close the
    others and show its file's folder."""
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook

    w = MainWindow()
    buttons = [b.text() for b in w.plugins_panel.findChildren(QtWidgets.QToolButton) if b.text()]
    assert "Install" in buttons and "Create with AI…" in buttons and "pip…" not in buttons
    install = next(b for b in w.plugins_panel.findChildren(QtWidgets.QToolButton) if b.text() == "Install")
    assert [a.text() for a in install.menu().actions()] == ["From file…", "With pip…"]
    path = tmp_path / "a.py"
    Notebook().save(str(path))
    w.open_notebook(str(path), run=False)
    w.panel.new_notebook()
    tabs = w.panel.tabs
    assert tabs.count() == 2
    tabs.close_others(1)
    QtWidgets.QApplication.processEvents()
    assert tabs.count() == 1
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
    assert acts["Move tab down"].isEnabled()            # into the panel
    wb.enable_actions()
    assert acts["Split right"].isEnabled()
    groups = len(wb.layout.groups())
    acts["Split right"].trigger()                        # the shortcut path checks too: no change
    assert len(wb.layout.groups()) == groups
    w.close()


def test_render_page_frame_on_the_3d_view(app, pipe, tmp_path):
    """The Render page shows the live 3D view with the picture's frame; dragging the frame sets the
    region (written to pipe.view), a side changes the picture size, Fit zooms the view into the frame."""
    from ewald.framing import frame_rect
    from ewald.gui.app import MainWindow
    from ewald.gui.render_frame import drag_rect, hit_test, keep_inside

    w = MainWindow(pipe)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    page, frame = w.render_page, w.render_frame
    assert frame is not None and not frame.visible
    w.set_workspace(1)
    app.processEvents()
    assert frame.visible and w.viewport.isVisible()
    assert w.viewport.geometry().size() == page.live_slot.size()      # the viewport sits in the page
    assert frame.aspect == pytest.approx(page.aspect)
    rect = frame.rect()
    assert hit_test(rect, rect[0], rect[1]) == "nw" and hit_test(rect, rect[0] + rect[2] / 2, rect[1]) == "n"
    assert hit_test(rect, rect[0] + rect[2] / 4, rect[1] + rect[3]) == "move"
    assert hit_test(rect, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) is None  # inside: turns the view

    def drag(part_pos, delta):
        widget = frame.widget
        x0, y0 = part_pos
        for kind, x, y, buttons in ((QtCore.QEvent.MouseButtonPress, x0, y0, QtCore.Qt.LeftButton),
                                    (QtCore.QEvent.MouseMove, x0 + delta[0], y0 + delta[1], QtCore.Qt.LeftButton),
                                    (QtCore.QEvent.MouseButtonRelease, x0 + delta[0], y0 + delta[1], QtCore.Qt.NoButton)):
            e = QtGui.QMouseEvent(kind, QtCore.QPointF(x, y), QtCore.QPointF(x, y), QtCore.Qt.LeftButton, buttons,
                                  QtCore.Qt.NoModifier)
            assert frame.eventFilter(widget, e)

    camera = tuple(w.viewport.plotter.camera.position)
    drag((rect[0] + rect[2], rect[1] + rect[3]), (-40, -10))        # se corner: smaller, same aspect
    assert tuple(w.viewport.plotter.camera.position) == camera      # the view did not turn
    assert frame.region is not None and frame.region[2] < 0.92
    assert frame.aspect == pytest.approx(page.aspect)
    assert w.pipeline.view.region == pytest.approx(list(frame.region), abs=1e-3)   # into the notebook view
    kw = page.render_kwargs()
    assert kw["region"] == pytest.approx(frame.region, abs=1e-3)
    width = page.width.value()
    rect = frame.rect()
    drag((rect[0] + rect[2], rect[1] + rect[3] / 2), (30, 0))      # east side: a wider picture
    assert page.preset.currentText() == "Custom (px)" and page.width.value() > width
    assert frame.aspect == pytest.approx(page.aspect, rel=0.02)
    assert keep_inside((-50, 10, 100, 80), (400, 300), "move")[0] == 0
    assert drag_rect((0, 0, 100, 50), "se", 20, 0, 2.0)[0][2:] == pytest.approx((120, 60))

    before = w.viewport.plotter.camera.parallel_scale
    page.fit_frame()
    assert w.viewport.plotter.camera.parallel_scale != before
    page.output.setText(str(tmp_path / "f.png"))
    page.quality.setCurrentText("Draft")
    page.set_renderer("opengl")
    w.render_from_page()
    w.wait_render()
    assert os.path.getsize(tmp_path / "f.png") > 1000 and page.left_stack.currentIndex() == 1
    assert not frame.visible                                         # the picture is shown, not the view
    assert "region=" in page.code.toPlainText()
    page.show_live(True)
    assert frame.visible
    state = page.state()
    page.set_region(None, write=True)
    assert frame.region is None and "region" not in w.pipeline.view.changed()
    page.set_state(state)
    assert frame.region == pytest.approx(state["region"])
    w.pipeline.view.update(region=(0.0, 0.0, 0.5))                   # a view cell sets the frame
    w.recompute()
    app.processEvents()
    assert frame.region == pytest.approx((0.0, 0.0, 0.5))
    w.set_workspace(0)
    assert not frame.visible
    w.settings["viewport.render_frame"] = True
    w._render_frame_visibility()
    assert frame.visible
    w.close()


def test_compute_device_lives_in_preferences(app, pipe):
    """No toolbar device switch: Preferences › Compute › Device (CPU by default) sets the window's
    device and the notebook pipe's, unless the pipe chose one itself."""
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.gui.preferences import apply_settings

    s = settings.Settings()
    w = MainWindow(pipe, settings=s)
    assert not hasattr(w, "device_combo") and w.device == "cpu" and w.pipeline.device == "cpu"
    s["compute.device"] = "auto"
    apply_settings(w, s)
    assert w.device == "auto" and w.pipeline.device == "auto"
    w.pipeline.device = "cuda"                      # chosen in a cell: kept
    s["compute.device"] = "cpu"
    apply_settings(w, s)
    assert w.device == "cpu" and w.pipeline.device == "cuda"
    w.close()


def test_ai_chat_does_not_insert_twice_when_the_model_wrote_cells(app, pipe):
    from ewald.gui.app import MainWindow

    class ToolAgent(_FakeAgent):
        def ask(self, question):
            self.on_event("tool_call", {"name": "insert_cell", "input": {}})
            return "I added this cell:\n\n```python\nx = 1\n```"

    w = MainWindow(pipe)
    w.ask_panel.agent_factory = lambda p, ev: ToolAgent(p, ev)
    n = len(w.panel.view.notebook.cells)
    w.ask_panel.ask("Add x")
    _wait(app, w.ask_panel)
    assert len(w.panel.view.notebook.cells) == n           # the tool call wrote it; the block only shows it
    w.close()


def test_ai_chat_links_the_cells_its_tools_wrote(app, pipe):
    """A tool that wrote a cell gets a link next to its chip ("Edited cell 2"): clicking it shows that
    notebook's tab and the cell (current, lit up), the link is saved with the chat (.ewd) and comes
    back, and a deleted cell's link is only a label."""
    from PySide6 import QtWidgets

    from ewald.gui.app import MainWindow
    from ewald.gui.ask_panel import Message
    from ewald.llm.notebook_tools import cell_summary
    from ewald.notebook_io import chat_messages

    w = MainWindow(pipe)
    view = w.panel.view
    target = view.add_cell("code", "x = 1")
    k = view.notebook.cells.index(target)
    summary = cell_summary(view.notebook, k)

    class ToolAgent(_FakeAgent):
        def ask(self, question):
            self.on_event("tool_call", {"name": "edit_cell", "input": {"index": k}})
            self.on_event("tool_result", {"name": "edit_cell", "result": {"edited": summary}})
            self.on_event("tool_call", {"name": "delete_cell", "input": {"index": 0}})
            self.on_event("tool_result", {"name": "delete_cell",
                                          "result": {"deleted": {"index": 0, "id": "gone"}, "cells": 3}})
            return "Done."

    w.ask_panel.agent_factory = lambda p, ev: ToolAgent(p, ev)
    w.ask_panel.ask("Change x")
    _wait(app, w.ask_panel)
    answer = w.ask_panel.messages[-1]
    links = answer.findChildren(QtWidgets.QToolButton, "chatCellLink")
    assert [b.text() for b in links] == [f"Edited cell {k + 1}", "Deleted cell 1"]
    assert links[0].isEnabled() and not links[1].isEnabled()
    saved = answer.to_dict()["cells"]
    assert saved[0] == {"step": 0, "id": target.id, "index": k, "verb": "Edited"} and saved[1]["step"] == 1

    other = w.panel.new_notebook()                      # the link leads back to the first tab
    w.ask_panel.sync()
    assert w.panel.view is other
    answer.cell_links[0].click()
    app.processEvents()
    assert w.panel.view is view and view.current is target
    assert view.widget_for(target).property("flash")

    entry = chat_messages([answer.to_dict(), {"role": "assistant", "text": "x",
                                              "cells": [{"step": -1, "index": 0, "verb": "Ran"}, "junk"]}])
    assert entry[0]["cells"] == saved and "cells" not in entry[1]
    again = Message.from_dict(entry[0])
    assert [b.text() for b in again.cell_links] == [f"Edited cell {k + 1}", "Deleted cell 1"]
    view.notebook.cells.remove(target)                  # gone since: a message instead of a jump
    w._show_ai_cell(view.notebook, target.id)
    w.close()


def test_opened_md_run_gets_a_run_check_and_runs_get_a_picker(app, lammps_dir, tmp_path):
    """Opening an MD run whose log says so adds a run check cell (run at once); a notebook that
    loads several runs shows a run picker that binds pipe, and with it the 3D view, to one run."""
    from ewald import import_file
    from ewald.gui.app import MainWindow

    w = MainWindow(import_file(lammps_dir / "thermo" / "dump.lammpstrj"))
    view = w.panel.view
    check = view.notebook.cells[-1]
    assert check.source.endswith("check = check_run(pipe)\ncheck") and check.state == "ok"
    assert "Run check: LAMMPS md NVE" in check.outputs[-1].data
    assert not view.run_bar.isVisible() and view._runs is None
    labels = [a.text() for a in w.analysis_menu.actions()]
    assert labels[0] == "Run check (convergence, equilibration)"
    for k in (1, 2):
        d = tmp_path / f"run{k}"
        d.mkdir()
        (d / "dump.lammpstrj").write_text((lammps_dir / "thermo" / "dump.lammpstrj").read_text())
    cell = view.add_cell("code", f"from ewald import load_runs\nruns = load_runs({str(tmp_path / 'run*' / 'dump.lammpstrj')!r})\n"
                                 "pipe = runs.pipe")
    view.run(cell)
    view.wait()
    assert not view.run_bar.isHidden() and view.run_combo.count() == 2
    runs = view.notebook.namespace["runs"]
    assert w.pipeline is runs[0].pipe and view.run_combo.currentText() == "run1"
    view._pick_run(1)
    assert w.pipeline is runs[1].pipe and view.notebook.namespace["pipe"] is runs[1].pipe and runs.current == 1
    view.restart()
    assert view.run_bar.isHidden()
    w.close()


def test_ai_chat_stop_button_and_failed_cells_in_the_question(app, pipe):
    import threading

    from PySide6 import QtCore, QtGui

    from ewald.gui.app import MainWindow
    from ewald.llm.agent import Cancelled

    started, prompts = threading.Event(), []

    class SlowAgent(_FakeAgent):
        def ask(self, question, cancel=None):
            prompts.append(question)
            if question.endswith("slow"):
                started.set()
                cancel.wait(10)                       # the request on the wire, until Stop
                raise Cancelled("stopped")
            return "fast"

    w = MainWindow(pipe)
    w.settings["llm.code_cells"] = "button"
    panel = w.ask_panel
    panel.agent_factory = lambda p, ev: SlowAgent(p, ev)
    view = w.panel.view
    cell = view.add_cell("code", "1 / 0")
    view.run(cell)
    view.wait()
    panel.ask("slow")
    assert started.wait(5) and panel.busy() and panel.ask_button.toolTip() == "Stop (Esc)"
    assert "These notebook cells failed" in prompts[0] and "ZeroDivisionError" in prompts[0]
    esc = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
    app.sendEvent(panel.input, esc)                    # Esc stops, like the Stop button
    assert not panel.busy() and panel.ask_button.toolTip() == "Send (Enter)"
    assert panel.messages[-1].text == "*Stopped.*"
    panel.ask("again")                                 # the chat is free at once
    _wait(app, panel)
    assert panel.messages[-1].text == "fast"
    assert "ZeroDivisionError" in prompts[1]           # the stopped question's context is sent again
    view.notebook.cells.remove(cell)
    panel.ask("and now")
    _wait(app, panel)
    assert "no longer show errors" in prompts[2]
    for worker in panel._stopped:
        worker.wait(5000)
    w.close()


def test_ai_chat_streams_the_answer(app, pipe):
    """A streaming agent's text shows while it arrives; the text of a round that went on with a tool
    stays above the tool's chip, and the final answer replaces the streamed one."""
    import threading
    import time

    from PySide6 import QtWidgets

    from ewald.gui.ask_panel import AskPanel

    go = threading.Event()

    class StreamingAgent(_FakeAgent):
        def ask(self, question):
            for piece in ("Let me ", "look."):
                self.on_event("text", piece)
            self.on_event("tool_call", {"name": "describe_frame", "input": {}})
            self.on_event("text", "It is ")
            go.wait(5)                                 # the rest of the answer is still on its way
            self.on_event("text", "fcc.")
            return "It is fcc."

    panel = AskPanel(lambda: (pipe, 0, []), agent_factory=lambda p, ev: StreamingAgent(p, ev))
    panel.ask("What is it?")
    answer = panel.messages[-1]
    t = time.time()
    while (answer.live is None or answer.live.text() != "It is ") and time.time() - t < 5:
        app.processEvents()
    assert answer.live.text() == "It is " and panel.busy()   # streamed so far, drawn as it came
    narration = answer.findChildren(QtWidgets.QLabel, "chatNarration")
    assert [n.text() for n in narration] == ["Let me look."]
    go.set()
    _wait(app, panel)
    assert answer.live is None and answer.text == "It is fcc." and answer.tools == ["describe_frame"]
    texts = [lb.text() for lb in answer.findChildren(QtWidgets.QLabel, "chatText")]
    assert texts == ["It is fcc."] and answer.to_dict()["tools"] == ["describe_frame"]


def test_ai_chat_shows_a_long_answer_whole(app):
    """A long Markdown answer (lists, a table, inline code) is as tall as its text: a live model's
    answer was cut off after a few paragraphs, with blank space below."""
    from PySide6 import QtWidgets

    from ewald.gui.ask_panel import Message

    rows = "\n".join(f"| {k} | {k * 0.37:.2f} | {k * 1.1:.2f} |" for k in range(15))
    text = ("**Barrier: 0.325 eV.** " + "The atom that moves is Fe id 8. " * 8 + "\n\n**Steps**\n"
            + "".join(f"- **Cell {k}:** `ExpressionSelection(\"(ParticleIdentifier == 8)\")` and more words\n"
                      for k in range(8))
            + "\n| id | a | b |\n|---|---|---|\n" + rows + "\n\n**Last line.**")
    area = QtWidgets.QScrollArea(widgetResizable=True)
    inner = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(inner)
    area.setWidget(inner)
    m = Message("assistant")
    lay.addWidget(m)
    lay.addStretch()
    area.resize(300, 600)
    area.show()
    m.set_answer(text)
    for _ in range(50):
        app.processEvents()
    label = next(lb for lb in m.findChildren(QtWidgets.QLabel, "chatText") if "Last line" in lb.text())
    assert label.height() >= label.heightForWidth(label.width()) > 0
    area.close()


def test_ai_chat_per_notebook_tab(app, pipe):
    """Each notebook tab has its own chat: switching tabs shows that tab's messages, an answer still
    on its way lands in its own tab's chat, the messages are kept on the notebook (saved in .ewd), and
    a notebook opened with a saved chat shows it and sends it with the next question, unless the
    notebook is not trusted."""
    import threading
    import time

    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook

    release, asked = threading.Event(), []

    class TabAgent(_FakeAgent):
        def ask(self, question):
            asked.append(question)
            if question.endswith("slow"):
                release.wait(5)
            return f"answer to {question.splitlines()[-1]}"

    w = MainWindow(pipe)
    w.settings["llm.code_cells"] = "button"
    panel = w.ask_panel
    panel.agent_factory = lambda p, ev: TabAgent(p, ev)
    first = w.panel.view
    panel.ask("about the first")
    _wait(app, panel)
    assert [m.text for m in panel.messages] == ["about the first", "answer to about the first"]
    assert [m["role"] for m in first.notebook.chat] == ["user", "assistant"]

    second = w.panel.new_notebook()
    app.processEvents()
    assert panel.conversation.key is second.notebook and panel.messages == [] and not panel.welcome.isHidden()
    panel.ask("slow")
    assert panel.busy()
    w.panel.tabs.setCurrentWidget(first)               # the answer keeps coming in the background
    app.processEvents()
    assert [m.text for m in panel.messages][-1] == "answer to about the first" and not panel.busy()
    assert panel.ask_button.toolTip() == "Send (Enter)"
    release.set()
    t = time.time()
    while second.notebook.chat[-1:] != [{"role": "assistant", "text": "answer to slow", "author": "AI"}] \
            and time.time() - t < 5:
        app.processEvents()
    assert second.notebook.chat[-1]["text"] == "answer to slow" and len(first.notebook.chat) == 2
    w.panel.tabs.setCurrentWidget(second)
    app.processEvents()
    assert [m.text for m in panel.messages] == ["slow", "answer to slow"]
    panel.reset()                                      # New chat empties this tab's chat only
    assert second.notebook.chat == [] and len(first.notebook.chat) == 2

    saved = [{"role": "user", "text": "What is the CN?"},
             {"role": "assistant", "text": "**12** for all atoms.", "author": "Claude", "tools": ["describe_frame"]}]
    for restricted in (True, False):
        nb = Notebook.from_py("# %%\nx = 1\n", name=f"saved {restricted}")
        nb.chat, nb.restricted = [dict(m) for m in saved], restricted
        view = w.panel.add_notebook(nb)
        w.panel.tabs.setCurrentWidget(view)
        app.processEvents()
        msgs = panel.messages
        assert [m.text for m in msgs[:2]] == ["What is the CN?", "**12** for all atoms."]
        assert msgs[1].tools == ["describe_frame"] and msgs[2].role == "note"
        assert ("once the notebook is trusted" in msgs[2].text) == restricted
        panel.ask("And the bonds?")
        _wait(app, panel)
        assert ("<earlier_chat>" in asked[-1]) == (not restricted)
        if not restricted:
            assert "User: What is the CN?" in asked[-1] and "You (after running describe_frame): **12**" in asked[-1]
            panel.ask("More")                          # the agent has it now: not sent twice
            _wait(app, panel)
            assert "<earlier_chat>" not in asked[-1]
        assert nb.chat[:2] == saved and nb.chat[-1]["role"] == "assistant"
    n = len(panel._conversations)
    w.panel.close_tab(w.panel.tabs.indexOf(view))     # a closed tab's chat is forgotten
    app.processEvents()
    assert len(panel._conversations) == n - 1 and panel.conversation.key is not nb
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_activity_bar_order_icons_and_tooltips(app, pipe, monkeypatch):
    """Work side bars on top, Plugins and Preferences at the bottom, the Window menu in the same
    order, and every tooltip names its shortcut (D35)."""
    from ewald.gui.app import MainWindow
    from ewald.gui.icons import PATHS

    w = MainWindow(pipe)
    acts = w.activity_bar.actions()
    titles = [a.text() for a in acts]
    assert titles.index("AI chat") == 1 and titles[-2:] == ["Plugins", "Preferences"]
    spacer = w.activity_bar.widgetForAction(acts[titles.index("Plugins") - 1])
    assert spacer.objectName() == "activitySpacer"             # pushes the management group down
    assert [w.activity_buttons[n].property("ewald_icon") for n in ("AI chat", "Remotes", "Preferences")] \
        == ["ai", "remote", "settings"]
    assert PATHS["claude"] == PATHS["ai"]                      # old icon name still works for plugins
    assert w.plugin_panels["workflows"]["plugin"].icon == "workflow"
    tips = {n: w.activity_buttons[n].toolTip() for n in ("AI chat", "Remotes", "Plugins", "Preferences")}
    assert all("Ctrl+" in t or "\u2318" in t for t in tips.values()), tips
    side = [a.text().replace("&", "") for a in w.side_bar_actions]
    assert side == ["Notebook", "AI chat", "Remotes", "Plugins"]
    bar_order = [t for t in titles if t in side]
    assert bar_order == side
    opened = []
    monkeypatch.setattr(w, "preferences", lambda page=None: opened.append(page))
    w.activity_buttons["Preferences"].trigger()
    assert opened == [None] and not w.activity_buttons["Preferences"].isCheckable()
    w.close()


def test_preferences_pages_use_the_shared_icons():
    """Preferences pages draw their icons from icons.PATHS, so Notebooks, AI and Plugins look
    the same as on the activity bar."""
    from ewald.gui.icons import PATHS
    from ewald.gui.pref_widgets import PAGE_ICONS
    from ewald.settings import PAGES

    assert set(PAGE_ICONS) == set(PAGES) and all(name in PATHS for name in PAGE_ICONS.values())
    assert (PAGE_ICONS["Notebooks"], PAGE_ICONS["AI"], PAGE_ICONS["Plugins"]) == ("notebook", "ai", "plugins")


def test_activity_bar_overflow_and_hidden_buttons(app, pipe, monkeypatch):
    """Too many side bars for the window's height: the top group ends in "More side bars" with
    the rest, Plugins and Preferences stay, an open side bar keeps its button and buttons hidden
    from the right-click menu are saved (D38)."""
    from ewald import plugins
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    for i in range(14):
        plugins.register_panel(f"p{i}", lambda context: QtWidgets.QLabel("x"), title=f"Panel {i}")
    w = MainWindow(pipe)
    w.resize(900, 600)
    w.show()
    for _ in range(5):
        app.processEvents()
    w.arrange_activity_bar()
    bar = w.activity_bar
    shown = lambda name: bar.widgetForAction(w.activity_buttons[name]).isVisible()   # noqa: E731
    overflow = [a.text() for a in w._activity_overflow]
    assert w.activity_more.isVisible() and "Panel 13" in overflow and "Remotes" in overflow
    assert shown("Plugins") and shown("Preferences") and shown("Notebook")
    ext = [b for b in bar.findChildren(QtWidgets.QToolButton) if b.objectName() == "qt_toolbar_ext_button"]
    assert not any(b.isVisible() for b in ext)                  # Qt's own overflow arrow never shows
    w.show_plugin_panel("p13")
    app.processEvents()
    assert w.activity_buttons["plugin:p13"].isVisible()         # the open side bar keeps its button
    w.set_activity_hidden("Panel 0", True)
    assert "Panel 0" not in [a.text() for a in w._activity_overflow] and not w.activity_buttons["plugin:p0"].isVisible()
    assert w.settings["general.activity_hidden"] == "Panel 0"
    assert "Panel 0" in [a.text() for a in w.plugin_panel_actions]   # still in the Window menu
    w.set_activity_hidden(None)
    assert w.settings["general.activity_hidden"] == ""
    w.resize(900, 1400)
    for _ in range(5):
        app.processEvents()
    w.arrange_activity_bar()
    if bar.height() > 22 * 46:                                  # tall enough for every button
        assert not w.activity_more.isVisible() and not w._activity_overflow
    w.close()


def test_plugin_panel_on_activity_bar_and_preferences_page(app, pipe, monkeypatch):
    """Installing a plugin with @panel and @settings_page adds an activity-bar button that opens
    its side bar (built on first open) and a page in Preferences; switching it off removes both."""
    import shutil
    from pathlib import Path

    from ewald import plugins
    from ewald.gui import preferences as prefs
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parents[1] / "examples" / "plugins" / "sim_agent_panel.py", folder)
    built = []

    def broken(context):
        built.append(context)
        raise RuntimeError("no server")

    plugins.register_panel("broken", broken, title="Broken")
    w = MainWindow(pipe)
    titles = [a.text() for a in w.activity_bar.actions() if a.text() and a is not w.activity_more]   # work on top
    assert titles[:2] == ["Notebook", "AI chat"] and titles[-3:] == ["Remotes", "Plugins", "Preferences"]
    assert sorted(titles[2:-3]) == ["Broken", "Sim agent", "Workflows"]
    assert w.plugin_panels["sim_agent"]["dock"] is None          # nothing built before it opens
    assert sorted(a.text() for a in w.plugin_panel_actions) == ["Agent jobs", "Broken", "Sim agent", "Workflows"]
    window = [a.text() for a in w.window_menu.actions()]            # with the side bars, before Remotes
    assert window.index("Workflows") < window.index("Re&motes") and "Plugin panels" not in window
    w.plugin_panels["sim_agent"]["action"].trigger()
    dock = w.plugin_panels["sim_agent"]["dock"]
    assert dock is not None and not dock.isHidden() and w.plugin_panels["sim_agent"]["action"].isChecked()
    labels = [lab.text() for lab in dock.widget().findChildren(QtWidgets.QLabel)]
    assert "Server: http://localhost:8000  (default)  no token" in labels and any("atoms, frame 0" in t for t in labels)
    edit = dock.widget().findChild(QtWidgets.QLineEdit)
    edit.setText("relax")
    edit.returnPressed.emit()
    assert "sim-agent task: relax" in w.panel.view.notebook.cells[-1].source
    w.set_frame(1)
    assert any("frame 1" in lab.text() for lab in dock.widget().findChildren(QtWidgets.QLabel))
    w.show_plugin_panel("sim_agent", False)
    assert dock.isHidden()
    w.show_plugin_panel("broken")
    assert built and isinstance(w.plugin_panels["broken"]["dock"].widget(), QtWidgets.QPlainTextEdit)
    assert "no server" in w.plugin_panels["broken"]["dock"].widget().toPlainText()

    dlg = prefs.PreferencesDialog(w.settings, w, page="Sim agent")
    assert dlg.page_names[-1] == "Sim agent" and dlg.stack.currentIndex() == len(dlg.page_names) - 1
    fields = dlg.plugin_pages["Sim agent"]["widgets"]
    assert fields["server"].text() == "http://localhost:8000" and fields["insert_cells"].isChecked()
    fields["server"].setText("http://hpc:9000")
    fields["insert_cells"].setChecked(False)
    dlg.restore_page()
    assert fields["server"].text() == "http://localhost:8000" and fields["insert_cells"].isChecked()
    fields["server"].setText("http://hpc:9000")
    dlg.applied.connect(lambda s: w._plugin_event("settings", s))     # as MainWindow.preferences does
    assert dlg.apply()
    assert plugins.plugin_settings("sim-agent")["server"] == "http://hpc:9000"
    assert any("http://hpc:9000" in lab.text() for lab in dock.widget().findChildren(QtWidgets.QLabel))
    dlg._search("access token")
    assert not dlg.pages.item(dlg.page_names.index("Sim agent")).isHidden()
    assert dlg.pages.item(0).isHidden()
    dlg.close()

    plugins.set_enabled("sim-agent-example", False)
    w.refresh_plugin_panels()
    assert [a.text() for a in w.activity_bar.actions() if a.text() and a is not w.activity_more][2:-3] \
        == ["Broken", "Workflows"]
    assert "sim_agent" not in w.plugin_panels
    w.close()


def test_plugin_menu_items_bottom_tab_and_layout_restore(app, pipe, monkeypatch, tmp_path):
    """@menu_item entries land in the menu bar (greyed out until their `when` holds) and the 3D
    view's right-click menu; a bottom @panel is a closable tab under the 3D view; open plugin
    side bars and tabs come back with the saved layout."""
    import shutil
    from pathlib import Path

    from ewald import plugins, settings
    from ewald.gui import prefs_apply
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parents[1] / "examples" / "plugins" / "sim_agent_panel.py", folder)
    calls = []
    plugins.register_menu_item("boom", lambda ctx: 1 / 0, title="Boom", menu="Help")
    plugins.register_menu_item("count", lambda ctx: calls.append(ctx.data.count), title="Count atoms",
                               menu="View/Plugin tools", when="data", shortcut="Ctrl+Alt+Shift+C")
    s = settings.Settings(path=tmp_path / "settings.json")
    s["general.restore_layout"] = True
    w = MainWindow(settings=s)

    tools = w._menu_bar_menu("Tools")
    sub = next(a.menu() for a in tools.actions() if a.menu() is not None and a.text() == "Sim agent")
    assert [a.text() for a in sub.actions()] == ["Send structure…", "Settings…"]
    send, _ = sub.actions()
    tools.aboutToShow.emit()
    assert not send.isEnabled()                # when="data" and nothing is loaded
    assert "Agent jobs" not in [a.text() for a in w.activity_bar.actions()]   # a bottom tab has no button
    menu = w.viewport_menu()
    assert not any(a.text() == "Send picked atoms to Sim agent" for a in menu.actions())
    w.set_pipeline(pipe)
    tools.aboutToShow.emit()
    assert send.isEnabled()
    view_sub = next(a.menu() for a in w._menu_bar_menu("View").actions() if a.text() == "Plugin tools")
    view_sub.actions()[0].trigger()
    assert calls == [w.data.count]
    next(a for a in w._menu_bar_menu("Help").actions() if a.text() == "Boom").trigger()   # error is reported
    assert not any(a.text() == "Send picked atoms to Sim agent" for a in w.viewport_menu().actions())
    w.viewport.set_picks([0, 2])
    menu = w.viewport_menu()
    act = next(a for a in menu.actions() if a.text() == "Send picked atoms to Sim agent")
    act.trigger()
    assert "relax atoms 0, 2" in w.panel.view.notebook.cells[-1].source
    w.viewport.set_picks([])

    # the bottom tab: built on first show, lists the jobs, closable, rebuilt on the next show
    table = w.show_plugin_panel("agent_jobs")
    assert isinstance(table, QtWidgets.QTableWidget) and w.workbench.is_shown("plugin:agent_jobs")
    assert table.rowCount() >= 1 and table.item(table.rowCount() - 1, 0).text() == "relax atoms 0, 2"
    w.workbench.close_view("plugin:agent_jobs")
    assert "plugin:agent_jobs" not in w.workbench.specs and w.plugin_panels["agent_jobs"]["widget"] is None
    w.show_plugin_panel("agent_jobs")
    dock = w.show_plugin_panel("sim_agent")
    w.resizeDocks([dock], [410], QtCore.Qt.Horizontal)
    app.processEvents()
    width = dock.width()
    prefs_apply.save_layout(w, s)
    w.close()

    w2 = MainWindow(settings=s)
    app.processEvents()
    entry = w2.plugin_panels["sim_agent"]
    assert entry["dock"] is not None and not entry["dock"].isHidden() and entry["action"].isChecked()
    assert abs(entry["dock"].width() - width) <= 20
    assert w2.plugin_panels["agent_jobs"]["widget"] is not None and "plugin:agent_jobs" in w2.workbench.specs
    plugins.set_enabled("sim-agent-example", False)
    w2.refresh_plugin_panels()
    assert "plugin:agent_jobs" not in w2.workbench.specs
    assert not any(a.menu() is not None and a.text() == "Sim agent" for a in w2._menu_bar_menu("Tools").actions())
    w2.close()


def test_plugin_items_on_toolbar_cell_and_plot_menus(app, pipe, monkeypatch):
    """@menu_item(menu="Toolbar" | "Notebook cell" | "Plot"): a toolbar button (or drop-down),
    entries in a cell's ⋯ menu and in a plot's right-click menu, with context.target."""
    import shutil
    from pathlib import Path

    from ewald import plugins
    from ewald.gui.app import MainWindow

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parents[1] / "examples" / "plugins" / "sim_agent_panel.py", folder)
    seen = []
    plugins.register_menu_item("a", lambda ctx: seen.append("a"), title="Alpha", menu="Toolbar/Tools+", when="picks")
    plugins.register_menu_item("b", lambda ctx: seen.append(ctx.target), title="Cell kind",
                               menu="Notebook cell/More")
    w = MainWindow(pipe)
    tb = w.findChild(QtWidgets.QToolBar, "main-toolbar")
    button = next(a for a in tb.actions() if a.text() == "Sim agent")
    assert not button.icon().isNull()
    button.trigger()                                   # shows the side bar
    assert w.plugin_panels["sim_agent"]["dock"] is not None
    drop = next(b for b in tb.findChildren(QtWidgets.QToolButton, "pluginToolButton") if b.text() == "Tools+")
    alpha = drop.menu().actions()[0]
    assert alpha.text() == "Alpha" and not alpha.isEnabled()     # nothing picked
    w.viewport.set_picks([1])
    assert alpha.isEnabled()
    alpha.trigger()
    assert seen == ["a"]

    cell_widget = w.panel.view.widgets[0]
    menu = cell_widget.findChild(QtWidgets.QToolButton, "cellMenu").menu()
    menu.aboutToShow.emit()
    titles = [a.text() for a in menu.actions()]
    assert "Send cell to Sim agent" in titles and "More" in titles
    menu.aboutToShow.emit()                            # rebuilt, not doubled
    assert [a.text() for a in menu.actions()].count("Send cell to Sim agent") == 1
    more = next(a.menu() for a in menu.actions() if a.text() == "More")
    more.actions()[0].trigger()
    assert seen[-1] is cell_widget.cell
    next(a for a in menu.actions() if a.text() == "Send cell to Sim agent").trigger()
    assert "sim-agent task: run cell:" in w.panel.view.notebook.cells[-1].source

    plot = w.plots
    target = plot.target()
    assert isinstance(target, plugins.PlotTarget)
    pmenu = plot.context_menu()
    ptitles = [a.text() for a in pmenu.actions()]
    assert ptitles[:2] == ["Save plot…", "Reset view"] and ptitles[-1] == "Send plotted data to Sim agent"
    pmenu.actions()[-1].trigger()
    assert "sim-agent task: analyze" in w.panel.view.notebook.cells[-1].source

    plugins.set_enabled("sim-agent-example", False)
    w.refresh_plugin_panels()
    assert not any(a.text() == "Sim agent" for a in tb.actions())
    w.close()


def test_plugin_settings_page_dialog(app, monkeypatch):
    from ewald import plugins, settings
    from ewald.gui.app import MainWindow
    from test_plugins import SETTINGS_PLUGIN

    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "demo_service.py").write_text(SETTINGS_PLUGIN)
    plugins.load_plugins(force=True)
    try:
        w = MainWindow(settings=settings.Settings())
        assert "Plugin &settings" not in [a.text() for a in w.edit_menu.actions()]   # Preferences has the page
        item = next(i for i in w.plugins_panel.items if i.id == "demo_service")
        dlg = w.plugins_panel.edit_settings(item, exec_=False)
        assert dlg.widgets["url"].text() == "http://old" and dlg.widgets["verify"].isChecked()
        assert dlg.widgets["key"].text() == "" and dlg.widgets["key"].placeholderText() == "not set"
        assert dlg.widgets["url"].toolTip() == "Where it runs"          # help as a tooltip, not a row
        assert dlg.advanced.isHidden()                  # "mode" is advanced and at its default: folded
        dlg.advanced_button.click()
        assert not dlg.advanced.isHidden() and dlg.advanced.isAncestorOf(dlg.widgets["mode"])
        dlg.widgets["url"].setText("ftp://nope")
        dlg.buttons.accepted.emit()
        assert dlg.isVisible() and "starts with http" in dlg.status.text()      # stays open to fix it
        dlg.widgets["url"].setText("http://new")
        dlg.widgets["key"].setText("sa_pat_2")
        dlg.buttons.accepted.emit()
        assert not dlg.isVisible() and dlg.message == "Saved" and dlg.widgets["key"].text() == ""
        dlg = w.edit_plugin_settings("demo-service", exec_=False)
        assert dlg.widgets["url"].text() == "http://new" and dlg.widgets["key"].placeholderText().startswith("saved")
        test = next(b for b in dlg.buttons.buttons() if b.text() == "Test")
        test.click()
        assert dlg.status.text() == "reachable" and dlg.isVisible()
        dlg.close()
        w.close()
    finally:
        (folder / "demo_service.py").unlink()
        plugins.load_plugins(force=True)


def test_open_location_opens_a_scheme_path(app, cu_fcc, monkeypatch):
    from ewald import plugins, settings
    from ewald.gui.app import MainWindow

    seen = []

    def fake(path, **kwargs):
        seen.append(path)
        return [cu_fcc]

    plugins.register_reader("fake-scheme", fake, schemes=("fakeloc",), priority=None)
    try:
        w = MainWindow(settings=settings.Settings())
        assert "fakeloc" in w.location_schemes()
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("fakeloc:1234/scf", True))
        assert w.open_location_dialog() is not None
        assert seen == ["fakeloc:1234/scf"] and w.pipeline.num_frames == 1
        assert w._last_location == "fakeloc:1234/scf"
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("", False))
        assert w.open_location_dialog() is None
        w.close()
    finally:
        plugins._registry.readers.pop("fake-scheme", None)


def test_plugin_settings_page_in_preferences_refuses_bad_values(app, monkeypatch):
    """A plugin page's save() raising ValueError keeps Preferences open on that page with the message."""
    from ewald import plugins, settings
    from ewald.gui import preferences as prefs
    from test_plugins import SETTINGS_PLUGIN

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "demo_service.py").write_text(SETTINGS_PLUGIN)
    dlg = prefs.PreferencesDialog(settings.Settings())
    entry = dlg.plugin_pages["Demo service"]
    assert entry["widgets"]["key"].placeholderText() == "not set"
    fold = next(b for b in entry["card"].parent().findChildren(QtWidgets.QToolButton) if b.text().startswith("Advanced"))
    assert entry["rows"]["mode"].parent().isHidden()        # the advanced card starts folded
    fold.click()
    assert not entry["rows"]["mode"].parent().isHidden()
    assert dlg.apply()                                    # nothing changed: save() is not called
    entry["widgets"]["url"].setText("ftp://nope")
    dlg.show_page("General")
    dlg.accept()
    assert dlg.result() == 0 and dlg.page_names[dlg.stack.currentIndex()] == "Demo service"
    assert "starts with http" in entry["status"].text()
    entry["widgets"]["url"].setText("http://new")
    entry["widgets"]["key"].setText("sa_pat_3")
    assert dlg.apply() and entry["status"].text() == "Saved" and entry["widgets"]["key"].text() == ""
    assert plugins.settings_pages()["demo-service"].values()["key"] is True
    test = next(b for b in entry["card"].parent().findChildren(QtWidgets.QPushButton) if b.text() == "Test")
    test.click()
    assert entry["status"].text() == "reachable"
    dlg.close()


def test_workflow_side_bar_lists_expands_and_opens_a_workflow_as_one_notebook(app, pipe, monkeypatch):
    """A workflow source plus workflow_panel(): the side bar follows an empty search page, expands
    a workflow into its nested steps and outputs, and a double click opens it as one notebook
    with a variable per step and pipe on the main result."""
    from ase.build import bulk

    from ewald import plugins, workflow
    from ewald.data import DataCollection
    from ewald.gui.app import MainWindow
    from ewald.workflow import Workflow, WorkflowOutput, WorkflowPage, WorkflowStep

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    monkeypatch.setattr(plugins._registry, "loaded", True)

    def full():
        return Workflow("relax", id="7", path="demo:7/relax", kind="vasp_relax", formula="Si2", state="finished",
                        label="Si relax", complete=True,
                        steps=[WorkflowStep("relax", path="demo:7/relax", state="finished", runs=2,
                                            outputs=[WorkflowOutput("vasprun.xml", "demo:7/relax/vasprun.xml")]),
                               WorkflowStep("bands", steps=[WorkflowStep("scf", path="demo:7/bands/scf",
                                                                         state="finished")])])

    class Demo:
        schemes = ("demo",)

        def account(self):
            return "me at demo"

        def list(self, query=None, cursor=None):
            if cursor is None:
                return WorkflowPage([], next="1")
            return WorkflowPage([Workflow("relax", id="7", path="demo:7/relax", kind="vasp_relax",
                                          state="finished", group="Si study")],
                                next="2" if cursor == "1" and query is None else None)

        def describe(self, ref):
            return full()

    plugins.register_reader("demo", lambda path, **kw: [DataCollection.from_ase(bulk("Si"))], schemes=("demo",))
    plugins.register_workflow_source("demo", Demo, title="Demo")
    plugins.register_panel("demo", workflow.workflow_panel("demo", title="Demo"))
    w = MainWindow(pipe)
    w.show_plugin_panel("demo")
    side = w.plugin_panels["demo"]["dock"].widget()
    side.wait()
    assert side.who.text() == "me at demo" and side.login_button.isHidden()
    top = side.tree.topLevelItem(0)
    assert side.tree.topLevelItemCount() == 1 and top.text(0) == "vasp_relax  <7>  ✓" and not side.more.isHidden()
    top.setExpanded(True)
    side.wait()
    assert [top.child(i).text(0) for i in range(top.childCount())] == ["relax  (2 runs)  ✓", "bands"]
    # the group is drawn under the title (a two-line row), not put into the item text
    from ewald.gui.workflow_panel import GROUP

    assert top.data(0, GROUP) == "Si study" and top.toolTip(0).startswith("Si study")
    assert side.tree.visualItemRect(top).height() > side.tree.visualItemRect(top.child(0)).height()
    assert top.child(1).child(0).text(0) == "scf  ✓" and top.child(0).child(0).text(0) == "vasprun.xml"
    side.tree.setCurrentItem(top.child(1).child(0))
    assert side.path.text() == "demo:7/bands/scf" and side.buttons["Open"].isEnabled()
    tabs = w.panel.tabs.count()
    side.tree.setCurrentItem(top)
    view = side.open(top)
    assert w.panel.tabs.count() == tabs + 1 and view.notebook.name == "Si relax"   # the pipe's notebook stays
    ns = view.notebook.namespace
    assert ns["pipe"] is ns["relax"] and ns["scf"].num_frames == 1 and w.pipeline is ns["relax"]
    side.search.setText("Si")
    side.refresh()
    side.wait()
    assert side.tree.topLevelItemCount() == 1 and side.more.isHidden()


def test_workflow_side_bar_switches_the_shown_step_and_plots_a_step_table(app, pipe, monkeypatch):
    """In a workflow notebook the side bar marks the step that is pipe (●) and the other loaded
    ones (○); a click on ○ rewrites the pipe = ... cell, and a "Plot Bands" row adds a view cell
    show(bands, table="Bands", panel=True) that plots the bands step, not pipe."""
    from ase.build import bulk

    from ewald import plugins, workflow
    from ewald.data import DataCollection
    from ewald.gui import workflow_panel as wp
    from ewald.gui.app import MainWindow
    from ewald.workflow import Workflow, WorkflowStep

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    monkeypatch.setattr(plugins._registry, "loaded", True)

    def read(path, **kw):
        d = DataCollection.from_ase(bulk("Si"))
        if path.endswith("bands"):
            d.tables["Bands"] = {"k": np.arange(3.0), "E": np.array([1.0, 2.0, 3.0])}
        else:
            d.tables["Steps"] = {"Step": np.arange(2.0), "Energy": np.array([-5.0, -5.4])}
        return [d]

    wf = Workflow("bands", id="8", path="demo:8", kind="vasp_bands", state="finished", complete=True,
                  steps=[WorkflowStep("relax", path="demo:8/relax", state="finished"),
                         WorkflowStep("bands", path="demo:8/bands", state="finished")])

    class Demo:
        schemes = ("demo",)

        def account(self):
            return "me"

        def list(self, query=None, cursor=None):
            return workflow.WorkflowPage([wf])

        def describe(self, ref):
            return wf

    plugins.register_reader("demo", read, schemes=("demo",))
    plugins.register_workflow_source("demo", Demo, title="Demo")
    plugins.register_panel("demo", workflow.workflow_panel("demo", title="Demo"))
    w = MainWindow(pipe)
    w.show_plugin_panel("demo")
    side = w.plugin_panels["demo"]["dock"].widget()
    side.wait()
    top = side.tree.topLevelItem(0)
    assert top.child(0).text(1) == ""                     # no workflow notebook on screen yet
    view = side.open(top)
    side.refresh_marks()
    side.wait()
    steps = [top.child(i) for i in range(top.childCount()) if not top.child(i).data(0, wp.TABLE)]
    relax, bands = steps
    assert [x.data(0, wp.VAR) for x in (top, relax, bands)] == ["result", "relax", "bands"]
    assert [x.text(1) for x in (top, relax, bands)] == [wp.ACTIVE, wp.OTHER, wp.OTHER]  # pipe = result
    rows = lambda item: [item.child(i).text(0) for i in range(item.childCount()) if item.child(i).data(0, wp.TABLE)]
    assert rows(relax) == ["Plot Steps"] and rows(bands) == ["Plot Bands"]

    cell = side.activate(bands)                            # ○ -> pipe = bands
    side.wait()
    assert cell.source == "pipe = bands" and view.notebook.pipeline is view.notebook.namespace["bands"]
    assert sum("pipe = " in c.source for c in view.notebook.cells) == 1
    assert [x.text(1) for x in (top, relax, bands)] == [wp.OTHER, wp.OTHER, wp.ACTIVE]
    side.activate(top)                                     # back to the workflow's own result
    side.wait()
    assert view.notebook.pipeline is view.notebook.namespace["result"] and top.text(1) == wp.ACTIVE

    row = next(bands.child(i) for i in range(bands.childCount()) if bands.child(i).data(0, wp.TABLE) == "Bands")
    cells = len(view.notebook.cells)
    vc = side.plot_table(row)
    assert vc.source == "show(bands, table='Bands', panel=True)" and len(view.notebook.cells) == cells + 1
    widget = next(x for x in view.widgets if x.cell is vc)
    assert widget.plot_panel.data is not None and "Bands" in widget.plot_panel.data.tables  # bands, not pipe
    assert side.plot_table(row) is vc and len(view.notebook.cells) == cells + 1           # once
    # a table name is a string literal in the cell, never code
    vc2 = view.notebook.add_view(source="bands", table="x')\nimport os\n#", panel=True)
    import ast
    assert [type(n).__name__ for n in ast.parse(vc2.source).body] == ["Expr"]


def test_dropping_a_folder_of_calculations_opens_it_as_a_workflow(app, tmp_path, vasp_dir):
    """A folder of calculations (here a relax and an MD restarted twice) dropped on the window
    opens as one notebook, a load() per step; the built-in Workflows side bar lists it after."""
    import shutil

    from ewald.gui.app import MainWindow

    study = tmp_path / "study"
    shutil.copytree(vasp_dir / "md", study / "1_relax")
    for run in ("run1", "run2"):
        shutil.copytree(vasp_dir / "md", study / "2_md" / run)
    w = MainWindow()
    tabs = w.panel.tabs.count()
    w.open_dropped([str(study)])
    assert w.panel.tabs.count() == tabs + 1
    assert w.pipeline.num_frames == 4                       # pipe = md: run1 + run2 joined
    w.show_plugin_panel("workflows")
    side = w.plugin_panels["workflows"]["dock"].widget()
    side.wait()
    assert not side.browse.isHidden() and side.tree.topLevelItemCount() == 1
    assert side.tree.topLevelItem(0).data(0, QtCore.Qt.UserRole).path == str(study)
    w.close()


def test_workflows_side_bar_compares_selected_workflows_in_a_table(app, tmp_path, vasp_dir):
    """Two folder workflows selected in the Workflows side bar: the right-click action adds and
    runs one cell workflow_table('<folder>', '<folder>') whose output is the side-by-side table."""
    import ast
    import shutil

    from ewald.gui.app import MainWindow
    from ewald.workflow_folder import add_recent_folder

    for name in ("pbe", "hse"):
        for step in ("1_relax", "2_md"):
            shutil.copytree(vasp_dir / "md", tmp_path / name / step)
        add_recent_folder(tmp_path / name)
    w = MainWindow()
    w.show_plugin_panel("workflows")
    side = w.plugin_panels["workflows"]["dock"].widget()
    side.wait()
    assert side.tree.topLevelItemCount() == 2
    for i in range(2):
        side.tree.topLevelItem(i).setSelected(True)
    chosen = side.selected_workflows()
    assert [wf.id for wf in chosen] == [str(tmp_path / "hse"), str(tmp_path / "pbe")]   # newest first
    cell = side.compare()
    assert w.panel.view.wait()
    code = ast.parse(cell.source)
    call = code.body[-1].value
    assert [a.value for a in call.args] == [str(tmp_path / "hse"), str(tmp_path / "pbe")]   # literals only
    html = next(o for o in cell.outputs if o.kind == "rich").mime["text/html"]
    assert "<th>hse</th>" in html and "<th>pbe</th>" in html and "1_relax" in html
    widget = next(x for x in w.panel.view.widgets if x.cell is cell)
    area = widget.findChild(QtWidgets.QScrollArea, "cellRichScroll")   # a wide table scrolls sideways
    assert area is not None and area.widget().text() == html
    w.close()


def test_wait_for_runs_off_the_gui_thread_with_cancel(app, pipe, tmp_path):
    import threading

    from ewald.gui import busy
    from ewald.gui.app import MainWindow

    said = []
    assert busy.wait_for(lambda: threading.current_thread() is threading.main_thread(), "Quick…", say=said.append) is False
    assert said == ["Quick…"]
    with pytest.raises(ZeroDivisionError):
        busy.wait_for(lambda: 1 / 0, "Failing…")
    release = threading.Event()
    seen = []

    def press_cancel():
        dialog = QtWidgets.QApplication.activeModalWidget() or next(
            w for w in QtWidgets.QApplication.topLevelWidgets() if w.objectName() == "waitDialog" and w.isVisible())
        seen.append(dialog.labelText())
        dialog.findChild(QtWidgets.QPushButton).click()

    QtCore.QTimer.singleShot(300, press_cancel)
    with pytest.raises(busy.Cancelled):          # a slow call gets a dialog with Cancel; Cancel returns at once
        busy.wait_for(lambda: release.wait(10), "Connecting to slow…", delay_ms=50)
    assert seen == ["Connecting to slow…"]
    release.set()                                # the abandoned call finishes in the background
    busy.finish_orphans(5000)
    assert not busy._orphans

    w = MainWindow()                             # File › Open goes through it
    path = tmp_path / "x.xyz"
    path.write_text("1\n\nH 0 0 0\n")
    assert w.open_file(str(path)).num_frames == 1
    w.close()


def test_notebook_status_drops_parts_instead_of_clipping(app):
    from ewald.gui.notebook_panel import _ElidedLabel

    label = _ElidedLabel("3 steps · not saved")
    fm = label.fontMetrics()
    label.resize(fm.horizontalAdvance("3 steps · not saved") + 4, 20)
    assert label.shown_text() == "3 steps · not saved"
    label.resize(fm.horizontalAdvance("3 steps") + 4, 20)
    assert label.shown_text() == "3 steps"           # not "3 st…ved"
    label.resize(4, 20)
    assert label.shown_text() == ""


def test_running_cell_shows_its_output_and_progress_bar_live(app, tmp_path, monkeypatch):
    """What a cell prints and its progress bars appear while it runs (a long download used to
    look frozen until it finished); afterwards the bar is full and the text is complete."""
    import ewald.gui.notebook_panel as npn
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.gui.cell_widget import ProgressOutput
    from ewald.notebook import Notebook

    nb = Notebook.from_py("# %%\nimport time\nfrom ewald.progress import progress\nprint('start')\n"
                          "for i in progress(range(10), desc='steps', mininterval=0):\n    time.sleep(0.15)\n"
                          "print('end')\n")
    nb.save(tmp_path / "live.ipynb")
    s = settings.Settings()
    s["python.run_notebooks"] = "never"
    w = MainWindow(settings=s)
    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
    monkeypatch.setattr(npn, "CELL_BUDGET", 0.05)
    try:
        view = w.open_notebook(str(tmp_path / "live.ipynb"))
        view.run(view.notebook.cells[0])
        assert view.running
        assert not view.wait(0.8)                                      # still running ...
        area = view.widget_for(view.notebook.cells[0]).outputs
        bars = [x for x in area._widgets if isinstance(x, ProgressOutput)]
        assert bars and 0 < bars[0].bar.value() < bars[0].bar.maximum()   # ... and the bar is moving
        assert any(getattr(x, "text", lambda: "")() == "start" for x in area._widgets)
        assert view.wait(30)
        bars = [x for x in area._widgets if isinstance(x, ProgressOutput)]
        assert len(bars) == 1 and bars[0].bar.value() == bars[0].bar.maximum()
        assert "steps" in bars[0].label.text() and "10/10" in bars[0].label.text()
        assert [o.kind for o in view.notebook.cells[0].outputs] == ["text", "rich", "text"]
    finally:
        w.close()


def test_rich_outputs_render_in_cells(app):
    """HTML (a pandas-style table), Markdown, LaTeX and SVG outputs are drawn, not shown as text."""
    from PySide6 import QtCore

    from ewald.gui.cell_widget import OutputArea
    from ewald.notebook import Cell, Notebook

    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="red"/></svg>'
    nb = Notebook([Cell("from IPython.display import HTML, Markdown, Math, SVG, display\n"
                        "display(HTML('<table><tr><td>Cu</td><td>4</td></tr></table>'))\n"
                        "display(Markdown('**bold**'))\ndisplay(Math(r'\\alpha^2'))\n"
                        f"display(SVG({svg!r}))\n")])
    assert nb.run_all(), nb.cells[0].outputs
    assert [o.kind for o in nb.cells[0].outputs] == ["rich"] * 4
    area = OutputArea(live=False)
    area.resize(400, 300)
    area.set_outputs(nb.cells[0].outputs)
    table, md, latex, picture = area._widgets
    assert isinstance(table, QtWidgets.QScrollArea)          # a table scrolls sideways, never cut off
    html = table.widget()
    assert html.textFormat() == QtCore.Qt.RichText and "<table>" in html.text()
    assert md.textFormat() == QtCore.Qt.MarkdownText
    assert latex.objectName() == picture.objectName() == "cellImage"       # mathtext and SVG as pictures
    assert len(area._images) == 2 and all(not pix.isNull() for _, pix in area._images)


def test_long_cell_output_scrolls_and_the_rule_folds_it(app, monkeypatch):
    """Output longer than python.output_scroll_after scrolls in a box python.output_box_lines
    tall (Jupyter's scrolled output). The rule beside the outputs is Jupyter's output prompt: a
    click (Shift+O) shows every line or scrolls again, also a shorter output by hand; a click on
    one shorter than the box, a double-click or O hides the outputs. A traceback shows its end, a
    running cell's log follows its newest line, and 0 scrolls nothing by itself."""
    from PySide6 import QtTest

    from ewald.gui import theme
    from ewald.gui.cell_widget import CellWidget
    from ewald.gui.code_editor import OutputText
    from ewald.notebook import Cell, Notebook

    def pump(wait=False):
        if wait:   # a single click waits out the double-click interval
            QtTest.QTest.qWait(QtWidgets.QApplication.doubleClickInterval() + 50)
        for _ in range(5):
            app.processEvents()

    def click(rule, double=False):
        if double:
            QtTest.QTest.mouseDClick(rule, QtCore.Qt.LeftButton, pos=QtCore.QPoint(3, 5))
        else:
            QtTest.QTest.mouseClick(rule, QtCore.Qt.LeftButton, pos=QtCore.QPoint(3, 5))
        pump(wait=True)

    monkeypatch.setattr(OutputText, "scroll_after", 50)
    monkeypatch.setattr(OutputText, "box_lines", 20)
    nb = Notebook([Cell("for i in range(100):\n    print('step', i)"), Cell("print('\\n'.join(map(str, range(30))))"),
                   Cell("print('short')"), Cell("def f(n):\n    return f(n - 1) if n else 1/0\nf(30)")])
    nb.run_all(stop_on_error=False)
    theme.apply(app, "graphite", font_size=12)
    cells = [CellWidget(nb, c) for c in nb.cells]
    for w in cells:
        w.resize(480, 600)
        w.show()
    pump()
    w = cells[0]
    text, rule = w.outputs.texts()[0], w.outputs.rule
    ls = text.fontMetrics().lineSpacing()
    assert text.line_count() == 100 and text.is_scrolled()
    assert text.height() < 22 * ls and text.verticalScrollBar().maximum() > 0
    assert rule.property("state") == "scrolled" and "100 lines" in rule.toolTip()
    click(rule)
    assert text.height() > 99 * ls and rule.property("state") == "expanded"
    click(rule)
    assert text.height() < 22 * ls and rule.property("state") == "scrolled"
    w.setFocus()
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_O, QtCore.Qt.ShiftModifier)   # Jupyter's command keys
    pump()
    assert text.height() > 99 * ls
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_O)
    pump()
    assert w.outputs.hidden and not text.isVisible() and rule.property("state") == "hidden"
    assert "100 lines" in w.outputs.hidden_note.text() and w.outputs.hidden_note.isVisible()
    w.outputs.hidden_note.click()
    pump()
    assert text.isVisible() and not w.outputs.hidden_note.isVisible()
    click(rule, double=True)                       # a double-click hides, without a click first
    assert w.outputs.hidden and text.height() > 99 * ls
    click(rule)
    assert not w.outputs.hidden

    mid = cells[1].outputs                          # 30 lines: whole, but can be scrolled by hand
    assert mid.rule.property("state") == "expanded" and not mid.is_scrolled()
    click(mid.rule)
    assert mid.is_scrolled() and mid.texts()[0].height() < 22 * ls

    short = cells[2].outputs                        # shorter than the box: a click hides it
    assert short.rule.property("state") == "short" and not short.can_scroll()
    click(short.rule)
    assert short.hidden and "(1 line)" in short.hidden_note.text()
    click(short.rule)
    assert not short.hidden

    err = cells[3].outputs.texts()[0]
    bar = err.verticalScrollBar()
    if err.is_scrolled():
        assert bar.value() == bar.maximum() > 0
    err.set_scrolled(True)
    pump()
    assert err.objectName() == "cellError" and bar.value() == bar.maximum() > 0

    log = OutputText("line 0")   # a running cell printing: the box follows the newest line
    log.resize(400, 100)
    log.show()
    for n in range(1, 60):
        log.setText(log.text() + f"\nline {n}")
    pump()
    assert log.is_scrolled() and log.verticalScrollBar().value() == log.verticalScrollBar().maximum() > 0
    log.verticalScrollBar().triggerAction(QtWidgets.QAbstractSlider.SliderToMinimum)   # scrolled up: stays
    pump()
    log.setText(log.text() + "\nline 60")
    pump()
    assert log.verticalScrollBar().value() == 0

    monkeypatch.setattr(OutputText, "scroll_after", 0)   # Preferences: nothing scrolls by itself
    w.outputs.set_scrolled(None)
    pump()
    assert not text.is_scrolled() and text.height() > 99 * ls
    for x in (*cells, log):
        x.close()


def test_cell_code_and_output_never_overlap(app):
    """The editor and the outputs are as tall as their laid-out text in the font the style
    sheet gives them, also after that font changes (Windows at 200 %, the font size preference),
    and a traceback's unbreakable rule wraps instead of cutting the last lines off. The outputs
    sit in their own area (cellOutput) in a monospaced font."""
    from PySide6 import QtCore, QtGui

    from ewald.gui import theme
    from ewald.gui.cell_widget import CellWidget
    from ewald.gui.code_editor import OutputText
    from ewald.notebook import Cell, Notebook

    def pump():
        for _ in range(5):
            app.processEvents()

    def text_height(w):
        doc = w.document()
        return sum(w.blockBoundingRect(doc.findBlockByNumber(i)).height() for i in range(doc.blockCount())) \
            + 2 * doc.documentMargin()

    before = app.styleSheet()
    nb = Notebook([Cell("for i in range(3):\n    print('step', i)\nprint('-' * 200)\n1/0")])
    nb.run_all(stop_on_error=False)
    try:
        theme.apply(app, "graphite", font_size=12)
        w = CellWidget(nb, nb.cells[0])
        w.resize(420, 400)
        w.show()
        pump()
        outs = w.outputs._widgets
        assert [o.objectName() for o in outs] == ["cellText", "cellError"]
        assert all(isinstance(o, OutputText) for o in outs)
        assert all(QtGui.QFontInfo(t.font()).fixedPitch() for t in (w.editor, *outs))   # not the UI font
        assert w.content.objectName() == "cellContent" and w.outputs.testAttribute(QtCore.Qt.WA_StyledBackground)
        for size in (12, 20):     # a bigger font after the widgets exist: measured again
            theme.apply(app, "graphite", font_size=size)
            pump()
            for t in (w.editor, *outs):
                assert t.viewport().height() + 1 >= text_height(t), (size, t.objectName())
            top = w.outputs.mapTo(w, QtCore.QPoint(0, 0)).y()
            assert w.editor.mapTo(w, QtCore.QPoint(0, w.editor.height())).y() <= top
        assert "ZeroDivisionError: division by zero" in outs[1].text()
        w.close()
    finally:
        app.setStyleSheet(before)


def test_console_shares_the_notebook_namespace(app, tmp_path, monkeypatch):
    """The Python console runs in the notebook's IPython namespace: a cell's variables are there
    and what the console defines is seen by the next cell, without copying."""
    import ewald.gui.notebook_panel as npn
    from ewald import settings
    from ewald.gui.app import MainWindow
    from ewald.notebook import Notebook

    pytest.importorskip("qtconsole")
    nb = Notebook.from_py("# %%\nfrom_cell = 41\n\n# %%\nboth = from_cell + from_console\n")
    nb.save(tmp_path / "ns.ipynb")
    s = settings.Settings()
    s["python.run_notebooks"] = "never"
    w = MainWindow(settings=s)
    monkeypatch.setattr(npn, "CONFIRM_UNSAVED", False)
    monkeypatch.setattr(npn, "CELL_BUDGET", None)
    try:
        view = w.open_notebook(str(tmp_path / "ns.ipynb"))
        view.run(view.notebook.cells[0])
        con = w.console.widget()
        w._push_console()
        shell = con.shell
        assert shell.user_ns is view.notebook.namespace and shell.user_ns["from_cell"] == 41
        shell.run_cell("from_console = 1")
        assert view.notebook.run_cell(1) and view.notebook.namespace["both"] == 42
        assert "viewer" in shell.user_ns_hidden                     # console helpers stay out of %who
    finally:
        w.close()


def test_restart_keeps_outputs_and_restart_and_clear_drops_them(app):
    """Restart clears the namespace but keeps the outputs, as in Jupyter; the Restart button's
    menu and Notebook › Restart & clear outputs also remove them."""
    from ewald.gui.app import MainWindow

    w = MainWindow()
    w.panel.new_notebook()
    view = w.panel.view
    cell = view.add_cell("code", "x = 41\nprint('hello')\nx + 1")
    view.run(cell)
    view.wait()
    assert [o.kind for o in cell.outputs] == ["text", "value"] and cell.state == "ok"
    view.restart()
    assert "x" not in view.notebook.namespace and cell.state == "stale"
    assert [o.kind for o in cell.outputs] == ["text", "value"]
    assert view.widget_for(cell).outputs.outputs == cell.outputs
    assert [a.text() for a in view.restart_button.menu().actions()] == ["Restart && clear outputs"]
    view.restart_button.menu().actions()[0].trigger()
    assert cell.outputs == [] and not view.widget_for(cell).outputs.outputs
    labels = [a.text() for a in w.notebook_menu.actions()]
    assert labels[labels.index("&Restart") + 1] == "Restart && cl&ear outputs"
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_next_notebook_does_not_inherit_the_previous_ones_bond_table(app, tmp_path):
    """MgB2 opened first, TiO2 next: the Display panel shows Ti-O bonds, not Mg-B (the window's style
    seeds an untouched view, but only with settings for the new structure's elements)."""
    from ase import Atoms
    from ase.io import write

    from ewald import settings
    from ewald.gui.app import MainWindow

    mgb2 = Atoms("MgB2", scaled_positions=[(0, 0, 0), (1 / 3, 2 / 3, 0.5), (2 / 3, 1 / 3, 0.5)],
                 cell=[[3.08, 0, 0], [-1.54, 2.667, 0], [0, 0, 3.52]], pbc=True)
    write(tmp_path / "POSCAR.MgB2", mgb2, format="vasp")
    u = 0.305   # rutile
    tio2 = Atoms("Ti2O4", scaled_positions=[(0, 0, 0), (.5, .5, .5), (u, u, 0), (1 - u, 1 - u, 0),
                                            (.5 + u, .5 - u, .5), (.5 - u, .5 + u, .5)], cell=[4.59, 4.59, 2.96], pbc=True)
    write(tmp_path / "POSCAR.TiO2", tio2, format="vasp")
    (tmp_path / "mgb2.py").write_text(f"# %%\nfrom ewald import load, view\npipe = load({str(tmp_path / 'POSCAR.MgB2')!r})\n"
                                      '\n# %%\nview(pipe, colors={"Mg": "#f2a33a", "B": "#3a6fd9"}, bonds=True)\n')
    (tmp_path / "tio2.py").write_text(f"# %%\nfrom ewald import load\npipe = load({str(tmp_path / 'POSCAR.TiO2')!r})\n")
    w = MainWindow(settings=settings.Settings())
    first = w.panel.open_notebook(str(tmp_path / "mgb2.py"), trusted=True)
    first.run_all()
    app.processEvents()
    assert any({a, b} == {"Mg", "B"} for a, b, *_ in w.style.bond_pairs)
    second = w.panel.open_notebook(str(tmp_path / "tio2.py"), trusted=True)
    second.run_all()
    app.processEvents()
    names = {s for a, b, *_ in w.style.bond_pairs for s in (a, b)}
    assert names and names <= {"Ti", "O"}, w.style.bond_pairs
    pipe = second.notebook.pipeline
    assert not {"Mg", "B"} & set(pipe.view.colors) and not pipe.view.bond_pairs
    w.close()


def test_viewport_sets_vtk_surface_format_before_any_gl_widget(app, monkeypatch):
    """The placeholder (the window's first OpenGL widget) already gets VTK's 3.2 core format, so
    the window's compositing context can share with the VTK view's on macOS (black view otherwise)."""
    import sys

    from ewald.gui.legacy_viewport import LegacyViewport
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport, vtk_surface_format

    QtGui.QSurfaceFormat.setDefaultFormat(QtGui.QSurfaceFormat())  # Qt's default: legacy profile
    monkeypatch.setattr(sys, "platform", "darwin")
    try:
        v = Viewport(ViewStyle())
        assert QtGui.QSurfaceFormat.defaultFormat() == vtk_surface_format()
        assert v._placeholder.format().profile() == QtGui.QSurfaceFormat.CoreProfile
        assert v._placeholder.format().version() == (3, 2)
        _dispose(v)
        assert Viewport.uses_vtk and not LegacyViewport.uses_vtk  # GLView asks for OpenGL 1.x itself
    finally:
        QtGui.QSurfaceFormat.setDefaultFormat(vtk_surface_format())


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_vtk_surface_format_stays_off_the_window_outside_macos(app, monkeypatch, platform):
    """On Windows the VTK default format (alpha, 8-bit depth) on the window's compositing surface
    turned the whole window black after a maximize or resize; only macOS needs it up front."""
    import sys

    from ewald.gui.application import use_vtk_surface_format
    from ewald.gui.viewport import vtk_surface_format

    QtGui.QSurfaceFormat.setDefaultFormat(QtGui.QSurfaceFormat())
    monkeypatch.setattr(sys, "platform", platform)
    try:
        use_vtk_surface_format()
        assert QtGui.QSurfaceFormat.defaultFormat() == QtGui.QSurfaceFormat()
    finally:
        QtGui.QSurfaceFormat.setDefaultFormat(vtk_surface_format())


def test_viewport_has_no_msaa_so_depth_peeling_keeps_see_through_atoms(app):
    """With Smooth edges off the VTK view kept pyvista's 4x MSAA, and depth peeling into those
    framebuffers dropped every see-through layer: NEB overlays and Transparency vanished."""
    from ewald.gui.style import ViewStyle
    from ewald.gui.viewport import Viewport

    v = Viewport(ViewStyle())
    v._want_anti_aliasing = False
    v.ensure_view()
    try:
        assert v.plotter.render_window.GetMultiSamples() == 0
        v.set_anti_aliasing(True)
        v.set_anti_aliasing(False)
        assert v.plotter.render_window.GetMultiSamples() == 0
    finally:
        _dispose(v)


def test_resize_handles_are_visible_in_every_theme(app):
    """Dock separators and split handles show a rule that thickens into a pill grip, in a 7 px hit
    area (the owner could only find the notebook/structure boundary by hovering, then found the
    plain border rule too faint under the grip), in every UI theme."""
    from PySide6 import QtGui

    from ewald.gui import theme

    saved = dict(theme.TOKENS), dict(theme.STYLE)
    # images are cached in the temp folder by content: an older grip drawing there is not reused
    first = theme.image("grip_v")
    entry = theme._IMAGES["grip_v"]
    try:
        theme._IMAGES["grip_v"] = (entry[0], entry[1].replace("v24", "v20"), *entry[2:])
        assert theme.image("grip_v") != first
    finally:
        theme._IMAGES["grip_v"] = entry
    # the pointer, wherever an earlier test left it, must not hover a separator (the hover color differs)
    QtGui.QCursor.setPos(app.primaryScreen().geometry().bottomRight())
    try:
        for name in theme.PALETTES:
            t = theme.use(name)
            for img in ("grip_v", "grip_h", "grip_v_hover", "grip_h_hover"):
                path = theme.image(img)      # PNG (Qt squares SVG style-sheet images) with a @2x copy
                small, large = QtGui.QImage(path), QtGui.QImage(path.replace(".png", "@2x.png"))
                assert sorted((small.width(), small.height())) == [7, 32] and large.width() == 2 * small.width()
            rule = theme._mix(t["border"], t["muted"], theme.RULE)
            w = QtWidgets.QMainWindow()
            w.setStyleSheet(theme.stylesheet())
            w.resize(400, 300)
            split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            split.addWidget(QtWidgets.QLabel("a"))
            split.addWidget(QtWidgets.QLabel("b"))
            w.setCentralWidget(split)
            dock = QtWidgets.QDockWidget("notebook")
            dock.setWidget(QtWidgets.QLabel("cells"))
            w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
            w.show()
            app.processEvents()
            im = w.grab().toImage()
            left, right = dock.geometry().right() + 1, w.centralWidget().geometry().left()
            assert right - left == 7, name                         # the separator's hit area
            row = [im.pixelColor(x, 20).name().upper() for x in range(left, right)]
            assert row == [t["panel"]] * 3 + [rule] + [t["panel"]] * 3, (name, row)
            mid = dock.geometry().center().y()
            grip = [im.pixelColor(x, mid).name().upper() for x in range(left, right)]
            assert grip[2:5] == [t["muted"]] * 3, (name, grip)     # the pill sits on the rule
            pill = [im.pixelColor(left + 3, y).name().upper() for y in range(mid - 20, mid + 20)]
            assert 18 <= pill.count(t["muted"]) <= 26, (name, pill)
            handle = split.handle(1)
            assert handle.height() == 7, name
            corner = handle.mapTo(w, QtCore.QPoint(0, 0))
            col = [im.pixelColor(corner.x() + 20, y).name().upper() for y in range(corner.y(), corner.y() + 7)]
            assert col == [t["panel"]] * 3 + [rule] + [t["panel"]] * 3, (name, col)
            w.close()
            _dispose(w)
    finally:
        theme.TOKENS.clear()
        theme.TOKENS.update(saved[0])
        theme.STYLE.clear()
        theme.STYLE.update(saved[1])
    css = theme.stylesheet()     # hover: an accent rule and a wider accent grip
    assert "QSplitter::handle:horizontal:hover" in css and "grip_v_hover" in css


def test_bundled_ui_fonts_are_registered(app):
    from PySide6 import QtGui

    from ewald.gui.fonts import load_fonts

    families = load_fonts()
    assert {"IBM Plex Sans", "IBM Plex Mono", "JetBrains Mono"} <= set(families)
    assert load_fonts() == families                      # once per process
    for family in ("JetBrains Mono", "IBM Plex Sans"):   # Qt adds "[foundry]" when the system has it too
        assert QtGui.QFontInfo(QtGui.QFont(family)).family().startswith(family)


def test_style_sheets_name_only_installed_font_families(app):
    """A family Qt cannot find (as the first choice or as a glyph fallback) makes it build its
    alias table: ~200 ms on macOS and a "Replace uses of missing font family" warning. The
    fallback lists name fonts of every platform; what reaches Qt is cut to what is installed."""
    import re

    from PySide6 import QtGui

    from ewald.gui import fonts, theme

    sans = fonts._system("GeneralFont")
    mac = {"ibm plex sans", "helvetica neue", "menlo"}        # no Segoe UI, Consolas or Inter
    windows = {"ibm plex sans", "segoe ui", "consolas"}
    rule = 'font-family: "IBM Plex Sans", "Inter", "Segoe UI", "Helvetica Neue", sans-serif;'
    assert fonts.css(rule, mac) == f'font-family: "IBM Plex Sans", "Helvetica Neue", "{sans}";'
    assert fonts.css(rule, windows) == f'font-family: "IBM Plex Sans", "Segoe UI", "{sans}";'
    mono = 'font-family: "IBM Plex Mono", "Menlo", "Consolas", monospace; }'
    fixed = fonts._system("FixedFont")
    assert fonts.css(mono, mac) == f'font-family: "Menlo", "{fixed}"; }}'
    assert fonts.installed(["Inter", "Fira Code"], set()) == [sans]   # never empty
    assert fonts.installed(["Menlo", "menlo", "MENLO"], mac) == ["Menlo"]

    known = {re.sub(r"\s*\[.*\]$", "", f).lower() for f in QtGui.QFontDatabase.families()}
    theme.apply(app, "graphite")
    lists = re.findall(r"font-family:\s*([^;}]+)", app.styleSheet())
    assert lists
    for names in lists:
        for name in names.split(","):
            assert name.strip().strip('"').lower() in known, name
    assert "Inter" not in theme.QSS


def test_plugins_side_bar_recommends_a_restart_when_pip_replaced_loaded_code(app):
    from ewald import settings
    from ewald.gui import plugins_panel

    panel = plugins_panel.PluginsPanel(settings=settings.Settings())
    panel.wait()
    asked = []
    panel.restartRequested.connect(lambda: asked.append(True))
    panel.recommend_restart("")                               # nothing to finish
    assert panel.restart_bar.isHidden()

    class Done:
        def deleteLater(self):
            pass

    panel._process, panel._before = Done(), {"numpy": "0.0.1"}   # pip upgraded numpy, which is loaded
    panel._pip_done("install", "ewald-something", 0)
    assert not panel.restart_bar.isHidden() and "numpy 0.0.1 -> " in panel.restart_label.text()
    restart = next(b for b in panel.restart_bar.findChildren(plugins_panel.QtWidgets.QPushButton)
                   if b.text() == "Restart now")
    restart.click()
    assert asked == [True]
    later = next(b for b in panel.restart_bar.findChildren(plugins_panel.QtWidgets.QToolButton) if b.text() == "Later")
    later.click()
    assert panel.restart_bar.isHidden()
    panel.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_plugins_side_bar_uses_an_installed_ui_theme_and_the_window_restarts(app, tmp_path, monkeypatch):
    import sys

    from ewald import plugin_catalog as pc
    from ewald import settings
    from ewald.gui import app as app_mod
    from ewald.gui.app import MainWindow

    src = tmp_path / "glow_theme.py"
    src.write_text('from ewald.plugin_api import ui_theme\n\n\n@ui_theme(name="Glow", base="graphite")\n'
                   'def glow():\n    """Green."""\n    return {"accent": "#33FF66", "font": "JetBrains Mono"}\n',
                   encoding="utf-8")
    w = MainWindow(settings=settings.Settings())
    panel = w.plugins_panel
    panel.install(pc.describe_file(src))
    panel.wait()
    assert not panel.theme_button.isHidden() and panel.theme_button.text() == "Use theme Glow"
    assert panel.restart_bar.isHidden()                       # a new file plugin works at once
    panel.theme_button.click()
    assert w.settings["general.ui_theme"] == "Glow" and settings.load()["general.ui_theme"] == "Glow"
    from ewald.gui import theme

    assert theme.TOKENS["accent"] == "#33FF66" and theme.STYLE["font"] == "JetBrains Mono"

    nb = tmp_path / "saved.py"
    nb.write_text("# %%\nx = 1\n", encoding="utf-8")
    w.panel.open_notebook(str(nb), trusted=True)
    started, quit_ = [], []
    monkeypatch.setattr(app_mod.QtCore.QProcess, "startDetached",
                        staticmethod(lambda prog, args, cwd: started.append([prog, *args]) or True))
    monkeypatch.setattr(app_mod.QtWidgets.QApplication, "quit", staticmethod(lambda: quit_.append(True)))
    panel.recommend_restart("test")
    panel.restartRequested.emit()
    assert quit_ == [True] and not w.isVisible()
    assert started and started[0][:4] == [sys.executable, "-m", "ewald", "gui"] and str(nb) in started[0]
    theme.use("graphite")
PHONON_DATA = os.path.join(os.path.dirname(__file__), "data", "phonon")


def test_phonon_band_plot_click_picks_mode(app):
    from matplotlib.backend_bases import MouseEvent

    from ewald import import_file
    from ewald.gui.plots import PlotsPanel

    pipe = import_file(os.path.join(PHONON_DATA, "band.yaml"), frames=4)
    pipe.modifiers.append(mods.PhononMode(qpoint=4, band=5))
    panel = PlotsPanel()
    panel.resize(600, 400)
    panel.set_data(pipe.compute(0))
    ax = panel.figure.axes[0]
    assert panel.table.currentText() == "Phonon Bands" and ax.get_ylabel() == "Frequency (THz)"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["$\\Gamma$", "X", "M", "$\\Gamma$", "R"]
    modes = pipe.source_frame(0).attributes["Phonon Modes"]
    assert list(panel._marker.get_ydata()) == [modes.frequencies[4, 5]]
    panel.set_data(pipe.compute(1))
    assert panel.figure.axes[0] is ax  # next frame of the animation: no redraw
    picked = []
    panel.phononModePicked.connect(lambda q, b: picked.append((q, b)))
    panel.canvas.draw()
    x, y = ax.transData.transform((modes.distances[12], modes.frequencies[12, 9]))
    panel._clicked(MouseEvent("button_press_event", panel.canvas, x, y, button=1))
    assert picked == [(12, 9)]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_plays_phonon_mode(app):
    from ewald.gui.app import MainWindow

    w = MainWindow()
    pipe = w.open_file(os.path.join(PHONON_DATA, "OUTCAR.phonon"))
    assert [type(m).__name__ for m in pipe.modifiers] == ["PhononMode"]  # added on open
    assert w.data.attributes["Phonon.Band"] == 3 and "Phonon Displacement" in w.data.vectors
    w.play_phonon_mode(0, 9)
    assert pipe.modifiers[0].band == 9 and w.data.attributes["Phonon.Band"] == 9
    assert w.timeline.play.isChecked()
    w.timeline.play.setChecked(False)
    menu = {a.text(): a for a in w.viewport_menu().actions()}
    assert "Phonon band structure" in menu and "Play" in menu
    w.close()


def test_notebook_follows_the_ai_cells_without_jumping(app, pipe):
    """While the AI chat works on cells the notebook does not hop: reading never scrolls, a
    rewritten cell keeps the focus where it was (it used to pass from cell to cell, scrolling to
    each), a written cell off screen is scrolled to until you scroll yourself, then a pill points
    at it; Preferences › AI › Follow the AI's cells: end scrolls once at the end, never not at all."""
    from PySide6 import QtWidgets

    from ewald.gui.app import MainWindow
    from ewald.llm.tools import Toolset

    def settle():
        for _ in range(5):
            app.processEvents()

    w = MainWindow(pipe)
    w.resize(1200, 700)
    w.show()
    view = w.panel.view
    for i in range(25):
        view.add_cell("code", f"x{i} = {i}")
    settle()
    nb, bar, follow = view.notebook, view.scroll.verticalScrollBar(), view.ai_follow
    assert bar.maximum() > 0
    ts = Toolset(None, notebook=w.ask_panel.notebook_access(nb))
    first = nb.cells[1]
    w.activateWindow()
    view.reveal(first)                                     # the keyboard is on a cell
    bar.setValue(0)
    settle()
    moves = []
    bar.valueChanged.connect(moves.append)

    def edit(k):
        moves.append("edit")    # every edit writes new source
        ts.call("edit_cell", {"index": k, "source": f"y = {len(moves)}", "run": False})
        settle()
        return nb.cells[k]

    w._ai_answer(nb, True)
    ts.call("list_cells", {})
    ts.call("read_cell", {"index": len(nb.cells) - 1})
    edit(2)                                                # on screen: nothing moves
    assert moves == ["edit"] and not follow.pill.isVisible()
    focused = QtWidgets.QApplication.focusWidget()
    if focused is not None:                                # the window has the keyboard (an X server)
        assert view.widget_for(first) in (focused, focused.parentWidget())
    far = edit(len(nb.cells) - 3)                          # off screen: shown
    assert bar.value() > 0 and follow.visible(far) and not follow.pill.isVisible()
    bar.triggerAction(QtWidgets.QAbstractSlider.SliderToMinimum)   # you scroll back up
    settle()
    last = edit(len(nb.cells) - 2)
    assert bar.value() == 0 and follow.pill.isVisible()
    assert follow.pill.text() == f"AI is on cell {len(nb.cells) - 1}"
    w._ai_answer(nb, False)
    assert follow.pill.text() == f"AI edited cell {len(nb.cells) - 1}"
    follow.pill.click()
    settle()
    assert follow.visible(last) and not follow.pill.isVisible() and view.current is last

    w.settings["llm.follow_cells"] = "end"
    bar.setValue(0)
    w._ai_answer(nb, True)
    cell = edit(len(nb.cells) - 1)
    assert bar.value() == 0
    w._ai_answer(nb, False)
    settle()
    assert follow.visible(cell) and bar.value() > 0

    w.settings["llm.follow_cells"] = "never"
    bar.setValue(0)
    w._ai_answer(nb, True)
    edit(len(nb.cells) - 1)
    w._ai_answer(nb, False)
    settle()
    assert bar.value() == 0 and follow.pill.isVisible()
    w.close()


def test_first_run_notice_stays_until_the_window_is_up(app):
    """The first start after an install or update shows a notice before gui.app is imported and
    closes it once the main window is on screen; the saved stamp keeps it away next time."""
    from ewald.gui import first_run

    assert first_run.is_first_run()
    notice = first_run.start([])
    assert notice is not None and notice.isVisible() and notice.objectName() == "firstRunNotice"
    assert "first start" in notice.findChild(QtWidgets.QLabel, "firstRunBody").text()
    first_run.finish()
    assert not notice.isVisible()
    assert not first_run.is_first_run()
    assert first_run.start([]) is None


def test_notice_comes_before_the_heavy_imports():
    """The notice needs only the QApplication: VTK, pyvista and matplotlib load after it is up."""
    import subprocess
    import sys

    code = ("import sys, ewald.gui.first_run, ewald.gui.application\n"
            "print([m for m in ('vtkmodules', 'pyvista', 'pyvistaqt', 'matplotlib', 'ewald.gui.app')"
            " if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_view_frame_minus_one_is_the_last_frame(app):
    """view(pipe, frame=-1) shows the last frame, as pipe.compute(-1) does (a model wrote it for "show
    the last frame" in the first real-model eval, and the window showed frame 0)."""
    from ewald import examples, import_file
    from ewald.gui.app import MainWindow

    w = MainWindow(import_file(examples.path("Cu_md.extxyz")))
    n = w.pipeline.num_frames
    w.pipeline.view.update(frame=-1)
    w.recompute()
    app.processEvents()
    assert w.frame == n - 1
    w.pipeline.view.update(frame=2)
    w.recompute()
    app.processEvents()
    assert w.frame == 2
    w.close()


def test_plot_style_file_is_the_source_of_the_preferences_form(app, monkeypatch):
    """Preferences › Plots › Style file: the form reads user.mplstyle when it opens (and again
    when the file changed while it was open), Apply writes back only what changed, a bad value
    keeps the dialog open, and Open as text creates the file from the template first."""
    import os

    from ewald import plotstyle, settings
    from ewald.gui.preferences import PreferencesDialog

    path = plotstyle.user_style_path()
    path.parent.mkdir(parents=True)
    path.write_text("# my style\nfont.size: 9\naxes.grid: True\n", encoding="utf-8")
    dlg = PreferencesDialog(settings.Settings(), page="Plots")
    assert dlg._style_form()["Font size"] == 9.0 and dlg._style_form()["Line width"] is None
    dlg.style_widgets["Line width"].setValue(2.0)
    assert dlg.apply()
    assert plotstyle.read_style()[0] == {"font.size": "9", "axes.grid": "True", "lines.linewidth": "2"}
    dlg.style_preset.setCurrentIndex(dlg.style_preset.findData("boston"))   # the rows preview the preset
    assert dlg._style_form()["Font size"] == 14.0
    dlg.style_widgets["Line width"].setValue(1.5)                         # ... and a row set on top of it
    assert dlg.apply() and plotstyle.preset_of() == "boston"
    rc = plotstyle.read_style()[0]
    assert rc["font.size"] == "14" and rc["lines.linewidth"] == "1.5" and rc["axes.grid"] == "True"
    plotstyle.write_style({"font.size": "12"})                 # edited in a text editor meanwhile
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    dlg.event(QtCore.QEvent(QtCore.QEvent.WindowActivate))
    assert dlg._style_form()["Font size"] == 12.0
    cycle = plotstyle.read_style()[0]["axes.prop_cycle"]          # the Boston preset's
    dlg.style_widgets["Color cycle"].setText("nocolor")
    assert not dlg.apply() and "Color cycle" in dlg.style_problems.text()
    assert plotstyle.read_style()[0]["axes.prop_cycle"] == cycle
    dlg.style_widgets["Color cycle"].setText("")
    dlg.restore_page()                                          # Reset page: the preset and the rows go
    assert dlg.apply() and plotstyle.preset_of() == "seoul"
    assert plotstyle.read_style()[0] == plotstyle.preset_rc("seoul")   # Boston's grid went with it
    dlg.style_widgets["Font size"].setValue(16)                  # Save as preset: listed from then on
    assert dlg.save_style_preset("Talk 16") and plotstyle.preset_of() == "talk 16"
    assert dlg.style_preset.currentData() == "talk 16" and dlg.style_preset.findData("seoul") >= 0
    assert plotstyle.preset_rc("Talk 16")["font.size"] == "16"
    assert not dlg.save_style_preset("Seoul") and "built-in" in dlg.style_problems.text()
    opened = []
    monkeypatch.setattr("ewald.gui.file_dialogs.open_in_editor", lambda p: opened.append(p) or True)
    path.unlink()
    assert dlg.open_style_file() and opened == [path] and plotstyle.read_style() == (plotstyle.preset_rc("seoul"), [])
    assert "# EWALD plot style" in path.read_text(encoding="utf-8")
    dlg.close()


def test_plots_panel_shares_the_style_file(app, tmp_path):
    """The Plots panel draws with the style file too: on screen without the theme's colors and
    the canvas font size, in saved files with all of it; notebooks and the console get it too."""
    from PIL import Image

    from ewald import DataCollection, plotstyle, settings
    from ewald.gui import console, plots, prefs_apply

    host = QtWidgets.QWidget()
    host.console, host.settings = None, settings.Settings()
    panel = plots.PlotsPanel(host)
    d = DataCollection(np.zeros((1, 3)), np.array(["C"]))
    d.tables["rdf"] = {"r": np.linspace(0, 5, 20), "g(r)": np.linspace(0, 2, 20)}
    panel.set_data(d)
    path = plotstyle.user_style_path()
    path.parent.mkdir(parents=True)
    path.write_text("lines.linewidth: 3\naxes.facecolor: FFEEDD\nfont.size: 6\n", encoding="utf-8")
    try:
        assert prefs_apply.apply_plot_style(host) == []
        assert plots.USER_RC["lines.linewidth"] == "3" and console.PLOT_STYLE[2] == plots.USER_RC
        ax = panel.figure.axes[0]
        assert ax.lines[0].get_linewidth() == 3 and ax.get_facecolor()[:3] != (1.0, 0xEE / 255, 0xDD / 255)
        assert ax.xaxis.label.get_fontsize() == plots.SCREEN_FONT
        panel.export(tmp_path / "rdf.png", quality="Draft")
        img = Image.open(tmp_path / "rdf.png").convert("RGB")
        assert (255, 238, 221) in {img.getpixel((x, img.height // 2)) for x in range(img.width)}
        plotstyle.write_style({"lines.linewidth": None, "axes.facecolor": None, "font.size": None})
        prefs_apply.apply_plot_style(host)
        assert plots.USER_RC == {} and panel.figure.axes[0].lines[0].get_linewidth() == 1.25
    finally:
        plots.USER_RC.clear()
        console.PLOT_STYLE[:] = ["publication", None, None]
        host.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_clicking_the_color_bars_folds_them_and_writes_the_view(app, pipe):
    """A click on the viewport's color bar cards folds them to chips (and back), never picks an atom
    behind them, and writes colorbar_collapsed into pipe.view; code sets it the other way round."""
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.panel.add_modifier(mods.ColorCoding("Position.Z", cmap="plasma"))
    w.recompute()
    app.processEvents()
    vp = w.viewport
    box = vp._colorbar_box
    assert box is not None and box[2] <= 0 < box[3]                 # pinned to the top right corner
    width = vp.plotter.interactor.width()
    x, y = width + (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    assert vp.on_colorbar(x, y) and not vp.on_colorbar(width / 2, (box[1] + box[3]) / 2 + 400)
    vp.set_tool("pick")
    vp.on_colorbar(x, y) and vp.toggle_colorbars()                 # what a click on the cards does
    app.processEvents()
    assert w.pipeline.view.colorbar_collapsed is True and not vp.picks
    folded = vp._colorbar_box
    assert folded[3] - folded[1] < 3 * vp.COLORBAR_FONT + 6 < box[3] - box[1]   # a one-line chip
    w.pipeline.view.update(colorbar_collapsed=False)                 # a view cell unfolds them
    w.recompute()
    app.processEvents()
    assert w.style.colorbar_collapsed is False
    assert vp._colorbar_box[3] - vp._colorbar_box[1] == pytest.approx(box[3] - box[1], abs=2)
    w.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_view_tabs_are_more_structure_tabs(app, tmp_path):
    """view(pipe, tab="Top", ...) opens another 3D viewport tab with the main view's settings plus its own;
    running it again updates the same tab, Window › New structure tab writes such a cell, a style picked
    in a tab goes into its cell, and closing the tab removes it and its cell."""
    from ase.build import bulk
    from ase.io import write

    from ewald import settings
    from ewald.gui.app import MainWindow

    write(tmp_path / "POSCAR", bulk("NaCl", "rocksalt", 5.64), format="vasp")
    (tmp_path / "nb.py").write_text(
        f"# %%\nfrom ewald import load, view\npipe = load({str(tmp_path / 'POSCAR')!r})\n"
        '\n# %%\nview(pipe, colors={"Na": "#ff0000"})\n'
        '\n# %%\nview(pipe, tab="Top", direction="z", style="Space-filling")\n', encoding="utf-8")
    w = MainWindow(settings=settings.Settings())
    nbv = w.panel.open_notebook(str(tmp_path / "nb.py"), trusted=True)
    nbv.run_all()
    app.processEvents()
    assert list(w.view_tabs) == ["Top"]
    t = w.view_tabs["Top"]
    assert t.view_id in w.workbench.specs and w.workbench.group_of(t.view_id) is w.workbench.group_of("viewport")
    assert t.style.preset == "Space-filling" and t.style.type_colors["Na"] == "#ff0000"
    assert w.style.preset == "Ball & stick"            # the main view keeps its own style
    assert t.viewport.data is w.data
    nbv.run_all()                                       # the same cell again: still one tab
    app.processEvents()
    assert list(w.view_tabs) == ["Top"]
    nb = nbv.notebook
    assert nb.view_cell("Top") is not None and nb.view_cell() is not None
    # Window › New structure tab
    name = w.new_view_tab()
    app.processEvents()
    assert name == "View 2" and "View 2" in w.pipeline.views and "View 2" in w.view_tabs
    assert nb.view_cell("View 2").source.endswith("view(pipe, tab='View 2')")
    # a style picked in that tab's viewport goes into its cell, not into the main view's
    w.view_tabs["View 2"].style.apply_preset("Points", w.data.types)
    w.write_view_tab("View 2")
    assert w.pipeline.views["View 2"].style == "Points"
    assert "style='Points'" in nb.view_cell("View 2").source and "Points" not in nb.view_cell().source
    # closing the tab removes it from the pipeline and the notebook
    w.workbench.close_view(w.view_tabs["View 2"].view_id)
    app.processEvents()
    assert "View 2" not in w.pipeline.views and "View 2" not in w.view_tabs and nb.view_cell("View 2") is None
    w.close()


def _hud_host(app, cls, data=None):
    """A viewport in a plain host widget, so tests can give it any size (the workbench sizes it
    with setGeometry too, below the controls' minimum width)."""
    from ewald.gui.icons import button_style
    from ewald.gui.style import ViewStyle

    host = QtWidgets.QWidget()
    host.resize(900, 700)
    v = cls(ViewStyle(), host)
    for b, text in v.styled_buttons:
        button_style(b, "Icons", text)
    v.set_button_style("Icons")
    host.show()
    v.show()
    for _ in range(100):   # the 3D widget is built after the first paint (Viewport.lazy_view)
        app.processEvents()
        if v.view_built:
            break
        QtCore.QThread.msleep(10)
    if data is not None:
        v.set_data(data)
    app.processEvents()
    return host, v


def _visible_rects(chip):
    return [w.geometry() for w in chip.findChildren(QtWidgets.QWidget) if w.isVisibleTo(chip) and w.parent() is chip]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="3D view needs an X display with OpenGL (xvfb-run)")
@pytest.mark.parametrize("font_size", [12, 16])   # 16: a larger UI font, like Windows at 150-200 % with big text
def test_viewport_hud_folds_without_overlap(app, cu_fcc, font_size):
    """The view chips float over the 3D view and fold into menus as it gets small (D62): at no size do
    the two top chips overlap each other, nor their buttons each other, and Fit always stays."""
    from ewald.gui import theme
    from ewald.gui.viewport import Viewport

    theme.apply(app, font_size=font_size)
    try:
        host, v = _hud_host(app, Viewport, cu_fcc)
        levels = []
        for width in (900, 640, 480, 400, 320, 260, 220):
            v.setGeometry(0, 0, width, 360)
            app.processEvents()
            levels.append(v.hud_level)
            left, right = v._viewbar.geometry(), v._displaybar.geometry()
            assert left.right() < right.left(), (width, left, right)
            assert right.right() < width and left.left() >= 0
            for chip in (v._viewbar, v._displaybar):
                rects = _visible_rects(chip)
                assert all(not a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1:]), (width, chip)
            assert v.fit_button.isVisible()
            # the chips are over the GL widget, not beside it: the 3D view keeps the whole viewport
            v.layout().activate()
            assert v.plotter.interactor.geometry() == v.rect()
        assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == len(v.HUD_LEVELS) - 1
        # folded: the a*/b*/c* and rotations are one click away in menus, the rest in ⋯
        assert not v.axis_buttons["a"].isVisible() and v.axis_menu.isVisible()
        assert [a.text() for a in v.axis_menu.menu().actions()] == [f"Look along {x}" for x in v.AXES]
        v.more_button.menu().aboutToShow.emit()
        titles = [a.text() for a in v.more_button.menu().actions()]
        assert titles == ["Rotate", "Orthographic projection", "Style"]
        ortho = v.more_button.menu().actions()[1]
        ortho.setChecked(False)
        assert not v.ortho and not v.ortho_button.isChecked()
        style = v.more_button.menu().actions()[2].menu()
        [a for a in style.actions() if a.text() == "Space-filling"][0].trigger()
        assert v.style_combo.currentText() == "Space-filling"
        v.setGeometry(0, 0, 900, 600)
        app.processEvents()
        assert v.hud_level == 0 and v.axis_buttons["c*"].isVisible() and not v.more_button.isVisible()
        v.close()
        _dispose(host)
    finally:
        theme.apply(app, font_size=12)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="3D view needs an X display with OpenGL (xvfb-run)")
def test_viewport_tool_chip_folds_into_more(app, cu_fcc):
    """A short view folds the last tools (Clear first) into ⋯, which looks checked while it holds
    the current tool; the tool chip ends above the orientation axes."""
    from ewald.gui.viewport import Viewport

    host, v = _hud_host(app, Viewport, cu_fcc)
    v.setGeometry(0, 0, 600, 600)
    app.processEvents()
    assert v.rail_tools == len(v.TOOLS) and v.clear_button.isVisible() and not v.rail_more.isVisible()
    v.setGeometry(0, 0, 600, 260)
    app.processEvents()
    assert 1 <= v.rail_tools < len(v.TOOLS) and not v.clear_button.isVisible() and v.rail_more.isVisible()
    assert v._rail.geometry().bottom() < 260 - 0.2 * 260
    hidden = v.TOOLS[-1][0]
    v.set_tool(hidden)
    assert v.rail_more.isChecked()
    v.rail_more.menu().aboutToShow.emit()
    texts = [a.text() for a in v.rail_more.menu().actions()]
    assert "Bond" in texts and "Clear picks and measurements" in texts
    v.set_tool("orbit")
    assert not v.rail_more.isChecked()
    v.close()
    _dispose(host)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="3D view needs an X display with OpenGL (xvfb-run)")
def test_viewport_overlays_start_below_the_chips(app, cu_fcc):
    """Color bars, the color legend and the frame caption are drawn below the top chips, so the
    chips never cover them."""
    from ewald.gui.viewport import Viewport

    host, v = _hud_host(app, Viewport)
    v.setGeometry(0, 0, 640, 480)
    data = cu_fcc.copy()
    data["Potential Energy"] = np.linspace(-1, 1, data.count)
    v.style.color_by = "Potential Energy"
    v.overlay_text = "Frame 0"
    v.set_data(data)
    app.processEvents()
    left, top = v.hud_insets()
    assert top >= v._displaybar.geometry().bottom() and left >= v._rail.geometry().right()
    assert v._colorbar_box is not None and v._colorbar_box[1] >= top
    assert "_overlay" in v.plotter.actors
    v.close()
    _dispose(host)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="GL window needs an X display (xvfb-run)")
def test_legacy_viewport_puts_the_chips_around_the_view(app, cu_fcc):
    """The OpenGL 1.x viewport's native GL window can't have widgets drawn over it: the same chips sit
    in a strip above it and a column left of it, fold the same way, and offset no overlay."""
    from ewald.gui.legacy_viewport import LegacyViewport

    host, v = _hud_host(app, LegacyViewport, cu_fcc)
    for width in (900, 300):
        v.setGeometry(0, 0, width, 400)
        app.processEvents()
        gl = v._gl_widget()
        origin = gl.mapTo(v, QtCore.QPoint(0, 0))
        area = QtCore.QRect(origin, gl.size())
        for chip in (v._viewbar, v._displaybar, v._rail):
            r = QtCore.QRect(chip.mapTo(v, QtCore.QPoint(0, 0)), chip.size())
            assert not r.intersects(area), (width, chip.property("hudName"))
    assert v.hud_level >= 2 and v.hud_insets() == (0, 0)
    v.close()
    _dispose(host)


def test_timeline_shows_timestep_and_time_not_the_3d_view(app, pipe):
    """The frame's timestep and time sit beside its number in the timeline (D63), not as text in the
    3D view; a single frame or a frame without them shows nothing, and the preference hides it."""
    from ewald.gui.timeline import Timeline

    t = Timeline()
    t.set_num_frames(10)
    t.set_info({"Timestep": np.int64(500), "Time": 0.5})
    assert t.info.isVisibleTo(t) and "500" in t.info.text() and "0.5" in t.info.text()
    t.set_info({})
    assert not t.info.isVisibleTo(t)
    t.show_info = False
    t.set_info({"Timestep": 500})
    assert not t.info.isVisibleTo(t)
    t.show_info = True
    t.set_num_frames(1)
    t.set_info({"Timestep": 500})
    assert not t.info.isVisibleTo(t)
    _dispose(t)
