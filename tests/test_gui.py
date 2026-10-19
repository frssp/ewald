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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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


def test_command_palette_fuzzy():
    from ewald.gui.palette import fuzzy_score

    assert fuzzy_score("coord", "Coordination analysis / RDF") > fuzzy_score("coord", "Color coding")
    assert fuzzy_score("xyz", "Slice") is None


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
    assert settings.load()["compute.device"] == "auto"       # saved to disk
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
    apply_element_defaults(style, cu_fcc.types, s)
    assert style.type_colors == {}                            # defaults: nothing redundant
    s["elements.overrides"] = {"Cu": {"color": "#c8794a"}}
    apply_element_defaults(style, cu_fcc.types, s)
    assert style.type_colors["Cu"] == "#c8794a"
    s["elements.overrides"] = {}
    s["elements.color_scheme"] = "CPK"
    apply_element_defaults(style, cu_fcc.types, s)            # previous fill replaced
    assert style.type_colors["Cu"] != "#c8794a"
    style.type_colors["Cu"] = "#000000"                       # session edit wins
    apply_element_defaults(style, cu_fcc.types, s)
    assert style.type_colors["Cu"] == "#000000"


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
    assert w.device == "cpu" and w.style.background == "#ffffff" and w.style.bonds_enabled
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
