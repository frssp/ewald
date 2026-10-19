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
    for menu in (w.panel.add_menu, menus["Pipeline"]):
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
def test_interactive_detail_coarse_then_refine(app, pipe):
    from ewald import settings
    from ewald.gui.app import MainWindow

    s = settings.Settings()
    s["viewport.interactive_detail"] = "coarse"
    s["performance.cache_memory"] = 64
    w = MainWindow(pipe, settings=s)
    vp, n = w.viewport, pipe.compute(0).count

    def impostors(name="atoms"):  # point-sprite actor -> its atom count, sphere mesh -> None
        mapper = vp.plotter.actors[name].GetMapper()
        if "PointGaussian" not in mapper.GetClassName():
            return None
        source = mapper.GetInputAlgorithm()
        source.Update()
        return source.GetOutputDataObject(0).GetNumberOfPoints()

    assert pipe.cache_memory == 64 * 2**20
    vp.interactive_atoms = 100  # below the Preferences minimum, to subsample the 256-atom test cell
    assert not vp.coarse and impostors() is None
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
    assert not vp.coarse and impostors() is None
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
