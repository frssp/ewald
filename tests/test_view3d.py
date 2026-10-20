"""pipe.view: every display setting of the 3D view as code (notebook, GUI, Claude)."""
import os

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, Pipeline, StaticSource, View, view
from ewald.view3d import SETTINGS, view_cell_args, view_source


@pytest.fixture
def srtio3():
    from ase import Atoms

    a = 3.905
    atoms = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
                  cell=[a, a, a], pbc=True)
    return Pipeline(StaticSource([DataCollection.from_ase(atoms)] * 3))


def test_settings_follow_the_style_preset_and_record_only_changes(srtio3):
    v = srtio3.view
    assert isinstance(v, View) and v.changed() == {} and v.version == 0
    assert (v.style, v.radius_scale, v.bonds, v.atom_style) == ("Ball & stick", 0.5, True, "auto")
    view(srtio3, style="Space-filling", colors={"O": "red"}, radii={"Ti": 0.6}, cell=False, direction="z")
    assert (v.radius_scale, v.bonds) == (1.0, False)                  # from the preset
    assert v.changed() == {"style": "Space-filling", "colors": {"O": "red"}, "radii": {"Ti": 0.6},
                           "cell": False, "direction": "z"}
    view(srtio3, style="Points", bonds=True)                         # style first, explicit settings win
    assert (v.radius_scale, v.bonds) == (0.25, True)
    v.style = "Simple (ASE)"
    assert v.theme == "ASE" and "theme" not in v.changed()             # implied by the preset
    assert (v.bonds, v.atom_style) == (True, "flat")                   # ASE look keeps its bonds
    v.theme = None
    assert v.changed()["theme"] is None                                 # explicitly no theme
    v.style = None                                                     # back to Ball & stick, drops the implied theme
    assert v.style == "Ball & stick" and "radius_scale" not in v.changed()
    before = v.version
    v.cell = False                                                     # unchanged: no new version
    assert v.version == before
    with pytest.raises(KeyError):
        v.update(colour="red")
    with pytest.raises(AttributeError):
        v.colour
    with pytest.raises(ValueError):
        v.atom_style = "cartoon"
    with pytest.raises(KeyError):
        v.theme = "No such theme"
    assert "bond_radius" in View.help() and set(SETTINGS) >= {"camera", "background", "boundary", "visuals"}


def test_helpers_colors_radii_camera(srtio3):
    v = srtio3.view
    v.color(["Sr", "Ti"], (255, 0, 0)).radius("O", 0.7)
    assert v.colors == {"Sr": (1.0, 0.0, 0.0), "Ti": (1.0, 0.0, 0.0)} and v.radii == {"O": 0.7}
    v.color("Ti", None)
    assert v.colors == {"Sr": (1.0, 0.0, 0.0)}
    v.look("x").rotate(azimuth=90)
    assert np.allclose(v.direction, [0, 1, 0], atol=1e-6)
    v.rotate(elevation=90)
    assert np.allclose(v.direction, [0, 0, 1], atol=1e-6)
    v.zoom_by(2)
    assert v.zoom == 2.0
    cam = {"position": (10, 0, 0), "focal_point": (0, 0, 0), "up": (0, 0, 1), "parallel_projection": True,
           "parallel_scale": 4.0, "view_angle": 30.0}
    v.camera = cam
    assert "direction" not in v.changed() and "zoom" not in v.changed()   # an exact camera replaces them
    v.rotate(azimuth=90).zoom_by(2)
    assert np.allclose(v.camera["position"], (0, 10, 0), atol=1e-9) and v.camera["parallel_scale"] == 2.0
    v.look("z")                                                         # a direction frames the structure again
    assert v.camera is None


def test_render_kwargs_and_pictures(srtio3, monkeypatch):
    from ewald import render as render_mod

    data = srtio3.compute(0)
    v = view(srtio3, colors={"O": "red"}, cell=False, background="black", text="frame [Frame]", frame=2)
    kw = v.render_kwargs(data)
    assert kw["type_colors"] == {"O": "red"} and kw["show_cell"] is False and kw["background"] == "black"
    assert kw["radius_scale"] == 0.5 and kw["text"] == "frame 2"
    pairs = {(a, b) for a, b, *_ in kw["bond_pairs"]}
    assert ("O", "Ti") in pairs and ("Sr", "Ti") not in pairs          # VESTA table, no metal-metal pairs
    v.bonds = 2.5
    assert v.render_kwargs(data)["bonds"] == 2.5 and "bond_pairs" not in v.render_kwargs(data)
    v.bonds = False
    assert "bonds" not in v.render_kwargs(data) and "bond_pairs" not in v.render_kwargs(data)

    calls = []
    monkeypatch.setattr(render_mod, "render", lambda d, f=None, **k: calls.append((f, k)) or np.zeros((2, 2, 3), np.uint8))
    srtio3.render("x.png", size=(10, 10))                                 # pipe.render draws with the view
    assert calls[-1][0] == "x.png" and calls[-1][1]["background"] == "black" and calls[-1][1]["size"] == (10, 10)
    from ewald.views import snapshot

    snapshot(srtio3).image()
    assert calls[-1][1]["type_colors"] == {"O": "red"} and calls[-1][1]["size"] == (480, 360)
    from ewald import display

    display.pipeline_png(srtio3)
    assert calls[-1][1]["show_cell"] is False and calls[-1][1]["renderer"] == "raytrace"


def test_view_cell_source_round_trip(srtio3):
    v = view(srtio3, colors={"O": (1.0, 0.0, 0.0)}, bond_pairs=[["Ti", "O", 0, 2.1]], boundary=[-0.1, 1.1])
    src = v.source()
    args = view_cell_args(src)
    assert args.pop("source") == "pipe" and View(**args).changed() == v.changed()
    long = view_source("pipe", **{k: "x" * 30 for k in ("text", "background", "cell_color", "theme")})
    assert long.startswith("view(pipe,\n     text=") and long.endswith(",\n)")
    assert view_cell_args("view(pipe, colour='red')") is None and view_cell_args("show(pipe)") is None


def test_notebook_view_cell(tmp_path):
    from ase.io import write

    from ewald.notebook import Notebook

    path = tmp_path / "cu.extxyz"
    write(path, bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2))
    nb = Notebook.from_py(f"from ewald import load, view\npipe = load({str(path)!r})\n\n"
                          "# %%\nview(pipe, style='Space-filling', colors={'Cu': 'orange'})\n")
    assert nb.run_all()
    pipe = nb.pipeline
    assert pipe.view.style == "Space-filling" and pipe.view.colors == {"Cu": "orange"}
    cell = nb.view_cell()
    assert cell is nb.cells[-1] and cell.outputs[-1].kind == "value"   # text, not a second picture
    pipe.view.update(bonds=True, direction="x")                          # the GUI changed it
    nb.set_view_cell(pipe.view.changed())
    assert cell.source == "view(pipe, style='Space-filling', colors={'Cu': 'orange'}, bonds=True, direction='x')"
    assert nb.run_all() and nb.pipeline.view.changed() == pipe.view.changed()
    plain = Notebook.from_py(f"from ewald import load\npipe = load({str(path)!r})\n")
    assert plain.run_all() and plain.set_view_cell({}) is None
    added = plain.set_view_cell({"cell": False})
    assert added.source == "from ewald import view\nview(pipe, cell=False)" and plain.cells[-1] is added


def test_claude_tools_drive_the_view(tmp_path, monkeypatch):
    from PIL import Image
    from ase.io import write

    from ewald.llm import tools as tools_mod
    from ewald.llm.tools import Toolset

    calls = []

    def render_png(data, path, **kw):
        calls.append(kw)
        Image.new("RGB", (8, 6), "white").save(path)

    monkeypatch.setattr(tools_mod, "render_png", render_png)
    path = tmp_path / "cu.extxyz"
    write(path, [bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2)] * 2)
    ts = Toolset()
    ts.call("load_structure", {"path": str(path)})
    schema = next(t for t in ts.definitions() if t["name"] == "set_view")["input_schema"]["properties"]
    assert {"colors", "bonds", "direction", "camera", "rotate", "reset"} <= set(schema)
    r = ts.call("set_view", {"colors": {"Cu": "gold"}, "bonds": False, "direction": "z", "rotate": [0, 30],
                             "background": "white", "frame": 1})
    assert not r.is_error and r.content["changed"]["colors"] == {"Cu": "gold"} and ts.frame == 1
    assert "colors={'Cu': 'gold'}" in r.content["code"] and "bonds=False" in r.content["code"]
    assert ts.pipeline.view.direction[2] < 0.9                         # tilted from z by 30 degrees
    assert ts.pipeline.view.bonds is False
    ts.call("render_image", {})
    assert calls[-1]["type_colors"] == {"Cu": "gold"} and calls[-1]["background"] == "white"
    assert "bond_pairs" not in calls[-1]
    ts.call("render_image", {"direction": "x"})                         # one picture only
    assert calls[-1]["direction"] == "x" and ts.pipeline.view.direction != "x"
    got = ts.call("get_view").content
    assert got["settings"]["background"] == "white" and "radius_scale" in got["help"]
    assert ts.call("set_view", {"atom_style": "cartoon"}).is_error
    assert ts.call("set_view", {"reset": True}).content["changed"] == {}
    ts.call("set_view", {"cell": False})
    assert "view(pipe, cell=False)" in ts.to_script()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_gui_follows_the_view_and_writes_the_view_cell(tmp_path):
    pytest.importorskip("PySide6")
    from ase.io import write
    from PySide6 import QtWidgets

    from ewald import settings
    from ewald.gui import compute, notebook_panel
    from ewald.gui.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    notebook_panel.CONFIRM_UNSAVED, compute.BUDGET = False, None
    path = tmp_path / "cu.extxyz"
    write(path, [bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2)] * 3)
    w = MainWindow(settings=settings.Settings())
    if w.layout_mode != "notebook":
        pytest.skip("notebook layout only")
    w.open_any(str(path), ask_types=False)
    view_nb = w.panel.view
    nb = view_nb.notebook
    assert w.pipeline.view.style == w.style.preset                        # seeded from the window
    cell = view_nb.add_cell("code", "view(pipe, style='Space-filling', colors={'Cu': '#ff0000'}, "
                                    "background='white', direction='x', frame=2)")
    view_nb.run(cell)
    w.recompute()
    assert w.style.preset == "Space-filling" and w.style.type_colors["Cu"] == "#ff0000"
    assert w.style.background == "#ffffff" and w.frame == 2
    cam = w.viewport.plotter.camera
    d = np.subtract(cam.position, cam.focal_point)
    assert abs(d[0]) > 10 * max(abs(d[1]), abs(d[2]))                   # looking along x
    w.viewport._preset("Ball & stick")                                   # the viewport writes the cell back
    assert nb.view_cell() is cell and "style=" not in cell.source and "colors={'Cu': '#ff0000'}" in cell.source
    assert w.pipeline.view.bonds is True
    w.style.show_cell = False
    w._style_changed()                                                   # Display panel edit
    assert "cell=False" in cell.source
    w.write_camera()                                                     # Insert › View settings with the camera
    assert "camera={" in cell.source and "frame=2" in cell.source
    w.close()
