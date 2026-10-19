"""The main window's panel layout model (Qt-free): moves, splits, clean-up and JSON."""
import pytest

from ewald.gui.layout_model import Group, Layout, Split, zone_at


def default():
    return Layout({"editor": Group(["viewport"]), "panel": Group(["console", "inspector", "plot:1", "log"])})


def test_drop_zones():
    assert zone_at(400, 300, 200, 150) == "center"
    assert zone_at(400, 300, 10, 150) == "left"
    assert zone_at(400, 300, 395, 150) == "right"
    assert zone_at(400, 300, 200, 5) == "top"
    assert zone_at(400, 300, 200, 290) == "bottom"
    assert zone_at(0, 0, 1, 1) == "center"


def test_plot_into_editor_as_tab_puts_3d_view_behind():
    lay = default()
    editor = lay.groups("editor")[0]
    g = lay.move("plot:1", editor)
    assert g is editor and editor.views == ["viewport", "plot:1"] and editor.current_view == "plot:1"
    assert lay.groups("panel")[0].views == ["console", "inspector", "log"]
    lay.show("viewport")
    assert editor.current_view == "viewport"


def test_split_right_then_back():
    lay = default()
    editor = lay.groups("editor")[0]
    new = lay.move("plot:1", editor, "right")
    root = lay.areas["editor"]
    assert isinstance(root, Split) and root.orient == "h" and root.children == [editor, new]
    assert lay.area_of(new) == "editor" and lay.neighbour(editor, "right") is new
    assert lay.neighbour(new, "left") is editor and lay.neighbour(editor, "top") is None
    lay.move("plot:1", lay.groups("panel")[0], index=2)   # back home: the split goes away
    assert lay.areas["editor"] is editor
    assert lay.groups("panel")[0].views == ["console", "inspector", "plot:1", "log"]


def test_same_direction_splits_stay_flat_and_sizes_share():
    lay = default()
    editor = lay.groups("editor")[0]
    b = lay.move("plot:1", editor, "right")
    lay.move("log", b, "right")
    root = lay.areas["editor"]
    assert len(root.children) == 3 and all(isinstance(c, Group) for c in root.children)
    assert root.sizes == [1, 0.5, 0.5]
    c = lay.move("console", b, "bottom")  # the other direction nests
    assert isinstance(root.children[1], Split) and root.children[1].orient == "v"
    assert lay.neighbour(c, "top") is b and lay.neighbour(c, "right").views == ["log"]


def test_reorder_and_self_split():
    lay = default()
    panel = lay.groups("panel")[0]
    lay.move("log", panel, index=0)
    assert panel.views == ["log", "console", "inspector", "plot:1"]
    lay.move("log", panel, index=3)
    assert panel.views == ["console", "inspector", "log", "plot:1"]
    editor = lay.groups("editor")[0]
    assert lay.move("viewport", editor, "left") is editor     # a lone view can't split off itself
    new = lay.move("inspector", panel, "left")
    assert lay.areas["panel"].children == [new, panel]


def test_empty_area_keeps_a_group():
    lay = default()
    editor = lay.groups("editor")[0]
    for v in ["console", "inspector", "plot:1", "log"]:
        lay.move(v, editor)
    assert len(lay.groups("panel")) == 1 and lay.groups("panel")[0].views == []
    lay.move("log", lay.groups("panel")[0])
    assert lay.groups("panel")[0].views == ["log"]
    lay.remove("log")
    assert lay.groups("panel")[0].views == []
    with pytest.raises(ValueError):
        lay.move("log", editor, "middle")


def test_json_round_trip_drops_unknown_and_restores_homes():
    lay = default()
    editor = lay.groups("editor")[0]
    lay.move("plot:1", editor, "bottom")
    lay.move("plot:7", lay.groups("editor")[1])   # e.g. a plugin view
    data = lay.to_json()
    assert data["version"] == 2 and data["editor"]["split"] == "v"
    homes = {"viewport": "editor", "console": "panel", "log": "panel", "inspector": "panel", "plot:1": "panel"}
    back = Layout.from_json(data, known=lambda v: v != "plot:7", homes=homes)
    assert back.to_json()["editor"] == {"split": "v", "sizes": [1, 1], "children": [
        {"group": ["viewport"], "current": 0}, {"group": ["plot:1"], "current": 0}]}
    data["panel"] = {"group": ["console", "console", 3, "log"], "current": 9}   # damaged file
    back = Layout.from_json(data, homes=homes)
    assert back.groups("panel")[0].views == ["console", "log", "inspector"]
    assert back.find("plot:1") is not None
    with pytest.raises(ValueError):
        Layout.from_json({"version": 1})
