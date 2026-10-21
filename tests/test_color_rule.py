"""The color rule behind the atoms' colors (ewald.color_rule): recorded by steps, read back for legends,
the View tab and the AI, and the view's display-only color_by."""
import json

import numpy as np
import pytest

from ewald import Pipeline, StaticSource, color_rule, view
from ewald import modifiers as mods


@pytest.fixture
def cu_pipe(cu_fcc, rng):
    data = cu_fcc.copy()
    pos = data.positions.copy()
    pos[:20] += rng.normal(0, 0.6, (20, 3))       # some disordered atoms: OTHER
    data.positions = pos
    return Pipeline(StaticSource([data]))


def test_structure_identification_records_its_colors(cu_pipe):
    cu_pipe.add(mods.CommonNeighborAnalysis())
    data = cu_pipe.compute(0)
    r = color_rule.rule(data)
    assert r["kind"] == "categorical" and r["property"] == "Structure Type"
    assert "CommonNeighborAnalysis" in r["source"]
    entries = {e["name"]: e for e in r["entries"]}
    assert entries["FCC"]["color"] == "#66ff66" and entries["OTHER"]["color"] == "#f2f2f2"
    assert sum(e["count"] for e in r["entries"]) == data.count
    assert entries["FCC"]["count"] == data.attributes["CommonNeighborAnalysis.counts.FCC"]
    text = color_rule.describe(data)
    assert "'Structure Type'" in text and "1 FCC = #66ff66" in text
    rows = color_rule.legend_entries(r)          # only the types that have atoms
    assert [name for name, _ in rows] == [color_rule.label(e["name"]) for e in r["entries"] if e["count"]]
    json.dumps(data.attributes[color_rule.KEY])  # travels with kernel frames (plain JSON)


def test_identify_diamond_labels():
    assert color_rule.label("CUBIC_DIAMOND_FIRST_NEIGHBOR") == "Cubic diamond (1st neighbor)"
    assert color_rule.label("HEX_DIAMOND") == "Hex diamond"
    assert color_rule.label("FCC") == "FCC" and color_rule.label("Coordination < 4") == "Coordination < 4"


def test_no_colors_means_type_colors(cu_pipe):
    data = cu_pipe.compute(0)
    r = color_rule.rule(data, {"Cu": "red"})
    assert r["kind"] == "type" and r["entries"][0]["color"] == "#ff0000" and r["entries"][0]["count"] == data.count
    assert color_rule.legend_entries(r) == []    # the type legend is its own overlay
    cu_pipe.add(mods.CommonNeighborAnalysis(color_by_type=False))
    assert color_rule.rule(cu_pipe.compute(0))["kind"] == "type"


def test_color_coding_and_paint(cu_pipe):
    from ewald.paint import paint

    cu_pipe.add(mods.ColorCoding("Position.Z", cmap="magma", start=0.0))
    r = color_rule.rule(cu_pipe.compute(0))
    assert r["kind"] == "continuous" and r["property"] == "Position.Z" and r["cmap"] == "magma"
    assert r["range"][0] == 0.0 and r["range"][1] > 10

    @cu_pipe.add
    def surface(frame, data):
        paint(data, "Position.X < 3", "gold")

    data = cu_pipe.compute(0)
    r = color_rule.rule(data)
    assert r["kind"] == "continuous" and r["painted"][0]["where"] == "Position.X < 3"
    assert r["painted"][0]["count"] == int((data.positions[:, 0] < 3).sum()) > 0
    assert "paint() painted" in color_rule.describe(r)
    assert color_rule.legend_entries(r)[0][0] == "Position.X < 3"


def test_unrecorded_colors_are_custom(cu_pipe):
    cu_pipe.add(mods.CommonNeighborAnalysis())

    @cu_pipe.add
    def gray(frame, data):
        data["Color"] = np.full((data.count, 3), 0.5)

    r = color_rule.rule(cu_pipe.compute(0))
    assert r["kind"] == "custom" and "CommonNeighborAnalysis" in r["overwrote"]
    assert r["colors"] == [{"color": "#808080", "count": cu_pipe.compute(0).count}]


def test_assign_color_on_selection(cu_pipe):
    cu_pipe.add(mods.CommonNeighborAnalysis())
    cu_pipe.add(mods.ExpressionSelection("Position.X < 2"))
    cu_pipe.add(mods.AssignColor("red"))
    r = color_rule.rule(cu_pipe.compute(0))
    assert r["kind"] == "categorical" and r["painted"][0]["where"] == "selected"
    assert r["painted"][0]["color"] == "#ff0000"


def test_view_color_by(cu_pipe):
    cu_pipe.add(mods.CommonNeighborAnalysis())
    data = cu_pipe.compute(0)
    typed = color_rule.apply(data, "type")
    assert "Color" not in typed.particles and "Color" in data.particles   # display copy only
    assert color_rule.rule(typed)["kind"] == "type"
    mapped = color_rule.apply(data, "Position.Z", "plasma", [None, 5.0])
    r = color_rule.rule(mapped)
    assert r["kind"] == "continuous" and r["cmap"] == "plasma" and r["range"][1] == 5.0
    assert "view" in r["source"] and mapped.attributes["ColorCoding.range"][1] == 5.0
    assert color_rule.apply(data, None) is data

    v = view(cu_pipe, color_by="Structure Type", colormap="plasma", color_range=[0, 4])
    assert v.changed() == {"color_by": "Structure Type", "colormap": "plasma", "color_range": [0.0, 4.0]}
    kw = v.render_kwargs(data)
    assert kw["color_by"] == "Structure Type" and kw["color_range"] == [0.0, 4.0]
    with pytest.raises(ValueError):
        view(cu_pipe, color_range=[5, 1])
    v.update(color_by=None)
    assert "colormap" not in v.render_kwargs(data)   # only with color_by


def test_view_style_round_trip(cu_pipe):
    from ewald.view_style import ViewStyle

    v = view(cu_pipe, color_by="Coordination", colormap="magma", color_range=[None, 12])
    st = ViewStyle()
    st.apply_view(v)
    kw = st.scene_kwargs()
    assert kw["color_by"] == "Coordination" and kw["colormap"] == "magma" and kw["color_range"] == [None, 12.0]
    out = st.view_settings()
    assert out["color_by"] == "Coordination" and out["color_range"] == [None, 12.0]
    assert "color_by" not in ViewStyle().scene_kwargs()


def test_legend_items_in_pictures(cu_pipe):
    from ewald.overlays import Rect, Text, overlay_layout

    data = cu_pipe.compute(0)
    assert "items" not in overlay_layout(data, (400, 300))       # type colors: no swatches
    cu_pipe.add(mods.CommonNeighborAnalysis())
    data = cu_pipe.compute(0)
    lay = overlay_layout(data, (400, 300))
    texts = [it.text for it in lay["items"] if isinstance(it, Text)]
    assert texts[0] == "Structure Type" and "FCC" in texts and "Other" in texts
    assert any(isinstance(it, Rect) and it.fill == pytest.approx((0.4, 1.0, 0.4)) for it in lay["items"])
    assert "items" not in overlay_layout(data, (400, 300), legend=False)


def test_vector_picture_has_the_legend(cu_pipe, tmp_path):
    cu_pipe.add(mods.CommonNeighborAnalysis())
    path = tmp_path / "cna.svg"
    cu_pipe.render(path, renderer="vector", size=(400, 300))
    svg = path.read_text()
    assert "Structure Type" in svg and "FCC" in svg


def test_ai_tools_say_how_atoms_are_colored(cu_pipe):
    from ewald.llm.tools import Toolset

    cu_pipe.add(mods.CommonNeighborAnalysis())
    ts = Toolset(cu_pipe)
    frame = json.loads(ts.call("describe_frame", {"cutoff": None}).text())
    assert "'Structure Type'" in frame["atom_colors"] and "FCC = #66ff66" in frame["atom_colors"]
    got = json.loads(ts.call("get_view", {}).text())
    assert got["atom_colors"]["rule"]["kind"] == "categorical"
    ts.call("set_view", {"color_by": "type"})
    got = json.loads(ts.call("get_view", {}).text())
    assert got["atom_colors"]["rule"]["kind"] == "type" and got["changed"] == {"color_by": "type"}
