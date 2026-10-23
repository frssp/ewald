"""Color maps (ewald.colormaps), themes (ewald.themes) and overlay elements (ewald.overlays)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource, plugins
from ewald.colormaps import get_colormap, is_perceptual, lightness, names
from ewald.overlays import (Attributes, Circle, CoordinateTripod, Image, OverlayContext, OverlayElement, Polyline,
                            Rect, Text, coerce_overlays, draw_overlays, overlay_items, overlay_to_dict)
from ewald.themes import Theme, apply_theme, element_colors, get_theme

EXTRAS = Path(__file__).parents[1] / "examples" / "plugins" / "figure_extras.py"
CAMERA = {"position": (30.0, 25.0, 20.0), "focal_point": (2.8, 2.8, 2.8), "up": (0.0, 0.0, 1.0),
          "parallel_projection": True, "parallel_scale": 6.0, "view_angle": 30.0}
TOP = {"position": (0.0, 0.0, 50.0), "focal_point": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0),
       "parallel_projection": True, "parallel_scale": 6.0, "view_angle": 30.0}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


@pytest.fixture
def extras():
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXTRAS, folder)
    plugins.load_plugins(force=True)
    assert not plugins.status()["errors"], plugins.status()["errors"]


def _crystal():
    from ase.build import bulk

    data = DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    data.attributes["Temperature"] = 300.0
    data.attributes["Timestep"] = 1200
    return data


# ---------------------------------------------------------------- color maps
def test_colormap_plugins_resolve_everywhere():
    plugins.register_colormap("two", ["navy", "gold"], kind="sequential")

    @plugins.colormap(kind="diverging")
    def bwr3():
        """blue white red"""
        return [(0, 0, 1), (1, 1, 1), (1, 0, 0)]

    assert np.allclose(get_colormap("two")(0.0)[:3], (0, 0, 0.5), atol=0.01)
    assert np.allclose(get_colormap("two_r")(0.0)[:3], get_colormap("two")(1.0)[:3])
    assert np.allclose(get_colormap("bwr3")(0.5)[:3], 1, atol=0.01)
    assert plugins.colormaps()["bwr3"].description == "blue white red"
    assert get_colormap("viridis").name == "viridis"
    assert names()[0] == "viridis" and {"two", "bwr3"} <= set(names())
    with pytest.raises(KeyError, match="unknown color map"):
        get_colormap("no-such-map")
    with pytest.raises(ValueError, match="kind"):
        plugins.register_colormap("x", ["red", "blue"], kind="wavy")
    with pytest.raises(ValueError, match="two colors"):
        plugins.register_colormap("y", ["red"])


def test_colormap_reaches_color_coding_and_legend():
    from ewald.modifiers import ColorCoding
    from ewald.render import render

    plugins.register_colormap("two", ["navy", "gold"])
    pipe = Pipeline(StaticSource(_crystal()))
    pipe.modifiers.append(ColorCoding("Position.X", cmap="two"))
    data = pipe.compute()
    x = data.positions[:, 0]
    t = (x - x.min()) / np.ptp(x)
    assert np.allclose(data["Color"], get_colormap("two")(t)[:, :3])
    img = render(data, size=(160, 120), renderer="vector", camera=CAMERA)
    from ewald.overlays import ColorRamp, overlay_layout

    ramp = next(it for it in overlay_layout(data, (160, 120))["items"] if isinstance(it, ColorRamp))
    bar = img[int(ramp.y0) + 3:int(ramp.y1) - 3, int((ramp.x0 + ramp.x1) / 2)]   # the color bar, top to bottom
    assert bar[0, 2] < 80 and bar[0, 0] > 150                 # gold at the top
    assert bar[-1, 2] > 60 and bar[-1, 0] < 60                # navy at the bottom


def test_lightness_and_grayscale_check():
    lum = lightness("Greys")
    assert lum[0] > 95 and lum[-1] < 5
    assert is_perceptual("viridis") and is_perceptual("cividis")
    assert not is_perceptual("jet") and not is_perceptual("hsv")
    assert is_perceptual("RdBu_r", "diverging") and not is_perceptual("viridis", "cyclic")


# ---------------------------------------------------------------- themes
def test_builtin_themes_and_schemes():
    reg = plugins.themes()
    assert {"Default", "VESTA", "CPK", "Journal", "Dark slide"} <= set(reg) and reg["VESTA"].builtin
    assert np.allclose(element_colors("VESTA")["O"], (0.9961, 0.0118, 0.0))
    from ase.data.colors import jmol_colors

    assert np.allclose(get_theme("Default").color_of("O"), jmol_colors[8])
    journal = get_theme("Journal").color_of("O")
    gray = np.dot(jmol_colors[8], [0.2126, 0.7152, 0.0722])
    assert np.allclose(journal, gray + 0.65 * (jmol_colors[8] - gray))
    assert get_theme("VESTA").color_of("Fe3+") == get_theme("VESTA").color_of("Fe")   # ions map to elements
    assert get_theme("VESTA").color_of("X99") is None


def test_apply_theme_lets_explicit_settings_win():
    data = _crystal()
    kw = apply_theme("Dark slide", data, {"background": None, "type_colors": {"Na": "#00ff00"}, "cell_color": None})
    assert kw["background"] == "black" and kw["cell_color"] == "#AEB4BE"
    assert kw["type_colors"]["Na"] == "#00ff00" and "Cl" in kw["type_colors"]
    kw = apply_theme("Dark slide", data, {"background": "white"})
    assert kw["background"] == "white"
    with pytest.raises(KeyError, match="unknown theme"):
        apply_theme("Nope", data, {})


def test_theme_plugins(extras):
    t = get_theme("Group style")
    assert t.base == "VESTA" and np.allclose(t.color_of("Li"), (0x7f / 255, 0x3f / 255, 0xbf / 255))
    with pytest.raises(ValueError, match="built-in"):
        plugins.register_theme("VESTA", Theme())
    with pytest.raises(TypeError):
        plugins.register_theme("bad", 42)
    plugins.register_theme("broken", lambda: "not a theme")
    with pytest.raises(TypeError, match="not a Theme"):
        get_theme("broken")


@pytest.mark.parametrize("renderer", ["raytrace", "vector"])
def test_render_theme(renderer):
    from ewald.render import render

    data = _crystal()
    kw = dict(size=(120, 90), renderer=renderer, camera=CAMERA, quality="draft", legend=False)
    plain, dark = render(data, **kw), render(data, theme="Dark slide", **kw)
    assert plain[0, 0].min() > 240 and dark[0, 0].max() < 15          # the corner is background
    vesta = render(data, theme="VESTA", **kw)
    assert np.abs(plain.astype(int) - vesta.astype(int)).sum() > 0   # other element colors
    assert np.array_equal(render(data, theme="VESTA", background="white", **kw), vesta)


def test_view_style_takes_a_theme():
    from ewald.gui.style import ViewStyle

    style = ViewStyle(type_colors={"Na": (1, 0, 0)})
    style.apply_theme("Dark slide", ["Na", "Cl"])
    assert style.theme == "Dark slide" and style.background == "black" and style.cell_color == "#AEB4BE"
    assert style.type_colors["Na"] != (1, 0, 0)
    assert style.scene_kwargs()["cell_color"] == "#AEB4BE"


# ---------------------------------------------------------------- overlays
def test_coordinate_tripod_follows_the_camera():
    ctx = OverlayContext((400, 300), TOP, font_size=12, small=10)
    items = CoordinateTripod(corner="bottom-left").layout(_crystal(), ctx)
    lines = {it.color: np.asarray(it.points, float) for it in items if isinstance(it, Polyline) and it.fill is None}
    x, y = lines[(0.85, 0.15, 0.15)], lines[(0.15, 0.6, 0.2)]
    assert x[1, 0] - x[0, 0] > 20 and abs(x[1, 1] - x[0, 1]) < 1         # x points right
    assert y[0, 1] - y[1, 1] > 20 and abs(y[1, 0] - y[0, 0]) < 1         # y points up (screen y is down)
    assert x[0, 0] < 200 and x[0, 1] > 150                               # bottom-left corner
    labels = [it.text for it in items if isinstance(it, Text)]
    assert sorted(labels) == ["x", "y", "z"]
    abc = CoordinateTripod(axes="abc").layout(_crystal(), ctx)
    assert sorted(it.text for it in abc if isinstance(it, Text)) == ["a", "b", "c"]
    assert CoordinateTripod().layout(_crystal(), OverlayContext((400, 300))) == []   # no camera, no tripod


def test_attributes_readout():
    ctx = OverlayContext((400, 300), font_size=12, small=10)
    items = Attributes().layout(_crystal(), ctx)
    assert [it.text for it in items] == ["Temperature = 300", "Timestep = 1200"]
    assert all(it.anchor == "rt" and it.x > 300 for it in items)
    items = Attributes("Temperature, Missing", corner="bottom-left", format="{:.1f}").layout(_crystal(), ctx)
    assert [it.text for it in items] == ["Temperature = 300.0"] and items[0].y > 200


def test_coerce_round_trip_and_errors(extras):
    (el,) = coerce_overlays([{"name": "Coordinate tripod", "params": {"axes": "abc"}, "enabled": False}])
    d = overlay_to_dict(el)
    assert d == {"name": "Coordinate tripod", "params": {"axes": "abc", "corner": "bottom-right", "size": 0.09},
                 "enabled": False}
    ctx = OverlayContext((200, 150), CAMERA)
    assert overlay_items(_crystal(), ctx, [d]) == []                 # disabled
    with pytest.raises(KeyError, match="no overlay"):
        coerce_overlays(["Nope"])
    assert plugins.overlays()["Inset g(r)"].category == "Plots"

    class Bad(OverlayElement):
        def layout(self, data, ctx):
            return [np.zeros(3)]

    with pytest.raises(TypeError, match="not an overlay item"):
        overlay_items(_crystal(), ctx, [Bad()])
    with pytest.raises(ValueError, match="built-in"):
        plugins.register_overlay(type("Attributes", (OverlayElement,), {"name": "Attributes"}))
    with pytest.raises(TypeError):
        plugins.register_overlay(object)


def test_every_item_kind_draws_with_pil():
    class All(OverlayElement):
        def layout(self, data, ctx):
            pic = np.zeros((4, 4, 3), np.uint8)
            pic[..., 1] = 255
            return [Rect(10, 10, 30, 30, fill="red", outline="ink"), Circle(50, 20, 8, fill="blue"),
                    Polyline([[60, 10], [80, 30], [60, 30]], closed=True, fill="yellow", width=0),
                    Text(5, 45, "hi", 12, "ink"), Image(90, 10, pic, 20, 20)]

    img = draw_overlays(np.full((60, 120, 3), 255, np.uint8), None, legend=False, overlays=[All()])
    assert tuple(img[20, 20, :3]) == (255, 0, 0) and tuple(img[20, 50, :3]) == (0, 0, 255)
    assert tuple(img[26, 64, :3]) == (255, 255, 0) and tuple(img[20, 100, :3]) == (0, 255, 0)
    assert img[40:55, 5:20, :3].min() < 100                         # the text in ink


@pytest.mark.parametrize("renderer", ["raytrace", "vector"])
def test_plugin_overlay_reaches_renderers(extras, renderer):
    from ewald.render import render

    data = _crystal()
    kw = dict(size=(200, 150), renderer=renderer, camera=CAMERA, quality="draft", legend=False)
    plain = render(data, **kw)
    with_inset = render(data, overlays=["Inset g(r)", "Coordinate tripod"], **kw)
    assert np.abs(plain.astype(int) - with_inset.astype(int)).sum() > 0


def test_session_keeps_overlays(tmp_path):
    from ewald.session import load_session, save_session

    ov = [{"name": "Attributes", "params": {"attributes": "Temperature"}, "enabled": True}]
    path = save_session(tmp_path / "s.ewald", Pipeline(StaticSource(_crystal())), render={"overlays": ov})
    assert load_session(path).render_kwargs()["overlays"] == ov


# ---------------------------------------------------------------- CLI and Claude tools
def test_cli_lists_and_renders(extras, tmp_path, capsys):
    from ase.io import write

    from ewald.__main__ import main

    main(["plugins"])
    out = capsys.readouterr().out
    assert "overlay Coordinate tripod" in out and "overlay Inset g(r)" in out
    assert "colormap charge_difference" in out and "theme Group style" in out and "theme VESTA" in out
    src = tmp_path / "nacl.xyz"
    write(src, _crystal().to_ase())
    out = tmp_path / "fig.svg"
    assert not main(["render", str(src), str(out), "--renderer", "vector", "--size", "160", "120",
                     "--theme", "Dark slide", "--overlay", "Coordinate tripod", "--overlay", "Attributes"])
    svg = out.read_text()
    assert "#aeb4be" in svg                                           # the theme's cell color
    assert "fill: #ffffff" not in svg.split('id="axes_1"')[0]          # black page (SVG's default fill)


def test_llm_tools(extras, monkeypatch):
    from ewald.llm import Toolset
    from ewald.llm import tools as tools_mod

    seen = {}

    def render_png(data, path, **kw):
        seen.update(kw)
        from PIL import Image as PILImage

        PILImage.new("RGB", (4, 4)).save(path)

    monkeypatch.setattr(tools_mod, "render_png", render_png)
    ts = Toolset(Pipeline(StaticSource(_crystal())))
    text = ts.call("list_styles", {}).text()
    assert "Group style" in text and "charge_difference" in text and "grayscale_safe" in text
    assert "Inset g(r)" in ts.call("list_overlays", {}).text()
    assert not ts.call("add_overlay", {"name": "Coordinate tripod", "params": {"axes": "abc"}}).is_error
    assert ts.call("add_overlay", {"name": "Nope"}).is_error
    assert ts.call("render_image", {"theme": "Nope"}).is_error
    res = ts.call("render_image", {"theme": "VESTA"})
    assert not res.is_error, res.text()
    assert seen["theme"] == "VESTA" and seen["overlays"][0]["name"] == "Coordinate tripod"
    assert ts.pipeline.view.theme is None                    # render_image's theme is for that picture only
    script = ts.to_script()   # the overlay went into the notebook's view cell
    assert "overlays=[{'name': 'Coordinate tripod'" in script and "theme" not in script
    assert not ts.call("remove_overlay", {"index": 0}).is_error and ts.pipeline.view.overlays == []
    assert "overlays" not in ts.to_script()


UI_THEMES = Path(__file__).parents[1] / "examples" / "plugins" / "ui_themes.py"


def test_ui_theme_plugins(tmp_path):
    from ewald import settings
    from ewald.gui import theme as ui

    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(UI_THEMES, folder)
    plugins.load_plugins(force=True)
    assert not plugins.status()["errors"], plugins.status()["errors"]
    names = list(plugins.ui_themes())
    assert names[:len(ui.PALETTES)] == list(ui.PALETTES) and {"Nord", "Sepia"} <= set(names)
    assert any(t["name"] == "Nord" and t["base"] == "graphite" for t in plugins.status()["ui_themes"])

    # overrides on top of the base palette; every token present
    p = ui.palette("Nord")
    assert set(p) == set(ui.PALETTES["paper"]) and p["accent"] == "#88C0D0"
    assert p["error"] == ui.PALETTES["graphite"]["error"]
    assert ui.use("Sepia")["text"] == "#2B2118" and ui.TOKENS["text"] == "#2B2118"
    ui.use("paper")

    # a dict may name its base; bad themes fall back to the default instead of failing
    plugins.register_ui_theme("Dict", {"base": "graphite", "accent": "#123456"})
    assert ui.palette("Dict")["bg"] == ui.PALETTES["graphite"]["bg"]
    plugins.register_ui_theme("Typo", {"acent": "#123456"})
    plugins.register_ui_theme("Not hex", {"accent": "teal"})
    plugins.register_ui_theme("Bad base", {"accent": "#123456"}, base="no such base")
    for name in ("Typo", "Not hex", "Bad base"):
        with pytest.raises(ValueError):
            plugins.ui_themes()[name].palette()
        assert ui.palette(name) == ui.PALETTES[ui.DEFAULT]
    assert ui.palette("no such theme") == ui.PALETTES[ui.DEFAULT]
    with pytest.raises(ValueError, match="built in"):
        plugins.register_ui_theme("graphite", {})
    with pytest.raises(TypeError):
        plugins.register_ui_theme("bad", 42)

    s = settings.Settings(path=tmp_path / "settings.json")
    s["general.ui_theme"] = "Nord"
    assert s["general.ui_theme"] == "Nord"


def test_ui_theme_style_keys():
    """A UI theme can set fonts, sizes, corner radius, spacing, icon stroke and extra QSS."""
    from ewald.gui import icons
    from ewald.gui import theme as ui
    from ewald.ui_tokens import STYLE_DEFAULTS, check_style

    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(UI_THEMES, folder)
    plugins.load_plugins(force=True)
    studio = plugins.ui_themes()["Studio"]
    st = studio.style()
    assert st["radius"] == 2 and st["density"] == "compact" and st["font"] == "Inter"
    assert set(studio.palette()) == set(ui.PALETTES["graphite"])    # style keys are not tokens
    assert ui.style("graphite") == STYLE_DEFAULTS and ui.style("Nord") == STYLE_DEFAULTS

    try:
        plain = ui.stylesheet()
        ui.use("Studio")
        css = ui.stylesheet()
        assert '"Inter", "IBM Plex Sans"' in css and '"Fira Code", "JetBrains Mono",' in css
        assert "border-radius: 6px" in plain and "border-radius: 6px" not in css
        assert "border-radius: 2px" in css
        assert "padding: 3px 10px" in plain and "padding: 2px 6px" in css     # compact: x0.6
        assert css.rstrip().endswith("QWidget#panelHeader { border-bottom: 2px solid #E0A040; }")
        assert 'stroke-width="1.4"' in icons.svg("chart", "#ffffff")
    finally:
        ui.use("paper")
    assert 'stroke-width="1.7"' in icons.svg("chart", "#ffffff")

    for bad in ({"radius": 40}, {"density": "tiny"}, {"font": 3}, {"icon_stroke": 9}, {"font_size": -1}):
        with pytest.raises(ValueError):
            check_style("bad", bad)
    plugins.register_ui_theme("Broken style", {"accent": "#123456", "radius": "round"})
    assert ui.style("Broken style") == STYLE_DEFAULTS     # logged, defaults used
    assert ui.palette("Broken style")["accent"] == "#123456"
