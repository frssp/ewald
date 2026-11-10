"""render() keywords (API review A2): a keyword no renderer knows raises on every renderer, before any
work, and the 3D view's setting names (``cell``, ``colors``, ...) are taken for the render keywords they
stand for. The ray tracer and vector files used to ignore unknown keywords, so ``sizee=`` or
``cell=False`` drew the default picture without a word."""
import inspect

import numpy as np
import pytest

import ewald
import ewald.examples
from ewald import render as render_mod
from ewald import view3d
from ewald.render import check_keywords, render, render_keywords

SMALL = {"size": (80, 60), "quality": "draft"}
needs_opengl = pytest.mark.skipif(not render_mod.opengl_available(), reason="needs an OpenGL context (xvfb-run)")
EVERY_RENDERER = ["raytrace", "vector", pytest.param("opengl", marks=needs_opengl)]


@pytest.fixture
def pipe():
    return ewald.load(ewald.examples.path("Cu_md.extxyz"))


@pytest.fixture
def data(pipe):
    return pipe.compute(0)


@pytest.fixture
def passed(monkeypatch):
    """What render() hands to the ray tracer, instead of a picture (fast: nothing is drawn)."""
    seen = []

    def fake(data, filename, **kwargs):
        seen.append(kwargs)
        return np.zeros((6, 8, 3), np.uint8)

    monkeypatch.setattr(render_mod, "_render_raytraced", fake)
    return seen


# ---------------------------------------------------------------- unknown keywords
@pytest.mark.parametrize("renderer", EVERY_RENDERER)
def test_unknown_keyword_raises_on_every_renderer(pipe, data, renderer):
    for call in (lambda **kw: pipe.render(None, **kw), lambda **kw: render(data, None, **kw)):
        with pytest.raises(TypeError, match=r"^render\(\) got an unexpected keyword argument 'bondz'\. "
                                            r"Did you mean 'bonds'\?$"):
            call(renderer=renderer, bondz=2.8, **SMALL)


def test_the_three_calls_of_the_review(pipe, tmp_path):
    out = tmp_path / "a.png"
    with pytest.raises(TypeError, match=r"'sizee'\. Did you mean 'size'\?"):
        pipe.render(out, renderer="raytrace", sizee=(100, 100))
    with pytest.raises(TypeError, match=r"'qualty'\. Did you mean 'quality'\?"):
        pipe.render(out, renderer="raytrace", qualty="draft")
    assert not out.exists()
    shown = pipe.render(None, renderer="raytrace", **SMALL)
    hidden = pipe.render(out, renderer="raytrace", cell=False, **SMALL)        # the view's name for show_cell
    assert out.exists()
    assert np.array_equal(hidden, pipe.render(None, renderer="raytrace", show_cell=False, **SMALL))
    assert not np.array_equal(hidden, shown)


@pytest.mark.parametrize("renderer", EVERY_RENDERER)
def test_cell_false_removes_the_cell(data, renderer):
    """Same camera for every picture, so only the cell lines can differ."""
    cam = render_mod.default_camera(data, (80, 60))
    kw = {"renderer": renderer, "camera": cam, "legend": False, **SMALL}
    hidden = render(data, None, cell=False, **kw)
    assert np.array_equal(hidden, render(data, None, show_cell=False, **kw))
    assert np.array_equal(render(data, None, cell=True, **kw), render(data, None, **kw))
    changed = (hidden != render(data, None, show_cell=True, **kw)).any(axis=-1)
    assert changed.sum() > 20                                                    # the cell lines are gone


def test_unknown_keyword_raises_before_any_work(pipe, data, monkeypatch, passed):
    def no_work(*args, **kwargs):
        raise AssertionError("the keywords are checked first")

    monkeypatch.setattr(render_mod, "opengl_available", no_work)
    monkeypatch.setattr(render_mod, "default_camera", no_work)
    monkeypatch.setattr("ewald.color_rule.take", no_work)
    with pytest.raises(TypeError, match="unexpected keyword argument 'colour_by'"):
        render(data, None, renderer="raytrace", colour_by="Position.Z")
    monkeypatch.setattr(type(pipe), "compute", no_work)                          # nor does the pipeline run
    with pytest.raises(TypeError, match="'sizee'"):
        pipe.render(None, renderer="raytrace", sizee=(100, 100))
    assert passed == []


def test_several_unknown_keywords_are_all_named(data):
    with pytest.raises(TypeError) as err:
        render(data, None, renderer="raytrace", bondz=2.8, xyzzy=1)
    text = str(err.value)
    assert "'bondz'. Did you mean 'bonds'?" in text and "'xyzzy'" in text
    # nothing close to xyzzy: the common keywords and where the rest are, not all eighty
    assert "Common keywords: filename, size, direction" in text and "sorted(ewald.render.render_keywords())" in text
    assert len(text) < 400
    with pytest.raises(TypeError) as err:
        render(data, None, renderer="raytrace", bondz=2.8)
    assert "keywords:" not in str(err.value)                                     # one line when a name is close


@pytest.mark.parametrize("wrong, meant", [("show_bonds", "bonds"), ("width", "size"), ("resolution", "size"),
                                          ("colour_by", "color_by"), ("bg", "background"), ("cmap", "colormap")])
def test_hints_name_the_keyword_that_was_meant(data, wrong, meant):
    """A close spelling of another keyword is no hint: show_bonds is not show_atoms."""
    with pytest.raises(TypeError, match=rf"'{wrong}'\. Did you mean '{meant}'\?"):
        render(data, None, renderer="raytrace", **{wrong: 1})


def test_view_setting_without_a_render_keyword_points_to_view(pipe, data):
    for name in ("style", "bond_factor", "colorbar_collapsed"):
        for call in (lambda **kw: pipe.render(None, **kw), lambda **kw: render(data, None, **kw)):
            with pytest.raises(TypeError, match=rf"unexpected keyword argument '{name}'\. It is a setting of the "
                                                rf"3D view .* view\(pipe, {name}=\.\.\.\)"):
                call(renderer="raytrace", **{name: view3d.SETTINGS[name].default})


def test_snapshot_checks_its_keywords_too(pipe):
    from ewald.views import Snapshot

    with pytest.raises(TypeError, match="'sizee'. Did you mean 'size'"):
        Snapshot(pipe, 0, sizee=(100, 100)).image(renderer="raytrace", **SMALL)
    with pytest.raises(TypeError, match="'bondz'. Did you mean 'bonds'"):
        pipe.view.snapshot(bondz=2.8)            # at once: a snapshot that cannot draw only shows its text
    assert pipe.view.snapshot(cell=False, **SMALL).kwargs == {"cell": False, **SMALL}


def test_snapshot_cell_with_a_wrong_keyword_raises_in_the_cell(pipe):
    """snapshot(pipe, sizee=...) as the last line of a cell: the picture swallows its errors (it
    falls back to text), so the call itself must raise. It used to draw with the typo ignored."""
    from ewald.views import snapshot

    for wrong, hint in ({"sizee": (100, 80)}, "Did you mean 'size'"), ({"style": "flat"}, r"view\(pipe, style="):
        with pytest.raises(TypeError, match=hint):
            snapshot(pipe, **wrong)
    assert snapshot(pipe, cell=False, **SMALL)._repr_png_()[:4] == b"\x89PNG"


def test_display_options_refuse_a_misspelt_key(pipe):
    from ewald import display

    with pytest.raises(TypeError, match=r"ewald\.display\.options has no option 'qualty'\. Did you mean 'quality'"):
        display.options["qualty"] = "draft"
    with pytest.raises(TypeError, match="'sizee'"):
        display.options.update(sizee=(10, 10))
    with pytest.raises(TypeError, match="'xyzzy'.*Options: enabled, size, renderer, quality, scale and the keywords"):
        display.options.setdefault("xyzzy", 1)
    assert "qualty" not in display.options and "sizee" not in display.options
    assert display.pipeline_png(pipe)[:4] == b"\x89PNG"


def test_movie_checks_its_keywords_before_any_work(pipe, tmp_path, monkeypatch):
    from ewald.movie import render_animation

    monkeypatch.setattr(type(pipe), "compute", lambda *a, **k: pytest.fail("computed a frame"))
    with pytest.raises(TypeError, match="'sizee'. Did you mean 'size'"):
        render_animation(pipe, tmp_path / "m.gif", sizee=(80, 60))
    assert not (tmp_path / "m.gif").exists()


def test_scene_builders_take_no_unknown_keywords(data):
    """Called directly they are as strict as render(): nothing is swallowed."""
    from ewald.raytrace import scene_from_data
    from ewald.vector import render_vector

    with pytest.raises(TypeError, match="bondz"):
        scene_from_data(data, bondz=2.8)
    with pytest.raises(TypeError, match="bondz"):
        render_vector(data, None, camera=render_mod.default_camera(data, (80, 60)), size=(80, 60), bondz=2.8)


# ---------------------------------------------------------------- the view's names
@pytest.mark.parametrize("name, key, value, expected", [
    ("cell", "show_cell", False, False),
    ("cell", "show_cell", "off", False),                 # checked as view(pipe, cell="off") checks it
    ("atoms", "show_atoms", False, False),
    ("colors", "type_colors", {"Cu": "red"}, {"Cu": "red"}),
    ("radii", "type_radii", {"Cu": 1}, {"Cu": 1.0}),
    ("transparency", "type_transparency", {"Cu": 0.5}, {"Cu": 0.5}),
    ("selection", "selection_color", True, "red"),
    ("selection", "selection_color", False, None),
    ("selection", "selection_color", "blue", "blue"),
])
def test_view_names_are_render_keywords(pipe, data, passed, name, key, value, expected):
    assert check_keywords({name: value}) == {key: expected}
    render(data, None, renderer="raytrace", **{name: value})
    render(data, None, renderer="raytrace", **{key: expected})
    pipe.render(None, renderer="raytrace", **{name: value})                      # an override of the view
    pipe.render(None, renderer="raytrace", **{key: expected})
    assert passed[0][key] == expected
    assert passed[0] == passed[1] and passed[2] == passed[3]


def test_view_name_checks_its_value_like_the_view(data):
    with pytest.raises(ValueError, match="transparency must be in 0..1"):
        render(data, None, renderer="raytrace", transparency={"Cu": 7})
    with pytest.raises(ValueError, match="selection is True, False or a color"):
        render(data, None, renderer="raytrace", selection="no such color")


@pytest.mark.parametrize("name, key", sorted(view3d._RENDER_NAMES.items()))
def test_both_spellings_raise(pipe, data, passed, name, key):
    value = view3d.SETTINGS[name].default
    for call in (lambda **kw: pipe.render(None, **kw), lambda **kw: render(data, None, **kw)):
        with pytest.raises(TypeError, match=rf"render\(\) got both '{name}' and '{key}'"):
            call(renderer="raytrace", **{name: value, key: value})
    assert passed == []


def test_show_cell_next_to_cell_raises_whatever_its_value(data, passed):
    for show_cell in (True, False):
        with pytest.raises(TypeError, match="got both 'cell' and 'show_cell'"):
            render(data, None, renderer="raytrace", show_cell=show_cell, cell=False)
    render(data, None, renderer="raytrace")
    assert passed[0]["show_cell"] is True                                        # the default, as before


def test_view_and_override_do_not_count_as_both_spellings(pipe, passed):
    """The view's own show_cell and a cell= override are one setting given once."""
    pipe.view.update(cell=True, colors={"Cu": "blue"}, selection="green")
    pipe.render(None, renderer="raytrace", cell=False, colors={"Cu": "red"}, selection=False)
    assert passed[0]["show_cell"] is False and passed[0]["type_colors"] == {"Cu": "red"}
    assert passed[0]["selection_color"] is None
    from ewald.views import Snapshot

    Snapshot(pipe, 0, cell=False).image(renderer="raytrace", atoms=False)
    assert passed[1]["show_cell"] is False and passed[1]["show_atoms"] is False


def test_display_options_take_the_views_names(pipe, passed, monkeypatch):
    """ewald.display.options holds "any render keyword": layered under the view's own show_cell, a
    view-named option must not count as the second spelling (the picture would fall back to text)."""
    from ewald import display
    from ewald.views import Snapshot

    monkeypatch.setitem(display.options, "cell", False)
    assert display.pipeline_png(pipe)[:4] == b"\x89PNG"
    assert passed[0]["show_cell"] is False                   # an option the user added wins over the view
    Snapshot(pipe, 0).image()
    assert passed[1]["show_cell"] is True                    # under a snapshot the view wins, as for show_cell


def test_merge_reads_the_views_names():
    merge = view3d.merge
    assert merge({"show_cell": True, "zoom": 2.0}, {"cell": False}) == {"show_cell": False, "zoom": 2.0}
    # layered defaults (ewald.display.options under the view's kwargs): the render keyword next to it wins
    assert merge({"cell": False, "show_cell": True}, {}) == {"show_cell": True}
    assert merge({"cell": False, "show_cell": True}, {"cell": False}) == {"show_cell": False}
    assert merge({"cell": False}, {}) == {"cell": False}                         # render() reads it
    with pytest.raises(TypeError, match="got both 'colors' and 'type_colors'"):
        merge({}, {"colors": {}, "type_colors": {}})


# ---------------------------------------------------------------- the known keywords are the real ones
def test_known_keywords_cover_everything_a_renderer_accepts():
    """Fails when a renderer (or the scene underneath) gains a keyword render() would refuse, or a way
    to swallow unknown ones again."""
    from ewald import scene

    known = render_keywords()
    own = set(inspect.signature(render).parameters)
    for renderer in render_mod.RENDERERS:
        params = inspect.signature(render_mod._builder(renderer)).parameters.values()
        assert not [p.name for p in params if p.kind is p.VAR_KEYWORD], f"{renderer} swallows unknown keywords"
        accepted = {p.name for p in params if p.kind is p.KEYWORD_ONLY}
        assert accepted - render_mod._BUILDER_ONLY <= known, renderer
        assert render_mod._BUILDER_ONLY.isdisjoint(own)
    # what the scene takes must come through a renderer's builder: a keyword only scene.build knows
    # would be refused by render() and swallowed by builtin_elements(**_ignored)
    for func in (scene.build, scene.builtin_elements):
        accepted = {p.name for p in inspect.signature(func).parameters.values() if p.kind is p.KEYWORD_ONLY}
        assert accepted <= known, func.__name__
    assert set(render_mod._CAMERA_KEYS) <= known


def test_color_rule_keywords_are_the_ones_take_pops(data):
    from ewald.color_rule import take

    kw = dict.fromkeys(render_mod._COLOR_RULE_KEYS)
    take(data, kw)
    assert kw == {} and set(render_mod._COLOR_RULE_KEYS) <= render_keywords()


def test_every_view_setting_is_a_keyword_an_alias_or_explained():
    known = render_keywords()
    sig = inspect.signature(render).parameters
    for name, key in view3d._RENDER_NAMES.items():
        assert name in view3d.SETTINGS and name not in known and key in known
        # a named parameter must be able to tell "not given", or both spellings cannot be refused
        assert key not in sig or sig[key].default is None
    for name in view3d.SETTINGS:
        if name in known or name in view3d._RENDER_NAMES:
            continue
        with pytest.raises(TypeError, match=rf"view\(pipe, {name}=\.\.\.\)"):
            check_keywords({name: 1})


def test_what_the_callers_pass_is_known(pipe, data):
    """View.render_kwargs (pipe.render, snapshots, movies, the AI's render_image), ViewStyle.scene_kwargs
    (sessions, the GUI) and ewald.display.options must never trip the check."""
    from ewald import display
    from ewald.modifiers import GenerateTrajectoryLines
    from ewald.view_style import ViewStyle

    known = render_keywords()
    pipe.add(GenerateTrajectoryLines())
    v = pipe.view
    v.update(theme="VESTA", atom_style="mesh", sphere_resolution=12, background="white", light=[1, 1, 1],
             atoms=False, color_by="Position.Z", colormap="magma", color_range=[0, 1], colors={"Cu": "red"},
             radii={"Cu": 1.0}, radius_preset="ionic", transparency={"Cu": 0.5}, selection="blue", labels=True,
             oxidation_states={"Cu": 2}, bonds=True, bond_radius=0.1, cell=False, cell_width=3.0,
             cell_color="red", cell_radius=0.1, boundary=[-0.1, 1.1], boundary_bonds="complete",
             replicate=[2, 1, 1], replicate_cell="unit", vectors="Position", surfaces={"s": {"visible": False}},
             measurements=[[0, 1]], trajectory_width=3.0, trajectory_color_by="Time",
             trajectory_colormap="magma", trajectory_color_range=[0, 1], trajectory_time_range=[0, 1],
             trajectory_trail=True, trajectory_style="tube", trajectory_radius=0.1, frame=1, direction="x",
             zoom=2.0, orthographic=False, region=[0, 0, 0.5], renderer="raytrace", quality="draft",
             size=[80, 60], ambient_occlusion=False, shadows=False, transparent_background=True, text="t",
             legend=False, type_legend=True, scale_bar=True, overlays=["Coordinate tripod"])
    kw = v.render_kwargs(data)
    assert len(kw) > 45 and set(kw) <= known
    assert check_keywords(kw) == kw                                              # and nothing is renamed
    v.update(camera=render_mod.default_camera(data, (80, 60)), bonds=2.8)
    assert set(v.render_kwargs(data)) <= known
    style = ViewStyle(type_colors={"Cu": "red"}, type_radii={"Cu": 1.0}, cell_color="red", vectors="Position",
                      trajectory_width=3.0, show_atoms=False, sphere_resolution=12, cell_radius=0.1,
                      surfaces={"s": {"visible": False}}, type_transparency={"Cu": 0.5}, labels=True,
                      measurements=[[0, 1]], visuals=[{"name": "x"}])
    assert set(style.scene_kwargs()) <= known
    assert set(display.options) - {"enabled"} <= known


@pytest.mark.parametrize("renderer", EVERY_RENDERER)
def test_every_known_keyword_draws_on_every_renderer(data, renderer):
    """One call works with any renderer: a keyword only another renderer uses (shading, mesh_limit,
    cell_radius, ...) is known and ignored, never an error from the builder underneath."""
    defaults = {}
    for r in render_mod.RENDERERS:
        for p in inspect.signature(render_mod._builder(r)).parameters.values():
            if p.kind is p.KEYWORD_ONLY and p.default is not p.empty:
                defaults.setdefault(p.name, p.default)
    own = set(inspect.signature(render).parameters)
    scene_only = {k: defaults[k] for k in render_keywords() - own - set(render_mod._COLOR_RULE_KEYS)}
    assert {"shading", "mesh_limit", "cell_radius", "arrow_resolution", "atom_style"} <= set(scene_only)
    img = render(data, None, renderer=renderer, color_by=None, colormap="viridis", color_range=None,
                 **scene_only, **SMALL)
    assert img.shape == (30, 40, 3) and img.std() > 5
    plain = render(data, None, renderer=renderer, **SMALL)
    assert np.array_equal(img, plain)                                            # defaults: the same picture


# ---------------------------------------------------------------- print(pipe.render(...))
def test_printing_an_image_is_one_line(pipe, tmp_path):
    out = tmp_path / "a.png"
    img = pipe.render(out, renderer="raytrace", **SMALL)
    assert str(img) == repr(img) == f"<Image 40x30 RGB, saved to {out}>"
    assert f"{img}" == str(img)
    grey = img.astype(float).mean(axis=2)                                        # computed arrays print as arrays
    assert str(grey) == str(np.asarray(grey)) and "Image" not in str(grey)
