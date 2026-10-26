"""ewald.colors: one parser for every color argument (view settings, steps, overlays, GUI forms)."""
import numpy as np
import pytest
from ase import Atoms

from ewald import DataCollection, Pipeline, StaticSource, view
from ewald.colors import is_color, normalize, to_hex, to_rgb, to_rgba

ORANGE = (1.0, 0x88 / 255, 0.0)


@pytest.mark.parametrize("color", ["#ff8800", "#FF8800", "#f80", "ff8800", "FF8800", "rgb(255, 136, 0)",
                                   "RGB(255,136,0)", "rgba(255, 136, 0, 1)", "255, 136, 0", "(255, 136, 0)",
                                   (255, 136, 0), [1.0, 136 / 255, 0.0], np.array([255, 136, 0]),
                                   (255, 136, 0, 255), 0xFF8800, b"#ff8800"])
def test_every_form_reads_the_same_color(color):
    assert np.allclose(to_rgb(color), ORANGE, atol=1e-3)
    assert to_hex(color) == "#ff8800"


@pytest.mark.parametrize("color, rgb", [("red", (1, 0, 0)), ("Red", (1, 0, 0)), ("CRIMSON", (220 / 255, 20 / 255, 60 / 255)),
                                        ("tab:blue", (0x1f / 255, 0x77 / 255, 0xb4 / 255)), ("0.5", (0.5, 0.5, 0.5)),
                                        ("rgb(100%, 50%, 0%)", (1, 0.5, 0)), ("abc", (0xaa / 255, 0xbb / 255, 0xcc / 255))])
def test_names_grey_levels_and_percentages(color, rgb):
    assert np.allclose(to_rgb(color), rgb, atol=1e-3)


def test_alpha_is_read_but_rgb_drops_it():
    assert to_rgba((1, 0, 0, 0.5)) == (1.0, 0.0, 0.0, 0.5)
    assert to_rgba((255, 0, 0, 128))[3] == pytest.approx(128 / 255)
    assert to_rgba("#ff000080")[3] == pytest.approx(128 / 255)
    assert to_rgba("rgba(255, 0, 0, 0.25)")[3] == 0.25
    assert to_rgb((1, 0, 0, 0.5)).shape == (3,)
    assert to_hex((1, 0, 0, 0.5)) == "#ff0000" and to_hex((1, 0, 0, 0.5), keep_alpha=True) == "#ff000080"


@pytest.mark.parametrize("bad", ["notacolor", "", "123", "#12", (1, 2), (300, 0, 0), (-1, 0, 0), (1, 0, 0, 2),
                                 "rgb(300, 0, 0)", None, True, 1.5, 0x1000000, (np.nan, 0, 0)])
def test_anything_else_fails_and_names_the_forms(bad):
    assert not is_color(bad)
    with pytest.raises(ValueError, match="not a color.*'#ff8800'"):
        to_rgba(bad)


def test_stored_colors_keep_readable_strings_and_make_the_rest_hex_or_floats():
    assert normalize(" red ") == "red" and normalize("tab:blue") == "tab:blue" and normalize("#f80") == "#f80"
    assert normalize("ff8800") == "#ff8800" and normalize("rgb(255, 136, 0)") == "#ff8800"
    assert normalize("255, 136, 0") == "#ff8800"
    assert normalize((255, 136, 0)) == (1.0, 0.5333, 0.0) and normalize((1, 0, 0, 0.5)) == (1.0, 0.0, 0.0)


@pytest.fixture
def srtio3():
    a = 3.905
    atoms = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
                  cell=[a, a, a], pbc=True)
    return Pipeline(StaticSource([DataCollection.from_ase(atoms)]))


def test_view_settings_take_every_form_and_reject_bad_colors_when_set(srtio3):
    v = view(srtio3, colors={"O": "rgb(255, 0, 0)", "Ti": "7f7fff", "Sr": (0, 255, 0)}, background="0.2",
             cell_color="Gray", surfaces={"poly": {"color": "ff8800"}}, labels={"color": "rgb(0, 0, 255)"})
    assert v.colors == {"O": "#ff0000", "Ti": "#7f7fff", "Sr": (0.0, 1.0, 0.0)}
    assert (v.background, v.cell_color) == ("0.2", "Gray")
    assert v.surfaces["poly"]["color"] == "#ff8800" and v.labels["color"] == "#0000ff"
    v.color("O", "200, 0, 0")
    assert v.colors["O"] == "#c80000"
    for bad in ({"colors": {"O": "notacolor"}}, {"background": "notacolor"}, {"cell_color": (1, 2)},
                {"surfaces": {"poly": {"color": "nope"}}}):
        with pytest.raises(ValueError, match="not a color"):
            view(srtio3, **bad)
    with pytest.raises(ValueError, match="not a color"):
        v.color("O", "nope")


def test_steps_take_every_form(srtio3):
    from ewald.modifiers import AssignColor, IdentifyFCCPlanarFaults
    from ewald.paint import paint

    srtio3.modifiers.append(AssignColor(color="rgb(255, 136, 0)"))
    data = srtio3.compute(0)
    assert np.allclose(data.particles["Color"], ORANGE, atol=1e-3)
    paint(data, np.arange(data.count) == 0, "00ff00")
    assert np.allclose(data.particles["Color"][0], (0, 1, 0))
    assert IdentifyFCCPlanarFaults(color_twin="blue").color_twin == "blue"   # read when the step runs
