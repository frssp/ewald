import io

import pytest

import ewald
from ewald import Pipeline, StaticSource, display
from ewald.notebook import display_output as _display

PNG = b"\x89PNG"


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(display, "options", {**display.options, "size": (80, 60)})


def _image(png):
    from PIL import Image

    return Image.open(io.BytesIO(png))


def test_jupyter_shows_a_pipeline_and_its_frames_as_pictures(cu_fcc, small):
    DisplayFormatter = pytest.importorskip("IPython.core.formatters").DisplayFormatter

    pipe = Pipeline(StaticSource(cu_fcc))
    fmt, _ = DisplayFormatter().format(pipe)
    assert fmt["image/png"].startswith(PNG) and _image(fmt["image/png"]).size == (80, 60)
    assert fmt["text/plain"].startswith("<Pipeline")                    # text stays for consoles
    assert DisplayFormatter().format(pipe.compute(0))[0]["image/png"].startswith(PNG)
    display.options["enabled"] = False
    assert "image/png" not in DisplayFormatter().format(pipe)[0]


def test_table_pictures_and_text_fallback(cu_fcc, small, monkeypatch):
    pipe = Pipeline(StaticSource(cu_fcc))
    g = ewald.rdf(pipe, cutoff=5.0, bins=50)
    assert g._repr_png_().startswith(PNG)
    out = _display(g)                                                    # the GUI notebook shows the curve
    assert out.kind == "image"
    assert _display(pipe.compute(0)).kind == "value"                    # a frame keeps its text summary there

    def boom(*a, **k):
        raise RuntimeError("no renderer")

    monkeypatch.setattr("ewald.render.render", boom)
    assert pipe._repr_png_() is None                                     # Jupyter falls back to the repr


def test_a_table_of_named_rows_is_bars_and_a_series_stays_a_curve():
    """One value per surface is a bar per surface, not a line through the row numbers; a table
    whose x is a real quantity (run temperatures) stays a curve although its rows have names."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from ewald.verbs import Table
    from ewald.views import _draw

    rows = Table("Work functions", {"Surface #": [0, 1], "Work function (eV)": [4.9, 4.6], "E_F (eV)": [-0.4, 0.2]},
                 labels=["111", "100"])
    ax = Figure().add_subplot()
    _draw(ax, rows.name, dict(rows), None, labels=rows.labels)
    assert [p.get_height() for p in ax.patches] == pytest.approx([4.9, 4.6]) and not ax.lines
    assert [x.get_text() for x in ax.get_xticklabels()] == ["111", "100"]
    assert 0 < ax.get_ylim()[0] < 4.6          # zoomed onto the values, so 4.6 and 4.9 differ
    assert {x.get_text() for x in ax.texts} == {"4.9", "4.6"}
    runs = Table("D", {"Run": [300.0, 600.0], "D": [1.0, 2.0]}, labels=["300K", "600K"])
    ax = Figure().add_subplot()
    _draw(ax, runs.name, dict(runs), None, labels=runs.labels)
    assert len(ax.lines) == 1 and not ax.patches
    assert display.table_png(rows).startswith(PNG)
