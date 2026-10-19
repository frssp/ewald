import io

import pytest

import ewald
from ewald import Pipeline, StaticSource, display
from ewald.notebook import _display

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
