import numpy as np

from ewald import Pipeline, StaticSource
from ewald.modifiers import ColorCoding, CoordinationAnalysis


def test_render_png(tmp_path, cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(3.0), ColorCoding("Position.Z")])
    out = tmp_path / "img.png"
    img = pipe.render(out, size=(320, 240), bonds=2.8)
    assert out.exists() and out.stat().st_size > 1000
    assert img.shape[:2] == (240, 320)
    assert img.std() > 5  # not a blank image
