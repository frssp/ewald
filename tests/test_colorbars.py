"""Color bars (ewald.colorbars): ticks, layout, the viewport's image and pictures."""
import numpy as np
import pytest

from ewald.colorbars import STYLES, colorbar_items, gradient_rgba, items_rgba, nice_ticks, tick_labels
from ewald.overlays import ColorRamp, Text, draw_overlays

BARS = [{"title": "Frame", "cmap": "viridis", "range": (0, 4)},
        {"title": "Charge (e)", "cmap": "coolwarm", "range": (-0.83, 1.2)}]


@pytest.mark.parametrize("lo, hi, labels", [
    (0, 4, ["0", "1", "2", "3", "4"]),                  # frames: whole numbers only
    (0, 2, ["0", "1", "2"]),
    (0, 5000, ["0", "1000", "2000", "3000", "4000", "5000"]),
    (0, 1, ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]),
    (-0.83, 1.2, ["−0.5", "0.0", "0.5", "1.0"]),  # a real minus sign
    (1e6, 5e6, ["1e6", "2e6", "3e6", "4e6", "5e6"]),
])
def test_ticks_are_round_and_share_one_format(lo, hi, labels):
    assert tick_labels(nice_ticks(lo, hi)) == labels


def test_ticks_of_an_empty_range():
    assert nice_ticks(3, 3) == [3.0] and tick_labels([3.0]) == ["3"]


@pytest.mark.parametrize("style", STYLES)
def test_bars_fit_in_the_image_and_do_not_overlap(style):
    items = colorbar_items(BARS, (800, 600), unit=12, style=style)
    ramps = [it for it in items if isinstance(it, ColorRamp)]
    assert len(ramps) == 2
    for it in ramps:
        assert 0 <= it.x0 < it.x1 <= 800 and 0 <= it.y0 < it.y1 <= 600
    a, b = ramps
    apart_x = a.x1 <= b.x0 or b.x1 <= a.x0
    apart_y = a.y1 <= b.y0 or b.y1 <= a.y0
    assert apart_x or apart_y
    titles = [it.text for it in items if isinstance(it, Text) and it.text in ("Frame", "Charge (e)")]
    assert sorted(titles) == ["Charge (e)", "Frame"]


def test_gradient_runs_low_to_high_with_rounded_ends():
    img = gradient_rgba("viridis", 8, 100, radius=4)
    assert img.shape == (100, 8, 4)
    assert img[0, 4, 1] > img[-1, 4, 1]                 # yellow (high) on top, purple at the bottom
    assert img[0, 0, 3] < 255 and img[50, 0, 3] == 255  # corners faded, the sides solid


def test_viewport_image_has_no_dark_fringes_on_dark_backgrounds():
    ink = (0.92, 0.92, 0.93)
    rgba, x0, y0 = items_rgba(colorbar_items(BARS[:1], (800, 600), unit=12, ink=ink, background="black"), ink)
    assert rgba.shape[-1] == 4 and rgba[..., 3].min() == 0
    text = (rgba[..., 3] > 0) & (rgba[..., 3] < 255) & (np.ptp(rgba[..., :3].astype(int), axis=-1) < 8)
    assert text.any()
    assert rgba[text][:, :3].min() > 200                # half-covered text pixels keep the light ink


@pytest.mark.parametrize("style", STYLES)
def test_pictures_stay_opaque(style, monkeypatch):
    """Translucent ticks and panels are blended into a picture, not written as holes."""
    monkeypatch.setenv("EWALD_COLORBAR_STYLE", style)
    img = np.full((300, 400, 3), 30, np.uint8)
    out = draw_overlays(img, None, background=(0.1, 0.1, 0.1), colorbars=BARS)
    assert out[..., 3].min() == 255 and (out[..., :3] != 30).any()


def test_vector_items_draw_on_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ewald.overlays import draw_items_mpl

    fig, ax = plt.subplots()
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    draw_items_mpl(ax, colorbar_items(BARS, (800, 600), unit=12, style="card"), (0, 0, 0), 0.75, 12)
    assert not ax.images and len(ax.collections) == 2 and len(ax.patches) >= 4   # vector, no raster
    plt.close(fig)
