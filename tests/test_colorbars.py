"""Color bars (ewald.colorbars): ticks, layout, the viewport's image and pictures."""
import numpy as np
import pytest

from ewald.colorbars import colorbar_items, gradient_rgba, items_rgba, nice_ticks, tick_labels
from ewald.overlays import ColorRamp, Rect, Text, draw_overlays

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


def test_bars_sit_on_cards_side_by_side_in_the_top_right_corner():
    items = colorbar_items(BARS, (800, 600), unit=12)
    ramps = [it for it in items if isinstance(it, ColorRamp)]
    assert len(ramps) == 2
    for it in ramps:
        assert 0 <= it.x0 < it.x1 <= 800 and 0 <= it.y0 < it.y1 <= 600
    a, b = ramps
    assert b.x1 < a.x0 and a.y0 == b.y0 < 100        # the second bar left of the first, both at the top
    cards = [it for it in items if isinstance(it, Rect) and it.fill is not None]
    assert len(cards) == 2 and all(c.radius > 0 and c.fill[3] < 1 for c in cards)
    assert max(c.x1 for c in cards) < 800
    titles = [it.text for it in items if isinstance(it, Text) and it.text in ("Frame", "Charge (e)")]
    assert sorted(titles) == ["Charge (e)", "Frame"]


def test_gradient_runs_low_to_high_with_rounded_ends():
    img = gradient_rgba("viridis", 8, 100, radius=4)
    assert img.shape == (100, 8, 4)
    assert img[0, 4, 1] > img[-1, 4, 1]                 # yellow (high) on top, purple at the bottom
    assert img[0, 0, 3] < 255 and img[50, 0, 3] == 255  # corners faded, the sides solid


def test_viewport_image_is_a_translucent_card_with_light_text_on_dark_backgrounds():
    ink = (0.92, 0.92, 0.93)
    rgba, x0, y0 = items_rgba(colorbar_items(BARS[:1], (800, 600), unit=12, ink=ink, background="black"), ink)
    alpha = rgba[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0                  # rounded corners, transparent around
    h, w = alpha.shape
    assert 150 < alpha[h // 2, w - 6] < 255                          # the card lets the view show through
    assert (rgba[..., :3].min(axis=-1)[alpha > 150] > 200).any()     # light labels on it


def test_pictures_stay_opaque():
    """Translucent ticks and cards are blended into a picture, not written as holes."""
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
    draw_items_mpl(ax, colorbar_items(BARS, (800, 600), unit=12), (0, 0, 0), 0.75, 12)
    assert not ax.images and len(ax.collections) == 2 and len(ax.patches) >= 4   # vector, no raster
    plt.close(fig)
