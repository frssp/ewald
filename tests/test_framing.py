"""Picture region (the Render frame): pure camera math, no window."""
import math

import numpy as np
import pytest

from ewald import framing

ORTHO = {"position": (0.0, 0.0, 10.0), "focal_point": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0),
         "parallel_projection": True, "parallel_scale": 5.0, "view_angle": 30.0}


def _project(camera, p, size):
    """Pixel (x right, y down) of world point ``p`` in a picture of ``size`` (orthographic or perspective)."""
    pos, focal, back, up, right, dist = framing._basis(camera)
    w, h = size
    d = np.asarray(p, float) - pos
    if camera.get("parallel_projection", True):
        scale = (h / 2) / camera["parallel_scale"]
        u, v = (d @ right) * scale, (d @ up) * scale
    else:
        depth = -(d @ back)
        f = (h / 2) / math.tan(math.radians(camera["view_angle"]) / 2)
        u, v = (d @ right) / depth * f, (d @ up) / depth * f
    return w / 2 + u, h / 2 - v


@pytest.mark.parametrize("ortho", [True, False])
def test_region_camera_shows_what_the_frame_covers(ortho):
    cam = dict(ORTHO, parallel_projection=ortho)
    view = (800, 600)
    region = (0.2, -0.1, 0.5)
    aspect = 16 / 9
    left, top, fw, fh = framing.frame_rect(region, view, aspect)
    pic = (320, 180)
    sub = framing.region_camera(cam, region)
    for p in [(0, 0, 0), (1.5, -2.0, 0), (-1.0, 0.7, 0)]:   # points in the focal plane: exact either way
        vx, vy = _project(cam, p, view)
        px, py = _project(sub, p, pic)
        assert px == pytest.approx((vx - left) / fw * pic[0], abs=1e-6)
        assert py == pytest.approx((vy - top) / fh * pic[1], abs=1e-6)


def test_full_region_is_the_camera_and_frame_rect_round_trips():
    assert framing.region_camera(ORTHO, None) == ORTHO
    assert framing.region_camera(ORTHO, [0, 0, 1]) == ORTHO
    rect = framing.frame_rect((0.1, 0.2, 0.4), (900, 500), 1.5)
    assert framing.rect_region(rect, (900, 500)) == pytest.approx((0.1, 0.2, 0.4))
    assert framing.as_region({"x": 0.1, "height": 0.5}) == (0.1, 0.0, 0.5)
    with pytest.raises(ValueError):
        framing.as_region((0, 0, 0))


def test_fit_region_stays_inside_the_view():
    # a wide picture in a tall view: limited by the view width
    x, y, h = framing.fit_region((400, 800), 2.0, margin=1.0)
    left, top, fw, fh = framing.frame_rect((x, y, h), (400, 800), 2.0)
    assert (x, y) == (0, 0) and fw == pytest.approx(400) and fh == pytest.approx(200)
    assert framing.fit_region((800, 400), 1.0, margin=1.0)[2] == 1.0


@pytest.mark.parametrize("ortho", [True, False])
def test_fit_camera_fills_the_frame(ortho):
    cam = dict(ORTHO, parallel_projection=ortho)
    pts = np.array([[3, 4, 0], [7, 5, 0], [5, 9, 0.0]])
    region, aspect, view = (0.1, 0.05, 0.6), 1.0, (1000, 800)
    fitted = framing.fit_camera(cam, pts, region, aspect, padding=0.0, radii=0.5)
    left, top, fw, fh = framing.frame_rect(region, view, aspect)
    xy = np.array([_project(fitted, p, view) for p in pts])
    assert left - 1e-6 <= xy[:, 0].min() and xy[:, 0].max() <= left + fw + 1e-6
    assert top - 1e-6 <= xy[:, 1].min() and xy[:, 1].max() <= top + fh + 1e-6
    # the tallest extent (y from 3.5 to 9.5 with radii) just fills the frame height
    span = _project(fitted, (5, 3.5, 0), view)[1] - _project(fitted, (5, 9.5, 0), view)[1]
    assert span == pytest.approx(fh, rel=1e-6)


def test_view_region_setting_goes_to_render():
    from ewald.view3d import View

    v = View(region=(0.1, 0, 0.5))
    assert v.region == [0.1, 0.0, 0.5] and v.render_kwargs()["region"] == [0.1, 0.0, 0.5]
    assert "region=[0.1, 0.0, 0.5]" in v.source()
    assert View(region=None).region is None
