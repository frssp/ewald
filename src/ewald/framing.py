"""Picture region: the part of a 3D view a rendered picture shows (the Render frame).

A region is ``(x, y, height)``: the center of the picture's frame, offset from the view center, and the
frame's height, all as fractions of the view height (``y`` points up). Its width follows the picture's
aspect ratio. ``(0, 0, 1)`` is a picture as tall as the view and centered on it, which is what
:func:`ewald.render.render` draws without a region.

:func:`region_camera` turns a view camera plus a region into the camera of the picture, so the picture
shows exactly what the frame covers in the view, however the view is zoomed. The functions here are
pure (no window), so the GUI frame, ``view(pipe, region=...)`` and ``render(..., region=...)`` agree.
"""
from __future__ import annotations

import math

import numpy as np

FULL = (0.0, 0.0, 1.0)


def as_region(region) -> tuple[float, float, float]:
    """``(x, y, height)`` floats from a region (list, tuple, dict with x/y/height, or None for FULL)."""
    if region is None:
        return FULL
    if isinstance(region, dict):
        region = (region.get("x", 0.0), region.get("y", 0.0), region.get("height", 1.0))
    x, y, h = (float(v) for v in region)
    if not h > 0:
        raise ValueError(f"region height must be positive, not {h}")
    return x, y, h


def _basis(camera: dict):
    pos = np.asarray(camera["position"], float)
    focal = np.asarray(camera["focal_point"], float)
    back = pos - focal
    dist = float(np.linalg.norm(back)) or 1.0
    back /= dist
    up = np.asarray(camera.get("up", (0, 0, 1)), float)
    up = up - back * float(up @ back)
    n = float(np.linalg.norm(up))
    if n < 1e-12:  # up along the view: any perpendicular
        up = np.cross(back, [1.0, 0, 0] if abs(back[0]) < 0.9 else [0, 1.0, 0])
        n = float(np.linalg.norm(up))
    up /= n
    right = np.cross(up, back)
    return pos, focal, back, up, right, dist


def half_height(camera: dict) -> float:
    """Half the view height in world units at the focal point."""
    if camera.get("parallel_projection", True):
        return float(camera.get("parallel_scale", 1.0))
    _, _, _, _, _, dist = _basis(camera)
    return dist * math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2)


def region_camera(camera: dict, region) -> dict:
    """The camera of a picture that shows ``region`` of the view seen through ``camera``.

    Orthographic views are exact. In perspective the camera moves sideways to the frame center (the
    picture plane stays parallel to the view), so points at the focal distance land exactly where the
    frame shows them and nearer or farther points shift slightly with parallax."""
    x, y, h = as_region(region)
    if (x, y, h) == FULL:
        return dict(camera)
    pos, focal, _, up, right, _ = _basis(camera)
    hh = half_height(camera)
    shift = right * (2 * hh * x) + up * (2 * hh * y)
    cam = dict(camera)
    cam["position"] = tuple(float(v) for v in pos + shift)
    cam["focal_point"] = tuple(float(v) for v in focal + shift)
    if camera.get("parallel_projection", True):
        cam["parallel_scale"] = float(camera.get("parallel_scale", 1.0)) * h
    else:
        t = math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2)
        cam["view_angle"] = math.degrees(2 * math.atan(t * h))
    return cam


def fit_region(view_size, aspect: float, margin: float = 0.92) -> tuple[float, float, float]:
    """The largest centered region of picture ``aspect`` (width / height) inside a view of
    ``view_size`` (width, height in pixels), shrunk by ``margin`` so its edges stay grabbable."""
    w, h = (max(float(v), 1.0) for v in view_size)
    return 0.0, 0.0, round(margin * min(1.0, (w / h) / max(aspect, 1e-9)), 6)


def frame_rect(region, view_size, aspect: float) -> tuple[float, float, float, float]:
    """``(left, top, width, height)`` of the region's frame in view pixels (y down, as Qt draws)."""
    x, y, rh = as_region(region)
    w, h = (max(float(v), 1.0) for v in view_size)
    fh = rh * h
    fw = fh * aspect
    cx, cy = w / 2 + x * h, h / 2 - y * h
    return cx - fw / 2, cy - fh / 2, fw, fh


def rect_region(rect, view_size) -> tuple[float, float, float]:
    """Inverse of :func:`frame_rect`: the region of a frame ``(left, top, width, height)`` in view pixels."""
    left, top, fw, fh = (float(v) for v in rect)
    w, h = (max(float(v), 1.0) for v in view_size)
    cx, cy = left + fw / 2, top + fh / 2
    return round((cx - w / 2) / h, 6), round((h / 2 - cy) / h, 6), round(max(fh, 1.0) / h, 6)


def fit_camera(camera: dict, points, region, aspect: float, padding: float = 0.06, radii=None) -> dict:
    """A view camera, moved and zoomed (same direction), so ``points`` (world positions, each with an
    optional radius) fill the region's frame with ``padding`` (a fraction of the frame) left around them."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if not len(pts):
        return dict(camera)
    r = np.zeros(len(pts)) if radii is None else np.broadcast_to(np.asarray(radii, float), (len(pts),))
    x, y, rh = as_region(region)
    pos, focal, back, up, right, dist = _basis(camera)
    u, v = (pts - focal) @ right, (pts - focal) @ up
    u0, u1, v0, v1 = (u - r).min(), (u + r).max(), (v - r).min(), (v + r).max()
    cu, cv = (u1 + u0) / 2, (v1 + v0) / 2
    need = max((v1 - v0) / 2, (u1 - u0) / 2 / max(aspect, 1e-9), 1e-6) * (1 + 2 * padding)
    hh = need / rh                      # the view half-height that makes the frame just big enough
    center = focal + right * cu + up * cv
    new_focal = center - right * (2 * hh * x) - up * (2 * hh * y)
    cam = dict(camera)
    if camera.get("parallel_projection", True):
        cam["parallel_scale"] = float(hh)
        new_pos = new_focal + back * dist
    else:
        t = math.tan(math.radians(float(camera.get("view_angle", 30.0))) / 2)
        new_pos = new_focal + back * (hh / max(t, 1e-9))
    cam["focal_point"] = tuple(float(c) for c in new_focal)
    cam["position"] = tuple(float(c) for c in new_pos)
    return cam
