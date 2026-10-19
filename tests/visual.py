"""Reference-image comparison for the visual regression tests (tests/test_visual.py).

References live in tests/data/visual/<name>.png. ``pytest --update-references`` rewrites
them from the current renderers; a failed comparison writes <name>_actual.png,
<name>_expected.png, <name>_diff.png (pixels beyond the tolerance in red, smaller
differences in orange, over a faded copy of the reference) and <name>_compare.png
(expected, actual and diff side by side) to ``visual-diffs/`` (``EWALD_VISUAL_DIFFS``
overrides the folder).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).parent / "data" / "visual"
DIFF_DIR = Path(os.environ.get("EWALD_VISUAL_DIFFS", Path(__file__).parent.parent / "visual-diffs"))


@dataclass
class Tolerance:
    """A pixel counts as changed when any channel differs by more than ``pixel``
    (0-255); the image fails when more than ``fraction`` of its pixels changed or the
    mean absolute difference over all channels exceeds ``mean``."""

    pixel: int = 16
    fraction: float = 0.005
    mean: float = 1.0


# the ray tracer (seeded per pixel) and matplotlib reproduce images bit for bit; allow rounding only
EXACT = Tolerance(pixel=4, fraction=0.001, mean=0.2)


def _load(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _save(img, path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(img, np.uint8)).save(path, optimize=True)


def diff_stats(actual, expected, tol: Tolerance):
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    changed = diff.max(-1) > tol.pixel
    return changed, float(changed.mean()), float(diff.mean())


def write_diff(name, actual, expected, changed):
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    marked = (255 - (255 - expected.astype(float)) * 0.3).astype(np.uint8)  # faded reference
    marked[(actual != expected).any(-1)] = (255, 170, 0)  # any difference: orange
    marked[changed] = (230, 0, 0)  # beyond the per-pixel tolerance: red
    _save(actual, DIFF_DIR / f"{name}_actual.png")
    _save(expected, DIFF_DIR / f"{name}_expected.png")
    _save(marked, DIFF_DIR / f"{name}_diff.png")
    gap = np.full((actual.shape[0], 4, 3), 128, np.uint8)
    _save(np.concatenate([expected, gap, actual, gap, marked], 1), DIFF_DIR / f"{name}_compare.png")


def check_image(name, img, update: bool, tol: Tolerance | None = None):
    """Compare an RGB(A) uint8 image with its reference (or rewrite the reference)."""
    tol = tol or Tolerance()
    img = np.asarray(img, np.uint8)[..., :3]
    ref = REF_DIR / f"{name}.png"
    for old in DIFF_DIR.glob(f"{name}_*.png"):  # left over from an earlier failure
        old.unlink()
    if update:
        _save(img, ref)
        return
    assert ref.exists(), f"no reference image {ref}; create it with pytest --update-references"
    expected = _load(ref)
    if expected.shape != img.shape:
        write_diff(name, img, np.zeros_like(img), np.ones(img.shape[:2], bool))
        raise AssertionError(f"{name}: size {img.shape[1]}x{img.shape[0]}, reference "
                             f"{expected.shape[1]}x{expected.shape[0]} (see {DIFF_DIR})")
    changed, fraction, mean = diff_stats(img, expected, tol)
    if fraction > tol.fraction or mean > tol.mean:
        write_diff(name, img, expected, changed)
        raise AssertionError(f"{name}: {fraction:.2%} of pixels changed (limit {tol.fraction:.2%}), mean "
                             f"difference {mean:.2f} (limit {tol.mean}); images in {DIFF_DIR}/{name}_*.png")


def check_json(name, value, update: bool):
    """Compare a JSON-serializable summary (e.g. SVG element counts) with its reference."""
    ref = REF_DIR / f"{name}.json"
    if update:
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        return
    assert ref.exists(), f"no reference {ref}; create it with pytest --update-references"
    assert value == json.loads(ref.read_text()), f"{name}: structure changed from {ref}"
