"""Visual regression tests: fixed scenes rendered and compared with tests/data/visual/*.png.

Run locally (no CI needed):  xvfb-run -a python -m pytest tests/test_visual.py
Refresh the references after an intended change:  ... --update-references, then review the
PNGs in the diff before committing. Failures leave actual/expected/diff images in visual-diffs/.

Scenes use fixed structures, cameras, sample counts and ray-tracer seeds. OpenGL references are
made with Mesa llvmpipe (xvfb-run); on other OpenGL drivers those cases skip, since GPU drivers
rasterize differently (EWALD_VISUAL_OPENGL=1 forces them). Ray-traced and vector images do not
depend on the driver and run everywhere.
"""

import collections
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource
from ewald.data import Cell, VoxelGrid
from ewald.modifiers import (ColorCoding, CoordinationPolyhedra, CreateBonds, CreateIsosurface, GridSlice)
from visual import EXACT, check_image, check_json

pytestmark = pytest.mark.visual

SIZE = (200, 150)


@pytest.fixture(scope="session")
def update(request):
    return request.config.getoption("--update-references")


@pytest.fixture(scope="session")
def opengl_ok():
    import pyvista as pv

    renderer = pv.GPUInfo().renderer
    if "llvmpipe" not in renderer and not os.environ.get("EWALD_VISUAL_OPENGL"):
        pytest.skip(f"OpenGL references are made with Mesa llvmpipe, this driver is {renderer!r}")


# ----------------------------------------------------------------------------- scenes
def copper():
    from ase.build import bulk

    return DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3)))


def perovskite():
    from ase.spacegroup import crystal

    atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                    cellpar=[3.905] * 3 + [90] * 3)
    return DataCollection.from_ase(atoms.repeat((2, 2, 2)))


def rocksalt():
    from ase.build import bulk

    return DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2)))


def density():
    cell = Cell(np.diag([6.0, 6.0, 6.0]))
    f = np.stack(np.meshgrid(*[np.arange(20) / 20] * 3, indexing="ij"), -1)
    rho = np.exp(-(((f - 0.5) @ cell.matrix) ** 2).sum(-1) / 2)
    data = DataCollection([[3.0, 3.0, 3.0]], ["Si"], cell)
    data.grids["charge-density"] = VoxelGrid(cell.copy(), {"Charge Density": rho})
    return data


def pipelines():
    return {
        "copper": Pipeline(StaticSource(copper()), [ColorCoding("Position.Z")]),
        "polyhedra": Pipeline(StaticSource(perovskite()), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)]),
                                                         CoordinationPolyhedra(["Ti"], transparency=0.0)]),
        "isosurface": Pipeline(StaticSource(density()), [CreateIsosurface(0.3, transparency=0.4),
                                                         GridSlice((0, 0, 1), 0.25)]),
        "rocksalt": Pipeline(StaticSource(rocksalt()), []),
    }


# name: (pipeline, render keyword arguments)
SCENES = {
    "copper_colorcoding": ("copper", dict(text="frame 0", legend=True, bonds=2.8)),
    "polyhedra": ("polyhedra", dict(radius_scale=0.4, show_cell=False, direction=(3, 2, 1))),
    "isosurface_slice": ("isosurface", dict()),
    "overlays": ("rocksalt", dict(text="NaCl", legend=False, type_legend=True, scale_bar=True, direction="z")),
    "radius_ionic": ("rocksalt", dict(radius_preset="ionic", show_cell=False)),
    "radius_vdw": ("rocksalt", dict(radius_preset="vdW", radius_scale=0.5, show_cell=False)),
}
OPENGL_ONLY = {"impostor": ("copper", dict(atom_style="impostor", show_cell=False))}


def _render(name, renderer, quality="standard", **extra):
    pipe_name, kw = {**SCENES, **OPENGL_ONLY}[name]
    data = pipelines()[pipe_name].compute(0)
    from ewald.render import render

    return render(data, size=SIZE, renderer=renderer, quality=quality, **{**kw, **extra})


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", [*SCENES, *OPENGL_ONLY])
def test_opengl(name, update, opengl_ok):
    check_image(f"opengl_{name}", _render(name, "opengl"), update)


@pytest.mark.parametrize("name", list(SCENES))
def test_raytrace(name, update):
    # 4 samples, 8 AO rays; the per-pixel random streams are seeded, so the image is reproducible
    check_image(f"raytrace_{name}", _render(name, "raytrace"), update, EXACT)


@pytest.mark.parametrize("name", list(SCENES))
def test_vector(name, update, tmp_path):
    svg = tmp_path / f"{name}.svg"
    img = _render(name, "vector", filename=svg)
    check_image(f"vector_{name}", img, update, EXACT)
    # structure of the SVG: counts of drawing elements (circles/polygons are <path>, text glyphs <use>)
    tags = collections.Counter(el.tag.rsplit("}", 1)[-1] for el in ET.parse(svg).iter())
    check_json(f"vector_{name}", {t: tags[t] for t in ("path", "use", "g", "image", "text") if tags[t]}, update)


def test_vector_pdf(tmp_path):
    out = tmp_path / "polyhedra.pdf"
    _render("polyhedra", "vector", filename=out)
    head = out.read_bytes()
    assert head.startswith(b"%PDF") and b"/Type /Page" in head and b"/Subtype /Image" not in head  # real vectors


@pytest.mark.parametrize("quality", ["draft", "standard", "publication"])
@pytest.mark.parametrize("renderer", ["opengl", "raytrace"])
def test_quality_presets(renderer, quality, update, request):
    if renderer == "opengl":
        request.getfixturevalue("opengl_ok")
    img = _render("copper_colorcoding", renderer, quality=quality)
    check_image(f"quality_{renderer}_{quality}", img, update, EXACT if renderer == "raytrace" else None)
