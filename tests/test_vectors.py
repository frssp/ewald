"""Per-particle vector arrows (VectorVis): geometry, modifier, ray tracer and vector renderer.

OpenGL rendering of arrows is in test_render.py (needs a display)."""
import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource
from ewald.arrows import (
    VectorStyle,
    arrow_geometry,
    arrow_mesh,
    coerce_styles,
    scene_arrows,
)
from ewald.data import Cell
from ewald.modifiers import Slice, VectorVis


@pytest.fixture
def two_atoms():
    d = DataCollection([[0, 0, 0], [3, 0, 0]], ["Fe", "Fe"], Cell(np.eye(3) * 6))
    d.particles["Force"] = np.array([[0, 0, 2.0], [1.0, 0, 0]])
    d.particles["Magnetic Moment"] = np.array([2.0, -2.0])
    return d


def test_alignment_reverse_offset_and_scaling(two_atoms):
    a = arrow_geometry(two_atoms, VectorStyle("Force", scaling=0.5))
    np.testing.assert_allclose(a.base, two_atoms.positions)
    np.testing.assert_allclose(a.tip - a.base, [[0, 0, 1.0], [0.5, 0, 0]])
    c = arrow_geometry(two_atoms, VectorStyle("Force", alignment="center", reverse=True, offset=(0, 1, 0)))
    np.testing.assert_allclose(0.5 * (c.base + c.tip), two_atoms.positions + [0, 1, 0])
    np.testing.assert_allclose(c.tip - c.base, -two_atoms["Force"])
    h = arrow_geometry(two_atoms, VectorStyle("Force", alignment="head"))
    np.testing.assert_allclose(h.tip, two_atoms.positions)


def test_head_never_longer_than_arrow(two_atoms):
    a = arrow_geometry(two_atoms, VectorStyle("Force", scaling=0.1, width=0.4))  # lengths 0.2 and 0.1
    lengths = np.linalg.norm(a.tip - a.base, axis=1)
    assert np.all(a.head_length <= lengths + 1e-12)
    assert np.all(a.head_radius < 2.5 * a.radius + 1e-12)  # shrunk head keeps its shape
    np.testing.assert_allclose(a.head_radius / a.head_length, 2.5 / 4.5)


def test_scalar_moment_points_along_z(two_atoms):
    a = arrow_geometry(two_atoms, VectorStyle("Magnetic Moment"))
    np.testing.assert_allclose(a.tip - a.base, [[0, 0, 2.0], [0, 0, -2.0]])


def test_color_mapping_and_skipping(two_atoms):
    a = arrow_geometry(two_atoms, VectorStyle("Force", color_mapping_property="Magnitude", cmap="viridis"))
    import matplotlib

    np.testing.assert_allclose(a.rgb, matplotlib.colormaps["viridis"]([1.0, 0.0])[:, :3])
    sel = two_atoms.copy()
    sel.particles["Selection"] = np.array([0, 1])
    assert list(arrow_geometry(sel, VectorStyle("Force", only_selected=True)).index) == [1]
    assert arrow_geometry(two_atoms, VectorStyle("Force", min_length=1.5)).count == 1
    with pytest.raises(KeyError, match="Force"):
        arrow_geometry(two_atoms, VectorStyle("Velocity"))


def test_modifier_writes_style_and_legend(two_atoms):
    pipe = Pipeline(StaticSource(two_atoms), [VectorVis(color_mapping_property="Force.X")])
    out = pipe.compute(0)
    style = out.vectors["Force"]  # property "" picks Force first
    assert style.color_mapping_interval == (0.0, 1.0)
    assert out.attributes["ColorCoding.range"] == (0.0, 1.0)
    assert "vectors 'Force'" in out.summary()
    assert two_atoms.vectors == {}  # the source frame is untouched


def test_arrows_follow_particles_through_later_modifiers(two_atoms):
    pipe = Pipeline(StaticSource(two_atoms), [VectorVis("Force"), Slice((1, 0, 0), 1.0)])
    (a,) = scene_arrows(pipe.compute(0))
    assert a.count == 1
    np.testing.assert_allclose(a.base, [[3, 0, 0]])
    np.testing.assert_allclose(a.tip, [[4, 0, 0]])


def test_render_argument_forms(two_atoms):
    assert [s.property for s in coerce_styles("Force")] == ["Force"]
    assert [s.scaling for s in coerce_styles({"Force": {"scaling": 2}})] == [2]
    data = two_atoms.copy()
    data.vectors = {"Force": VectorStyle("Force", color="red")}
    (a,) = scene_arrows(data, {"Force": {"color": "blue"}})  # the render argument wins
    np.testing.assert_allclose(a.rgb[0], [0, 0, 1])


def test_arrow_mesh_is_closed_with_unit_normals(two_atoms):
    a = arrow_geometry(two_atoms, VectorStyle("Force"))
    verts, faces, normals, owner = arrow_mesh(a, sides=8)
    assert faces.max() < len(verts) and len(owner) == len(verts)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    # the mesh spans from base to tip of each arrow
    for k in range(a.count):
        v = verts[owner == k]
        t = (v - a.base[k]) @ a.direction[k]
        assert t.min() == pytest.approx(0) and t.max() == pytest.approx(np.linalg.norm(a.tip[k] - a.base[k]))
    # watertight except where the shaft meets the head disc (it ends inside the disc, no gap)
    key = np.round(verts, 9)
    uniq, weld = np.unique(key, axis=0, return_inverse=True)
    f = weld.ravel()[faces]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]  # cone tip triangles are degenerate after welding
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) <= {1, 2}
    open_pts = uniq[edges[counts == 1].ravel()]
    neck_dist = np.min([np.abs((open_pts - a.neck[k]) @ a.direction[k]) for k in range(a.count)], axis=0)
    assert np.all(neck_dist < 1e-6)


_CAMERA = {"position": (1.5, -20, 0.5), "focal_point": (1.5, 0, 0.5), "up": (0, 0, 1), "parallel_projection": True,
           "parallel_scale": 3.0, "view_angle": 30.0}


def test_raytraced_arrows_are_visible(two_atoms):
    from ewald.raytrace import raytrace, scene_from_data

    data = two_atoms.copy()
    data.vectors = {"Force": VectorStyle("Force", color=(0, 0, 1), width=0.4)}
    scene = scene_from_data(data, show_cell=False, radius_scale=0.3, arrow_resolution=12)
    assert len(scene.cyl_r) == 2 and len(scene.tri_alpha) == 2 * 3 * 12
    img = raytrace(scene, _CAMERA, (120, 120), samples=1, ao_samples=0, shadows=False)
    # pixels above the first atom (its arrow points along +z) are blue
    col = img[20:55, 30, :3]
    assert (col[:, 2] > 2 * col[:, 0]).sum() > 5


def test_vector_renderer_draws_arrows(tmp_path, two_atoms):
    from ewald.vector import render_vector

    data = two_atoms.copy()
    plain = render_vector(data, tmp_path / "plain.svg", camera=_CAMERA, size=(120, 120), show_cell=False,
                          radius_scale=0.3)
    out = tmp_path / "arrows.svg"
    img = render_vector(data, out, camera=_CAMERA, size=(120, 120), show_cell=False, radius_scale=0.3,
                        vectors={"Force": {"color": (0, 0, 1)}})
    assert out.read_text().count("<path") > (tmp_path / "plain.svg").read_text().count("<path")  # real shapes
    blue = (img[..., 2].astype(int) - img[..., 0]) > 100
    assert blue.sum() > 50 and ((plain[..., 2].astype(int) - plain[..., 0]) > 100).sum() == 0


def test_vector_renderer_arrow_toward_viewer_is_a_disc(two_atoms):
    from ewald.vector import _arrow_prims

    data = two_atoms.copy()
    data.particles["Force"] = np.array([[0, -2.0, 0], [0, 0, 0]])  # points at the camera
    prims = _arrow_prims(data, "Force", _CAMERA, (120, 120), 0.72, "gradient")
    head = prims[-1][1][0].get_xy()
    span = np.ptp(head, axis=0)
    assert span[0] == pytest.approx(span[1], rel=0.1)  # round, not a sliver
