"""Scene IR (ewald.scene) and visual element plugins (ewald.plugins.visual)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource, plugins
from ewald.data import Cell
from ewald.scene import (Cylinders, Lines, ParticlesVis, Scene, SceneContext, Spheres, VisualElement, build,
                         coerce_visuals, visual_to_dict)

CELL_AXES = Path(__file__).parents[1] / "examples" / "plugins" / "cell_axes.py"
CAMERA = {"position": (30.0, 25.0, 20.0), "focal_point": (2.0, 2.0, 2.0), "up": (0.0, 0.0, 1.0),
          "parallel_projection": True, "parallel_scale": 6.0, "view_angle": 30.0}


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


@pytest.fixture
def example_plugin():
    folder = plugins.plugin_dir()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(CELL_AXES, folder)
    return plugins.visuals()


def _crystal():
    from ase.build import bulk

    data = DataCollection.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    data.particles["Force"] = np.tile([0.0, 0.0, 1.0], (data.count, 1))
    return data


def test_builtin_elements_make_one_layer_each():
    data = _crystal()
    scene = build(data, bonds=3.0, vectors="Force")
    assert [lay.name for lay in scene.layers] == ["atoms", "bonds", "vectors:Force", "cell"]
    assert [lay.kind for lay in scene.layers] == ["spheres", "cylinders", "arrows", "lines"]
    atoms = scene["atoms"].primitive
    assert atoms.count == data.count and atoms.rgb is None  # per-type tables, not per-atom arrays
    assert atoms.colors().shape == (data.count, 3) and atoms.sizes().shape == (data.count,)
    bonds = scene["bonds"].primitive
    assert bonds.count % 2 == 0 and set(bonds.owner) <= set(range(data.count))
    cell = scene["cell"].primitive
    assert cell.count == 12 and cell.rgb is None and cell.radius > 0
    assert len(build(data, show_cell=False).layers) == 1


def test_particles_leave_out_invisible_atoms_and_keep_opacity():
    data = _crystal()
    t = np.zeros(data.count)
    t[0], t[1] = 1.0, 0.5
    data.particles["Transparency"] = t
    ctx = SceneContext()
    ((_, sph),) = ParticlesVis().build(data, ctx)
    assert sph.count == data.count - 1 and 0 not in sph.index
    assert np.isclose(sph.alpha[list(sph.index).index(1)], 0.5)
    assert ctx.particle_colors.shape == (data.count, 3)  # bonds and plugins reuse every atom's color


def test_hidden_atoms_take_their_half_bonds_with_them():
    data = _crystal()
    full = build(data, bonds=3.0)["bonds"].primitive
    t = np.zeros(data.count)
    t[0] = 1.0                                       # paint.hide(): Transparency 1
    data.particles["Transparency"] = t
    bonds = build(data, bonds=3.0)["bonds"].primitive
    assert 0 not in bonds.owner and bonds.count == full.count - np.count_nonzero(full.owner == 0)
    data.particles["Transparency"] = np.ones(data.count)
    assert "bonds" not in [lay.name for lay in build(data, bonds=3.0).layers]


def test_scene_rejects_non_primitives_and_renames_duplicates():
    scene = Scene()
    scene.add("x", Spheres([[0, 0, 0]]))
    assert scene.add("x", Spheres([[1, 0, 0]])).name == "x:2"
    with pytest.raises(TypeError, match="not a scene primitive"):
        scene.add("y", np.zeros(3))


def test_lines_segments_follow_strips():
    lines = Lines(np.zeros((5, 3)), [[0, 1, 2], [3, 4]])
    assert lines.segments().tolist() == [[0, 1], [1, 2], [3, 4]]


def test_plugin_registered_from_folder(example_plugin):
    reg = example_plugin
    assert {"Particles", "Bonds", "Simulation cell"} <= set(reg)
    assert reg["Cell axes"].category == "Annotations" and not reg["Cell axes"].builtin
    assert reg["Particles"].builtin and reg["Particles"].origin == "built-in"
    scene = build(_crystal(), visuals=["Cell axes"])
    layer = scene["visual:cell-axes"]
    assert layer.kind == "arrows" and layer.primitive.count == 3 and layer.element == "Cell axes"


def test_register_checks():
    with pytest.raises(TypeError):
        plugins.register_visual(object)
    with pytest.raises(ValueError, match="built-in"):
        plugins.register_visual(type("Particles", (VisualElement,), {"name": "Particles"}))


def test_coerce_and_round_trip(example_plugin):
    (el,) = coerce_visuals([{"name": "Selection halo", "params": {"scale": 2.5}, "enabled": False}])
    assert el.scale == 2.5 and el.enabled is False
    d = visual_to_dict(el)
    assert d == {"name": "Selection halo", "params": {"scale": 2.5, "opacity": 0.35, "color": (1.0, 0.85, 0.1)},
                 "enabled": False}
    assert coerce_visuals([d])[0].scale == 2.5
    with pytest.raises(KeyError, match="no visual element"):
        coerce_visuals(["Nope"])
    # a disabled element is kept but not drawn
    data = _crystal()
    data.particles["Selection"] = np.ones(data.count, int)
    assert "visual:selection-halo" not in build(data, visuals=[d])


def _render(data, renderer, **kw):
    from ewald.render import render

    return render(data, size=(120, 90), renderer=renderer, camera=CAMERA, quality="draft", legend=False, **kw)


@pytest.mark.parametrize("renderer", ["raytrace", "vector"])
def test_plugin_element_reaches_raytracer_and_vector(example_plugin, renderer):
    data = _crystal()
    plain = _render(data, renderer)
    with_axes = _render(data, renderer, visuals=["Cell axes"])
    assert plain.shape == with_axes.shape
    assert np.abs(plain.astype(int) - with_axes.astype(int)).sum() > 0


def test_raytracer_tessellates_see_through_spheres():
    from ewald.raytrace import scene_from_ir

    scene = Scene()
    scene.add("glass", Spheres([[0, 0, 0], [3, 0, 0]], 1.0, (1, 0, 0), [0.4, 1.0]))
    sc = scene_from_ir(scene)
    assert len(sc.sph_r) == 1 and len(sc.tri_alpha) > 100 and np.allclose(sc.tri_alpha, 0.4)


def test_raytracer_draws_lines_with_a_radius_only():
    from ewald.raytrace import scene_from_ir

    scene = Scene()
    scene.add("screen-only", Lines(np.eye(3), [[0, 1, 2]], width=2))
    scene.add("tube", Lines(np.eye(3), [[0, 1, 2]], [(0, 1, 0)], radius=0.1))
    scene.add("sticks", Cylinders([[0, 0, 0]], [[0, 0, 1]], 0.2, (0, 0, 1), cap_end=False))
    sc = scene_from_ir(scene)
    assert len(sc.cyl_r) == 3 and len(sc.sph_r) == 3  # two tube pieces + one stick; three joints
    assert np.allclose(sc.cyl_rgb[:2], [0, 1, 0])


def test_session_and_style_keep_visuals(tmp_path, example_plugin):
    from ewald.gui.style import ViewStyle
    from ewald.session import load_session, save_session

    data = _crystal()
    style = ViewStyle(visuals=[{"name": "Cell axes", "params": {"length": 3.0}, "enabled": True}])
    path = save_session(tmp_path / "s.ewald", Pipeline(StaticSource(data)), style=style)
    kw = load_session(path).render_kwargs()
    assert kw["visuals"] == [{"name": "Cell axes", "params": {"length": 3.0}, "enabled": True}]
    scene = build(data, visuals=kw["visuals"])
    tips = scene["visual:cell-axes"].primitive.tip
    assert np.allclose(np.linalg.norm(tips, axis=1), 3.0)


def test_llm_tools_list_and_add_visuals(example_plugin):
    from ewald.llm import Toolset

    ts = Toolset(Pipeline(StaticSource(_crystal())))
    listed = ts.call("list_visuals", {}).text()
    assert "Cell axes" in listed and "Particles" in listed
    assert ts.call("add_visual", {"name": "Particles"}).is_error
    res = ts.call("add_visual", {"name": "Cell axes", "params": {"length": 1.5}})
    assert not res.is_error, res.text()
    added = [{"name": "Cell axes", "params": {"length": 1.5, "width": 0.25}, "enabled": True}]
    assert ts.pipeline.view.visuals == added                    # through the view cell, which ran
    assert "visuals=[{'name': 'Cell axes'" in ts.to_script()
    assert not ts.call("remove_visual", {"index": 0}).is_error and ts.pipeline.view.visuals == []


def test_cli_lists_visual_elements(example_plugin, capsys):
    from ewald.__main__ import main

    main(["plugins"])
    out = capsys.readouterr().out
    assert "visual Cell axes" in out and "visual Particles" in out


def test_elements_do_not_touch_the_data(example_plugin):
    data = _crystal()
    data.particles["Selection"] = np.ones(data.count, int)
    before = {k: np.array(v, copy=True) for k, v in data.particles.items()}
    build(data, bonds=3.0, vectors="Force", visuals=["Cell axes", "Selection halo"])
    for k, v in before.items():
        assert np.array_equal(v, data.particles[k])


def test_empty_cell_and_no_atoms():
    data = DataCollection(np.zeros((0, 3)), [], Cell(np.zeros((3, 3))))
    assert build(data).layers == []


def test_raytrace_cell_casts_no_shadow():
    """Cell edges are seen by camera rays but skipped by shadow and AO rays."""
    from ewald.raytrace import Scene as RTScene, raytrace

    sc = RTScene()
    sc.add_spheres([[0, 0, 0]], 1.0, (0.8, 0.8, 0.8))
    cam = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_scale": 1.5}
    kw = dict(size=(40, 30), samples=1, ao_samples=8, light_direction=(0, 0, 1))
    clean = raytrace(sc, cam, **kw)
    # a thin rod between the light and the sphere, off to the side of the camera's view of it
    rod = RTScene()
    rod.add_spheres([[0, 0, 0]], 1.0, (0.8, 0.8, 0.8))
    rod.add_cylinders([[-0.3, -3, 3]], [[-0.3, 3, 3]], 0.05, (0, 0, 0), cast_shadows=False)
    img = raytrace(rod, cam, **kw)
    away = np.abs(np.arange(40) - 20 + 0.3 * 40 / 4) > 3  # columns not covered by the rod itself
    assert np.allclose(img[:, away], clean[:, away])
    rod.cyl_cast[:] = True
    assert not np.allclose(raytrace(rod, cam, **kw)[:, away], clean[:, away])


# ----------------------------------------------------------------------------- bond caps (D50)
def _caps(scene):
    cyl = next(lay.primitive for lay in scene.layers if lay.kind == "cylinders")
    return cyl, np.bincount(cyl.start_cap, minlength=3), np.bincount(cyl.end_cap, minlength=3)


def test_bond_caps_flat_where_cut_round_where_atoms_do_not_hide_the_end():
    data = _crystal()   # cubic NaCl: half of the 24 bonds cross the periodic cell
    cyl, start, end = _caps(build(data, bonds=3.0))
    assert cyl.count == 48 and start.tolist() == [48, 0, 0]   # every atom end inside its atom
    assert end.tolist() == [24, 24, 0]                         # met halves open, cut halves flat
    cut = cyl.end_cap == 1
    met = ~cut
    # an open end meets its partner's; a flat one is not at any other half-bond's end
    ends = {tuple(np.round(e, 6)) for e in cyl.end[met]}
    assert len(ends) == 12 and not ends & {tuple(np.round(e, 6)) for e in cyl.end[cut]}
    for kw in (dict(show_atoms=False), dict(radius_scale=0.05)):   # atoms off or smaller than the bond
        assert _caps(build(data, bonds=3.0, **kw))[1].tolist() == [0, 0, 48]
    see_through = _crystal()
    see_through.particles["Transparency"] = np.where(np.arange(see_through.count) < 4, 0.5, 0.0)
    cyl, start, end = _caps(build(see_through, bonds=3.0))
    assert start.tolist() == [24, 0, 24] and np.all(cyl.start_cap[cyl.owner < 4] == 2)


def test_bond_caps_close_a_half_bond_whose_partner_is_gone():
    data = _crystal()
    data.particles["Transparency"] = np.where(np.arange(data.count) == 0, 1.0, 0.0)  # atom 0 hidden
    cyl, start, end = _caps(build(data, bonds=3.0))
    # atom 0 takes 3 cut halves with it; the halves of its 3 in-cell partners are now cut too
    assert cyl.count == 48 - 6 and end[1] == 24 - 3 + 3 and end[0] == 24 - 3 - 3
    cyl, _, end = _caps(build(_crystal(), bonds=3.0, boundary=((-0.01, 1.01),) * 3))
    assert end[1] > 24  # boundary half-bonds end flat at the cut


def test_cylinder_caps_values_and_old_cap_end():
    one = Cylinders([[0, 0, 0]], [[0, 0, 1]], 0.2)
    assert one.start_cap.tolist() == [1] and one.end_cap.tolist() == [1]    # closed by default
    assert Cylinders([[0, 0, 0]], [[0, 0, 1]], cap_end=False).end_cap.tolist() == [0]
    two = Cylinders(np.zeros((2, 3)), np.eye(3)[:2], 0.2, start_cap=["round", "none"], end_cap=2)
    assert two.start_cap.tolist() == [2, 0] and two.end_cap.tolist() == [2, 2]
    center, axis, radius, *_, idx = two.caps("round")
    assert idx.tolist() == [0, 0, 1] and np.allclose(axis, [[-1, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(center, [[0, 0, 0], [1, 0, 0], [0, 1, 0]]) and np.allclose(radius, 0.2)
    with pytest.raises(ValueError, match="dome"):
        Cylinders([[0, 0, 0]], [[0, 0, 1]], end_cap="dome")


def test_cap_meshes_close_the_tube_without_cracks():
    from ewald.render import TUBE_SIDES, cap_mesh, tube_mesh

    start, end = np.array([[0.0, 0, 0], [1, 1, 0]]), np.array([[0.0, 0, 2], [1, 3, 1]])
    verts, _, _ = tube_mesh(start, end, 0.3)
    rims = verts.reshape(2, 2, TUBE_SIDES, 3)
    axis = (end - start) / np.linalg.norm(end - start, axis=1, keepdims=True)
    for kind in ("flat", "round"):
        for which, (p, d) in enumerate(((start, -axis), (end, axis))):
            v, f, n = cap_mesh(p, d, np.full(2, 0.3), kind)
            v = v.reshape(2, -1, 3)
            # the cap's rim is the tube's ring at that end (as a set: the start cap turns the other way)
            for k in range(2):
                rim = {tuple(np.round(x, 9)) for x in rims[k, which]}
                assert rim <= {tuple(np.round(x, 9)) for x in v[k]}
            assert np.allclose(np.linalg.norm(n, axis=1), 1) and f.max() < len(v.reshape(-1, 3))
            if kind == "round":   # every vertex on the hemisphere, on the outer side of the end
                rel = v - p[:, None]
                assert np.allclose(np.linalg.norm(rel, axis=2), 0.3)
                assert (np.einsum("kij,kj->ki", rel, d) > -1e-9).all()
            # triangles face outward
            tri = v.reshape(-1, 3)[f]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            out = tri.mean(1) - np.repeat(p, len(f) // 2, axis=0)
            assert (np.einsum("ij,ij->i", normal, out + 1e-3 * np.repeat(d, len(f) // 2, axis=0)) > 0).all()


def test_raytracer_flat_caps_are_disks_and_round_caps_spheres():
    from ewald import raytrace as rt
    from ewald.raytrace import _hit_cylinder, scene_from_ir

    o, d = np.array([0.0, 0.05, 5.0]), np.array([0.0, 0.0, -1.0])   # down the axis, inside the rim
    for caps, t in (((0, 0), -1.0), ((0, 1), 3.0), ((1, 0), 5.0), ((1, 1), 3.0)):
        cyl = np.array([[0, 0, 0, 0, 0, 2, 0.2, *caps]], float)
        assert _hit_cylinder(o, d, cyl, 0) == pytest.approx(t)
    scene = Scene()
    scene.add("sticks", Cylinders(np.zeros((2, 3)), [[0, 0, 2], [2, 0, 0]], 0.2,
                                  start_cap="round", end_cap=["flat", "none"]))
    sc = scene_from_ir(scene)
    assert sc.cyl_cap.tolist() == [[0, 1], [0, 0]] and np.allclose(sc.sph_r, 0.2) and len(sc.sph_r) == 2
    img = rt.raytrace(sc, {"position": (0, 0, 6), "focal_point": (0, 0, 0), "up": (0, 1, 0),
                           "parallel_projection": True, "parallel_scale": 0.5}, (20, 20), samples=1,
                      ao_samples=0, shadows=False)
    centre = img[10, 10, :3]
    assert centre.max() > 0.3 and np.allclose(centre, centre[0], atol=0.05)  # a lit disk, not a hole


def test_vector_draws_caps():
    from ewald.vector import scene_prims

    cam = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
           "parallel_scale": 3.0}
    def count(**caps):
        scene = Scene()
        scene.add("sticks", Cylinders([[0, 0, 0]], [[1, 0, 1]], 0.2, **caps))
        return len(scene_prims(scene, cam, (200, 150), 1.0))
    assert count(start_cap="none", end_cap="none") == 1
    assert count(start_cap="none", end_cap="flat") == 2     # the end faces the camera: an ellipse
    assert count(start_cap="flat", end_cap="none") == 1     # the start faces away: hidden by the stick
    assert count(start_cap="round", end_cap="round") == 5   # a disc under and over each end
