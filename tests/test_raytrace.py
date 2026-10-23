"""Ray tracer acceleration structures (grid + BVH) and adaptive sampling; no display needed."""

import numpy as np
import pytest

from ewald import raytrace as rt

CAM = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
       "parallel_scale": 2.0}


def random_scene(seed=0, n_sph=60, n_cyl=30, n_tri=400, surface=True):
    rng = np.random.default_rng(seed)
    sc = rt.Scene()
    sc.add_spheres(rng.uniform(-3, 3, (n_sph, 3)), rng.uniform(0.05, 0.4, n_sph), rng.uniform(0, 1, (n_sph, 3)))
    a = rng.uniform(-3, 3, (n_cyl, 3))
    sc.add_cylinders(a, a + rng.normal(0, 0.8, (n_cyl, 3)), rng.uniform(0.02, 0.1, n_cyl), (0.5, 0.5, 0.5),
                     cast_shadows=False)
    c = rng.uniform(-3, 3, (n_tri, 3))
    sc.add_triangles(c, c + rng.normal(0, 0.15, (n_tri, 3)), c + rng.normal(0, 0.15, (n_tri, 3)), (1.0, 0.8, 0.0),
                     alpha=np.where(rng.uniform(size=n_tri) < 0.3, 0.5, 1.0), surface=surface)
    return sc


def brute_force(o, d, tmax, accel, any_hit):
    """Nearest hit over every primitive, without any acceleration structure."""
    best_t, best = tmax, -1
    for kind, arr in ((rt.SPHERE, accel.sph), (rt.CYLINDER, accel.cyl), (rt.TRIANGLE, accel.tri)):
        for i in range(len(arr)):
            if kind == rt.SPHERE:
                t = rt._hit_sphere(o, d, arr, i)
            elif kind == rt.CYLINDER:
                if any_hit:  # the random scene's cylinders cast no shadows
                    continue
                t = rt._hit_cylinder(o, d, arr, i)
            else:
                if any_hit and arr[i, 9] < 0.999:
                    continue
                t = rt._hit_triangle(o, d, arr, i)
            if 0 < t < best_t:
                best_t, best = t, (kind << 40) + i
    return best_t, best


@pytest.mark.parametrize("min_triangles", [0, 10 ** 9])   # triangles in the BVH, or in the grid
def test_trace_matches_brute_force(monkeypatch, min_triangles):
    monkeypatch.setattr(rt, "BVH_MIN_TRIANGLES", min_triangles)
    sc = random_scene()
    accel = rt.Accel(sc)
    assert (len(accel.nodes) > 0) == (min_triangles == 0)
    rng = np.random.default_rng(1)
    stack = np.empty(accel.depth + 2, np.int64); stack_t = np.empty(accel.depth + 2)
    hits = 0
    for k in range(400):
        o = rng.uniform(-5, 5, 3)
        d = rng.uniform(-3, 3, 3) - o if k % 2 else rng.normal(size=3)   # half of them aimed into the scene
        if k % 4 == 0:
            d[rng.integers(3)] = 0.0       # axis-parallel components: the slab test's 1/0 case
        d /= np.linalg.norm(d)
        any_hit = bool(k % 3 == 0)
        tmax = 1.5 if any_hit else 1e30    # short AO-style rays and unbounded camera rays
        t, code = rt._trace(o, d, tmax, accel.grid, accel.starts, accel.items, accel.nodes, accel.bvh_items,
                            stack, stack_t, accel.sph, accel.cyl, accel.tri, any_hit)
        tb, cb = brute_force(o, d, tmax, accel, any_hit)
        assert (code >= 0) == (cb >= 0)
        if code >= 0:
            hits += 1
            assert t == pytest.approx(tb, rel=1e-9, abs=1e-12)
            if not any_hit:  # any-hit rays may stop at any blocker
                assert code & rt.INDEX_MASK == cb & ((1 << 40) - 1) and code >> 40 == cb >> 40
    assert hits > 50


def test_bvh_structure():
    sc = random_scene(n_sph=0, n_cyl=0, n_tri=3000)
    nodes, items, depth = rt.build_bvh(sc)
    count = nodes[:, 7].astype(int)
    leaves = count > 0
    assert sorted(items & rt.INDEX_MASK) == list(range(3000))       # every triangle in exactly one leaf
    assert count[leaves].max() <= rt.MAX_LEAF and count.sum() == 3000
    interior = np.flatnonzero(~leaves)
    for k in interior:  # children lie inside their parent's box
        for c in (int(nodes[k, 6]), int(nodes[k, 6]) + 1):
            assert (nodes[c, :3] >= nodes[k, :3] - 1e-12).all() and (nodes[c, 3:6] <= nodes[k, 3:6] + 1e-12).all()
    assert depth < 40


def test_only_fine_surface_meshes_go_into_the_bvh():
    fine = random_scene(n_sph=20, n_tri=2000)
    assert rt._fine_surface(fine).all()
    assert rt._fine_surface(random_scene(n_sph=20, n_tri=2000, surface=False)).sum() == 0   # glass, arrow heads
    assert rt._fine_surface(random_scene(n_sph=20, n_tri=200)).sum() == 0                   # small mesh
    # coarse mesh: triangles as large as the atom spacing (polyhedra)
    rng = np.random.default_rng(0)
    sc = rt.Scene()
    sc.add_spheres(rng.uniform(0, 30, (5000, 3)), 0.3, (1, 0, 0))
    c = rng.uniform(0, 30, (2000, 3))
    sc.add_triangles(c, c + (2.5, 0, 0), c + (0, 2.5, 0), (0, 1, 0), surface=True)
    assert rt._fine_surface(sc).sum() == 0


def test_mesh_layers_are_surfaces():
    from ewald import DataCollection, Pipeline, StaticSource
    from ewald.data import Cell, VoxelGrid
    from ewald.modifiers import CreateIsosurface, VectorVis

    cell = Cell(np.diag([6.0, 6.0, 6.0]))
    f = np.stack(np.meshgrid(*[np.arange(20) / 20] * 3, indexing="ij"), -1)
    data = DataCollection([[3.0, 3.0, 3.0]], ["Si"], cell)
    data.grids["charge-density"] = VoxelGrid(cell.copy(), {"Charge Density": np.exp(-(((f - 0.5) * 6) ** 2).sum(-1))})
    data.particles["Force"] = np.array([[1.0, 0.0, 0.0]])
    out = Pipeline(StaticSource(data), [CreateIsosurface(0.3), VectorVis("Force")]).compute(0)
    sc = rt.scene_from_data(out)
    n_iso = out.surfaces["isosurface"].face_count
    assert sc.tri_surface.sum() == n_iso and len(sc.tri_alpha) > n_iso   # arrow head triangles are solids


def test_adaptive_pixels_match_the_full_render():
    sc = rt.Scene()
    sc.add_spheres([[0, 0, 0], [1.2, 0.3, 0]], [1.0, 0.6], [(1.0, 0.0, 0.0), (0.0, 0.4, 1.0)])
    full = rt.raytrace(sc, CAM, (48, 36), samples=16, ao_samples=4)
    # a tiny threshold: only exactly flat pixels (the background) stop early, every other one
    # runs to 16 samples over several passes and must equal the one-pass render
    tiny = rt.raytrace(sc, CAM, (48, 36), samples=16, ao_samples=4, noise_threshold=1e-12)
    assert np.allclose(tiny, full, rtol=0, atol=1e-12)
    loose = rt.raytrace(sc, CAM, (48, 36), samples=16, ao_samples=4, noise_threshold=0.02)
    assert np.abs(loose - full).mean() < 0.01
    assert np.array_equal(rt.raytrace(sc, CAM, (48, 36), samples=16, ao_samples=4, noise_threshold=0.0), full)


def test_adaptive_sampling_stops_flat_pixels(monkeypatch):
    seen = {}
    noisy = rt._still_noisy

    def spy(acc, sq, nsamp, threshold, grow=2):
        seen["n"] = nsamp.copy()
        return noisy(acc, sq, nsamp, threshold, grow)

    monkeypatch.setattr(rt, "_still_noisy", spy)
    sc = rt.Scene()
    sc.add_spheres([[0, 0, 0]], [0.8], [(1.0, 0.0, 0.0)])
    rt.raytrace(sc, CAM, (40, 40), samples=32, ao_samples=0, shadows=False, noise_threshold=0.01)
    n = seen["n"]
    assert (n[:6] == 8).all() and (n[:, :6] == 8).all()   # the background stops at min_samples
    assert (n[12:28, 12:28] > 8).mean() > 0.9            # the shaded sphere keeps going


def test_adaptive_progress_and_cancel():
    sc = rt.Scene()
    sc.add_spheres([[0, 0, 0]], [1.0], [(1.0, 0.0, 0.0)])
    seen = []
    a = rt.raytrace(sc, CAM, (40, 30), samples=16, ao_samples=4, noise_threshold=0.01,
                    progress=lambda d, t: seen.append((d, t)))
    b = rt.raytrace(sc, CAM, (40, 30), samples=16, ao_samples=4, noise_threshold=0.01)
    assert np.array_equal(a, b) and seen[-1] == (30, 30) and seen == sorted(seen)
    with pytest.raises(rt.RenderCancelled):
        rt.raytrace(sc, CAM, (40, 30), samples=16, noise_threshold=0.01, progress=lambda d, t: False)


def test_quality_presets_carry_noise_threshold():
    from ewald.render import QUALITY_PRESETS, quality_settings

    assert all("noise_threshold" in p for p in QUALITY_PRESETS.values())
    assert QUALITY_PRESETS["publication"]["noise_threshold"] > 0
    assert quality_settings("publication", noise_threshold=0.0)["noise_threshold"] == 0.0
