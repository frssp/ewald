import numpy as np

from ewald import DataCollection, Pipeline, StaticSource
from ewald.data import Cell
from ewald.modifiers import ColorCoding, CoordinationAnalysis


def test_render_png(tmp_path, cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(3.0), ColorCoding("Position.Z")])
    out = tmp_path / "img.png"
    img = pipe.render(out, size=(320, 240), bonds=2.8)
    assert out.exists() and out.stat().st_size > 1000
    assert img.shape[:2] == (240, 320)
    assert img.std() > 5  # not a blank image


def test_impostor_shading_has_no_black_rims(tmp_path, cu_fcc):
    from ewald.render import render

    img = render(cu_fcc, tmp_path / "imp.png", size=(320, 240), atom_style="impostor", background="white",
                 show_cell=False, ambient_occlusion=False)
    atoms = img[..., :3].astype(int).sum(-1) < 740
    assert atoms.sum() > 1000  # sprites are drawn (not blended away)
    assert np.percentile(img[..., :3].astype(int).sum(-1)[atoms], 1) > 60  # rims stay lit


def test_raytrace_png(tmp_path, cu_fcc):
    from ewald.render import render

    pipe = Pipeline(StaticSource(cu_fcc), [ColorCoding("Position.Z")])
    out = tmp_path / "rt.png"
    img = render(pipe.compute(0), out, size=(96, 72), renderer="raytrace", samples=2, ao_samples=4, bonds=2.8,
                 text="frame 0")
    assert out.exists() and img.shape == (72, 96, 3)
    assert img.std() > 5


def test_raytrace_hits_sphere_center():
    from ewald.raytrace import Scene, raytrace

    sc = Scene()
    sc.add_spheres([[0, 0, 0]], [1.0], [(1.0, 0.0, 0.0)])
    cam = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
           "parallel_scale": 2.0}
    img = raytrace(sc, cam, (21, 21), samples=1, ao_samples=0, shadows=False)
    assert img[10, 10, 0] > 0.5 and img[10, 10, 1] < 0.1      # red sphere in the middle
    assert np.allclose(img[0, 0, :3], 1.0)                    # white background in the corner


def test_trajectory_lines_render(tmp_path):
    from ewald.modifiers import GenerateTrajectoryLines

    t = np.linspace(0, 1, 20)
    frames = []
    for s in t:
        d = DataCollection(np.array([[2.0 + 6 * s, 5.0, 5.0], [5.0, 2.0 + 6 * s, 5.0]]), np.array(["Cu", "O"]),
                           Cell(np.diag([10.0] * 3)))
        frames.append(d)
    pipe = Pipeline(StaticSource(frames), [GenerateTrajectoryLines()])
    with_lines = pipe.render(tmp_path / "a.png", frame=0, size=(160, 120), direction="z", show_cell=False)
    without = pipe.render(tmp_path / "b.png", frame=0, size=(160, 120), direction="z", show_cell=False,
                          trajectory_width=0)
    assert np.abs(with_lines.astype(int) - without.astype(int)).sum() > 0


def _perovskite():
    from ase.spacegroup import crystal

    atoms = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                    cellpar=[3.905] * 3 + [90] * 3)
    return DataCollection.from_ase(atoms)


def test_render_bonds_and_polyhedra(tmp_path):
    from ewald.modifiers import CoordinationPolyhedra, CreateBonds

    pipe = Pipeline(StaticSource(_perovskite()), [CreateBonds(mode="pairwise", pairs=[("Ti", "O", 2.2)])])
    kw = dict(size=(160, 120), direction="iso", show_cell=False, radius_scale=0.3)
    bonds_only = pipe.render(tmp_path / "b.png", **kw)
    pipe.modifiers.append(CoordinationPolyhedra(["Ti"], transparency=0.0))
    poly = pipe.render(tmp_path / "p.png", **kw)
    assert np.abs(poly.astype(int) - bonds_only.astype(int)).sum(-1).astype(bool).mean() > 0.05
    rt = pipe.render(tmp_path / "rt.png", renderer="raytrace", samples=1, ao_samples=2, **kw)
    assert rt.std() > 5


def test_raytrace_triangles_and_transparency():
    from ewald.raytrace import Scene, raytrace

    cam = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
           "parallel_scale": 2.0}

    def scene(alpha):
        sc = Scene()
        sc.add_spheres([[0, 0, -1]], [0.5], [(0.0, 0.0, 1.0)])          # blue sphere behind the plane
        sc.add_triangles([[-2, -2, 0], [-2, -2, 0]], [[2, -2, 0], [2, 2, 0]], [[2, 2, 0], [-2, 2, 0]],
                         [(1.0, 0.0, 0.0)], alpha)
        return raytrace(sc, cam, (21, 21), samples=1, ao_samples=0, shadows=False)

    opaque, glass = scene(1.0), scene(0.5)
    assert opaque[10, 10, 0] > 0.4 and opaque[10, 10, 2] < 0.05     # red plane hides the sphere
    assert glass[10, 10, 0] > 0.2 and glass[10, 10, 2] > 0.2        # sphere shows through
    assert glass[2, 2, 1] > 0.3                                     # white background through the corner


def test_render_isosurface_and_slice(tmp_path):
    from ewald.data import VoxelGrid
    from ewald.modifiers import CreateIsosurface, GridSlice

    cell = Cell(np.diag([6.0, 6.0, 6.0]))
    f = np.stack(np.meshgrid(*[np.arange(20) / 20] * 3, indexing="ij"), -1)
    rho = np.exp(-(((f - 0.5) @ cell.matrix) ** 2).sum(-1))
    data = DataCollection([[3.0, 3.0, 3.0]], ["Si"], cell)
    data.grids["charge-density"] = VoxelGrid(cell.copy(), {"Charge Density": rho})
    pipe = Pipeline(StaticSource(data), [CreateIsosurface(0.2, transparency=0.5), GridSlice((0, 0, 1), 0.25)])
    for renderer in ("opengl", "raytrace"):
        img = pipe.render(tmp_path / f"{renderer}.png", size=(160, 120), renderer=renderer, samples=1, ao_samples=2)
        assert img.shape[:2] == (120, 160) and img.std() > 5
        yellow = (img[..., 0] > 150) & (img[..., 1] > 120) & (img[..., 2] < 80)
        assert yellow.sum() > 50, renderer  # the isosurface is drawn in its colour


def test_render_vector_arrows(tmp_path):
    from ewald.modifiers import VectorVis
    from ewald.render import render

    data = DataCollection([[1, 1, 1], [3, 3, 3]], ["Fe", "Fe"], Cell(np.eye(3) * 4))
    data.particles["Force"] = np.array([[0, 0, 1.5], [1.5, 0, 0]])
    kw = dict(size=(160, 120), direction="y", show_cell=False, radius_scale=0.3, background="white")
    plain = render(data, **kw)
    out = Pipeline(StaticSource(data), [VectorVis("Force", color=(0, 0, 1), width=0.4)]).compute(0)
    arrows = render(out, tmp_path / "a.png", **kw)
    blue = lambda img: ((img[..., 2].astype(int) - img[..., 0]) > 80).sum()
    assert blue(plain) == 0 and blue(arrows) > 100
    lines = render(out, mesh_limit=1, **kw)  # above the limit arrows become lines
    assert 0 < blue(lines) < blue(arrows)
    rt = render(out, renderer="raytrace", samples=1, ao_samples=0, **kw)
    assert blue(rt) > 100


def test_type_groups_first_appearance_order():
    from ewald.render import type_groups

    types = np.array(["Zr", "Cu", "Zr", "O", "Cu"])
    names, inverse = type_groups(types)
    assert names == ["Zr", "Cu", "O"] and list(inverse) == [0, 1, 0, 2, 1]
    many = np.array([f"T{k % 50}" for k in range(500)][::-1])  # more types than the scan handles
    names, inverse = type_groups(many, max_scan=8)
    assert names == list(dict.fromkeys(many)) and all(np.array(names)[inverse] == many)
    names, inverse = type_groups(np.array([], dtype=str))
    assert names == [] and len(inverse) == 0
