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
