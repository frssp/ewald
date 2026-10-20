"""Publication output without OpenGL: overlays, vector figures, movies, session files, CLI."""
import json
import os
import shutil

import numpy as np
import pytest

from ewald import DataCollection, Pipeline, PythonModifier, StaticSource
from ewald.data import Cell
from ewald.modifiers import ColorCoding, PythonScript, Replicate, SelectType

CAMERA = {"position": (30.0, 30.0, 30.0), "focal_point": (7.2, 7.2, 7.2), "up": (0.0, 0.0, 1.0),
          "parallel_projection": True, "parallel_scale": 12.0, "view_angle": 30.0}


@pytest.fixture
def moving():
    frames = [DataCollection(np.array([[2.0 + 6 * s, 5.0, 5.0], [5.0, 2.0 + 6 * s, 5.0]]), np.array(["Cu", "O"]),
                             Cell(np.diag([10.0] * 3))) for s in np.linspace(0, 1, 5)]
    for k, d in enumerate(frames):
        d.attributes["Timestep"] = 100 * k
    return Pipeline(StaticSource(frames))


def test_format_label():
    from ewald.overlays import format_label

    d = DataCollection(np.zeros((1, 3)), ["Cu"])
    d.attributes.update(Timestep=500, Temperature=301.5)
    assert format_label("f=[Frame] ts=[Timestep] t=[Time] ps T=[Temperature] K", 5, d, 0.2) == \
        "f=5 ts=500 t=1 ps T=301.5 K"
    assert format_label("", 1, d) == "" and format_label(None, 1) == ""


def test_scale_bar_length_and_pixels():
    from ewald.overlays import nice_length, overlay_layout, pixels_per_angstrom

    assert [nice_length(x) for x in (0.7, 3.2, 7.9, 12, 49)] == [0.5, 2.0, 5.0, 10.0, 20.0]
    assert pixels_per_angstrom(CAMERA, (400, 240)) == pytest.approx(10.0)  # 240 px / 24 A
    lay = overlay_layout(None, (400, 240), scale_bar=True, camera=CAMERA)
    sb = lay["scale_bar"]
    assert sb["length"] == 5.0 and sb["x1"] - sb["x0"] == pytest.approx(50.0) and sb["label"] == "5 Å"
    assert overlay_layout(None, (400, 240), scale_bar=20, camera=CAMERA)["scale_bar"]["label"] == "2 nm"


def test_raytrace_overlays_drawn(tmp_path, cu_fcc):
    from ewald.render import render

    plain = render(cu_fcc, None, size=(160, 120), renderer="raytrace", samples=1, ao_samples=0, camera=CAMERA)
    marked = render(cu_fcc, tmp_path / "rt.png", size=(160, 120), renderer="raytrace", samples=1, ao_samples=0,
                    camera=CAMERA, type_legend=True, scale_bar=True, dpi=300)
    assert plain.shape == marked.shape == (120, 160, 3)
    # swatch top-left, bar bottom-left; the rest of the image is unchanged
    assert np.abs(marked[:20, :30].astype(int) - plain[:20, :30]).sum() > 0
    assert (marked[108:114, 8:40].sum(-1) < 100).any()
    from PIL import Image

    assert round(Image.open(tmp_path / "rt.png").info["dpi"][0]) == 300


@pytest.mark.parametrize("ext", ["svg", "pdf", "eps"])
def test_vector_formats(tmp_path, cu_fcc, ext):
    from ewald.render import render

    pipe = Pipeline(StaticSource(cu_fcc), [ColorCoding("Position.Z")])
    out = tmp_path / f"fig.{ext}"
    img = render(pipe.compute(0), out, size=(320, 240), renderer="vector", camera=CAMERA, bonds=2.8,
                 text="Cu fcc", type_legend=True, scale_bar=True)
    head = out.read_bytes()[:200]
    assert {"svg": b"<svg", "pdf": b"%PDF", "eps": b"%!PS"}[ext] in head
    assert img.shape == (240, 320, 3) and img.std() > 5
    if ext == "svg":
        svg = out.read_text()
        assert "Cu fcc" in svg or "Cu" in svg  # text stays text
        assert "<image" not in svg             # no embedded raster


def test_vector_paints_near_atoms_last():
    from ewald.render import render

    d = DataCollection(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), ["Cu", "O"])  # O nearer to a +z camera
    cam = {"position": (0, 0, 20), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
           "parallel_scale": 3.0}
    img = render(d, None, size=(60, 60), renderer="vector", camera=cam, show_cell=False, shading="flat")
    center = img[30, 30].astype(int)
    assert center[0] > 200 and center[1] < 100  # red O in front of the copper atom


def test_raster_renderer_to_pdf(tmp_path, cu_fcc):
    from ewald.render import render

    out = tmp_path / "rt.pdf"
    render(cu_fcc, out, size=(80, 60), renderer="raytrace", samples=1, ao_samples=0, camera=CAMERA)
    assert out.read_bytes().startswith(b"%PDF")


def test_unknown_renderer(cu_fcc):
    from ewald.render import render

    with pytest.raises(ValueError, match="vector"):
        render(cu_fcc, None, renderer="ospray", camera=CAMERA)


def test_gif_and_png_sequence(tmp_path, moving):
    from PIL import Image

    from ewald.render import render_animation

    seen = []
    n = render_animation(moving, tmp_path / "m.gif", fps=5, size=(81, 61), renderer="vector", camera=CAMERA,
                         text="frame [Frame]", progress=lambda i, n: seen.append(i))
    assert n == 5 and seen == [1, 2, 3, 4, 5]
    gif = Image.open(tmp_path / "m.gif")
    assert gif.n_frames == 5 and gif.size == (81, 61)
    os.makedirs(tmp_path / "seq")
    n = render_animation(moving, tmp_path / "seq" / "f.png", frames=[0, 2, 4], size=(40, 30), renderer="vector", camera=CAMERA,
                         transparent_background=True)
    files = sorted(os.listdir(tmp_path / "seq"))
    assert n == 3 and files == ["f_0000.png", "f_0001.png", "f_0002.png"]
    assert Image.open(tmp_path / "seq" / files[0]).mode == "RGBA"
    stop = render_animation(moving, tmp_path / "s.gif", size=(40, 30), renderer="vector", camera=CAMERA,
                            progress=lambda i, n: i < 2)
    assert stop == 2


@pytest.mark.parametrize("ext", ["mp4", "avi"])
def test_video_formats(tmp_path, moving, ext):
    from ewald.movie import ffmpeg_executable
    from ewald.render import render_animation

    if ffmpeg_executable() is None:
        pytest.skip("no ffmpeg")
    out = tmp_path / f"m.{ext}"
    assert render_animation(moving, out, fps=5, size=(81, 61), renderer="vector", camera=CAMERA) == 5
    assert out.stat().st_size > 1000


def test_movie_without_ffmpeg_explains(tmp_path, moving, monkeypatch):
    from ewald import movie

    monkeypatch.setattr(movie, "ffmpeg_executable", lambda: None)
    with pytest.raises(RuntimeError, match="pip install imageio-ffmpeg"):
        movie.MovieWriter(tmp_path / "m.mp4")
    with pytest.raises(ValueError, match="unsupported"):
        movie.MovieWriter(tmp_path / "m.xyz")


def _tag(frame, data, *, value=1):
    data["Tag"] = np.full(data.count, value)


def test_session_roundtrip_static_and_code(tmp_path, cu_fcc):
    from ewald.session import load_session, save_session

    cu_fcc.attributes["Note"] = "bulk"
    pipe = Pipeline(StaticSource(cu_fcc), [Replicate((2, 1, 1)), SelectType({"Cu"}),
                                           ColorCoding("Position.Z", cmap="plasma", start=0.0, end=5.0),
                                           PythonScript(), PythonModifier(_tag, value=7)])
    pipe.modifiers[1].enabled = False
    style = {"radius_scale": 0.5, "type_colors": {"Cu": "#c8794a"}, "bonds_enabled": True,
             "bond_pairs": [["Cu", "Cu", 0.0, 2.8]], "preset": "Ball & stick"}
    path = save_session(tmp_path / "a.ewald", pipe, frame=0, camera=CAMERA, style=style,
                        render={"renderer": "vector", "size": (200, 150), "scale_bar": True})
    doc = json.loads(open(path).read())
    assert doc["ewald_session"] == 1 and [m["type"] for m in doc["modifiers"]][:3] == \
        ["Replicate", "SelectType", "ColorCoding"]

    with pytest.warns(UserWarning, match="not run"):
        safe = load_session(path)
    names = [(type(m).__name__, m.enabled) for m in safe.pipeline.modifiers]
    assert names == [("Replicate", True), ("SelectType", False), ("ColorCoding", True), ("PythonScript", False),
                     ("PythonScript", False)]
    assert len(safe.warnings) == 2
    assert safe.pipeline.modifiers[0].n == (2, 1, 1) and list(safe.pipeline.modifiers[1].types) == ["Cu"]
    assert safe.camera["position"] == CAMERA["position"]
    out = safe.pipeline.compute(0)
    assert out.count == 2 * cu_fcc.count and out.attributes["Note"] == "bulk"
    np.testing.assert_allclose(out.positions[: cu_fcc.count], cu_fcc.positions)

    trusted = load_session(path, allow_code=True)
    data = trusted.pipeline.compute(0)
    assert (data["Tag"] == 7).all() and "Upper" in data  # console function and script both ran
    kw = trusted.render_kwargs()
    assert kw["renderer"] == "vector" and kw["size"] == (200, 150) and kw["radius_scale"] == 0.5
    assert kw["bond_pairs"] == [("Cu", "Cu", 0.0, 2.8)] and kw["type_colors"] == {"Cu": "#c8794a"}
    from ewald.render import render

    img = render(data, None, camera=trusted.camera, **kw)
    assert img.shape == (150, 200, 3)


def test_session_file_source_moves_with_folder(tmp_path, vasp_dir):
    from ewald import import_file
    from ewald.session import load_session, save_session

    src = next(p for p in sorted(vasp_dir.iterdir()) if "XDATCAR" in p.name)
    work = tmp_path / "project"
    work.mkdir()
    shutil.copy(src, work / "XDATCAR")
    pipe = import_file(str(work / "XDATCAR"))
    save_session(work / "s.ewald", pipe)
    moved = tmp_path / "moved"
    shutil.move(str(work), str(moved))
    s = load_session(moved / "s.ewald")
    assert s.pipeline.num_frames == pipe.num_frames
    assert os.path.dirname(s.pipeline.source.path) == str(moved)


def test_session_rejects_unknown_and_unsaveable(tmp_path, cu_fcc):
    from ewald.session import load_session, save_session

    pipe = Pipeline(StaticSource(cu_fcc), [lambda frame, data: None])
    with pytest.raises(TypeError, match="PythonScript"):
        save_session(tmp_path / "x.ewald", pipe)  # a lambda has no retrievable source
    path = tmp_path / "y.ewald"
    save_session(path, Pipeline(StaticSource(cu_fcc), [Replicate()]))
    doc = json.loads(path.read_text())
    doc["modifiers"].append({"type": "Nope", "module": "not_a_module", "parameters": {}})
    path.write_text(json.dumps(doc))
    with pytest.warns(UserWarning, match="not available"):
        s = load_session(path)
    assert [m.name for m in s.pipeline.modifiers] == ["Replicate"]


def test_cli_render_and_movie_from_session(tmp_path, moving, capsys):
    from ewald.__main__ import main
    from ewald.session import save_session

    sess = save_session(tmp_path / "m.ewald", moving, camera=CAMERA,
                        render={"renderer_name": "vector", "size": (64, 48), "text_template": "f[Frame]"})
    main(["render", str(sess), str(tmp_path / "f.svg"), "--frame", "2", "--type-legend", "--scale-bar"])
    assert "f2" in (tmp_path / "f.svg").read_text()
    main(["movie", str(sess), str(tmp_path / "m.gif"), "--every", "2"])
    assert "(3 frames)" in capsys.readouterr().out


def test_quality_presets(cu_fcc):
    from ewald.render import QUALITY_PRESETS, quality_settings, render

    assert quality_settings()["samples"] == QUALITY_PRESETS["publication"]["samples"]
    assert quality_settings("draft", samples=3)["samples"] == 3            # explicit wins
    assert quality_settings("Standard")["ao_samples"] == QUALITY_PRESETS["standard"]["ao_samples"]
    with pytest.raises(ValueError, match="publication"):
        quality_settings("ultra")
    img = render(cu_fcc, None, size=(160, 120), renderer="raytrace", camera=CAMERA, quality="draft")
    assert img.shape == (60, 80, 3)                                         # draft renders at half size
    img = render(cu_fcc, None, size=(160, 120), renderer="vector", camera=CAMERA, quality="draft", scale=1.0)
    assert img.shape == (120, 160, 3)


def test_session_quality_preset_wins_over_stored_fields(moving):
    from ewald.session import Session

    s = Session(pipeline=moving, render={"quality": "Draft", "samples": 16, "shadows": True})
    kw = s.render_kwargs()
    assert kw["quality"] == "draft" and "samples" not in kw and "shadows" not in kw
    s = Session(pipeline=moving, render={"quality": "Custom", "samples": 3})
    assert s.render_kwargs() == {"samples": 3}


def test_vector_draws_bonds_and_polyhedra(tmp_path):
    from ase.spacegroup import crystal

    from ewald.modifiers import CoordinationPolyhedra, CreateBonds
    from ewald.render import render

    a = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                cellpar=[3.905] * 3 + [90] * 3)
    pipe = Pipeline(StaticSource(DataCollection.from_ase(a)), [CreateBonds(cutoff=2.0)])
    cam = {"position": (1.95, -30, 1.95), "focal_point": (1.95, 1.95, 1.95), "up": (0, 0, 1),
           "parallel_projection": True, "parallel_scale": 3.5}
    bonds_only = render(pipe.compute(0), None, size=(120, 120), renderer="vector", camera=cam, radius_scale=0.3,
                        show_cell=False)
    pipe.modifiers.append(CoordinationPolyhedra(center_types=["Ti"], transparency=0.0))
    out = tmp_path / "poly.svg"
    with_poly = render(pipe.compute(0), out, size=(120, 120), renderer="vector", camera=cam, radius_scale=0.3,
                       show_cell=False)
    red = (bonds_only[..., 0] > 200) & (bonds_only[..., 1] < 80)
    assert red.sum() > 50                                   # O atoms and their red half-bonds (data.bonds)
    def gray(img):
        return ((np.ptp(img.astype(int), axis=-1) < 20) & (img.astype(int).sum(-1) < 600)).sum()

    assert gray(with_poly) > gray(bonds_only) + 1000         # the opaque gray octahedron covers the middle
    assert out.read_text().count("<path") > 20


def test_session_keeps_bond_modifiers(tmp_path, cu_fcc):
    from ewald.modifiers import CoordinationPolyhedra, CreateBonds, EditBonds
    from ewald.session import load_session, save_session

    pipe = Pipeline(StaticSource(cu_fcc), [CreateBonds(mode="pairwise", pairs=[("Cu", "Cu", 0.0, 2.7)]),
                                           EditBonds(add=[(0, 5)], remove=[(0, 1)]),
                                           CoordinationPolyhedra(center_types=["Cu"], transparency=0.4)])
    s = load_session(save_session(tmp_path / "b.ewald", pipe))
    assert [repr(m) for m in s.pipeline.modifiers] == [repr(m) for m in pipe.modifiers]
    assert s.pipeline.compute(0).bonds.count == pipe.compute(0).bonds.count
