"""ewald as a plain Python library: in-memory sources, frame indexing, interop, headless
rendering and the error messages a script author sees (the API audit of 2026-10-24)."""
import warnings

import numpy as np
import pytest
from ase.build import bulk

import ewald
from ewald import DataCollection, examples, export_file, load
from ewald import modifiers as M
from ewald import render as render_mod


@pytest.fixture
def cu():
    return bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))


@pytest.fixture
def dump():
    return load(examples.path("Cu_void.lammpstrj"))


def test_load_takes_frames_in_memory(cu):
    assert load(cu).num_frames == 1
    assert load([cu, cu]).num_frames == 2
    assert load(DataCollection.from_ase(cu))[0].count == 32
    with pytest.raises(TypeError, match="frames in memory"):
        load(cu, format="vasp")
    with pytest.raises(TypeError, match="cannot use a int"):
        load([1, 2])


def test_pymatgen_round_trip(cu):
    pytest.importorskip("pymatgen.core")
    structure = DataCollection.from_ase(cu).to_pymatgen()
    assert type(structure).__name__ == "Structure" and len(structure) == 32
    data = load(structure)[0]
    assert data.count == 32 and set(data.types) == {"Cu"}
    np.testing.assert_allclose(data.cell.matrix, cu.cell[:], atol=1e-9)


def test_pipeline_index_iterate_and_to_ase(dump):
    assert dump[-1].attributes["Timestep"] == dump.compute(4).attributes["Timestep"]
    assert [d.attributes["SourceFrame"] for d in dump[1:4]] == [1, 2, 3]
    assert sum(1 for _ in dump) == len(dump) == 5
    with pytest.raises(IndexError, match="out of range"):
        dump[5]
    with pytest.raises(TypeError):
        dump["Position"]
    images = dump.to_ase([0, -1])
    assert len(images) == 2 and len(images[1]) == 462


def test_data_apply_leaves_the_frame_alone(cu):
    data = DataCollection.from_ase(cu)
    out = data.apply(M.CoordinationAnalysis(cutoff=3.0))
    assert (out["Coordination"] == 12).all() and "Coordination" not in data


def test_export_frames_forms(dump, tmp_path):
    export_file(dump, tmp_path / "all.extxyz", frames="all")
    export_file(dump, tmp_path / "even.extxyz", frames=slice(None, None, 2))
    export_file(dump, tmp_path / "POSCAR", format="vasp", frames="last")
    assert load(tmp_path / "all.extxyz").num_frames == 5
    assert load(tmp_path / "even.extxyz").num_frames == 3
    assert load(tmp_path / "POSCAR")[0].count == 462
    with pytest.raises(TypeError, match="'all', 'last'"):
        export_file(dump, tmp_path / "x.extxyz", frames="every")
    with pytest.raises(IndexError):
        export_file(dump, tmp_path / "x.extxyz", frames=[7])


def test_type_map_wins_over_the_element_column():
    # Cu_void.lammpstrj has both a numeric type and an element column
    assert set(load(examples.path("Cu_void.lammpstrj"))[0].types) == {"Cu"}
    assert set(load(examples.path("Cu_void.lammpstrj"), type_map={1: "Ag"})[0].types) == {"Ag"}


@pytest.mark.parametrize("modifier", [M.CoordinationAnalysis(cutoff=-1), M.ClusterAnalysis(cutoff=0),
                                      M.CreateBonds(cutoff=-2.0)])
def test_nonsense_cutoffs_raise(dump, modifier):
    dump.add(modifier)
    with pytest.raises(ValueError, match="must be a number above 0"):
        dump.compute()


def test_select_type_warns_about_a_missing_type(dump):
    dump.add(M.SelectType("Ag"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = dump.compute()
    assert data.attributes["SelectType.num_selected"] == 0
    assert any("no particles of type Ag" in str(w.message) for w in caught)


def test_unknown_view_setting_is_a_type_error_with_a_hint(cu):
    with pytest.raises(TypeError, match="Did you mean 'style'") as info:
        ewald.view(load(cu), stlye="Points")
    assert isinstance(info.value, KeyError)  # what it raised before


def test_writing_into_a_frame_says_how_to_replace_it(dump):
    @dump.add
    def shift(frame, data):
        data.positions[:, 0] += 1

    with pytest.raises(ValueError, match="data.positions = data.positions"):
        dump.compute()
    assert "replace them" in dump.modifiers[0].status


def test_show_is_top_level_and_paint_stays_a_module():
    assert "show" in ewald.__all__ and callable(ewald.show)
    # ewald.paint is the module (paint, hide, isolate, ...); a function of that name would hide it
    from ewald import paint

    assert hasattr(paint, "hide")


@pytest.mark.parametrize("env,libs,expected", [
    ({"DISPLAY": ":0"}, (), True),
    ({}, (), False),                       # cluster node, container: VTK would abort the process
    ({}, ("EGL",), True),
    ({}, ("OSMesa",), True),
    ({"EWALD_OPENGL": "0", "DISPLAY": ":0"}, (), False),
    ({"EWALD_OPENGL": "1"}, (), True),
])
def test_opengl_available_on_linux(monkeypatch, env, libs, expected):
    monkeypatch.setattr("sys.platform", "linux")
    assert render_mod.opengl_available(env, find_library=lambda name: name if name in libs else None) is expected


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_opengl_available_on_windows_and_macos(monkeypatch, platform):
    monkeypatch.setattr("sys.platform", platform)
    assert render_mod.opengl_available({}, find_library=lambda name: None)


def test_headless_render_falls_back_to_the_ray_tracer(monkeypatch, cu):
    monkeypatch.setenv("EWALD_OPENGL", "0")
    used = {}

    def fake_raytrace(data, filename, **kwargs):
        used["raytrace"] = True
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(render_mod, "_render_raytraced", fake_raytrace)
    load(cu).render(size=(40, 30), quality="draft")
    assert used == {"raytrace": True}
    with pytest.raises(RuntimeError, match="needs an OpenGL context"):
        load(cu).render(size=(40, 30), renderer="opengl")
