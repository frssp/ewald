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


# ---- behaviour changes chosen by the owner (D-number in docs/developer/decisions.md)

def test_compute_out_of_range_raises(dump):
    assert dump.compute(-5).attributes["SourceFrame"] == 0
    for frame in (5, -6, 999):
        with pytest.raises(IndexError, match="out of range"):
            dump.compute(frame)


def test_verb_never_changes_a_users_step(dump):
    dump.add(M.CoordinationAnalysis(cutoff=3.0))
    before = dump.compute()["Coordination"].copy()
    g = ewald.rdf(dump, cutoff=6.0)
    data = dump.compute()
    np.testing.assert_array_equal(data["Coordination"], before)
    assert g.name == "coordination-rdf (rdf)" and g.x.max() > 5.5
    assert data.tables["coordination-rdf"]["r"].max() < 3.0           # the user's table stays
    assert set(data.tables) == {"coordination-rdf", "coordination-rdf (rdf)"}
    assert len(dump.modifiers) == 2
    ewald.rdf(dump, cutoff=5.0)                                         # replaces its own step
    assert len(dump.modifiers) == 2 and dump.modifiers[1].cutoff == 5.0


def test_verb_alone_still_writes_its_results(dump):
    counts = ewald.cna(dump).counts
    data = dump.compute()
    assert counts["FCC"] == data.attributes["CommonNeighborAnalysis.counts.FCC"] > 0
    assert "Structure Type" in data and "Color" in data


def test_verb_table_attributes_are_its_own(dump):
    dump.add(M.CommonNeighborAnalysis(mode="fixed", cutoff=2.0))         # finds no FCC
    table = ewald.cna(dump)
    assert table.attributes["CommonNeighborAnalysis.counts.FCC"] > 0
    assert dump.compute().attributes["CommonNeighborAnalysis.counts.FCC"] == 0


def test_render_returns_an_image_with_a_short_repr(monkeypatch, cu, tmp_path):
    monkeypatch.setattr(render_mod, "_render_raytraced",
                        lambda data, filename, **kw: np.full((30, 40, 3), 200, np.uint8))
    img = load(cu).render(tmp_path / "x.png", renderer="raytrace", size=(40, 30))
    assert repr(img) == f"<Image 40x30 RGB, saved to {tmp_path / 'x.png'}>"
    assert img.shape == (30, 40, 3) and img[0, 0, 0] == 200 and img._repr_png_()[:4] == b"\x89PNG"
    assert "Image" not in repr(img.astype(float).mean(axis=2))         # computed arrays print as arrays


def test_show_returns_a_plot_or_the_structure(monkeypatch, dump, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from ewald.views import Plot, Snapshot, view_args

    ewald.rdf(dump, cutoff=5.0)
    plot = ewald.show(dump, table="coordination-rdf", save=tmp_path / "g.pdf")
    assert isinstance(plot, Plot) and plot.figure.axes and (tmp_path / "g.pdf").stat().st_size > 0
    assert plot.savefig == plot.figure.savefig
    monkeypatch.setattr(Snapshot, "image", lambda self, **kw: np.zeros((3, 4, 3), np.uint8))
    assert isinstance(ewald.show(dump), Snapshot)
    # a bare show(pipe) cell is a picture, not a plot cell of the GUI
    assert view_args("show(pipe)") is None and view_args("show(pipe, histogram='Coordination')")
    import matplotlib.pyplot as plt

    plt.close("all")   # Agg's plt.show() leaves them open; later notebook tests collect open figures


# ---- one name per idea (D55): new names, OVITO's and older spellings still accepted

@pytest.mark.parametrize("cls,kwargs,old,new", [
    (M.CoordinationAnalysis, {"cutoff": 3.0}, "number_of_bins", "bins"),
    (M.StructureFactor, {}, "number_of_bins", "bins"),
    (M.Histogram, {"property": "Mass"}, "bin_count", "bins"),
    (M.SpatialBinning, {"property": "Mass"}, "bin_count", "bins"),
    (M.VanHoveCorrelation, {}, "nbins", "bins"),
    (M.SiteHopping, {}, "nbins", "bins"),
    (M.PowderDiffraction, {}, "step", "two_theta_step"),
    (M.ClusterAnalysis, {"cutoff": 3.0}, "unwrap_particles", "unwrapped"),
    (M.GenerateTrajectoryLines, {}, "unwrap_trajectories", "unwrapped"),
])
def test_old_argument_names_still_work(cls, kwargs, old, new):
    value = 7 if new != "unwrapped" else True
    m = cls(**kwargs, **{old: value})
    assert getattr(m, new) == value and getattr(m, old) == value
    assert new in m.parameters() and old not in m.parameters() and f"{new}=" in repr(m)
    setattr(m, old, value)      # an old attribute name sets the new one
    assert old not in vars(m)
    with pytest.raises(TypeError, match="old name"):
        cls(**kwargs, **{old: value, new: value})


def test_histogram_and_color_coding_take_a_range(dump):
    h = M.Histogram("Position.Z", bin_count=4, fix_xrange=True, xrange_start=1.0, xrange_end=9.0)
    assert h.range == (1.0, 9.0) and h.bins == 4
    assert M.Histogram("Position.Z", fix_xrange=False, xrange_start=1.0).range is None
    c = M.ColorCoding("Position.Z", start=2.0, end=8.0)
    assert c.range == (2.0, 8.0) and (c.start, c.end, c.start_value) == (2.0, 8.0, 2.0)
    c.end = 9.0
    assert c.range == (2.0, 9.0)
    dump.add(M.ColorCoding("Position.Z", range=(0.0, 10.0)))
    assert dump.compute().attributes["ColorCoding.range"] == (0.0, 10.0)


def test_sessions_and_subclasses_keep_old_names():
    from ewald.session import modifier_from_dict

    m, warning = modifier_from_dict({"type": "CoordinationAnalysis", "parameters": {"cutoff": 3.0, "number_of_bins": 30}})
    assert warning is None and m.bins == 30

    class MyHistogram(M.Histogram):   # no __init__ of its own
        pass

    assert MyHistogram("Mass", bin_count=9).bins == 9


def test_transport_steps_use_every_particle_unless_asked():
    for cls in (M.VanHoveCorrelation, M.SelfIntermediateScattering, M.NonGaussianParameter, M.SiteHopping,
                M.MeanSquaredDisplacement, M.IonicConductivity):
        assert cls().only_selected is False


def test_kernel_bins_and_unported_ovito_modifiers():
    from ewald import kernels
    from ewald.data import Cell

    pos = np.random.default_rng(0).random((30, 3)) * 6
    a = kernels.rdf(pos, Cell(np.eye(3) * 6), 2.5, bins=40)
    b = kernels.rdf(pos, Cell(np.eye(3) * 6), 2.5, nbins=40)
    np.testing.assert_array_equal(a["g(r)"], b["g(r)"])
    with pytest.raises(TypeError, match="unexpected keyword"):
        kernels.rdf(pos, Cell(np.eye(3) * 6), 2.5, bns=40)
    with pytest.raises(ImportError, match="OvitoModifier"):
        from ewald.modifiers import ConstructSurfaceModifier  # noqa: F401
    assert M.IdentifyDiamondStructureModifier is M.IdentifyDiamond
