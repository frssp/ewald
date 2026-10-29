"""Trajectory modifiers: MSD/diffusion, VACF, time averaging, time series, trajectory lines."""
import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource, import_file
from ewald.backend import CPU, Backend
from ewald.data import Cell
from ewald.kernels import (autocorrelation_fft, collect_trajectory, diffusion_fit, fit_lags, green_kubo, msd_fft,
                           particle_msd, path_statistics, trajectory_rdf, trajectory_vacf)
from ewald.modifiers import (CoordinationAnalysis, ExpressionSelection, GenerateTrajectoryLines, MeanSquaredDisplacement,
                             SelectType,
                             TimeAveraging, TimeSeries, VelocityAutocorrelation)


class _NumpyAsGPU:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


@pytest.fixture(params=["cpu", "fake"])
def backend(request):
    return CPU if request.param == "cpu" else Backend("fake-cuda", _NumpyAsGPU())


BOX = 10.0


class _CountingSource(StaticSource):
    reads = 0

    def compute(self, frame):
        self.reads += 1
        return super().compute(frame)


def _walk_pipeline(rng, T=40, n=60, sigma=0.3, shuffle=True, ids=True, velocities=False):
    """Random walk wrapped into a cubic box; every stored frame shuffles the particle order."""
    true = np.cumsum(rng.normal(scale=sigma, size=(T, n, 3)), axis=0) + rng.random((n, 3)) * BOX
    types = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    frames = []
    for k, x in enumerate(true):
        perm = rng.permutation(n) if shuffle else np.arange(n)
        d = DataCollection((x % BOX)[perm], types[perm], Cell(np.diag([BOX] * 3)))
        if ids:
            d.particles["Particle Identifier"] = perm + 1
        if velocities:
            v = true[min(k + 1, T - 1)] - true[k] if k < T - 1 else np.zeros((n, 3))
            d.particles["Velocity"] = v[perm]
        d.attributes["Timestep"] = 100 * k
        d.attributes["Energy"] = float(k)
        frames.append(d)
    return Pipeline(_CountingSource(frames)), true


def test_msd_modifier_matches_fft_and_einstein(rng, backend):
    pipe, true = _walk_pipeline(rng, T=60, n=400)
    pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=0.5))
    data = pipe.compute(3, device=backend)
    table = data.tables["msd"]
    np.testing.assert_allclose(table["MSD"], msd_fft(true), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(table["MSD B"], msd_fft(true[:, 200:]), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(table["Time"], np.arange(60) * 0.5)
    # random walk: MSD = 3 sigma^2 t / dt -> D = sigma^2 / (2 dt) = 0.09
    d = data.attributes["MeanSquaredDisplacement.diffusion_coefficient"]
    assert d == pytest.approx(0.09, rel=0.15)
    assert "MeanSquaredDisplacement.diffusion_coefficient.A" in data.attributes
    assert len(table["MSD fit"]) == 60


def test_particle_msd_is_each_atoms_time_averaged_msd(rng, backend):
    x = np.cumsum(rng.normal(size=(30, 9, 3)), axis=0)
    lags = slice(4, 20)
    got = particle_msd(x, lags, backend, chunk_bytes=2000)          # several chunks of particles
    direct = [np.mean([((x[m:, i] - x[:-m, i]) ** 2).sum(-1).mean() for m in range(4, 20)]) for i in range(9)]
    np.testing.assert_allclose(got, direct, rtol=1e-9)
    assert got.mean() == pytest.approx(msd_fft(x)[lags].mean())
    assert fit_lags(100, (0.2, 0.8)) == slice(20, 80)


def test_msd_writes_per_atom_msd_and_top_selection_traces_the_fastest(rng):
    """The atoms with the largest MSD, as steps: MSD writes 'MSD' per atom (matched by id although
    every frame shuffles the order), ExpressionSelection(top=, by=) ranks, trajectory lines trace."""
    pipe, true = _walk_pipeline(rng, T=40, n=60)
    fast = np.arange(60) % 7 == 3                                    # every 7th atom moves 3x farther
    true = true[0] + (true - true[0]) * np.where(fast, 3.0, 1.0)[None, :, None]
    for d, x in zip(pipe.source.frames, true):
        d.positions = (x % BOX)[np.asarray(d["Particle Identifier"]) - 1]
    pipe.modifiers.extend([MeanSquaredDisplacement(), ExpressionSelection("ParticleType == 'A'", top=3, by="MSD"),
                           GenerateTrajectoryLines()])
    data = pipe.compute(5)
    ids = np.asarray(data["Particle Identifier"])
    expected = particle_msd(true, fit_lags(40, (0.2, 0.8)))
    np.testing.assert_allclose(data["MSD"], expected[ids - 1], rtol=1e-9)
    a_fast = np.nonzero(fast[:30])[0] + 1                            # the fast atoms among type A (ids 1..30)
    top3 = (np.arange(1, 31))[np.argsort(-expected[:30])[:3]]
    assert set(top3) <= set(a_fast)
    traced = set(np.unique(data.tables["trajectory-lines"]["Particle Identifier"]).tolist())
    assert traced == set(top3.tolist())
    smallest = ExpressionSelection("True", top=-2, by="MSD")
    pipe.modifiers[1:] = [smallest]
    first = pipe.compute(0)
    picked = np.asarray(first["Particle Identifier"])[first.selection.astype(bool)]
    assert set(picked.tolist()) == set((np.argsort(expected)[:2] + 1).tolist())
    smallest.by = ""
    with pytest.raises(ValueError, match="needs by"):
        pipe.compute(0)


def test_msd_by_axis_shows_diffusion_confined_to_a_plane(rng):
    pipe, true = _walk_pipeline(rng, T=60, n=400, shuffle=False)
    flat = true.copy()
    flat[:, :, 2] = flat[0, :, 2]          # no motion along z: diffusion in the xy plane
    frames = []
    for d, x in zip(pipe.source.frames, flat):
        d = d.copy()
        d.positions = x % BOX
        frames.append(d)
    pipe = Pipeline(StaticSource(frames))
    pipe.modifiers.append(MeanSquaredDisplacement(time_per_frame=0.5, by_axis=True))
    data = pipe.compute(0)
    t, a = data.tables["msd"], data.attributes
    np.testing.assert_allclose(t["MSD x"] + t["MSD y"] + t["MSD z"], t["MSD"], rtol=1e-9, atol=1e-9)
    assert np.allclose(t["MSD z"], 0, atol=1e-9) and "MSD A z" in t
    name = "MeanSquaredDisplacement.diffusion_coefficient"
    # 1D coefficients (slope/2): each in-plane axis has D = sigma^2 / (2 dt) = 0.09, z none
    assert a[name + ".x"] == pytest.approx(0.09, rel=0.15) and a[name + ".z"] == pytest.approx(0, abs=1e-9)
    assert a[name + ".B.y"] == pytest.approx(0.09, rel=0.2)


def test_trajectory_result_is_cached_until_inputs_change(rng):
    pipe, _ = _walk_pipeline(rng, T=12, n=10)
    msd = MeanSquaredDisplacement()
    pipe.modifiers.append(msd)
    pipe.compute(0)
    reads = pipe.source.reads
    for k in range(12):
        pipe.compute(k)
    assert pipe.source.reads - reads <= 12  # one source read per compute, no re-sweep of the trajectory
    first = pipe.compute(0).tables["msd"]["MSD"]
    msd.step = 2
    assert len(pipe.compute(0).tables["msd"]["MSD"]) == 6
    msd.step = 1
    pipe.modifiers.insert(0, SelectType("A"))
    msd.only_selected = True
    sub = pipe.compute(0).tables["msd"]["MSD"]
    assert not np.allclose(sub, first)


def test_msd_only_selected_uses_upstream_selection(rng):
    pipe, true = _walk_pipeline(rng, T=20, n=30)
    pipe.modifiers.extend([SelectType("B"), MeanSquaredDisplacement(only_selected=True, by_type=False)])
    np.testing.assert_allclose(pipe.compute(0).tables["msd"]["MSD"], msd_fft(true[:, 15:]), rtol=1e-9, atol=1e-9)


def test_msd_without_identifiers_and_drift(rng):
    pipe, true = _walk_pipeline(rng, T=20, n=30, shuffle=False, ids=False)
    pipe.modifiers.append(MeanSquaredDisplacement(subtract_drift=True))
    got = pipe.compute(0).tables["msd"]["MSD"]
    np.testing.assert_allclose(got, msd_fft(true - true.mean(1, keepdims=True)), rtol=1e-9, atol=1e-9)


def test_autocorrelation_matches_direct(rng, backend):
    v = rng.normal(size=(25, 7, 3))
    direct = np.array([np.mean(np.sum(v[m:] * v[:len(v) - m], -1)) for m in range(len(v))])
    np.testing.assert_allclose(autocorrelation_fft(v, backend, chunk_bytes=3000), direct, rtol=1e-10, atol=1e-12)


def test_diffusion_helpers():
    t = np.arange(50) * 0.1
    d, icpt, (t0, t1) = diffusion_fit(t, 6 * 0.25 * t + 1.0)
    assert d == pytest.approx(0.25) and icpt == pytest.approx(1.0) and 0 < t0 < t1 < t[-1] + 1e-12
    np.testing.assert_allclose(green_kubo(t, np.full(50, 3.0)), t)


def test_fit_time_fits_the_same_lag_window_in_runs_of_any_length(rng):
    t = np.arange(50) * 0.1
    assert fit_lags(50, time=t, fit_time=(1.0, 2.0)) == slice(10, 21)
    with pytest.raises(ValueError, match="lags run from 0 to 4.9"):
        fit_lags(50, time=t, fit_time=(7.0, 9.0))
    pipe, _ = _walk_pipeline(rng, T=60, n=50)
    windows = []
    for stop in (30, 60):
        pipe.modifiers[:] = [MeanSquaredDisplacement(time_per_frame=0.5, stop=stop, fit_time=(2.0, 6.0))]
        a = pipe.compute(0).attributes
        windows.append((a["MeanSquaredDisplacement.fit_start"], a["MeanSquaredDisplacement.fit_end"]))
    assert windows == [(2.0, 6.0), (2.0, 6.0)]
    pipe.modifiers[:] = [MeanSquaredDisplacement(time_per_frame=0.5, stop=30)]
    a = pipe.compute(0).attributes                # the default fraction (0.2, 0.8) of 30 lags
    assert (a["MeanSquaredDisplacement.fit_start"], a["MeanSquaredDisplacement.fit_end"]) == (3.0, 11.5)


def test_vacf_from_positions_and_velocities_agree(rng, backend):
    pipe, true = _walk_pipeline(rng, T=50, n=300, velocities=True)
    pos = VelocityAutocorrelation(velocities="positions", time_per_frame=1.0)
    pipe.modifiers.append(pos)
    a = pipe.compute(0, device=backend)
    v = np.diff(true, axis=0)
    np.testing.assert_allclose(a.tables["vacf"]["VACF"], autocorrelation_fft(v), rtol=1e-9, atol=1e-12)
    assert a.tables["vacf"]["VACF (normalized)"][0] == pytest.approx(1.0)
    # uncorrelated steps: D = (1/3) * c0 * dt / 2 = sigma^2 / 2, like the MSD
    assert a.attributes["VelocityAutocorrelation.diffusion_coefficient"] == pytest.approx(0.045, rel=0.2)
    pos.velocities = "property"
    b = pipe.compute(0, device=backend).tables["vacf"]
    np.testing.assert_allclose(b["VACF"][:-1][:5], autocorrelation_fft(np.concatenate([v, 0 * v[:1]]))[:5],
                               rtol=1e-9)
    table = trajectory_vacf(pipe, velocities="auto")
    assert len(table["VACF"]) == 50  # the 'Velocity' property was found


def test_time_averaging_rdf_attribute_and_property(rng):
    pipe, true = _walk_pipeline(rng, T=6, n=80)
    pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0, number_of_bins=30))
    pipe.modifiers.append(TimeAveraging(("table:coordination-rdf", "attribute:Energy",
                                         "property:Coordination"), step=2))
    data = pipe.compute(1)
    avg = data.tables["coordination-rdf[average]"]
    plain = Pipeline(pipe.source, [CoordinationAnalysis(cutoff=3.0, number_of_bins=30)])
    per_frame = [plain.compute(k).tables["coordination-rdf"]["g(r)"] for k in (0, 2, 4)]
    np.testing.assert_allclose(avg["g(r)"], np.mean(per_frame, 0))
    np.testing.assert_allclose(avg["g(r)"], trajectory_rdf(pipe, 3.0, 30, step=2)["g(r)"])
    assert data.attributes["Energy (Average)"] == pytest.approx(2.0)
    # per-particle average, matched by identifier in the current (shuffled) frame
    coords = []
    for k in (0, 2, 4):
        d = plain.compute(k)
        coords.append(d["Coordination"][np.argsort(d["Particle Identifier"])])
    expect = np.mean(coords, 0)[data["Particle Identifier"] - 1]
    np.testing.assert_allclose(data["Coordination (Average)"], expect)
    with pytest.raises(ValueError, match="operate_on"):
        TimeAveraging("rdf")._targets()


def test_time_series(rng):
    pipe, _ = _walk_pipeline(rng, T=8, n=5)
    pipe.modifiers.append(TimeSeries())
    ts = pipe.compute(0).tables["time-series"]
    np.testing.assert_array_equal(ts["Frame"], np.arange(8))
    np.testing.assert_array_equal(ts["Timestep"], np.arange(8) * 100)
    np.testing.assert_array_equal(ts["Energy"], np.arange(8.0))


def test_trajectory_lines_and_path_statistics(rng, backend):
    pipe, true = _walk_pipeline(rng, T=15, n=20, sigma=1.0)
    pipe.modifiers.extend([SelectType("A"), GenerateTrajectoryLines(stop=10)])
    data = pipe.compute(12, device=backend)
    lines = data.tables["trajectory-lines"]
    assert lines["Position"].shape == (10 * 10, 3)
    np.testing.assert_array_equal(lines["Particle Identifier"], np.repeat(np.arange(1, 11), 10))
    np.testing.assert_array_equal(lines["Time"][:10], np.arange(10) * 100)
    # unwrapped lines reproduce the true path (up to the wrapped start position)
    first = lines["Position"][:10]
    np.testing.assert_allclose(first - first[0], true[:10, 0] - true[0, 0], atol=1e-9)
    path, net, peak = path_statistics(true[:10])
    ids = data["Particle Identifier"]
    sel = ids <= 10
    np.testing.assert_allclose(data["Path Length"][sel], path[ids[sel] - 1])
    np.testing.assert_allclose(data["Net Displacement"][sel], net[ids[sel] - 1])
    np.testing.assert_allclose(data["Max Displacement"][sel], peak[ids[sel] - 1])
    assert np.all(data["Path Length"][~sel] == 0)


def test_collect_trajectory_rejects_missing_ids(rng):
    pipe, _ = _walk_pipeline(rng, T=3, n=6)
    frames = [pipe.source_frame(k) for k in range(3)]
    frames[2] = frames[2].take(np.arange(5))
    with pytest.raises(ValueError, match="identifiers"):
        collect_trajectory(frames)


def test_trajectory_modifiers_on_lazy_readers(lammps_dir, vasp_dir):
    pipe = import_file(lammps_dir / "dump.custom.lammpstrj")
    pipe.modifiers.extend([MeanSquaredDisplacement(), VelocityAutocorrelation(), GenerateTrajectoryLines(),
                           CoordinationAnalysis(cutoff=3.0), TimeAveraging("table:coordination-rdf")])
    data = pipe.compute(pipe.num_frames - 1)
    assert len(data.tables["msd"]["MSD"]) == pipe.num_frames
    assert np.isfinite(data.tables["vacf"]["VACF"]).all()
    assert len(data.tables["vacf"]["VACF"]) == pipe.num_frames  # 'Velocity' read from vx vy vz
    assert "coordination-rdf[average]" in data.tables

    pipe = import_file(vasp_dir / "XDATCAR")
    pipe.modifiers.extend([MeanSquaredDisplacement(time_per_frame=0.002), VelocityAutocorrelation(),
                           GenerateTrajectoryLines(only_selected=False)])
    data = pipe.compute(0)
    assert abs(data.tables["msd"]["MSD"][0]) < 1e-10 and data.tables["msd"]["MSD"][1] > 0
    assert len(data.tables["vacf"]["VACF"]) == pipe.num_frames - 1  # differentiated positions
    assert data.tables["trajectory-lines"]["Position"].shape == (pipe.num_frames * data.count, 3)


def _traced(rng, **lines):
    pipe, true = _walk_pipeline(rng, T=12, n=8, sigma=0.4)
    pipe.modifiers.append(GenerateTrajectoryLines(only_selected=False, **lines))
    return pipe, true


def _lines(data, frame=0, **kw):
    from ewald import scene as ir

    s = ir.build(data, [ir.TrajectoryLinesVis(**kw)], ir.SceneContext(frame=frame))
    return s, (s["trajectories"].primitive if "trajectories" in s else None)


def test_trajectory_lines_color_by_time_window_and_trail(rng):
    """Trajectory lines follow a color map by Time over a chosen Time window or a trail that ends at the
    current frame, with a color bar that spans the map."""
    pipe, _ = _traced(rng, start=2)
    data = pipe.compute(7)
    table = data.tables["trajectory-lines"]
    np.testing.assert_array_equal(table["Frame"][:10], np.arange(2, 12))
    scene, lines = _lines(data, color_by="Time")
    assert lines.rgb is None and lines.point_rgb.shape == (80, 3) and len(lines.strips) == 8
    assert scene.legends == [{"title": "Time", "cmap": "viridis", "range": (200.0, 1100.0)}]
    from ewald.colormaps import get_colormap

    np.testing.assert_allclose(lines.point_rgb[0], get_colormap("viridis")(0.0)[:3])
    np.testing.assert_allclose(lines.point_rgb[9], get_colormap("viridis")(1.0)[:3])
    scene, lines = _lines(data, color_by="Time", time_range=(400, 800), color_range=(0, None))
    drawn = np.concatenate(lines.strips)
    assert set(table["Time"][drawn]) == {400, 500, 600, 700, 800}
    assert scene.legends[0]["range"] == (0.0, 800.0)   # lo fixed, hi from what is drawn
    # trail: the frame shown (the data's SourceFrame) is 7, Time 700; 'True' draws up to it, a number only that much before it
    scene, lines = _lines(data, 7, trail=True, color_by="Time")
    assert set(table["Time"][np.concatenate(lines.strips)]) == set(range(200, 701, 100))
    assert scene.legends[0]["range"] == (200.0, 1100.0)   # the whole path, so colors keep their meaning
    _, lines = _lines(data, 7, trail=True)
    _, lines = _lines(data, 7, trail=250)
    assert set(table["Time"][np.concatenate(lines.strips)]) == {500, 600, 700}
    assert lines.point_rgb is None and lines.rgb.shape == (8, 3)   # atom colors without color_by
    _, lines = _lines(pipe.compute(1), 1, trail=True)
    assert lines is None                                            # frame 1 is before the traced range
    scene, _ = _lines(data, color_by="Time", colorbar=False)
    assert scene.legends == []


def test_trajectory_lines_as_tubes(rng):
    """style='tube' draws the lines as thin cylinders: one per piece in its color (the mean of its ends for
    a color map), round at both ends of a strip and at bent joints, of ``radius`` (default 0.08)."""
    from ewald.scene import Cylinders, TrajectoryLinesVis

    pipe, _ = _traced(rng)
    data = pipe.compute(5)
    _, lines = _lines(data)
    _, tubes = _lines(data, style="tube")
    assert isinstance(tubes, Cylinders) and tubes.count == len(lines.segments()) == 8 * 11
    seg = lines.segments()
    np.testing.assert_allclose(tubes.start, lines.points[seg[:, 0]])
    np.testing.assert_allclose(tubes.end, lines.points[seg[:, 1]])
    np.testing.assert_allclose(tubes.rgb, lines.segment_colors())
    assert np.all(tubes.radius == 0.08)
    assert np.flatnonzero(tubes.start_cap == 2).tolist() == list(range(0, 88, 11))   # each strip's start
    assert np.all(tubes.start_cap[tubes.start_cap != 2] == 0)
    assert np.all(tubes.end_cap[10::11] == 2)                                        # and its end
    _, lines = _lines(data, color_by="Time", trail=True)
    _, tubes = _lines(data, color_by="Time", trail=True, style="tube", radius=0.2)
    np.testing.assert_allclose(tubes.rgb, lines.segment_colors())
    assert np.all(tubes.radius == 0.2) and tubes.count == len(lines.segments())
    with pytest.raises(ValueError, match="line, tube"):
        TrajectoryLinesVis(style="cylinder")


def test_line_tubes_round_only_bent_joints():
    """A straight run has no caps inside (no beads in the vector renderer); a bend gets a round one."""
    from ewald.scene import Lines, line_tubes

    pts = [[0, 0, 0], [1, 0, 0], [2, 0.1, 0], [3, 0.2, 0], [3, 1.2, 0]]
    tubes = line_tubes(Lines(pts, [[0, 1, 2, 3, 4]], [(1, 0, 0)]), 0.1)
    assert tubes.start_cap.tolist() == [2, 0, 0, 0] and tubes.end_cap.tolist() == [0, 0, 2, 2]
    assert np.allclose(tubes.rgb, (1, 0, 0))


def test_collect_trajectory_matches_by_index_when_the_first_frame_has_no_ids(cu_fcc):
    """A band whose first frame has no identifiers (a POSCAR) and later ones do (OUTCARs) is matched by index."""
    a = cu_fcc.copy()
    b = cu_fcc.copy()
    b.particles["Particle Identifier"] = np.arange(1, b.count + 1)[::-1]
    traj = collect_trajectory([a, b])
    assert traj["Particle Identifier"] is None
    np.testing.assert_array_equal(traj["Position"][1], b.positions)


def test_trajectory_lines_take_the_views_type_colors(rng):
    from ewald import scene as ir

    pipe, _ = _traced(rng)
    data = pipe.compute(3)
    green = {name: "#008000" for name in set(np.asarray(data.types))}
    lines = next(e for e in ir.builtin_elements(type_colors=green) if isinstance(e, ir.TrajectoryLinesVis))
    s = ir.build(data, [lines], ir.SceneContext(frame=3))
    np.testing.assert_allclose(s["trajectories"].primitive.rgb, np.tile([0, 128 / 255, 0], (8, 1)), atol=1e-6)


def test_trajectory_lines_color_by_step_length_and_property(rng):
    pipe, true = _traced(rng)
    data = pipe.compute(0)
    table = data.tables["trajectory-lines"]
    el_scene, lines = _lines(data, color_by="Step Length", colormap="plasma")
    first = table["Particle Identifier"] == 1
    steps = np.linalg.norm(np.diff(true[:, 0], axis=0), axis=1)
    lo, hi = el_scene.legends[0]["range"]
    from ewald.colormaps import get_colormap

    np.testing.assert_allclose(lines.point_rgb[np.flatnonzero(first)[3]],
                               get_colormap("plasma")((steps[2] - lo) / (hi - lo))[:3], atol=1e-9)
    assert hi == pytest.approx(np.linalg.norm(np.diff(true, axis=0), axis=2).max())
    _, lines = _lines(data, color_by="Path Length")      # a particle property: one color per line
    for s in lines.strips:
        assert np.ptp(lines.point_rgb[s], axis=0).max() == 0
    with pytest.raises(KeyError, match="Step Length"):
        _lines(data, color_by="Nope")


def test_lines_vertex_and_segment_colors():
    from ewald.scene import Lines

    pts = np.zeros((4, 3))
    plain = Lines(pts, [[0, 1, 2], [2, 3]], [(1, 0, 0), (0, 0, 1)])
    np.testing.assert_allclose(plain.segment_colors(), [(1, 0, 0), (1, 0, 0), (0, 0, 1)])
    np.testing.assert_allclose(plain.vertex_colors()[[0, 3]], [(1, 0, 0), (0, 0, 1)])
    mapped = Lines(pts, [[0, 1, 2, 3]], point_rgb=[(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 0, 0)])
    np.testing.assert_allclose(mapped.segment_colors()[0], (0.5, 0.5, 0.5))
    assert Lines(pts, [[0, 1]]).vertex_colors() is None
    np.testing.assert_allclose(Lines(pts, [[0, 1]]).segment_colors((0.2, 0.2, 0.2)), [(0.2, 0.2, 0.2)])


def test_trajectory_view_settings_and_elements(rng):
    """The color map is a pipe.view setting, also as view.elements['Trajectory lines'], and the GUI style
    round-trips it."""
    from ewald.view_style import ViewStyle

    pipe, _ = _traced(rng)
    el = pipe.view.elements["Trajectory lines"]
    el.set(color_by="Time", colormap="magma", time_range=(100, None), trail=True)
    assert pipe.view.trajectory_time_range == [100.0, None] and pipe.view.trajectory_trail is True
    kw = pipe.view.render_kwargs(pipe.compute(5), 5)
    assert kw["trajectory_color_by"] == "Time" and kw["trajectory_colormap"] == "magma" and kw["frame"] == 5
    assert "trajectory_color_by='Time'" in pipe.view.source()
    with pytest.raises(ValueError):
        pipe.view.trajectory_color_range = [5, 1]
    with pytest.raises(ValueError):
        pipe.view.trajectory_trail = -3
    with pytest.raises(KeyError):
        pipe.view.trajectory_colormap = "no-such-map"
    style = ViewStyle()
    style.apply_view(pipe.view)
    assert style.scene_kwargs()["trajectory_time_range"] == [100.0, None]
    settings = style.view_settings()
    assert settings["trajectory_colormap"] == "magma" and settings["trajectory_trail"] is True
    pipe.view.update(**settings)
    assert pipe.view.trajectory_color_by == "Time"
    # tubes: a view setting and an element attribute, carried by the GUI style and sessions
    assert "trajectory_style" not in style.scene_kwargs()   # 'line' is the default
    el.set(style="tube", radius=0.12)
    assert pipe.view.trajectory_style == "tube" and "trajectory_style='tube'" in pipe.view.source()
    kw = pipe.view.render_kwargs(pipe.compute(5), 5)
    assert kw["trajectory_style"] == "tube" and kw["trajectory_radius"] == 0.12
    with pytest.raises(ValueError):
        pipe.view.trajectory_style = "ribbon"
    style.apply_view(pipe.view)
    assert style.scene_kwargs()["trajectory_style"] == "tube" and style.view_settings()["trajectory_style"] == "tube"


def test_overlay_layout_stacks_color_bars():
    from ewald.overlays import overlay_layout

    d = DataCollection(np.zeros((1, 3)), np.array(["A"]), Cell(np.diag([BOX] * 3)))
    d.attributes.update({"ColorCoding.range": (0.0, 1.0), "ColorCoding.property": "Charge"})
    lay = overlay_layout(d, (800, 600), colorbars=[{"title": "Time", "cmap": "magma", "range": (0, 5000)}])
    bars = lay["colorbars"]
    assert [b["title"] for b in bars] == ["Charge", "Time"]
    texts = {it.text: it for it in lay["items"] if hasattr(it, "text")}
    assert texts["Time"].x < texts["Charge"].x                    # side by side, the second to the left
    assert [t for t in ("0", "1000", "2000", "3000", "4000", "5000") if t in texts] == \
        ["0", "1000", "2000", "3000", "4000", "5000"]
    assert "colorbars" not in overlay_layout(d, (800, 600), legend=False, colorbars=bars)
