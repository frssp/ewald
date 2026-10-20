import numpy as np
import pytest

from ewald import Pipeline, StaticSource
from ewald import modifiers as mods
from ewald.backend import Backend, get_backend
from ewald.modifiers import (AssignColor, CalculateDisplacements, ColorCoding, ComputeProperty,
                             CoordinationAnalysis, DeleteSelected, ExpressionSelection, Replicate,
                             SelectType, Slice)


def test_fcc_coordination_and_rdf(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(cutoff=3.0, number_of_bins=60)])
    d = pipe.compute(0)
    assert np.all(d["Coordination"] == 12)
    rdf = d.tables["coordination-rdf"]
    assert abs(rdf["r"][np.argmax(rdf["g(r)"])] - 3.615 / np.sqrt(2)) < 0.06


def test_gpu_kernels_run_on_any_array_module(cu_fcc):
    """The CUDA path is the same xp code; run it with numpy standing in for CuPy."""
    fake_gpu = Backend("cuda-sim", np)
    d = Pipeline(StaticSource(cu_fcc), [CoordinationAnalysis(cutoff=3.0)]).compute(0, device=fake_gpu)
    assert np.all(d["Coordination"] == 12)


def test_cuda_request_without_cupy_has_clear_error():
    try:
        import cupy  # noqa: F401
        pytest.skip("CuPy present")
    except ImportError:
        with pytest.raises(RuntimeError, match="CuPy"):
            get_backend("cuda")


def test_selection_delete_and_python_modifier(cu_fcc):
    calls = []

    def tag(frame, data):
        calls.append(frame)
        data.particles["Tag"] = np.arange(data.count)

    pipe = Pipeline(StaticSource(cu_fcc))
    pipe.modifiers.append(tag)
    pipe.modifiers.append(ExpressionSelection("(Position.Z < 3.0) & (ParticleType == 'Cu')"))
    pipe.modifiers.append(DeleteSelected())
    d = pipe.compute(0)
    assert calls == [0]
    assert d.count == cu_fcc.count - int((cu_fcc.positions[:, 2] < 3.0).sum())
    assert "Selection" not in d
    # source frame must stay untouched by modifiers
    assert "Tag" not in pipe.source_frame(0)


def test_expression_error_hint(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [ExpressionSelection("Position.Z > 1 and Position.X > 1")])
    with pytest.raises(ValueError, match="&"):
        pipe.compute(0)


def test_expression_takes_property_names_with_spaces(cu_fcc):
    data = cu_fcc.copy()
    data.particles["Particle Identifier"] = np.arange(cu_fcc.count)
    d = Pipeline(StaticSource(data), [ExpressionSelection("Particle Identifier")]).compute(0)
    assert d["Selection"].sum() == cu_fcc.count - 1
    with pytest.raises(NameError, match="no property 'Charge Density'.*Available"):
        Pipeline(StaticSource(cu_fcc), [ColorCoding("Charge Density")]).compute(0)


def test_displacement_across_pbc(cu_fcc):
    moved = cu_fcc.copy()
    pos = cu_fcc.positions.copy()
    L = cu_fcc.cell.matrix[0, 0]
    pos[0, 0] = pos[0, 0] - 0.3 + (L if pos[0, 0] < 0.3 else 0)  # wraps around x
    moved.positions = pos
    moved.particles["Particle Identifier"] = np.arange(cu_fcc.count)
    ref = cu_fcc.copy()
    ref.particles["Particle Identifier"] = np.arange(cu_fcc.count)
    d = Pipeline(StaticSource([ref, moved]), [CalculateDisplacements(0)]).compute(1)
    np.testing.assert_allclose(d["Displacement"][0], [-0.3, 0, 0], atol=1e-9)
    assert d.attributes["MSD"] == pytest.approx(0.09 / cu_fcc.count)


def test_slice_replicate_color(cu_fcc):
    pipe = Pipeline(StaticSource(cu_fcc), [
        Replicate((2, 1, 1)),
        Slice(normal=(0, 0, 1), distance=1.8, width=1.0),
        SelectType("Cu"), AssignColor("red"),
        ComputeProperty("Height", "Position.Z - CellOrigin.Z"),
        ColorCoding("Height", cmap="plasma"),
    ])
    d = pipe.compute(0)
    assert d.count == 2 * int((np.abs(cu_fcc.positions[:, 2] - 1.8) <= 0.5).sum())
    assert d["Color"].shape == (d.count, 3)
    assert d.attributes["ColorCoding.range"][0] == pytest.approx(d["Height"].min())


def test_frame_cache_respects_memory_budget(cu_fcc):
    frames = [cu_fcc.copy() for _ in range(6)]
    per_frame = cu_fcc.nbytes
    pipe = Pipeline(StaticSource(frames), cache_memory=2.5 * per_frame)
    for f in range(6):
        pipe.compute(f)
    assert list(pipe._cache) == [4, 5]
    pipe.cache_memory = 1  # smaller than one frame: the newest frame is still kept
    pipe.compute(2)
    assert list(pipe._cache) == [2]
    pipe.cache_memory = None
    for f in range(6):
        pipe.compute(f)
    assert len(pipe._cache) == 6


def test_cancel_stops_between_modifiers_and_in_input_frame(cu_fcc):
    import threading

    from ewald import ComputeCancelled, Modifier

    seen = []

    class Record(Modifier):
        def modify(self, data, ctx):
            seen.append(ctx.frame)

    class Trajectory(Modifier):  # reads every frame, like MSD; stops at the first input_frame after cancel
        def modify(self, data, ctx):
            for f in range(ctx.pipeline.num_frames):
                ctx.input_frame(f)
                if f == 1 and armed:
                    cancel.set()

    pipe = Pipeline(StaticSource([cu_fcc] * 5), [Record(), Trajectory(), Record()])
    cancel, armed = threading.Event(), [True]
    with pytest.raises(ComputeCancelled):
        pipe.compute(0, cancel=cancel, errors="skip")  # never swallowed as a modifier error
    assert pipe.modifiers[1].status is None
    assert seen == [0, 0, 1]  # frame 0 itself, then input frames 0 and 1; the last Record never ran
    cancel.clear()
    seen.clear()
    armed.clear()
    pipe.compute(0, cancel=cancel)
    assert len(seen) == 1 + 5 + 1


def test_source_frame_cache_is_thread_safe(cu_fcc):
    from concurrent.futures import ThreadPoolExecutor

    pipe = Pipeline(StaticSource([cu_fcc] * 20), cache_size=3)
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda f: pipe.compute(f % 20).count, range(400)))
    assert set(out) == {cu_fcc.count} and len(pipe._cache) <= 3


def test_output_cache_reuses_results_until_settings_change(cu_fcc):
    calls = []

    class Counting(mods.CoordinationAnalysis):
        cacheable = True

        def modify(self, data, ctx):
            calls.append(ctx.frame)
            return super().modify(data, ctx)

    seen = []

    def script(frame, data):  # never cached: runs on top of the cached result every time
        seen.append(frame)
        data.particles["Marked"] = np.ones(data.count)

    pipe = Pipeline(StaticSource([cu_fcc, cu_fcc]), [Counting(3.0), script])
    a = pipe.compute(0)
    b = pipe.compute(0)
    assert calls == [0] and seen == [0, 0]
    np.testing.assert_array_equal(a["Coordination"], b["Coordination"])
    b.particles["Coordination"] = np.zeros(b.count)  # a caller's copy, not the cache
    assert (pipe.compute(0)["Coordination"] == 12).all()
    pipe.compute(1)
    assert calls == [0, 1]
    pipe.modifiers[0].cutoff = 2.0  # a GUI form or a script changed a setting
    assert (pipe.compute(0)["Coordination"] == 0).all() and calls == [0, 1, 0]
    pipe.modifiers[0].enabled = False
    assert "Coordination" not in pipe.compute(0) and calls == [0, 1, 0]
    pipe.modifiers[0].enabled = True
    pipe.compute(0)
    assert calls == [0, 1, 0]  # that setting was computed before
    pipe.invalidate()
    pipe.compute(0)
    assert calls == [0, 1, 0, 0]


def test_output_cache_extends_a_cached_prefix_and_skips_scripts(cu_fcc):
    from ewald.pipeline import _modifier_key

    pipe = Pipeline(StaticSource([cu_fcc]), [mods.CoordinationAnalysis(3.0)])
    pipe.compute(0)
    first = pipe.modifiers[0]
    pipe.add(mods.ExpressionSelection("Coordination < 12"))
    t = first.last_time
    pipe.compute(0)
    assert first.last_time == t  # the new step started from the cached coordination
    assert _modifier_key(mods.PythonModifier(lambda f, d: None)) is None
    assert _modifier_key(mods.PythonScript("data.attributes['x'] = 1")) is None
    pipe.cache_outputs = False
    pipe.compute(0)
    assert first.last_time != t


def test_cached_frames_are_read_only(cu_fcc):
    """Writing into an array of a computed frame fails instead of changing the cached source frame;
    replacing the array works, and the user's own arrays stay writable."""
    original = cu_fcc.positions.copy()

    def nudge_in_place(frame, data):
        data.positions[0] += 1.0

    def nudge(frame, data):
        data.positions = data.positions + 1.0

    pipe = Pipeline(StaticSource(cu_fcc), [nudge_in_place])
    with pytest.raises(ValueError, match="read-only"):
        pipe.compute(0)
    pipe.modifiers[:] = [nudge]
    assert np.allclose(pipe.compute(0).positions, original + 1.0)
    assert np.allclose(pipe.compute(0).positions, original + 1.0)   # the cached frame is unchanged
    assert cu_fcc.positions.flags.writeable and np.allclose(cu_fcc.positions, original)


def test_missing_package_fails_once_not_every_frame(cu_fcc):
    from ewald import Modifier

    calls = []

    class NeedsPackage(Modifier):
        def __init__(self, level=1):
            self.level = level

        def modify(self, data, ctx):
            calls.append(ctx.frame)
            raise ImportError("NeedsPackage needs spglib")

    pipe = Pipeline(StaticSource([cu_fcc] * 4), [NeedsPackage()])
    for f in range(4):
        pipe.compute(f, errors="skip")
    assert calls == [0]  # later frames report the same error without trying again
    assert pipe.modifiers[0].status == "ImportError: NeedsPackage needs spglib"
    pipe.modifiers[0].level = 2  # new settings: one more try
    pipe.compute(1, errors="skip")
    pipe.invalidate()  # e.g. after installing the package
    pipe.compute(2, errors="skip")
    assert calls == [0, 1, 2]
    with pytest.raises(ImportError):  # scripts always get the real exception
        pipe.compute(3)
    assert calls == [0, 1, 2, 3]
