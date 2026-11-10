"""Pipeline defects from the API review (A1, A5, A6): results that were reused although a Python
step above computed from something that had changed, frame indices that wrapped or ran past the
end without a word, and mistakes in adding or setting up steps that surfaced late or never."""
import copy
import functools
import pickle
import warnings

import numpy as np
import pytest

import ewald
import ewald.examples
from ewald import AllFramesModifier, Pipeline, PythonModifier, StaticSource, all_frames
from ewald.modifiers import CoordinationAnalysis, MeanSquaredDisplacement, PythonScript, VelocityAutocorrelation
from ewald.pipeline import ModifierList

SCALE = 1.0          # read by the module-level steps below; tests change them with monkeypatch
OFFSET = 0.0
HANDLE = None


@pytest.fixture
def pipe():
    return ewald.load(ewald.examples.path("Cu_md.extxyz"))


def msd_end(pipe, frame=0):
    return float(pipe.compute(frame).tables["msd"]["MSD"][-1])


def fresh(*steps, frame=0):
    """The value a new pipeline gives: what a reused result must agree with."""
    return msd_end(Pipeline(ewald.load(ewald.examples.path("Cu_md.extxyz")).source, steps), frame)


def _count_msd_runs(monkeypatch) -> list:
    """One entry for every time MeanSquaredDisplacement really computes."""
    from ewald.modifiers import trajectory

    runs, real = [], trajectory.msd_curves
    monkeypatch.setattr(trajectory, "msd_curves", lambda *a, **k: runs.append(1) or real(*a, **k))
    return runs


def swell(frame, data):
    data.positions = data.positions * (1 + SCALE * 0.01 * frame)


def _amount():
    return OFFSET + 0.01


def swell_by_helper(frame, data):
    data.positions = data.positions * (1 + _amount() * frame)


def swell_with_handle(frame, data):
    data.positions = data.positions * (1 + HANDLE.scale * 0.01 * frame)


class Handle:
    scale = 1.0


# --- A1: all-frames results below a Python step ---------------------------------------------------

def test_msd_follows_a_global_the_step_above_reads(pipe, monkeypatch):
    pipe.add(swell, MeanSquaredDisplacement())
    first = msd_end(pipe)
    monkeypatch.setattr(f"{__name__}.SCALE", 5.0)
    second = msd_end(pipe)
    assert second != pytest.approx(first)                    # used to be the old 0.838
    assert second == pytest.approx(fresh(swell, MeanSquaredDisplacement()))
    assert msd_end(pipe, 3) == pytest.approx(second)         # and stays, at every frame


def test_msd_follows_a_closure_variable(pipe):
    scale = 1.0

    def swell(frame, data):
        data.positions = data.positions * (1 + scale * 0.01 * frame)

    pipe.add(swell, MeanSquaredDisplacement())
    first = msd_end(pipe)
    scale = 5.0
    assert msd_end(pipe) == pytest.approx(fresh(swell, MeanSquaredDisplacement()))
    assert msd_end(pipe) != pytest.approx(first)


def test_msd_follows_a_default_argument(pipe):
    def swell(frame, data, ctx=None, scale=1.0, *, power=1):
        data.positions = data.positions * (1 + scale * 0.01 * frame) ** power

    pipe.add(swell, MeanSquaredDisplacement())
    first = msd_end(pipe)
    swell.__defaults__ = (None, 5.0)
    second = msd_end(pipe)
    assert second != pytest.approx(first) and second == pytest.approx(fresh(swell, MeanSquaredDisplacement()))
    swell.__kwdefaults__ = {"power": 2}
    third = msd_end(pipe)
    assert third != pytest.approx(second) and third == pytest.approx(fresh(swell, MeanSquaredDisplacement()))


def test_msd_follows_a_keyword_parameter_of_the_step(pipe):
    def swell(frame, data, *, scale):
        data.positions = data.positions * (1 + scale * 0.01 * frame)

    step = PythonModifier(swell, scale=1.0)
    pipe.add(step, MeanSquaredDisplacement())
    first = msd_end(pipe)
    step.params["scale"] = 5.0
    assert msd_end(pipe) != pytest.approx(first)


def test_msd_follows_a_global_of_a_helper_the_step_calls(pipe, monkeypatch):
    pipe.add(swell_by_helper, MeanSquaredDisplacement())
    first = msd_end(pipe)
    monkeypatch.setattr(f"{__name__}.OFFSET", 0.04)
    second = msd_end(pipe)
    assert second != pytest.approx(first)
    assert second == pytest.approx(fresh(swell_by_helper, MeanSquaredDisplacement()))


def test_msd_follows_new_code_in_the_same_function(pipe):
    def swell(frame, data):
        data.positions = data.positions * (1 + 0.01 * frame)

    def swell_more(frame, data):
        data.positions = data.positions * (1 + 0.05 * frame)

    pipe.add(swell, MeanSquaredDisplacement())
    first = msd_end(pipe)
    swell.__code__ = swell_more.__code__                     # same function object, same repr
    assert msd_end(pipe) != pytest.approx(first)


def test_python_script_parameters_count(pipe):
    script = PythonScript("def modify(frame, data, ctx, scale=1.0):\n"
                          "    data.positions = data.positions * (1 + scale * 0.01 * frame)\n", scale=1.0)
    pipe.add(script, MeanSquaredDisplacement())
    first = msd_end(pipe)
    script.params["scale"] = 5.0
    assert msd_end(pipe) != pytest.approx(first)


def test_step_reading_an_object_follows_its_attributes(pipe, monkeypatch):
    """An object counts by its class and attributes: reused while they stay, recomputed after."""
    from ewald.pipeline import _Same, _token

    handle = Handle()
    monkeypatch.setattr(f"{__name__}.HANDLE", handle)
    assert type(_token(handle)) is not _Same and _token(handle) == _token(handle)
    runs = _count_msd_runs(monkeypatch)
    pipe.add(swell_with_handle, MeanSquaredDisplacement(by_type=False))
    first = msd_end(pipe)
    assert msd_end(pipe, 3) == pytest.approx(first) and len(runs) == 1
    handle.scale = 5.0
    changed = msd_end(pipe)
    assert len(runs) == 2
    assert changed == pytest.approx(fresh(swell_with_handle, MeanSquaredDisplacement())) and changed > 2 * first


REFERENCE = WINDOW = PATTERN = TABLE = None


def displaced(frame, data):
    shift = data.positions[WINDOW] - REFERENCE.positions[WINDOW]
    moved = np.linalg.norm(shift, axis=1) * (PATTERN.match("Cu") is not None) * TABLE.attributes["gain"]
    data.attributes["Moved"] = float(moved.max())


def test_ordinary_values_a_step_reads_do_not_cost_the_result_below(pipe, monkeypatch):
    """A reference frame, a slice, a compiled pattern, a table, a random generator, a jitted
    helper and the pipeline itself are things steps read all the time: the all-frames step
    below is computed once, not at every frame change."""
    import re

    from ewald.verbs import Table

    table = Table("levels", {"x": np.arange(3.0)}, attributes={"gain": 1.0})
    for name, value in dict(REFERENCE=pipe.compute(0), WINDOW=slice(0, 50), PATTERN=re.compile("Cu"),
                            TABLE=table).items():
        monkeypatch.setattr(f"{__name__}.{name}", value)
    rng, frames = np.random.default_rng(1), pipe.num_frames
    helper = functools.lru_cache(lambda k: k + 1)

    def noisy(frame, data):
        data.attributes["Noise"] = float(rng.normal()) + helper(frame) + pipe.num_frames + frames

    runs = _count_msd_runs(monkeypatch)
    pipe.add(displaced, noisy, MeanSquaredDisplacement(by_type=False))
    first = msd_end(pipe)
    for k in range(1, pipe.num_frames):
        assert msd_end(pipe, k) == first
    assert len(runs) == 1
    table.attributes["gain"] = 2.0          # state a dict subclass carries beside its items
    msd_end(pipe)
    assert len(runs) == 2
    monkeypatch.setattr(f"{__name__}.WINDOW", slice(0, 20))
    msd_end(pipe)
    assert len(runs) == 3


def test_step_that_changes_what_it_reads_settles(pipe, monkeypatch):
    """A step that appends to a list it also reads has a new fingerprint after every run. Seen
    twice, it counts by identity, so the step below is not computed at every frame for good."""
    seen = []

    def collect(frame, data):
        seen.append(frame)

    runs = _count_msd_runs(monkeypatch)
    pipe.add(collect, MeanSquaredDisplacement(by_type=False))
    values = [msd_end(pipe, k % pipe.num_frames) for k in range(12)]
    assert len(set(values)) == 1 and len(runs) <= 3
    before = len(runs)
    for k in range(5):
        msd_end(pipe, k)
    assert len(runs) == before


def test_large_arrays_in_a_token(monkeypatch):
    """A read-only array nobody can change counts by identity (not read again at every
    compute); one that can be written to counts by content."""
    from ewald import pipeline
    from ewald.pipeline import _token

    monkeypatch.setattr(pipeline, "_HASHED_BYTES", 64)
    digests = []
    real = pipeline._digest
    monkeypatch.setattr(pipeline, "_digest", lambda a: digests.append(1) or real(a))
    array = np.arange(1000.0)
    token = _token(array)
    array[5] = -1.0
    assert _token(array) != token and len(digests) == 2
    array.flags.writeable = False
    fixed = _token(array)
    assert fixed == _token(array) and len(digests) == 2
    assert _token(array[::2]) != fixed and len(digests) == 2            # a view of it: read-only too
    writable = np.arange(1000.0)
    view = writable[:500]
    view.flags.writeable = False                                        # its base can still change
    assert _token(view) == _token(view) and len(digests) == 4
    assert _token(list(range(5000))) == _token(list(range(5000))) != _token(list(range(5001)))


def test_token_of_objects():
    import collections
    import dataclasses
    import datetime
    import re
    import types

    from ewald.pipeline import _Same, _token

    @dataclasses.dataclass(frozen=True)
    class Setup:
        cutoff: float = 3.0

    class Slotted:
        __slots__ = ("gain",)

        def __init__(self, gain):
            self.gain = gain

    for make in (lambda: slice(0, 2), lambda: re.compile("Cu"), lambda: datetime.date(2026, 1, 1), lambda: ...,
                 lambda: np.dtype("f4"), lambda: Setup(), lambda: types.SimpleNamespace(a=1), lambda: Slotted(2),
                 lambda: collections.deque([1, 2]), lambda: np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])):
        assert _token(make()) == _token(make()) and type(_token(make())) is not _Same, make()
    assert _token(Setup(3.0)) != _token(Setup(3.5)) and _token(Slotted(1)) != _token(Slotted(2))
    assert _token(slice(0, 2)) != _token(slice(0, 3)) and _token(re.compile("a")) != _token(re.compile("a", re.I))
    masked = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
    before = _token(masked)
    masked.mask = [1, 0, 0]
    assert _token(masked) != before
    for unreadable in (np.random.default_rng(1), object(), Setup):      # by identity, and hashable
        assert _token(unreadable) == _token(unreadable) and type(_token(unreadable)) is _Same
        assert hash(_token(unreadable)) == hash(_token(unreadable))
    knot = types.SimpleNamespace()
    knot.me = knot                                                       # ends, and is stable
    assert _token(knot) == _token(knot)
    assert pickle.loads(pickle.dumps(_token(object()))) != copy.deepcopy(_token(object()))


def test_slow_fingerprint_is_told_once(pipe, monkeypatch):
    from ewald.pipeline import Pipeline as P

    monkeypatch.setattr(P, "_SLOW_TOKEN", 0.0)
    pipe.add(swell, MeanSquaredDisplacement(by_type=False))
    with pytest.warns(RuntimeWarning, match="step 'swell'.*read-only"):
        pipe.compute(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pipe.compute(1)


def test_unchanged_python_step_keeps_the_result_below(pipe, monkeypatch):
    runs = _count_msd_runs(monkeypatch)
    pipe.add(swell, MeanSquaredDisplacement(by_type=False))
    values = [msd_end(pipe, k) for k in (0, 1, 5, 0)]
    assert len(runs) == 1 and len(set(values)) == 1
    monkeypatch.setattr(f"{__name__}.SCALE", 1.0)             # set again to the same value: no change
    monkeypatch.setattr(f"{__name__}.OFFSET", 7.0)            # a global the step does not read
    msd_end(pipe, 2)
    assert len(runs) == 1
    monkeypatch.setattr(f"{__name__}.SCALE", 2.0)
    msd_end(pipe, 2)
    assert len(runs) == 2


def test_all_frames_step_reused_until_what_it_or_the_step_above_reads_changes(pipe):
    calls = []
    scale, gain = 1.0, 1.0

    def swell(frame, data):
        data.positions = data.positions * (1 + scale * 0.01 * frame)

    @all_frames
    def spread(frames, data):
        calls.append(len(frames))                             # the step's own bookkeeping is no change
        z = [gain * float(d.positions[:, 2].max()) for d in frames]
        data.tables["spread"] = {"Frame": np.arange(len(z)), "Max z": np.array(z)}

    pipe.add(swell, spread)
    top = [float(pipe.compute(k).tables["spread"]["Max z"][-1]) for k in range(4)]
    assert calls == [10] and len(set(top)) == 1
    scale = 3.0                                               # the step above
    wider = float(pipe.compute(0).tables["spread"]["Max z"][-1])
    assert calls == [10, 10] and wider > top[0]
    gain = 2.0                                                # the all-frames function itself
    assert float(pipe.compute(1).tables["spread"]["Max z"][-1]) == pytest.approx(2 * wider)
    pipe.compute(2)
    assert calls == [10, 10, 10]


def test_all_frames_step_below_another_runs_once_per_change(pipe):
    """A step that reads every frame computes the steps above it at each of them: the one above
    must come from its kept result there, also when its function reads an object."""
    calls = {"first": 0, "second": 0}
    handle = Handle()

    @all_frames
    def first(frames, data):
        calls["first"] += 1
        data.attributes["Top"] = handle.scale * max(float(d.positions.max()) for d in frames)

    @all_frames
    def second(frames, data):
        calls["second"] += 1
        data.attributes["Tops"] = [d.attributes["Top"] for d in frames]

    pipe.add(first, second)
    out = pipe.compute(0)
    assert calls == {"first": 1, "second": 1} and len(set(out.attributes["Tops"])) == 1
    handle.scale = 2.0
    again = pipe.compute(0)                                   # an object: recomputed, once each
    assert calls == {"first": 2, "second": 2}
    assert again.attributes["Top"] == pytest.approx(2 * out.attributes["Top"])


def test_trajectory_steps_still_share_one_pass_below_a_python_step(pipe, monkeypatch):
    """msd then vacf read the frames once; a changed step above makes both read them again."""
    from ewald.modifiers import trajectory

    passes = []
    real = trajectory.collect_trajectory
    monkeypatch.setattr(trajectory, "collect_trajectory", lambda *a, **k: passes.append(1) or real(*a, **k))
    pipe.add(swell, MeanSquaredDisplacement(), VelocityAutocorrelation(velocities="positions"))
    pipe.compute(0)
    pipe.compute(1)
    assert len(passes) == 1
    monkeypatch.setattr(f"{__name__}.SCALE", 4.0)
    expected = fresh(swell, MeanSquaredDisplacement())
    del passes[:]
    assert msd_end(pipe) == pytest.approx(expected)          # not from the frames stacked before the change
    assert len(passes) == 1


def _by_identity(token) -> bool:
    """Whether some part of ``token`` stands for an object that could not be read."""
    from ewald.pipeline import _Same

    return type(token) is _Same or isinstance(token, tuple) and any(_by_identity(part) for part in token)


def test_function_token():
    from ewald.pipeline import _Same, _function_token, _token

    def with_default(frame, data, cutoff=3.0): ...
    def other_default(frame, data, cutoff=3.5): ...
    def uses_numpy(frame, data): return np.linalg.norm(data.positions), np.sqrt(2.0), functools.reduce

    assert _function_token(swell) == _function_token(swell)
    assert _function_token(with_default) != _function_token(other_default)
    assert not _by_identity(_function_token(uses_numpy))       # modules and installed functions: by name
    assert _token(functools.partial(with_default, cutoff=1.0)) != _token(functools.partial(with_default, cutoff=2.0))
    assert _token(np.arange(3)) != _token(np.arange(3) + 1)
    assert not _by_identity(_token(open)) and type(_token(object())) is _Same

    def recursive(n):
        return recursive(n - 1) if n else SCALE
    assert _function_token(recursive) == _function_token(recursive) and not _by_identity(_function_token(recursive))

    def attribute_only(frame, data):
        return data.HANDLE                                    # an attribute, not the global HANDLE
    assert not _by_identity(_function_token(attribute_only))


def test_keys_survive_pickle_and_deepcopy(pipe):
    pipe.add(swell, MeanSquaredDisplacement())
    value = msd_end(pipe)
    for clone in (pickle.loads(pickle.dumps(pipe)), copy.deepcopy(pipe)):
        assert type(clone.modifiers) is ModifierList and clone.modifiers is not pipe.modifiers
        assert msd_end(clone) == pytest.approx(value)


# --- A5: frame indices -----------------------------------------------------------------------------

def test_source_frame_takes_the_indices_compute_takes(pipe):
    n = pipe.num_frames
    assert n == 10
    for bad in (n, 999, -n - 1):
        with pytest.raises(IndexError, match="out of range: the pipeline has 10 frames"):
            pipe.source_frame(bad)
    assert np.array_equal(pipe.source_frame(-1).positions, pipe.source_frame(n - 1).positions)
    assert not np.array_equal(pipe.source_frame(-1).positions, pipe.source_frame(0).positions)
    with pytest.raises(TypeError, match="frame must be an int"):
        pipe.source_frame(1.5)


def test_reference_frame_out_of_range_raises(pipe):
    from ewald.modifiers import CalculateDisplacements

    pipe.add(CalculateDisplacements(reference_frame=999))     # used to take frame 9 for it
    with pytest.raises(IndexError, match="frame 999 out of range"):
        pipe.compute(0)

    def to_last(frame, data, ctx):
        data["Step"] = ctx.source_frame(-1).positions - data.positions

    pipe.modifiers[:] = [to_last]
    assert np.abs(pipe.compute(0)["Step"]).max() > 0 and np.abs(pipe.compute(9)["Step"]).max() == 0


def test_frame_offset_stays_inside_the_trajectory(pipe):
    from ewald.modifiers import AtomicStrain, WignerSeitzAnalysis

    for step in (AtomicStrain(cutoff=3.2, frame_offset=-1), WignerSeitzAnalysis(frame_offset=-1)):
        pipe.modifiers[:] = [step]
        pipe.compute(1)
        with pytest.raises(IndexError, match="frame_offset=-1 at frame 0 points to frame -1"):
            pipe.compute(0)                                   # used to take the last frame
        step.frame_offset = 5
        with pytest.raises(IndexError, match="points to frame 13: the pipeline has frames 0 to 9"):
            pipe.compute(8)


def test_frame_ranges_of_the_trajectory_kernels_clip_as_frames_does(pipe):
    """``stop`` past the end read the first frames again and gave numbers from them."""
    from ewald import kernels

    whole = kernels.trajectory_msd(pipe)
    past = kernels.trajectory_msd(pipe, stop=999)
    assert len(past["MSD"]) == len(whole["MSD"]) and np.allclose(past["MSD"], whole["MSD"])
    assert len(kernels.trajectory_vacf(pipe, stop=999, velocities="positions")["VACF"]) == \
        len(kernels.trajectory_vacf(pipe, velocities="positions")["VACF"])
    assert np.allclose(kernels.trajectory_rdf(pipe, 3.0, stop=999)["g(r)"], kernels.trajectory_rdf(pipe, 3.0)["g(r)"])
    assert len(kernels.trajectory_msd(pipe, start=-5)["MSD"]) == 5
    pipe.add(MeanSquaredDisplacement(start=-5, by_type=False))
    assert len(pipe.compute(0).tables["msd"]["MSD"]) == 5      # used to be 15 rows of 10 frames
    with pytest.raises(ValueError, match="empty frame range"):
        kernels.trajectory_msd(pipe, start=20)


def test_frames_follow_slice_semantics(pipe):
    marker = []
    pipe.add(lambda frame, data: marker.append(frame))
    n = pipe.num_frames

    def which(*args, **kwargs):
        del marker[:]
        count = sum(1 for _ in pipe.frames(*args, **kwargs))
        assert count == len(marker)
        return list(marker)

    assert which() == list(range(n))
    assert which(-2) == [8, 9]                                # used to be 12 frames: range(-2, 10)
    assert which(2, 5) == [2, 3, 4]
    assert which(0, 999) == list(range(n))                    # clipped, as a slice is
    assert which(None, None, 3) == list(range(n))[::3]
    assert which(step=-1) == list(range(n))[::-1]
    assert which(1, -1, 4) == list(range(n))[1:-1:4]
    with pytest.raises(ValueError, match="step cannot be zero"):
        pipe.frames(step=0)
    with pytest.raises(TypeError, match="whole numbers or None, not 1.5"):
        pipe.frames(1.5)
    assert [d.count for d in pipe] == [108] * n


def test_pipeline_longer_than_its_source_still_computes(tmp_path, pipe):
    """LoadTrajectory puts ten frames on a one-frame structure: the frames past the source's own
    count are valid for the pipeline (and each starts from the source's frame)."""
    from ewald.modifiers import LoadTrajectory

    first = pipe.compute(0)
    structure = Pipeline(StaticSource(first))
    structure.add(LoadTrajectory(ewald.examples.path("Cu_md.extxyz")))
    assert structure.source.num_frames == 1 and structure.num_frames == 10
    assert np.allclose(structure.compute(7).positions, pipe.compute(7).positions)
    assert structure.source_frame(7).count == 108 and structure.source_frame(-1).count == 108
    assert len(list(structure.frames(-3))) == 3
    with pytest.raises(IndexError):
        structure.source_frame(10)
    structure.add(MeanSquaredDisplacement())
    assert msd_end(structure) == pytest.approx(msd_end(_with(pipe, MeanSquaredDisplacement())))


def _with(pipe, *steps):
    pipe.add(*steps)
    return pipe


# --- A6: late or missing errors --------------------------------------------------------------------

def test_adding_a_class_instead_of_a_step_raises_at_once(pipe):
    for add in (pipe.add, pipe.modifiers.append, lambda m: pipe.modifiers.insert(0, m),
                lambda m: pipe.modifiers.extend([m]), lambda m: Pipeline(pipe.source, [m])):
        with pytest.raises(TypeError, match=r"pipe\.add\(CoordinationAnalysis\(\.\.\.\)\)"):
            add(CoordinationAnalysis)
    with pytest.raises(TypeError, match="dict is a class, not a step"):
        pipe.add(dict)
    assert len(pipe.modifiers) == 0


def test_python_step_returning_something_else_says_so(pipe):
    def doubled(frame, data):
        return data.positions * 2

    pipe.add(doubled)
    with pytest.raises(TypeError, match="step 'doubled' returned ndarray: a step sets its result on data"):
        pipe.compute(0)
    out = pipe.compute(0, errors="skip")                      # the GUI's way: reported on the step, frame intact
    assert out.count == 108 and pipe.modifiers[0].status.startswith("TypeError: step 'doubled' returned ndarray")

    @all_frames
    def flag(frames, data):
        data.attributes["Seen"] = len(frames)
        return True

    late = ewald.load(ewald.examples.path("Cu_md.extxyz"))
    late.add(flag)
    for _ in range(2):                                        # not kept either: the second look fails the same way
        with pytest.raises(TypeError, match="step 'flag' returned bool"):
            late.compute(0)

    script = ewald.load(ewald.examples.path("Cu_md.extxyz"))
    script.add(PythonScript("def modify(frame, data):\n    return 3\n", title="three"))
    with pytest.raises(TypeError, match="step 'three' returned int"):
        script.compute(0)

    fine = ewald.load(ewald.examples.path("Cu_md.extxyz"))
    fine.add(lambda frame, data: data.copy(), lambda frame, data: None)
    assert fine.compute(0).count == 108


def test_every_way_into_the_modifier_list_wraps_functions(pipe):
    def mark(frame, data):
        data.attributes["Marks"] = data.attributes.get("Marks", 0) + 1

    @all_frames
    def count(frames, data):
        data.attributes["Frames"] = len(frames)

    pipe.modifiers += [mark]                                  # used to leave the bare function in the list
    pipe.modifiers += (m for m in [count])
    assert [type(m) for m in pipe.modifiers] == [PythonModifier, AllFramesModifier]
    assert type(pipe.modifiers) is ModifierList
    pipe.modifiers = [mark, mark]                             # used to replace the list by a plain one
    assert type(pipe.modifiers) is ModifierList and [type(m) for m in pipe.modifiers] == [PythonModifier] * 2
    assert pipe.compute(0).attributes["Marks"] == 2
    pipe.modifiers = (m for m in [mark, CoordinationAnalysis(3.0)])
    assert [type(m).__name__ for m in pipe.modifiers] == ["PythonModifier", "CoordinationAnalysis"]
    pipe.modifiers[1:] = [mark, mark]
    pipe.modifiers[0] = mark
    assert all(type(m) is PythonModifier for m in pipe.modifiers) and pipe.compute(0).attributes["Marks"] == 3
    assert [type(m) for m in ModifierList([mark, count])] == [PythonModifier, AllFramesModifier]

    kept = pipe.modifiers
    pipe.modifiers = kept                                     # the list itself stays the list
    assert pipe.modifiers is kept
    other = Pipeline(pipe.source)
    other.modifiers = []
    other.modifiers.append(mark)
    assert type(other.modifiers[0]) is PythonModifier
    for wrong in (mark, CoordinationAnalysis(3.0)):
        with pytest.raises(TypeError, match=r"pipe\.modifiers = \[step\]"):
            pipe.modifiers = wrong
    with pytest.raises(TypeError, match="not a Modifier or callable"):
        pipe.modifiers = [3]
    assert pipe.modifiers is kept


@pytest.mark.parametrize("wrong", [None, "abc", 3, CoordinationAnalysis(3.0), swell])
def test_one_step_where_a_list_belongs(pipe, wrong):
    with pytest.raises(TypeError, match=r"pipe\.modifiers = \[step\], or pipe\.add\(step\)"):
        pipe.modifiers = wrong
    with pytest.raises(TypeError, match=r"pipe\.modifiers \+= \[step\], or pipe\.add\(step\)"):
        pipe.modifiers += wrong
    assert pipe.modifiers == []


def test_misspelled_setting_raises(pipe):
    m = CoordinationAnalysis(cutoff=3.0)
    with pytest.raises(AttributeError, match="CoordinationAnalysis has no setting 'cutof'. Did you mean 'cutoff'"):
        m.cutof = 9
    with pytest.raises(AttributeError, match="Did you mean 'partial'"):
        setattr(m, "partail", True)
    with pytest.raises(AttributeError, match="Did you mean 'bins'"):
        m.number_of_bin = 50                                  # nearly the old name: the current one is offered
    assert m.cutoff == 3.0 and "cutof" not in vars(m) and "partail" not in vars(m)


def test_misspelled_setting_raises_in_a_helper_too():
    """A function that takes the step as its first argument is not a method of the step."""
    def tune(step, value):
        step.cutof = value

    m = CoordinationAnalysis(cutoff=3.0)
    with pytest.raises(AttributeError, match="Did you mean 'cutoff'"):
        tune(m, 5.0)
    with pytest.raises(AttributeError, match="Did you mean 'partial'"):
        (lambda step: setattr(step, "partail", True))(m)
    assert m.cutoff == 3.0 and "cutof" not in vars(m)
    tune.__name__ = "modify"                                  # a namesake of a method is not it either
    with pytest.raises(AttributeError):
        tune(m, 5.0)


def test_setting_guard_leaves_everything_else_alone():
    m = CoordinationAnalysis(cutoff=3.0)
    m.cutoff = 4.0
    m.number_of_bins = 40                                     # an old name still sets the new one
    assert (m.cutoff, m.bins) == (4.0, 40)
    m.enabled = False
    m.status = "note"
    m.label = "first shell"                                   # nothing like a setting: the caller's own attribute
    m._cutof = 1                                              # private names are never checked
    m.unrelated_value = 2
    m.label = "again"
    assert vars(m)["label"] == "again" and m._cutof == 1

    class Shells(ewald.Modifier):
        """A modifier's own methods name their attributes as they like, also close to a setting."""

        def __init__(self, cutoff: float = 3.0):
            self.cutoff = cutoff
            self.cutoffs = [cutoff, 2 * cutoff]

        def modify(self, data, ctx):
            self.cutof = self.cutoff
            data.attributes["Shells"] = len(self.cutoffs)

    s = Shells()
    assert s.cutoffs == [3.0, 6.0] and s.parameters() == {"cutoff": 3.0}
    s.cutoffs = [1.0]                                         # exists now: an ordinary attribute
    assert Pipeline(StaticSource(ewald.load(ewald.examples.path("Cu_md.extxyz")).compute(0)), [s]) \
        .compute(0).attributes["Shells"] == 1
    with pytest.raises(AttributeError, match="Did you mean 'cutoff'"):
        s.cuttoff = 2.0
    clone = copy.copy(s)
    assert clone.cutoffs == [1.0] and pickle.loads(pickle.dumps(m)).label == "again"
