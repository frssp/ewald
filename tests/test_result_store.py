"""Results kept across pipelines by content (ewald.result_store): a notebook's Run all makes a new
pipeline with new steps, and finds what the last run computed from the same files and settings."""
import ntpath
import os
import time

import numpy as np
import pytest

import ewald
from ewald.modifiers import MeanSquaredDisplacement
from ewald.notebook import Notebook
from ewald.pipeline import Modifier, _file_stamps, _portable, _Same, all_frames
from ewald.result_store import ResultStore, shared

CALLS: list = []


class Slow(Modifier):
    """A built-in-like step that counts its runs."""

    cacheable = True

    def __init__(self, scale: float = 1.0, path: str | None = None):
        self.scale = scale
        self.path = path

    def modify(self, data, ctx):
        CALLS.append((self.scale, ctx.frame))
        time.sleep(0.03)   # above Pipeline.checkpoint_seconds: kept as a checkpoint too
        data.particles[f"Scaled x{self.scale:g}"] = np.asarray(data.positions)[:, 0] * self.scale
        data.attributes["path"] = self.path


@pytest.fixture(autouse=True)
def _calls():
    CALLS.clear()
    yield
    CALLS.clear()


def _trajectory(path, frames=4, n=8):
    from ase import Atoms
    from ase.io import write

    rng = np.random.default_rng(0)
    images = [Atoms("Cu" * n, positions=rng.random((n, 3)) * 5 + 0.1 * k, cell=[6, 6, 6], pbc=True)
              for k in range(frames)]
    write(str(path), images, format="extxyz")
    return str(path)


def _bump(path):
    """Change a file's contents and modification time."""
    with open(path, "ab") as f:
        f.write(b"\n")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_a_new_pipeline_over_the_same_file_reuses_the_result(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    first = ewald.load(path)
    first.add(Slow(2.0))
    a = first.compute(1)
    assert CALLS == [(2.0, 1)]
    again = ewald.load(path)              # what Run all does: new pipeline, new step
    step = again.add(Slow(2.0))
    b = again.compute(1)
    assert CALLS == [(2.0, 1)]            # not run again
    np.testing.assert_array_equal(a["Scaled x2"], b["Scaled x2"])
    assert step.last_output is not None and step.last_output["properties"] == ["Scaled x2"]
    assert step.status is None
    again.compute(2)                      # another frame is computed as before
    assert CALLS[-1] == (2.0, 2)


def test_other_settings_or_a_changed_file_compute_again(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    ewald.load(path).add(Slow(2.0)) and None
    pipe = ewald.load(path)
    pipe.add(Slow(2.0))
    pipe.compute(0)
    other = ewald.load(path)
    other.add(Slow(3.0))
    other.compute(0)
    assert CALLS == [(2.0, 0), (3.0, 0)]
    _bump(path)
    changed = ewald.load(path)
    changed.add(Slow(2.0))
    changed.compute(0)
    assert CALLS[-1] == (2.0, 0) and len(CALLS) == 3


def test_a_file_a_setting_names_is_part_of_the_key(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    ref = tmp_path / "ref.txt"
    ref.write_text("1\n", encoding="utf-8", newline="\n")
    for _ in range(2):
        pipe = ewald.load(path)
        pipe.add(Slow(1.0, path=str(ref)))
        pipe.compute(0)
    assert len(CALLS) == 1
    _bump(ref)
    pipe = ewald.load(path)
    pipe.add(Slow(1.0, path=str(ref)))
    pipe.compute(0)
    assert len(CALLS) == 2


def test_only_the_steps_after_an_edit_run_again(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    pipe = ewald.load(path)
    pipe.add(Slow(2.0), Slow(5.0))
    pipe.compute(0)
    edited = ewald.load(path)
    edited.add(Slow(2.0), Slow(7.0))     # the second cell changed
    edited.compute(0)
    assert CALLS == [(2.0, 0), (5.0, 0), (7.0, 0)]


def test_all_frames_results_survive_a_new_pipeline(tmp_path, monkeypatch):
    path = _trajectory(tmp_path / "t.extxyz", frames=6)
    runs = []
    real = MeanSquaredDisplacement._compute
    monkeypatch.setattr(MeanSquaredDisplacement, "_compute", lambda self, ctx: runs.append(1) or real(self, ctx))
    tables = []
    for frame in (0, 3):   # another frame: the frame's own result is new, the trajectory's is not
        pipe = ewald.load(path)
        pipe.add(MeanSquaredDisplacement())
        tables.append(pipe.compute(frame).tables["msd"])
    assert len(runs) == 1
    np.testing.assert_allclose(tables[0]["MSD"], tables[1]["MSD"])


def test_python_all_frames_step_is_shared_by_what_its_function_reads(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz", frames=5)
    seen = []

    def make(offset):
        @all_frames
        def mean_x(frames, data):
            seen.append(offset)
            data.attributes["mean x"] = float(np.mean([f.positions[:, 0].mean() for f in frames])) + offset
        return mean_x

    for offset in (0.0, 0.0, 1.0):
        pipe = ewald.load(path)
        pipe.add(make(offset))
        pipe.compute(0)
    assert seen == [0.0, 1.0]


def test_a_step_reading_a_random_generator_is_not_shared(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    rng = np.random.default_rng(1)
    seen = []

    @all_frames
    def noise(frames, data):
        seen.append(1)
        data.attributes["noise"] = float(rng.random())

    for _ in range(2):
        pipe = ewald.load(path)
        pipe.add(noise)
        pipe.compute(0)
    # the generator counts by identity (its state cannot be read): a second pipeline must not
    # be handed the first number as if nothing changed
    assert len(seen) == 2 and not _portable(_Same(rng))


def test_turned_off_and_cleared(tmp_path, monkeypatch):
    path = _trajectory(tmp_path / "t.extxyz")
    monkeypatch.setenv("EWALD_SHARED_RESULTS", "0")
    for _ in range(2):
        pipe = ewald.load(path)
        pipe.add(Slow(2.0))
        pipe.compute(0)
    assert len(CALLS) == 2 and len(shared) == 0
    monkeypatch.setenv("EWALD_SHARED_RESULTS", "1")
    pipe.compute(1)
    assert len(shared) > 0
    pipe.clear_caches()                   # View › Redraw from scratch: for every pipeline of the file
    assert len(shared) == 0
    pipe.share_results = False
    ewald.load(path).add(Slow(2.0)) and None
    pipe.compute(1)
    assert len(shared) == 0


def test_store_limits():
    store = ResultStore(max_bytes=1000, max_entries=3)
    big = np.zeros(200)                   # 1600 bytes: more than the whole store
    store.put("big", big)
    assert store.get("big") is None
    for k in range(5):
        store.put(k, np.zeros(10))
    assert len(store) == 3 and store.get(0) is None and store.get(4) is not None
    store.put("x", np.zeros(100))         # 800 bytes: the oldest go
    assert store.nbytes() <= 1000 and store.get("x") is not None
    store.drop(None)
    assert len(store) == 0


def test_relative_and_absolute_paths_share(tmp_path, monkeypatch):
    path = _trajectory(tmp_path / "t.extxyz")
    monkeypatch.chdir(tmp_path)
    a = ewald.load("t.extxyz")
    b = ewald.load(os.path.abspath(path))
    assert a.source.content_key() == b.source.content_key() is not None


def test_file_stamps_of_other_platforms_and_remote_paths(tmp_path):
    step = Slow(1.0, path=ntpath.join("C:\\", "data", "missing.txt"))   # a Windows path seen on Linux
    assert _file_stamps(step) == ()       # not a file here: nothing to stamp, still shareable
    step.path = "cluster:/scratch/run/ref.txt"
    assert _file_stamps(step) is None     # remote: results stay in their pipeline
    step.path = str(tmp_path / "*.txt")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8", newline="\n")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8", newline="\n")
    stamps = _file_stamps(step)
    assert [os.path.basename(s[0]) for s in stamps] == ["a.txt", "b.txt"]
    assert all(s[0] == os.path.normcase(s[0]) for s in stamps)


def test_notebook_run_all_reuses_unchanged_cells(tmp_path):
    path = _trajectory(tmp_path / "t.extxyz")
    nb_path = tmp_path / "nb.py"
    nb_path.write_text(
        "# %%\nimport ewald\nfrom test_result_store import Slow\n"
        f"pipe = ewald.load({path!r})\n# %%\npipe.add(Slow(2.0))\n# %%\npipe.add(Slow(5.0))\n"
        "# %%\ndata = pipe.compute(0)\n", encoding="utf-8", newline="\n")
    nb = Notebook.load(str(nb_path))
    nb.trust()
    assert nb.run_all()
    assert CALLS == [(2.0, 0), (5.0, 0)]
    assert nb.run_all()
    assert CALLS == [(2.0, 0), (5.0, 0)]  # nothing ran again
    nb.cells[2].source = nb.cells[2].source.replace("5.0", "6.0")
    assert nb.run_all()
    assert CALLS[-1] == (6.0, 0) and len(CALLS) == 3
    assert [m.last_output["properties"] for m in nb.pipeline.modifiers] == [["Scaled x2"], ["Scaled x6"]]


def test_remote_files_are_shared_by_size_and_mtime(server):
    for _ in range(2):
        pipe = ewald.load("hpc:/data/dump.custom.lammpstrj")
        pipe.add(Slow(2.0))
        pipe.compute(0)
    assert len(CALLS) == 1
    assert pipe.source.content_key()[1][0][0] == "hpc:/data/dump.custom.lammpstrj"
