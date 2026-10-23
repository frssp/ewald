"""Several files as one trajectory: natural order, file lists, restart segments joined on Timestep."""
import os
import shutil
import warnings

import numpy as np
import pytest

from ewald import import_file, load
from ewald.data import Cell, DataCollection
from ewald.io import FileSource, write_lammps_dump


def frame(step, n=4):
    rng = np.random.default_rng(step)
    data = DataCollection(rng.random((n, 3)) * 5, cell=Cell(np.eye(3) * 5))
    data.particles["Particle Identifier"] = np.arange(1, n + 1)
    data.particles["Particle Type"] = np.array(["1"] * n)
    data.attributes["Timestep"] = step
    return data


def dump(path, steps):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    write_lammps_dump(path, [frame(s) for s in steps])
    return str(path)


def steps(pipe):
    return [pipe.compute(i).attributes["Timestep"] for i in range(pipe.num_frames)]


def log(path, rows):
    with open(path, "w") as fh:
        fh.write("LAMMPS (test)\n   Step          Temp\n")
        fh.writelines(f"{s:10d}   {t}\n" for s, t in rows)
        fh.write("Loop time of 0.1 on 1 procs for 1 steps with 4 atoms\n")


def test_wildcard_in_natural_order(tmp_path):
    for s in (0, 200, 1000, 30):
        dump(tmp_path / f"dump.{s}", [s])
    pipe = import_file(tmp_path / "dump.*")
    assert steps(pipe) == [0, 30, 200, 1000]  # sorted(glob) gave 0, 1000, 200, 30


def test_restart_segments_later_wins(tmp_path):
    a = dump(tmp_path / "run1" / "dump.lammpstrj", range(0, 10001, 1000))
    b = dump(tmp_path / "run2" / "dump.lammpstrj", range(8000, 15001, 1000))
    pipe = load([a, b])
    assert steps(pipe) == list(range(0, 15001, 1000))
    assert pipe.source.segments == [(a, 8, 11), (b, 8, 8)]
    assert steps(load([a, b], join="concat"))[10:13] == [10000, 8000, 9000]
    # a wildcard over the run folders does the same, run10 after run2
    c = dump(tmp_path / "run10" / "dump.lammpstrj", range(14000, 16001, 1000))
    pipe = load(str(tmp_path / "run*" / "dump.lammpstrj"))
    assert pipe.source.paths == [a, b, c]
    assert steps(pipe) == list(range(0, 16001, 1000))


def test_list_keeps_given_order_and_reports_missing(tmp_path):
    a = dump(tmp_path / "a.dump", [500, 600])
    b = dump(tmp_path / "b.dump", [0, 100])
    assert steps(load([a, b], join="concat")) == [500, 600, 0, 100]
    with pytest.warns(UserWarning, match="left out"):  # b restarts before a: b wins from step 0 on
        assert steps(load([a, b])) == [0, 100]
    with pytest.raises(FileNotFoundError):
        load([a, str(tmp_path / "missing.dump")])
    with pytest.raises(ValueError, match="join"):
        load([a, b], join="sorted")


def test_fully_overlapped_file_is_dropped_with_a_warning(tmp_path):
    a = dump(tmp_path / "1.dump", [100, 200, 300])
    b = dump(tmp_path / "2.dump", [0, 100, 200, 300, 400])
    with pytest.warns(UserWarning, match="left out"):
        pipe = load([a, b])
    assert steps(pipe) == [0, 100, 200, 300, 400]
    assert pipe.source.segments[0] == (a, 0, 3)


def test_files_all_starting_at_one_step_are_appended(tmp_path):
    # independent runs (or NEB replicas): steps tell nothing about their order
    paths = [dump(tmp_path / f"r{i}.dump", [0, 10]) for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load(paths).num_frames == 6


def test_vasp_runs_are_appended(vasp_dir, tmp_path):
    for name in ("run1", "run2"):
        (tmp_path / name).mkdir()
        shutil.copy(vasp_dir / "XDATCAR", tmp_path / name / "XDATCAR")
    one = load(vasp_dir / "XDATCAR").num_frames
    assert load([str(tmp_path / "run1" / "XDATCAR"), str(tmp_path / "run2" / "XDATCAR")]).num_frames == 2 * one
    assert load(str(tmp_path / "run*" / "XDATCAR")).num_frames == 2 * one


def test_thermo_of_every_segment(tmp_path):
    a = dump(tmp_path / "run1" / "dump.lammpstrj", [0, 10, 20])
    b = dump(tmp_path / "run2" / "dump.lammpstrj", [10, 20, 30])
    log(tmp_path / "run1" / "log.lammps", [(0, 1.0), (10, 2.0), (20, 3.0)])
    log(tmp_path / "run2" / "log.lammps", [(10, 20.0), (20, 30.0), (30, 40.0)])
    pipe = load([a, b])
    assert [pipe.compute(i).attributes["Temp"] for i in range(pipe.num_frames)] == [1.0, 20.0, 30.0, 40.0]


def test_refresh_follows_a_restart_appearing(tmp_path):
    a = dump(tmp_path / "run1" / "dump.lammpstrj", range(0, 1001, 100))
    pipe = load(str(tmp_path / "run*" / "dump.lammpstrj"))
    assert pipe.num_frames == 11 and pipe.source.refresh() is None
    b = dump(tmp_path / "run2" / "dump.lammpstrj", [800, 900])
    assert pipe.source.refresh() == 8  # run1 from step 800 on is replaced by run2
    assert steps(pipe) == list(range(0, 1000, 100))
    with open(b, "a") as fh:  # run2 keeps writing
        write_lammps_dump(tmp_path / "extra", [frame(1000), frame(1100)])
        fh.write(open(tmp_path / "extra").read())
    assert pipe.source.refresh() == 9
    assert steps(pipe)[-3:] == [900, 1000, 1100] and pipe.source.paths == [a, b]


def test_session_and_notebook_code_keep_the_list(tmp_path):
    from ewald.notebook import load_code
    from ewald.session import source_from_dict, source_to_dict

    a = dump(tmp_path / "run1" / "dump.lammpstrj", [0, 10, 20])
    b = dump(tmp_path / "run2" / "dump.lammpstrj", [10, 20, 30])
    pipe = load([a, b], join="concat")
    entry = source_to_dict(pipe.source, str(tmp_path))
    assert entry["relative_path"] == [os.path.join("run1", "dump.lammpstrj"), os.path.join("run2", "dump.lammpstrj")]
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "run1", moved / "run1")
    shutil.copytree(tmp_path / "run2", moved / "run2")
    shutil.rmtree(tmp_path / "run1")
    src = source_from_dict(entry, str(moved))
    assert isinstance(src, FileSource) and src.join == "concat" and src.num_frames == 6
    assert load_code(pipe) == f"pipe = load({[a, b]!r}, join='concat')"


def test_remote_list_and_restart(server):
    data = server.local("/data")
    dump(os.path.join(data, "run1", "dump.lammpstrj"), [0, 10, 20, 30])
    dump(os.path.join(data, "run2", "dump.lammpstrj"), [20, 30, 40])
    dump(os.path.join(data, "run10", "dump.lammpstrj"), [40, 50])
    pipe = load(["hpc:/data/run1/dump.lammpstrj", "hpc:/data/run2/dump.lammpstrj"])
    assert steps(pipe) == [0, 10, 20, 30, 40]
    pipe = load("hpc:/data/run*/dump.lammpstrj")
    assert pipe.source.paths[-1] == "hpc:/data/run10/dump.lammpstrj"
    assert steps(pipe) == [0, 10, 20, 30, 40, 50]


def appended(path, *runs):
    """One dump that several runs wrote one after the other (``dump_modify append yes``)."""
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    for k, run in enumerate(runs):
        write_lammps_dump(str(path) + ".part", [frame(s) for s in run])
        with open(path, "ab" if k else "wb") as out, open(str(path) + ".part", "rb") as part:
            out.write(part.read())
    os.remove(str(path) + ".part")
    return str(path)


def test_restart_inside_one_appended_dump(tmp_path):
    path = appended(tmp_path / "dump.lammpstrj", range(0, 1201, 100), range(800, 2001, 100))
    pipe = load(path)
    assert steps(pipe) == list(range(0, 2001, 100))
    assert pipe.source.restarts == [(path, 13, 800)]
    assert pipe.source.segments == [(path, 21, 26)]
    concat = load(path, join="concat")
    assert concat.num_frames == 26 and concat.source.restarts == []
    # restarting at the last step written repeats that frame: the later one wins
    path = appended(tmp_path / "same.lammpstrj", [0, 100, 200], [200, 300])
    pipe = load(path)
    assert steps(pipe) == [0, 100, 200, 300] and pipe.compute(2).attributes["SourceFrame"] == 2
    assert np.allclose(pipe.compute(2).particles["Position"], load(path, join="concat").compute(3).particles["Position"])


def test_restarts_inside_files_and_across_them(tmp_path):
    a = appended(tmp_path / "run1" / "dump.lammpstrj", [0, 10, 20, 30], [20, 30, 40, 50])
    b = dump(tmp_path / "run2" / "dump.lammpstrj", [40, 50, 60])
    assert steps(load([a, b])) == [0, 10, 20, 30, 40, 50, 60]
    c = appended(tmp_path / "run3" / "dump.lammpstrj", [55, 65], [60, 70])   # run3 restarted once more
    assert steps(load([a, b, c])) == [0, 10, 20, 30, 40, 50, 55, 60, 70]


def test_reset_timestep_inside_one_dump_keeps_both_counts(tmp_path):
    # minimize, reset_timestep 0, then MD that was restarted from step 300 into the same dump
    path = appended(tmp_path / "dump.lammpstrj", [0, 5, 10], [0, 100, 200, 300, 400], [300, 400, 500])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = load(path)
    assert steps(pipe) == [0, 5, 10, 0, 100, 200, 300, 400, 500]
    assert pipe.source.restarts == [(path, 8, 300)]
    # a file of independent snapshots, all at step 0, stays whole
    path = appended(tmp_path / "snapshots.lammpstrj", [0], [0], [0])
    assert load(path).num_frames == 3 and load(path).source.restarts == []


def test_restart_steps_come_from_the_index_scan(tmp_path, monkeypatch):
    from ewald.io.lammps_dump import LammpsDumpReader

    path = appended(tmp_path / "dump.lammpstrj", range(0, 500, 10), range(300, 800, 10))
    reader = LammpsDumpReader(path)
    monkeypatch.setattr("ewald.io.lammps_dump.open_binary", lambda *a: pytest.fail("read a frame for its step"))
    assert reader.run_starts() == [50] and reader.timestep(50) == 300


def test_restart_in_a_crlf_and_a_gzipped_dump(tmp_path):
    import gzip

    path = appended(tmp_path / "dump.lammpstrj", [0, 10, 20, 30], [20, 30, 40])
    raw = open(path, "rb").read()
    crlf = tmp_path / "crlf.lammpstrj"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    gz = tmp_path / "dump.lammpstrj.gz"
    gz.write_bytes(gzip.compress(raw))
    for p in (crlf, gz):
        pipe = load(str(p))
        assert steps(pipe) == [0, 10, 20, 30, 40], p
        assert [r[1:] for r in pipe.source.restarts] == [(4, 20)]


def test_refresh_sees_a_restart_appended_while_running(tmp_path):
    path = appended(tmp_path / "dump.lammpstrj", [0, 10, 20, 30])
    pipe = load(path)
    assert pipe.num_frames == 4
    appended(tmp_path / "more", [20, 30, 40])
    with open(path, "ab") as fh:
        fh.write(open(tmp_path / "more", "rb").read())
    assert pipe.source.refresh() == 2
    assert steps(pipe) == [0, 10, 20, 30, 40] and pipe.source.restarts[0][1:] == (4, 20)


def test_remote_restart_inside_one_dump(server):
    data = server.local("/data")
    os.makedirs(data, exist_ok=True)
    appended(os.path.join(data, "dump.lammpstrj"), [0, 10, 20, 30], [20, 30, 40])
    raw = open(os.path.join(data, "dump.lammpstrj"), "rb").read()
    with open(os.path.join(data, "crlf.lammpstrj"), "wb") as fh:
        fh.write(raw.replace(b"\n", b"\r\n"))
    for name in ("dump.lammpstrj", "crlf.lammpstrj"):
        pipe = load(f"hpc:/data/{name}")
        assert steps(pipe) == [0, 10, 20, 30, 40], name
        assert pipe.source.restarts == [(f"hpc:/data/{name}", 4, 20)]


def test_join_modes_for_steps_going_back_inside_a_file(tmp_path):
    # a run restarted from step 0 into the same dump looks like reset_timestep: kept by default
    path = appended(tmp_path / "dump.lammpstrj", [0, 10, 20], [0, 10, 20, 30])
    assert steps(load(path)) == [0, 10, 20, 0, 10, 20, 30]
    with pytest.warns(UserWarning, match="left out"):   # every step going back is a restart
        pipe = load(path, join="restart")
    assert steps(pipe) == [0, 10, 20, 30] and pipe.source.restarts == [(path, 3, 0)]
    restarted = appended(tmp_path / "restarted.lammpstrj", [0, 10, 20, 30], [20, 30, 40])
    b = dump(tmp_path / "next.lammpstrj", [40, 50])
    pipe = load([restarted, b], join="files")   # only between files: the inside stays whole
    assert steps(pipe) == [0, 10, 20, 30, 20, 30, 40, 50] and pipe.source.restarts == []
    assert steps(load([restarted, b])) == [0, 10, 20, 30, 40, 50]
