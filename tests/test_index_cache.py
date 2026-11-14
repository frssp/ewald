"""Frame indexes of big trajectories kept on disk: a second open of an unchanged file skips the scan."""
import os

import numpy as np
import pytest

from ewald.io import extxyz, index_cache, lammps_dump, xdatcar

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EWALD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EWALD_INDEX_CACHE", "1")
    monkeypatch.setattr(index_cache, "MIN_BYTES", 0)   # the test files are small
    return tmp_path / "cache" / "index"


def _copy(src, dst, crlf=False):
    with open(src, "rb") as f:
        raw = f.read()
    if crlf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    with open(dst, "wb") as f:
        f.write(raw)
    return str(dst)


def _no_scan(*a, **k):
    raise AssertionError("scanned although the index was kept")


@pytest.mark.parametrize("crlf", [False, True])
def test_lammps_dump_index_is_kept(cache, tmp_path, monkeypatch, crlf):
    path = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "dump.lammpstrj", crlf)
    first = lammps_dump.LammpsDumpReader(path)
    assert len(list(cache.glob("lammps-dump-*.npz"))) == 1
    with monkeypatch.context() as m:
        m.setattr(lammps_dump, "_scan_frames", _no_scan)
        again = lammps_dump.LammpsDumpReader(path)
    assert again._offsets == first._offsets and again._timesteps == first._timesteps and len(again) == len(first)
    np.testing.assert_allclose(again.read(len(again) - 1).positions, first.read(len(first) - 1).positions)
    assert [again.timestep(i) for i in range(len(again))] == [first.timestep(i) for i in range(len(first))]


def test_a_changed_file_is_scanned_again(cache, tmp_path, monkeypatch):
    path = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "dump.lammpstrj")
    n = len(lammps_dump.LammpsDumpReader(path))
    with open(path, "rb") as f:
        raw = f.read()
    with open(path, "ab") as f:   # a run appended a frame
        f.write(raw[raw.index(b"ITEM: TIMESTEP"):])
    calls = []
    real = lammps_dump._scan_frames
    monkeypatch.setattr(lammps_dump, "_scan_frames", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert len(lammps_dump.LammpsDumpReader(path)) == 2 * n and calls
    calls.clear()
    assert len(lammps_dump.LammpsDumpReader(path)) == 2 * n and not calls   # the new index is kept


@pytest.mark.parametrize("reader, name, kind, scan", [
    (xdatcar.XdatcarReader, os.path.join("vasp", "XDATCAR"), "xdatcar", None),
    (xdatcar.XdatcarReader, os.path.join("vasp", "XDATCAR.npt"), "xdatcar", None),
    (extxyz.ExtxyzReader, "interface_gaas_alas_110.extxyz", "extxyz", "_scan_frames")])
def test_xdatcar_and_extxyz_indexes_are_kept(cache, tmp_path, monkeypatch, reader, name, kind, scan):
    path = _copy(os.path.join(DATA, name), tmp_path / os.path.basename(name))
    first = reader(path)
    assert len(list(cache.glob(f"{kind}-*.npz"))) == 1
    with monkeypatch.context() as m:
        if scan:
            m.setattr(extxyz, scan, _no_scan)
        else:
            m.setattr(xdatcar.XdatcarReader, "_scan", _no_scan)
        again = reader(path)
    assert again._frames == first._frames
    np.testing.assert_allclose(again.read(len(again) - 1).positions, first.read(len(first) - 1).positions)


def test_off_small_remote_and_broken_entries(cache, tmp_path, monkeypatch):
    path = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "dump.lammpstrj")
    monkeypatch.setenv("EWALD_INDEX_CACHE", "0")
    lammps_dump.LammpsDumpReader(path)
    assert not cache.exists()
    monkeypatch.setenv("EWALD_INDEX_CACHE", "1")
    monkeypatch.setattr(index_cache, "MIN_BYTES", 1 << 40)   # small files scan in milliseconds
    lammps_dump.LammpsDumpReader(path)
    assert not cache.exists()
    monkeypatch.setattr(index_cache, "MIN_BYTES", 0)
    monkeypatch.setenv("EWALD_INDEX_CACHE", "0")
    assert index_cache.store(path, "lammps-dump", offsets=[0]) is False
    monkeypatch.setenv("EWALD_INDEX_CACHE", "1")
    monkeypatch.setattr(index_cache, "MIN_BYTES", 1 << 40)
    monkeypatch.setattr(index_cache, "MIN_BYTES", 0)
    lammps_dump.LammpsDumpReader(path)
    (entry,) = cache.glob("*.npz")
    entry.write_bytes(b"not a zip")          # another process half wrote it, or an older format
    assert len(lammps_dump.LammpsDumpReader(path)) >= 1
    assert index_cache.lookup(path, "lammps-dump") is not None   # the scan wrote it again
    index_cache.clear()
    assert not cache.exists()


def test_entries_differ_by_path_and_reader(cache, tmp_path):
    a = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "a.lammpstrj")
    b = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "b.lammpstrj")
    assert index_cache._entry(a, "lammps-dump") != index_cache._entry(b, "lammps-dump")
    assert index_cache._entry(a, "lammps-dump") != index_cache._entry(a, "extxyz")
    # names differing only in case are one file on Windows and macOS: normcase keys them alike there
    assert index_cache._entry(a, "x").name.startswith("x-")


def _append_frames(path, src):
    """Append every frame of ``src`` to ``path`` (a run went on writing), with a newer mtime."""
    with open(src, "rb") as f:
        raw = f.read()
    with open(path, "ab") as f:
        f.write(raw)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def _counting(module, name, monkeypatch):
    calls = []
    real = getattr(module, name)

    def wrapped(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return calls


@pytest.mark.parametrize("crlf", [False, True])
def test_a_grown_dump_is_scanned_from_its_old_last_frame(cache, tmp_path, monkeypatch, crlf):
    src = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "src.lammpstrj", crlf)
    path = _copy(src, tmp_path / "dump.lammpstrj")
    first = lammps_dump.LammpsDumpReader(path)
    n, last = len(first), first._offsets[-1]
    _append_frames(path, src)
    calls = _counting(lammps_dump, "_scan_frames", monkeypatch)
    grown = lammps_dump.LammpsDumpReader(path)
    assert len(grown) == 2 * n
    assert [c[0][1] if len(c[0]) > 1 else c[1].get("start", 0) for c in calls] == [last]   # not from byte 0
    monkeypatch.undo()
    steps = {}
    offsets, _ = lammps_dump._scan_frames(path, steps=steps)
    assert grown._offsets == offsets and grown._timesteps == steps


@pytest.mark.parametrize("reader, name, kind", [
    (xdatcar.XdatcarReader, os.path.join("vasp", "XDATCAR"), "xdatcar"),
    (extxyz.ExtxyzReader, "interface_gaas_alas_110.extxyz", "extxyz")])
def test_grown_xdatcar_and_extxyz_resume(cache, tmp_path, monkeypatch, reader, name, kind):
    src = os.path.join(DATA, name)
    path = _copy(src, tmp_path / os.path.basename(name))
    first = reader(path)
    if kind == "xdatcar":   # appended configurations, no new header (VASP's fixed-cell MD)
        with open(src, "rb") as f:
            raw = f.read()
        body = raw[first._frames[0][1]:first._frames[1][1]] if len(first) > 1 else raw[first._frames[0][1]:]
        with open(path, "ab") as f:
            f.write(body)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        scans = _counting(xdatcar.XdatcarReader, "_scan", monkeypatch)
    else:
        _append_frames(path, src)
        scans = _counting(extxyz, "_scan_frames", monkeypatch)
    grown = reader(path)
    assert len(grown) > len(first) and len(scans) == 1
    monkeypatch.undo()
    monkeypatch.setenv("EWALD_INDEX_CACHE", "0")
    fresh = reader(path)
    assert grown._frames == fresh._frames
    np.testing.assert_allclose(grown.read(len(grown) - 1).positions, fresh.read(len(fresh) - 1).positions)


def test_a_rewritten_file_of_the_same_start_is_not_resumed(cache, tmp_path, monkeypatch):
    src = _copy(os.path.join(DATA, "lammps", "dump.custom.lammpstrj"), tmp_path / "src.lammpstrj")
    path = _copy(src, tmp_path / "dump.lammpstrj")
    lammps_dump.LammpsDumpReader(path)
    raw = open(path, "rb").read()
    changed = raw[:-200] + raw[-200:].replace(b"1", b"2") + raw   # the old part's end changed too
    with open(path, "wb") as f:
        f.write(changed)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert index_cache.lookup_prefix(path, "lammps-dump") is None
    calls = _counting(lammps_dump, "_scan_frames", monkeypatch)
    lammps_dump.LammpsDumpReader(path)
    assert [c[0][1] if len(c[0]) > 1 else c[1].get("start", 0) for c in calls] == [0]


def test_remote_dump_index_is_kept(cache, server, monkeypatch):
    monkeypatch.setenv("EWALD_CACHE_DIR", str(cache.parent))
    from ewald.io import remote

    path = "hpc:/data/dump.custom.lammpstrj"
    first = lammps_dump.LammpsDumpReader(path)
    assert len(list(cache.glob("lammps-dump-*.npz"))) == 1
    calls = _counting(remote, "line_offsets", monkeypatch)   # the grep on the host
    again = lammps_dump.LammpsDumpReader(path)
    assert not calls and again._offsets == first._offsets
    # another user or port is another file
    assert index_cache._entry(path, "x") != index_cache._entry("bob@hpc:/data/dump.custom.lammpstrj", "x")
    assert index_cache._entry(path, "x") != index_cache._entry("ssh://hpc:2222/data/dump.custom.lammpstrj", "x")


def test_windows_paths_name_entries_without_a_host(cache):
    # 'C:\\...' is a drive, never an SSH host: it is named as a local path
    name = index_cache._name("C:\\runs\\dump.lammpstrj")
    assert not name.startswith("remote")
