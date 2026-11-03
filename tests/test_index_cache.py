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
    assert index_cache.store("host:/scratch/dump.lammpstrj", "lammps-dump", offsets=[0]) is False
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
