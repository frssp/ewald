"""Live reload: following trajectory files that a running simulation is still writing."""
import shutil
import warnings

import numpy as np
import pytest
from ase.build import bulk

from ewald import import_file
from ewald.io import write_extxyz
from ewald.data import DataCollection


def _cuts(blob: bytes, step: int):
    """Byte positions to cut a file at: every ``step`` bytes (mid-line and mid-frame), every
    line end, and the end."""
    ends = {i + 1 for i, c in enumerate(blob) if c == 10}
    return sorted(set(range(step, len(blob), step)) | ends | {len(blob)})


def _fresh(path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return import_file(path, **kwargs)
        except ValueError:  # not even one complete frame yet
            return None


def _grow_and_compare(path, blob, step, first_cut=None, **kwargs):
    """Write ``blob`` to ``path`` piece by piece; after each piece the refreshed pipeline must
    match a pipeline opened from scratch on the same bytes. Returns the final pipeline."""
    cuts = _cuts(blob, step)
    start = first_cut or next(c for c in cuts if _write(path, blob[:c]) and _fresh(path, **kwargs))
    path.write_bytes(blob[:start])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = import_file(path, **kwargs)
    pipe.compute(-1)  # cache the last frame: refresh() must drop it if it can change
    for cut in [c for c in cuts if c > start]:
        with open(path, "ab") as fh:
            fh.write(blob[len(path.read_bytes()):cut])
        before = pipe.num_frames
        added = pipe.refresh()
        if blob[cut - 1:cut] != b"\n":
            continue  # half a line: a reader opened now may also see half a number
        ref = _fresh(path, **kwargs)
        assert pipe.num_frames == ref.num_frames, cut
        assert added == pipe.num_frames - before
        for i in {0, pipe.num_frames - 1, min(max(before - 1, 0), pipe.num_frames - 1)}:
            a, b = pipe.compute(i), ref.compute(i)
            np.testing.assert_allclose(a.positions, b.positions)
            np.testing.assert_allclose(a.cell.matrix, b.cell.matrix)
            for key, value in b.attributes.items():
                if np.isscalar(value):
                    assert a.attributes.get(key) == value, (cut, i, key)
    assert pipe.refresh() == 0  # nothing new
    return pipe


def _write(path, data):
    path.write_bytes(data)
    return True


def test_lammps_dump_grows(tmp_path, lammps_dir):
    blob = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    pipe = _grow_and_compare(tmp_path / "dump.lammpstrj", blob, 97)
    assert pipe.num_frames == import_file(lammps_dir / "dump.custom.lammpstrj").num_frames


def test_triclinic_dump_grows(tmp_path, lammps_dir):
    blob = (lammps_dir / "dump.triclinic.lammpstrj").read_bytes()
    _grow_and_compare(tmp_path / "dump.lammpstrj", blob, 61)


@pytest.mark.parametrize("name", ["XDATCAR", "XDATCAR.npt", "md/XDATCAR"])
def test_xdatcar_grows(tmp_path, vasp_dir, name):
    blob = (vasp_dir / name).read_bytes()
    pipe = _grow_and_compare(tmp_path / "XDATCAR", blob, 23, thermo=None)
    assert pipe.num_frames == import_file(vasp_dir / name).num_frames


@pytest.mark.parametrize("name", ["OUTCAR", "md/OUTCAR"])
def test_outcar_grows(tmp_path, vasp_dir, name):
    blob = (vasp_dir / name).read_bytes()
    pipe = _grow_and_compare(tmp_path / "OUTCAR", blob, 211)
    full = import_file(vasp_dir / name)
    assert pipe.num_frames == full.num_frames
    for key in ("Energy", "Free Energy", "Pressure"):
        if key in full.compute(-1).attributes:
            assert pipe.compute(-1).attributes[key] == full.compute(-1).attributes[key]
    np.testing.assert_allclose(pipe.source.thermo.table["Step"], full.source.thermo.table["Step"])


def test_extxyz_grows(tmp_path):
    rng = np.random.default_rng(1)
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    frames = []
    for k in range(5):
        d = DataCollection.from_ase(atoms)
        d.particles["Position"] = d.positions + rng.normal(0, 0.05, d.positions.shape)
        d.attributes["Energy"] = -float(k)
        frames.append(d)
    src = tmp_path / "src.extxyz"
    write_extxyz(src, frames)
    _grow_and_compare(tmp_path / "traj.extxyz", src.read_bytes(), 173)


def test_rewritten_file_is_indexed_again(tmp_path, lammps_dir):
    path = tmp_path / "dump.lammpstrj"
    shutil.copy(lammps_dir / "dump.custom.lammpstrj", path)
    pipe = import_file(path)
    n = pipe.num_frames
    reader = pipe.source.readers[0]
    one = path.read_bytes()[reader._offsets[1]:reader._offsets[2]]
    path.write_bytes(one)  # a new run overwrote the file with one (different) frame
    assert pipe.refresh() == 1 - n
    assert pipe.num_frames == 1
    np.testing.assert_allclose(pipe.compute(0).positions, import_file(path).compute(0).positions)
    assert pipe.compute(0).attributes["Timestep"] == import_file(path).compute(0).attributes["Timestep"]


def test_wildcard_picks_up_new_files(tmp_path, lammps_dir):
    blob = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    (tmp_path / "dump.0.lammpstrj").write_bytes(blob)
    pipe = import_file(tmp_path / "dump.*.lammpstrj")
    n = pipe.num_frames
    (tmp_path / "dump.1.lammpstrj").write_bytes(b"ITEM: TIMESTEP\n")  # being created: not a frame yet
    assert pipe.refresh() == 0
    (tmp_path / "dump.1.lammpstrj").write_bytes(blob)
    assert pipe.refresh() == n
    np.testing.assert_allclose(pipe.compute(n).positions, pipe.compute(0).positions)


def test_compressed_and_static_sources_do_not_refresh(lammps_dir):
    pipe = import_file(lammps_dir / "dump.atom.gz")
    assert pipe.refresh() == 0
    from ewald.io import StaticSource
    from ewald.pipeline import Pipeline

    assert Pipeline(StaticSource(DataCollection.from_ase(bulk("Cu")))).refresh() == 0


def test_remote_dump_grows_over_ssh(server, lammps_dir):
    blob = (lammps_dir / "dump.custom.lammpstrj").read_bytes()
    remote_file = server.root + "/data/live.lammpstrj"
    local = import_file(lammps_dir / "dump.custom.lammpstrj")
    offsets = local.source.readers[0]._offsets
    with open(remote_file, "wb") as fh:
        fh.write(blob[:offsets[1] + 40])  # one frame and the start of the next
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = import_file("hpc:/data/live.lammpstrj")
    assert pipe.num_frames == 1
    with open(remote_file, "ab") as fh:
        fh.write(blob[offsets[1] + 40:])
    assert pipe.refresh() == local.num_frames - 1
    np.testing.assert_allclose(pipe.compute(-1).positions, local.compute(-1).positions)
    if shutil.which("grep"):
        assert any(c.startswith("tail -c +") for c in server.commands)  # only the new bytes were searched
