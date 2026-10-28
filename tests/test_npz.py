"""The built-in NumPy .npz trajectory reader and writer (ewald.io.npz)."""
import numpy as np
import pytest

from ewald import Cell, DataCollection, export_file, import_file, plugins
from ewald.io import NpzTrajectoryReader, is_npz_trajectory


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """A registry loaded from scratch, whatever earlier tests left in it."""
    from ewald import plugins

    monkeypatch.setattr(plugins, "_registry", plugins._Registry())
    yield
    monkeypatch.setattr(plugins, "_registry", plugins._Registry())


def _npz(path, frames=4, atoms=5, **extra):
    rng = np.random.default_rng(1)
    arrays = dict(positions=rng.random((frames, atoms, 3)) * 10, types=np.array(["Li"] * 2 + ["O"] * 3),
                  cell=np.eye(3) * 10, forces=rng.random((frames, atoms, 3)), energies=np.arange(frames, dtype=float))
    arrays.update(extra)
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})
    return path


def test_reads_positions_types_cell_properties_and_attributes(tmp_path):
    pipe = import_file(_npz(tmp_path / "md.npz"))
    assert isinstance(pipe.source.readers[0], NpzTrajectoryReader)
    assert pipe.num_frames == 4
    d = pipe.compute(2)
    assert d.count == 5 and list(d.types) == ["Li", "Li", "O", "O", "O"]
    assert d["Force"].shape == (5, 3) and d.attributes["Energy"] == 2.0
    assert np.isclose(d.cell.volume, 1000.0) and d.cell.pbc.all()


def test_found_by_contents_not_only_by_name(tmp_path):
    path = _npz(tmp_path / "md.npz")
    assert plugins.find_reader(str(path)).name == "npz-trajectory"
    other = tmp_path / "weights.npz"
    np.savez(other, w=np.zeros(3))
    assert not is_npz_trajectory(other) and not is_npz_trajectory(tmp_path / "missing.npz")
    text = tmp_path / "x.npz"
    text.write_text("not a zip", encoding="utf-8")
    assert not is_npz_trajectory(text)
    with pytest.raises(ValueError, match="no 'positions' array"):
        NpzTrajectoryReader(other)
    assert import_file(path, format="npz").num_frames == 4        # alias


def test_other_key_spellings_and_type_map(tmp_path):
    path = tmp_path / "mlip.npz"
    np.savez(path, coords=np.zeros((2, 3)), numbers=np.array([3, 8]), box=np.diag([4.0, 5, 6]),
             pbc=np.array([True, True, False]), origin=np.array([1.0, 0, 0]))
    d = import_file(path).compute(0)
    assert list(d.types) == ["Li", "O"]
    assert list(d.cell.pbc) == [True, True, False] and d.cell.origin[0] == 1.0
    path = tmp_path / "lammps.npz"
    np.savez(path, positions=np.zeros((1, 2, 3)), types=np.array([1, 2]))
    assert list(import_file(path, type_map={1: "Cu", 2: "Zr"}).compute(0).types) == ["Cu", "Zr"]


def test_pickled_arrays_are_refused(tmp_path):
    """A .npz from elsewhere must not run code: object arrays need pickle, which stays off."""
    path = tmp_path / "evil.npz"
    np.savez(path, positions=np.zeros((1, 1, 3)), types=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        NpzTrajectoryReader(path)


def test_written_trajectory_reads_back(tmp_path):
    frames = []
    for i in range(3):
        d = DataCollection(np.arange(9, dtype=float).reshape(3, 3) + i, ["Cu", "Zr", "Cu"],
                           Cell(np.eye(3) * (10 + i), pbc=(True, True, False)))
        d.particles["Force"] = np.full((3, 3), float(i))
        d.particles["Structure Type"] = np.array([1, 2, 1])
        d.attributes["Energy"] = -1.5 * i
        d.attributes["Note"] = "text is left out"
        frames.append(d)
    path = tmp_path / "out.npz"
    export_file(frames, path)
    with np.load(path) as z:
        assert {"positions", "types", "cell", "pbc", "forces", "Structure Type", "energies"} <= set(z.files)
        assert z["cell"].shape == (3, 3, 3)        # the cell changes, so one per frame
    pipe = import_file(path)
    back = pipe.compute(2)
    assert np.allclose(back.positions, frames[2].positions) and list(back.types) == ["Cu", "Zr", "Cu"]
    assert np.isclose(back.cell.matrix[0, 0], 12) and list(back.cell.pbc) == [True, True, False]
    assert np.allclose(back["Force"], 2.0) and list(back["Structure Type"]) == [1, 2, 1]
    assert back.attributes["Energy"] == -3.0 and "Note" not in back.attributes


def test_writer_needs_one_atom_count(tmp_path):
    with pytest.raises(ValueError, match="same number of atoms"):
        export_file([DataCollection(np.zeros((1, 3)), ["H"]), DataCollection(np.zeros((2, 3)), ["H", "H"])],
                    tmp_path / "bad.npz")


def test_builtin_group_can_be_switched_off(tmp_path):
    assert plugins.installed()["npz"].contributions == {"readers": ["npz-trajectory"], "writers": ["npz-trajectory"]}
    plugins.set_enabled("npz", False)
    try:
        assert "npz-trajectory" not in plugins.readers()
    finally:
        plugins.set_enabled("npz", True)
    assert "npz-trajectory" in plugins.readers()
