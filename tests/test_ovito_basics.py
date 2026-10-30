"""OVITO's everyday helper modifiers: FreezeProperty, UnwrapTrajectories, ExpandSelection,
LoadTrajectory and CombineDatasets."""
import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection, Pipeline, StaticSource, export_file
from ewald.bonds import Bonds
from ewald.data import Cell
from ewald.modifiers import (CombineDatasets, CreateBonds, ExpandSelection, ExpressionSelection, FreezeProperty,
                             LoadTrajectory, SelectType, UnwrapTrajectories)

BOX = 10.0


class _CountingSource(StaticSource):
    reads = 0

    def compute(self, frame):
        self.reads += 1
        return super().compute(frame)


def _walk(rng, T=40, n=50, sigma=0.4, ids=True, images=False):
    """Random walk wrapped into a cubic box; every frame shuffles the particle order."""
    true = np.cumsum(rng.normal(scale=sigma, size=(T, n, 3)), axis=0) + rng.random((n, 3)) * BOX
    frames = []
    for x in true:
        perm = rng.permutation(n) if ids else np.arange(n)
        d = DataCollection((x % BOX)[perm], np.array(["A"] * n), Cell(np.diag([BOX] * 3)))
        if ids:
            d.particles["Particle Identifier"] = perm + 1
        if images:
            d.particles["Periodic Image"] = np.floor(x / BOX).astype(np.int32)[perm]
        frames.append(d)
    return Pipeline(_CountingSource(frames)), true


def _by_id(data, values):
    return np.asarray(values)[np.argsort(data["Particle Identifier"])]


# --------------------------------------------------------------------------- FreezeProperty

def test_freeze_property_keeps_the_values_of_the_reference_frame(rng):
    pipe, true = _walk(rng, T=10, n=40)
    pipe.modifiers += [ExpressionSelection("Position.X < 5"),
                       FreezeProperty("Selection", "Initial Selection", freeze_at=2)]
    expected = (true[2] % BOX)[:, 0] < 5
    for frame in (0, 2, 7):
        data = pipe.compute(frame)
        np.testing.assert_array_equal(_by_id(data, data["Initial Selection"]).astype(bool), expected)
        now = (true[frame] % BOX)[:, 0] < 5
        np.testing.assert_array_equal(_by_id(data, data["Selection"]).astype(bool), now)


def test_freeze_property_overwrites_in_place_and_matches_by_index(rng):
    pipe, true = _walk(rng, T=5, n=20, ids=False)
    pipe.modifiers += [ExpressionSelection("Position.Z > 5"), FreezeProperty()]
    data = pipe.compute(4)
    np.testing.assert_array_equal(data["Selection"].astype(bool), (true[0] % BOX)[:, 2] > 5)


def test_freeze_property_fills_particles_that_did_not_exist(rng):
    a = DataCollection(np.zeros((3, 3)), ["Cu", "Zr", "Cu"])
    a.particles["Particle Identifier"] = np.array([1, 2, 3])
    b = DataCollection(np.zeros((4, 3)), ["Zr", "Cu", "Cu", "Cu"])
    b.particles["Particle Identifier"] = np.array([2, 4, 1, 3])
    pipe = Pipeline(StaticSource([a, b]), [FreezeProperty("Particle Type", "Type at start")])
    assert list(pipe.compute(1)["Type at start"]) == ["Zr", "", "Cu", "Cu"]


def test_freeze_property_caches_the_reference(rng):
    pipe, _ = _walk(rng, T=6, n=10)
    pipe.modifiers.append(FreezeProperty("Position", "Start"))
    pipe.compute(1)
    reads = pipe.source.reads
    for frame in (2, 3, 4):
        pipe.compute(frame)
    assert pipe.source.reads - reads == 3  # one read per frame, frame 0 not read again


# --------------------------------------------------------------------------- UnwrapTrajectories

def test_unwrap_trajectories_follows_crossings(rng):
    pipe, true = _walk(rng, T=40, n=50)
    pipe.modifiers.append(UnwrapTrajectories())
    for frame in (0, 39, 5, 22, 23, 38):  # jumps back and forth use the checkpoints
        data = pipe.compute(frame)
        np.testing.assert_allclose(_by_id(data, data.positions), true[frame] - np.floor(true[0] / BOX) * BOX,
                                   atol=1e-9)


def test_unwrap_trajectories_replays_only_from_the_nearest_checkpoint(rng):
    pipe, _ = _walk(rng, T=64, n=10)
    unwrap = UnwrapTrajectories()
    pipe.modifiers.append(unwrap)
    pipe.compute(63)
    pipe.source.reads = 0
    pipe.compute(61)
    assert pipe.source.reads <= unwrap.max_checkpoints // 4 + 2


def test_unwrap_trajectories_uses_periodic_images(rng):
    pipe, true = _walk(rng, T=8, n=30, images=True)
    pipe.modifiers.append(UnwrapTrajectories())
    data = pipe.compute(7)
    assert pipe.source.reads == 1  # no trajectory walk with image flags
    np.testing.assert_allclose(_by_id(data, data.positions), true[7], atol=1e-9)
    assert not data["Periodic Image"].any()


def test_unwrap_trajectories_keeps_bonds(rng):
    cell = Cell(np.diag([BOX] * 3))
    frames = []
    for x0 in (9.0, 9.8):  # the second atom crosses the boundary at frame 1
        d = DataCollection([[x0 - 1.0, 5, 5], [x0 % BOX if x0 < BOX else x0 - BOX, 5, 5]], ["A", "A"], cell)
        d.positions = [[x0 - 1.0, 5, 5], [(x0 + 0.5) % BOX, 5, 5]]
        d.bonds = Bonds([[0, 1]], [[1 if x0 + 0.5 >= BOX else 0, 0, 0]])
        frames.append(d)
    pipe = Pipeline(StaticSource(frames), [UnwrapTrajectories()])
    data = pipe.compute(1)
    np.testing.assert_allclose(data.positions[:, 0], [8.8, 10.3])
    np.testing.assert_allclose(data.bonds.vectors(data)[0], [1.5, 0, 0])


# --------------------------------------------------------------------------- ExpandSelection

@pytest.fixture
def cu_one_selected():
    data = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((4, 4, 4)))
    sel = np.zeros(data.count, np.int8)
    sel[0] = 1
    data.particles["Selection"] = sel
    return data


@pytest.mark.parametrize("mod, count", [
    (ExpandSelection("cutoff", cutoff=3.0), 13),
    (ExpandSelection("cutoff", cutoff=3.0, iterations=2), 55),     # fcc coordination sequence 12, 42
    (ExpandSelection("nearest", num_neighbors=12), 13),
    (ExpandSelection("nearest"), 2),
])
def test_expand_selection_counts_fcc_shells(cu_one_selected, mod, count):
    data = Pipeline(StaticSource(cu_one_selected), [mod]).compute(0)
    assert int(data["Selection"].sum()) == count
    assert data.attributes["ExpandSelection.num_added"] == count - 1


def test_expand_selection_along_bonds(cu_one_selected):
    pipe = Pipeline(StaticSource(cu_one_selected), [CreateBonds(cutoff=2.7), ExpandSelection("bonded", iterations=2)])
    assert int(pipe.compute(0)["Selection"].sum()) == 55
    with pytest.raises(ValueError, match="needs bonds"):
        Pipeline(StaticSource(cu_one_selected), [ExpandSelection("bonded")]).compute(0)


def test_expand_selection_with_nothing_selected(cu_one_selected):
    pipe = Pipeline(StaticSource(cu_one_selected), [SelectType("Zr"), ExpandSelection()])
    assert pipe.compute(0)["Selection"].sum() == 0


# --------------------------------------------------------------------------- LoadTrajectory, CombineDatasets

@pytest.fixture
def topology_and_dump(tmp_path, rng):
    """A topology with names, charges and bonds plus a 3-frame dump with numeric types in shuffled order."""
    n = 6
    cell = Cell(np.diag([BOX] * 3))
    oxygens = np.array([[9.7, 5.0, 5.0], [3.0, 3.0, 3.0]])  # the first water straddles the x boundary
    offsets = np.array([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
    base = np.concatenate([o + offsets for o in oxygens])
    topo = DataCollection(base % BOX, ["O", "H", "H", "O", "H", "H"], cell)
    topo.particles["Particle Identifier"] = np.arange(1, n + 1)
    topo.particles["Charge"] = np.array([-0.8, 0.4, 0.4, -0.8, 0.4, 0.4])
    topo.bonds = Bonds([[0, 1], [0, 2], [3, 4], [3, 5]])
    frames, truth = [], []
    for k in range(3):
        x = (base + 0.4 * k + rng.normal(scale=0.02, size=(n, 3))) % BOX
        perm = rng.permutation(n)
        d = DataCollection(x[perm], np.array(["2", "1", "1", "2", "1", "1"])[perm], cell)
        d.particles["Particle Identifier"] = perm + 1
        d.attributes["Timestep"] = 1000 * k
        frames.append(d)
        truth.append(x)
    path = tmp_path / "run.lammpstrj"
    export_file(frames, path)
    return topo, path, truth


def test_load_trajectory_puts_dump_frames_on_the_topology(topology_and_dump):
    topo, path, truth = topology_and_dump
    pipe = Pipeline(StaticSource(topo), [LoadTrajectory(str(path))])
    assert pipe.num_frames == 3
    for k in range(3):
        data = pipe.compute(k)
        np.testing.assert_allclose(data.positions, truth[k], atol=1e-4)
        assert list(data.types) == ["O", "H", "H", "O", "H", "H"]
        np.testing.assert_allclose(data["Charge"], topo["Charge"])
        assert data.attributes["Timestep"] == 1000 * k
        assert data.bonds.count == 4
        assert np.all(data.bonds.lengths(data) < 2.0)  # minimum image, also across the boundary


def test_load_trajectory_needs_matching_particles(topology_and_dump):
    topo, path, _ = topology_and_dump
    small = topo.take(np.arange(4))
    small.particles.pop("Particle Identifier")
    with pytest.raises(ValueError, match="no identifiers"):
        Pipeline(StaticSource(small), [LoadTrajectory(str(path))]).compute(0)
    with pytest.raises(ValueError, match="set 'file'"):
        Pipeline(StaticSource(topo), [LoadTrajectory()]).compute(0)
    assert Pipeline(StaticSource(topo), [LoadTrajectory()]).num_frames == 1


def test_combine_datasets_merges_particles_and_bonds(topology_and_dump, tmp_path):
    topo, path, _ = topology_and_dump
    extra = DataCollection([[1, 1, 1], [2, 2, 2]], ["Na", "Cl"], topo.cell)
    extra.particles["Particle Identifier"] = np.array([1, 2])
    xyz = tmp_path / "ions.extxyz"
    export_file(extra, xyz)
    data = Pipeline(StaticSource(topo), [CombineDatasets(str(xyz))]).compute(0)
    assert data.count == 8
    assert list(data.types[-2:]) == ["Na", "Cl"]
    np.testing.assert_array_equal(data["Particle Identifier"], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_allclose(data["Charge"][-2:], 0)
    assert data.attributes["CombineDatasets.num_merged"] == 2
    assert data.bonds.count == 4
    # the other file's bonds are appended with shifted indices; frame k uses the file's frame k
    piped = Pipeline(StaticSource([extra, extra]), [CombineDatasets(str(path), type_map={1: "H", 2: "O"})])
    merged = piped.compute(1)
    assert merged.count == 8 and sorted(set(merged.types)) == ["Cl", "H", "Na", "O"]


def test_particles_answer_ovito_count_and_positions():
    """Code written the OVITO way (data.particles.count / .positions) reads the same numbers; any
    dict assigned to particles keeps the shortcuts (the AI writes OVITO code first)."""
    data = DataCollection(np.zeros((3, 3)), ["Si"] * 3)
    assert data.particles.count == data.count == 3 and data.particles.positions is data.positions
    new = data.copy()
    new.particles = {k: v[:2] for k, v in data.particles.items()}
    assert new.particles.count == 2 and isinstance(new.particles, dict) and data.count == 3
    assert DataCollection().particles.count == 0


def test_cell_reads_like_ovitos_3x4_matrix():
    """data.cell[...] is OVITO's cell matrix (columns a, b, c, origin), read-only."""
    from ewald.data import Cell

    cell = Cell(np.diag([2.0, 3.0, 4.0]) + np.triu(np.ones((3, 3)), 1), origin=[1.0, 2.0, 3.0])
    assert cell[...].shape == (3, 4) and np.array_equal(cell[:, :3], cell.matrix.T)
    assert np.array_equal(cell[:, 0], cell.matrix[0]) and np.array_equal(cell[:, 3], cell.origin)
    with pytest.raises(ValueError):
        cell[...][0, 0] = 5.0
