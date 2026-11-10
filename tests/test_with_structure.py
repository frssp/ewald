"""``data.with_structure(...)``: a structure edited with ASE or pymatgen goes back into the frame
without losing what the pipeline computed (design review B3, D103)."""
import warnings

import numpy as np
import pytest

from ewald import Cell, DataCollection, Pipeline, StaticSource
from ewald.bonds import Bonds
from ewald.modifiers import CommonNeighborAnalysis, CoordinationAnalysis
from ewald.verbs import Table

ase = pytest.importorskip("ase")


def frame(origin=(0.0, 0.0, 0.0), types=None) -> DataCollection:
    from ase.build import bulk

    atoms = bulk("Cu", cubic=True) * (2, 2, 2)                      # 32 atoms
    data = DataCollection(atoms.get_positions() + np.asarray(origin), types if types is not None else ["Cu"] * 32,
                          Cell(np.array(atoms.cell), np.asarray(origin, float), np.ones(3, bool)))
    data["Structure Type"] = np.arange(32) % 3
    data["Label"] = np.array([f"a{i}" for i in range(32)])          # text: ASE does not carry it
    data.attributes["Run.temperature"] = 300.0
    data.tables["rdf"] = {"r": np.arange(3.0), "g(r)": np.ones(3)}
    data.tables["types"] = Table("types", {"Type": [1, 2], "Count": [30, 2]}, labels=["FCC", "HCP"])
    return data




def test_an_edit_with_the_same_atoms_keeps_everything():
    data = frame()
    data.bonds = Bonds([[0, 1], [1, 2]])
    atoms = data.to_ase()
    atoms.rattle(0.05, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = data.with_structure(atoms)
    assert np.allclose(new.positions, atoms.get_positions()) and not np.allclose(new.positions, data.positions)
    assert new.attributes["Run.temperature"] == 300.0
    assert set(new.tables) == {"rdf", "types"} and new.tables["types"].labels == ["FCC", "HCP"]
    assert set(new.particles) == set(data.particles)                 # no "Structure_Type" beside "Structure Type"
    assert np.array_equal(new["Structure Type"], data["Structure Type"]) and list(new["Label"][:2]) == ["a0", "a1"]
    assert len(new.bonds.properties["Topology"]) == 2
    assert np.allclose(data.positions, frame().positions)            # the frame it came from is untouched
    assert new.tables["rdf"] is not data.tables["rdf"] and new.particles is not data.particles


def test_the_cell_keeps_its_origin():
    data = frame(origin=(5.0, -2.0, 1.5))
    new = data.with_structure(data.to_ase())                         # ASE has no origin: to_ase() takes it off
    assert np.allclose(new.positions, data.positions) and np.allclose(new.cell.origin, [5.0, -2.0, 1.5])
    atoms = data.to_ase()
    atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
    scaled = data.with_structure(atoms)
    assert np.allclose(scaled.cell.matrix, data.cell.matrix * 1.1) and np.allclose(scaled.cell.origin, data.cell.origin)
    assert np.allclose(scaled.cell.to_fractional(scaled.positions), data.cell.to_fractional(data.positions))


def test_types_that_are_no_elements_keep_their_names():
    names = np.array(["1"] * 16 + ["Cu_surface"] * 15 + ["Zr"])
    data = frame(types=names)
    assert set(data.to_ase().get_chemical_symbols()) == {"X", "Zr"}
    new = data.with_structure(data.to_ase())
    assert list(new.types) == list(names)
    atoms = data.to_ase()
    atoms[0].symbol = "Ni"                                           # a substitution made in ASE wins
    assert list(data.with_structure(atoms).types[:2]) == ["Ni", "1"]


def test_when_the_atom_count_changes_what_cannot_follow_is_dropped_with_a_warning():
    data = frame()
    data.bonds = Bonds([[0, 1]])
    atoms = data.to_ase()
    del atoms[0]
    with pytest.warns(UserWarning) as caught:
        new = data.with_structure(atoms)
    assert len(caught) == 1 and caught[0].filename == __file__      # points at the caller's line
    text = str(caught[0].message)
    assert "32 -> 31" in text and "Label, the bonds could not be kept" in text and "attributes and tables are kept" in text
    assert new.count == 31 and "Label" not in new.particles and new.bonds is None
    assert np.array_equal(new["Structure Type"], data["Structure Type"][1:])   # ASE carried the numbers along
    assert new.attributes["Run.temperature"] == 300.0 and set(new.tables) == {"rdf", "types"}
    assert data.count == 32 and "Label" in data.particles and data.bonds is not None


def test_no_warning_when_nothing_is_lost():
    data = frame()
    del data.particles["Label"]
    atoms = data.to_ase() * (2, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = data.with_structure(atoms)
    assert new.count == 64 and np.array_equal(new["Structure Type"], np.tile(data["Structure Type"], 2))
    assert np.allclose(new.cell.matrix[0], data.cell.matrix[0] * 2)


def test_what_the_structure_carries_replaces_what_the_frame_had():
    from ase.calculators.singlepoint import SinglePointCalculator

    data = frame()
    data.attributes["Energy"] = -1.0
    atoms = data.to_ase()
    atoms.calc = SinglePointCalculator(atoms, energy=-2.5, forces=np.full((32, 3), 0.1))
    new = data.with_structure(atoms)
    assert new.attributes["Energy"] == -2.5 and np.allclose(new["Force"], 0.1)
    assert new.attributes["Run.temperature"] == 300.0


def test_a_pymatgen_structure():
    pytest.importorskip("pymatgen")
    data = frame()
    structure = data.to_pymatgen()
    structure.translate_sites(range(len(structure)), [0.01, 0.0, 0.0], frac_coords=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = data.with_structure(structure)
    assert new.count == 32 and new.attributes["Run.temperature"] == 300.0 and "rdf" in new.tables
    assert list(new["Label"][:2]) == ["a0", "a1"] and np.array_equal(new["Structure Type"], data["Structure Type"])
    assert not np.allclose(new.positions, data.positions)
    structure.remove_sites([0, 1])
    with pytest.warns(UserWarning, match="32 -> 30"):
        fewer = data.with_structure(structure)
    assert fewer.count == 30 and "rdf" in fewer.tables


def test_another_frame_as_the_structure():
    data = frame()
    other = DataCollection(np.zeros((2, 3)), ["H", "H"], Cell(np.eye(3) * 3.0, np.array([1.0, 0, 0]), np.ones(3, bool)))
    other.bonds = Bonds([[0, 1]])
    other.attributes["Energy"] = -6.7
    with pytest.warns(UserWarning, match="Structure Type, Label could not be kept"):
        new = data.with_structure(other)
    assert new.count == 2 and list(new.types) == ["H", "H"] and len(new.bonds.properties["Topology"]) == 1
    assert np.allclose(new.cell.origin, [1.0, 0, 0]) and np.allclose(new.cell.matrix, np.eye(3) * 3.0)
    assert new.attributes == {"Run.temperature": 300.0, "Energy": -6.7} and set(new.tables) == {"rdf", "types"}
    assert new.bonds is not other.bonds and new.cell is not other.cell


def test_other_things_are_refused_by_name():
    with pytest.raises(TypeError, match="takes an ASE Atoms, a pymatgen Structure or Molecule, or a DataCollection, not ndarray"):
        frame().with_structure(np.zeros((3, 3)))


def test_a_step_that_edits_with_ase_keeps_the_results_above_it():
    """The case from the review: returning ``DataCollection.from_ase(...)`` lost attributes and tables."""
    pipe = Pipeline(StaticSource(frame()))
    pipe.add(CommonNeighborAnalysis())
    pipe.add(CoordinationAnalysis(cutoff=3.0))

    def lost(frame, data):
        return DataCollection.from_ase(data.to_ase())

    def kept(frame, data):
        atoms = data.to_ase()
        atoms.rattle(0.01, seed=frame)
        return data.with_structure(atoms)

    before = pipe.compute(0)
    pipe.add(lost)
    assert not pipe.compute(0).tables and "Run.temperature" not in pipe.compute(0).attributes
    pipe.modifiers[-1] = kept
    after = pipe.compute(0)
    assert set(after.tables) == set(before.tables) and set(after.attributes) >= set(before.attributes)
    assert set(after.particles) == set(before.particles)
    assert np.array_equal(after["Coordination"], before["Coordination"])
    assert not np.allclose(after.positions, before.positions)
    assert np.allclose(pipe.source_frame(0).positions, frame().positions)   # the cached source frame is untouched
