"""``output=`` on every step and the warning when a step replaces an earlier step's result (D106)."""
import inspect
import warnings

import numpy as np
import pytest

import ewald
from ewald import session
from ewald.modifiers import (AssignColor, CommonNeighborAnalysis, ComputeProperty, CoordinationAnalysis,
                             ExpressionSelection, InvertSelection, PolyhedralTemplateMatching)
from ewald.pipeline import _modifier_key


@pytest.fixture
def pipe():
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(3)
    atoms.rattle(0.05, seed=1)
    return ewald.load(atoms)


def _compute(pipe):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = pipe.compute(0)
    return data, [str(w.message) for w in caught if "replaced what" in str(w.message)]


def test_a_second_step_of_the_same_kind_warns_once(pipe):
    pipe.add(CoordinationAnalysis(cutoff=3.0))
    pipe.compute(0)                      # the first step's result comes from the cache below
    pipe.add(CoordinationAnalysis(cutoff=4.5))
    data, told = _compute(pipe)
    assert len(told) == 1
    assert "an earlier CoordinationAnalysis" in told[0] and "property 'Coordination'" in told[0]
    assert "table 'coordination-rdf'" in told[0] and "output='name'" in told[0]
    pipe.invalidate()
    assert _compute(pipe)[1] == []       # once per step and name


def test_output_keeps_both_results_under_their_names(pipe):
    pipe.add(CoordinationAnalysis(cutoff=3.0))
    pipe.add(CoordinationAnalysis(cutoff=4.5, output="far"))
    data, told = _compute(pipe)
    assert told == []
    assert list(data.particles)[-2:] == ["Coordination", "Coordination (far)"]
    assert (data["Coordination"] == 12).all() and data["Coordination (far)"].min() > 12
    assert set(data.tables) >= {"coordination-rdf", "coordination-rdf (far)"}
    summary = pipe.modifiers[-1].last_output
    assert summary["properties"] == ["Coordination (far)"] and summary["tables"] == ["coordination-rdf (far)"]
    selected = pipe.compute(0)
    pipe.add(ExpressionSelection("Coordinationfar > 12"))
    assert pipe.compute(0).selection.sum() == (selected["Coordination (far)"] > 12).sum()


def test_cna_beside_ptm_keeps_both_structure_types(pipe):
    pipe.add(PolyhedralTemplateMatching())
    pipe.add(CommonNeighborAnalysis(output="cna"))
    data, told = _compute(pipe)
    assert told == []
    assert "Structure Type" in data.particles and "Structure Type (cna)" in data.particles
    assert {"structures", "structures (cna)"} <= set(data.tables)
    assert "CommonNeighborAnalysis.counts.FCC (cna)" in data.attributes
    assert "PolyhedralTemplateMatching.counts.FCC" in data.attributes
    # the colors are CNA's, and the color rule names the property they came from
    assert data.attributes["Color.rule"]["property"] == "Structure Type (cna)"


def test_shared_names_and_equal_values_are_no_overwrite(pipe):
    pipe.add(CoordinationAnalysis(cutoff=3.0))
    pipe.add(AssignColor("red"))
    pipe.add(ExpressionSelection("Coordination > 11"))
    pipe.add(InvertSelection())
    pipe.add(AssignColor("blue"))
    pipe.add(CoordinationAnalysis(cutoff=3.0))   # the same numbers again
    assert _compute(pipe)[1] == []


def test_output_is_a_setting_of_every_step():
    cna = CommonNeighborAnalysis(output="cna")
    assert cna.parameters()["output"] == "cna" and repr(cna).endswith("output='cna')")
    assert "output" not in CommonNeighborAnalysis().parameters()
    assert _modifier_key(cna) != _modifier_key(CommonNeighborAnalysis())
    assert InvertSelection(output="x").output == "x"           # a step without a constructor of its own
    assert ComputeProperty("Energy2", "1").output == "Energy2"   # its own output: the property it writes
    assert "output" not in str(inspect.signature(CommonNeighborAnalysis))
    with pytest.raises(ValueError, match="output must be a name"):
        CommonNeighborAnalysis(output="")
    again = session.modifier_from_dict(session.modifier_to_dict(cna))[0]
    assert type(again) is CommonNeighborAnalysis and again.output == "cna"


def test_verbs_take_output(pipe):
    from ewald.verbs import cna

    first = cna(pipe)
    second = cna(pipe, output="b")
    assert first.name == "structures" and second.name == "structures (b)"
    assert len(pipe.modifiers) == 2
    np.testing.assert_array_equal(np.asarray(first["Count"]), np.asarray(second["Count"]))


def test_dataclass_steps_keep_their_generated_constructor():
    from dataclasses import dataclass

    from ewald.pipeline import Modifier

    @dataclass
    class Scale(Modifier):
        factor: float = 2.0

        def modify(self, data, ctx):
            return None

    assert Scale(factor=3.0).factor == 3.0 and Scale().parameters() == {"factor": 2.0}
