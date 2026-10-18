"""Optional bridge to the `ovito` package; skipped when it is not installed."""
import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("ovito")

from ewald import DataCollection, Pipeline, StaticSource  # noqa: E402
from ewald.modifiers import DislocationAnalysis, OvitoModifier  # noqa: E402


def _fcc():
    return DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((6, 6, 6)))


def test_ovito_modifier_roundtrip():
    from ovito.modifiers import CentroSymmetryModifier

    d = Pipeline(StaticSource(_fcc()), [OvitoModifier(CentroSymmetryModifier(num_neighbors=12))]).compute()
    np.testing.assert_allclose(d["Centrosymmetry"], 0.0, atol=1e-8)
    assert d.count == 864


def test_dxa_perfect_crystal_has_no_dislocations():
    d = Pipeline(StaticSource(_fcc()), [DislocationAnalysis()]).compute()
    assert d.attributes["DislocationAnalysis.total_line_length"] == 0
    assert d.attributes["DislocationAnalysis.counts.FCC"] == 864
    assert len(d.tables["dislocations"]["Length"]) == 0
