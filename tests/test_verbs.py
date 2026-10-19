import numpy as np
import pytest

import ewald
from ewald import Pipeline, StaticSource
from ewald.notebook import Notebook
from ewald.verbs import Table


@pytest.fixture
def cu_pipe(cu_fcc):
    return Pipeline(StaticSource(cu_fcc))


def test_rdf_adds_one_step_and_returns_the_table(cu_pipe):
    g = ewald.rdf(cu_pipe, cutoff=5.0, bins=100)
    assert isinstance(g, Table) and g.name == "coordination-rdf" and len(g.x) == 100
    y = g.columns[1]
    assert abs(g.x[np.argmax(g[y])] - 3.615 / np.sqrt(2)) < 0.1          # first fcc shell
    assert [type(m).__name__ for m in cu_pipe.modifiers] == ["CoordinationAnalysis"]
    assert "Coordination" in cu_pipe.compute(0).particles
    g2 = ewald.rdf(cu_pipe, cutoff=4.0, bins=50)                           # replaces, no duplicate step
    assert len(cu_pipe.modifiers) == 1 and cu_pipe.modifiers[0].cutoff == 4.0 and len(g2.x) == 50
    assert "coordination-rdf" in repr(g2) and "50 rows" in repr(g2)


def test_verbs_keep_other_steps_and_msd_runs_over_the_trajectory(rng):
    from test_trajectory import _walk_pipeline

    pipe, _ = _walk_pipeline(rng, T=20, n=50)
    ewald.rdf(pipe, cutoff=3.0)
    d = ewald.msd(pipe, time_per_frame=0.5)
    assert [type(m).__name__ for m in pipe.modifiers] == ["CoordinationAnalysis", "MeanSquaredDisplacement"]
    assert d.columns[:2] == ["Time", "MSD"] and d.x[1] == 0.5
    assert ewald.msd(pipe).x[1] == 1.0 and len(pipe.modifiers) == 2


def test_verb_in_a_notebook_cell_is_replaced_when_the_cell_runs_again(cu_fcc, tmp_path):
    from ase.io import write

    write(tmp_path / "cu.extxyz", cu_fcc.to_ase())
    nb = Notebook.from_py(f"# %%\nfrom ewald import load, rdf\npipe = load({str(tmp_path / 'cu.extxyz')!r})\n\n"
                          "# %%\ng = rdf(pipe, cutoff=4.0)\nprint(len(g.x))\n")
    assert nb.run_all()
    assert nb.run_cell(1) and len(nb.pipeline.modifiers) == 1
    assert nb.cells[1].outputs[-1].data.strip() == "200"


def test_sq_table_is_a_plain_dict_of_columns(cu_pipe):
    s = ewald.sq(cu_pipe, q_max=6.0, bins=60)
    assert s.name == "structure-factor" and s.columns[0] == "q" and s.x.max() <= 6.0
    assert all(len(v) == len(s.x) for v in s.values())
