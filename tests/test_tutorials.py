"""The tutorial notebooks in examples/notebooks run as written (docs/user/tutorials/)."""
from pathlib import Path

import pytest

from ewald.notebook import Notebook

NOTEBOOKS = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


def run(name, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)                 # notebooks write their figures into the working folder
    nb = Notebook.load(NOTEBOOKS / name)
    ok = nb.run_all()
    errors = [o.data for c in nb.cells for o in c.outputs if o.kind == "error"]
    assert ok and not errors, errors
    texts = "".join(o.data for c in nb.cells for o in c.outputs if o.kind == "text")
    return nb, texts


def test_every_tutorial_notebook_is_tested():
    assert sorted(p.name for p in NOTEBOOKS.glob("*.py")) == ["cu_md_trajectory.py", "cu_void.py", "srtio3_octahedra.py"]


def test_cu_void(tmp_path, monkeypatch):
    nb, out = run("cu_void.py", tmp_path, monkeypatch)
    assert "78 surface atoms of 462" in out              # quoted in first-notebook.md
    assert sum(c.is_step for c in nb.cells) == 1 and sum(c.is_view for c in nb.cells) == 3
    surface = nb.pipeline.compute(4).tables["surface"]           # the @all_frames step
    assert list(surface["Surface atoms"]) == [78, 78, 78, 82, 107]   # quoted in first-notebook.md


def test_srtio3_octahedra(tmp_path, monkeypatch):
    nb, out = run("srtio3_octahedra.py", tmp_path, monkeypatch)
    assert "Pm-3m 221 Sr 1a, Ti 1b, O 3c" in out           # quoted in structure-figure.md
    assert (tmp_path / "srtio3.png").stat().st_size > 0
    assert nb.pipeline.compute(0).attributes["CoordinationPolyhedra.count"] == 8
    assert nb.pipeline.view.boundary == [-0.05, 1.05] and nb.pipeline.view.theme == "VESTA"   # the view cell


def test_cu_md_trajectory(tmp_path, monkeypatch):
    nb, out = run("cu_md_trajectory.py", tmp_path, monkeypatch)
    data = nb.pipeline.compute(9)
    assert "msd" in data.tables and "Displacement Magnitude" in data
    assert len(data.tables["run"]["Energy"]) == 10 and "Force Magnitude" in data   # @all_frames, @each_frame
