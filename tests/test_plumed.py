"""PLUMED support: CVs through the PLUMED library match ``plumed driver`` on the same frames, and
COLVAR/HILLS parsing and the FES match ``plumed sum_hills`` (reference files made by
tests/data/plumed/make_plumed_reference.py). Tests that run PLUMED skip when it is missing."""
import subprocess
import sys

import numpy as np
import pytest

from ewald import DataCollection, Pipeline, StaticSource
from ewald import plumed as pl
from ewald.data import Cell
from ewald.modifiers import FreeEnergySurface, LoadColvar, Plumed, TimeSeries

DATA = __import__("pathlib").Path(__file__).parent / "data" / "plumed"
needs_plumed = pytest.mark.skipif(not pl.available(), reason="PLUMED library (plumed package + kernel) not installed")
needs_driver = pytest.mark.skipif(not pl.available() or pl.plumed_executable() is None,
                                  reason="plumed command line tool not installed")

CELL = np.array([[12.0, 0, 0], [1.5, 11.0, 0], [0.7, -0.4, 10.0]])


def _frames(n=40, nframes=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, (n, 3)) @ CELL
    types = np.array(["O" if i % 3 == 0 else "H" for i in range(n)])
    out = []
    for k in range(nframes):
        x = base + rng.normal(0, 0.4, (n, 3)) * np.sqrt(k + 1)
        out.append(DataCollection(x, types, Cell(CELL)))
    return out


def _driver(text, frames, tmp_path, kt=0.0):
    """Run ``plumed driver`` on ``frames`` (written as xyz with the cell on the comment line)."""
    with open(tmp_path / "traj.xyz", "w") as fh:
        for d in frames:
            fh.write(f"{d.count}\n" + " ".join(repr(float(v)) for v in d.cell.matrix.ravel()) + "\n")
            for t, p in zip(d.types, d.positions):
                fh.write(f"{t} " + " ".join(repr(float(v)) for v in p) + "\n")
    with open(tmp_path / "mc.dat", "w") as fh:
        fh.write("#! FIELDS index mass charge\n")
        for i, m in enumerate(pl._masses(frames[0])):
            fh.write(f"{i} {float(m)!r} 0\n")
    (tmp_path / "plumed.dat").write_text(text)
    cmd = [pl.plumed_executable(), "driver", "--plumed", "plumed.dat", "--ixyz", "traj.xyz", "--length-units", "A",
           "--mc", "mc.dat"] + (["--kt", repr(kt)] if kt else [])
    res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]


CVS = """
d: DISTANCE ATOMS=1,2
dc: DISTANCE ATOMS=3,5 COMPONENTS
dn: DISTANCE ATOMS=3,5 NOPBC
t: TORSION ATOMS=1,2,3,4
c: COORDINATION GROUPA=1-40:3 GROUPB=2-40:3 R_0=0.3
g: GROUP ATOMS=1-5
cm: COM ATOMS=1-3
dcm: DISTANCE ATOMS=cm,7
m: METAD ARG=d SIGMA=0.05 HEIGHT=1.2 PACE=2 BIASFACTOR=5 TEMP=300 FILE=HILLS
"""
PRINTED = ["d", "dc.x", "dc.y", "dc.z", "dn", "t", "c", "dcm", "m.bias"]


@needs_driver
def test_values_match_plumed_driver(tmp_path):
    frames = _frames()
    res = pl.run(CVS, frames, temperature=300)
    _driver(CVS + f"PRINT ARG={','.join(PRINTED)} FILE=COLVAR FMT=%.17g\n", frames, tmp_path, pl.KB_KJMOL * 300)
    ref = pl.read_colvar(tmp_path / "COLVAR")
    for name in PRINTED:
        np.testing.assert_allclose(res["values"][name], ref[name], rtol=0, atol=1e-12, err_msg=name)
    # labels without values (GROUP) are skipped, virtual-atom components are kept, the run left no files
    assert "g" not in res["labels"] and "cm.mass" in res["values"]
    assert res["workdir"] is None
    assert res["bias"][0] == 0 and res["bias"][-1] > 0


@needs_driver
def test_modifier_attributes_table_and_ids(tmp_path):
    frames = _frames(nframes=6)
    # PLUMED atom i is the particle with identifier i, wherever it is stored
    perm = np.random.default_rng(1).permutation(frames[0].count)
    shuffled = []
    for d in frames:
        s = DataCollection(d.positions[perm], d.types[perm], d.cell)
        s["Particle Identifier"] = perm + 1
        shuffled.append(s)
    text = "d: DISTANCE ATOMS=1,2\nt: TORSION ATOMS=1,2,3,4\n"
    pipe = Pipeline(StaticSource(shuffled), [Plumed(text), TimeSeries("Plumed.d")])
    _driver(text + "PRINT ARG=d,t FILE=COLVAR FMT=%.17g\n", frames, tmp_path)
    ref = pl.read_colvar(tmp_path / "COLVAR")
    for k in range(len(frames)):
        out = pipe.compute(k)
        assert out.attributes["Plumed.d"] == pytest.approx(ref["d"][k], abs=1e-12)
        assert out.attributes["Plumed.t"] == pytest.approx(ref["t"][k], abs=1e-12)
    table = out.tables["plumed"]
    np.testing.assert_allclose(table["d"], ref["d"], atol=1e-12)
    assert list(table["Frame"]) == list(range(len(frames)))
    np.testing.assert_allclose(out.tables["time-series"]["Plumed.d"], ref["d"], atol=1e-12)


@needs_plumed
def test_per_atom_values_become_properties():
    frames = _frames(n=12, nframes=3)
    perm = np.arange(12)[::-1]
    for d in frames:
        d["Particle Identifier"] = perm + 1          # particle 0 is PLUMED atom 12
    text = "cn: COORDINATIONNUMBER SPECIES=1-12 SWITCH={RATIONAL R_0=0.3}\ns: SUM ARG=cn PERIODIC=NO\n"
    try:
        out = Pipeline(StaticSource(frames), [Plumed(text)]).compute(2)
    except RuntimeError as exc:
        if "not known" in str(exc):
            pytest.skip("PLUMED built without the symfunc module")
        raise
    cn = out["Plumed cn"]
    assert cn.shape == (12,)
    assert out.attributes["Plumed.s"] == pytest.approx(cn.sum())
    plain = Pipeline(StaticSource([DataCollection(d.positions[perm], d.types, d.cell) for d in frames]),
                     [Plumed(text, by_identifier=False)]).compute(2)
    np.testing.assert_allclose(plain["Plumed cn"][perm], cn, atol=1e-12)


@needs_plumed
def test_input_file_semicolons_and_errors(tmp_path):
    frames = _frames(nframes=3)
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "plumed.dat").write_text("INCLUDE FILE=cvs.dat\nPRINT ARG=d FILE=COLVAR\n")
    (tmp_path / "ref" / "cvs.dat").write_text("d: DISTANCE ATOMS=1,2\n")
    (tmp_path / "ref" / "COLVAR").write_text("keep me\n")
    a = Pipeline(StaticSource(frames), [Plumed(str(tmp_path / "ref" / "plumed.dat"))]).compute(1)
    b = Pipeline(StaticSource(frames), [Plumed("d: DISTANCE ATOMS=1,2; e: DISTANCE ATOMS=2,3")]).compute(1)
    assert a.attributes["Plumed.d"] == b.attributes["Plumed.d"] and "Plumed.e" in b.attributes
    assert (tmp_path / "ref" / "COLVAR").read_text() == "keep me\n"     # PLUMED ran elsewhere
    work = tmp_path / "work"
    Pipeline(StaticSource(frames), [Plumed("d: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=MYCV", workdir=str(work))]).compute(0)
    assert len(pl.read_colvar(work / "MYCV")["d"]) == 3
    with pytest.raises(RuntimeError, match="NOSUCHACTION"):
        Pipeline(StaticSource(frames), [Plumed("x: NOSUCHACTION ATOMS=1")]).compute(0)


def test_missing_plumed_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "plumed", None)
    assert not pl.available()
    pipe = Pipeline(StaticSource(_frames(nframes=2)), [Plumed("d: DISTANCE ATOMS=1,2")])
    with pytest.raises(pl.PlumedNotFound, match="conda install"):
        pipe.compute(0)
    out = pipe.compute(0, errors="skip")               # the GUI keeps the rest of the pipeline
    assert "Plumed.d" not in out.attributes and "conda install" in pipe.modifiers[0].status


def test_labels_and_referenced_files(tmp_path):
    text = """# comment
d: DISTANCE ATOMS=1,2   # trailing
DISTANCE ATOMS=1,3 LABEL=e
METAD ...
  ARG=d SIGMA=0.1 HEIGHT=1 PACE=10
  LABEL=m FILE=HILLS
...
r: RMSD REFERENCE=ref.pdb TYPE=OPTIMAL
"""
    assert pl.labels(text) == ["d", "e", "m", "r"]
    (tmp_path / "ref.pdb").write_text("x")
    assert [p.name for p in pl._referenced_files(text, tmp_path)] == ["ref.pdb"]


# ---- files: COLVAR, HILLS, FES
def test_read_colvar_restarts_and_constants(tmp_path):
    f = tmp_path / "COLVAR"
    f.write_text("#! FIELDS time d\n#! SET min_t -pi\n 0 1.0\n 1 1.5\n#! FIELDS time d t\n 2 2.0 0.3\n 3 2.5\n")
    t = pl.read_colvar(f)
    assert list(t) == ["time", "d", "t"]
    np.testing.assert_allclose(t["d"], [1.0, 1.5, 2.0])   # the truncated last line is dropped
    assert np.isnan(t["t"][0]) and t["t"][2] == 0.3
    assert pl._const(t.constants["min_t"]) == pytest.approx(-np.pi)


@pytest.mark.parametrize("name, kw", [("1d", dict(bins=60)),
                                      ("2d", dict(bins=(30, 40), ranges=[(0.0, 0.6), None]))])
def test_fes_matches_plumed_sum_hills(name, kw):
    ref = pl.read_fes(DATA / f"fes_{name}.dat")
    mine = pl.fes_from_hills(DATA / f"HILLS_{name}", **kw)
    assert mine["cvs"] == ref["cvs"]
    for a, b in zip(mine["grid"], ref["grid"]):
        np.testing.assert_allclose(a, b, atol=1e-8)
    np.testing.assert_allclose(mine["fes"], ref["fes"], atol=1e-8)   # sum_hills prints 9 decimals


@needs_driver
def test_sum_hills_wrapper():
    out = pl.sum_hills(DATA / "HILLS_1d", bins=60)
    np.testing.assert_allclose(out["fes"], pl.read_fes(DATA / "fes_1d.dat")["fes"], atol=1e-8)


def test_fes_modifier_and_history(tmp_path):
    frames = [DataCollection(np.zeros((1, 3)), np.array(["C"]))] * 3
    mod = FreeEnergySurface(str(DATA / "HILLS_2d"), bins=(30, 40), ranges=[(0.0, 0.6), None])
    out = Pipeline(StaticSource(frames), [mod]).compute(0)
    fes = out.tables["fes"]
    assert set(fes) == {"d", "t", "Free Energy"} and len(fes["d"]) == 31 * 40
    assert min(fes["Free Energy"]) == 0
    n = len(pl.read_hills(DATA / "HILLS_2d"))
    assert out.attributes["FreeEnergySurface.hills"] == n
    # FES at the time of a frame: HILLS time 2, 4, ... against frame * time_per_frame
    early = FreeEnergySurface(str(DATA / "HILLS_1d"), bins=60, up_to_frame=True, time_per_frame=1.0)
    out = Pipeline(StaticSource(frames), [early]).compute(2)
    assert out.attributes["FreeEnergySurface.hills"] == 1
    hist = pl.fes_from_hills(DATA / "HILLS_1d", bins=60, stride=50)
    np.testing.assert_allclose(hist["history"][-1], hist["fes"])


def test_load_colvar_alignment(tmp_path):
    f = tmp_path / "COLVAR"
    rows = "\n".join(f" {0.5 * i} {i * 10.0}" for i in range(20))
    f.write_text("#! FIELDS time d\n" + rows + "\n")
    frames = []
    for k in range(5):
        d = DataCollection(np.zeros((1, 3)), np.array(["C"]))
        d.attributes["Timestep"] = 1000 * k
        frames.append(d)
    by_index = Pipeline(StaticSource(frames), [LoadColvar(str(f), stride=4, offset=5)])
    assert by_index.compute(2).attributes["Colvar.d"] == 130.0
    assert np.isnan(by_index.compute(4).attributes["Colvar.d"])           # row 21 of 20
    by_time = Pipeline(StaticSource(frames), [LoadColvar(str(f), align="time", md_timestep=0.002)])
    assert by_time.compute(3).attributes["Colvar.d"] == 120.0               # t = 6 ps
    assert by_time.compute(0).tables["colvar"]["d"][-1] == 190.0


def test_plot_fes_draws_1d_and_2d():
    import matplotlib
    matplotlib.use("Agg")
    ax1 = pl.plot_fes(pl.read_fes(DATA / "fes_1d.dat"))
    ax2 = pl.plot_fes(pl.read_fes(DATA / "fes_2d.dat"))
    assert ax1.get_xlabel() == "t" and ax2.get_ylabel() == "t"


# ---- Claude tools
@needs_plumed
def test_claude_tools_compute_cvs():
    from ewald.llm.tools import Toolset

    ts = Toolset(Pipeline(StaticSource(_frames(nframes=5))))
    r = ts.call("compute_plumed_cvs", {"input": "d: DISTANCE ATOMS=1,2\nt: TORSION ATOMS=1,2,3,4"})
    assert not r.is_error, r.content
    assert set(r.content["values"]) == {"d", "t", "Bias"}
    assert len(r.content["table"]["Frame"]) == 5
    assert "Plumed(" in r.content["pipeline"][0]
    assert "Plumed(" in ts.to_script() and "from ewald.modifiers import Plumed" in ts.to_script()
    ev = ts.call("evaluate_expression", {"expression": "Plumed.d", "frame": 2})   # attributes in expressions
    assert ev.content["value"] == pytest.approx(r.content["table"]["d"][2], rel=1e-3)
    bad = ts.call("compute_plumed_cvs", {"input": "x: NOSUCHACTION"})
    assert bad.is_error and len(ts.pipeline.modifiers) == 1


def test_claude_tools_read_plumed_output(tmp_path):
    from ewald.llm.tools import Toolset

    ts = Toolset(workdir=str(tmp_path))
    r = ts.call("read_plumed_output", {"path": str(DATA / "HILLS_2d"), "bins": 40})
    assert not r.is_error, r.content
    assert r.content["kind"] == "hills" and r.content["cvs"] == ["d", "t"] and r.images
    f = tmp_path / "COLVAR"
    f.write_text("#! FIELDS time d\n0 1\n1 3\n")
    r = ts.call("read_plumed_output", {"path": str(f)})
    assert r.content["kind"] == "colvar" and r.content["columns"]["d"]["mean"] == 2.0
