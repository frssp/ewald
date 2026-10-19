"""Histogram, SpatialBinning, ReduceProperty, bond angle/length distributions and FindRings against
OVITO 3.16; SteinhardtOrderParameters against freud.

Reference data come from tests/data/make_analysis_reference.py (needs `ovito` and `freud-analysis`).
"""
import os

import numpy as np
import pytest
from ase.build import bulk

from ewald import DataCollection
from ewald.backend import CPU, Backend
from ewald.bonds import Bonds
from ewald.data import Cell
from ewald.modifiers import (BondAngleDistribution, BondLengthDistribution, CreateBonds, FindRings, Histogram,
                             ReduceProperty, SpatialBinning, SteinhardtOrderParameters)
from ewald.modifiers.topology import shortest_path_rings
from ewald.pipeline import ComputeContext

REF = np.load(os.path.join(os.path.dirname(__file__), "data", "analysis_reference.npz"))
CTX = ComputeContext(0, None, CPU)
FAKE_GPU = ComputeContext(0, None, Backend("cuda", np))   # the CuPy code path, run with numpy
TAGS = ["tri", "open"]


def run(mod, data, ctx=CTX):
    data = data.copy()
    out = mod.modify(data, ctx)
    return data if out is None else out


def binned(tag):
    pos = REF[f"{tag}/pos"]
    d = DataCollection(pos, np.array(["Cu"] * len(pos)), Cell(REF[f"{tag}/cell"], np.zeros(3), REF[f"{tag}/pbc"]))
    d.particles["Mass"] = REF[f"{tag}/mass"]
    d.particles["Selection"] = REF[f"{tag}/sel"].astype(np.int8)
    return d


# ------------------------------------------------------------------------------------------------- histogram
HIST_CASES = {"plain": dict(property="Mass", bin_count=37),
              "fixed": dict(property="Mass", bin_count=10, fix_xrange=True, xrange_start=1.2, xrange_end=1.9),
              "rel": dict(property="Mass", bin_count=13, normalization="frequency"),
              "pdf": dict(property="Position.Y", bin_count=21, normalization="density"),
              "onlysel": dict(property="Position.Z", bin_count=9, only_selected=True),
              "select": dict(property="Mass", bin_count=8, select_elements=True, selection_start=1.7,
                             selection_end=1.3)}


@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("case", sorted(HIST_CASES))
def test_histogram_matches_ovito(tag, case):
    kw = HIST_CASES[case]
    d = run(Histogram(**kw), binned(tag))
    table = d.tables[f"histogram[{kw['property']}]"]
    np.testing.assert_allclose(np.column_stack(list(table.values())), REF[f"{tag}/hist/{case}/xy"], rtol=0, atol=1e-9)
    if f"{tag}/hist/{case}/num_selected" in REF:
        assert d.attributes["Histogram.num_selected"] == REF[f"{tag}/hist/{case}/num_selected"]
        np.testing.assert_array_equal(d["Selection"], REF[f"{tag}/hist/{case}/Selection"])


def test_histogram_of_bonds_and_constant_values():
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3)))
    d = run(CreateBonds(cutoff=2.8), d)
    t = run(Histogram("Topology.A", bin_count=4, operate_on="bonds"), d).tables["histogram[Topology.A]"]
    assert t["Count"].sum() == d.bonds.count
    d.particles["Constant"] = np.full(d.count, 2.0)
    t = run(Histogram("Constant", bin_count=5), d).tables["histogram[Constant]"]
    assert t["Count"].tolist() == [d.count, 0, 0, 0, 0]


# ------------------------------------------------------------------------------------------- spatial binning
BIN_COUNTS = {"X": 7, "Y": 5, "Z": 9, "XY": (4, 3), "XZ": (3, 5), "YZ": (2, 4), "XYZ": (3, 4, 5)}
BIN_CASES = [(d, op) for d in BIN_COUNTS for op in ("Mean", "Sum", "SumVol", "Min", "Max")] + \
            [("Y", "deriv"), ("Z", "deriv"), ("X", "onlysel"), ("XYZ", "unity")]


def _binning_kwargs(direction, case):
    kw = dict(property="Mass", direction=direction)
    if case == "deriv":
        return dict(kw, bin_count=6, first_derivative=True)
    if case == "onlysel":
        return dict(kw, bin_count=4, only_selected=True)
    if case == "unity":
        return dict(kw, property="", bin_count=(2, 3, 2), reduction_operation="sum")
    return dict(kw, bin_count=BIN_COUNTS[direction], reduction_operation=case.lower())


@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("direction,case", BIN_CASES)
def test_spatial_binning_matches_ovito(tag, direction, case):
    d = run(SpatialBinning(**_binning_kwargs(direction, case)), binned(tag))
    key = f"{tag}/bin/{direction}/{case}"
    if len(direction) == 1:
        table = d.tables["binning"]
        x, y = list(table.values())
        np.testing.assert_allclose(x, REF[f"{key}/x"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(y, REF[f"{key}/y"], rtol=0, atol=1e-9)
    else:
        grid = d.grids["binning"]
        values = next(iter(grid.properties.values()))
        assert values.shape == REF[f"{key}/grid"].shape
        np.testing.assert_allclose(values, REF[f"{key}/grid"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("tag", TAGS)
def test_spatial_binning_vector_property(tag):
    d = run(SpatialBinning("Position", direction="Z", bin_count=4), binned(tag))
    table = d.tables["binning"]
    assert list(table) == ["Position", "Position.X", "Position.Y", "Position.Z"]
    for comp in "XYZ":
        one = run(SpatialBinning(f"Position.{comp}", direction="Z", bin_count=4), binned(tag)).tables["binning"]
        np.testing.assert_array_equal(table[f"Position.{comp}"], one[f"Position.{comp}"])
    # OVITO 3.16 divides the mean of a vector property by its component count as well.
    np.testing.assert_allclose(np.column_stack(list(table.values())[1:]) / 3, REF[f"{tag}/bin/Z/vector/y"], atol=1e-9)


def test_spatial_binning_number_density():
    d = DataCollection.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 4, 4)))
    t = run(SpatialBinning("", "sumvol", "Z", 8), d).tables["binning"]
    np.testing.assert_allclose(t["Unity"], 4 / 3.6 ** 3)


# -------------------------------------------------------------------------------------------- reduce property
@pytest.mark.parametrize("tag", TAGS)
def test_reduce_property_matches_ovito(tag):
    checked = 0
    for key in REF.files:
        if not key.startswith(f"{tag}/reduce/"):
            continue
        which, attr = key.split("/", 3)[2:]
        _, on, op, prop = attr.split(".", 3)
        d = run(ReduceProperty(prop.split(".")[0], op.lower(), only_selected=which == "sel", operate_on=on),
                binned(tag))
        value = d.attributes[attr]
        if op in ("Nonzero", "ArgMin", "ArgMax"):
            assert value == REF[key], attr
        else:
            np.testing.assert_allclose(value, REF[key], rtol=1e-12, atol=1e-12, err_msg=attr)
        checked += 1
    assert checked == 10 * (1 + 3 + 1)


# --------------------------------------------------------------------------------- bond angle / length
def bonded():
    types = np.where(REF["bond/types"] == 1, "A", "B")
    d = DataCollection(REF["bond/pos"], types, Cell(REF["bond/cell"], np.zeros(3), np.ones(3, bool)))
    d.particles["Selection"] = REF["bond/sel"].astype(np.int8)
    d.bonds = Bonds(REF["bond/topology"], REF["bond/image"])
    d.bonds["Bond Type"] = REF["bond/bond_type"]          # 1 = "s", 2 = "l" in OVITO
    d.bonds["Selection"] = REF["bond/bond_sel"].astype(np.int8)
    return d


PARTITIONS = {"off": {}, "cos": dict(cosine_mode=True, bins=40), "ptype": dict(partition="particle_type"),
              "psel": dict(partition="particle_selection"), "btype": dict(partition="bond_type"),
              "bsel": dict(partition="bond_selection", bins=30)}


def _columns(table):
    return np.column_stack(list(table.values())[1:]), [k.removeprefix("Count").strip() for k in list(table)[1:]]


def _ovito_labels(key, labels):
    names = [str(s) for s in REF[f"bond/{key}/names"]]
    return [n.replace("s", "1").replace("l", "2") if "btype" in key else n for n in names] if labels[0] else ["Count"]


@pytest.mark.parametrize("case", sorted(PARTITIONS))
def test_bond_angle_distribution_matches_ovito(case):
    d = run(BondAngleDistribution(**PARTITIONS[case]), bonded())
    counts, labels = _columns(d.tables["bond-angle-distr"])
    ref = REF[f"bond/angle/{case}/y"].reshape(len(counts), -1)
    np.testing.assert_array_equal(counts, ref)
    assert labels == ([""] if not labels[0] else _ovito_labels(f"angle/{case}", labels))
    assert d.tables["bond-angle-distr"]["Cosine" if case == "cos" else "Angle"][0] > -1.0


@pytest.mark.parametrize("case", sorted(PARTITIONS))
def test_bond_length_distribution_matches_ovito(case):
    kw = {k: v for k, v in PARTITIONS[case].items() if k != "cosine_mode"}
    d = run(BondLengthDistribution(**kw), bonded())
    counts, labels = _columns(d.tables["bond-length-distr"])
    np.testing.assert_array_equal(counts, REF[f"bond/length/{case}/y"].reshape(len(counts), -1))
    assert labels == ([""] if not labels[0] else _ovito_labels(f"length/{case}", labels))
    d = run(BondLengthDistribution(bins=25, length_cutoff=2.6), bonded())
    np.testing.assert_array_equal(d.tables["bond-length-distr"]["Count"], REF["bond/length/cut/y"])


def test_bond_distributions_need_bonds():
    d = DataCollection.from_ase(bulk("Si", "diamond", a=5.43))
    with pytest.raises(ValueError, match="CreateBonds"):
        run(BondAngleDistribution(), d)


# ---------------------------------------------------------------------------------------------- rings
def _canon(ring):
    r = [int(v) for v in ring]
    k = r.index(min(r))
    r = r[k:] + r[:k]
    return tuple([r[0]] + r[:0:-1]) if r[1] > r[-1] else tuple(r)


def test_diamond_has_only_six_rings():
    # 3x3 cells: a chain closing through the periodic boundary needs 12 bonds, more than 8.
    d = DataCollection.from_ase(bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)))
    d = run(FindRings(3, 8), run(CreateBonds(cutoff=2.5), d))
    hist = d.tables["ring-size-histogram"]
    assert dict(zip(hist["Ring Size"].tolist(), hist["Count"].tolist())) == {3: 0, 4: 0, 5: 0, 6: 432, 7: 0, 8: 0}
    assert d.attributes["FindRings.count.6"] == 432
    np.testing.assert_array_equal(d["Ring Count"], 12)       # every atom sits on 12 six-rings
    rings = d.tables["6-rings"]["Particle Indices"]
    assert rings.shape == (432, 6) and all(r[0] == r.min() and r[1] < r[-1] for r in rings)


@pytest.mark.parametrize("tag", ["ringa", "ringb", "ringc"])
def test_rings_contain_every_ovito_ring(tag):
    """Every ring OVITO finds is a shortest-path ring; OVITO keeps fewer of the large rings that
    differ from another ring of the same size by a smaller ring (exact match on ringc)."""
    rmax = int(REF[f"{tag}/max"])
    ours = shortest_path_rings(REF[f"{tag}/topology"], len(REF[f"{tag}/pos"]), rmax)
    theirs = {_canon(r) for key in REF.files if key.startswith(f"{tag}/") and key.endswith("-rings")
              for r in REF[key]}
    assert theirs <= ours
    hist = REF[f"{tag}/hist"]
    for size, count in hist:
        mine = sum(len(r) == size for r in ours)
        assert mine >= count
        if size <= 6:
            assert mine == count, size
    if tag == "ringc":
        assert ours == theirs


def test_rings_through_modifier_match_function():
    d = DataCollection(REF["ringc/pos"], None, Cell(REF["ringc/cell"], np.zeros(3), np.ones(3, bool)))
    d.bonds = Bonds(REF["ringc/topology"], REF["ringc/image"])
    d = run(FindRings(4, 8), d)
    for size, count in REF["ringc/hist"]:
        if size >= 4:
            assert len(d.tables[f"{size}-rings"]["Particle Indices"]) == count


# ------------------------------------------------------------------------------------------- Steinhardt
def steinhardt_system(tag):
    return DataCollection(REF[f"st/{tag}/pos"], None, Cell(REF[f"st/{tag}/cell"], np.zeros(3), np.ones(3, bool)))


@pytest.mark.parametrize("ctx", [CTX, FAKE_GPU], ids=["cpu", "cuda-path"])
@pytest.mark.parametrize("tag", ["fcc", "bcc", "hcp", "liquid"])
@pytest.mark.parametrize("neighbors", ["k12", "k8", "rc"])
def test_steinhardt_matches_freud(ctx, tag, neighbors):
    kw = {"k12": dict(num_neighbors=12), "k8": dict(num_neighbors=8),
          "rc": dict(cutoff=2.6 if tag == "bcc" else 3.2)}[neighbors]
    for average in (False, True):
        d = run(SteinhardtOrderParameters((4, 6, 8), average=average, wl=True, **kw), steinhardt_system(tag), ctx)
        key = f"st/{tag}/{neighbors}/{'avg' if average else 'plain'}"
        q = np.column_stack([d[f"Steinhardt Q{l}"] for l in (4, 6, 8)])
        w = np.column_stack([d[f"Steinhardt W{l}"] for l in (4, 6, 8)])
        np.testing.assert_allclose(q, REF[f"{key}/Q"], atol=5e-5)      # freud computes in float32
        np.testing.assert_allclose(w, REF[f"{key}/W"], atol=5e-5)
    d = run(SteinhardtOrderParameters(6, wl=True, wl_normalize=False, **kw), steinhardt_system(tag), ctx)
    np.testing.assert_allclose(d["Steinhardt W6"], REF[f"st/{tag}/{neighbors}/raw/W6"], atol=5e-6)


@pytest.mark.parametrize("structure,k,q4,q6", [("fcc", 12, 0.19094, 0.57452), ("bcc", 8, 0.50918, 0.62854),
                                               ("sc", 6, 0.76376, 0.35355)])
def test_steinhardt_perfect_lattices(structure, k, q4, q6):
    atoms = bulk("Po", "sc", a=3.0) if structure == "sc" else bulk("Fe", structure, a=3.0, cubic=True)
    d = run(SteinhardtOrderParameters(num_neighbors=k, wl=True), DataCollection.from_ase(atoms.repeat((3, 3, 3))))
    np.testing.assert_allclose(d["Steinhardt Q4"], q4, atol=1e-5)
    np.testing.assert_allclose(d["Steinhardt Q6"], q6, atol=1e-5)
    assert d.attributes["SteinhardtOrderParameters.Q6"] == pytest.approx(q6, abs=1e-5)
