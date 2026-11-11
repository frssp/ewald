"""Units as labels (design review B2, D105): numbers stay as read, results say what they are in."""
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import ewald.examples as examples
from ewald import DataCollection, Cell, load, msd, rdf, vacf
from ewald import units
from ewald.plotstyle import axis_label
from ewald.verbs import Table
from ewald.views import compare_data, figure

LAMMPS = Path(__file__).parent / "data" / "lammps"


def test_unit_strings_per_system():
    assert units.unit("length") == "Å" and units.unit("length^2") == "Å$^2$" and units.unit("1/length") == "Å$^{-1}$"
    assert units.unit("energy", "real") == "kcal/mol" and units.unit("length^2", "nano") == "nm$^2$"
    assert units.unit("length^2", "lj") == "σ$^2$" and units.unit("charge^2 length^2") == "e$^2$ Å$^2$"
    assert units.unit("energy") == units.unit("energy", "metal") == "eV"
    assert units.time_unit(0.0) == units.time_unit(None) == "frames" and units.time_unit(0.002) == "ps"
    assert units.per_time("Å$^2$", 0) == "Å$^2$/frame" and units.per_time("Å$^2$", 0.5) == "Å$^2$/ps"
    assert units.check_style(" Real ") == "real" and units.check_style(None) is None
    with pytest.raises(ValueError, match="units='kcal' is not a LAMMPS unit style; use one of metal, real, si, cgs, electron, micro, nano, lj"):
        units.check_style("kcal")


def test_labels_without_a_unit_system_are_what_they_were():
    """Files without a unit system (VASP, extxyz, ...) keep the labels they always had."""
    for name, label in (("r", "$r$ (Å)"), ("g(r) Cu-Zr", "$g(r)$"), ("MSD Cu", "MSD (Å$^2$)"), ("q", "$q$ (Å$^{-1}$)"),
                        ("Energy", "Energy (eV)"), ("Count Cu", "Count"), ("Temp", "Temp")):
        assert axis_label(name) == label
        assert axis_label(name, units.column_unit(name)) == label, name
    assert axis_label("r", "nm") == "$r$ (nm)" and axis_label("Energy", "kcal/mol") == "Energy (kcal/mol)"
    assert axis_label("Time", "ps") == "Time (ps)" and axis_label("Time", "frames") == "Time (frames)"
    assert axis_label("VACF (normalized)", "") == "VACF (normalized)"          # a name with words in brackets
    assert axis_label("Charge MSD") == "Charge MSD (e$^2$ Å$^2$)"              # it was labeled Å² alone


def test_msd_without_a_time_step_says_frames():
    pipe = examples.load("Cu_md.extxyz")
    d = msd(pipe)
    D = "MeanSquaredDisplacement.diffusion_coefficient"
    assert d.units["Time"] == "frames" and d.units[D] == "Å$^2$/frame" and d.units["MSD"] == "Å$^2$"
    assert d.units["MeanSquaredDisplacement.fit_start"] == "frames"
    assert np.array_equal(d["Time"], np.arange(len(d["Time"])))               # the numbers stay frame lags
    ax = figure(pipe, table="msd").axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (frames)", "MSD (Å$^2$)")
    timed = msd(pipe, time_per_frame=0.002)
    assert timed.units["Time"] == "ps" and timed.units[D] == "Å$^2$/ps"
    assert np.allclose(timed["Time"], d["Time"] * 0.002)                       # the same numbers as before
    assert figure(pipe, table="msd").axes[0].get_xlabel() == "Time (ps)"
    assert pipe.compute(0).units[("msd", "Time")] == "ps"


def test_vacf_labels_d_only_when_it_came_from_positions():
    pipe = examples.load("Cu_md.extxyz")
    v = vacf(pipe, velocities="positions")
    assert v.units["Time"] == "frames" and v.units["VelocityAutocorrelation.diffusion_coefficient"] == "Å$^2$/frame"
    assert v.units["D(t)"] == "Å$^2$/frame"
    data = pipe.compute(0)
    moving = DataCollection(data.positions, data.types, data.cell)
    moving["Velocity"] = np.ones((data.count, 3))
    from ewald.io import StaticSource
    from ewald.pipeline import Pipeline

    p = Pipeline(StaticSource([moving, moving.copy()]))
    w = vacf(p, velocities="property")
    assert w.units["Time"] == "frames" and "VelocityAutocorrelation.diffusion_coefficient" not in w.units


def test_lammps_units_from_the_file_and_from_load(tmp_path):
    dump = load(LAMMPS / "dump.custom.lammpstrj")                              # ITEM: UNITS metal
    assert dump.compute(0).attributes["Units"] == "metal" and units.system(dump.compute(1)) == "metal"
    data = load(LAMMPS / "data.full.lmp", units="real")
    assert data.compute(0).attributes["Units"] == "real"
    assert units.table_units(data.compute(0), "x", ["r", "Energy"]) == {"r": "Å", "Energy": "kcal/mol"}
    plain = tmp_path / "dump.lammpstrj"
    text = (LAMMPS / "dump.custom.lammpstrj").read_text(encoding="utf-8")
    plain.write_text(text.replace("ITEM: UNITS\nmetal\n", ""), encoding="utf-8", newline="\n")
    assert "Units" not in load(plain).compute(0).attributes
    nano = load(plain, units="nano")
    first = nano.compute(0)
    assert first.attributes["Units"] == "nano" and np.allclose(first.positions, load(plain).compute(0).positions)
    r = rdf(nano, cutoff=3.0)
    assert r.units["r"] == "nm" and figure(nano, table="coordination-rdf").axes[0].get_xlabel() == "$r$ (nm)"
    with pytest.warns(UserWarning, match="says ITEM: UNITS metal, but units='real' was given"):
        told = load(LAMMPS / "dump.custom.lammpstrj", units="real")
    assert told.compute(0).attributes["Units"] == "real"
    with pytest.raises(ValueError, match="not a LAMMPS unit style"):
        load(plain, units="angstrom")
    with pytest.raises(TypeError, match="unexpected keyword argument 'units'"):
        load(examples.path("Cu_md.extxyz"), units="real")                     # only LAMMPS files take it


def test_lammps_lengths_in_nm_label_the_msd_in_nm(tmp_path):
    plain = tmp_path / "dump.lammpstrj"
    text = (LAMMPS / "dump.custom.lammpstrj").read_text(encoding="utf-8")
    plain.write_text(text.replace("ITEM: UNITS\nmetal\n", ""), encoding="utf-8", newline="\n")
    d = msd(load(plain, units="nano"), time_per_frame=0.0)
    assert d.units["MSD"] == "nm$^2$" and d.units["MeanSquaredDisplacement.diffusion_coefficient"] == "nm$^2$/frame"


def test_a_verb_table_beside_an_earlier_one_keeps_both_units():
    from ewald.modifiers import MeanSquaredDisplacement

    pipe = examples.load("Cu_md.extxyz")
    pipe.add(MeanSquaredDisplacement(time_per_frame=0.002))
    d = msd(pipe)                                                             # frames, beside the user's ps step
    assert d.name == "msd (msd)" and d.units["Time"] == "frames"
    data = pipe.compute(0)
    assert data.units[("msd", "Time")] == "ps" and data.units[("msd (msd)", "Time")] == "frames"


def test_units_travel_with_copies_and_tables():
    data = DataCollection(np.zeros((1, 3)), ["Cu"], Cell(np.eye(3)))
    data.units[("t", "Time")] = "ps"
    assert data.copy().units == {("t", "Time"): "ps"} and data.copy().units is not data.units
    t = Table("t", {"Time": [0.0, 1.0], "MSD": [0.0, 1.0]})
    assert t.units == {"MSD": "Å$^2$"}                                         # from the names alone
    t2 = Table("t", {"Time": [0.0]}, units={"Time": "ps"})
    import copy

    assert copy.copy(t2).units == {"Time": "ps"}
    title, x, series = compare_data(t2, y="Time") if False else ("t", "Time", [])
    data.tables["t"] = {"Time": np.arange(3.0), "MSD": np.arange(3.0)}
    title, x, series = compare_data(data, table="t")
    assert series[0][4] == {"Time": "ps", "MSD": "Å$^2$"}


def test_arrhenius_labels_d_with_the_unit_of_the_runs(tmp_path):
    from test_runs import write_vasp_md

    from ewald import arrhenius, load_runs

    for t in (600, 900, 1200):
        write_vasp_md(tmp_path / f"T{t}", t, incar=False)
    runs = load_runs(str(tmp_path / "T*" / "XDATCAR"), label=r"T(\d+)")
    fit = arrhenius(msd(runs))
    assert fit.name == "$D$ (Å$^2$/frame)"
