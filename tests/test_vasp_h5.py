"""HDF5 readers: VASP 6 vaspout.h5 / vaspwave.h5 and MDTraj/OpenMM HDF5.

Fixtures come from tests/data/vasp/h5/make_vasp_h5.py (py4vasp layout) and
tests/data/mdtraj/make_mdtraj_h5.py; when py4vasp / mdtraj are installed the same
files are read with them and compared.
"""
import os

import numpy as np
import pytest

pytest.importorskip("h5py")

import ewald  # noqa: E402
from ewald import plugins  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
MD = os.path.join(DATA, "vasp", "h5", "md", "vaspout.h5")
RELAX = os.path.join(DATA, "vasp", "h5", "relax", "vaspout.h5")
WAVE = os.path.join(DATA, "vasp", "h5", "relax", "vaspwave.h5")
WATER = os.path.join(DATA, "mdtraj", "water.h5")


def test_detection():
    assert plugins.find_reader(MD).name == "vaspout"
    assert plugins.find_reader(WAVE).name == "vaspwave"
    assert plugins.find_reader(WATER).name == "mdtraj-h5"
    assert ewald.import_file(MD, format="vasp-h5").num_frames == 3


def test_vaspout_md_frames():
    pipe = ewald.import_file(MD)
    assert pipe.num_frames == 3
    d = pipe.compute(1)
    assert list(d.types) == ["Fe", "Fe", "O", "O"]
    assert d.attributes["Timestep"] == 2
    assert d.attributes["Free Energy"] == pytest.approx(-30.05)
    assert d.attributes["Temperature"] == pytest.approx(320.0)
    assert d.attributes["Total Energy"] == pytest.approx(-29.93)
    assert d.particles["Force"].shape == (4, 3)
    assert d.particles["Velocity"].shape == (4, 3)
    np.testing.assert_allclose(d.particles["Magnetic Moment"], [3.6, -3.6, 0, 0])
    assert "potential" not in d.grids
    last = pipe.compute(-1)
    grid = last.grids["potential"]
    assert grid.shape == (4, 5, 6)
    assert set(grid.names()) >= {"Potential", "Potential Magnetization", "Hartree Potential"}
    thermo = pipe.source.thermo.table
    np.testing.assert_allclose(thermo["Temperature"], [300, 320, 340])


def test_vaspout_relax_cell_scale_and_stress():
    pipe = ewald.import_file(RELAX)
    first, last = pipe.compute(0), pipe.compute(-1)
    np.testing.assert_allclose(first.cell.matrix[0], [0, 2.7 * 1.01, 2.7 * 1.01])
    np.testing.assert_allclose(last.cell.matrix[0], [0, 2.72 * 1.01, 2.72 * 1.01])
    assert first.attributes["Energy"] == pytest.approx(-10.805)
    assert first.attributes["Pressure"] == pytest.approx(5.0)
    assert last.attributes["Pressure"] == pytest.approx(2.0)
    s = last.attributes["Stress"]  # ASE convention: -stress(kB) in eV/A^3, xx yy zz yz xz xy
    assert s[0] < 0 and s[2] < s[0] and np.allclose(s[3:], 0)


def test_vaspwave_charge_density():
    data = ewald.import_file(WAVE).compute(0)
    grid = data.grids["charge-density"]
    assert grid.shape == (4, 6, 8)
    assert grid.integrate("Charge Density") == pytest.approx(8.0 + 8.0 / grid["Charge Density"].size)
    assert "Magnetization" in grid and "Spin Up" in grid
    assert list(data.types) == ["Si", "Si"]


def test_vaspwave_needs_structure(tmp_path):
    import shutil

    shutil.copy(WAVE, tmp_path / "vaspwave.h5")
    with pytest.raises(FileNotFoundError):
        ewald.import_file(tmp_path / "vaspwave.h5")
    data = ewald.import_file(tmp_path / "vaspwave.h5", structure=RELAX).compute(0)
    assert "charge-density" in data.grids


def test_vaspout_matches_py4vasp():
    py4vasp = pytest.importorskip("py4vasp")
    calc = py4vasp.Calculation.from_path(os.path.dirname(MD))
    pipe = ewald.import_file(MD)
    for i in range(pipe.num_frames):
        d = pipe.compute(i)
        np.testing.assert_allclose(d.positions, calc.structure[i].to_ase().positions)
        np.testing.assert_allclose(d.particles["Force"], calc.force[i].read()["forces"])
        np.testing.assert_allclose(d.particles["Velocity"], calc.velocity[i].read()["velocities"])
        np.testing.assert_allclose(d.particles["Magnetic Moment"], calc.local_moment[i].read()["total"].sum(axis=-1))
    pot = calc.potential.read()
    grid = pipe.compute(-1).grids["potential"]
    np.testing.assert_allclose(grid["Potential"], pot["total"])
    np.testing.assert_allclose(grid["Potential"] + grid["Potential Magnetization"], pot["total_up"])
    relax = py4vasp.Calculation.from_path(os.path.dirname(RELAX))
    d = ewald.import_file(RELAX).compute(0)
    np.testing.assert_allclose(d.cell.matrix, relax.structure[0].to_ase().cell[:])
    w = ewald.import_file(WAVE).compute(0)
    np.testing.assert_allclose(w.grids["charge-density"]["Charge Density"] * w.cell.volume,
                               relax.density.read()["charge"])


def test_mdtraj_h5():
    pipe = ewald.import_file(WATER)
    assert pipe.num_frames == 3
    d = pipe.compute(2)
    assert list(d.types) == ["O", "H", "H"] * 2
    np.testing.assert_allclose(d.particles["Molecule Identifier"], [1, 1, 1, 2, 2, 2])
    np.testing.assert_allclose(np.linalg.norm(d.cell.matrix, axis=1), [10, 12, 15], rtol=1e-6)
    np.testing.assert_allclose(d.particles["Velocity"], 0.005, rtol=1e-6)  # 0.5 nm/ps = 0.005 A/fs
    assert d.attributes["Time"] == pytest.approx(4.0)
    assert d.attributes["Temperature"] == pytest.approx(302.0)
    assert d.attributes["Kinetic Energy"] == pytest.approx(12.0 * 0.010364269656, rel=1e-6)
    assert d.bonds.topology.tolist() == [[0, 1], [0, 2], [3, 4], [3, 5]]


def test_mdtraj_h5_matches_mdtraj():
    md = pytest.importorskip("mdtraj")
    traj = md.load(WATER)
    pipe = ewald.import_file(WATER)
    for i in range(len(traj)):
        d = pipe.compute(i)
        np.testing.assert_allclose(d.positions, traj.xyz[i] * 10, atol=1e-5)
        np.testing.assert_allclose(d.cell.matrix, traj.unitcell_vectors[i] * 10, atol=1e-4)
