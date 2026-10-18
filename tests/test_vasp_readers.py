"""VASP readers checked against the hand-made fixtures in tests/data/vasp."""
import gzip
import shutil

import numpy as np
import pytest

from ewald import import_file
from ewald.io import XdatcarReader, open_reader
from ewald.io.xdatcar import is_xdatcar, species_name


def test_poscar_vasp5_direct(vasp_dir):
    d = import_file(vasp_dir / "POSCAR").compute(0)
    assert d.count == 8 and set(d.types) == {"Si"}
    np.testing.assert_allclose(d.cell.matrix, np.eye(3) * 5.43)
    np.testing.assert_allclose(d.positions[4], [1.3575] * 3)


def test_poscar_vasp4_takes_species_from_comment(vasp_dir):
    d = import_file(vasp_dir / "POSCAR.vasp4").compute(0)
    assert list(d.types) == ["Ga", "As"]
    np.testing.assert_allclose(d.positions[1], [1.4125] * 3)


def test_poscar_selective_dynamics_cartesian_scaled(vasp_dir):
    d = import_file(vasp_dir / "POSCAR.selective").compute(0)
    np.testing.assert_allclose(d.cell.matrix.diagonal(), [3.3, 3.3, 11.0])
    np.testing.assert_allclose(d.positions[1], [1.65, 1.65, 2.2])  # Cartesian coords are scaled too
    np.testing.assert_array_equal(d["Selective Dynamics"],
                                  [[False, False, False], [True, True, True], [False, False, True]])


def test_poscar_negative_scale_is_volume(vasp_dir):
    d = import_file(vasp_dir / "POSCAR.negscale").compute(0)
    assert d.cell.volume == pytest.approx(64.0)


def test_contcar_velocities_in_angstrom_per_fs(vasp_dir):
    d = import_file(vasp_dir / "CONTCAR").compute(0)
    assert list(d.types) == ["O", "H", "H"]
    np.testing.assert_allclose(d["Velocity"], np.diag([1e-3, 2e-3, 3e-3]), atol=1e-12)
    assert "Momentum" not in d


def test_contcar_vasp64_species_hash(vasp_dir):
    assert list(import_file(vasp_dir / "CONTCAR.vasp64").compute(0).types) == ["Fe", "O"]


@pytest.mark.parametrize("label,name", [("Fe_pv/2c5c46b5", "Fe"), ("Si_GW", "Si"), ("O", "O"),
                                        ("H.75", "H"), ("Mg/a1b2", "Mg")])
def test_species_name(label, name):
    assert species_name(label) == name


def test_outcar_trajectory_energy_forces_stress(vasp_dir):
    pipe = import_file(vasp_dir / "OUTCAR")
    assert pipe.num_frames == 2
    d0, d1 = pipe.compute(0), pipe.compute(1)
    assert list(d0.types) == ["Si", "O", "O"]
    assert d0.attributes["Energy"] == pytest.approx(-20.45)  # energy(sigma->0)
    assert d1.attributes["Energy"] == pytest.approx(-20.55)
    np.testing.assert_allclose(d0["Force"][0], [0.1, 0, 0])
    np.testing.assert_allclose(d1.positions[1], [1.49, 0, 0])
    assert np.shape(d0.attributes["Stress"]) == (6,)


def test_vasprun_variable_cell(vasp_dir):
    pipe = import_file(vasp_dir / "vasprun.xml")
    assert pipe.num_frames == 2
    d = pipe.compute(1)
    np.testing.assert_allclose(d.cell.matrix.diagonal(), [5.1, 5.0, 5.0])
    np.testing.assert_allclose(d.positions[1], [0.298 * 5.1, 0, 0])
    np.testing.assert_allclose(d["Force"][1], [-0.005, 0, 0])
    assert d.attributes["Energy"] == pytest.approx(-20.55)


def test_xdatcar_fixed_cell_fixture(vasp_dir):
    pipe = import_file(vasp_dir / "XDATCAR")
    assert pipe.num_frames == 3
    d = pipe.compute(2)
    assert d.attributes["Timestep"] == 3
    np.testing.assert_allclose(d.positions[0], [0.03 * 5.43, 0, 0])


def test_xdatcar_npt_with_vasp64_labels(vasp_dir):
    pipe = import_file(vasp_dir / "XDATCAR.npt")
    assert pipe.num_frames == 3
    d = pipe.compute(-1)
    assert list(d.types) == ["Mg", "O"]
    assert d.attributes["Timestep"] == 30
    np.testing.assert_allclose(d.cell.matrix, np.eye(3) * 4.2)
    np.testing.assert_allclose(d.positions[1], [2.1] * 3)


def test_xdatcar_truncated_last_frame_is_skipped(vasp_dir, tmp_path):
    text = (vasp_dir / "XDATCAR").read_text().splitlines(keepends=True)
    path = tmp_path / "XDATCAR"
    path.write_text("".join(text[:-3]))  # job killed while writing frame 3
    with pytest.warns(UserWarning, match="incomplete"):
        pipe = import_file(path)
    assert pipe.num_frames == 2


def test_xdatcar_detected_by_content_and_gzip(vasp_dir, tmp_path):
    plain = tmp_path / "md_run.traj"
    shutil.copy(vasp_dir / "XDATCAR", plain)
    assert is_xdatcar(plain)
    assert isinstance(open_reader(plain), XdatcarReader)
    gz = tmp_path / "XDATCAR.gz"
    gz.write_bytes(gzip.compress((vasp_dir / "XDATCAR").read_bytes()))
    assert import_file(gz).num_frames == 3
    assert not is_xdatcar(vasp_dir / "POSCAR")


def test_crlf_line_endings(vasp_dir, tmp_path):
    for name in ("XDATCAR", "POSCAR"):
        lf = (vasp_dir / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(lf.replace(b"\n", b"\r\n"))
    assert import_file(tmp_path / "XDATCAR").num_frames == 3
    assert import_file(tmp_path / "POSCAR").compute(0).count == 8
