"""Volumetric data: VASP CHGCAR/LOCPOT/ELFCAR and cube readers, isosurfaces, slices, differences."""
import gzip

import numpy as np
import pytest

from ewald import import_file
from ewald.data import Cell, DataCollection, VoxelGrid
from ewald.io import export_file
from ewald.modifiers import CreateIsosurface, GridDifference, GridSlice

SIGMA = 0.7
SHAPE = (24, 24, 30)
LATTICE = np.array([[8.0, 0.0, 0.0], [1.0, 8.0, 0.0], [0.0, 0.0, 10.0]])  # triclinic on purpose
CENTERS = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])


def gaussians(amplitude=2.0, centers=CENTERS, shape=SHAPE, lattice=LATTICE):
    """Periodic sum of Gaussians (e/A^3) on the VASP grid points i/N."""
    f = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), -1)
    rho = np.zeros(shape)
    for c in centers:
        d = f - c
        d -= np.round(d)
        rho += amplitude * np.exp(-((d @ lattice) ** 2).sum(-1) / (2 * SIGMA ** 2))
    return rho


def write_vasp(path, blocks, per_volume=True, per_line=5, augmentation=True, vasp4=False):
    vol = abs(np.linalg.det(LATTICE))
    with open(path, "w") as fh:
        fh.write("Si O\n   1.00000000000000\n")
        for v in LATTICE:
            fh.write(" ".join(f"{x:12.6f}" for x in v) + "\n")
        if not vasp4:
            fh.write("   Si   O\n")
        fh.write("     1     1\nDirect\n  0.500000  0.500000  0.500000\n  0.000000  0.000000  0.000000\n")
        for k, block in enumerate(blocks):
            fh.write("\n" + " ".join(f"{n:5d}" for n in SHAPE) + "\n")
            flat = (block * (vol if per_volume else 1.0)).transpose(2, 1, 0).ravel()
            for i in range(0, len(flat), per_line):
                fh.write(" ".join(f"{x:.11E}" for x in flat[i:i + per_line]) + "\n")
            if augmentation:
                fh.write("augmentation occupancies   1  4\n  0.1234E+00  0.2E-01 -0.3E-02  0.4E+00\n"
                         "augmentation occupancies   2  4\n  0.1234E+00  0.2E-01 -0.3E-02  0.4E+00\n")
                if k == 0 and len(blocks) > 1:
                    fh.write("  0.1000000E+01  0.2000000E+01\n")  # per-atom moments before the 2nd block


@pytest.fixture
def chgcar(tmp_path):
    rho = gaussians()
    mag = 0.3 * rho * np.where(np.arange(SHAPE[2]) < SHAPE[2] // 2, -1.0, 1.0)
    path = tmp_path / "CHGCAR"
    write_vasp(path, [rho, mag])
    return path, rho, mag


def test_read_spin_chgcar(chgcar):
    path, rho, mag = chgcar
    data = import_file(path).compute()
    assert data.count == 2 and list(data.types) == ["Si", "O"]
    np.testing.assert_allclose(data.positions[0], np.array([0.5, 0.5, 0.5]) @ LATTICE)
    grid = data.grids["charge-density"]
    assert grid.shape == SHAPE
    np.testing.assert_allclose(grid["Charge Density"], rho, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(grid["Magnetization"], mag, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(grid["Spin Up"] - grid["Spin Down"], mag, atol=1e-12)
    # two Gaussians of amplitude 2: 2 * 2 * (2 pi)^1.5 sigma^3 electrons
    assert grid.integrate("Charge Density") == pytest.approx(4 * (2 * np.pi) ** 1.5 * SIGMA ** 3, rel=1e-4)
    text = data.summary()
    assert "grid 'charge-density': 24x24x30" in text and "'Spin Up'" in text


def test_noncollinear_locpot_elfcar_vasp4_gzip(tmp_path):
    rho = gaussians()
    write_vasp(tmp_path / "CHGCAR.ncl", [rho, rho, 2 * rho, 3 * rho])
    g = import_file(tmp_path / "CHGCAR.ncl").compute().grids["charge-density"]
    assert g["Magnetization"].shape == SHAPE + (3,)
    np.testing.assert_allclose(g["Magnetization.Z"], 3 * rho, atol=1e-12)

    write_vasp(tmp_path / "LOCPOT", [-rho], per_volume=False, augmentation=False)
    g = import_file(tmp_path / "LOCPOT").compute().grids["potential"]
    np.testing.assert_allclose(g["Potential"], -rho, atol=1e-12)  # potentials are not scaled by volume

    write_vasp(tmp_path / "ELFCAR", [rho / rho.max(), rho / rho.max()], per_volume=False, per_line=10,
               augmentation=False)
    g = import_file(tmp_path / "ELFCAR").compute().grids["elf"]
    assert set(g.properties) == {"ELF", "ELF 2"} and g["ELF"].max() == pytest.approx(1.0)

    write_vasp(tmp_path / "PARCHG", [rho], vasp4=True)
    with open(tmp_path / "PARCHG", "rb") as src, gzip.open(tmp_path / "PARCHG.gz", "wb") as dst:
        dst.write(src.read())
    data = import_file(tmp_path / "PARCHG.gz").compute()
    assert list(data.types) == ["Si", "O"]  # VASP 4: species from the comment line
    np.testing.assert_allclose(data.grids["charge-density"]["Charge Density"], rho, atol=1e-12)


def test_truncated_second_block_warns(chgcar, tmp_path):
    path, rho, _ = chgcar
    lines = path.read_text().splitlines()
    (tmp_path / "CHG").write_text("\n".join(lines[:-200]) + "\n")
    with pytest.warns(UserWarning, match="incomplete"):
        g = import_file(tmp_path / "CHG").compute().grids["charge-density"]
    assert list(g.properties) == ["Charge Density"]


def test_cube_roundtrip_and_units(chgcar, tmp_path):
    path, rho, _ = chgcar
    data = import_file(path).compute()
    export_file(data, tmp_path / "rho.cube", property="Charge Density")
    back = import_file(tmp_path / "rho.cube", pbc=True).compute()
    g = back.grids["cube"]
    np.testing.assert_allclose(g.domain.matrix, LATTICE, atol=1e-5)  # Bohr -> Angstrom
    np.testing.assert_allclose(back.positions, data.positions, atol=1e-5)
    np.testing.assert_allclose(g["Value"], rho, rtol=1e-4, atol=1e-9)
    assert list(back.types) == ["Si", "O"] and g.domain.pbc.all()
    # and back to CHGCAR format (multiplied by the volume again)
    export_file(data, tmp_path / "CHGCAR_out", format="vasp-chgcar", property="Spin Up")
    again = import_file(tmp_path / "CHGCAR_out").compute().grids["charge-density"]
    np.testing.assert_allclose(again["Charge Density"], data.grids["charge-density"]["Spin Up"], rtol=1e-9)


def test_isosurface_area_matches_sphere(chgcar):
    path, rho, _ = chgcar
    level = 0.5
    pipe = import_file(path)
    pipe.modifiers.append(CreateIsosurface(level))
    data = pipe.compute()
    surf = data.surfaces["isosurface"]
    r = SIGMA * np.sqrt(2 * np.log(2.0 / level))
    # two spheres; the one on the cell corner is closed across the periodic faces
    assert surf.area == pytest.approx(2 * 4 * np.pi * r ** 2, rel=0.03)
    assert data.attributes["isosurface.isolevel"] == level
    v = surf.vertices
    assert surf.vertex_properties["Normal"].shape == v.shape and len(surf.faces) > 100
    # every vertex lies on the level set
    g = data.grids["charge-density"]
    np.testing.assert_allclose(g.interpolate("Charge Density", v[::25]), level, atol=0.05)


def test_signed_isosurface_and_auto_level(chgcar):
    path, rho, mag = chgcar
    pipe = import_file(path)
    pipe.modifiers.append(CreateIsosurface(0.1, property="Magnetization", signed=True, identifier="spin"))
    pipe.modifiers.append(CreateIsosurface(property="Spin Up"))
    data = pipe.compute()
    assert len(data.surfaces["spin"].faces) and len(data.surfaces["spin-"].faces)
    assert not np.allclose(data.surfaces["spin-"].face_colors()[0], data.surfaces["spin"].face_colors()[0])
    spin_up = 0.5 * (rho + mag)
    assert data.attributes["isosurface.isolevel"] == pytest.approx(np.abs(spin_up).mean(), rel=1e-6)
    # sources are untouched
    assert pipe.source_frame(0).surfaces == {}


def test_grid_slice_values_and_legend(chgcar):
    path, rho, _ = chgcar
    pipe = import_file(path)
    pipe.modifiers.append(GridSlice((0, 0, 1), 0.5, cmap="viridis"))
    data = pipe.compute()
    s = data.surfaces["slice"]
    frac = Cell(LATTICE).to_fractional(s.vertices)
    np.testing.assert_allclose(frac[:, 2], 0.5, atol=1e-9)
    assert frac[:, :2].min() > -1e-9 and frac[:, :2].max() < 1 + 1e-9
    assert s.area == pytest.approx(abs(np.linalg.det(LATTICE[:2, :2])), rel=1e-6)
    lo, hi = data.attributes["slice.range"]
    assert hi == pytest.approx(rho[:, :, 15].max(), rel=0.02)  # the Si Gaussian peaks in this plane
    assert data.attributes["ColorCoding.range"] == (lo, hi)
    assert s.vertex_properties["Color"].shape == s.vertices.shape
    # a diagonal (1 1 0) plane is a rectangle through the cell; log scale
    pipe.modifiers[0] = GridSlice((1, 1, 0), 1.0, scale="log")
    s2 = pipe.compute().surfaces["slice"]
    f2 = Cell(LATTICE).to_fractional(s2.vertices)
    np.testing.assert_allclose(f2[:, 0] + f2[:, 1], 1.0, atol=1e-9)
    with pytest.raises(ValueError, match="does not cut"):
        pipe.modifiers[0] = GridSlice((0, 0, 1), 1.5)
        pipe.compute()


def test_grid_difference(chgcar, tmp_path):
    path, rho, _ = chgcar
    a = gaussians(centers=CENTERS[:1])
    write_vasp(tmp_path / "CHGCAR_A", [a])
    write_vasp(tmp_path / "CHGCAR_B", [rho - a - 0.01])
    pipe = import_file(path)
    pipe.modifiers.append(GridDifference([tmp_path / "CHGCAR_A", tmp_path / "CHGCAR_B"]))
    pipe.modifiers.append(CreateIsosurface(0.005, signed=True))
    data = pipe.compute()
    np.testing.assert_allclose(data.grids["charge-density"]["Charge Density"], 0.01, atol=1e-9)
    assert pipe.source_frame(0).grids["charge-density"]["Charge Density"].max() == pytest.approx(rho.max())
    assert len(data.surfaces["isosurface"].faces) == 0  # constant +0.01 field: no crossing
    bad = DataCollection()
    bad.grids["x"] = VoxelGrid(Cell(LATTICE), {"Charge Density": np.zeros((4, 4, 4))})
    with pytest.raises(ValueError, match="does not match"):
        GridDifference(str(tmp_path / "CHGCAR_A")).modify(bad, None)


def test_raytrace_interpolates_vertex_colors():
    from ewald.raytrace import Scene, raytrace

    camera = {"position": (0, 0, 10), "focal_point": (0, 0, 0), "up": (0, 1, 0), "parallel_projection": True,
              "parallel_scale": 2.0, "view_angle": 30.0}
    sc = Scene()
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    sc.add_triangles([[-2, -2, 0], [-2, -2, 0]], [[2, -2, 0], [2, 2, 0]], [[2, 2, 0], [-2, 2, 0]], [(0.5, 0.5, 0.5)],
                     normals=np.tile([0.0, 0.0, 1.0], (2, 3, 1)),
                     vertex_colors=np.array([[red, blue, blue], [red, blue, red]]))  # red left, blue right
    img = raytrace(sc, camera, (41, 41), samples=1, ao_samples=0, shadows=False)
    assert img[20, 3, 0] > img[20, 3, 2] and img[20, 37, 2] > img[20, 37, 0]
