"""Regenerate the example files in ``files/`` (``python -m ewald.examples.make_examples``).

The structures are textbook crystals built with ASE; the trajectories, forces and the charge
density are synthetic (seeded random displacements, harmonic forces, Gaussians on atoms and
bond centres), so the files stay small and need no simulation code. ``Cu3Au_band.yaml`` is a
copy of the phonopy fixture in ``tests/data/phonon/`` (see ``make_phonon_reference.py``).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.cluster import Octahedron

from ewald import DataCollection, export_file
from ewald.data import VoxelGrid

HERE = Path(__file__).resolve().parent
OUT = HERE / "files"


def _frames(base: DataCollection, n: int, amp, rng, forces=False, k=2.0):
    """``n`` copies of ``base`` with growing random displacements (a fake heating run)."""
    frames = []
    for i in range(n):
        f = base.copy()
        du = rng.normal(0.0, amp * (1 + i) / n, base.positions.shape)
        f.positions = base.positions + du
        f.attributes["Timestep"] = 100 * i
        if forces:
            f.particles["Force"] = -k * du
            f.attributes["Energy"] = float(-3.54 * base.count + 0.5 * k * np.sum(du ** 2))
        frames.append(f)
    return frames


def main(out: Path = OUT):
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    # CIF: diamond Si, conventional cell (8 atoms)
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    export_file(DataCollection.from_ase(si), out / "Si.cif", format="cif")

    # POSCAR: cubic perovskite SrTiO3, 2x2x2 (40 atoms); try Polyhedra on Ti
    sto = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
                cell=np.eye(3) * 3.905, pbc=True).repeat((2, 2, 2))
    export_file(DataCollection.from_ase(sto), out / "POSCAR.SrTiO3", format="vasp")

    # plain XYZ: Pt truncated octahedron nanoparticle (no cell)
    pt = Octahedron("Pt", length=7, cutoff=2, latticeconstant=3.924)
    pt.center()
    pt.pbc = False
    export_file(DataCollection.from_ase(pt), out / "Pt_nanoparticle.xyz", format="xyz")

    # extended XYZ trajectory: fcc Cu 3x3x3 (108 atoms), 10 frames with forces and energy
    cu = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3)))
    export_file(_frames(cu, 10, 0.15, rng, forces=True), out / "Cu_md.extxyz")

    # XDATCAR: diamond Si 2x2x2 (64 atoms), 10 frames
    si64 = DataCollection.from_ase(si.repeat((2, 2, 2)))
    export_file(_frames(si64, 10, 0.12, rng), out / "XDATCAR.Si", format="vasp-xdatcar")

    # LAMMPS dump: Cu block with a spherical void, 5 frames, with an element column
    cu5 = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((5, 5, 5)))
    centre = cu5.cell.matrix.diagonal() / 2
    cu5 = cu5.delete(np.linalg.norm(cu5.positions - centre, axis=1) < 5.0)
    cu5.particles["Particle Identifier"] = np.arange(1, cu5.count + 1)
    export_file(_frames(cu5, 5, 0.12, rng), out / "Cu_void.lammpstrj",
                columns=("id", "type", "element", "x", "y", "z"))

    # LAMMPS data: B2 CuZr 4x4x4 (128 atoms), types named in Masses comments
    cuzr = bulk("CuZr", "cesiumchloride", a=3.26).repeat((4, 4, 4))
    export_file(DataCollection.from_ase(cuzr), out / "CuZr.data", atom_style="atomic")

    # CHGCAR: diamond Si with a synthetic valence density (atoms + bond charges), 16^3 grid
    n = 16
    cell = si.cell.array
    frac = np.stack(np.meshgrid(*[np.arange(n) / n] * 3, indexing="ij"), axis=-1)
    atoms_f = si.get_scaled_positions()
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    centres = []
    for i, j in pairs:  # bond midpoints: nearest-image pairs 2.35 A apart
        d = atoms_f[j] - atoms_f[i]
        d -= np.round(d)
        if abs(np.linalg.norm(d @ cell) - 2.352) < 0.05:
            centres.append(atoms_f[i] + d / 2)
    rho = np.zeros((n, n, n))
    for sites, width, weight in ((atoms_f, 0.55, 1.0), (np.array(centres), 0.45, 0.8)):
        for s in sites:
            d = frac - s
            d -= np.round(d)
            r2 = np.sum((d @ cell) ** 2, axis=-1)
            rho += weight * np.exp(-r2 / (2 * width ** 2))
    rho *= 32.0 / (rho.mean() * si.get_volume())  # 8 atoms x 4 valence electrons
    data = DataCollection.from_ase(si)
    grid = VoxelGrid(data.cell.copy())
    grid["Charge Density"] = rho
    data.grids["charge-density"] = grid
    export_file(data, out / "CHGCAR.Si", format="chgcar")

    # phonopy band.yaml with eigenvectors (L1_2 Cu3Au): PhononMode animates a mode
    fixture = HERE.parents[2] / "tests" / "data" / "phonon" / "band.yaml"
    if fixture.exists():
        shutil.copyfile(fixture, out / "Cu3Au_band.yaml")


if __name__ == "__main__":
    main()
