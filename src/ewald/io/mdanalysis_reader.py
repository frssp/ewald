"""GROMACS XTC/TRR, CHARMM/NAMD/LAMMPS DCD and AMBER NetCDF trajectories via MDAnalysis.

Needs ``MDAnalysis`` (``pip install ewald[formats]``). These files hold coordinates only,
so atom types come from a topology: ``topology="conf.gro"`` (also .pdb, .tpr, .psf,
.data, ...), or else a file with the trajectory's stem and one of those extensions,
or ``conf.gro``/``topol.tpr`` next to it. Without one every atom is type "X".
MDAnalysis builds a frame-offset index on first open (and caches it next to the file),
so frames load lazily in any order. Lengths are Angstrom; velocities are written as
MDAnalysis gives them (Angstrom/ps).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader
from .xdatcar import species_name

FORMATS = {".xtc": "XTC", ".trr": "TRR", ".dcd": "DCD", ".nc": "NCDF", ".ncdf": "NCDF"}
TOPOLOGY_EXTENSIONS = (".tpr", ".gro", ".pdb", ".psf", ".prmtop", ".parm7", ".data", ".mol2", ".xyz")


def _mda():
    try:
        import MDAnalysis
    except ImportError:
        raise ImportError("XTC/TRR/DCD files need MDAnalysis: pip install MDAnalysis "
                          "(or ewald[formats])") from None
    return MDAnalysis


def is_md_trajectory(path) -> bool:
    return os.path.splitext(str(path))[1].lower() in FORMATS


def find_topology(path) -> str | None:
    """A topology file for trajectory ``path``: same stem first, then GROMACS defaults."""
    stem, _ = os.path.splitext(str(path))
    folder = os.path.dirname(str(path)) or "."
    candidates = [stem + ext for ext in TOPOLOGY_EXTENSIONS]
    candidates += [os.path.join(folder, n) for n in ("topol.tpr", "conf.gro", "confout.gro", "start.gro")]
    return next((c for c in candidates if os.path.isfile(c)), None)


class MDAnalysisReader(FrameReader):
    """Coordinate trajectory (XTC, TRR, DCD, NetCDF) with an optional topology, lazy."""

    def __init__(self, path, topology: str | None = None, format: str | None = None):
        mda = _mda()
        self.path = str(path)
        self.topology = topology or find_topology(path)
        fmt = format or FORMATS.get(os.path.splitext(self.path)[1].lower())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # MDAnalysis warns about guessed masses, missing elements, ...
            if self.topology:
                self.universe = mda.Universe(self.topology, self.path, format=fmt)
            else:
                reader = mda.coordinates.core.get_reader_for(self.path, format=fmt)(self.path)
                self.universe = mda.Universe.empty(reader.n_atoms, trajectory=False)
                self.universe.trajectory = reader
        self.types = self._types()

    def _types(self):
        atoms = self.universe.atoms
        for attr in ("elements", "types", "names"):
            values = getattr(atoms, attr, None)
            if values is not None and len(values) and all(str(v).strip() for v in values):
                if attr == "types" and all(str(v).isdigit() for v in values):
                    return np.asarray(values).astype(str)  # LAMMPS numeric types stay numbers
                return np.array([species_name(str(v)) for v in values])
        return np.full(len(atoms), "X")

    def __len__(self):
        return len(self.universe.trajectory)

    def read(self, index: int) -> DataCollection:
        ts = self.universe.trajectory[index]
        cell = Cell()
        if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0):
            cell = Cell(np.asarray(ts.triclinic_dimensions, float), np.zeros(3), np.ones(3, bool))
        data = DataCollection(np.asarray(ts.positions, float), self.types, cell)
        data.particles["Particle Identifier"] = np.arange(1, ts.n_atoms + 1)
        if ts.has_velocities:
            data.particles["Velocity"] = np.asarray(ts.velocities, float)
        if ts.has_forces:
            data.particles["Force"] = np.asarray(ts.forces, float)
        data.attributes["Timestep"] = int(ts.data.get("step", ts.frame))
        data.attributes["Time"] = float(ts.time)
        return data


def write_md_trajectory(path, frames, format: str | None = None):
    """Stream frames to XTC/TRR/DCD with MDAnalysis (positions, box; TRR also velocity/force).

    These formats store a lower-triangular box, so a cell in another orientation is
    rotated rigidly (positions, velocities and forces with it) and the origin moved to 0."""
    mda = _mda()
    from MDAnalysis.lib.mdamath import triclinic_box, triclinic_vectors

    fmt = format or FORMATS.get(os.path.splitext(str(path))[1].lower())
    writer = None
    try:
        for t, data in enumerate(frames):
            if writer is None:
                n = data.count
                writer = mda.Writer(os.fspath(path), n_atoms=n, format=fmt)
                u = mda.Universe.empty(n, trajectory=True, velocities="Velocity" in data,
                                       forces="Force" in data)
            ts = u.trajectory.ts
            if data.cell.is_valid:
                # GROMACS/CHARMM boxes are lower-triangular; rotate the frame into that orientation
                box = triclinic_box(*data.cell.matrix)
                rotation = np.linalg.solve(data.cell.matrix, triclinic_vectors(box))
                ts.positions = (data.positions - data.cell.origin) @ rotation
                ts.dimensions = box
            else:
                rotation = np.eye(3)
                ts.positions = data.positions
                ts.dimensions = None
            if ts.has_velocities and "Velocity" in data:
                ts.velocities = data["Velocity"] @ rotation
            if ts.has_forces and "Force" in data:
                ts.forces = data["Force"] @ rotation
            ts.data["step"] = int(data.attributes.get("Timestep", t))
            ts.time = float(data.attributes.get("Time", t))
            writer.write(u.atoms)
    finally:
        if writer is not None:
            writer.close()
