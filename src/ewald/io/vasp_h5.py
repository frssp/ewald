"""VASP 6 HDF5 output: ``vaspout.h5`` (structures, forces, energies, ...) and ``vaspwave.h5`` (grids).

Needs ``h5py`` (``pip install ewald[formats]``). The layout is the one py4vasp reads
(``py4vasp/_raw/definition.py``); HDF5 stores VASP's Fortran arrays with the axes
reversed, so a grid dataset is ``(component, z, y, x)``.

``vaspout.h5`` gives one frame per ionic step from ``/intermediate/ion_dynamics``
(or the final structure in ``/results/positions`` when there is no history), read on
demand, with the same names as the OUTCAR/vasprun readers: particle ``Force`` (eV/A),
``Velocity`` (A/fs, MD) and ``Magnetic Moment`` (mu_B, summed over orbitals; ``N x 3``
for non-collinear runs, VASP >= 6.5), attributes ``Energy`` (energy(sigma->0)),
``Free Energy`` (TOTEN), ``Stress`` (eV/A^3, ASE sign and order), ``Pressure`` (kB) and,
for MD, ``Temperature``, ``Kinetic Energy``, ``Total Energy``, ... ``Timestep`` is the
ionic step counted from 1. ``reader.thermo`` holds the per-step energies as a table.
The local potential (``WRT_POTENTIAL``/``LVTOT``, ``/results/potential``) goes into
``data.grids["potential"]`` of the last frame.

``vaspwave.h5`` (``LCHARGH5``, ``LH5``) holds the charge density but no structure; the
structure comes from ``vaspout.h5`` in the same folder (or ``structure=``). Like a
CHGCAR it stores rho * V_cell, so "Charge Density" is divided by the cell volume
(e/A^3); "Magnetization" is the second (collinear) or second to fourth component.
"""
from __future__ import annotations

import os

import numpy as np

from ..data import Cell, DataCollection, VoxelGrid
from .base import FrameReader
from .h5md import _h5py
from .vasp_output import _KB_TO_EV_A3, _table
from .xdatcar import species_name

_MAGIC = b"\x89HDF\r\n\x1a\n"
_DYN = "intermediate/ion_dynamics"
_FINAL = "results/positions"
# last word of a VASP energy label -> ewald attribute (OUTCAR/OSZICAR names)
_ENERGY_TAGS = {"TOTEN": "Free Energy", "EKIN": "Kinetic Energy", "EKIN_LAT": "Lattice Kinetic Energy",
                "TEIN": "Temperature", "ES": "Nose Potential", "EPS": "Nose Kinetic", "ETOTAL": "Total Energy"}
_POTENTIALS = {"total": "Potential", "hartree": "Hartree Potential", "ionic": "Ionic Potential",
               "xc": "XC Potential"}


def _hdf5(path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == _MAGIC
    except OSError:
        return False


def _has(path, *groups) -> bool:
    if not _hdf5(path):
        return False
    try:
        with _h5py().File(path, "r") as f:
            return all(g in f for g in groups)
    except (OSError, ImportError):
        return False


def is_vaspout(path) -> bool:
    return _has(path, "version", _FINAL + "/ion_types") or (
        os.path.basename(str(path)).lower().startswith("vaspout") and _hdf5(path))


def is_vaspwave(path) -> bool:
    return _has(path, "charge/charge") or (os.path.basename(str(path)).lower().startswith("vaspwave")
                                           and _hdf5(path))


def _text(x) -> str:
    return (x.decode() if isinstance(x, bytes) else str(x)).strip()


def _energy_name(label: str) -> str:
    label = " ".join(label.split())
    if label.startswith("energy(sigma->0)"):
        return "Energy"
    if label.startswith("energy without entropy"):
        return "Energy without Entropy"
    return _ENERGY_TAGS.get(label.split()[-1], label) if label else label


def _grid(values) -> np.ndarray:
    """``(component, z, y, x)`` -> ``(x, y, z)`` or ``(x, y, z, component)``."""
    values = np.asarray(values, float)
    if values.ndim == 3:
        return values.T
    out = np.moveaxis(values, 0, -1)                  # (z, y, x, c)
    return np.ascontiguousarray(np.transpose(out, (2, 1, 0, 3)))


def _add_components(grid: VoxelGrid, name: str, values, extra: str):
    """First component under ``name``; the rest (magnetic part) under ``extra``."""
    values = _grid(values)
    if values.ndim == 3:
        grid[name] = values
        return
    grid[name] = np.ascontiguousarray(values[..., 0])
    if values.shape[-1] == 2:
        grid[extra] = np.ascontiguousarray(values[..., 1])
    elif values.shape[-1] > 2:
        grid[extra] = np.ascontiguousarray(values[..., 1:4])


class VaspoutReader(FrameReader):
    """VASP 6 ``vaspout.h5``: every ionic step with forces, energies, stress, velocities, lazy."""

    def __init__(self, path, grids: bool = True):
        h5py = _h5py()
        self.path = str(path)
        self.grids = grids
        with h5py.File(self.path, "r") as f:
            if _FINAL + "/ion_types" not in f:
                raise ValueError(f"{path}: no {_FINAL}/ion_types; not a vaspout.h5?")
            names = [species_name(_text(t)) for t in f[_FINAL + "/ion_types"][()]]
            counts = np.atleast_1d(f[_FINAL + "/number_ion_types"][()]).astype(int)
            self.types = np.repeat(names, counts)
            self.version = tuple(int(f[f"version/{k}"][()]) for k in ("major", "minor", "patch")
                                 if f"version/{k}" in f)
            dyn = f.get(_DYN + "/position_ions")
            self._history = dyn is not None and dyn.ndim == 3 and dyn.shape[0] > 0
            self._n = dyn.shape[0] if self._history else 1
            if not self._history and _FINAL + "/position_ions" not in f:
                raise ValueError(f"{path}: no ionic positions")
            self._energies = {}
            if self._history and _DYN + "/energies" in f:
                tags = [_energy_name(_text(t)) for t in f[_DYN + "/energies_tags"][()]]
                values = np.atleast_2d(f[_DYN + "/energies"][()])
                self._energies = {t: values[:, i].astype(float) for i, t in enumerate(tags) if t}
            self._stress = np.asarray(f[_DYN + "/stress"][()], float) if self._history and \
                _DYN + "/stress" in f else None
            self.electronic = None
            if "results/electron_dos" in f or "results/electron_eigenvalues" in f:
                try:
                    self.electronic = self._read_electronic(f)
                except (KeyError, ValueError, IndexError) as exc:
                    import warnings

                    warnings.warn(f"{path}: DOS / eigenvalues could not be read ({exc})", stacklevel=2)
        rows = []
        for i in range(self._n):
            row = {"Step": i + 1, **{k: v[i] for k, v in self._energies.items() if i < len(v)}}
            p = self._pressure(i)
            if p is not None:
                row["Pressure"] = p
            rows.append(row)
        self.thermo = _table(rows) if self._energies or self._stress is not None else None

    def __len__(self):
        return self._n

    def _read_electronic(self, f):
        """DOS from ``results/electron_dos``, bands from ``results/electron_eigenvalues_kpoints_opt``
        (KPOINTS_OPT, the band path of a self-consistent run) or else ``results/electron_eigenvalues``."""
        from ..electronic import ElectronicStructure, line_ticks, path_distances, pretty_label

        es = ElectronicStructure(types=self.types.copy(), source=os.path.basename(self.path))
        if "results/electron_dos/dos" in f:
            g = f["results/electron_dos"]
            es.efermi = float(g["efermi"][()])
            dos = np.atleast_2d(np.asarray(g["dos"][()], float))
            es.dos_energies, es.dos = np.asarray(g["energies"][()], float), dos[:1] if len(dos) == 4 else dos
            if "dospar" in g and g["dospar"].ndim == 4:
                par = np.asarray(g["dospar"][()], float)            # (spin, atom, orbital, E)
                es.pdos = np.ascontiguousarray(np.transpose(par[:1] if len(par) == 4 else par, (1, 0, 2, 3)))
        if "results/projectors/lchar" in f:
            es.orbitals = [_text(o) for o in f["results/projectors/lchar"][()]]
        suffix = "_kpoints_opt" if "results/electron_eigenvalues_kpoints_opt/eigenvalues" in f else ""
        eig = f.get(f"results/electron_eigenvalues{suffix}")
        if eig is None or "eigenvalues" not in eig:
            return es
        if suffix and "results/electron_dos_kpoints_opt/efermi" in f and not es.has_dos:
            es.efermi = float(f["results/electron_dos_kpoints_opt/efermi"][()])
        es.eigenvalues = np.asarray(eig["eigenvalues"][()], float)
        es.occupations = np.asarray(eig["fermiweights"][()], float) if "fermiweights" in eig else None
        k = np.asarray(eig["kpoint_coords"][()], float)
        es.kpoints = k
        es.weights = np.asarray(eig["kpoints_symmetry_weight"][()], float) \
            if "kpoints_symmetry_weight" in eig else None
        inp = f.get(f"input/kpoints{suffix}")
        mode = _text(inp["mode"][()]).lower()[:1] if inp is not None and "mode" in inp else ""
        segment = int(inp["number_kpoints"][()]) if mode == "l" and "number_kpoints" in inp else None
        lattice = self._cell(f, _FINAL, 0).matrix
        es.distances, es.breaks = path_distances(k, lattice, segment)
        labels = [_text(x) for x in inp["labels_kpoints"][()]] if inp is not None and "labels_kpoints" in inp else []
        where = np.asarray(inp["positions_labels_kpoints"][()], int) if labels and \
            "positions_labels_kpoints" in inp else None
        if segment:
            ends = [""] * (2 * (len(k) // segment))
            for lab, pos in zip(labels, where if where is not None else []):
                if 0 < pos <= len(ends):
                    ends[pos - 1] = lab
            es.ticks = line_ticks(es.distances, segment, ends, k)
        elif where is not None:
            es.ticks = [(float(es.distances[p - 1]), pretty_label(lab)) for lab, p in zip(labels, where)
                        if 0 < p <= len(k)]
        par = f"results/projectors{suffix}/par"
        if par in f:
            path = self.path

            def load():
                with _h5py().File(path, "r") as h:
                    p = np.asarray(h[par][()], float)                # (spin, atom, orbital, k, band)
                p = p[:1] if len(p) == 4 else p
                return np.ascontiguousarray(np.transpose(p, (0, 3, 4, 1, 2)))

            es.projections_loader = load
        return es

    def _pressure(self, i):
        if self._stress is None or i >= len(self._stress) or self._stress.ndim != 3:
            return None
        return float(np.trace(self._stress[i]) / 3)

    @staticmethod
    def _step(ds, i):
        """Step ``i`` of a per-step dataset, or the dataset itself when it has no step axis."""
        return np.asarray(ds[i] if ds.ndim == 3 else ds[()], float)

    def _cell(self, f, group, i):
        lattice = self._step(f[group + "/lattice_vectors"], i)
        scale = np.asarray(f[group + "/scale"][()], float) if group + "/scale" in f else np.ones(())
        if scale.ndim == 0 and scale < 0:  # POSCAR convention: -volume
            scale = (-scale / abs(np.linalg.det(lattice))) ** (1 / 3)
        matrix = lattice * (scale if scale.ndim == 0 else scale[:, None])
        return Cell(matrix, np.zeros(3), np.ones(3, bool))

    def read(self, index: int) -> DataCollection:
        h5py = _h5py()
        index = range(self._n)[index]
        with h5py.File(self.path, "r") as f:
            if self._history:
                group = _DYN if _DYN + "/lattice_vectors" in f else _FINAL
                cell = self._cell(f, group, index)
                frac = np.asarray(f[_DYN + "/position_ions"][index], float)
            else:
                cell = self._cell(f, _FINAL, 0)
                frac = np.asarray(f[_FINAL + "/position_ions"][()], float)
            n = len(self.types)
            if frac.shape != (n, 3):
                raise ValueError(f"{self.path}: positions of step {index + 1} have shape {frac.shape}, not ({n}, 3)")
            data = DataCollection(cell.to_cartesian(frac), self.types, cell)
            data.particles["Particle Identifier"] = np.arange(1, n + 1)
            if self._history:
                for name, key in (("Force", "forces"), ("Velocity", "ion_velocities")):
                    ds = f.get(f"{_DYN}/{key}")
                    if ds is not None and ds.ndim == 3 and index < ds.shape[0] and ds.shape[1:] == (n, 3):
                        data.particles[name] = np.asarray(ds[index], float)
                moments = f.get(_DYN + "/magnetism/spin_moments/values")
                if moments is not None and moments.ndim == 4 and index < moments.shape[0] \
                        and moments.shape[1] > 1 and moments.shape[2] == n:
                    m = np.asarray(moments[index], float).sum(axis=-1)  # (component, N): sum over orbitals
                    data.particles["Magnetic Moment"] = m[1] if m.shape[0] == 2 else np.ascontiguousarray(m[1:4].T)
            elif _FINAL + "/ion_velocities" in f and f[_FINAL + "/ion_velocities"].shape == (n, 3):
                data.particles["Velocity"] = np.asarray(f[_FINAL + "/ion_velocities"][()], float)
            if self.grids and index == self._n - 1 and "results/potential" in f:
                grid = VoxelGrid(cell.copy())
                for key, name in _POTENTIALS.items():
                    if key in f["results/potential"]:
                        _add_components(grid, name, f["results/potential"][key][()], name + " Magnetization")
                if grid.properties:
                    data.grids["potential"] = grid
        data.attributes["Timestep"] = index + 1
        if self.electronic is not None:
            self.electronic.attach(data)
        for k, v in self._energies.items():
            if index < len(v):
                data.attributes[k] = float(v[index])
        if self._stress is not None and index < len(self._stress) and self._stress.ndim == 3:
            kb = self._stress[index]
            data.attributes["Stress"] = -np.array([kb[0, 0], kb[1, 1], kb[2, 2], kb[1, 2], kb[0, 2], kb[0, 1]]) \
                * _KB_TO_EV_A3
            data.attributes["Pressure"] = self._pressure(index)
        return data


class VaspwaveReader(FrameReader):
    """VASP 6 ``vaspwave.h5`` charge density (and kinetic energy density) on the final structure."""

    def __init__(self, path, structure=None):
        self.path = str(path)
        self.structure = str(structure) if structure is not None else \
            os.path.join(os.path.dirname(os.path.abspath(self.path)), "vaspout.h5")
        if not os.path.exists(self.structure):
            raise FileNotFoundError(f"{path}: vaspwave.h5 has no structure; expected vaspout.h5 next to it "
                                    f"(or pass structure=)")

    def __len__(self):
        return 1

    def read(self, index: int) -> DataCollection:
        range(1)[index]
        data = VaspoutReader(self.structure, grids=False).read(-1)
        data.attributes.pop("Timestep", None)
        cell = data.cell
        grid = VoxelGrid(cell.copy())
        with _h5py().File(self.path, "r") as f:
            if "charge/charge" in f:
                _add_components(grid, "Charge Density", np.asarray(f["charge/charge"][()], float) / cell.volume,
                                "Magnetization")
            if "kinetic_energy_density/values" in f:
                _add_components(grid, "Kinetic Energy Density",
                                np.asarray(f["kinetic_energy_density/values"][()], float) / cell.volume,
                                "Kinetic Energy Density Magnetization")
        if not grid.properties:
            raise ValueError(f"{self.path}: no charge density (set LCHARGH5 = .TRUE.)")
        data.grids["charge-density"] = grid
        return data
