"""Native lazy reader for LAMMPS text dump files (``dump atom``/``dump custom``)."""
from __future__ import annotations

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader, _scan_lines

# LAMMPS column name -> (OVITO-style property, component)
_COLUMN_MAP = {
    "id": ("Particle Identifier", None), "type": ("Particle Type", None),
    "element": ("Particle Type", None), "mol": ("Molecule Identifier", None),
    "mass": ("Mass", None), "q": ("Charge", None), "radius": ("Radius", None),
    "x": ("Position", 0), "y": ("Position", 1), "z": ("Position", 2),
    "xu": ("Position", 0), "yu": ("Position", 1), "zu": ("Position", 2),
    "xs": ("Position.reduced", 0), "ys": ("Position.reduced", 1), "zs": ("Position.reduced", 2),
    "xsu": ("Position.reduced", 0), "ysu": ("Position.reduced", 1), "zsu": ("Position.reduced", 2),
    "ix": ("Periodic Image", 0), "iy": ("Periodic Image", 1), "iz": ("Periodic Image", 2),
    "vx": ("Velocity", 0), "vy": ("Velocity", 1), "vz": ("Velocity", 2),
    "fx": ("Force", 0), "fy": ("Force", 1), "fz": ("Force", 2),
    "c_pe": ("Potential Energy", None), "c_ke": ("Kinetic Energy", None),
}


def is_lammps_dump(path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.readline().startswith(b"ITEM: TIMESTEP")
    except OSError:
        return False


class LammpsDumpReader(FrameReader):
    def __init__(self, path, type_map: dict | None = None, sort_by_id: bool = True):
        self.path = str(path)
        self.type_map = {str(k): v for k, v in (type_map or {}).items()}
        self.sort_by_id = sort_by_id
        self._offsets = _scan_lines(self.path, lambda l: l.startswith(b"ITEM: TIMESTEP"))
        if not self._offsets:
            raise ValueError(f"{path}: no 'ITEM: TIMESTEP' found; not a LAMMPS dump file")

    def __len__(self):
        return len(self._offsets)

    def read(self, index: int) -> DataCollection:
        with open(self.path, "rb") as fh:
            fh.seek(self._offsets[index])
            return self._parse(fh)

    def _parse(self, fh) -> DataCollection:
        fh.readline()
        timestep = int(fh.readline().split()[0])
        fh.readline()  # ITEM: NUMBER OF ATOMS
        n = int(fh.readline())
        header = fh.readline().decode().split()  # ITEM: BOX BOUNDS [xy xz yz] pp pp pp
        flags = [t for t in header[3:] if t not in ("xy", "xz", "yz", "abc", "origin")]
        pbc = np.array([f.startswith("pp") for f in flags[:3]]) if len(flags) >= 3 else np.ones(3, bool)
        bounds = np.array([fh.readline().split() for _ in range(3)], dtype=float)
        cell = _cell_from_bounds(bounds, "xy" in header, pbc)
        cols = fh.readline().decode().split()[2:]
        raw = b"".join(fh.readline() for _ in range(n)).split()
        table = np.array(raw, dtype=object).reshape(n, len(cols)) if n else np.empty((0, len(cols)))

        data = DataCollection(np.zeros((n, 3)), cell=cell)
        vectors: dict[str, dict[int, np.ndarray]] = {}
        for k, col in enumerate(cols):
            name, comp = _COLUMN_MAP.get(col, (col, None))
            column = table[:, k]
            if name == "Particle Type":
                if col == "element" or "Particle Type" not in data.particles:
                    values = column.astype(str)
                    if col == "type":
                        values = np.array([self.type_map.get(v, v) for v in values]) if self.type_map else values
                    data.particles["Particle Type"] = values
                if col == "type":
                    data.particles["Type Index"] = column.astype(int)
            elif comp is None:
                data.particles[name] = _numeric(column)
            else:
                vectors.setdefault(name, {})[comp] = column.astype(float)
        for name, comps in vectors.items():
            arr = np.zeros((n, 3))
            for c, v in comps.items():
                arr[:, c] = v
            data.particles[name] = arr if name != "Periodic Image" else arr.astype(int)
        if "Position.reduced" in data.particles:
            red = data.particles.pop("Position.reduced")
            if "Position" not in vectors:
                data.particles["Position"] = cell.to_cartesian(red)
        if self.sort_by_id and "Particle Identifier" in data.particles:
            order = np.argsort(data.particles["Particle Identifier"], kind="stable")
            if np.any(order != np.arange(n)):
                data = data.take(order)
        data.attributes["Timestep"] = timestep
        return data


def _numeric(column):
    try:
        values = column.astype(float)
    except ValueError:
        return column.astype(str)
    if np.all(values == np.round(values)) and np.all(np.abs(values) < 2**62):
        return values.astype(np.int64)
    return values


def _cell_from_bounds(bounds, triclinic, pbc) -> Cell:
    """Convert LAMMPS (possibly triclinic) box bounds to a cell matrix + origin."""
    if triclinic:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = bounds
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds[:, :2]
        xy = xz = yz = 0.0
    matrix = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    return Cell(matrix, [xlo, ylo, zlo], pbc)


def write_lammps_dump(path, frames, columns=("id", "type", "x", "y", "z")):
    """Write orthogonal or triclinic frames as a text dump (mostly for tests/export)."""
    with open(path, "w") as fh:
        for step, data in enumerate(frames):
            m, o = data.cell.matrix, data.cell.origin
            xy, xz, yz = m[1, 0], m[2, 0], m[2, 1]
            tri = any(abs(v) > 1e-12 for v in (xy, xz, yz))
            lo = o
            hi = o + np.array([m[0, 0], m[1, 1], m[2, 2]])
            ts = data.attributes.get("Timestep", step)
            fh.write(f"ITEM: TIMESTEP\n{ts}\nITEM: NUMBER OF ATOMS\n{data.count}\n")
            pb = " ".join("pp" if p else "ff" for p in data.cell.pbc)
            if tri:
                xlo_b = lo[0] + min(0.0, xy, xz, xy + xz)
                xhi_b = hi[0] + max(0.0, xy, xz, xy + xz)
                ylo_b = lo[1] + min(0.0, yz)
                yhi_b = hi[1] + max(0.0, yz)
                fh.write(f"ITEM: BOX BOUNDS xy xz yz {pb}\n{xlo_b} {xhi_b} {xy}\n{ylo_b} {yhi_b} {xz}\n{lo[2]} {hi[2]} {yz}\n")
            else:
                fh.write(f"ITEM: BOX BOUNDS {pb}\n" + "".join(f"{lo[i]} {hi[i]}\n" for i in range(3)))
            fh.write("ITEM: ATOMS " + " ".join(columns) + "\n")
            types = data.types
            uniq = {t: i + 1 for i, t in enumerate(dict.fromkeys(types))}
            ids = data.particles.get("Particle Identifier", np.arange(1, data.count + 1))
            for i in range(data.count):
                vals = []
                for c in columns:
                    if c == "id":
                        vals.append(str(int(ids[i])))
                    elif c == "type":
                        vals.append(str(uniq[types[i]]))
                    elif c == "element":
                        vals.append(str(types[i]))
                    elif c in "xyz":
                        vals.append(f"{data.positions[i, 'xyz'.index(c)]:.8f}")
                    else:
                        name, comp = _COLUMN_MAP.get(c, (c, None))
                        v = data.particles[name][i] if comp is None else data.particles[name][i, comp]
                        vals.append(f"{v:.8g}")
                fh.write(" ".join(vals) + "\n")
