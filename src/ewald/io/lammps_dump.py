"""Native lazy reader for LAMMPS text dump files (``dump atom``/``dump custom``).

Handles orthogonal, restricted-triclinic (``xy xz yz``) and general-triclinic
(``abc origin``) boxes, ``dump_modify units/time yes`` headers, scaled/unwrapped
coordinates, per-atom vectors written as ``c_name[1] c_name[2] ...``, compressed
files (``.gz``/``.bz2``/``.xz``) and a truncated last frame from a killed run.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..data import Cell, DataCollection
from . import remote
from .base import FrameReader, head_lines, open_binary

# LAMMPS column name -> (OVITO-style property, component)
_COLUMN_MAP = {
    "id": ("Particle Identifier", None), "type": ("Particle Type", None),
    "element": ("Particle Type", None), "mol": ("Molecule Identifier", None),
    "mass": ("Mass", None), "q": ("Charge", None), "radius": ("Radius", None),
    "diameter": ("Diameter", None),
    "x": ("Position", 0), "y": ("Position", 1), "z": ("Position", 2),
    "xu": ("Position.unwrapped", 0), "yu": ("Position.unwrapped", 1), "zu": ("Position.unwrapped", 2),
    "xs": ("Position.reduced", 0), "ys": ("Position.reduced", 1), "zs": ("Position.reduced", 2),
    "xsu": ("Position.reduced_unwrapped", 0), "ysu": ("Position.reduced_unwrapped", 1),
    "zsu": ("Position.reduced_unwrapped", 2),
    "ix": ("Periodic Image", 0), "iy": ("Periodic Image", 1), "iz": ("Periodic Image", 2),
    "vx": ("Velocity", 0), "vy": ("Velocity", 1), "vz": ("Velocity", 2),
    "fx": ("Force", 0), "fy": ("Force", 1), "fz": ("Force", 2),
    "mux": ("Dipole Orientation", 0), "muy": ("Dipole Orientation", 1), "muz": ("Dipole Orientation", 2),
    "omegax": ("Angular Velocity", 0), "omegay": ("Angular Velocity", 1), "omegaz": ("Angular Velocity", 2),
    "angmomx": ("Angular Momentum", 0), "angmomy": ("Angular Momentum", 1), "angmomz": ("Angular Momentum", 2),
    "tqx": ("Torque", 0), "tqy": ("Torque", 1), "tqz": ("Torque", 2),
    "c_pe": ("Potential Energy", None), "c_ke": ("Kinetic Energy", None),
}
_VECTORS3 = {name for name, comp in _COLUMN_MAP.values() if comp is not None}
# Which coordinate flavour becomes 'Position' when several are present.
_POSITION_PRIORITY = ("Position", "Position.unwrapped", "Position.reduced", "Position.reduced_unwrapped")
_FRAME_START = (b"ITEM: TIMESTEP", b"ITEM: UNITS", b"ITEM: TIME")


def is_lammps_dump(path) -> bool:
    head = head_lines(path, 1)
    return bool(head) and head[0].startswith(_FRAME_START)


class LammpsDumpReader(FrameReader):
    def __init__(self, path, type_map: dict | None = None, sort_by_id: bool = True):
        self.path = str(path)
        self.type_map = {str(k): v for k, v in (type_map or {}).items()}
        self.sort_by_id = sort_by_id
        self._offsets = _scan_frames(self.path)
        if not self._offsets:
            raise ValueError(f"{path}: no 'ITEM: TIMESTEP' found; not a LAMMPS dump file")
        self._last = None
        self._units = None  # LAMMPS writes ITEM: UNITS only in the first snapshot
        self._units = self.read(0).attributes.get("Units") if len(self._offsets) > 1 else None
        try:  # a run killed mid-write leaves a partial last frame; drop it
            self.read(len(self._offsets) - 1)
        except (ValueError, IndexError, EOFError):
            warnings.warn(f"{path}: last frame is incomplete and was skipped", stacklevel=2)
            self._offsets.pop()
            if not self._offsets:
                raise ValueError(f"{path}: the only frame is incomplete") from None

    def __len__(self):
        return len(self._offsets)

    def read(self, index: int) -> DataCollection:
        index = range(len(self._offsets))[index]
        if self._last is None or self._last[0] != index:  # the frame checked in __init__ is reused
            with open_binary(self.path) as fh:
                fh.seek(self._offsets[index])
                self._last = (index, self._parse(fh))
        return self._last[1].copy()

    def _parse(self, fh) -> DataCollection:
        attrs: dict[str, object] = {}
        n = None
        cell = None
        line = fh.readline()
        while True:
            if not line:
                raise EOFError("unexpected end of file inside a frame")
            item = line.decode().strip()
            if not item.startswith("ITEM:"):
                line = fh.readline()
                continue
            item = item[5:].strip()
            if item == "TIMESTEP":
                attrs["Timestep"] = int(fh.readline().split()[0])
            elif item == "UNITS":
                attrs["Units"] = fh.readline().decode().strip()
            elif item == "TIME":
                attrs["Time"] = float(fh.readline())
            elif item == "NUMBER OF ATOMS":
                n = int(fh.readline())
            elif item.startswith("BOX BOUNDS"):
                cell = _parse_box(item.split()[2:], [fh.readline().split() for _ in range(3)])
            elif item.startswith("ATOMS"):
                cols = item.split()[1:]
                break
            line = fh.readline()
        if n is None or cell is None:
            raise ValueError("frame lacks NUMBER OF ATOMS or BOX BOUNDS")
        lines = [fh.readline() for _ in range(n)]
        if n and (not lines[-1].strip() or lines[-1].startswith(b"ITEM:")):
            raise ValueError(f"frame has fewer than {n} atom lines")
        table = _read_table(b"".join(lines), n, cols)

        data = DataCollection(np.zeros((n, 3)), cell=cell)
        vectors: dict[str, dict[int, np.ndarray]] = {}
        for k, col in enumerate(cols):
            name, comp = _column_property(col)
            column = table[k]
            if name == "Particle Type":
                index = _numeric(column) if col == "type" else None
                if index is not None and index.dtype.kind != "i":
                    index = None  # type labels instead of numbers
                if col == "element" or "Particle Type" not in data.particles:
                    values = index.astype(str) if index is not None else column.astype(str)
                    if col == "type" and self.type_map:
                        values = np.array([self.type_map.get(v, v) for v in values])
                    data.particles["Particle Type"] = values
                if index is not None:
                    data.particles["Type Index"] = index
            elif comp is None:
                data.particles[name] = _numeric(column)
            else:
                vectors.setdefault(name, {})[comp] = column.astype(float)
        for name, comps in vectors.items():
            arr = np.zeros((n, max(max(comps) + 1, 3 if name in _VECTORS3 else 1)))
            for c, v in comps.items():
                arr[:, c] = v
            data.particles[name] = arr if name != "Periodic Image" else arr.astype(int)
        _resolve_positions(data, cell, vectors)
        if self.sort_by_id and "Particle Identifier" in data.particles:
            order = np.argsort(data.particles["Particle Identifier"], kind="stable")
            if np.any(order != np.arange(n)):
                data = data.take(order)
        data.attributes.update(attrs)
        data.attributes.setdefault("Timestep", 0)
        if self._units is not None:
            data.attributes.setdefault("Units", self._units)
        return data


def _read_table(block: bytes, n: int, cols) -> list[np.ndarray]:
    """Atom lines -> one array per column (float, or bytes for text columns such as 'element')."""
    import io

    if n == 0:
        return [np.empty(0) for _ in cols]
    text = [k for k, c in enumerate(cols) if c == "element"]
    numeric = [k for k in range(len(cols)) if k not in text]
    try:  # numpy's C parser: ~4x faster than splitting in Python
        values = np.loadtxt(io.BytesIO(block), dtype=float, usecols=numeric, ndmin=2,
                            comments=None)
        if values.shape != (n, len(numeric)):
            raise ValueError
        out: list = [None] * len(cols)
        for j, k in enumerate(numeric):
            out[k] = values[:, j]
        if text:
            raw = np.array(block.split(), dtype=bytes).reshape(n, len(cols))
            for k in text:
                out[k] = raw[:, k]
        return out
    except ValueError:
        raw = block.split()
        if len(raw) != n * len(cols):
            raise ValueError(f"atom table has {len(raw)} values, expected {n} x {len(cols)}") from None
        raw = np.array(raw, dtype=bytes).reshape(n, len(cols))
        return [raw[:, k] for k in range(len(cols))]


def _scan_frames(path) -> list[int]:
    """Byte offset of the first ITEM line of every frame (UNITS/TIME may precede TIMESTEP)."""
    offsets, in_header = [], False
    for pos, line in _item_lines(path):
        if line.startswith(_FRAME_START):
            if not in_header:
                offsets.append(pos)
                in_header = True
        elif line.startswith(b"ITEM: ATOMS"):
            in_header = False
    return offsets


def _item_lines(path):
    """(offset, line) of every line; for a remote dump only the ITEM lines, found by grep on the host."""
    hits = remote.line_offsets(path, "^ITEM: (TIME|UNITS|ATOMS)") if remote.is_remote(path) else None
    if hits is not None:
        yield from hits
        return
    pos = 0
    with open_binary(path) as fh:
        try:
            for line in fh:
                yield pos, line
                pos += len(line)
        except EOFError:  # truncated compressed stream; the partial frame is dropped later
            pass


def _column_property(col: str):
    if col in _COLUMN_MAP:
        return _COLUMN_MAP[col]
    if col.endswith("]") and "[" in col:  # c_stress[3] -> ('c_stress', 2)
        base, idx = col[:-1].split("[", 1)
        if idx.isdigit() and int(idx) >= 1:
            return base, int(idx) - 1
    return col, None


def _resolve_positions(data: DataCollection, cell: Cell, columns) -> None:
    found = [k for k in _POSITION_PRIORITY if k in columns]
    if not found:
        return
    best = data.particles.pop(found[0])
    if found[0] in ("Position.reduced", "Position.reduced_unwrapped"):
        best = cell.to_cartesian(best)
    data.particles["Position"] = best
    for k in found[1:]:
        data.particles.pop(k)


def _parse_box(flags, rows) -> Cell:
    rows = np.array(rows, dtype=float)
    general = "abc" in flags
    bc = [f for f in flags if f not in ("xy", "xz", "yz", "abc", "origin")]
    pbc = np.array([f.startswith("pp") for f in bc[:3]]) if len(bc) >= 3 else np.ones(3, bool)
    if general:  # LAMMPS >= 2024 general triclinic: rows are a, b, c vectors + origin component
        return Cell(rows[:, :3], rows[:, 3], pbc)
    return _cell_from_bounds(rows, "xy" in flags, pbc)


def _numeric(column):
    try:
        values = np.asarray(column).astype(float)
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
