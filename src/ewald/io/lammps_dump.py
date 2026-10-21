"""Native lazy reader for LAMMPS text dump files (``dump atom``/``dump custom``).

Handles orthogonal, restricted-triclinic (``xy xz yz``) and general-triclinic
(``abc origin``) boxes, ``dump_modify units/time yes`` headers, scaled/unwrapped
coordinates, per-atom vectors written as ``c_name[1] c_name[2] ...``, compressed
files (``.gz``/``.bz2``/``.xz``) and a truncated last frame from a killed run.
"""
from __future__ import annotations

import io

import numpy as np

from ..data import Cell, DataCollection
from . import remote
from ._textparse import parse_table
from .base import FrameReader, IncompleteFileError, file_size, head_lines, open_binary

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
        self._size = file_size(self.path)
        self._offsets, last_complete = _scan_frames(self.path)
        if not self._offsets:
            raise ValueError(f"{path}: no 'ITEM: TIMESTEP' found; not a LAMMPS dump file")
        self._end = None  # where the last frame ends (None: end of file)
        if not last_complete:  # a run killed mid-write leaves a partial last frame; drop it
            self._end = self._offsets.pop()
            if not self._offsets:
                raise IncompleteFileError(f"{path}: the only frame is incomplete (LAMMPS still writing it, "
                                          f"or stopped)") from None
            self._set_incomplete(f"{len(self._offsets)} frame(s) read, the partial last one skipped")
        self._last = None
        self._timesteps: dict[int, int] = {}  # frame offset -> ITEM: TIMESTEP, read on demand
        self._units = None  # LAMMPS writes ITEM: UNITS only in the first snapshot
        self._read_units()

    def _read_units(self):
        if len(self._offsets) > 1:
            with open_binary(self.path) as fh:
                self._units = _parse_header(fh)[0].get("Units")

    def __len__(self):
        return len(self._offsets)

    def refresh(self) -> int | None:
        """Index frames a running LAMMPS appended since the last look (only the new bytes are
        scanned); the old last frame is scanned again, so it counts as changed."""
        size = file_size(self.path)
        if size is None or size == self._size:
            return None
        if size < self._size:  # truncated or rewritten (a new run): index from scratch
            self._size, start, keep = size, 0, 0
        else:
            self._size = size
            start = self._end if self._end is not None else self._offsets[-1]
            keep = len(self._offsets) - (self._end is None)
        offsets, complete = _scan_frames(self.path, start)
        end = None
        if offsets and not complete:  # the frame being written right now
            end = offsets.pop()
        offsets = self._offsets[:keep] + offsets
        if not offsets:
            return None
        # the old last frame was scanned again: its last line may have been half written
        changed = keep < len(self._offsets) or offsets != self._offsets
        self._offsets, self._end, self._last = offsets, end, None
        if keep == 0:  # rewritten: the same offsets may hold other frames now
            self._timesteps = {}
        self._set_incomplete(f"{len(offsets)} frame(s) read, the partial last one skipped" if end is not None
                             else None, warn=False)
        if keep == 0 or (self._units is None and len(offsets) > 1):
            self._read_units()
        return keep if changed else None

    def timestep(self, index: int) -> int | None:
        """``ITEM: TIMESTEP`` of frame ``index``, from its first lines only (cached)."""
        index = range(len(self._offsets))[index]
        cache = self._timesteps
        offset = self._offsets[index]
        if offset not in cache:
            with open_binary(self.path) as fh:
                fh.seek(offset)
                for _ in range(4):  # UNITS and TIME may come first
                    line = fh.readline()
                    if not line:
                        return None
                    if line.startswith(b"ITEM: TIMESTEP"):
                        cache[offset] = int(fh.readline().split()[0])
                        break
                else:
                    return None
        return cache[offset]

    def read(self, index: int) -> DataCollection:
        index = range(len(self._offsets))[index]
        if self._last is None or self._last[0] != index:
            with open_binary(self.path) as fh:
                fh.seek(self._offsets[index])
                end = self._offsets[index + 1] if index + 1 < len(self._offsets) else self._end
                self._last = (index, self._parse(fh, end))
        return self._last[1].copy()

    def _parse(self, fh, end=None) -> DataCollection:
        attrs, n, cell, cols = _parse_header(fh)
        # One read of the whole atom block (up to the next frame) instead of n readline calls.
        block = fh.read() if end is None else fh.read(max(end - fh.tell(), 0))
        table = _read_table(block, n, cols)

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
                    data.particles["Particle Type"] = _type_labels(
                        index if index is not None else column, self.type_map if col == "type" else None)
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
            order = _id_order(data.particles["Particle Identifier"])
            if order is not None:
                data = data.take(order)
        data.attributes.update(attrs)
        data.attributes.setdefault("Timestep", 0)
        if self._units is not None:
            data.attributes.setdefault("Units", self._units)
        return data


def _parse_header(fh):
    """Read the ITEM lines of one frame up to and including 'ITEM: ATOMS'."""
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
    return attrs, n, cell, cols


def _read_table(block: bytes, n: int, cols) -> list[np.ndarray]:
    """Atom lines -> one array per column (float, or bytes for text columns such as 'element')."""
    if n == 0:
        return [np.empty(0) for _ in cols]
    text = [k for k, c in enumerate(cols) if c == "element"]
    numeric = [k for k in range(len(cols)) if k not in text]
    try:  # parallel numba parser, np.loadtxt for rows it cannot convert exactly
        values = parse_table(block, n, len(cols), numeric)
        out: list = [None] * len(cols)
        for j, k in enumerate(numeric):
            out[k] = values[:, j]
        if text:
            raw = np.array(block.split()[:n * len(cols)], dtype=bytes).reshape(n, len(cols))
            for k in text:
                out[k] = raw[:, k]
        return out
    except ValueError:
        raw = block.split()[:n * len(cols)]
        if len(raw) != n * len(cols):
            raise ValueError(f"atom table has {len(raw)} values, expected {n} x {len(cols)}") from None
        raw = np.array(raw, dtype=bytes).reshape(n, len(cols))
        return [raw[:, k] for k in range(len(cols))]


def _id_order(ids) -> np.ndarray | None:
    """Permutation sorting atoms by id, or None when they are already sorted.

    MPI runs write ids in scrambled order, usually a permutation of a dense range;
    then a scatter (O(n)) replaces the argsort.
    """
    n = len(ids)
    if n == 0 or np.all(ids[1:] >= ids[:-1]):
        return None
    lo = ids.min()
    if ids.dtype.kind in "iu" and ids.max() - lo == n - 1:
        order = np.full(n, -1, dtype=np.int64)
        order[ids - lo] = np.arange(n)
        if order.min() >= 0:  # every slot filled: ids are a permutation
            return order
    return np.argsort(ids, kind="stable")


def _type_labels(values, type_map=None) -> np.ndarray:
    """Per-atom type names as a compact string array, converting each distinct value once.

    ``astype(str)`` on 1e6+ int64 values is slow and yields 21-character strings (84 bytes
    per atom); mapping the few distinct types through a lookup table avoids both.
    """
    values = np.asarray(values)
    if values.dtype.kind == "i" and len(values) and 0 <= values.min() and values.max() < 1 << 16:
        uniq = np.arange(values.max() + 1)
        inverse = values
    else:
        uniq, inverse = np.unique(values, return_inverse=True)
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in uniq.tolist()]
    if type_map:
        names = [type_map.get(v, v) for v in names]
    return np.array(names, dtype=str)[inverse]


def _scan_frames(path, start: int = 0, chunk_size: int = 1 << 24) -> tuple[list[int], bool]:
    """Byte offset of the first ITEM line of every frame (UNITS/TIME may precede TIMESTEP),
    and whether the last frame holds all of its atom lines. ``start`` (a frame start)
    skips the bytes before it.

    Reads the file in large chunks and only looks at 'ITEM:' lines, so indexing a
    multi-GB dump costs about as much as reading it from disk. A remote dump is indexed
    by ``grep`` on the host instead (only the ITEM lines cross the network).
    """
    hits = remote.line_offsets(path, "^ITEM: (TIME|UNITS|ATOMS)", start) if remote.is_remote(path) else None
    if hits is not None:
        return _scan_remote(path, hits)
    offsets: list[int] = []
    in_header = False
    natoms = None  # NUMBER OF ATOMS of the current frame
    ncols = 0  # columns in its ATOMS line
    counted = None  # atom lines seen since the ATOMS line, None outside the atom block
    want_n = False  # the next line holds the atom count
    pos = start  # file offset of buf[0]
    carry = b""
    with open_binary(path) as fh:
        if start:
            fh.seek(start)
        # read1 on a compressed stream returns what decompressed so far, so a truncated
        # .gz still yields every complete frame before the cut
        read = fh.read if isinstance(fh, io.BufferedReader) else fh.read1
        while True:
            try:
                chunk = read(chunk_size)
            except EOFError:  # truncated compressed stream; the partial frame is dropped later
                chunk = b""
            buf = carry + chunk
            if chunk:
                end = buf.rfind(b"\n") + 1
                if end == 0:
                    carry = buf
                    continue
            else:
                end = len(buf)
            seg, carry = buf[:end], buf[end:]
            i = 0
            if want_n and seg:
                eol = seg.find(b"\n") + 1 or end
                natoms, want_n, i = int(seg[:eol]), False, eol
            while True:
                j = seg.find(b"ITEM:", i)
                while j > 0 and seg[j - 1] != 10:  # 'ITEM:' not at a line start
                    j = seg.find(b"ITEM:", j + 5)
                stop = end if j < 0 else j
                if counted is not None:
                    counted += seg.count(b"\n", i, stop)
                    if not chunk and stop == end and not seg.endswith(b"\n"):
                        tail = seg[seg.rfind(b"\n", i, stop) + 1:stop]  # unterminated last line
                        counted += len(tail.split()) >= ncols
                if j < 0:
                    break
                counted = None
                eol = seg.find(b"\n", j) + 1 or end
                line = seg[j:eol]
                if line.startswith(_FRAME_START):
                    if not in_header:
                        offsets.append(pos + j)
                        in_header, natoms = True, None
                elif line.startswith(b"ITEM: NUMBER OF ATOMS"):
                    nl = seg.find(b"\n", eol) + 1
                    if nl:
                        natoms, eol = int(seg[eol:nl]), nl
                    else:
                        want_n = True
                elif line.startswith(b"ITEM: ATOMS"):
                    in_header, counted, ncols = False, 0, len(line.split()) - 2
                i = eol
            pos += end
            if not chunk:
                break
    complete = bool(offsets) and natoms is not None and counted is not None and counted >= natoms
    return offsets, complete


def _scan_remote(path, hits) -> tuple[list[int], bool]:
    """Frame offsets from remote ITEM-line hits; the last frame is checked by reading only it."""
    offsets, in_header = [], False
    for pos, line in hits:
        if line.startswith(_FRAME_START):
            if not in_header:
                offsets.append(pos)
                in_header = True
        elif line.startswith(b"ITEM: ATOMS"):
            in_header = False
    if not offsets:
        return offsets, False
    try:
        with open_binary(path) as fh:
            fh.seek(offsets[-1])
            _, n, _, cols = _parse_header(fh)
            block = fh.read()
    except (ValueError, IndexError, EOFError):
        return offsets, False
    lines = block.count(b"\n")
    if block and not block.endswith(b"\n"):
        lines += len(block[block.rfind(b"\n") + 1:].split()) >= len(cols)
    return offsets, lines >= n


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
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
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
