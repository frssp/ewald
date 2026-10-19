"""Native lazy reader and streaming writer for XYZ and extended XYZ (extxyz) trajectories.

Indexing finds the atom-count line of each frame from newline positions (numpy, in
16 MB chunks) without reading the atom lines one by one, so opening a multi-gigabyte
extxyz from an MLIP training set or an MD run is fast. A frame is parsed on demand:
the comment line (``Lattice=``, ``Properties=``, ``pbc=``, per-frame info) by ASE's
key=value parser, the atom table by the numba kernel in ``_textparse``, and the result
goes through ASE's own Atoms/calculator assembly, so names and units match
``ase.io.read``. Anything the fast path does not handle falls back to ASE's parser.
Plain XYZ is the same format without the comment-line keys.
"""
from __future__ import annotations

import io
import os

import numpy as np

from ..data import DataCollection
from ._textparse import parse_mixed
from .ase_reader import _ovito_names
from .base import FrameReader, IncompleteFileError, file_size, head_lines, open_binary, strip_compression


def is_extxyz(path) -> bool:
    if not strip_compression(path).lower().endswith((".xyz", ".extxyz")):
        return False
    head = head_lines(path, 1)
    return bool(head) and head[0].strip().isdigit()


class ExtxyzReader(FrameReader):
    """XYZ / extended XYZ, lazy (frames indexed from newline positions)."""

    def __init__(self, path):
        self.path = str(path)
        self._size = file_size(self.path)
        self._frames, complete = _scan_frames(self.path)  # (byte offset, byte length)
        if not self._frames:
            if not complete:
                raise IncompleteFileError(f"{path}: the only frame is incomplete (still being written, or the "
                                          f"run stopped)")
            raise ValueError(f"{path}: no frames found")
        if not complete:
            self._set_incomplete(f"{len(self._frames)} frame(s) read, the partial last one skipped")

    def refresh(self) -> int | None:
        """Index frames appended since the last look; the old last frame is scanned again,
        so it counts as changed."""
        size = file_size(self.path)
        if size is None or size == self._size:
            return None
        if size < self._size:  # rewritten: index from scratch
            self._size = size
            frames, complete = _scan_frames(self.path, growing=True)
            if not frames:
                return None
            self._frames = frames
            self._set_incomplete(None if complete else f"{len(frames)} frame(s) read, the partial last one skipped",
                                 warn=False)
            return 0
        self._size = size
        frames, complete = _scan_frames(self.path, offset=self._frames[-1][0], growing=True)
        if frames[:1] == self._frames[-1:] and len(frames) == 1:
            return None
        first = len(self._frames) - 1
        # an old last frame read with half its last line is dropped until it is complete
        self._frames = self._frames[:-1] + frames or self._frames
        self._set_incomplete(None if complete else f"{len(self._frames)} frame(s) read, the partial last one skipped",
                             warn=False)
        return first

    def __len__(self):
        return len(self._frames)

    def read(self, index: int) -> DataCollection:
        offset, size = self._frames[index]
        with open_binary(self.path) as fh:
            fh.seek(offset)
            text = fh.read(size)
        try:
            atoms = _read_frame(text)
        except Exception:  # anything unusual: ASE's parser decides (and raises its own errors)
            from ase.io.extxyz import read_xyz

            atoms = next(read_xyz(io.StringIO(text.decode()), 0))
        data = DataCollection.from_ase(atoms)
        _ovito_names(data, atoms)
        data.attributes.setdefault("Timestep", index)
        return data


def _scan_frames(path, chunk_size: int = 1 << 24, offset: int = 0,
                 growing: bool = False) -> tuple[list[tuple[int, int]], bool]:
    """(byte offset, byte length) of every complete frame, and whether the file ends cleanly.
    ``offset`` (a frame start) skips the bytes before it, and ``growing`` ignores a last line
    without a newline (still being written), for following a growing file.

    Walks newline positions found by numpy, jumping over each frame's n + 1 body lines;
    blank lines between frames are skipped. Python work is per frame, not per line.
    """
    frames: list[tuple[int, int]] = []
    skip = 0  # body lines (comment + atoms) still to pass in the current frame
    start = 0
    base = offset  # file offset of data[0]
    carry = b""
    with open_binary(path) as fh:
        if offset:
            fh.seek(offset)
        while True:
            chunk = fh.read(chunk_size)
            # EOF: close a last line without a newline (a file ending in one gets no empty extra line)
            data = carry + (chunk if chunk else b"\n" if carry and not growing else b"")
            nl = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == 10)
            m = len(nl)
            i = 0
            while i < m:
                if skip:
                    take = min(skip, m - i)
                    i += take
                    skip -= take
                    if not skip:
                        frames.append((start, base + int(nl[i - 1]) + 1 - start))
                    continue
                ls = int(nl[i - 1]) + 1 if i else 0
                line = data[ls:int(nl[i])]
                i += 1
                if not line.strip():
                    continue
                try:
                    n = int(line.split()[0])
                except ValueError:
                    raise ValueError(f"{path}: expected an atom count at byte {base + ls}, got "
                                     f"{line[:60].decode(errors='replace')!r}") from None
                start, skip = base + ls, n + 1
            if not chunk:
                return frames, skip == 0
            cut = int(nl[-1]) + 1 if m else 0
            carry = data[cut:]
            base += cut


def _read_frame(text: bytes):
    """One extxyz frame -> ase.Atoms, as ``ase.io.extxyz._read_xyz_frame`` builds it."""
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian
    from ase.data import atomic_numbers
    from ase.io.extxyz import key_val_str_to_dict, parse_properties, set_calc_and_arrays

    first = text.index(b"\n")
    second = text.find(b"\n", first + 1)
    if second < 0:
        second = len(text)
    natoms = int(text[:first].split()[0])
    line = text[first + 1:second].decode().strip()
    info = key_val_str_to_dict(line) if line else {}
    pbc = info.pop("pbc") if "pbc" in info else ([True] * 3 if "Lattice" in info else None)
    cell = info.pop("Lattice").T if "Lattice" in info else None
    properties, names, dtype, _ = parse_properties(info.pop("Properties", "species:S:1:pos:R:3"))

    kinds = [dtype.fields[f][0] for f in dtype.names]  # one per column
    numeric = [k for k, t in enumerate(kinds) if t.kind in "fi"]
    textcols = [k for k, t in enumerate(kinds) if t.kind not in "fi"]
    values, tokens = parse_mixed(text[second + 1:], natoms, len(kinds), numeric, textcols)
    columns: dict[int, np.ndarray] = {}
    for j, k in enumerate(numeric):
        col = values[:, j]
        if kinds[k].kind == "i":
            if not np.array_equal(col, np.trunc(col)):
                raise ValueError("non-integer value in an integer column")
            col = col.astype(kinds[k])
        columns[k] = col
    labels = {}  # string column -> (distinct values, index of each atom's value)
    for j, k in enumerate(textcols):
        tok = tokens[j]
        if kinds[k] == np.bool_:
            columns[k] = np.isin(tok, (b"T", b"True"))
        else:  # strings: decode each distinct value once
            uniq, inverse = np.unique(tok, return_inverse=True)
            labels[k] = ([u.decode() for u in uniq], inverse.ravel())
            columns[k] = np.array(labels[k][0], dtype=object)[labels[k][1]]

    arrays = {}
    k = 0
    for name in names:
        ase_name, cols = properties[name]
        if ase_name == "symbols" and cols == 1:  # element per distinct label, not per atom
            names_, inverse = labels[k]
            arrays[ase_name] = np.array([atomic_numbers[x.capitalize()] for x in names_], dtype=int)[inverse]
        else:
            arrays[ase_name] = columns[k] if cols == 1 else np.column_stack([columns[k + c] for c in range(cols)])
        k += cols
    numbers = arrays.pop("numbers", None)
    symbols = arrays.pop("symbols", None)  # atomic numbers here; a Z column wins, as in ASE
    atoms = Atoms(numbers=symbols if numbers is None else numbers, positions=arrays.pop("positions", None),
                  charges=arrays.pop("initial_charges", None), cell=cell, pbc=pbc, info=info)
    if "move_mask" in arrays:
        move_mask = arrays.pop("move_mask").astype(bool)
        if properties["move_mask"][1] == 3:
            atoms.set_constraint([FixCartesian(a, mask=~move_mask[a, :]) for a in range(natoms)])
        elif properties["move_mask"][1] == 1:
            atoms.set_constraint(FixAtoms(mask=~move_mask))
        else:
            raise ValueError("unsupported move_mask")
    set_calc_and_arrays(atoms, arrays)
    return atoms


def write_extxyz(path, frames, columns=None, **kwargs):
    """Stream ``frames`` (an iterable of DataCollection) to an extxyz file, one frame at a time.

    Every numeric particle property is written (``Force`` as ASE's ``forces``) unless
    ``columns`` names the ones to keep; scalar attributes go into the comment line.
    """
    from ase.io.extxyz import write_xyz

    with open(os.fspath(path), "w") as fh:
        for data in frames:
            atoms = _to_atoms(data, columns)
            write_xyz(fh, [atoms], **kwargs)


def _to_atoms(data: DataCollection, columns=None):
    from ase.calculators.singlepoint import SinglePointCalculator

    if columns is not None:
        keep = {"Position", "Particle Type", *columns}
        data = data.copy()
        data.particles = {k: v for k, v in data.particles.items() if k in keep}
    atoms = data.to_ase()
    results = {}
    if "Force" in atoms.arrays:
        results["forces"] = atoms.arrays.pop("Force")
    for key, value in data.attributes.items():
        if key == "Energy":
            results["energy"] = float(value)
        elif key == "Stress":
            results["stress"] = np.asarray(value)
        elif np.isscalar(value) and key not in ("SourceFrame",):
            atoms.info[key] = value
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms
