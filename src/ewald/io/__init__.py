"""File import/export."""
from __future__ import annotations

import glob
import os

from ..data import DataCollection
from .ase_reader import AseReader
from .base import FrameReader
from .lammps_data import LammpsDataReader, is_lammps_data
from .lammps_dump import LammpsDumpReader, is_lammps_dump, write_lammps_dump
from .xdatcar import XdatcarReader, is_xdatcar


class FileSource:
    """Pipeline source backed by one or more files, frames loaded lazily."""

    def __init__(self, path, format: str | None = None, **kwargs):
        paths = sorted(glob.glob(str(path))) if any(c in str(path) for c in "*?[") else [str(path)]
        if not paths:
            raise FileNotFoundError(path)
        self.paths = paths
        self.path = str(path)
        self.format = format
        self.kwargs = kwargs
        self.readers: list[FrameReader] = [open_reader(p, format, **kwargs) for p in paths]
        self._index = [(r, i) for r in self.readers for i in range(len(r))]

    @property
    def num_frames(self) -> int:
        return len(self._index)

    def compute(self, frame: int) -> DataCollection:
        if not -self.num_frames <= frame < self.num_frames:
            raise IndexError(f"frame {frame} out of range (0..{self.num_frames - 1})")
        reader, i = self._index[frame]
        data = reader.read(i)
        data.attributes["SourceFrame"] = frame % self.num_frames
        return data


class StaticSource:
    """Pipeline source wrapping an in-memory DataCollection or list of them."""

    def __init__(self, frames):
        self.frames = frames if isinstance(frames, (list, tuple)) else [frames]
        self.frames = [DataCollection.from_ase(f) if not isinstance(f, DataCollection) else f
                       for f in self.frames]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def compute(self, frame: int) -> DataCollection:
        data = self.frames[frame].copy()
        data.attributes["SourceFrame"] = frame % self.num_frames
        return data


def open_reader(path, format: str | None = None, **kwargs) -> FrameReader:
    fmt = (format or "").lower()
    if fmt in ("lammps/dump", "lammps-dump", "lammps-dump-text") or (not fmt and is_lammps_dump(path)):
        return LammpsDumpReader(path, **kwargs)
    if fmt in ("vasp-xdatcar", "xdatcar") or (not fmt and is_xdatcar(path)):
        return XdatcarReader(path, **kwargs)
    if fmt in ("lammps/data", "lammps-data", "lammps-data-native") or (not fmt and is_lammps_data(path)):
        if "style" in kwargs:  # ASE's keyword name
            kwargs["atom_style"] = kwargs.pop("style")
        kwargs.pop("units", None)
        return LammpsDataReader(path, **kwargs)
    if fmt == "lammps-data-ase":
        return AseReader(path, format="lammps-data", **kwargs)
    return AseReader(path, format=format, **kwargs)


def import_file(path, format: str | None = None, **kwargs):
    """Create a Pipeline reading ``path`` (wildcards allowed: ``dump.*.lammpstrj``).

    Keyword arguments go to the reader, e.g. ``type_map={1: "Cu", 2: "Zr"}`` for
    LAMMPS dumps/data files that only contain numeric types, or
    ``atom_style="atomic"`` for LAMMPS data files without an ``Atoms # style`` hint.
    """
    from ..pipeline import Pipeline

    return Pipeline(FileSource(path, format, **kwargs))


def export_file(source, path, format: str | None = None, frames=None, **kwargs):
    """Write a DataCollection, a list of them, or a Pipeline's frames to ``path``.

    ``format="lammps/dump"`` uses the native writer; anything else goes through ASE
    (``vasp``, ``vasp-xdatcar``, ``extxyz``, ``lammps-data``, ``cif``, ...).
    """
    from ..pipeline import Pipeline

    if isinstance(source, Pipeline):
        idx = range(source.num_frames) if frames is None else frames
        items = [source.compute(i) for i in idx]
    elif isinstance(source, DataCollection):
        items = [source]
    else:
        items = list(source)
    if (format or "").lower() in ("lammps/dump", "lammps-dump") or str(path).endswith((".dump", ".lammpstrj")):
        write_lammps_dump(path, items, **kwargs)
        return
    import ase.io

    images = [d.to_ase() for d in items]
    ase.io.write(os.fspath(path), images if len(images) > 1 else images[0], format=format, **kwargs)


__all__ = ["FileSource", "StaticSource", "import_file", "export_file", "open_reader",
           "LammpsDumpReader", "LammpsDataReader", "XdatcarReader", "AseReader"]
