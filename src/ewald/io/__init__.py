"""File import/export."""
from __future__ import annotations

import glob
import os

from ..data import DataCollection
from . import remote
from .ase_reader import AseReader
from .base import FrameReader
from .lammps_data import LammpsDataReader, is_lammps_data
from .lammps_dump import LammpsDumpReader, is_lammps_dump, write_lammps_dump
from .volumetric import (VolumetricReader, is_cube, is_vasp_volumetric, read_cube, read_vasp_volumetric,
                         write_cube, write_vasp_volumetric)
from .xdatcar import XdatcarReader, is_xdatcar


class FileSource:
    """Pipeline source backed by one or more files, frames loaded lazily."""

    def __init__(self, path, format: str | None = None, **kwargs):
        path = str(path)
        wildcard = any(c in path for c in "*?[")
        if remote.is_remote(path):  # host:/path or ssh://host/path
            paths = remote.glob(path) if wildcard else [path]
            if not wildcard:
                remote.stat(path)  # connect (and authenticate) once, up front; raises if missing
        else:
            paths = sorted(glob.glob(path)) if wildcard else [path]
        if not paths:
            raise FileNotFoundError(path)
        self.paths = paths
        self.path = path
        self.format = format
        self.kwargs = kwargs
        self.readers: list[FrameReader] = [open_reader(_readable_path(p, format), format, **kwargs)
                                           for p in paths]
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


_STREAMING_FORMATS = ("lammps/dump", "lammps-dump", "lammps-dump-text", "vasp-xdatcar", "xdatcar",
                      "lammps/data", "lammps-data", "lammps-data-native")


def _readable_path(path: str, format: str | None) -> str:
    """Remote files go to the native readers as they are (read in place, frame by frame);
    for every other reader, ASE and plugins included, they are downloaded to the local cache."""
    if not remote.is_remote(path):
        return path
    fmt = (format or "").lower()
    if fmt in _STREAMING_FORMATS or (not fmt and (is_lammps_dump(path) or is_xdatcar(path)
                                                  or is_lammps_data(path))):
        return path
    return remote.fetch(path)


def open_reader(path, format: str | None = None, **kwargs) -> FrameReader:
    fmt = (format or "").lower()
    if fmt in ("cube", "gaussian-cube") or (not fmt and is_cube(path)):
        return VolumetricReader(path, "cube", pbc=kwargs.pop("pbc", False))
    if fmt in ("vasp-chgcar", "chgcar", "vasp-volumetric", "vasp-locpot", "locpot", "vasp-elfcar", "elfcar") \
            or (not fmt and is_vasp_volumetric(path)):
        kind = kwargs.pop("kind", None) or ("potential" if "locpot" in fmt else "elf" if "elfcar" in fmt else
                                            "charge" if "chgcar" in fmt else None)
        return VolumetricReader(path, "vasp", kind=kind)
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

    ``path`` may be on another machine: ``host:/path``, ``user@host:path`` or
    ``ssh://user@host:port/path`` (see :mod:`ewald.io.remote`). LAMMPS dumps and
    XDATCARs are then read frame by frame over SFTP; other formats are downloaded
    once to a local cache.

    Keyword arguments go to the reader, e.g. ``type_map={1: "Cu", 2: "Zr"}`` for
    LAMMPS dumps/data files that only contain numeric types, or
    ``atom_style="atomic"`` for LAMMPS data files without an ``Atoms # style`` hint.
    Volumetric files (VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR, ``*.cube``) give the
    atoms plus a grid in ``data.grids``; ``kind="charge"|"potential"|"elf"`` overrides the
    VASP file-name guess and ``pbc=True`` marks a cube file periodic.
    """
    from ..pipeline import Pipeline

    return Pipeline(FileSource(path, format, **kwargs))


def export_file(source, path, format: str | None = None, frames=None, **kwargs):
    """Write a DataCollection, a list of them, or a Pipeline's frames to ``path``.

    ``format="lammps/dump"`` uses the native writer; anything else goes through ASE
    (``vasp``, ``vasp-xdatcar``, ``extxyz``, ``lammps-data``, ``cif``, ...).
    ``format="cube"`` (or a ``.cube`` name) and ``format="vasp-chgcar"`` write a grid of
    one frame; pass ``property="Charge Density"`` (and ``grid=``) to pick what.
    """
    from ..pipeline import Pipeline

    if isinstance(source, Pipeline):
        idx = range(source.num_frames) if frames is None else frames
        items = [source.compute(i) for i in idx]
    elif isinstance(source, DataCollection):
        items = [source]
    else:
        items = list(source)
    fmt = (format or "").lower()
    if fmt in ("cube", "vasp-chgcar", "chgcar") or (not fmt and str(path).lower().endswith(".cube")):
        writer = write_cube if fmt in ("cube", "") else write_vasp_volumetric
        writer(path, items[0], **kwargs)
        return
    if (format or "").lower() in ("lammps/dump", "lammps-dump") or str(path).endswith((".dump", ".lammpstrj")):
        write_lammps_dump(path, items, **kwargs)
        return
    import ase.io

    images = [d.to_ase() for d in items]
    ase.io.write(os.fspath(path), images if len(images) > 1 else images[0], format=format, **kwargs)


__all__ = ["FileSource", "StaticSource", "import_file", "export_file", "open_reader",
           "LammpsDumpReader", "LammpsDataReader", "XdatcarReader", "AseReader", "VolumetricReader",
           "read_vasp_volumetric", "read_cube", "write_vasp_volumetric", "write_cube"]
