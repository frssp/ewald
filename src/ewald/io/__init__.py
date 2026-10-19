"""File import/export."""
from __future__ import annotations

import glob
import os

from ..data import DataCollection
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
    """Reader for ``path``: chosen from the reader registry (built-in formats and plugins)."""
    from .. import plugins

    return plugins.open_reader(path, format, **kwargs)


def _open_lammps_data(path, **kwargs):
    if "style" in kwargs:  # ASE's keyword name
        kwargs["atom_style"] = kwargs.pop("style")
    kwargs.pop("units", None)
    return LammpsDataReader(path, **kwargs)


def _open_ase(path, format=None, **kwargs):
    return AseReader(path, format=format, **kwargs)


def _open_cube(path, pbc=False, **kwargs):
    return VolumetricReader(path, "cube", pbc=pbc)


def _open_vasp_volumetric(path, format=None, kind=None, **kwargs):
    fmt = (format or "").lower()
    kind = kind or ("potential" if "locpot" in fmt else "elf" if "elfcar" in fmt else
                    "charge" if "chgcar" in fmt else None)
    return VolumetricReader(path, "vasp", kind=kind)


def _open_ase_lammps_data(path, **kwargs):
    return AseReader(path, format="lammps-data", **kwargs)


def builtin_readers():
    """ewald's own formats, registered in :mod:`ewald.plugins` like any reader plugin."""
    from ..plugins import ReaderPlugin

    B = "built-in"
    return [
        ReaderPlugin("cube", _open_cube, ("*.cube", "*.cub"), is_cube,
                     "Gaussian cube volumetric data (atoms + grid)", B, 5, ("gaussian-cube",), category="volumetric"),
        ReaderPlugin("vasp-volumetric", _open_vasp_volumetric,
                     ("CHGCAR*", "CHG*", "PARCHG*", "AECCAR*", "LOCPOT*", "ELFCAR*"), is_vasp_volumetric,
                     "VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR (atoms + grid)", B, 6,
                     ("vasp-chgcar", "chgcar", "vasp-locpot", "locpot", "vasp-elfcar", "elfcar"),
                     takes_format=True, category="volumetric"),
        ReaderPlugin("lammps-dump", LammpsDumpReader, ("*.lammpstrj", "*.dump"), is_lammps_dump,
                     "LAMMPS text dump (atom/custom), lazy", B, 10, ("lammps/dump", "lammps-dump-text")),
        ReaderPlugin("xdatcar", XdatcarReader, ("XDATCAR*",), is_xdatcar,
                     "VASP XDATCAR, fixed or variable cell, lazy", B, 11, ("vasp-xdatcar",)),
        ReaderPlugin("lammps-data", _open_lammps_data, ("*.data",), is_lammps_data,
                     "LAMMPS data file", B, 12, ("lammps/data", "lammps-data-native")),
        ReaderPlugin("lammps-data-ase", _open_ase_lammps_data, (), None,
                     "LAMMPS data file via ASE", B, None),
        ReaderPlugin("ase", _open_ase, ("POSCAR*", "CONTCAR*", "OUTCAR*", "vasprun*.xml", "*.xyz", "*.extxyz",
                                        "*.cif", "*.vasp"), lambda path: True,
                     "anything ase.io.read understands (POSCAR, OUTCAR, vasprun.xml, extxyz, cif, ...); "
                     "other format= names are passed to ASE", B, 100, takes_format=True),
    ]


def import_file(path, format: str | None = None, **kwargs):
    """Create a Pipeline reading ``path`` (wildcards allowed: ``dump.*.lammpstrj``).

    ``format`` names a built-in format, an ASE format, or a reader plugin (see
    :mod:`ewald.plugins`); without it the format is detected from the file.
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
