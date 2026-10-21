"""File import/export."""
from __future__ import annotations

import glob
import os

from ..data import DataCollection
from . import remote
from .ase_reader import AseDbReader, AseReader, AseTrajectoryReader, is_ase_db, is_ase_traj
from .base import FrameReader, IncompleteFileError, IncompleteFileWarning, natural_key
from .extxyz import ExtxyzReader, is_extxyz, write_extxyz
from .h5md import H5mdReader, is_h5md, write_h5md
from .lammps_data import LammpsDataReader, is_lammps_data, write_lammps_data
from .lammps_dump import LammpsDumpReader, is_lammps_dump, write_lammps_dump
from .thermo import Thermo, merge_tables, read_lammps_log, read_thermo, sibling_files
from .vasp_output import OutcarReader, VasprunReader, is_outcar, is_vasprun, read_oszicar
from .volumetric import (VolumetricReader, is_cube, is_vasp_volumetric, read_cube, read_vasp_volumetric,
                         write_cube, write_vasp_volumetric)
from .mdtraj_h5 import MdtrajH5Reader, is_mdtraj_h5
from .vasp_h5 import VaspoutReader, VaspwaveReader, is_vaspout, is_vaspwave
from .mdanalysis_reader import MDAnalysisReader, is_md_trajectory, write_md_trajectory
from .neb import NebReader, is_neb, neb_run_of, read_lammps_neb_log
from .xdatcar import XdatcarReader, is_xdatcar
from .xsf import XsfReader, is_xsf, write_xsf


class FileSource:
    """Pipeline source backed by one or more files, frames loaded lazily.

    ``path`` is one file, a wildcard (``dump.*.lammpstrj``, expanded in natural order, so
    ``dump.200`` comes before ``dump.1000``) or a list of files and wildcards, read one after the
    other in the order given. ``join`` says how the files are put together: ``"timestep"``
    (default) treats them as segments of one run that was restarted, so where a later file starts
    at a step an earlier one already reached, the later file wins and the earlier frames from that
    step on are dropped (LAMMPS dumps; formats whose steps restart in every run, such as XDATCAR
    and OUTCAR, are simply appended). ``"concat"`` keeps every frame of every file.

    ``thermo`` adds per-step results (see :mod:`ewald.io.thermo`): ``"auto"`` (default) picks
    up an OSZICAR next to an XDATCAR and a log.lammps next to a LAMMPS dump, a path or list
    of paths names LAMMPS logs / OSZICARs, ``None`` turns it off. OUTCAR and vasprun.xml
    carry their own. The result is ``self.thermo`` (a :class:`Thermo`, or None).
    """

    def __init__(self, path, format: str | None = None, thermo="auto", join: str = "timestep", **kwargs):
        if join not in _JOINS:
            raise ValueError(f"join must be one of {', '.join(map(repr, _JOINS))}, not {join!r}")
        self.path = [os.fspath(p) for p in path] if _is_list(path) else os.fspath(path)
        self.format = format
        self.join = join
        self.kwargs = kwargs
        paths = self._expand()
        if not paths:
            raise FileNotFoundError(self.path)
        self.paths = paths
        self.readers: list[FrameReader] = [open_reader(_readable_path(p, format), format, **kwargs)
                                           for p in paths]
        self._index = self._build_index(warn=True)
        self.thermo_arg = thermo
        self.thermo = self._load_thermo(thermo)

    def _patterns(self) -> list[str]:
        return list(self.path) if isinstance(self.path, list) else [self.path]

    def _expand(self, check: bool = True) -> list[str]:
        """The files ``path`` names, in reading order (wildcards in natural order)."""
        from .. import plugins

        neb = (self.format or "").lower() in _NEB_FORMATS  # a NEB reader takes the whole wildcard
        paths: list[str] = []
        seen: set[str] = set()
        for pattern in self._patterns():
            wildcard = any(c in pattern for c in "*?[") and not neb
            if plugins.scheme_reader(pattern) is not None:  # aiida:1234 and other plugin schemes
                found = [pattern]
            elif remote.is_remote(pattern):  # host:/path or ssh://host/path
                found = sorted(remote.glob(pattern), key=natural_key) if wildcard else [pattern]
                if check and not any(c in pattern for c in "*?["):
                    remote.stat(pattern)  # connect (and authenticate) once, up front; raises if missing
            else:
                found = sorted(glob.glob(pattern), key=natural_key) if wildcard else [pattern]
            if check and not found and isinstance(self.path, list):
                raise FileNotFoundError(pattern)
            paths += [p for p in found if not (p in seen or seen.add(p))]
        return paths

    def _build_index(self, warn: bool = False) -> list[tuple[FrameReader, int]]:
        """(reader, frame) of every frame shown. With ``join="timestep"`` each file is cut where
        a later one starts (a restart from an earlier step), found by bisection on the steps."""
        keep = [len(r) for r in self.readers]
        starts = [r.timestep(0) if len(r) and self.join == "timestep" else None for r in self.readers]
        known = [s for s in starts if s is not None]
        if len(known) > 1 and len(set(known)) == 1:  # all start at the same step: not one run, append them
            starts = [None] * len(starts)
        cutoff = None  # the earliest step a later file starts at
        for k in range(len(self.readers) - 1, -1, -1):
            if starts[k] is None:
                continue
            if cutoff is not None:
                keep[k] = _frames_before(self.readers[k], cutoff)
                if keep[k] == 0 and warn:
                    import warnings

                    warnings.warn(f"{self.paths[k]}: every frame is at or after Timestep {cutoff}, where a later "
                                  "file starts, so it is left out (pass join='concat' to keep it)", stacklevel=4)
            cutoff = starts[k] if cutoff is None else min(cutoff, starts[k])
        return [(r, i) for r, n in zip(self.readers, keep) for i in range(n)]

    @property
    def segments(self) -> list[tuple[str, int, int]]:
        """``(file, frames used, frames in the file)`` for every file, in reading order."""
        used: dict[int, int] = {}
        for r, _ in self._index:
            used[id(r)] = used.get(id(r), 0) + 1
        return [(p, used.get(id(r), 0), len(r)) for p, r in zip(self.paths, self.readers)]

    def _load_thermo(self, thermo) -> Thermo | None:
        tables = [r.thermo for r in self.readers[:1] if len(self.readers) == 1 and isinstance(getattr(r, "thermo", None), dict)]
        files: list[str] = []
        if thermo == "auto":
            kinds = {XdatcarReader: "xdatcar", LammpsDumpReader: "lammps-dump"}
            for p, r in zip(self.paths, self.readers):
                files += [f for f in sibling_files(p, kinds.get(type(r), "")) if f not in files]
        elif thermo:
            files = [str(thermo)] if isinstance(thermo, (str, os.PathLike)) else [str(t) for t in thermo]
        for f in files:
            try:
                tables.append(read_thermo(f))
            except (OSError, ValueError) as exc:
                if thermo != "auto":
                    raise
                import warnings

                warnings.warn(f"ignoring {f}: {exc}", stacklevel=3)
        return Thermo(merge_tables(tables), files) if tables else None

    @property
    def num_frames(self) -> int:
        return len(self._index)

    @property
    def incomplete(self) -> list[str]:
        """``"file: what is missing"`` for each file of a calculation that is still running or
        was stopped (see :class:`~ewald.io.base.IncompleteFileWarning`); empty when all are complete."""
        return [f"{os.path.basename(str(p))}: {r.incomplete}" for p, r in zip(self.paths, self.readers)
                if getattr(r, "incomplete", None)]

    def refresh(self) -> int | None:
        """Follow files a running simulation is still writing: frames appended to the files
        (LAMMPS dump, XDATCAR, OUTCAR, vasprun.xml, extxyz, H5MD, NEB images), and for a wildcard,
        files that appeared.

        Returns the first frame whose data may have changed (see :meth:`FrameReader.refresh`),
        or None when nothing changed. Thermo tables (log.lammps, OSZICAR) are read again.
        Use :meth:`ewald.pipeline.Pipeline.refresh`, which also drops stale cached frames.
        """
        readers = dict(zip(self.paths, self.readers))
        old_index = self._index
        paths = self.paths
        if any(c in p for p in self._patterns() for c in "*?[") and (self.format or "").lower() not in _NEB_FORMATS:
            try:
                paths = self._expand(check=False)
            except OSError:
                paths = self.paths
        changed_at: dict[int, int] = {}  # id(reader) -> its first frame that may have changed
        for k, p in enumerate(paths):
            reader = readers.get(p)
            if reader is None:  # a new file of the series
                try:
                    reader = open_reader(_readable_path(p, self.format), self.format, **self.kwargs)
                except (OSError, ValueError):
                    paths = paths[:k]  # still being created; try again next time
                    break
                readers[p] = reader
                changed = 0
            elif p != (self.paths[k] if k < len(self.paths) else None):
                changed = 0  # the series was renumbered around it
            else:
                changed = reader.refresh()
            if changed is not None:
                changed_at[id(reader)] = changed
        if not changed_at and list(paths) == self.paths:
            return None
        self.paths = list(paths)
        self.readers = [readers[p] for p in paths]
        self._index = self._build_index()
        # the first frame shown that is another frame than before, or one whose file changed there
        first = next((k for k, (r, i) in enumerate(self._index)
                      if k >= len(old_index) or old_index[k] != (r, i) or i >= changed_at.get(id(r), i + 1)), None)
        if first is None:
            if len(self._index) == len(old_index):
                return None
            first = len(self._index)  # frames went away
        try:
            self.thermo = self._load_thermo(self.thermo_arg)
        except (OSError, ValueError):
            pass
        return first

    def compute(self, frame: int) -> DataCollection:
        if not -self.num_frames <= frame < self.num_frames:
            raise IndexError(f"frame {frame} out of range (0..{self.num_frames - 1})")
        reader, i = self._index[frame]
        data = reader.read(i)
        data.attributes["SourceFrame"] = frame % self.num_frames
        if self.thermo is not None:
            self.thermo.annotate(data)
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


_NEB_FORMATS = ("neb", "vasp-neb", "lammps-neb")
_JOINS = ("timestep", "concat")


def _is_list(path) -> bool:
    return isinstance(path, (list, tuple))


def _frames_before(reader: FrameReader, step: int) -> int:
    """How many leading frames of ``reader`` have a Timestep below ``step`` (steps increase in a file)."""
    lo, hi = 0, len(reader)
    if hi == 0 or (reader.timestep(hi - 1) or 0) < step:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if (reader.timestep(mid) or 0) < step:
            lo = mid + 1
        else:
            hi = mid
    return lo

_STREAMING_FORMATS = ("lammps/dump", "lammps-dump", "lammps-dump-text", "vasp-xdatcar", "xdatcar",
                      "lammps/data", "lammps-data", "lammps-data-native", "outcar", "vasp-outcar", "vasprun",
                      "vasp-vasprun")


def _readable_path(path: str, format: str | None) -> str:
    """Remote files go to the native readers as they are (read in place, frame by frame);
    for every other reader, ASE and plugins included, they are downloaded to the local cache."""
    from .. import plugins

    if not remote.is_remote(path) or plugins.scheme_reader(path) is not None:
        return path
    fmt = (format or "").lower()
    if fmt in _NEB_FORMATS or (not fmt and _is_remote_dir(path)):
        return path  # a NEB folder: its image files are read (or fetched) one by one
    if fmt in _STREAMING_FORMATS or (not fmt and (is_lammps_dump(path) or is_xdatcar(path)
                                                  or is_lammps_data(path) or is_outcar(path)
                                                  or is_vasprun(path))):
        return path
    return remote.fetch(path)


def _is_remote_dir(path: str) -> bool:
    import stat

    try:
        return stat.S_ISDIR(remote.stat(path).st_mode or 0)
    except OSError:
        return False


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


def _open_cif(path, **kwargs):
    return AseReader(path, format="cif", **kwargs)


def _open_vaspout(path, **kwargs):
    """vaspout.h5, or when VASP has not written a readable one yet (still running, or killed
    before closing it) the vasprun.xml, OUTCAR or XDATCAR of the same run next to it."""
    try:
        return VaspoutReader(path, **kwargs)
    except IncompleteFileError as exc:
        folder = os.path.dirname(os.path.abspath(str(path)))
        for name in ("vasprun.xml", "OUTCAR", "XDATCAR"):
            other = os.path.join(folder, name)
            if not os.path.isfile(other):
                continue
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IncompleteFileWarning)
                try:
                    reader = open_reader(other)
                except (OSError, ValueError):
                    continue
            for w in caught:
                if not issubclass(w.category, IncompleteFileWarning):
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            note = f"not readable yet, read from {name} instead"
            reader._set_incomplete(note + (f": {reader.incomplete}" if reader.incomplete else ""), stacklevel=4)
            return reader
        raise exc from None


def _open_md_trajectory(path, format=None, topology=None, **kwargs):
    fmt = (format or "").upper()
    fmt = {"NETCDF": "NCDF", "MDANALYSIS": None, "": None}.get(fmt, fmt)
    return MDAnalysisReader(path, topology=topology, format=fmt)


def builtin_readers():
    """ewald's own formats, registered in :mod:`ewald.plugins` like any reader plugin."""
    from ..plugins import ReaderPlugin

    B = "built-in"
    return [
        ReaderPlugin("neb", NebReader, (), is_neb,
                     "NEB band, one frame per image: VASP NEB folder (00..NN) or LAMMPS neb run folder "
                     "(detected), or format='neb' with one file per image (dump.neb.*)", B, 4, _NEB_FORMATS[1:]),
        ReaderPlugin("cube", _open_cube, ("*.cube", "*.cub"), is_cube,
                     "Gaussian cube volumetric data (atoms + grid)", B, 5, ("gaussian-cube",), category="volumetric"),
        ReaderPlugin("vasp-volumetric", _open_vasp_volumetric,
                     ("CHGCAR*", "CHG*", "PARCHG*", "AECCAR*", "LOCPOT*", "ELFCAR*"), is_vasp_volumetric,
                     "VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR (atoms + grid)", B, 6,
                     ("vasp-chgcar", "chgcar", "vasp-locpot", "locpot", "vasp-elfcar", "elfcar"),
                     takes_format=True, category="volumetric"),
        ReaderPlugin("vaspout", _open_vaspout, ("vaspout*.h5",), is_vaspout,
                     "VASP 6 vaspout.h5: ionic steps with forces, energies, stress, velocities, moments, "
                     "local potential grid, DOS/PDOS and bands, lazy; a file VASP has not closed yet is read from "
                     "the vasprun.xml/OUTCAR next to it; needs h5py", B, 7, ("vasp-h5", "vaspout.h5")),
        ReaderPlugin("vaspwave", VaspwaveReader, ("vaspwave*.h5",), is_vaspwave,
                     "VASP 6 vaspwave.h5 charge density (structure from vaspout.h5 next to it); needs h5py",
                     B, 7, ("vaspwave.h5",), category="volumetric"),
        ReaderPlugin("mdtraj-h5", MdtrajH5Reader, ("*.h5",), is_mdtraj_h5,
                     "MDTraj/OpenMM HDF5 trajectory (HDF5Reporter), topology and bonds, lazy; needs h5py", B, 7,
                     ("mdtraj", "openmm-h5")),
        ReaderPlugin("h5md", H5mdReader, ("*.h5md", "*.h5"), is_h5md,
                     "H5MD trajectory (LAMMPS dump h5md, MDAnalysis, ...), lazy; needs h5py", B, 7),
        ReaderPlugin("xsf", XsfReader, ("*.xsf", "*.axsf"), is_xsf,
                     "XCrySDen XSF/AXSF: structure, ANIMSTEPS trajectory, DATAGRID_3D grids", B, 8),
        ReaderPlugin("lammps-dump", LammpsDumpReader, ("*.lammpstrj", "*.dump"), is_lammps_dump,
                     "LAMMPS text dump (atom/custom), lazy", B, 10, ("lammps/dump", "lammps-dump-text")),
        ReaderPlugin("xdatcar", XdatcarReader, ("XDATCAR*",), is_xdatcar,
                     "VASP XDATCAR, fixed or variable cell, lazy", B, 11, ("vasp-xdatcar",)),
        ReaderPlugin("lammps-data", _open_lammps_data, ("*.data",), is_lammps_data,
                     "LAMMPS data file", B, 12, ("lammps/data", "lammps-data-native")),
        ReaderPlugin("ase-db", AseDbReader, ("*.db",), is_ase_db,
                     "ASE database (SQLite .db or .json), one row per frame, lazy; selection= query", B, 13,
                     ("db", "ase.db")),
        ReaderPlugin("ase-traj", AseTrajectoryReader, ("*.traj",), is_ase_traj,
                     "ASE binary trajectory (.traj), lazy", B, 14, ("traj",)),
        ReaderPlugin("md-trajectory", _open_md_trajectory, ("*.xtc", "*.trr", "*.dcd", "*.nc", "*.ncdf"),
                     is_md_trajectory, "GROMACS XTC/TRR, DCD, AMBER NetCDF via MDAnalysis, lazy; "
                     "topology= (.gro/.pdb/.tpr/...) for atom types", B, 15,
                     ("xtc", "trr", "dcd", "netcdf", "ncdf", "mdanalysis"), takes_format=True),
        ReaderPlugin("extxyz", ExtxyzReader, ("*.xyz", "*.extxyz"), is_extxyz,
                     "XYZ / extended XYZ (Lattice=, Properties=), lazy", B, 20, ("xyz", "ext-xyz")),
        ReaderPlugin("cif", _open_cif, ("*.cif",), None, "Crystallographic Information File via ASE", B, 90),
        ReaderPlugin("outcar", OutcarReader, ("OUTCAR*",), is_outcar,
                     "VASP OUTCAR: ionic steps with forces, energies, stress, magnetic moments, lazy", B, 16,
                     ("vasp-outcar",)),
        ReaderPlugin("vasprun", VasprunReader, ("vasprun*.xml",), is_vasprun,
                     "VASP vasprun.xml: ionic steps with cell, forces, stress, energies, DOS/PDOS and bands, lazy", B, 17,
                     ("vasp-vasprun",)),
        ReaderPlugin("lammps-data-ase", _open_ase_lammps_data, (), None,
                     "LAMMPS data file via ASE", B, None),
        ReaderPlugin("ase", _open_ase, ("*POSCAR*", "*CONTCAR*", "*.vasp", "*.poscar", "OUTCAR*", "vasprun*.xml",
                                        "*.xyz", "*.extxyz", "*.cif", "*.pdb", "*.gro", "*.gen", "*.res",
                                        "*.cell", "*.castep", "*.pwi", "*.pwo", "geometry.in"), lambda path: True,
                     "anything ase.io.read understands (POSCAR, OUTCAR, vasprun.xml, extxyz, cif, ...); "
                     "other format= names are passed to ASE", B, 100, takes_format=True),
    ]


def import_file(path, format: str | None = None, **kwargs):
    """Create a Pipeline reading ``path`` (wildcards allowed: ``dump.*.lammpstrj``).

    Several files make one trajectory: a wildcard (natural order, ``dump.200`` before
    ``dump.1000``) or a list such as ``["run1/dump.lammpstrj", "run2/dump.lammpstrj"]`` (read in
    the order given). ``join="timestep"`` (default) joins the segments of a restarted run: a
    later file wins from the Timestep it starts at, so the frames an earlier file wrote past the
    restart point are dropped; ``join="concat"`` keeps every frame (see :class:`FileSource`).

    ``format`` names a built-in format, an ASE format, or a reader plugin (see
    :mod:`ewald.plugins`); without it the format is detected from the file.
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
    A NEB run (a VASP folder with image folders ``00 .. NN``, or ``format="neb"`` with a
    wildcard naming one file per image) is read with one frame per image, and the pipeline
    starts with ``NebEnergyProfile`` and ``OverlayFrames`` (every image see-through behind the
    current one; see :mod:`ewald.io.neb`).
    """
    from ..pipeline import Pipeline

    pipe = Pipeline(FileSource(path, format, **kwargs))
    if any(isinstance(r, NebReader) for r in pipe.source.readers):
        from ..modifiers import NebEnergyProfile, OverlayFrames

        pipe.modifiers += [NebEnergyProfile(), OverlayFrames()]
    return pipe


def _write_poscar(path, data, sort: bool = True, direct: bool = True, **kwargs):
    import ase.io

    from .extxyz import _to_atoms

    atoms = _to_atoms(data)
    if sort:  # one species block per element, in order of first appearance (stable)
        first = {s: i for i, s in reversed(list(enumerate(atoms.get_chemical_symbols())))}
        atoms = atoms[sorted(range(len(atoms)), key=lambda i: first[atoms[i].symbol])]
    ase.io.write(os.fspath(path), atoms, format="vasp", direct=direct, **kwargs)


def _write_with_ase(format):
    def write(path, frames, **kwargs):
        import ase.io

        from .extxyz import _to_atoms

        images = [_to_atoms(d) for d in frames]
        ase.io.write(os.fspath(path), images if len(images) > 1 else images[0], format=format, **kwargs)

    return write


def _write_ase_fallback(path, frames, format=None, **kwargs):
    return _write_with_ase(format)(path, frames, **kwargs)


def _write_ase_traj(path, frames, **kwargs):
    from ase.io.trajectory import Trajectory

    from .extxyz import _to_atoms

    with Trajectory(os.fspath(path), "w", **kwargs) as traj:
        for d in frames:
            traj.write(_to_atoms(d))


def _write_ase_db(path, frames, append: bool = True, **kwargs):
    import re

    import ase.db
    from ase.db.core import reserved_keys

    from .extxyz import _to_atoms

    db = ase.db.connect(os.fspath(path), append=append)
    ok = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for d in frames:
        atoms = _to_atoms(d)
        pairs = {k: v.item() if hasattr(v, "item") else v for k, v in atoms.info.items()
                 if ok.match(k) and k not in reserved_keys and isinstance(getattr(v, "item", lambda: v)(),
                                                                       (bool, int, float, str))}
        atoms.info = {}
        db.write(atoms, key_value_pairs=pairs, **kwargs)


def _write_md_trajectory(path, frames, format=None, **kwargs):
    fmt = (format or "").upper()
    return write_md_trajectory(path, frames, format={"MD-TRAJECTORY": None, "": None}.get(fmt, fmt), **kwargs)


def _write_cube(path, data, **kwargs):
    write_cube(path, data, **kwargs)


def _write_vasp_volumetric(path, data, format=None, **kwargs):
    write_vasp_volumetric(path, data, **kwargs)


def builtin_writers():
    """ewald's own writers, registered in :mod:`ewald.plugins` like any writer plugin."""
    from ..plugins import WriterPlugin

    B = "built-in"
    return [
        WriterPlugin("cube", _write_cube, ("*.cube",), "Gaussian cube (one grid property + atoms)", B, 5,
                     ("gaussian-cube",), multi_frame=False, category="volumetric"),
        WriterPlugin("vasp-volumetric", _write_vasp_volumetric, ("CHGCAR*", "LOCPOT*", "ELFCAR*", "PARCHG*"),
                     "VASP CHGCAR-style grid (VESTA reads it)", B, 6, ("vasp-chgcar", "chgcar"),
                     takes_format=True, multi_frame=False, category="volumetric"),
        WriterPlugin("h5md", write_h5md, ("*.h5md", "*.h5"), "H5MD trajectory, streamed; needs h5py", B, 7),
        WriterPlugin("xsf", write_xsf, ("*.xsf", "*.axsf"), "XCrySDen XSF (structure, ANIMSTEPS, grid)", B, 8),
        WriterPlugin("lammps-dump", write_lammps_dump, ("*.lammpstrj", "*.dump"), "LAMMPS text dump", B, 10,
                     ("lammps/dump",)),
        WriterPlugin("poscar", _write_poscar, ("POSCAR*", "CONTCAR*", "*.vasp", "*.poscar"),
                     "VASP POSCAR (species grouped, Direct)", B, 11, ("vasp",), multi_frame=False),
        WriterPlugin("xdatcar", _write_with_ase("vasp-xdatcar"), ("XDATCAR*",), "VASP XDATCAR", B, 11,
                     ("vasp-xdatcar",)),
        WriterPlugin("lammps-data", write_lammps_data, ("*.data", "*.lmp"),
                     "LAMMPS data file (atomic/charge/molecular/full)", B, 12, ("lammps/data",), multi_frame=False),
        WriterPlugin("ase-db", _write_ase_db, ("*.db",), "ASE database, one row per frame", B, 13, ("db", "ase.db")),
        WriterPlugin("ase-traj", _write_ase_traj, ("*.traj",), "ASE binary trajectory", B, 14, ("traj",)),
        WriterPlugin("md-trajectory", _write_md_trajectory, ("*.xtc", "*.trr", "*.dcd"),
                     "GROMACS XTC/TRR or DCD via MDAnalysis, streamed", B, 15, ("xtc", "trr", "dcd"),
                     takes_format=True),
        WriterPlugin("extxyz", write_extxyz, ("*.extxyz", "*.xyz"), "extended XYZ, streamed", B, 20, ("xyz",)),
        WriterPlugin("cif", _write_with_ase("cif"), ("*.cif",), "CIF via ASE", B, 90),
        WriterPlugin("ase", _write_ase_fallback, (), "any format ase.io.write knows (format= name)", B, 100,
                     takes_format=True),
    ]


def export_file(source, path, format: str | None = None, frames=None, **kwargs):
    """Write a DataCollection, a list of them, or a Pipeline's frames to ``path``.

    The writer comes from ``format`` (a writer name or alias such as ``"lammps/dump"``,
    ``"poscar"``, ``"xtc"``; unknown names go to ``ase.io.write``) or from the file name
    (``*.extxyz``, ``POSCAR``, ``*.data``, ``*.h5md``, ``*.xtc``, ``*.xsf``, ``*.cube``, ...;
    ``ewald plugins`` lists them, writer plugins included). Pipeline frames are computed and
    written one at a time. ``frames=`` picks frames of a Pipeline; one-frame formats (POSCAR,
    LAMMPS data, CHGCAR, cube) need exactly one, e.g. ``frames=[-1]``. Grid writers take
    ``property="Charge Density"`` (and ``grid=``); other keyword arguments go to the writer
    (``columns=`` for LAMMPS dumps, ``atom_style=`` for LAMMPS data).
    """
    from .. import plugins
    from ..pipeline import Pipeline

    if isinstance(source, Pipeline):
        idx = range(source.num_frames) if frames is None else frames
        items = (source.compute(i) for i in idx)
    elif isinstance(source, DataCollection):
        items = [source]
    else:
        items = list(source)
        if frames is not None:
            items = [items[i] for i in frames]
    return plugins.write_file(path, items, format=format, **kwargs)


__all__ = ["FileSource", "StaticSource", "IncompleteFileError", "IncompleteFileWarning", "import_file", "export_file", "open_reader", "builtin_readers",
           "builtin_writers", "ExtxyzReader", "XsfReader", "H5mdReader", "MDAnalysisReader", "AseDbReader",
           "AseTrajectoryReader", "write_extxyz", "write_xsf", "write_h5md", "write_md_trajectory",
           "write_lammps_data", "write_lammps_dump",
           "LammpsDumpReader", "LammpsDataReader", "XdatcarReader", "AseReader", "VolumetricReader",
           "OutcarReader", "VasprunReader", "VaspoutReader", "VaspwaveReader", "MdtrajH5Reader", "NebReader",
           "read_lammps_neb_log", "Thermo", "read_lammps_log", "read_oszicar", "read_thermo",
           "read_vasp_volumetric", "read_cube", "write_vasp_volumetric", "write_cube"]
