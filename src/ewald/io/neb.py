"""Nudged elastic band (NEB) runs read as a trajectory: one frame per image.

Three layouts are recognized:

- a VASP (VTST) NEB folder with numbered image folders ``00``, ``01``, ..., ``NN``. Each
  image is read from its OUTCAR (last ionic step: positions, forces, energies) when there
  is one, else from CONTCAR, else from POSCAR; endpoints often have only a POSCAR, and
  their energy is then unknown unless ``energies=`` gives it. An endpoint with both an OUTCAR
  and a POSCAR takes its atoms from the POSCAR and its energy from the OUTCAR.
- a set of files with one file per image, given as a wildcard (``dump.neb.*``, LAMMPS
  ``neb`` with one dump per replica): the last frame of each file is the image, files in
  natural order (``dump.neb.10`` after ``dump.neb.9``). The energies come from the NEB
  table of the master ``log.lammps`` (``PE1 .. PEN`` of its last row), picked up from the
  same folder or named with ``log=``.
- a LAMMPS NEB run folder: its master ``log.lammps`` and one dump per replica, numbered
  like the replicas (``dump.neb.1 .. dump.neb.N`` from ``dump ... dump.neb.$u``).

Every frame gets ``Timestep`` = image number and, when known, the image energy as the
``Energy`` attribute (eV). The :class:`~ewald.modifiers.NebEnergyProfile` modifier turns the
band into the energy profile and barrier; :func:`ewald.import_file` adds it for these sources.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..data import DataCollection
from .base import FrameReader, IncompleteFileError, open_binary


def _natural_key(path: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def vasp_neb_images(folder) -> list[str]:
    """Image folders ``00 .. NN`` of a VASP NEB run (empty if ``folder`` is not one)."""
    folder = str(folder)
    if not os.path.isdir(folder):
        return []
    names = sorted(n for n in os.listdir(folder) if n.isdigit() and os.path.isdir(os.path.join(folder, n)))
    if len(names) < 3 or [int(n) for n in names] != list(range(len(names))):
        return []
    dirs = [os.path.join(folder, n) for n in names]
    if not all(_image_file(d) for d in dirs):
        return []
    return dirs


def _image_file(folder: str) -> str | None:
    for name in ("OUTCAR", "CONTCAR", "POSCAR"):
        for cand in (name, name + ".gz", name + ".xz", name + ".bz2"):
            p = os.path.join(folder, cand)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return p
    return None


def lammps_neb_images(folder) -> tuple[list[str], str | None]:
    """Replica files and master log of a LAMMPS NEB run folder: a ``log.lammps`` with the
    NEB table and one file per replica named with the replica number (``dump.neb.1`` ..
    ``dump.neb.N``, any prefix/suffix), as many as the table has images."""
    folder = str(folder)
    log = os.path.join(folder, "log.lammps")
    if not os.path.isdir(folder) or not os.path.isfile(log) or not _is_lammps_neb_log(log):
        return [], None
    n = len(read_lammps_neb_log(log)["Energy"])
    groups: dict[tuple[str, str], dict[int, str]] = {}
    for name in os.listdir(folder):
        m = re.fullmatch(r"(.*?)(\d+)((?:\.[A-Za-z]\w*)*)", name)
        if m and not name.startswith(("log.lammps", "screen")) and os.path.isfile(os.path.join(folder, name)):
            groups.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = os.path.join(folder, name)
    from .lammps_dump import is_lammps_dump

    for files in groups.values():
        ids = sorted(files)
        if len(ids) == n and ids == list(range(ids[0], ids[0] + n)) and is_lammps_dump(files[ids[0]]):
            return [files[i] for i in ids], log
    return [], log


def is_neb(path) -> bool:
    """A VASP NEB folder (image folders ``00 .. NN``) or a LAMMPS NEB run folder (master
    ``log.lammps`` plus one dump per replica); wildcards need ``format="neb"``."""
    return bool(vasp_neb_images(path)) or bool(lammps_neb_images(path)[0])


def read_lammps_neb_log(path) -> dict:
    """Last row of the NEB table in a LAMMPS master log (``neb`` run on partitions):
    ``{"Reaction Coordinate": RD1..RDN (normalized), "Energy": PE1..PEN, "Step": step}``."""
    cols, last = None, None
    with open_binary(path) as fh:
        for line in fh:
            tok = line.split()
            if tok[:2] == [b"Step", b"MaxReplicaForce"]:
                cols = [t.decode() for t in tok]
                continue
            if cols is not None and len(tok) == len(cols):
                try:
                    last = [float(t) for t in tok]
                except ValueError:
                    pass
    if last is None:
        raise ValueError(f"{path}: no NEB table (Step MaxReplicaForce ...); not a LAMMPS NEB master log?")
    row = dict(zip(cols, last))
    n = max(int(m.group(1)) for c in cols if (m := re.fullmatch(r"PE(\d+)", c)))
    return {"Step": int(row["Step"]),
            "Reaction Coordinate": np.array([row[f"RD{i}"] for i in range(1, n + 1)]),
            "Energy": np.array([row[f"PE{i}"] for i in range(1, n + 1)])}


def _is_lammps_neb_log(path) -> bool:
    try:
        with open_binary(path) as fh:
            return any(line.lstrip().startswith(b"Step") and b"MaxReplicaForce" in line for line in fh)
    except OSError:
        return False


class NebReader(FrameReader):
    """One frame per NEB image; see the module docstring. ``frame`` picks the frame of
    each image file (default: the last one), ``energies`` overrides the image energies
    (None or NaN keeps an image's own), ``log`` names the LAMMPS master log."""

    def __init__(self, path, frame: int = -1, energies=None, log: str | None = None, **kwargs):
        self.path = str(path)
        self.frame = frame
        dirs = vasp_neb_images(self.path)
        lammps, found_log = lammps_neb_images(self.path) if not dirs else ([], None)
        # an endpoint's OUTCAR is often a separate relaxation (atoms in another order); VASP's
        # NEB uses its POSCAR, so that gives the structure and the OUTCAR the energy (as pymatgen)
        self._endpoint_poscars: dict[int, str] = {}
        if dirs:
            self.files = [_image_file(d) for d in dirs]
            for i in (0, len(dirs) - 1):
                poscar = os.path.join(dirs[i], "POSCAR")
                if os.path.basename(self.files[i]).startswith("OUTCAR") and os.path.isfile(poscar):
                    self._endpoint_poscars[i] = poscar
        elif lammps:
            self.files, log = lammps, log or found_log
        else:
            self.files = sorted(glob.glob(self.path), key=_natural_key)
            if os.path.isdir(self.path):
                raise ValueError(f"{path}: not a NEB folder (needs image folders 00, 01, ... with "
                                 f"OUTCAR, CONTCAR or POSCAR, or a LAMMPS NEB log.lammps with one dump "
                                 f"per replica)")
            if len(self.files) < 2:
                raise ValueError(f"{path}: a NEB band needs at least two image files, found {len(self.files)}")
        self.kwargs = kwargs
        self.log = log
        self.energies = None
        if energies is not None:
            self.energies = np.asarray(energies, dtype=float)
        elif not dirs:
            if log is None:
                cand = os.path.join(os.path.dirname(os.path.abspath(self.files[0])), "log.lammps")
                log = cand if os.path.isfile(cand) and _is_lammps_neb_log(cand) else None
            if log is not None:
                self.log = log
                self.energies = read_lammps_neb_log(log)["Energy"]
        if self.energies is not None and len(self.energies) != len(self.files):
            raise ValueError(f"{len(self.energies)} energies for {len(self.files)} NEB images")
        self._readers: dict[int, FrameReader] = {}

    def __len__(self):
        return len(self.files)

    def _reader(self, i: int) -> FrameReader:
        r = self._readers.get(i)
        if r is None:
            from . import open_reader

            try:
                r = open_reader(self.files[i], **self.kwargs)
            except IncompleteFileError:
                # an image whose OUTCAR has no finished ionic step yet: its CONTCAR or POSCAR
                folder = os.path.dirname(self.files[i])
                start = next((os.path.join(folder, n) for n in ("CONTCAR", "POSCAR")
                              if os.path.isfile(os.path.join(folder, n)) and os.path.getsize(os.path.join(folder, n))),
                             None)
                if start is None:
                    raise
                r = open_reader(start)
                r.incomplete = f"{os.path.basename(self.files[i])} has no finished ionic step yet, " \
                               f"{os.path.basename(start)} shown"
            self._readers[i] = r
        return r

    @property
    def incomplete(self) -> str | None:
        """Images whose files are still being written (of the images read so far)."""
        notes = [f"image {i:02d}: {r.incomplete}" for i, r in sorted(self._readers.items())
                 if getattr(r, "incomplete", None)]
        return "; ".join(notes) or None

    def refresh(self) -> int | None:
        """A running NEB: images whose files grew (or that can now be read) are read again."""
        changed = []
        for i, r in list(self._readers.items()):
            if os.path.abspath(self.files[i]) != os.path.abspath(getattr(r, "path", self.files[i])):
                from . import open_reader

                try:  # shown from CONTCAR/POSCAR: has the OUTCAR finished a step now?
                    self._readers[i] = open_reader(self.files[i], **self.kwargs)
                except (OSError, ValueError):
                    continue
                changed.append(i)
            elif r.refresh() is not None:
                changed.append(i)
        return min(changed) if changed else None

    def _endpoint(self, index: int, outcar: DataCollection) -> DataCollection:
        """The POSCAR of endpoint ``index`` with the energies of its OUTCAR, and the OUTCAR's
        forces too when its atoms are the POSCAR's."""
        from . import open_reader

        data = open_reader(self._endpoint_poscars[index]).read(0)
        if data.count == outcar.count:
            frac = data.cell.to_fractional(np.asarray(outcar.positions)) - data.cell.to_fractional(np.asarray(data.positions))
            if "Force" in outcar and np.abs((frac - np.round(frac)) @ data.cell.matrix).max() < 1e-3:
                data.particles["Force"] = outcar["Force"]
        data.attributes.update({k: v for k, v in outcar.attributes.items() if k != "Timestep"})
        return data

    def read(self, index: int) -> DataCollection:
        r = self._reader(index)
        data = r.read(self.frame % len(r))
        if index in self._endpoint_poscars:
            data = self._endpoint(index, data)
        data.attributes["Timestep"] = index
        data.attributes["NEB.Source"] = self.files[index]
        if self.energies is not None:
            e = float(self.energies[index])
            if e == e:  # None/NaN keeps the image file's own energy
                data.attributes["Energy"] = e
        return data
