"""Native lazy readers for VASP results: OUTCAR, vasprun.xml and OSZICAR.

OUTCAR and vasprun.xml are indexed by searching only for the lines that matter
(:func:`~ewald.io.base.grep_lines`, done by ``grep`` on the host for a remote file),
so the per-step energies are known as soon as the file is open and each ionic
step's structure, forces and magnetic moments are read when that frame is needed.

Per frame (OVITO-style names): particle properties ``Force`` (eV/A) and
``Magnetic Moment`` (mu_B; ``N x 3`` for non-collinear runs, OUTCAR only), and the
attributes ``Energy`` (energy(sigma->0), eV), ``Free Energy`` (TOTEN, eV),
``Pressure`` (external pressure, kB), ``Stress`` (6 Voigt components xx yy zz yz xz xy
in eV/A^3, ASE's sign and order), ``Magnetization`` (total, mu_B) and, for MD,
``Temperature`` (K), ``Kinetic Energy`` and ``Total Energy`` (eV). ``Timestep`` is the
ionic step number counted from 1, as in XDATCAR and OSZICAR. ``reader.thermo`` holds
the same values for every ionic step as a table.
"""
from __future__ import annotations

import os
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader, floats, grep_lines, head_lines, open_binary, strip_compression
from .xdatcar import species_name

_KB_TO_EV_A3 = 1e-1 / 160.21766208  # kBar -> GPa -> eV/A^3 (ase.units.GPa)


def _voigt_from_kb(xx, yy, zz, xy, yz, zx):
    """VASP stress in kB (OUTCAR order XX YY ZZ XY YZ ZX) -> ASE stress (eV/A^3, xx yy zz yz xz xy)."""
    return -np.array([xx, yy, zz, yz, zx, xy]) * _KB_TO_EV_A3


def _table(rows: list[dict], key="Step") -> dict:
    names = [key] + [k for k in dict.fromkeys(k for r in rows for k in r) if k != key]
    out = {}
    for k in names:
        col = [r.get(k, np.nan) for r in rows]
        out[k] = np.array(col, dtype=int if k == key else float)
    return out


# ----------------------------------------------------------------------- OUTCAR
def is_outcar(path) -> bool:
    if os.path.basename(strip_compression(path)).upper().startswith("OUTCAR"):
        return True
    head = head_lines(path, 1)
    return bool(head) and head[0].strip().startswith(b"vasp.") and b"(build" in head[0]


_OUTCAR_KEYS = "|".join((
    r"Iteration +[0-9]+\(", "direct lattice vectors", "POSITION +TOTAL-FORCE", "free  energy   TOTEN",
    "energy  without entropy", "^  in kB", "external pressure", r"^ magnetization \([xyz]\)",
    "number of electron .*magnetization", "kinetic energy EKIN", "total energy   ETOTAL", "POTCAR:",
    "ions per type", "NIONS =",
))


class OutcarReader(FrameReader):
    """VASP OUTCAR: every ionic step with forces, energies, stress and magnetic moments, lazy."""

    def __init__(self, path):
        self.path = str(path)
        potcars, counts, natoms = [], None, None
        steps: dict[int, dict] = {}
        ionic, lattice = 0, None
        for off, line in grep_lines(self.path, _OUTCAR_KEYS):
            if b"Iteration" in line:
                ionic = int(re.search(rb"Iteration +(\d+)\(", line).group(1))
                continue
            rec = steps.setdefault(ionic, {"values": {}, "mag": {}})
            v = rec["values"]
            if b"direct lattice vectors" in line:
                lattice = off
            elif b"TOTAL-FORCE" in line:
                rec["pos"], rec["lat"] = off, lattice
            elif b"free  energy   TOTEN" in line:
                v["Free Energy"] = floats(line.split(b"=")[-1])[0]
            elif b"energy  without entropy" in line:
                v["Energy"] = floats(line)[-1]
            elif line.startswith(b"  in kB"):
                f = floats(line)
                if len(f) == 6:
                    rec["kb"] = f
            elif b"external pressure" in line:
                v["Pressure"] = floats(line)[0]
            elif line.startswith(b" magnetization ("):
                rec["mag"]["xyz".index(chr(line[16]))] = off
            elif b"number of electron" in line:
                f = floats(line.split(b"magnetization")[-1])
                if f:
                    v["Magnetization"] = f[0] if len(f) == 1 else float(np.linalg.norm(f[:3]))
            elif b"kinetic energy EKIN" in line:
                f = floats(line.split(b"=")[-1])
                v["Kinetic Energy"] = f[0]
                if len(f) > 1:
                    v["Temperature"] = f[1]
            elif b"total energy   ETOTAL" in line:
                v["Total Energy"] = floats(line.split(b"=")[-1])[0]
            elif b"POTCAR:" in line and counts is None:
                potcars.append(line.split(b":", 1)[1].split())
            elif b"ions per type" in line and counts is None:
                counts = [int(x) for x in line.split(b"=")[1].split()]
            elif b"NIONS =" in line:
                natoms = int(line.split(b"NIONS =")[1].split()[0])
        if counts is None:
            raise ValueError(f"{path}: no 'ions per type' line; not an OUTCAR?")
        if len(potcars) == 2 * len(counts) and potcars[: len(counts)] == potcars[len(counts):]:
            potcars = potcars[: len(counts)]
        species = [species_name(p[1].decode() if len(p) > 1 else p[0].decode()) for p in potcars[: len(counts)]]
        if len(species) != len(counts):
            raise ValueError(f"{path}: {len(potcars)} POTCAR lines for {len(counts)} species")
        self.types = np.repeat(species, counts)
        self.natoms = natoms or int(sum(counts))
        self._steps = [(n, rec) for n, rec in sorted(steps.items()) if "pos" in rec]
        for _, rec in self._steps:
            if "kb" in rec:
                rec["values"]["Stress"] = _voigt_from_kb(*rec["kb"])
                rec["values"].setdefault("Pressure", sum(rec["kb"][:3]) / 3)
        if self._steps:
            try:
                self.read(len(self._steps) - 1)
            except (ValueError, IndexError):
                warnings.warn(f"{path}: last ionic step is incomplete and was skipped", stacklevel=2)
                self._steps.pop()
        if not self._steps:
            raise ValueError(f"{path}: no ionic steps with positions found")
        self.thermo = _table([{"Step": n, **{k: x for k, x in rec["values"].items() if np.isscalar(x)}}
                              for n, rec in self._steps])

    def __len__(self):
        return len(self._steps)

    def _block(self, fh, offset, skip, n):
        fh.seek(offset)
        for _ in range(skip):
            fh.readline()
        rows = [floats(fh.readline()) for _ in range(n)]
        if any(not r for r in rows):
            raise ValueError(f"{self.path}: truncated block at byte {offset}")
        return rows

    def read(self, index: int) -> DataCollection:
        step, rec = self._steps[index]
        n = self.natoms
        with open_binary(self.path) as fh:
            if rec["lat"] is None:
                raise ValueError(f"{self.path}: no lattice vectors before ionic step {step}")
            lattice = np.array([r[:3] for r in self._block(fh, rec["lat"], 1, 3)])
            table = np.array([r[:6] for r in self._block(fh, rec["pos"], 2, n)])
            mags = {axis: np.array([r[-1] for r in self._block(fh, off, 4, n)])
                    for axis, off in sorted(rec["mag"].items())}
        if table.shape != (n, 6):
            raise ValueError(f"{self.path}: bad POSITION block in ionic step {step}")
        data = DataCollection(table[:, :3], self.types, Cell(lattice, np.zeros(3), np.ones(3, bool)))
        data.particles["Particle Identifier"] = np.arange(1, n + 1)
        data.particles["Force"] = table[:, 3:]
        if mags:
            data.particles["Magnetic Moment"] = mags[0] if len(mags) == 1 else \
                np.stack([mags.get(a, np.zeros(n)) for a in range(3)], axis=1)
        data.attributes["Timestep"] = step
        data.attributes.update(rec["values"])
        return data


# ----------------------------------------------------------------------- vasprun.xml
def is_vasprun(path) -> bool:
    base = os.path.basename(strip_compression(path)).lower()
    if base.startswith("vasprun") and base.endswith(".xml"):
        return True
    return any(b"<modeling>" in line for line in head_lines(path, 4))


_VASPRUN_KEYS = "|".join((
    "<calculation>", "</calculation>", "<structure", "<varray name=\"forces\"", "<varray name=\"stress\"",
    "<scstep>", "</scstep>", "<atominfo>", "<i name=\"(e_fr_energy|e_0_energy|kinetic|total)\"",
))
_VASPRUN_VALUES = {b"e_fr_energy": "Free Energy", b"e_0_energy": "Energy", b"kinetic": "Kinetic Energy",
                   b"total": "Total Energy"}


class VasprunReader(FrameReader):
    """VASP vasprun.xml: every ionic step with cell, forces, stress and energies, lazy."""

    def __init__(self, path):
        self.path = str(path)
        self._steps: list[dict] = []
        atominfo, calc, in_scstep = None, None, False
        for off, line in grep_lines(self.path, _VASPRUN_KEYS):
            if b"<calculation>" in line:
                calc = {"values": {}}
            elif b"</calculation>" in line:
                if calc is not None and "structure" in calc:
                    self._steps.append(calc)
                calc = None
            elif b"<atominfo>" in line:
                atominfo = off
            elif calc is None:
                continue
            elif b"<scstep>" in line:
                in_scstep = True
            elif b"</scstep>" in line:
                in_scstep = False
            elif b"<structure" in line:
                calc["structure"] = off
            elif b"\"forces\"" in line:
                calc["forces"] = off
            elif b"\"stress\"" in line:
                calc["stress"] = off
            elif not in_scstep:
                name = re.search(rb'name="(\w+)"', line).group(1)
                try:
                    calc["values"][_VASPRUN_VALUES[name]] = float(line.split(b">", 1)[1].split(b"<")[0])
                except ValueError:  # ***** overflow
                    pass
        if calc is not None and "structure" in calc:  # running job: the open last step
            self._steps.append(calc)
        if atominfo is None:
            raise ValueError(f"{path}: no <atominfo>; not a vasprun.xml?")
        root = self._element(atominfo, b"</atominfo>")
        rows = root.find("array[@name='atoms']/set")
        self.types = np.array([species_name(rc[0].text.strip()) for rc in rows], dtype=str)
        for rec in self._steps:
            if "stress" in rec:
                try:
                    kb = self._varray(rec["stress"])
                except ValueError:
                    continue
                rec["values"]["Stress"] = _voigt_from_kb(kb[0, 0], kb[1, 1], kb[2, 2], kb[0, 1], kb[1, 2], kb[2, 0])
                rec["values"]["Pressure"] = float(np.trace(kb) / 3)
        if self._steps:
            try:
                self.read(len(self._steps) - 1)
            except (ValueError, ET.ParseError, IndexError):
                warnings.warn(f"{path}: last ionic step is incomplete and was skipped", stacklevel=2)
                self._steps.pop()
        if not self._steps:
            raise ValueError(f"{path}: no ionic steps found")
        self.thermo = _table([{"Step": i + 1, **{k: x for k, x in rec["values"].items() if np.isscalar(x)}}
                              for i, rec in enumerate(self._steps)])

    def __len__(self):
        return len(self._steps)

    def _element(self, offset, closing: bytes):
        with open_binary(self.path) as fh:
            fh.seek(offset)
            lines = []
            for line in fh:
                lines.append(line)
                if closing in line:
                    break
            else:
                raise ValueError(f"{self.path}: element at byte {offset} is not closed")
        return ET.fromstring(b"".join(lines))

    def _varray(self, offset) -> np.ndarray:
        el = self._element(offset, b"</varray>")
        return np.array([v.text.split() for v in el.findall("v")], dtype=float)

    def read(self, index: int) -> DataCollection:
        rec = self._steps[index]
        st = self._element(rec["structure"], b"</structure>")
        basis = np.array([v.text.split() for v in st.findall("crystal/varray[@name='basis']/v")], dtype=float)
        frac = np.array([v.text.split() for v in st.findall("varray[@name='positions']/v")], dtype=float)
        n = len(self.types)
        if basis.shape != (3, 3) or frac.shape != (n, 3):
            raise ValueError(f"{self.path}: bad structure in ionic step {index + 1}")
        cell = Cell(basis, np.zeros(3), np.ones(3, bool))
        data = DataCollection(cell.to_cartesian(frac), self.types, cell)
        data.particles["Particle Identifier"] = np.arange(1, n + 1)
        if "forces" in rec:
            forces = self._varray(rec["forces"])
            if forces.shape == (n, 3):
                data.particles["Force"] = forces
        data.attributes["Timestep"] = index + 1
        data.attributes.update(rec["values"])
        return data


# ----------------------------------------------------------------------- OSZICAR
_OSZICAR_NAMES = {"F": "Free Energy", "E0": "Energy", "T": "Temperature", "E": "Total Energy",
                  "EK": "Kinetic Energy", "SP": "Nose Potential", "SK": "Nose Kinetic", "mag": "Magnetization"}
_OSZICAR_KEY = re.compile(rb"([A-Za-z][A-Za-z0-9]*(?: E)?)\s*=")


def is_oszicar(path) -> bool:
    return os.path.basename(strip_compression(path)).upper().startswith("OSZICAR")


def read_oszicar(path) -> dict:
    """OSZICAR ionic-step lines as a table: ``Step`` plus ``Energy`` (E0), ``Free Energy`` (F),
    ``Magnetization`` and, for MD, ``Temperature``, ``Total Energy``, ``Kinetic Energy``, ..."""
    rows = []
    for _, line in grep_lines(path, r"^ *[0-9]+ +(T|F)="):
        parts = _OSZICAR_KEY.split(line)
        row = {"Step": int(parts[0])}
        for key, value in zip(parts[1::2], parts[2::2]):
            name = _OSZICAR_NAMES.get(key.decode())
            f = floats(value)
            if name and f:
                row[name] = f[0] if len(f) == 1 else float(np.linalg.norm(f[:3]))
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no ionic-step lines found; not an OSZICAR?")
    return _table(rows)
