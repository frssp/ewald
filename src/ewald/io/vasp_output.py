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
from .base import (FrameReader, complete_size, file_size, floats, grep_lines, head_lines, open_binary,
                   strip_compression)
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
_OUTCAR_LITERALS = (b"Iteration", b"direct lattice vectors", b"TOTAL-FORCE", b"TOTEN", b"without entropy", b"in kB",
                    b"external pressure", b"magnetization", b"EKIN", b"ETOTAL", b"POTCAR:", b"ions per type",
                    b"NIONS =")


class OutcarReader(FrameReader):
    """VASP OUTCAR: every ionic step with forces, energies, stress and magnetic moments, lazy."""

    def __init__(self, path):
        self.path = str(path)
        self._potcars, self._counts, self._natoms = [], None, None
        self._all_steps: dict[int, dict] = {}
        self._ionic, self._lattice = 0, None
        size = file_size(self.path)
        # a running VASP may have written half a line; it is looked at again by refresh()
        self._scanned = complete_size(self.path, size) if size else None
        hits = grep_lines(self.path, _OUTCAR_KEYS, _OUTCAR_LITERALS)
        self._consume(hits if self._scanned is None else [h for h in hits if h[0] < self._scanned])
        counts = self._counts
        if counts is None:
            raise ValueError(f"{path}: no 'ions per type' line; not an OUTCAR?")
        # VASP prints the POTCAR lines more than once (once per type, then again); the first set counts
        potcars = self._potcars
        species = [species_name(p[1].decode() if len(p) > 1 else p[0].decode()) for p in potcars[: len(counts)]]
        if len(species) != len(counts):
            raise ValueError(f"{path}: {len(potcars)} POTCAR lines for {len(counts)} species")
        self.types = np.repeat(species, counts)
        self.natoms = self._natoms or int(sum(counts))
        if not self._collect():
            warnings.warn(f"{path}: last ionic step is incomplete and was skipped", stacklevel=2)
        if not self._steps:
            raise ValueError(f"{path}: no ionic steps with positions found")

    def _consume(self, hits):
        steps = self._all_steps
        for off, line in hits:
            if b"Iteration" in line:
                self._ionic = int(re.search(rb"Iteration +(\d+)\(", line).group(1))
                continue
            rec = steps.setdefault(self._ionic, {"values": {}, "mag": {}})
            v = rec["values"]
            if b"direct lattice vectors" in line:
                self._lattice = off
            elif b"TOTAL-FORCE" in line:
                rec["pos"], rec["lat"] = off, self._lattice
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
            elif b"POTCAR:" in line and self._counts is None:
                self._potcars.append(line.split(b":", 1)[1].split())  # 'PAW_PBE Fe_pv 02Aug2007'
            elif b"ions per type" in line and self._counts is None:
                self._counts = [int(x) for x in line.split(b"=")[1].split()]
            elif b"NIONS =" in line:
                self._natoms = int(line.split(b"NIONS =")[1].split()[0])

    def _collect(self) -> bool:
        """Ionic steps with positions -> ``self._steps`` and ``self.thermo``; False when the
        last one could not be read (VASP is still writing it) and was left out."""
        steps = [(n, rec) for n, rec in sorted(self._all_steps.items()) if "pos" in rec]
        for _, rec in steps:
            if "kb" in rec:
                rec["values"]["Stress"] = _voigt_from_kb(*rec["kb"])
                rec["values"].setdefault("Pressure", sum(rec["kb"][:3]) / 3)
        self._steps = steps
        complete = True
        if steps:
            try:
                self.read(len(steps) - 1)
            except (ValueError, IndexError):
                self._steps = steps = steps[:-1]
                complete = False
        self.thermo = _table([{"Step": n, **{k: x for k, x in rec["values"].items() if np.isscalar(x)}}
                              for n, rec in steps])
        return complete

    def refresh(self) -> int | None:
        """Index ionic steps VASP appended since the last look: only the new part of the
        file is searched. The old last step may gain energies or stress, so it counts as changed."""
        size = file_size(self.path)
        if size is None or self._scanned is None or size == self._scanned:
            return None
        if size < self._scanned:  # rewritten (a new run): index from scratch
            try:
                fresh = OutcarReader(self.path)
            except ValueError:
                return None
            self.__dict__.update(fresh.__dict__)
            return 0
        end = complete_size(self.path, size)
        if end <= self._scanned:
            return None
        hits = grep_lines(self.path, _OUTCAR_KEYS, _OUTCAR_LITERALS, start=self._scanned)
        before = self._signature()
        self._consume([h for h in hits if h[0] < end])
        self._scanned = end
        self._collect()
        if self._signature() == before:  # only electronic steps were written
            return None
        return max(before[0] - 1, 0)

    def _signature(self):
        n, rec = self._steps[-1] if self._steps else (None, {"values": {}, "mag": {}})
        return len(self._steps), n, tuple(rec["values"]), tuple(rec["mag"]), "kb" in rec

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
    "^ *<(kpoints|dos|eigenvalues|projected)>", "</projected>",
    "^ *</?(electronvelocities|eigenvalues_kpoints_opt|projected_kpoints_opt)[ >]",
))
_VASPRUN_LITERALS = (b"calculation>", b"<structure", b"<varray name=\"f", b"<varray name=\"s", b"scstep>",
                     b"<atominfo>", b"<i name=\"e_", b"<i name=\"kinetic\"", b"<i name=\"total\"",
                     b"<kpoints>", b"<dos>", b"<eigenvalues>", b"projected>", b"electronvelocities>",
                     b"_kpoints_opt")
# blocks with their own <kpoints> / <eigenvalues> (LVEL velocities, KPOINTS_OPT bands), not the run's
_VASPRUN_NESTED = (b"<electronvelocities", b"<eigenvalues_kpoints_opt", b"<projected_kpoints_opt")
_VASPRUN_VALUES = {b"e_fr_energy": "Free Energy", b"e_0_energy": "Energy", b"kinetic": "Kinetic Energy",
                   b"total": "Total Energy"}


class VasprunReader(FrameReader):
    """VASP vasprun.xml: every ionic step with cell, forces, stress and energies, lazy."""

    def __init__(self, path):
        self.path = str(path)
        self._steps: list[dict] = []
        self._electronic_offsets: dict[str, int] = {}
        self.electronic = None
        atominfo, calc, in_scstep, in_projected, nested = None, None, False, False, False
        fallback, last_values = {}, {}
        for off, line in grep_lines(self.path, _VASPRUN_KEYS, _VASPRUN_LITERALS):
            tag = line.strip()
            if tag.replace(b"</", b"<", 1).startswith(_VASPRUN_NESTED):
                nested = not tag.startswith(b"</")
                continue
            if nested:
                continue
            if tag in (b"<kpoints>", b"<dos>", b"<eigenvalues>", b"<projected>"):
                if tag == b"<projected>":
                    in_projected = True
                if not (in_projected and tag == b"<eigenvalues>"):  # the copy inside <projected>
                    self._electronic_offsets[tag[1:-1].decode()] = off
                continue
            if tag == b"</projected>":
                in_projected = False
                continue
            if b"<calculation>" in line:
                calc = {"values": {}}
            elif b"</calculation>" in line:
                if calc is not None:
                    # VASP 5 writes a wrong e_0_energy (0 or the -sigma/2 term) after the last
                    # <scstep>, whose own values lack the vdW correction: E0 = F + (E0 - F)
                    # of the last <scstep>, as ASE reads it
                    scf, v = calc.pop("scf", {}), calc["values"]
                    if "Free Energy" in v and "e_0_energy" in scf and "e_fr_energy" in scf:
                        v["Energy"] = v["Free Energy"] + scf["e_0_energy"] - scf["e_fr_energy"]
                    last_values = calc["values"]
                    if "structure" in calc:
                        self._steps.append(calc)
                calc = None
            elif b"<atominfo>" in line:
                atominfo = off
            elif calc is None:
                name = re.search(rb'<structure name="(\w+)"', line)
                if name:  # initialpos / finalpos
                    fallback[name.group(1).decode()] = off
                continue
            elif b"<scstep>" in line:
                in_scstep = True
            elif b"</scstep>" in line:
                in_scstep = False
            elif in_scstep:
                name = re.search(rb'name="(e_0_energy|e_fr_energy)"', line)
                if name:
                    scf, key = calc.setdefault("scf", {}), name.group(1).decode()
                    try:
                        scf[key] = float(line.split(b">", 1)[1].split(b"<")[0])
                    except ValueError:  # ***** overflow
                        scf.pop(key, None)
            elif b"<structure" in line:
                calc["structure"] = off
            elif b"\"forces\"" in line:
                calc["forces"] = off
            elif b"\"stress\"" in line:
                calc["stress"] = off
            else:
                name = re.search(rb'name="(\w+)"', line).group(1)
                try:
                    calc["values"][_VASPRUN_VALUES[name]] = float(line.split(b">", 1)[1].split(b"<")[0])
                except ValueError:  # ***** overflow
                    pass
        if calc is not None and "structure" in calc:  # running job: the open last step
            self._steps.append(calc)
        if not self._steps and ("finalpos" in fallback or "initialpos" in fallback):
            # GW, BSE and optics runs have no ionic step: one frame of the structure (with DOS / bands)
            self._steps.append({"structure": fallback.get("finalpos", fallback.get("initialpos")),
                                "values": dict(last_values)})
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
        if "dos" in self._electronic_offsets or "eigenvalues" in self._electronic_offsets:
            try:
                self.electronic = self._read_electronic(self.read(len(self._steps) - 1).cell.matrix)
            except (ValueError, ET.ParseError, IndexError, KeyError) as exc:
                warnings.warn(f"{path}: DOS / eigenvalues could not be read ({exc})", stacklevel=2)

    def __len__(self):
        return len(self._steps)

    # ------------------------------------------------------------ DOS and bands
    def _scan(self, offset, closing: bytes, sections=()):
        """Rows (``<r>``), field names and ``<set comment="word n">`` counts of the element at
        ``offset``, per section; a line starting with one of ``sections`` (tag prefixes) starts
        the section of that name. Line by line, without building a tree: a PDOS or projection
        block can hold millions of numbers."""
        out = {"": {"rows": [], "fields": [], "sets": {}, "values": {}}}
        cur = out[""]
        with open_binary(self.path) as fh:
            fh.seek(offset)
            for line in fh:
                s = line.strip()
                if s.startswith(b"<r>"):
                    cur["rows"].append(s[3:-4])
                    continue
                for prefix in sections:
                    if s.startswith(prefix):
                        cur = out.setdefault(prefix.decode(), {"rows": [], "fields": [], "sets": {}, "values": {}})
                if s.startswith(b"<field"):
                    cur["fields"].append(s.split(b">", 1)[1].split(b"<", 1)[0].strip().decode())
                elif s.startswith(b"<set comment="):
                    word = re.match(rb'<set comment="([a-z]+)', s)
                    if word:
                        w = word.group(1).decode()
                        cur["sets"][w] = cur["sets"].get(w, 0) + 1
                elif s.startswith(b"<i ") and b"name=" in s:
                    name = re.search(rb'name="(\w+)"', s).group(1).decode()
                    try:
                        cur["values"][name] = float(s.split(b">", 1)[1].split(b"<", 1)[0])
                    except ValueError:
                        pass
                if closing in line:
                    break
            else:
                raise ValueError(f"{self.path}: element at byte {offset} is not closed")
        for sec in out.values():
            rows = sec["rows"]
            sec["array"] = np.array(b" ".join(rows).split(), dtype=float).reshape(len(rows), -1) \
                if rows else np.zeros((0, 0))
        return out

    def _read_electronic(self, lattice):
        from ..electronic import ElectronicStructure, kpoints_file_labels, line_ticks, path_distances

        offs = self._electronic_offsets
        es = ElectronicStructure(types=self.types.copy(), source=os.path.basename(self.path))
        natoms, band = len(self.types), None
        if "dos" in offs:
            dos = self._scan(offs["dos"], b"</dos>", (b"<total", b"<partial"))
            es.efermi = dos[""]["values"].get("efermi", 0.0)
            tot = dos.get("<total", {}).get("array")
            if tot is not None and tot.size:
                nspin = dos["<total"]["sets"].get("spin", 1)
                ne = tot.shape[0] // nspin
                tot = tot.reshape(nspin, ne, -1)[: 1 if nspin == 4 else nspin]
                es.dos_energies, es.dos = tot[0, :, 0].copy(), tot[:, :, 1].copy()
                part = dos.get("<partial")
                if part is not None and part["array"].size:
                    nsets = part["sets"].get("spin", 1)
                    ns = nsets // natoms
                    a = part["array"].reshape(natoms, ns, ne, -1)[:, : 1 if ns == 4 else ns]
                    es.orbitals = part["fields"][1:]
                    es.pdos = np.ascontiguousarray(np.transpose(a[..., 1:], (0, 1, 3, 2)))
        if "eigenvalues" in offs:
            eig = self._scan(offs["eigenvalues"], b"</eigenvalues>")[""]
            a = eig["array"]
            nspin = eig["sets"].get("spin", 1)
            nk = eig["sets"].get("kpoint", 1) // nspin
            a = a.reshape(nspin, nk, -1, a.shape[1])
            es.eigenvalues, es.occupations = a[..., 0].copy(), a[..., 1].copy()
        if "kpoints" in offs and es.eigenvalues is not None:
            kp = self._element(offs["kpoints"], b"</kpoints>")
            k = np.array([v.text.split() for v in kp.findall("varray[@name='kpointlist']/v")], dtype=float)
            w = np.array([v.text.split()[0] for v in kp.findall("varray[@name='weights']/v")], dtype=float)
            if len(k) == es.eigenvalues.shape[1]:
                gen = kp.find("generation")
                div = gen.find("i[@name='divisions']") if gen is not None and \
                    gen.get("param", "").lower().startswith("list") else None
                segment = int(div.text.split()[0]) if div is not None else None
                band = np.flatnonzero(w == 0) if (w == 0).any() and (w > 0).any() else None
                if band is not None:  # hybrid functional band run: the zero-weight points are the path
                    es.eigenvalues, es.occupations = es.eigenvalues[:, band], es.occupations[:, band]
                    k, w, segment = k[band], w[band], None
                es.kpoints, es.weights = k, w
                es.distances, es.breaks = path_distances(k, lattice, segment)
                if segment:
                    labels = kpoints_file_labels(os.path.join(os.path.dirname(os.path.abspath(self.path)), "KPOINTS"))
                    labels = labels[1] if labels and labels[0] == segment else None
                    es.ticks = line_ticks(es.distances, segment, labels, k)
        if "projected" in offs and es.eigenvalues is not None:
            off, shape = offs["projected"], es.eigenvalues.shape

            def load():
                p = self._scan(off, b"</projected>", (b"</eigenvalues>",)).get("</eigenvalues>")
                if p is None or not p["array"].size:
                    return None
                ns = p["sets"].get("spin", 1)
                a = p["array"].reshape(ns, -1, shape[2], natoms, p["array"].shape[1])[: 1 if ns == 4 else ns]
                if band_idx is not None:
                    a = a[:, band_idx]
                norb, have = len(es.orbitals), a.shape[-1]
                if have < norb and p["fields"] == es.orbitals[:have]:  # VASP 5.2: fewer columns than the PDOS
                    a = np.concatenate([a, np.zeros(a.shape[:-1] + (norb - have,))], axis=-1)
                return a if a.shape[:3] == shape and a.shape[-1] == norb else None

            band_idx = band
            es.orbitals = es.orbitals or self._scan_fields(off)
            es.projections_loader = load
        return es

    def _scan_fields(self, offset) -> list[str]:
        """Field names of the projection array at ``offset`` (just the header)."""
        fields = []
        with open_binary(self.path) as fh:
            fh.seek(offset)
            seen_eig = False
            for line in fh:
                s = line.strip()
                if s.startswith(b"</eigenvalues>"):
                    seen_eig = True
                elif seen_eig and s.startswith(b"<field"):
                    fields.append(s.split(b">", 1)[1].split(b"<", 1)[0].strip().decode())
                elif seen_eig and s.startswith(b"<set"):
                    break
        return fields

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
        el = self._element(offset, b"</varray>")  # ***** (Fortran overflow) reads as NaN
        return np.array([[np.nan if t[0] == "*" else t for t in v.text.split()] for v in el.findall("v")],
                        dtype=float)

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
        if self.electronic is not None:
            self.electronic.attach(data)
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
