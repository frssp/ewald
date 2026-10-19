"""Per-step results next to a trajectory: LAMMPS ``log.lammps`` thermo output, VASP OSZICAR.

A :class:`Thermo` is a table with a ``Step`` column. A pipeline source holding one
copies the row whose ``Step`` equals a frame's ``Timestep`` into that frame's
attributes (``Temp``, ``PotEng``, ``Press``, ... for LAMMPS, as the log names them;
``Energy``, ``Free Energy``, ``Temperature``, ... for VASP) and puts the whole table
into ``data.tables["Thermo"]``, which the GUI's Plots tab draws with a marker at the
current frame. Frames without a matching step get the table but no attributes.
"""
from __future__ import annotations

import os
import posixpath
import re

import numpy as np

from . import remote
from .base import open_binary


def read_lammps_log(path) -> dict:
    """Thermo output of every run in a LAMMPS log (one-line ``thermo_style`` or ``multi``) as one table.

    Runs with different ``thermo_style`` columns are joined by ``Step``: columns a run does
    not print are NaN, and a step printed twice (the last step of a run is the first of
    the next) keeps the values of both, the later run winning where both print a column.
    WARNING lines inside a run are skipped; a truncated last line (running job) is dropped.
    """
    rows: dict[int, dict] = {}
    names: dict[str, None] = {}
    cols, multi = None, None
    with open_binary(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0].startswith(b"-----") and len(tok) > 2 and tok[1] == b"Step" and _is_number(tok[2]):
                # thermo_style multi: '---- Step 100 ----- CPU = 0.1 (sec) ----' then 'Name = value' lines
                cols, multi = None, rows.setdefault(int(float(tok[2])), {})
                names["Step"] = None
                if b"CPU" in tok[3:-2] and _is_number(tok[tok.index(b"CPU") + 2]):
                    multi["CPU"] = float(tok[tok.index(b"CPU") + 2])
                    names["CPU"] = None
                continue
            if multi is not None:
                pairs = _MULTI_PAIR.findall(line)
                if pairs and all(_is_number(v) for _, v in pairs):
                    for k, v in pairs:
                        multi[k.decode()] = float(v)
                        names[k.decode()] = None
                    continue
                multi = None
            if cols is not None:
                if len(tok) == len(cols):
                    try:
                        values = [float(t) for t in tok]
                    except ValueError:
                        values = None
                    if values is not None:
                        step = int(values[cols.index("Step")])
                        row = rows.setdefault(step, {})
                        row.update(zip(cols, values))
                        continue
                if tok[0] in (b"Loop", b"ERROR:", b"ERROR"):
                    cols = None
                continue
            if b"Step" in tok and not _is_number(tok[0]):
                cols = [t.decode() for t in tok]
                names.update(dict.fromkeys(cols))
    if not rows:
        raise ValueError(f"{path}: no thermo output found; not a LAMMPS log?")
    order = sorted(rows)
    table = {"Step": np.array(order, dtype=np.int64)}
    for name in names:
        if name != "Step":
            table[name] = np.array([rows[s].get(name, np.nan) for s in order], dtype=float)
    return table


_MULTI_PAIR = re.compile(rb"(\S+)\s+=\s+(\S+)")


def _is_number(tok: bytes) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def read_thermo(path) -> dict:
    """Table from a LAMMPS log or a VASP OSZICAR (chosen by name, then by trying both)."""
    from .vasp_output import is_oszicar, read_oszicar

    if is_oszicar(path):
        return read_oszicar(path)
    try:
        return read_lammps_log(path)
    except ValueError:
        return read_oszicar(path)


def merge_tables(tables) -> dict:
    """Join tables on ``Step``; later tables win where both have a value."""
    tables = [t for t in tables if t]
    if len(tables) == 1:
        return tables[0]
    steps = np.unique(np.concatenate([np.asarray(t["Step"]) for t in tables]))
    out = {"Step": steps}
    for t in tables:
        idx = np.searchsorted(steps, t["Step"])
        for name, col in t.items():
            if name == "Step":
                continue
            merged = out.get(name)
            if merged is None:
                merged = np.full(len(steps), np.nan)
            else:
                merged = merged.copy()
            vals = np.asarray(col, dtype=float)
            ok = ~np.isnan(vals)
            merged[idx[ok]] = vals[ok]
            out[name] = merged
    return out


class Thermo:
    """A per-step table and the files it came from; see the module docstring."""

    def __init__(self, table: dict, sources=()):
        self.table = table
        self.sources = list(sources)
        self._row = {int(s): i for i, s in enumerate(table["Step"])}

    @property
    def columns(self) -> list[str]:
        return [k for k in self.table if k != "Step"]

    def __len__(self):
        return len(self.table["Step"])

    def row(self, step) -> dict:
        """Values at ``step`` (NaNs left out); empty when the step is not in the table."""
        try:
            i = self._row.get(int(step))
        except (TypeError, ValueError):
            i = None
        if i is None:
            return {}
        out = {}
        for k, col in self.table.items():
            if k != "Step":
                v = float(col[i])
                if v == v:
                    out[k] = v
        return out

    def annotate(self, data) -> None:
        for k, v in self.row(data.attributes.get("Timestep")).items():
            data.attributes.setdefault(k, v)
        data.tables["Thermo"] = self.table

    def __repr__(self):
        return f"<Thermo {len(self)} steps, columns {self.columns}>"


_SIBLINGS = {"xdatcar": ("OSZICAR",), "lammps-dump": ("log.lammps",)}


def sibling_files(path, kind: str) -> list[str]:
    """Result files next to ``path`` that belong to the same run (OSZICAR next to an
    XDATCAR, log.lammps next to a dump), local or remote."""
    path = str(path)
    if remote.is_remote(path):
        rp = remote.parse_remote(path)
        cands = [str(rp.with_path(posixpath.join(posixpath.dirname(rp.path), n))) for n in _SIBLINGS.get(kind, ())]
        return [c for c in cands if remote.exists(c)]
    folder = os.path.dirname(os.path.abspath(path))
    return [os.path.join(folder, n) for n in _SIBLINGS.get(kind, ()) if os.path.isfile(os.path.join(folder, n))]
