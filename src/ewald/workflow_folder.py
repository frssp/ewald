"""Folders of calculations read as one workflow, with no database behind them.

A project folder usually holds the steps of one study side by side: ``01_relax/``, ``02_neb/``
with its images ``00 .. NN``, ``phonon/`` with phonopy's files, an MD that was restarted in
``run1/ run2/``. :func:`describe_folder` walks such a folder (locally or on an SSH host,
``host:/path``) and returns a :class:`~ewald.workflow.Workflow`, so it shows, opens as one notebook
and reaches the AI the same way an AiiDA workflow does::

    from ewald.workflow_folder import describe_folder
    from ewald.workflow import notebook_text, text
    wf = describe_folder("~/calc/li_diffusion")
    print(text(wf))      # relax_IS ✓ -> .../relax_IS/vasprun.xml, neb -> .../neb, ...

What a folder is, from the files in it (nothing is run, files are only listed and their heads
or tails read):

- NEB: VASP image folders ``00 .. NN`` or a LAMMPS NEB run (:func:`ewald.io.is_neb`); the step
  opens the whole band;
- phonopy: ``phonopy_disp.yaml``, ``phonopy.yaml``, ``phonopy_params.yaml``, ``FORCE_SETS`` or
  ``band.yaml``; its displacement folders (``disp-001``, ...) are counted, not listed;
- VASP: ``vasprun.xml``, ``OUTCAR``, ``XDATCAR``, ``OSZICAR`` or ``INCAR`` with ``POSCAR``; the kind
  (relax, md, static, phonon, neb) comes from the INCAR (:func:`ewald.runinfo.run_info`), the
  result opened is the XDATCAR for MD, else vasprun.xml, OUTCAR, XDATCAR, CONTCAR, POSCAR;
- LAMMPS: dump files (``ITEM: TIMESTEP``) with or without a ``log.lammps``;
- restarts: a folder whose sub-folders are all calculations named ``run1, run2``, ``restart_1``,
  ``1, 2, 3``, ``r0, r1`` ... is one step with ``runs`` set and the segments as its sub-steps;
- any other folder with calculations below it is a group step; hidden folders are skipped.

Steps are in natural order of their folder names (``2_neb`` before ``10_dos``). The workflow's
own ``path`` is the last step's result, so ``pipe`` starts at the end of the study.

:class:`FolderSource` is the built-in workflow source ``folders`` over this: it lists the folders
opened as workflows lately (newest first) and describes any folder path given to it.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import time

from .io import remote
from .io.neb import _entries, _join, _natural_key, lammps_neb_images, vasp_neb_images
from .workflow import Workflow, WorkflowOutput, WorkflowPage, WorkflowStep

__all__ = ["describe_folder", "is_workflow_folder", "FolderSource", "recent_folders", "add_recent_folder"]

#: how deep below the workflow folder calculations are looked for
MAX_DEPTH = 3
#: folders with more entries than this are not walked into (a scratch folder, a dataset)
MAX_ENTRIES = 2000
#: folders remembered as recent workflows
MAX_RECENT = 30

_PHONOPY = ("band.yaml", "phonopy_params.yaml", "phonopy.yaml", "phonopy_disp.yaml", "FORCE_SETS")
_VASP_MARKS = ("vasprun.xml", "OUTCAR", "XDATCAR", "OSZICAR")
_VOLUMETRIC = ("CHGCAR", "LOCPOT", "ELFCAR", "PARCHG", "AECCAR0", "AECCAR2")
_SEGMENT = re.compile(r"^(?:run|restart|segment|seg|part|r|job|cont|continue)?[_.-]?(\d+)$", re.IGNORECASE)
_SEGMENT_PREFIX = re.compile(r"^(\D*)\d+$")
_DISPLACEMENT = re.compile(r"^(?:disp|displacement|POSCAR)?[_.-]?\d{3,}$", re.IGNORECASE)
_SKIP = re.compile(r"^(\.|__pycache__$|node_modules$)")


# ----------------------------------------------------------------------- files
def _files(entries: dict) -> dict:
    """``{name: size}`` of the non-empty files of a listing (compressed ones under their plain name too)."""
    out = {}
    for name, (is_dir, size) in entries.items():
        if not is_dir and size > 0:
            out[name] = size
            for ext in (".gz", ".xz", ".bz2"):
                if name.endswith(ext):
                    out.setdefault(name[: -len(ext)], size)
    return out


def _real_name(entries: dict, name: str) -> str:
    """The file in the listing that holds ``name`` (``OUTCAR`` or ``OUTCAR.gz``)."""
    for cand in (name, name + ".gz", name + ".xz", name + ".bz2"):
        if cand in entries and not entries[cand][0]:
            return cand
    return name


def _subdirs(entries: dict) -> list[str]:
    return sorted((n for n, (is_dir, _) in entries.items() if is_dir and not _SKIP.match(n)), key=_natural_key)


def _basename(path: str) -> str:
    if remote.is_remote(path):
        return posixpath.basename(remote.parse_remote(path).path.rstrip("/")) or path
    return os.path.basename(os.path.normpath(path))


def _tail(path: str, size: int = 1 << 16) -> bytes:
    from .io.base import open_binary

    try:
        with open_binary(path) as fh:
            try:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - size))
            except (OSError, ValueError):
                pass
            return fh.read()
    except OSError:
        return b""


def _last_energy(oszicar: str) -> float | None:
    """E0 (else F) of the last ionic step in an OSZICAR."""
    for line in reversed(_tail(oszicar).decode("utf-8", "replace").splitlines()):
        m = re.search(r"\bE0=\s*([-+.0-9Ee]+)", line) or re.search(r"\bF=\s*([-+.0-9Ee]+)", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _ionic_steps(oszicar: str) -> int | None:
    lines = [ln for ln in _tail(oszicar).decode("utf-8", "replace").splitlines() if " F=" in ln]
    if not lines:
        return None
    try:
        return int(lines[-1].split()[0])
    except ValueError:
        return None


def _head(path: str, size: int = 1 << 14) -> bytes:
    from .io.base import open_binary

    try:
        with open_binary(path) as fh:
            return fh.read(size)
    except OSError:
        return b""


def _atom_count(folder: str, entries: dict) -> int | None:
    """Atoms in the cell: the counts line of CONTCAR or POSCAR, else ``NIONS`` in the OUTCAR head."""
    files = _files(entries)
    for name in ("CONTCAR", "POSCAR"):
        if name not in files:
            continue
        lines = _head(_join(folder, _real_name(entries, name)), 1 << 12).decode("utf-8", "replace").splitlines()
        for line in lines[5:7]:          # VASP 5 has a species line first, VASP 4 does not
            words = line.split()
            if words and all(w.isdigit() for w in words):
                return sum(int(w) for w in words)
    if "OUTCAR" in files:
        m = re.search(rb"NIONS\s*=\s*(\d+)", _head(_join(folder, _real_name(entries, "OUTCAR")), 1 << 16))
        if m:
            return int(m.group(1))
    return None


def _max_force(outcar: str, atoms: int | None) -> float | None:
    """Largest force (eV/Å) on an atom in the last ``TOTAL-FORCE`` block of an OUTCAR (its tail
    only: None when the block is not in the last part of the file or is cut off)."""
    size = max(1 << 16, 120 * (atoms or 0) + (1 << 15))
    text = _tail(outcar, min(size, 1 << 23)).decode("utf-8", "replace")
    at = text.rfind("TOTAL-FORCE")
    if at < 0:
        return None
    best, rows = None, 0
    lines = text[at:].splitlines()[2:]
    for line in lines:
        words = line.split()
        if len(words) != 6:
            break
        try:
            f = [float(w) for w in words[3:]]
        except ValueError:
            break
        norm = (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5
        best = norm if best is None else max(best, norm)
        rows += 1
    if not rows or (atoms and rows != atoms):
        return None
    return round(best, 6)


#: EIGENVAL files larger than this are not read for the band gap (bytes)
MAX_EIGENVAL = 32 << 20


def _band_gap(eigenval: str) -> float | None:
    """Band gap (eV, 0 for a metal) from the band energies and occupations of an EIGENVAL file
    (VASP 5.4 or later writes the occupations); None when it has none."""
    import numpy as np

    from .io.base import open_binary

    try:
        with open_binary(eigenval) as fh:
            lines = fh.read(MAX_EIGENVAL + 1).decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    try:
        spin = int(lines[0].split()[3])
        _, nk, nb = (int(w) for w in lines[5].split()[:3])
    except (IndexError, ValueError):
        return None
    energies, occupations, i = [], [], 6
    for _ in range(nk):
        while i < len(lines) and not lines[i].strip():
            i += 1
        i += 1                                       # the k-point line
        for _ in range(nb):
            if i >= len(lines):
                return None
            words = lines[i].split()
            i += 1
            if len(words) < (5 if spin == 2 else 3):
                return None
            try:
                values = [float(w) for w in words[1:]]
            except ValueError:
                return None
            if spin == 2:
                energies += values[0:2]
                occupations += values[2:4]
            else:
                energies.append(values[0])
                occupations.append(values[1])
    e, occ = np.asarray(energies), np.asarray(occupations)
    full, empty = e[occ > 0.5], e[occ <= 0.5]
    if not len(full) or not len(empty):
        return None
    return round(max(0.0, float(empty.min() - full.max())), 6)


# ----------------------------------------------------------------------- one folder
def _neb_step(name: str, folder: str, entries: dict) -> WorkflowStep | None:
    images, _log = vasp_neb_images(folder), None
    if images:
        code = "vasp"
    else:
        images, _log = lammps_neb_images(folder)
        code = "lammps"
    if not images:
        return None
    step = WorkflowStep(name, path=folder, work_dir=folder,
                        outputs=[WorkflowOutput("band", folder, kind="band", note=f"{len(images)} images")])
    step.scalars["code"] = code
    step.scalars["kind"] = "neb"
    step.scalars["images"] = len(images)
    energies = None
    if code == "lammps" and _log:
        from .io.neb import read_lammps_neb_log

        try:
            energies = list(read_lammps_neb_log(_log)["Energy"])
        except (OSError, ValueError, KeyError):
            energies = None
        if b"Total wall time" in _tail(_log, 1 << 13):
            step.state = "finished"
    elif code == "vasp":
        listings = [_entries(d) for d in images]
        finished = [_tail(_join(d, _real_name(e, "OUTCAR")), 1 << 14)
                    for d, e in zip(images[1:-1], listings[1:-1])]
        if finished and all(b"General timing" in t for t in finished):
            step.state = "finished"
        energies = [_last_energy(_join(d, _real_name(e, "OSZICAR"))) if "OSZICAR" in _files(e) else None
                    for d, e in zip(images, listings)]
    if energies and energies[0] is not None and all(e is not None for e in energies[1:-1]):
        known = [e for e in energies if e is not None]
        step.scalars["barrier"] = round(float(max(known) - energies[0]), 6)
    return step


def _phonopy_step(name: str, folder: str, entries: dict) -> WorkflowStep | None:
    files = _files(entries)
    found = [f for f in _PHONOPY if f in files]
    if not found:
        return None
    main = next((f for f in _PHONOPY[:4] if f in files), None)
    path = _join(folder, _real_name(entries, main)) if main else ""
    step = WorkflowStep(name, path=path, work_dir=folder)
    step.scalars["code"] = "phonopy"
    step.scalars["kind"] = "phonon"
    displacements = [d for d in _subdirs(entries) if _DISPLACEMENT.match(d)]
    if displacements:
        step.scalars["displacements"] = len(displacements)
    for f in found:
        if _join(folder, _real_name(entries, f)) != path:
            step.outputs.append(WorkflowOutput(f, _join(folder, _real_name(entries, f)),
                                               kind="data" if f != "band.yaml" else "file"))
    from . import plugins

    if "phonopy" not in plugins.readers():
        step.outputs.insert(0, WorkflowOutput("needs the phonopy plugin", "", kind="data",
                                              note="Plugins side bar: install phonopy"))
    if "band.yaml" in files or "FORCE_SETS" in files or "FORCE_CONSTANTS" in files:
        step.state = "finished"
    return step


def _vasp_step(name: str, folder: str, entries: dict) -> WorkflowStep | None:
    files = _files(entries)
    if not (any(m in files for m in _VASP_MARKS) or ("INCAR" in files and "POSCAR" in files)):
        return None
    from .runinfo import run_info

    def path_of(f):
        return _join(folder, _real_name(entries, f))

    probe = next((f for f in ("INCAR", "OUTCAR", "vasprun.xml", "XDATCAR", "POSCAR") if f in files), None)
    kind = None
    if probe:
        try:
            kind = run_info(path_of(probe)).kind
        except (OSError, ValueError, UnicodeError):
            kind = None
    order = (("XDATCAR", "vasprun.xml", "OUTCAR") if kind == "md"
             else ("vasprun.xml", "OUTCAR", "XDATCAR", "CONTCAR", "POSCAR"))
    main = next((f for f in order if f in files), None)
    step = WorkflowStep(name, path=path_of(main) if main else "", work_dir=folder)
    step.scalars["code"] = "vasp"
    if kind:
        step.scalars["kind"] = kind
    if "OUTCAR" in files:
        step.state = "finished" if b"General timing" in _tail(path_of("OUTCAR"), 1 << 14) else "running"
    if "OSZICAR" in files:
        energy = _last_energy(path_of("OSZICAR"))
        if energy is not None:
            step.scalars["energy"] = energy
        n = _ionic_steps(path_of("OSZICAR"))
        if n:
            step.scalars["ionic steps"] = n
    atoms = _atom_count(folder, entries)
    if atoms:
        step.scalars["atoms"] = atoms
        if "energy" in step.scalars:
            step.scalars["energy/atom"] = round(step.scalars["energy"] / atoms, 6)
    if "OUTCAR" in files:
        force = _max_force(path_of("OUTCAR"), atoms)
        if force is not None:
            step.scalars["max force"] = force
    if "EIGENVAL" in files and files["EIGENVAL"] <= MAX_EIGENVAL:
        gap = _band_gap(path_of("EIGENVAL"))
        if gap is not None:
            step.scalars["gap"] = gap
    if "CONTCAR" in files and main != "CONTCAR":
        step.outputs.append(WorkflowOutput("CONTCAR", path_of("CONTCAR"), kind="structure", note="final structure"))
    for f in _VOLUMETRIC:
        if f in files:
            step.outputs.append(WorkflowOutput(f, path_of(f), kind="volumetric"))
    return step


def _lammps_step(name: str, folder: str, entries: dict) -> WorkflowStep | None:
    from .io.lammps_dump import is_lammps_dump

    files = _files(entries)
    cands = [f for f in sorted(files, key=_natural_key)
             if (f.startswith("dump") or f.endswith((".lammpstrj", ".dump", ".lammpstrj.gz", ".dump.gz")))
             and f == _real_name(entries, f)]
    dumps = []
    for f in cands[:50]:
        try:
            if is_lammps_dump(_join(folder, f)):
                dumps.append(f)
        except OSError:
            pass
    if not dumps:
        return None
    step = WorkflowStep(name, path=_join(folder, dumps[0]), work_dir=folder)
    step.scalars["code"] = "lammps"
    if len(dumps) > 1:
        step.outputs += [WorkflowOutput(f, _join(folder, f), kind="trajectory") for f in dumps[1:]]
    if "log.lammps" in files:
        from .runinfo import run_info

        try:
            kind = run_info(_join(folder, dumps[0])).kind
        except (OSError, ValueError, UnicodeError):
            kind = None
        if kind:
            step.scalars["kind"] = kind
        tail = _tail(_join(folder, "log.lammps"), 1 << 13)
        step.state = "finished" if b"Total wall time" in tail else "running"
    return step


def _calculation(name: str, folder: str, entries: dict) -> WorkflowStep | None:
    """The step a folder is when it holds a calculation itself, else None."""
    for kind in (_neb_step, _phonopy_step, _vasp_step, _lammps_step):
        step = kind(name, folder, entries)
        if step is not None:
            return step
    return None


def _walk(name: str, folder: str, depth: int, entries: dict | None = None) -> WorkflowStep | None:
    entries = _entries(folder) if entries is None else entries
    if not entries or len(entries) > MAX_ENTRIES:
        return None
    step = _calculation(name, folder, entries)
    if step is not None:
        return step
    if depth >= MAX_DEPTH:
        return None
    children = []
    for sub in _subdirs(entries):
        child = _walk(sub, _join(folder, sub), depth + 1)
        if child is not None:
            children.append(child)
    if not children:
        return None
    step = WorkflowStep(name, work_dir=folder, steps=children)
    if len(children) > 1 and all(_SEGMENT.match(c.name) for c in children) and all(not c.steps for c in children):
        # restarts of one calculation: run1, run2, ... in time order
        step.runs = len(children)
        last = children[-1]
        step.path = _segments_path(folder, children)
        step.state = last.state
        step.scalars = dict(last.scalars)
    elif children:
        states = {c.state for c in children}
        step.state = "finished" if states == {"finished"} else ""
    return step


def _segments_path(folder: str, segments) -> str:
    """One path that opens every segment of a restarted run as one trajectory: a wildcard such
    as ``md/run*/XDATCAR`` when the segments keep their result under the same name and the
    wildcard matches exactly them (``load`` joins the files in natural order), else the last
    segment's result."""
    last = segments[-1].path
    names = {_basename(s.path) for s in segments}
    prefixes = {_SEGMENT_PREFIX.match(s.name).group(1) for s in segments}
    if len(names) != 1 or len(prefixes) != 1 or not all(s.path for s in segments):
        return last
    pattern = _join(_join(folder, f"{prefixes.pop()}*"), names.pop())
    if remote.is_remote(folder):
        try:
            found = remote.glob(pattern)
        except OSError:
            return last
    else:
        import glob

        found = glob.glob(pattern)
    if sorted(found) != sorted(s.path for s in segments):
        return last
    return pattern


# ----------------------------------------------------------------------- workflow
def _mtime(folder: str) -> float:
    try:
        if remote.is_remote(folder):
            return float(remote.stat(folder).st_mtime or 0)
        return os.path.getmtime(folder)
    except OSError:
        return 0.0


def _expand(path) -> str:
    path = str(path)
    if remote.is_remote(path):
        return path.rstrip("/") or path
    return os.path.abspath(os.path.expanduser(path))


def _is_folder(path: str) -> bool:
    if remote.is_remote(path):
        import stat as st

        try:
            return st.S_ISDIR(remote.stat(path).st_mode or 0)
        except OSError:
            return False
    return os.path.isdir(path)


def _last_result(steps) -> WorkflowStep | None:
    """The last step with a result; a group without one stands for its own last step."""
    for s in reversed(steps):
        if s.path:
            return s
        inner = _last_result(s.steps)
        if inner is not None:
            return inner
    return None


def describe_folder(path) -> Workflow:
    """The workflow a folder of calculations is (see the module docstring). Raises
    ``LookupError`` when the folder holds no calculation ewald recognizes."""
    folder = _expand(path)
    if not _is_folder(folder):
        raise LookupError(f"{path}: not a folder")
    name = _basename(folder)
    entries = _entries(folder)
    top = _walk(name, folder, 0, entries)
    if top is None:
        raise LookupError(f"{path}: no VASP, LAMMPS, NEB or phonopy calculation found in it "
                          f"(looked {MAX_DEPTH} folders deep)")
    # one calculation, or restarts of one, is a workflow of one step; otherwise its steps
    steps = [top] if not top.steps or top.runs > 1 else top.steps
    kinds = []
    for _, s in Workflow("", steps=steps).walk():
        k = s.scalars.get("kind")
        if k and (not kinds or kinds[-1] != k) and not s.steps:
            kinds.append(str(k))
    last = _last_result(steps)
    mtime = _mtime(folder)
    wf = Workflow(name, id=folder, path=last.path if last else "", steps=steps,
                  kind=" → ".join(kinds[:6]) or "folder", label=name, work_dir=folder,
                  ctime=time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "",
                  complete=True)
    states = {s.state for _, s in wf.walk() if not s.steps}
    wf.state = "finished" if states == {"finished"} else "running" if "running" in states else ""
    return wf


def is_workflow_folder(path) -> bool:
    """A folder with more than one calculation in it (or below it): what opens as a workflow
    rather than as one data file. A single NEB folder is not one (it opens as the band)."""
    try:
        folder = _expand(path)
        if not _is_folder(folder):
            return False
        wf = describe_folder(folder)
    except (LookupError, OSError):
        return False
    return sum(1 for _, s in wf.walk() if not s.steps) > 1 or any(s.runs > 1 for _, s in wf.walk())


# ----------------------------------------------------------------------- recent folders
def _recent_path():
    from .settings import config_dir

    return config_dir() / "workflow_folders.json"


def recent_folders() -> list[str]:
    """Folders opened as workflows lately, newest first."""
    try:
        got = json.loads(_recent_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(p) for p in got if isinstance(p, str)][:MAX_RECENT] if isinstance(got, list) else []


def add_recent_folder(path) -> None:
    folder = _expand(path)
    items = [folder] + [p for p in recent_folders() if p != folder]
    try:
        _recent_path().parent.mkdir(parents=True, exist_ok=True)
        _recent_path().write_text(json.dumps(items[:MAX_RECENT]), encoding="utf-8", newline="\n")
    except OSError:
        pass


class FolderSource:
    """Folders of VASP, LAMMPS, NEB and phonopy calculations on this computer or an SSH host."""

    title = "Folders"
    schemes = ()
    browse_folders = True      # the side bar offers "Open folder…"

    def account(self):
        return "Folders on this computer and SSH hosts"

    def list(self, query=None, cursor=None):
        """The recent workflow folders (newest first), or ``query`` itself when it is a folder."""
        out = []
        if query and (remote.is_remote(query) or os.path.isdir(os.path.expanduser(query))):
            try:
                return WorkflowPage([describe_folder(query)])
            except (LookupError, OSError) as exc:
                raise LookupError(str(exc)) from exc
        for folder in recent_folders():
            if query and query.lower() not in folder.lower():
                continue
            if not remote.is_remote(folder) and not os.path.isdir(folder):
                continue
            out.append(Workflow(_basename(folder), id=folder, path=folder, label=_basename(folder),
                                kind="folder", work_dir=folder))
        return WorkflowPage(out)

    def describe(self, ref):
        return describe_folder(ref)
