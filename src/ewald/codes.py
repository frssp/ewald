"""Simulation codes: what EWALD knows about each program that wrote the files, in one place.

VASP and LAMMPS are built-in entries of this registry; any other code (Quantum ESPRESSO, CP2K,
GPAW, ...) is added the same way by a plugin, with the :func:`ewald.plugins.simulation_code`
decorator or an ``ewald.codes`` entry point::

    from ewald.plugin_api import SimulationCode, simulation_code

    @simulation_code
    def qe():
        return SimulationCode("qe", "Quantum ESPRESSO", files=("*.pwo", "pw.out"),
                              markers=("pw.in",), run_info=read_qe_run)

A code bundles

- ``files``: file names it writes (case-insensitive patterns), so :func:`code_of` knows a
  file's code; ``detect(path)`` for files known by content; ``markers`` are files whose presence
  next to a file (or in a folder) says the run is this code's (``INCAR``, ``log.lammps``);
- ``formats``: the reader names of its file formats (shown in Preferences › Codes);
- ``options``: :class:`ewald.settings.Option` rows under ``codes.<name>.*``, which Preferences ›
  Codes shows and scripts read with ``settings.load()["codes.lammps.type_map"]``;
- hooks, each a callable or a lazy ``"module:function"`` string: ``run_info(info, path)`` fills
  :class:`ewald.runinfo.RunInfo` from the input files, ``check(check, info, path)`` adds
  :class:`ewald.runinfo.Finding`\\ s, ``frame_steps(info, path, steps)`` returns MD steps between
  stored frames, ``workflow_step(name, folder, entries)`` turns a folder into a
  :class:`ewald.workflow.WorkflowStep` (or None), ``open_options(path, settings, ask)`` returns
  reader keyword arguments when the GUI opens a file (``ask(title, label, text)`` asks the user;
  None = cancelled), ``read_thermo(path)`` reads a per-step table;
- ``thermo``: reader name -> per-step result files next to such a file (``{"xdatcar":
  ("OSZICAR",)}``), joined to the frames by ``import_file(..., thermo="auto")``;
- ``plugin``: the built-in plugin group whose switch also turns the code off.

``codes()`` lists them (built-ins first), :func:`code_of` finds a file's code, :func:`get` one
by name. Nothing here runs code from the files; hooks only parse them.
"""
from __future__ import annotations

import fnmatch
import importlib
import posixpath
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Callable

__all__ = ["SimulationCode", "codes", "get", "code_of", "code_for_folder", "builtin_codes", "parse_type_map"]

_HOOKS = ("run_info", "check", "frame_steps", "workflow_step", "open_options", "read_thermo", "detect")


@dataclass
class SimulationCode:
    """One simulation code; see the module docstring for the fields."""

    name: str
    title: str = ""
    description: str = ""
    files: tuple = ()
    markers: tuple = ()
    formats: tuple = ()
    options: tuple = ()
    thermo: dict = field(default_factory=dict)
    plugin: str = ""
    run_info: Callable | str | None = None
    check: Callable | str | None = None
    frame_steps: Callable | str | None = None
    workflow_step: Callable | str | None = None
    open_options: Callable | str | None = None
    read_thermo: Callable | str | None = None
    detect: Callable | str | None = None
    origin: str = "python"

    def __post_init__(self):
        self.name = str(self.name).strip().lower()
        if not self.name or "." in self.name or any(c.isspace() for c in self.name):
            raise ValueError(f"simulation code name {self.name!r}: one word without dots")
        self.title = self.title or self.name
        for attr in ("files", "markers", "formats", "options"):
            value = getattr(self, attr)
            setattr(self, attr, (value,) if isinstance(value, str) else tuple(value or ()))
        prefix = f"codes.{self.name}."
        for opt in self.options:
            if not getattr(opt, "key", "").startswith(prefix):
                raise ValueError(f"simulation code {self.name!r}: option {getattr(opt, 'key', opt)!r} "
                                 f"must be named {prefix}<name>")
        self.thermo = {str(k): (v,) if isinstance(v, str) else tuple(v) for k, v in dict(self.thermo or {}).items()}

    def hook(self, name: str) -> Callable | None:
        """The hook ``name`` as a callable (a ``"module:function"`` string is imported now)."""
        fn = getattr(self, name)
        if isinstance(fn, str):
            module, _, attr = fn.partition(":")
            fn = getattr(importlib.import_module(module), attr)
            setattr(self, name, fn)
        return fn

    def matches(self, path) -> bool:
        """Whether the file ``path`` (local, remote or a bare name) is one this code writes,
        by its name and then by ``detect``."""
        name = _basename(str(path)).lower()
        if any(fnmatch.fnmatchcase(name, p.lower()) for p in self.files):
            return True
        detect = self.hook("detect")
        if detect is None:
            return False
        try:
            return bool(detect(str(path)))
        except (OSError, ValueError, UnicodeError):
            return False

    def as_dict(self) -> dict:
        """Plain data for the CLI, the AI tools and the inventory."""
        return {"name": self.name, "title": self.title, "description": self.description,
                "files": list(self.files), "markers": list(self.markers), "formats": list(self.formats),
                "options": [o.key for o in self.options], "thermo": {k: list(v) for k, v in self.thermo.items()},
                "hooks": [h for h in _HOOKS if getattr(self, h) is not None], "plugin": self.plugin,
                "origin": self.origin}

    def __repr__(self):
        return f"<SimulationCode {self.name} ({self.origin})>"


def _basename(path: str) -> str:
    from .io import remote

    if remote.is_remote(path):
        return posixpath.basename(remote.parse_remote(path).path)
    return PureWindowsPath(path).name   # "/" and "\\" both separate: a path from any OS


def codes() -> dict[str, SimulationCode]:
    """Registered codes by name: the built-in ones (VASP, LAMMPS) first, then plugins'."""
    from . import plugins

    return plugins.simulation_codes()


def get(name: str) -> SimulationCode:
    """The code ``name`` (case-insensitive); KeyError lists the known ones."""
    found = codes()
    key = str(name).lower()
    if key not in found:
        raise KeyError(f"no simulation code {name!r}; known: {', '.join(found)}")
    return found[key]


def code_of(path) -> SimulationCode | None:
    """The code that wrote the file ``path``: by file name first (every code), then by content."""
    found = list(codes().values())
    name = _basename(str(path)).lower()
    for c in found:
        if any(fnmatch.fnmatchcase(name, p.lower()) for p in c.files):
            return c
    for c in found:
        if c.detect is not None and c.matches(path):
            return c
    return None


def code_for_folder(path, exists: Callable[[str, str], str | None]) -> SimulationCode | None:
    """The code whose ``markers`` sit next to ``path``; ``exists(path, name)`` returns the
    marker's path or None (local or remote)."""
    for c in codes().values():
        if any(exists(path, m) for m in c.markers):
            return c
    return None


def parse_type_map(text: str) -> dict:
    """``"1=Cu, 2=Zr"`` -> ``{"1": "Cu", "2": "Zr"}``; ValueError says what is wrong."""
    out = {}
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"type map entry {part!r}: write type=element, e.g. 1=Cu")
        out[key] = value
    return out


# ------------------------------------------------------------------------ built-in codes
def _lammps_open_options(path, settings, ask) -> dict | None:
    """Ask for a type -> element map when a LAMMPS dump is opened (``codes.lammps.type_prompt``)."""
    from .io.lammps_dump import is_lammps_dump

    if not is_lammps_dump(path):
        return {}
    default = settings["codes.lammps.type_map"]
    if settings["codes.lammps.type_prompt"] == "ask":
        text = ask("LAMMPS types", "Optional type → element map (e.g. 1=Cu,2=Zr):", default)
        if text is None:
            return {}
    else:
        text = default
    return {"type_map": parse_type_map(text)} if text.strip() else {}


def _lammps_detect(path) -> bool:
    from .io.lammps_dump import is_lammps_dump

    return is_lammps_dump(path)


def builtin_codes() -> dict[str, SimulationCode]:
    """VASP and LAMMPS, registered like any plugin code (origin ``built-in``)."""
    from .settings import Option

    vasp = SimulationCode(
        "vasp", "VASP", "Vienna Ab initio Simulation Package: structures, MD and relaxations, "
        "electronic structure, phonons and NEB runs.",
        files=("XDATCAR*", "OUTCAR*", "vasprun*", "OSZICAR*", "POSCAR*", "CONTCAR*", "vaspout*"),
        markers=("INCAR",),
        formats=("poscar", "xdatcar", "outcar", "vasprun", "vaspout", "vaspwave", "vasp-volumetric", "vasp-phonon",
                 "neb"),
        thermo={"xdatcar": ("OSZICAR",)}, plugin="vasp",
        run_info="ewald.runinfo:_vasp_info", check="ewald.runinfo:_check_vasp",
        frame_steps="ewald.runinfo:_vasp_frame_steps", workflow_step="ewald.workflow_folder:_vasp_step",
        origin="built-in")
    lammps = SimulationCode(
        "lammps", "LAMMPS", "Large-scale Atomic/Molecular Massively Parallel Simulator: text dumps, data files "
        "and log.lammps thermo output.",
        files=("*.lammpstrj", "*.lammpstrj.gz", "*.dump", "*.dump.gz", "dump*", "log.lammps*", "*.data"),
        markers=("log.lammps",),
        formats=("lammps-dump", "lammps-data", "lammps-data-ase"),
        options=(Option("codes.lammps.type_prompt", "ask", "Type names",
                        "Ask for a type → element map when opening a LAMMPS dump.", choices=("ask", "never")),
                 Option("codes.lammps.type_map", "", "Default type map", "e.g. 1=Cu,2=Zr; pre-fills the prompt.")),
        thermo={"lammps-dump": ("log.lammps",)}, plugin="lammps",
        run_info="ewald.runinfo:_lammps_info", check="ewald.runinfo:_check_lammps",
        frame_steps="ewald.runinfo:_lammps_frame_steps", workflow_step="ewald.workflow_folder:_lammps_step",
        open_options=_lammps_open_options, read_thermo="ewald.io.thermo:read_lammps_log", detect=_lammps_detect,
        origin="built-in")
    vasp.read_thermo = "ewald.io.vasp_output:read_oszicar"
    return {"vasp": vasp, "lammps": lammps}


def builtin_option(key: str):
    """The :class:`ewald.settings.Option` of a built-in code (no plugin loading), or None."""
    name = key.split(".")[1] if key.count(".") >= 2 else ""
    code = _builtin_cache().get(name)
    return next((o for o in code.options if o.key == key), None) if code else None


_BUILTIN: dict = {}


def _builtin_cache() -> dict:
    if not _BUILTIN:
        _BUILTIN.update(builtin_codes())
    return _BUILTIN
