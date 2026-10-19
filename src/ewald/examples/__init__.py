"""Small example files shipped with ewald, one per common format, to open first.

>>> from ewald import examples, import_file
>>> pipe = import_file(examples.path("POSCAR.SrTiO3"))
>>> examples.load("Cu_md.extxyz").num_frames
10

The GUI lists them under File › Open example; ``ewald examples`` prints their paths and
``ewald examples --copy DIR`` copies them somewhere writable. ``make_examples.py`` regenerates
the files (never edit them by hand).

Published data from papers is not shipped but downloaded on demand: :func:`fetch` (see
:mod:`ewald.examples.papers`) feeds the reproduction scripts in ``examples/papers/``.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

FILES = Path(__file__).resolve().parent / "files"


@dataclass(frozen=True)
class Example:
    file: str
    format: str
    description: str

    @property
    def path(self) -> Path:
        return FILES / self.file


EXAMPLES = [
    Example("POSCAR.SrTiO3", "VASP POSCAR", "cubic perovskite SrTiO3, 2x2x2 (40 atoms); try CoordinationPolyhedra on Ti"),
    Example("Si.cif", "CIF", "diamond Si, conventional cell (8 atoms)"),
    Example("Pt_nanoparticle.xyz", "XYZ", "Pt truncated-octahedron nanoparticle, no cell"),
    Example("Cu_md.extxyz", "extended XYZ", "fcc Cu (108 atoms), 10 frames with forces and energy"),
    Example("XDATCAR.Si", "VASP XDATCAR", "diamond Si (64 atoms), 10-frame trajectory"),
    Example("CHGCAR.Si", "VASP CHGCAR", "diamond Si with a synthetic valence density; try CreateIsosurface(isolevel=0.3)"),
    Example("Cu_void.lammpstrj", "LAMMPS dump", "Cu block with a spherical void, 5 frames"),
    Example("CuZr.data", "LAMMPS data", "B2 CuZr (128 atoms), types named in Masses"),
    Example("Cu3Au_band.yaml", "phonopy band.yaml", "Cu3Au phonon bands with eigenvectors; opens playing a mode"),
]


def path(name: str) -> Path:
    """Path of the example file ``name`` (a file name from :data:`EXAMPLES`)."""
    for ex in EXAMPLES:
        if ex.file == name:
            return ex.path
    raise KeyError(f"no example {name!r}; choose one of {[ex.file for ex in EXAMPLES]}")


def load(name: str, **kwargs):
    """``import_file`` on the example ``name``."""
    from .. import import_file

    return import_file(path(name), **kwargs)


def copy(dest) -> Path:
    """Copy every example file into the directory ``dest`` (created if needed)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for ex in EXAMPLES:
        shutil.copyfile(ex.path, dest / ex.file)
    return dest


def fetch(name: str, dest=None, quiet: bool = False) -> Path:
    """Download the published dataset ``name`` (see :data:`papers.PAPERS`) once and return its folder."""
    from . import papers

    return papers.fetch(name, dest, quiet)
