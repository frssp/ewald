"""Space group, Wyckoff positions and symmetry-equivalent atoms through spglib (installed with
ewald)."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier

_SYSTEMS = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
            (167, "trigonal"), (194, "hexagonal"), (230, "cubic"))


def crystal_system(number: int) -> str:
    """Crystal system of space group ``number`` (1-230)."""
    for last, name in _SYSTEMS:
        if number <= last:
            return name
    raise ValueError(f"no space group {number}")


def _spglib():
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("SymmetryAnalysis needs spglib, which ewald installs; reinstall ewald or pip install spglib") from exc
    return spglib


def symmetry_dataset(data, symprec: float = 0.01, angle_tolerance: float = -1.0):
    """spglib's dataset for a ``DataCollection`` (types are the species), or None when spglib
    finds no symmetry at this tolerance."""
    if not data.cell.is_valid:
        raise ValueError("symmetry analysis needs a periodic cell")
    names, codes = np.unique(np.asarray(data.types).astype(str), return_inverse=True)
    frac = data.cell.to_fractional(data.positions)
    cell = (np.asarray(data.cell.matrix, float), frac % 1.0, codes.astype(np.int32) + 1)
    spglib = _spglib()
    err = getattr(spglib, "error", None)
    if err is not None and hasattr(err, "OLD_ERROR_HANDLING"):
        err.OLD_ERROR_HANDLING = False   # spglib >= 2.5: raise instead of returning None
    try:
        return spglib.get_symmetry_dataset(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
    except Exception as exc:   # SpglibError: no symmetry found at this tolerance
        if type(exc).__name__ == "SpglibError":
            return None
        raise


class SymmetryAnalysis(Modifier):
    """Space group, Wyckoff positions and symmetry-equivalent atoms (spglib).

    Particle types are the species; ``symprec`` is the distance tolerance in Angstrom (0.01
    finds the symmetry of relaxed DFT cells; lower it to tell close structures apart) and
    ``angle_tolerance`` in degrees (-1: spglib's default). Writes the attributes
    ``SymmetryAnalysis.space_group`` (Hermann-Mauguin, e.g. 'Fm-3m'), ``.space_group_number``,
    ``.hall_symbol``, ``.point_group``, ``.crystal_system``, ``.num_operations`` (in the input
    cell, lattice translations of a supercell included) and ``.wyckoff_sites`` (e.g.
    'Na 4a, Cl 4b'), and the particle properties 'Wyckoff Position' ('4a': multiplicity in the
    conventional cell and letter), 'Site Symmetry' and 'Equivalent Atom' (index of the first atom
    of its orbit). ``select_unique`` selects one atom per orbit, the asymmetric unit. Without
    symmetry at this tolerance the space group is 'P1'.
    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = -1.0, select_unique: bool = False):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.select_unique = select_unique

    def modify(self, data, ctx):
        n = data.count
        ds = symmetry_dataset(data, self.symprec, self.angle_tolerance) if n else None
        a = data.attributes
        if ds is None:
            equiv = np.arange(n)
            a.update({"SymmetryAnalysis.space_group": "P1", "SymmetryAnalysis.space_group_number": 1,
                      "SymmetryAnalysis.hall_symbol": "P 1", "SymmetryAnalysis.point_group": "1",
                      "SymmetryAnalysis.crystal_system": "triclinic", "SymmetryAnalysis.num_operations": 1})
            wyckoff = np.array(["1a"] * n)
            site = np.array(["1"] * n)
        else:
            equiv = np.asarray(ds.equivalent_atoms, np.int64)
            orbit = np.bincount(equiv, minlength=n)[equiv]
            per_cell = len(ds.std_types) / n          # conventional cell atoms per input atom
            mult = np.rint(orbit * per_cell).astype(int)
            wyckoff = np.array([f"{m}{w}" for m, w in zip(mult, ds.wyckoffs)])
            site = np.array([s.replace(".", "") or "1" for s in ds.site_symmetry_symbols])
            a.update({"SymmetryAnalysis.space_group": str(ds.international),
                      "SymmetryAnalysis.space_group_number": int(ds.number),
                      "SymmetryAnalysis.hall_symbol": str(ds.hall), "SymmetryAnalysis.point_group": str(ds.pointgroup),
                      "SymmetryAnalysis.crystal_system": crystal_system(int(ds.number)),
                      "SymmetryAnalysis.num_operations": int(len(ds.rotations))})
        unique = np.unique(equiv)
        types = np.asarray(data.types).astype(str)
        a["SymmetryAnalysis.num_unique_sites"] = int(len(unique))
        a["SymmetryAnalysis.wyckoff_sites"] = ", ".join(f"{types[i]} {wyckoff[i]}" for i in unique)
        data.particles["Wyckoff Position"] = wyckoff
        data.particles["Site Symmetry"] = site
        data.particles["Equivalent Atom"] = equiv
        if self.select_unique:
            sel = np.zeros(n, np.int8)
            sel[unique] = 1
            data.particles["Selection"] = sel
