"""Regenerate the phonon fixtures in this folder (needs ``pip install phonopy``).

L1_2 Cu3Au with ASE's EMT potential, force constants from phonopy on a 2x2x2 supercell:

- ``band.yaml``: band structure G-X-M-G-R with eigenvectors (``phonopy --eigvecs`` layout).
- ``qpoints.yaml``: modes at G and X with eigenvectors.
- ``OUTCAR.phonon``: the G modes written the way VASP prints them for IBRION=5-8 (header with
  the lines ewald's OUTCAR reader needs, then "Eigenvectors and eigenvalues of the dynamical
  matrix", highest mode first, imaginary modes as ``f/i=``).
- ``modulation_reference.npz``: phonopy's own modulated supercells (``run_modulations``) for a few
  modes, which ``tests/test_phonon.py`` compares ewald's displacement patterns against.

Run ``python tests/data/phonon/make_phonon_reference.py`` from the repository root; never edit
the outputs by hand.
"""
from __future__ import annotations

import os

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms

HERE = os.path.dirname(os.path.abspath(__file__))
A = 3.75
MODES = [((0.0, 0.0, 0.0), 5, (1, 1, 1)), ((0.5, 0.0, 0.0), 2, (2, 1, 1)), ((0.5, 0.5, 0.0), 7, (2, 2, 1)),
         ((0.5, 0.5, 0.5), 10, (2, 2, 2)), ((0.25, 0.0, 0.0), 4, (4, 1, 1))]


def unitcell():
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return PhonopyAtoms(symbols=["Au", "Cu", "Cu", "Cu"], cell=np.eye(3) * A, scaled_positions=frac)


def to_ase(cell):
    from ase import Atoms

    return Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)


def main():
    ph = Phonopy(unitcell(), supercell_matrix=np.diag([2, 2, 2]))
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        atoms = to_ase(sc)
        atoms.calc = EMT()
        forces.append(atoms.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()

    path = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["$\\Gamma$", "X", "M", "$\\Gamma$", "R"]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=5)
    ph.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=True)
    ph.write_yaml_band_structure(filename=os.path.join(HERE, "band.yaml"))

    ph.run_qpoints([[0, 0, 0], [0.5, 0, 0]], with_eigenvectors=True)
    cwd = os.getcwd()
    os.chdir(HERE)  # phonopy writes qpoints.yaml into the working directory
    try:
        ph.write_yaml_qpoints_phonon()
    finally:
        os.chdir(cwd)
    freqs, vecs = ph.qpoints.frequencies[0], ph.qpoints.eigenvectors[0]
    write_outcar(os.path.join(HERE, "OUTCAR.phonon"), ph.primitive, freqs, vecs.real)

    out = {}
    for k, (q, band, dim) in enumerate(MODES):
        ph.run_modulations(dim, [[q, band, 1.0, 0.0]])
        (u,), sc = ph.modulation.modulations, ph.modulation.supercell
        out[f"q{k}"] = np.array(q)
        out[f"band{k}"] = band
        out[f"dim{k}"] = np.array(dim)
        out[f"positions{k}"] = sc.positions
        out[f"symbols{k}"] = np.array(sc.symbols)
        out[f"u{k}"] = u
    np.savez_compressed(os.path.join(HERE, "modulation_reference.npz"), **out)


def write_outcar(path, prim, freqs, vecs):
    """Gamma modes in VASP's OUTCAR layout (eigenvectors of the dynamical matrix, which are
    real at Gamma; phonopy's are real here too because the dynamical matrix is real)."""
    pos, cell = prim.positions, prim.cell
    species = list(dict.fromkeys(prim.symbols))
    counts = [list(prim.symbols).count(s) for s in species]
    masses = {s: m for s, m in zip(prim.symbols, prim.masses)}
    lines = [" vasp.6.4.2 18Apr23 (build Jan 01 2024 00:00:00) complex", ""]
    lines += [f" POTCAR:    PAW_PBE {s} 06Sep2000" for s in species]
    lines += [f"   POMASS =  {masses[s]:7.3f}; ZVAL   =   11.000    mass and valenz" for s in species]
    lines += ["   number of dos      NEDOS =    301   number of ions     NIONS =      %d" % len(pos),
              "   ions per type =  " + "".join(f"{c:4d}" for c in counts),
              "   IBRION =      8    ionic relax: 0-MD 1-quasi-New 2-CG",
              "   Mass of Ions in am",
              "   POMASS =" + "".join(f"{masses[s]:6.2f}" for s in species), ""]
    lines += ["--------------------------------------- Iteration      1(   1)  ---------------------------------------", "",
              " direct lattice vectors                 reciprocal lattice vectors"]
    inv = np.linalg.inv(cell).T
    lines += ["  " + "".join(f"{x:13.9f}" for x in cell[i]) + "  " + "".join(f"{x:13.9f}" for x in inv[i]) for i in range(3)]
    lines += ["", " POSITION                                       TOTAL-FORCE (eV/Angst)",
              " -----------------------------------------------------------------------------------"]
    lines += ["   " + "".join(f"{x:13.5f}" for x in p) + "      " + "".join(f"{0.0:13.6f}" for _ in range(3)) for p in pos]
    lines += [" -----------------------------------------------------------------------------------",
              "    total drift:                                0.000000      0.000000      0.000000", "",
              "  free  energy   TOTEN  =       -12.34567890 eV", "",
              "  energy  without entropy=      -12.34567890  energy(sigma->0) =      -12.34567890", "",
              " SECOND DERIVATIVES (NOT SYMMETRIZED)", " ------------------------------------", "",
              " Eigenvectors and eigenvalues of the dynamical matrix",
              " ----------------------------------------------------", "", ""]
    n = len(pos)
    order = np.argsort(-freqs)  # VASP prints the highest mode first
    for k, b in enumerate(order, 1):
        f = freqs[b]
        tag = "f  =" if f >= 0 else "f/i="
        f = abs(f)
        lines.append(f"{k:4d} {tag}{f:12.6f} THz {2 * np.pi * f:12.6f} 2PiTHz {f * 33.35641:12.6f} cm-1 "
                     f"{f * 4.135668:12.6f} meV")
        lines.append("             X         Y         Z           dx          dy          dz")
        e = vecs[:, b].reshape(n, 3)
        lines += ["  " + "".join(f"{x:10.6f}" for x in pos[i]) + "  " + "".join(f"{x:12.6f}" for x in e[i])
                  for i in range(n)]
        lines.append("")
    lines += ["", " Eigenvectors after division by SQRT(mass)", " -----------------------------------------", "", ""]
    w = np.repeat([masses[s] for s in prim.symbols], 3)[:, None]
    dv = vecs / np.sqrt(w)
    for k, b in enumerate(order, 1):
        f = abs(freqs[b])
        lines.append(f"{k:4d} {'f  =' if freqs[b] >= 0 else 'f/i='}{f:12.6f} THz {2 * np.pi * f:12.6f} 2PiTHz "
                     f"{f * 33.35641:12.6f} cm-1 {f * 4.135668:12.6f} meV")
        lines.append("             X         Y         Z           dx          dy          dz")
        e = dv[:, b].reshape(n, 3)
        lines += ["  " + "".join(f"{x:10.6f}" for x in pos[i]) + "  " + "".join(f"{x:12.6f}" for x in e[i])
                  for i in range(n)]
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
