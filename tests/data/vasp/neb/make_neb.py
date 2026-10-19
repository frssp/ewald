"""Write the VASP NEB folder used by tests/test_neb.py (images 00..04).

The band is a real NEB run, done with ASE's EMT potential instead of VASP: a vacancy hop
in a 2x2x2 fcc Cu supercell, three moving images, climbing image. Each image folder gets
what a VTST run leaves there: POSCAR (the initial band), and for every image an OUTCAR
holding one ionic step (lattice, positions, forces, energies) in VASP 6 layout; the
intermediate images also get a CONTCAR. Image 04 has no OUTCAR, so a missing endpoint
energy is covered too. Run from this folder: ``python make_neb.py``.
"""
import os

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from ase.mep import NEB
from ase.optimize import BFGS, FIRE

N_MOVING = 3

initial = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
del initial[0]  # vacancy at the origin
final = initial.copy()
hop = int(np.argmin(np.linalg.norm(initial.positions, axis=1)))  # a nearest neighbour of the vacancy
final.positions[hop] = 0.0
for end in (initial, final):
    end.calc = EMT()
    BFGS(end, logfile=None).run(fmax=0.01)

images = [initial] + [initial.copy() for _ in range(N_MOVING)] + [final]
for im in images[1:-1]:
    im.calc = EMT()
band = NEB(images, climb=True, method="improvedtangent")
band.interpolate(mic=True)
start = [im.copy() for im in images]
FIRE(band, logfile=None).run(fmax=0.02, steps=500)

HEAD = """ vasp.6.4.2 20Jul23 (build Nov 20 2023 10:00:00) complex
 POTCAR:    PAW_PBE Cu 22Jun2005
 POTCAR:    PAW_PBE Cu 22Jun2005
   number of dos      NEDOS =    301   number of ions     NIONS = {n:6d}
   ions per type = {n:15d}
"""


def outcar(atoms, path):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    with open(path, "w") as fh:
        fh.write(HEAD.format(n=len(atoms)))
        fh.write("\n      direct lattice vectors                 reciprocal lattice vectors\n")
        rec = np.linalg.inv(atoms.cell.array).T
        for a, b in zip(atoms.cell.array, rec):
            fh.write("  " + "".join(f"{x:13.9f}" for x in a) + "  " + "".join(f"{x:13.9f}" for x in b) + "\n")
        fh.write("\n------------------------ Iteration      1(   1)  ------------------------\n\n")
        fh.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        fh.write(" " + "-" * 83 + "\n")
        for p, f in zip(atoms.positions, forces):
            fh.write("  " + "".join(f"{x:12.5f}" for x in p) + "    " + "".join(f"{x:14.6f}" for x in f) + "\n")
        fh.write(" " + "-" * 83 + "\n")
        fh.write(f"\n  free  energy   TOTEN  = {energy:18.8f} eV\n\n")
        fh.write(f"  energy  without entropy= {energy:18.8f}  energy(sigma->0) = {energy:18.8f}\n")
        fh.write("\n\n General timing and accounting informations for this job:\n"
                 " ========================================================\n")


for i, (im, im0) in enumerate(zip(images, start)):
    folder = f"{i:02d}"
    os.makedirs(folder, exist_ok=True)
    write(os.path.join(folder, "POSCAR"), im0, format="vasp", direct=True)
    if 0 < i < len(images) - 1:
        write(os.path.join(folder, "CONTCAR"), im, format="vasp", direct=True)
    if i < len(images) - 1:
        outcar(im, os.path.join(folder, "OUTCAR"))
    print(folder, round(im.get_potential_energy(), 5))
