"""Write the small vaspout.h5 / vaspwave.h5 fixtures in this folder.

The layout follows py4vasp's schema (py4vasp/_raw/definition.py, VASP 6.5): ion
history under /intermediate/ion_dynamics, final structure under /results/positions,
grids as (component, z, y, x). tests/test_vasp_h5.py reads them back with py4vasp
too when it is installed. Run: python tests/data/vasp/h5/make_vasp_h5.py
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)


def common(f, types, counts, lattice, scale, final_frac):
    for k, v in zip(("major", "minor", "patch"), (6, 5, 1)):
        f[f"version/{k}"] = v
    f["input/incar/SYSTEM"] = np.bytes_("ewald test")
    f["results/positions/ion_types"] = np.array([t.ljust(2) for t in types], dtype="S2")
    f["results/positions/number_ion_types"] = np.array(counts)
    f["results/positions/scale"] = scale
    f["results/positions/lattice_vectors"] = lattice
    f["results/positions/position_ions"] = final_frac


def md():
    """FeO, 4 atoms, 3 MD steps, ISPIN=2, fixed cell, local potential written."""
    nsteps, n = 3, 4
    lattice = np.array([[4.3, 0, 0], [0, 4.3, 0], [0.1, 0, 4.2]])
    frac0 = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], float)
    frac = (frac0 + 0.01 * rng.standard_normal((nsteps, n, 3))) % 1.0
    with h5py.File(os.path.join(HERE, "md", "vaspout.h5"), "w") as f:
        common(f, ["Fe", "O"], [2, 2], lattice, 1.0, frac[-1])
        d = f.create_group("intermediate/ion_dynamics")
        d["scale"] = 1.0
        d["lattice_vectors"] = np.repeat(lattice[None], nsteps, axis=0)
        d["position_ions"] = frac
        d["forces"] = rng.standard_normal((nsteps, n, 3))
        d["ion_velocities"] = 0.01 * rng.standard_normal((nsteps, n, 3))
        stress = rng.standard_normal((nsteps, 3, 3)) * 10
        d["stress"] = 0.5 * (stress + stress.transpose(0, 2, 1))
        labels = ["ion-electron   TOTEN", "kinetic energy EKIN", "kin. lattice   EKIN_LAT",
                  "temperature    TEIN", "nose potential ES", "nose kinetic   EPS", "total energy   ETOTAL"]
        d["energies_tags"] = np.array(labels, dtype="S")
        e = np.zeros((nsteps, len(labels)))
        e[:, 0] = [-30.1, -30.05, -30.02]
        e[:, 1] = [0.10, 0.12, 0.14]
        e[:, 3] = [300.0, 320.0, 340.0]
        e[:, 6] = e[:, 0] + e[:, 1]
        d["energies"] = e
        moments = np.zeros((nsteps, 2, n, 3))
        moments[:, 0] = rng.random((nsteps, n, 3))                   # charge
        moments[:, 1, :2] = [[0.5, 0.5, 2.6], [-0.5, -0.5, -2.6]]    # Fe up / down (s p d)
        d["magnetism/spin_moments/values"] = moments
        pot = rng.standard_normal((2, 6, 5, 4))                      # (component, z, y, x)
        f["results/potential/total"] = pot
        f["results/potential/hartree"] = pot[:1]


def relax():
    """Si2, 2 relaxation steps with a changing cell and scale 1.01; plus vaspwave.h5 charge."""
    n = 2
    lat = np.array([[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]],
                    [[0, 2.72, 2.72], [2.72, 0, 2.72], [2.72, 2.72, 0]]])
    frac = np.array([[[0, 0, 0], [0.25, 0.25, 0.25]], [[0, 0, 0], [0.26, 0.25, 0.25]]])
    os.makedirs(os.path.join(HERE, "relax"), exist_ok=True)
    with h5py.File(os.path.join(HERE, "relax", "vaspout.h5"), "w") as f:
        common(f, ["Si"], [2], lat[-1], 1.01, frac[-1])
        d = f.create_group("intermediate/ion_dynamics")
        d["scale"] = 1.01
        d["lattice_vectors"] = lat
        d["position_ions"] = frac
        d["forces"] = np.array([[[0.1, 0, 0], [-0.1, 0, 0]], [[0.01, 0, 0], [-0.01, 0, 0]]])
        d["stress"] = np.array([np.diag([5.0, 5.0, 5.0]), np.diag([1.0, 2.0, 3.0])])
        d["energies_tags"] = np.array(["free energy    TOTEN   ", "energy without entropy ",
                                       "energy(sigma->0)       "], dtype="S")
        d["energies"] = np.array([[-10.80, -10.81, -10.805], [-10.85, -10.86, -10.855]])
    with h5py.File(os.path.join(HERE, "relax", "vaspwave.h5"), "w") as f:
        rho = np.full((2, 8, 6, 4), 0.0)
        rho[0] = 8.0                       # rho * V_cell, as in a CHGCAR: 8/V e/A^3
        rho[0, 1, 2, 3] = 16.0
        rho[1] = 0.5
        f["charge/charge"] = rho


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "md"), exist_ok=True)
    md()
    relax()
