"""Write water.h5, a 3-frame MDTraj HDF5 file (what OpenMM's HDF5Reporter writes).

Needs mdtraj and tables. Run: python tests/data/mdtraj/make_mdtraj_h5.py
"""
import os

import mdtraj as md
import numpy as np

top = md.Topology()
chain = top.add_chain()
for k in range(2):
    res = top.add_residue("HOH", chain)
    o = top.add_atom("O", md.element.oxygen, res)
    top.add_bond(o, top.add_atom("H1", md.element.hydrogen, res))
    top.add_bond(o, top.add_atom("H2", md.element.hydrogen, res))
water = np.array([[0, 0, 0], [0.09572, 0, 0], [-0.024, 0.0927, 0]])
xyz = np.concatenate([water, water + [0.5, 0.5, 0.5]])
xyz = np.array([xyz + 0.01 * i for i in range(3)], np.float32)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water.h5")
with md.formats.HDF5TrajectoryFile(path, "w") as f:
    f.topology = top
    f.write(coordinates=xyz, time=np.array([0.0, 2.0, 4.0], np.float32),
            cell_lengths=np.array([[1.0, 1.2, 1.5]] * 3, np.float32),
            cell_angles=np.array([[90.0, 90.0, 120.0]] * 3, np.float32),
            velocities=np.full((3, 6, 3), 0.5, np.float32),
            kineticEnergy=np.array([10.0, 11.0, 12.0], np.float32),
            potentialEnergy=np.array([-100.0, -101.0, -102.0], np.float32),
            temperature=np.array([300.0, 301.0, 302.0], np.float32))
