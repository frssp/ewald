"""ewald step-1 demo: read LAMMPS/VASP trajectories, run modifiers, render PNGs.

    python examples/demo.py [outdir]          # CPU
    python examples/demo.py [outdir] --cuda   # GPU analysis (needs CuPy + NVIDIA GPU)

Generates its own sample data so it runs anywhere.
"""
import sys
import time
from pathlib import Path

import numpy as np
from ase.build import bulk
import ase.io

from ewald import DataCollection, import_file
from ewald.io.lammps_dump import write_lammps_dump
from ewald.modifiers import (CalculateDisplacements, ColorCoding, CoordinationAnalysis, DeleteSelected,
                             ExpressionSelection, Slice)

out = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "demo_out")
device = "cuda" if "--cuda" in sys.argv else "cpu"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# --- 1. sample data: Cu block with a spherical void, "annealed" over 5 frames (LAMMPS dump)
cu = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((16, 16, 16)))
center = cu.cell.matrix.diagonal() / 2
cu = cu.delete(np.linalg.norm(cu.positions - center, axis=1) < 12.0)
cu.particles["Particle Identifier"] = np.arange(1, cu.count + 1)
frames = []
for k in range(5):
    f = cu.copy()
    f.positions = cu.positions + rng.normal(0, 0.015 + 0.015 * k, cu.positions.shape)
    f.attributes["Timestep"] = 1000 * k
    frames.append(f)
write_lammps_dump(out / "cu_void.lammpstrj", frames, columns=("id", "type", "x", "y", "z"))

# --- 2. LAMMPS pipeline: coordination -> color -> cut open with a slice
pipe = import_file(out / "cu_void.lammpstrj", type_map={1: "Cu"})
print(pipe, "| atoms:", pipe.compute(0).count)
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.08, number_of_bins=150))
pipe.modifiers.append(ColorCoding("Coordination", cmap="coolwarm_r", start=6, end=12))
pipe.modifiers.append(Slice(normal=(0, 1, 0), distance=center[1], inverse=True))
t = time.perf_counter()
data = pipe.compute(4, device=device)
print(f"frame 4 pipeline on {device}: {time.perf_counter() - t:.2f} s")
pipe.render(out / "cu_void_coordination.png", frame=4, direction=(0.4, 1, 0.3), size=(1600, 1200),
            text="Cu void, frame 4: coordination (cut at y = L/2)")

# RDF table -> matplotlib figure (publication style)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rdf = data.tables["coordination-rdf"]
fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
ax.plot(rdf["r"], rdf["g(r)"], lw=1.2)
ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)"); ax.set_ylabel(r"$g(r)$")
fig.tight_layout(); fig.savefig(out / "cu_rdf.png")

# --- 3. displacement + an OVITO-style Python modifier (user function)
def surface_shell(frame, data):
    """Custom modifier: flag under-coordinated atoms and record their fraction."""
    data.particles["Surface"] = (data["Coordination"] < 12).astype(np.int8)
    data.attributes["SurfaceFraction"] = float(data["Surface"].mean())

disp = import_file(out / "cu_void.lammpstrj", type_map={1: "Cu"})
disp.modifiers.append(CalculateDisplacements(reference_frame=0))
disp.modifiers.append(CoordinationAnalysis(cutoff=3.08))
disp.modifiers.append(surface_shell)
disp.modifiers.append(ExpressionSelection("Surface == 0"))   # keep only the void surface
disp.modifiers.append(DeleteSelected())
disp.modifiers.append(ColorCoding("Displacement Magnitude", cmap="inferno"))
for d in disp.frames(device=device):
    print(f"  timestep {d.attributes['Timestep']:5d}: MSD = {d.attributes.get('MSD', 0):.4f} A^2, "
          f"surface atoms = {d.count}")
disp.render(out / "cu_void_surface_displacement.png", frame=4, direction="iso", size=(1400, 1200),
            text="Void surface atoms colored by |displacement|")

# --- 4. VASP XDATCAR: Si diamond MD-like trajectory with bonds
si = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
images = []
for k in range(3):
    a = si.copy()
    a.positions += rng.normal(0, 0.03 * (k + 1), a.positions.shape)
    images.append(a)
ase.io.write(out / "XDATCAR", images, format="vasp-xdatcar")
vasp = import_file(out / "XDATCAR")
vasp.render(out / "si_xdatcar_bonds.png", frame=2, bonds=2.6, direction=(1, 0.3, 0.2), size=(1400, 1200),
            radius_scale=0.5, text="Si (XDATCAR, frame 2) with bonds")
print("images written to", out.resolve())
