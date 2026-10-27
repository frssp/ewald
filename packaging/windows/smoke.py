"""Smoke test for the frozen app: ``ewald run smoke.py DATA_DIR OUT_DIR``.

Touches each bundled stack once: readers (native + ASE), the C++ extension, numba kernels,
VTK offscreen rendering, the numba ray tracer, and the Qt main window with its console.
"""
import faulthandler
import os
import sys
from pathlib import Path

# a crash in native code (VTK, numba, Qt) then names the line, and output is not lost in the pipe
faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

data_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

import ewald
from ewald import import_file
from ewald import modifiers as m
from ewald import _native  # noqa: F401  C++ extension
from ewald.neighbors import find_pairs

print("ewald", getattr(ewald, "__version__", "?"), "frozen:", getattr(sys, "frozen", False))

pipe = import_file(data_dir / "lammps" / "dump.custom.lammpstrj")
print("lammps dump:", pipe.num_frames, "frames")
pipe = import_file(data_dir / "vasp" / "XDATCAR")
print("XDATCAR:", pipe.num_frames, "frames")
pipe = import_file(data_dir / "vasp" / "OUTCAR")   # ASE reader
print("OUTCAR:", pipe.num_frames, "frames")

from ase.build import bulk
from ewald import DataCollection

cu = DataCollection.from_ase(bulk("Cu", "fcc", a=3.615, cubic=True).repeat((6, 6, 6)))
i, j = find_pairs(cu.positions, cu.cell, 3.0)[:2]
assert len(i) > 0, "numba neighbor list returned no pairs"
print("pairs:", len(i))

import ase.io

ase.io.write(out / "POSCAR_cu", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((6, 6, 6)), format="vasp")
pipe = import_file(out / "POSCAR_cu")
pipe.modifiers.append(m.CommonNeighborAnalysis())
pipe.modifiers.append(m.CoordinationAnalysis(cutoff=3.0))
data = pipe.compute(0)
print("CNA counts:", {k: v for k, v in data.attributes.items() if "CommonNeighborAnalysis" in k})

from ewald.render import render

render(data, out / "opengl.png", size=(320, 240))
render(data, out / "raytrace.png", size=(160, 120), renderer="raytrace", samples=1, ao_samples=2)
for f in ("opengl.png", "raytrace.png"):
    assert (out / f).stat().st_size > 1000, f
print("renders ok")

from ewald import settings

print("devices:", [(k, probe()[0]) for k, _, probe in settings.DEVICES])

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from ewald.gui.app import MainWindow, _app

app = _app()
win = MainWindow(pipe)
win.show()
app.processEvents()
print("console:", type(win.console).__name__)
win.close()
print("SMOKE OK")
