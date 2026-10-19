# File formats

`import_file(path)` detects the format from the file name and the first lines of the file.
Pass `format=` to force one. Wildcards read a numbered series as one trajectory
(`import_file("dump.*.lammpstrj")`), and the native readers read `.gz`, `.bz2` and `.xz`
files directly.

| Format | Reader | Notes |
|---|---|---|
| LAMMPS text dump (`dump atom/custom`) | native, lazy | orthogonal, `xy xz yz` and general (`abc origin`) triclinic boxes; `x`/`xu`/`xs`/`xsu`; `c_x[1] c_x[2]` become one vector property; `dump_modify units/time yes`; a partial last frame is skipped with a warning |
| LAMMPS data file | native | styles atomic, charge, bond/angle/molecular, full, sphere, dipole (from `Atoms # style`, inferred, or `atom_style=`); type names from `Atom Type Labels` or `Masses` comments; velocities, image flags, bonds in `data.bonds` |
| VASP XDATCAR | native, lazy | fixed and variable cell (NpT), VASP 6.4 `Si/hash` species labels |
| VASP POSCAR/CONTCAR | ASE | VASP 4 and 5, selective dynamics, CONTCAR velocities (`Velocity`, Å/fs) |
| VASP OUTCAR, vasprun.xml | ASE | every ionic step, with `Energy`, `Force`, `Stress` |
| extxyz, CIF, and others | ASE | anything `ase.io.read` understands |

"Lazy" means the file is indexed once and each frame is read when it is needed, so a
multi-gigabyte trajectory opens immediately and uses memory for the frames in the cache only.

## Reader options

Keyword arguments to `import_file` go to the reader:

```python
import_file("dump.lammpstrj", type_map={1: "Cu", 2: "Zr"})     # LAMMPS numeric types to elements
import_file("system.data", atom_style="full")                  # data file without an Atoms # style hint
import_file("structure.data", format="lammps-data-ase")        # use ASE's LAMMPS data reader instead
```

LAMMPS values stay in the file's unit style; the dump's `Units` attribute is kept when the
file has one.

## Property names

Readers name properties as OVITO does: `Position`, `Particle Type`, `Particle Identifier`,
`Velocity`, `Force`, `Charge`, `Mass`, and so on. `data.summary()` lists what a file produced.

## Writing files

```python
from ewald.io import export_file

export_file(pipe, "out.lammpstrj", format="lammps/dump")     # native writer, every frame
export_file(pipe.compute(0), "POSCAR", format="vasp")       # anything else goes through ASE
export_file(pipe, "traj.extxyz", frames=range(0, 100, 10))
```

## Formats ewald does not read

A [reader plugin](../plugins/readers.md) adds a format without changing ewald: once it is
installed, `import_file`, the GUI's open dialog and the Claude tools all use it.
