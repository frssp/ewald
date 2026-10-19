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
| VASP CHGCAR/LOCPOT/ELFCAR, Gaussian cube | native | atoms plus a grid in `data.grids` |
| XYZ, extended XYZ (`*.xyz`, `*.extxyz`) | native, lazy | frames indexed by their atom-count lines, each parsed on demand by ASE's extxyz parser: `Lattice=`, `Properties=`, `pbc=`, per-frame info (attributes), `forces` (`Force`), `energy` (`Energy`); a partial last frame is skipped with a warning |
| XCrySDen XSF/AXSF (`*.xsf`, `*.axsf`) | native, lazy | `CRYSTAL`/`SLAB`/`POLYMER`/`MOLECULE`, `ANIMSTEPS` trajectories with fixed or variable cell, forces (`Force`), every `DATAGRID_3D` as a property of `data.grids["xsf"]` (Quantum ESPRESSO `pp.x`, VESTA); the repeated end points of XSF's general grid are dropped for periodic axes |
| H5MD (`*.h5md`, `*.h5`) | native, lazy, needs `h5py` | LAMMPS `dump h5md`, MDAnalysis, ESPResSo; fixed or time-dependent box, `velocity`, `force`, `image`, `id`, `charge`, `mass`, scalar `/observables` as attributes; species names from string species, `/parameters/vmd_structure` or `type_map=`; `nm` converted to Å; `group=` picks a particle group |
| GROMACS XTC/TRR, DCD, AMBER NetCDF | MDAnalysis, lazy, needs `MDAnalysis` | atom types from `topology=` (`.gro`, `.pdb`, `.tpr`, `.psf`, LAMMPS `.data`, ...) or a file with the same stem, or `conf.gro`/`topol.tpr` next to it; type "X" without one; TRR velocities in Å/ps |
| ASE database (`*.db`, `.json`) | ASE, lazy | one row per frame; `selection="natoms<100,relaxed=1"` is an `ase db` query; key-value pairs, `Row Id`, `Unique Id` become attributes |
| ASE trajectory (`*.traj`) | ASE, lazy | written by ASE optimizers and MD |
| CIF | ASE | every data block is a frame |
| others | ASE | anything `ase.io.read` understands, with `format=` set to the ASE name |

`ewald plugins` prints every reader and writer with its file-name patterns. The H5MD and
MDAnalysis formats need the `formats` extra: `pip install "ewald[formats]"` (h5py and
MDAnalysis).

"Lazy" means the file is indexed once and each frame is read when it is needed, so a
multi-gigabyte trajectory opens immediately and uses memory for the frames in the cache only.

## Reader options

Keyword arguments to `import_file` go to the reader:

```python
import_file("dump.lammpstrj", type_map={1: "Cu", 2: "Zr"})     # LAMMPS numeric types to elements
import_file("system.data", atom_style="full")                  # data file without an Atoms # style hint
import_file("structure.data", format="lammps-data-ase")        # use ASE's LAMMPS data reader instead
import_file("md.xtc", topology="conf.gro")                     # GROMACS: atom types from the topology
import_file("lammps.h5", type_map={1: "Cu", 2: "Zr"})          # H5MD with numeric species
import_file("train.db", selection="energy<-3")                 # rows of an ASE database
import_file("run.xyz", format="ase")                           # ASE's own reader instead of the native one
```

LAMMPS values stay in the file's unit style; the dump's `Units` attribute is kept when the
file has one.

## Property names

Readers name properties as OVITO does: `Position`, `Particle Type`, `Particle Identifier`,
`Velocity`, `Force`, `Charge`, `Mass`, and so on. `data.summary()` lists what a file produced.

## Writing files

`export_file(source, path)` writes a `DataCollection`, a list of them or a pipeline's frames
(after its modifiers). The writer follows the file name, or `format=` names one (or an ASE
format). Pipeline frames are computed and written one at a time, so exporting a long
trajectory to extxyz, H5MD or XTC never holds it in memory.

```python
from ewald.io import export_file

export_file(pipe, "traj.extxyz", frames=range(0, 100, 10))   # every 10th frame
export_file(pipe, "md.h5md")                                 # H5MD, streamed
export_file(pipe, "md.xtc")                                  # GROMACS XTC (MDAnalysis)
export_file(pipe, "CONTCAR", frames=[-1])                    # last frame as a POSCAR
export_file(pipe.compute(0), "cu.data", atom_style="atomic") # LAMMPS data file
export_file(pipe, "out.lammpstrj", columns=("id", "type", "x", "y", "z"))
export_file(data, "rho.xsf")                                 # structure and its grid
```

| Writer | File names | Frames | Notes |
|---|---|---|---|
| `poscar` (`vasp`) | `POSCAR*`, `CONTCAR*`, `*.vasp` | one | species grouped in order of first appearance, `Direct`; `sort=False` keeps the atom order |
| `xdatcar` | `XDATCAR*` | many | through ASE |
| `lammps-data` | `*.data`, `*.lmp` | one | `atom_style` `atomic`, `charge` (default when there is a `Charge`), `molecular`, `full`; type names as `Masses` comments, so ewald and OVITO read them back; `Velocities`; a cell not in LAMMPS' lower-triangular form is rotated rigidly |
| `lammps-dump` | `*.lammpstrj`, `*.dump` | many | `columns=` |
| `extxyz` | `*.extxyz`, `*.xyz` | many | every numeric property, `Force` as `forces`, `Energy` and scalar attributes in the comment line; `columns=` limits the properties |
| `cif` | `*.cif` | many | through ASE |
| `xsf` | `*.xsf`, `*.axsf` | many | `ANIMSTEPS` for a trajectory; the first frame's grid as `DATAGRID_3D` (`grid=`, `property=`) |
| `h5md` | `*.h5md`, `*.h5` | many | position, species with names, box, other numeric properties and scalar attributes as observables; `group=` (default `all`, use `trajectory` for MDAnalysis), `compression=` |
| `md-trajectory` (`xtc`, `trr`, `dcd`) | `*.xtc`, `*.trr`, `*.dcd` | many | lower-triangular box (rotated rigidly when needed); TRR also velocities and forces |
| `ase-db` | `*.db` | many | one row per frame, scalar attributes as key-value pairs |
| `ase-traj` | `*.traj` | many | with `Energy` and `Force` |
| `cube`, `vasp-volumetric` (`chgcar`) | `*.cube`, `CHGCAR*`, `LOCPOT*` | one | one grid property (`property=`, `grid=`) |
| `ase` | any other `format=` | many | `ase.io.write` |

One-frame formats raise an error when given several frames; pass `frames=[i]`. The GUI's
File › Export data lists the same writers and asks whether to write the current frame or the
whole trajectory. A [writer plugin](../plugins/writers.md) adds a format.

## Formats ewald does not read

A [reader plugin](../plugins/readers.md) adds a format without changing ewald: once it is
installed, `import_file`, the GUI's open dialog and the Claude tools all use it. A
[writer plugin](../plugins/writers.md) does the same for `export_file` and File › Export.
