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
| VASP OUTCAR | native, lazy | every ionic step: `Force`, `Magnetic Moment` (collinear, or N x 3 non-collinear), energies, stress, pressure, magnetization, MD temperature (see [Simulation results](#simulation-results)); a partial last step is skipped with a warning |
| VASP OUTCAR of a phonon run (IBRION = 5-8) | native | Gamma-point phonon modes, animated by `PhononMode` (see [Phonon modes](phonons.md)) |
| phonopy `band.yaml`, `mesh.yaml`, `qpoints.yaml` | native | phonon modes with eigenvectors, band structure as the table "Phonon Bands" (see [Phonon modes](phonons.md)) |
| VASP vasprun.xml | native, lazy | every ionic step: cell, `Force`, energies, stress; a running job's open last step is skipped with a warning; total and projected DOS, bands and band projections as `data.attributes["Electronic Structure"]` and the tables "DOS" and "Bands" (see [DOS and band structure](electronic.md)) |
| VASP CHGCAR/LOCPOT/ELFCAR, Gaussian cube | native | atoms plus a grid in `data.grids` |
| VASP 6 `vaspout.h5` | native, lazy, needs `h5py` | every ionic step from `/intermediate/ion_dynamics` (the final structure when there is no history): cell with `scale`, `Force`, `Velocity` (Å/fs), `Magnetic Moment` (summed over orbitals, N x 3 non-collinear), energies, `Stress`, `Pressure`; the local potential (`WRT_POTENTIAL`/`LVTOT`) as `data.grids["potential"]` of the last frame; DOS, PDOS and bands (also KPOINTS_OPT) as for vasprun.xml |
| VASP 6 `vaspwave.h5` | native, needs `h5py` | `LCHARGH5` charge density as `data.grids["charge-density"]` (divided by the cell volume like a CHGCAR, e/Å³, plus `Magnetization`); the structure comes from `vaspout.h5` next to it or `structure=` |
| XYZ, extended XYZ (`*.xyz`, `*.extxyz`) | native, lazy | frames indexed by their atom-count lines and parsed on demand; the atom table goes through the same parallel numba parser as LAMMPS dumps and the rest through ASE's extxyz code, so the result is identical to `ase.io.read`: `Lattice=`, `Properties=`, `pbc=`, per-frame info (attributes), `forces` (`Force`), `energy` (`Energy`); a partial last frame is skipped with a warning |
| XCrySDen XSF/AXSF (`*.xsf`, `*.axsf`) | native, lazy | `CRYSTAL`/`SLAB`/`POLYMER`/`MOLECULE`, `ANIMSTEPS` trajectories with fixed or variable cell, forces (`Force`), every `DATAGRID_3D` as a property of `data.grids["xsf"]` (Quantum ESPRESSO `pp.x`, VESTA); the repeated end points of XSF's general grid are dropped for periodic axes |
| H5MD (`*.h5md`, `*.h5`) | native, lazy, needs `h5py` | LAMMPS `dump h5md`, MDAnalysis, ESPResSo; fixed or time-dependent box, `velocity`, `force`, `image`, `id`, `charge`, `mass`, scalar `/observables` as attributes; species names from string species, `/parameters/vmd_structure` or `type_map=`; `nm` converted to Å; `group=` picks a particle group |
| MDTraj / OpenMM HDF5 (`*.h5`) | native, lazy, needs `h5py` | OpenMM `HDF5Reporter`, `mdtraj.save_hdf5`: elements, residues (`Molecule Identifier`) and bonds from the JSON topology, cell from lengths and angles, `Velocity` (Å/fs), `Time` (ps), `Kinetic Energy`/`Potential Energy` (eV), `Temperature` |
| GROMACS XTC/TRR, DCD, AMBER NetCDF | MDAnalysis, lazy, needs `MDAnalysis` | atom types from `topology=` (`.gro`, `.pdb`, `.tpr`, `.psf`, LAMMPS `.data`, ...) or a file with the same stem, or `conf.gro`/`topol.tpr` next to it; type "X" without one; TRR velocities in Å/ps |
| ASE database (`*.db`, `.json`) | ASE, lazy | one row per frame; `selection="natoms<100,relaxed=1"` is an `ase db` query; key-value pairs, `Row Id`, `Unique Id` become attributes |
| ASE trajectory (`*.traj`) | ASE, lazy | written by ASE optimizers and MD |
| NEB runs (VASP folder `00 .. NN`, LAMMPS `neb`) | native | one frame per image, energies per image; see [NEB runs](#neb-runs) |
| CIF | ASE | every data block is a frame |
| others | ASE | anything `ase.io.read` understands, with `format=` set to the ASE name |

`ewald plugins` prints every reader and writer with its file-name patterns. The H5MD and
MDAnalysis formats need the `formats` extra: `pip install "ewald[formats]"` (h5py and
MDAnalysis).

### HDF5 files

Several `*.h5` formats share the extension; ewald looks inside the file to pick the reader:
`vaspout.h5`/`vaspwave.h5` by their VASP groups, MDTraj files by `/coordinates` and the
"Pande" convention, H5MD by its `/h5md` group. Other simulation codes that write HDF5 (or
netCDF-4, which is HDF5 underneath), and where they stand in ewald:

| Code | HDF5 output | In ewald |
|---|---|---|
| VASP 6 | `vaspout.h5` (structures, forces, energies, DOS, bands, ...), `vaspwave.h5` (charge, wavefunctions) | read (structures, energies, potential, charge) |
| LAMMPS, ESPResSo, MDAnalysis, pyiron | H5MD | read and written |
| GROMACS 2026 | H5MD from `gmx mdrun` (experimental) | the H5MD reader; not tried on GROMACS output yet |
| OpenMM, MDTraj | MDTraj HDF5 (`HDF5Reporter`) | read |
| Quantum ESPRESSO (built with HDF5) | `charge-density.hdf5`, `wfc*.hdf5` in `prefix.save` (G-space) | not yet |
| ABINIT | netCDF-4 `*_HIST.nc`, `*_DEN.nc`, `*_GSR.nc` | not yet |
| phonopy, phono3py | `band.hdf5`, `mesh.hdf5`, `kappa-*.hdf5` | not yet |
| BerkeleyGW, QMCPACK, FLEUR, PySCF (chkfile), DeePMD-kit/dpdata | HDF5 | not planned |

"Lazy" means the file is indexed once and each frame is read when it is needed, so a
multi-gigabyte trajectory opens immediately and uses memory for the frames in the cache only.

## Simulation results

Energies, temperature, pressure and the other per-step numbers a code writes become frame
attributes, matched to frames by `Timestep` = step, and the whole series is the table
`data.tables["Thermo"]`, which the GUI's Plots tab draws with a marker at the current frame.

| Source | Found how | Attributes |
|---|---|---|
| OUTCAR, vasprun.xml, vaspout.h5 | carried by the file | `Energy` (energy(sigma→0), eV), `Free Energy` (TOTEN), `Pressure` (external pressure, kB), `Stress` (xx yy zz yz xz xy, eV/Å³, ASE's sign), `Magnetization` (μB, OUTCAR); MD: `Temperature` (K), `Kinetic Energy`, `Total Energy` |
| OSZICAR | next to an XDATCAR | `Energy` (E0), `Free Energy` (F), `Magnetization` (mag, \|m\| when non-collinear); MD: `Temperature`, `Total Energy` (E), `Kinetic Energy` (EK), `Nose Potential` (SP), `Nose Kinetic` (SK) |
| log.lammps | next to a LAMMPS dump | every `thermo_style` column under its log name (`Temp`, `PotEng`, `Press`, `c_myT`, ...), in the run's units |

VASP frames from OUTCAR, vasprun.xml, vaspout.h5 and XDATCAR all count ionic steps from 1, as OSZICAR
does. A LAMMPS log may hold several runs with different `thermo_style` columns: they are
joined by `Step`, a column a run does not print is missing for its frames, and the step two
runs share keeps the values of both. Frames whose timestep the log does not print (dump
every 10 steps, thermo every 100) get the table but no attributes. Only one-line thermo
output is read (not `thermo_style multi` or YAML).

```python
pipe = import_file("XDATCAR")                          # OSZICAR in the same folder is picked up
pipe = import_file("dump.lammpstrj", thermo="../log.lammps")   # or a list of logs; thermo=None turns it off
data = pipe.compute(10)
data.attributes["Temp"], data.attributes["PotEng"]
pipe.source.thermo.table["Step"], pipe.source.thermo.table["Press"]   # the whole series (numpy arrays)

from ewald.io import read_lammps_log, read_oszicar     # the parsers alone, as {column: array}
```

OUTCAR and vasprun.xml are indexed by searching only for the lines that matter, so a large
OUTCAR opens about three times faster than with ASE and holds one ionic step in memory at a
time; over SSH the search runs as `grep` on the host. `format="vasp-out"` / `"vasp-xml"`
still use ASE's readers.

## NEB runs

A nudged elastic band run opens as a trajectory with one frame per image (`Timestep` = image
number), and the pipeline starts with `NebEnergyProfile`, which gives the energy profile and
barrier ([Modifiers](modifiers.md#nudged-elastic-band)), and `OverlayFrames`, which draws every
other image see-through behind the current one, so the moving atoms show their whole path.

| Layout | Open with | Each image | Energies |
|---|---|---|---|
| VASP (VTST) folder with `00`, `01`, ..., `NN` | `import_file("neb/")` | OUTCAR (last ionic step, with forces), else CONTCAR, else POSCAR | `Energy` from the OUTCAR; endpoints with only a POSCAR have none unless `energies=` gives them |
| LAMMPS `neb` run folder | `import_file("run/")` | last frame of the replica's dump (`dump ... dump.neb.$u`) | `PE1 .. PEN` of the last row of the master `log.lammps` NEB table |
| one file per image | `import_file("dump.neb.*", format="neb")` | last frame of each file (`frame=` picks another), natural order | from each file, or `log=` names the LAMMPS log |

```python
pipe = import_file("neb")                              # VASP NEB folder
data = pipe.compute(0)
data.attributes["NEB.Barrier"], data.attributes["NEB.Reverse Barrier"]   # eV, from the images
data.tables["NEB"]                                     # Reaction Coordinate, Energy, Tangent Force, Image, Frame
pipe = import_file("neb", energies=[-120.31, None, None, None, -120.12])  # endpoint energies by hand (None: from file)
```

In the GUI, File › Load NEB folder opens either folder layout and shows the profile in the
Plots tab. Remote (SSH) NEB folders are not read yet.

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

## Files on an SSH host

Any path can be on another machine, written like `scp` or as a URL (needs the `remote` extra):
`import_file("cluster:/scratch/me/run1/dump.lammpstrj")`, `"me@login.hpc.edu:run1/XDATCAR"`
(relative to the remote home) or `"ssh://me@login.hpc.edu:2222/scratch/me/dump.*.lammpstrj"`.
In the GUI, File › Load remote file (SSH)… browses the remote folders.

Connections follow `~/.ssh/config` (HostName, User, Port, IdentityFile, IdentitiesOnly,
ProxyJump, ProxyCommand, UserKnownHostsFile, StrictHostKeyChecking), the SSH agent and your
keys, and host keys are checked against `known_hosts`. If key login fails, ewald asks for the
password or one-time code and never saves it. LAMMPS dumps, LAMMPS data files, XDATCARs, OUTCARs and
vasprun.xml files are read in place, frame by frame (a dump is indexed with `grep -b` on the host), so a large
trajectory is not copied. Other formats, including plugin formats, are downloaded once into a
local cache (`ewald.io.remote.cache_dir()`, `EWALD_CACHE_DIR` overrides) and reused while the
remote file is unchanged. Host aliases need at least two characters, so `C:\data` stays local.

## Runs that are still writing

A LAMMPS dump, XDATCAR, OUTCAR or extxyz file can be opened while the simulation is still
writing it, locally or over SSH. `pipe.refresh()` looks for frames appended since the file was
opened (or last refreshed), reads only the new bytes (on an SSH host `tail -c | grep -b` runs
there), drops cached frames that may have changed and returns how many frames were added.
A half-written last frame is left out until it is complete, a file that shrank (a new run
overwrote it) is indexed again, and a wildcard such as `dump.*.lammpstrj` also picks up files
that appeared. Thermo tables next to the trajectory (log.lammps, OSZICAR) are read again too.
Compressed files are not followed.

```python
pipe = import_file("cluster:/scratch/me/run1/dump.lammpstrj")
...
if pipe.refresh():               # new frames since the last look
    data = pipe.compute(-1)      # the newest one
```

The GUI does this by itself every two seconds (Preferences › General › Live reload, and
File › Live reload for the current session): the timeline grows, and if it was on the last
frame it moves on to the newest one, as in OVITO and VMD.

## Formats ewald does not read

A [reader plugin](../plugins/readers.md) adds a format without changing ewald: once it is
installed, `import_file`, the GUI's open dialog and the Claude tools all use it. A
[writer plugin](../plugins/writers.md) does the same for `export_file` and File › Export.
