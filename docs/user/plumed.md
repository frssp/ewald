# PLUMED collective variables

ewald runs [PLUMED](https://www.plumed.org) inputs on its trajectories, so every collective
variable PLUMED knows (distances, torsions, coordination numbers, RMSD, path CVs, METAD biases,
...) becomes a per-frame attribute you can plot, select on and hand to Claude. It also reads
PLUMED's own COLVAR and HILLS files and draws the metadynamics free energy surface.

This is the **ewald-plumed plugin** (id `plumed`), not part of ewald itself: install it from the
Plugins panel or with

```bash
pip install 'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-plumed'
```

Its modifiers then import from `ewald.modifiers` like built-in ones, its library is
`ewald_plumed.plumed` (`import ewald.plumed` still works), and Claude gets its two tools.

```python
from ewald import import_file
from ewald.modifiers import Plumed, TimeSeries

pipe = import_file("dump.lammpstrj")
pipe.modifiers.append(Plumed("""
d: DISTANCE ATOMS=1,2
phi: TORSION ATOMS=5,7,9,15
cn: COORDINATIONNUMBER SPECIES=1-100 SWITCH={RATIONAL R_0=0.3}
""", time_per_frame=0.1))
data = pipe.compute(10)
data.attributes["Plumed.d"]        # nm, as PLUMED reports it
data["Plumed cn"]                  # one value per atom
data.tables["plumed"]              # Frame, d, phi, Bias over the trajectory
```

## Installing PLUMED

`LoadColvar`, `FreeEnergySurface` and `read_plumed_output` need only the plugin. `Plumed` and
`compute_plumed_cvs` also need the PLUMED library (the "kernel") and its Python binding, and
fail with a message saying what to install when they are missing.

| Route | Command |
|---|---|
| conda-forge (library and binding, sets `PLUMED_KERNEL`) | `conda install -c conda-forge plumed py-plumed` |
| pip binding + your own PLUMED build | `pip install plumed` (or the plugin with its `[engine]` extra), then `export PLUMED_KERNEL=<prefix>/lib/libplumedKernel.so` |

The binding finds the library through `PLUMED_KERNEL`; `Plumed(kernel="/path/libplumedKernel.so")`
overrides it per modifier. Values of vector CVs (one per atom) need PLUMED 2.10 or later.
`ewald_plumed.plumed.available()` tells you whether everything loads.

## How frames reach PLUMED

`Plumed` talks to PLUMED through its MD-engine interface, the way LAMMPS or GROMACS do, one
frame per MD step:

- positions and cell go in Angstrom and PLUMED converts them to nm (`setMDLengthUnits 0.1`),
  so CVs come back in PLUMED units: nm, rad, kJ/mol, ps;
- masses come from the 'Mass' property or the element (a numeric LAMMPS type without a mass
  gets 1), charges from 'Charge' (0 otherwise), so `COM`, `CENTER` and charge-based CVs work;
- PLUMED atom `i` (1-based, `ATOMS=1,2`) is the particle whose 'Particle Identifier' is `i`,
  the LAMMPS numbering, whatever order the dump stores them in; without identifiers, or with
  `by_identifier=False`, it is the i-th particle;
- frame `k` is MD step `k` and `time_per_frame` ps after the previous one, so `STRIDE` and
  `PACE` count frames, exactly as `plumed driver` reads a trajectory; `temperature` (K) sets
  kBT for METAD and other biases.

The test suite checks this against `plumed driver` on the same frames: distances with and
without PBC in a triclinic cell, components, torsions, coordinations, centers of mass and METAD
biases agree to the last digit.

PLUMED is stateful (METAD deposits hills, averages accumulate), so `Plumed` runs once over
frames `start:stop:step` the first time any frame is computed, and caches the result until a
parameter, an upstream modifier or the input file changes.

## Inputs and output files

`input` is the text of a `plumed.dat` or its path. In the GUI the parameter is a text box; a
one-line input may separate actions with `;`. Relative files the input mentions
(`REFERENCE=ref.pdb`, `INCLUDE FILE=cvs.dat`, `MOLINFO STRUCTURE=`, a HILLS file to restart
from) are read from the `plumed.dat`'s folder, or the current folder for text input.

PLUMED runs in a scratch folder, so the `PRINT` and `METAD` lines of a production input never
overwrite the COLVAR and HILLS next to it. To keep what PLUMED writes, give
`workdir="analysis/"`.

Outputs:

| Output | What |
|---|---|
| attribute `Plumed.<name>` | every value of every label, components included (`Plumed.d.x`, `Plumed.metad.bias`) |
| attribute `Plumed.bias` | total bias at that frame |
| property `Plumed <label>` | a label with one value per atom (e.g. `COORDINATIONNUMBER`) |
| `tables["plumed"]` | `Frame`, every value and `Bias` over the analyzed frames (Plots tab) |

Labels without values (`GROUP`, `MOLINFO`) and virtual atoms used only as positions are
handled: virtual atoms give `.x .y .z .mass .charge` components.

## Existing COLVAR files

`LoadColvar("COLVAR")` attaches a COLVAR (any file with a `#! FIELDS` header) to the frames
as attributes `Colvar.<field>`, and puts the whole file in `tables["colvar"]`.

- `align="index"` (default): frame `k` takes row `offset + k * stride`; `stride` is the number
  of COLVAR rows per stored frame (dump every 1000 steps, COLVAR every 100: `stride=10`).
- `align="time"`: frame `k` takes the row whose `time` is nearest to the frame's time, which is
  `Timestep * md_timestep` when the frame has a LAMMPS 'Timestep' and `md_timestep` (ps) is given,
  else `k * time_per_frame`. Frames with no row within half a COLVAR interval get NaN.

Restarted runs, whose COLVAR repeats the header, read as one file; a field that appears later
is NaN before it.

## Free energy from HILLS

`FreeEnergySurface("HILLS", bins=100)` sums the metadynamics hills on a grid,
F(s) = -sum_i G_i(s), the way `plumed sum_hills` does: PLUMED's stretched-Gaussian kernels,
periodic CVs from the `#! SET min_/max_` lines, and well-tempered heights (which PLUMED already
writes rescaled by gamma/(gamma-1)). The tests compare it with `plumed sum_hills` output for
one and two CVs. The result, minimum shifted to 0, is `tables["fes"]`: the Plots tab draws one
CV as a curve and two CVs as a filled contour map.

- `ranges=[(0.0, 0.6), None]` fixes the grid range per CV (None: the periodic domain, or the
  hills +- 3 sigma).
- `up_to_frame=True` uses only the hills deposited up to the current frame's time
  (`frame * time_per_frame` against HILLS `time`), so stepping through the trajectory shows the
  FES converge.
- `use_plumed=True` runs `plumed sum_hills` itself, for more than two CVs or multivariate
  hills.

From Python:

```python
from ewald_plumed import plumed

fes = plumed.fes_from_hills("HILLS", bins=(60, 60))
plumed.plot_fes(fes).figure.savefig("fes.pdf")
history = plumed.fes_from_hills("HILLS", bins=60, stride=500)["history"]   # FES every 500 hills
```

`plumed.read_colvar`, `read_hills`, `read_fes` (a `sum_hills` grid) and `sum_hills` are there
too, and `plumed.run(input, frames)` evaluates an input on any iterable of frames without a
pipeline.

## With Claude

The AI tools `compute_plumed_cvs` (runs an input on the loaded trajectory and keeps the
`Plumed` step in the notebook, written and run like `add_modifier`, so `get_script` reproduces it) and `read_plumed_output`
(COLVAR statistics, or the FES of a HILLS file with its minimum and a plot) let you ask, for
example, "compute the O-H coordination of the first water and tell me when it changes".
`Plumed.*` attributes then work in expressions and the other tools like any attribute.

## Examples on published data

Two PLUMED-NEST eggs are in [Reproducing papers](papers.md): `ammonia_borane_fes.py` sums the
hills of four walkers with `fes_from_hills` and compares them with the authors' `sum_hills`
output, and `alanine_dipeptide_fes.py` draws the phi/psi free energy surface.
