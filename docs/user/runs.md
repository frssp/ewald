# Runs: what they are, whether to trust them, and comparing several

A trajectory file alone does not say how it was made. The time between frames, the ensemble, the
target temperature and whether the run converged are in the input and log files next to it.
EWALD reads them, checks the run, and lets one notebook hold a whole series of runs.

## What a run is: `run_info`

```python
from ewald import load, run_info

pipe = load("md/T600/XDATCAR")
run_info(pipe)
# <RunInfo VASP md NVT, 600 K, 2 fs steps, 0.01 ps per frame>
```

For VASP it reads `INCAR` in the same folder (or, without one, the parameters VASP echoes at the
top of `OUTCAR` or in `vasprun.xml`) and `KPOINTS`. For LAMMPS it reads `log.lammps`, which echoes
every input command, or an input file (`in.*`, `*.in`) when there is no log. Files on an SSH host
are read in place, the same as the trajectory.

| Field | From |
|---|---|
| `kind` | `md`, `relax`, `static`, `neb` or `phonon`: VASP `IBRION`, `NSW`, `IMAGES`; LAMMPS `run`, `minimize`, `neb` |
| `ensemble` | VASP `MDALGO`, `SMASS`, `ISIF`, `ANDERSEN_PROB`; LAMMPS `fix nvt/npt/nph/nve` plus thermostat fixes (`langevin`, `temp/*`) |
| `temperature` | VASP `TEBEG`/`TEEND`; the LAMMPS thermostat's start and end |
| `timestep` (fs) | VASP `POTIM`; LAMMPS `timestep` in its `units` style (default for the style when not set) |
| `time_per_frame` (ps) | the steps between stored frames (XDATCAR: `NBLOCK`; dump: the frames' timesteps, else the `dump` interval) times the time step |

`notes` lists every value that was assumed rather than read (a default `POTIM`, parameters taken
from OUTCAR). LAMMPS `lj` units have no time in ps: `time_per_frame` stays `None` and
`parameters["time_per_frame_tau"]` gives it in reduced units.

`msd(pipe)` and `vacf(pipe)` use `time_per_frame` when you do not pass one, so the MSD is against
time in ps and the diffusion coefficient in Å²/ps without looking up `POTIM`. The VACF does this
only where velocities are in Å/ps (VASP, LAMMPS `metal`).

## Can I trust it: `check_run`

```python
from ewald import check_run

check_run(pipe)
# Run check: VASP md NVT, 600 K, 2 fs steps, 0.01 ps per frame, 200 frame(s)
#   [ok  ] SCF converged at every ionic step: at most 12 of NELM = 40 iterations (OSZICAR)
#   [ok  ] Equilibrated from step 120 (frame 23): Energy: 12% of the run discarded, 410 independent samples
#   [ok  ] Temperature at the target: mean 598.2 ± 29.7 K over the equilibrated part, target 600 K
#   [ok  ] Energy drift small: Total Energy: -0.0002 meV/atom/ps (limit 1)
#   [ok  ] No close contacts at frame 0: ...
```

What it checks:

- files a running or killed job left half written;
- ionic steps whose SCF loop hit `NELM` (VASP, counted in OSZICAR);
- a relaxation that never printed "reached required accuracy" (VASP OUTCAR);
- where an MD run is equilibrated: the start that maximizes the number of independent samples of
  the potential energy (Chodera, *J. Chem. Theory Comput.* 12, 1799 (2016)); a warning when that
  start is past the middle of the run. `check.equilibrated` is the frame, ready for
  `start=` of the trajectory analyses;
- the mean temperature against the thermostat target (NVT, NPT);
- energy drift where energy is conserved (VASP `E=`, which includes the thermostat; LAMMPS
  `TotEng` in NVE), in meV per atom per ps;
- LAMMPS `WARNING` and `ERROR` lines in the log;
- atoms closer than half the sum of their covalent radii in the first and last frame.

Each finding says what it rests on. `check.ok` is False when any is a warning or an error,
`check.to_dict()` is the same as JSON, and `ewald check FILE...` prints it on the command line
(exit status 1 when something is wrong). The AI chat has the same check as its `check_run` tool.

In the GUI, opening an MD run or a relaxation whose input files say so adds a `check_run(pipe)`
cell under the load cell, run at once. Insert › Analysis › Run check adds one to any notebook.

## Several runs: `load_runs`

```python
from ewald import load_runs, msd, rdf, arrhenius
from ewald.modifiers import WrapAtoms

runs = load_runs("T*/XDATCAR", label=r"T(\d+)")   # <Runs 3: 300, 600, 900>
pipe = runs.pipe          # the run the 3D view shows

runs.add(WrapAtoms())     # a step on every run (a copy each)
g = rdf(runs, cutoff=5)   # one g(r) table per run, shown overlaid
d = msd(runs)             # time per frame from each run's INCAR
arrhenius(d)              # <Arrhenius Ea = 0.301 ± 0.004 eV, ...>
```

`load_runs` takes a wildcard (also on an SSH host) or a list of files. `label` is a regular
expression whose first group names each run, a function of the path, or a list; by default it is
the folder name that differs between the files. When every label holds a number the runs are
sorted by it. `runs["600"]` or `runs[1]` is one `Run` with its `label`, `value`, `path` and
`pipe`, an ordinary pipeline.

Every analysis verb (`rdf`, `msd`, `vacf`, `sq`, `xrd`, `cna`, ...) takes `runs` and returns a
`RunTables`: one table per run, drawn overlaid under the cell (`.plot(y=...)` for another column,
`.figure()` for the matplotlib figure, `.attribute(name)` for a global result per run).
`runs.attribute(name)` tabulates any attribute over the runs, `runs.check()` checks each one, and
`runs.temperatures()` gives each run's thermostat target (its label's number when the input does
not say).

`arrhenius(d)` fits `D = A exp(-Ea/kT)` to the diffusion coefficients of `msd(runs)` against each
run's temperature and shows ln D against 1000/T with the line; `arrhenius([300, 600], [d1, d2])`
fits any values you give it.

In the GUI, a notebook that holds a `Runs` shows a run picker above its cells. Picking a run binds
`pipe` to it, so the 3D view, the timeline and later `pipe.add(...)` cells follow that run;
`runs.add(...)` and verbs called on `runs` still cover all of them.
