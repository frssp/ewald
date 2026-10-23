# Workflows: a folder of calculations as one notebook

A study usually lives in one folder: the end points relaxed in `01_relax_IS/` and `01_relax_FS/`,
the band in `02_neb/` (`00 .. NN`), an MD restarted in `03_md/run1`, `run2`, phonons in
`04_phonon/`. EWALD reads such a folder as one **workflow** and opens it as one notebook, with a
`load` line per step and `pipe` on the last result. Nothing needs to be set up and nothing in
the folder is run; files are only listed and their heads or tails read.

## Opening one

- **Desktop app:** File › Open from › Workflow folder…, drop the folder on the window, or the
  **Workflows** side bar (activity bar): Open folder…, or a folder path typed in its search box.
  The side bar lists the folders opened lately; expanding one shows its steps, their state
  (✓ finished, … still running) and their outputs (CONTCAR, CHGCAR, NEB band, ...). A double
  click opens a step, or the whole workflow as one notebook.
- **Notebook or script:**

```python
from ewald import workflow

wf = workflow.describe("~/calc/li_diffusion")      # or "hpc:/scratch/me/li_diffusion" over SSH
print(workflow.text(wf))
# relax → neb → md  <...>  li_diffusion  -> .../03_md/run*/XDATCAR
#   01_relax  ✓  -> .../01_relax/vasprun.xml
#     - CONTCAR  (final structure)  -> .../01_relax/CONTCAR
#   02_neb  ✓  -> .../02_neb
#     - band  (7 images)  -> .../02_neb
#   03_md  (2 runs)  ✓  -> .../03_md/run*/XDATCAR
print(workflow.notebook_text(wf))                   # the notebook it opens as
```

The notebook it opens holds one cell that loads every step:

```python
relax = load(".../01_relax/vasprun.xml")
neb = load(".../02_neb")
md = load(".../03_md/run*/XDATCAR")
pipe = md
```

so `pipe = neb` or `show(neb, ...)` brings any other step into the 3D view. Folder names lose a
leading step number (`02_neb` becomes `neb`).

## What is recognized

| Folder | Step | Opens |
|---|---|---|
| VASP image folders `00 .. NN`, or a LAMMPS NEB run (`log.lammps` + one dump per replica) | NEB (`barrier` from OSZICARs / the NEB log) | the whole band, one frame per image |
| `phonopy_disp.yaml`, `phonopy.yaml`, `phonopy_params.yaml`, `FORCE_SETS`, `band.yaml` | phonon (`disp-001 ...` counted, not listed) | `band.yaml`, else phonopy's yaml (needs the phonopy plugin) |
| `vasprun.xml`, `OUTCAR`, `XDATCAR`, `OSZICAR`, or `INCAR` + `POSCAR` | relax, md, static, phonon or neb from the INCAR; `energy` and `ionic steps` from OSZICAR, `atoms` (CONTCAR/POSCAR) and `energy/atom`, `max force` from the OUTCAR's last force block, `gap` from EIGENVAL | XDATCAR for MD, else vasprun.xml, OUTCAR, XDATCAR, CONTCAR, POSCAR |
| LAMMPS dump files (`ITEM: TIMESTEP`) | the kind from `log.lammps` | the first dump; the others as outputs |
| sub-folders that are all calculations named `run1, run2`, `restart_1`, `1, 2`, `r0, r1`, ... | one step with **runs** = the number of segments | all segments joined: `md/run*/XDATCAR` |
| any other folder with calculations below it (up to three levels) | a group of steps | |

Steps are in natural order of their folder names, so `run10` comes after `run9` (wildcards given
to `load` are joined in the same order). Hidden folders are skipped. A finished VASP step has
`General timing` at the end of its OUTCAR, a finished LAMMPS one `Total wall time` in its log;
others show as still running.

Restarted segments are joined the way `load` joins several files (`join="timestep"`): LAMMPS
dumps that overlap after a restart from an earlier step keep the later segment's frames; VASP
segments, which carry no timestep, are put one after the other.

## The numbers of every step, and several workflows side by side

The notebook a workflow opens as starts with a table of its steps and their numbers. The same
table, live, and for several workflows at once (another functional, a bigger cell, a different
dopant), comes from `workflow_table`:

```python
from ewald import workflow_table

workflow_table("~/calc/li_pbe")                          # a row per step, a column per number
t = workflow_table("~/calc/li_pbe", "~/calc/li_r2scan", labels=["PBE", "r2SCAN"])
t                                                        # each number gets a column per workflow
t.difference("PBE")                                      # r2SCAN − PBE, step by step
t["energy/atom"]                                         # {"PBE": array, "r2SCAN": array}
t.plot("gap")                                            # the steps on x, a line per workflow
t.to_csv("compare.csv")                                 # pandas.read_csv reads it back
```

| step | state PBE | state r2SCAN | energy/atom (eV) PBE | energy/atom (eV) r2SCAN | gap (eV) PBE | gap (eV) r2SCAN |
|---|---|---|---|---|---|---|
| 01_relax | ✓ | ✓ | -5.4250 | -6.0120 | | |
| 02_static | ✓ | ✓ | -5.4310 | -6.0170 | 1.0120 | 1.3840 |

(made-up numbers; in a notebook the heads of one number span its workflows)

Rows are the steps that ran a calculation: a group of steps adds its name in front
(`bands / scf`), a restarted MD is one row. Steps of different workflows are matched by name,
ignoring case and a leading step number (`01_relax` and `1-Relax` are one row); folders named
differently line up by order with `align="position"`. `columns=["energy/atom", "gap"]` picks the
numbers. Any source works: `workflow_table("aiida:1234", "aiida:1240")` compares AiiDA workflows
with the numbers their source reads.

In the **Workflows** side bar, select two or more workflows (Ctrl or Shift click) and pick
**Compare 2 workflows in a table** from the right-click menu (one workflow: **Table of the steps'
numbers**): a cell `workflow_table(...)` is added to the
notebook on screen and run.

## For the AI

The AI tools `list_workflows` (source `folders`: the folders opened lately) and
`describe_workflow` (any folder path, local or `host:/path`) read the same tree, with each step's
path, state and numbers, so "open the NEB of this study and plot the barrier" needs no file
names. `list_workflows` also filters in code, for every source: `formula` (exactly those
elements in any ratio, so `TiN` finds the Hill formula `N4Ti4` and a TiN slab), `elements`,
`kind` (template), `status` (succeeded, failed, running) and `since`/`until` days (`today`,
`yesterday`, `3d`, `2026-10-22`; the day a workflow was created), and it returns today's date,
so "open yesterday's finished TiN calculation" does not depend on the model guessing.
`open_workflow` then loads the workflow, a step (`bands`, `bands/scf`) or an output of it
(`CHGCAR`) as the pipeline, and refuses a name that is not there with the ones that are.
Other workflow sources (AiiDA, sim-agent) come from plugins
([Workflow sources](../plugins/workflows.md)).

Plugins can also offer **analyses** of a finished workflow: the notebook a sim-agent
`vasp_band_offset` workflow opens ends with a "Potential lineup" section that reads the three
statics' planar-averaged potentials and band edges and puts the potential, the macroscopic
average, VBM, CBM and the interface Fermi level on one energy axis, with the band offset or
Schottky barrier in the result. `describe_workflow` lists the analyses, with the reason when one
cannot run (a missing or failed step), and `add_analysis` adds one to the notebook and runs it,
so asking the AI to "draw the potential with VBM, CBM and E_F" runs these cells instead of the
AI aligning energies itself.
