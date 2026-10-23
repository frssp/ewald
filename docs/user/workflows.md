# Workflows: a folder of calculations as one notebook

A study usually lives in one folder: the end points relaxed in `01_relax_IS/` and `01_relax_FS/`,
the band in `02_neb/` (`00 .. NN`), an MD restarted in `03_md/run1`, `run2`, phonons in
`04_phonon/`. EWALD reads such a folder as one **workflow** and opens it as one notebook, with a
`load` line per step and `pipe` on the last result. Nothing needs to be set up and nothing in
the folder is run; files are only listed and their heads or tails read.

## Opening one

- **Desktop app:** File › Open workflow folder…, drop the folder on the window, or the
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
| `vasprun.xml`, `OUTCAR`, `XDATCAR`, `OSZICAR`, or `INCAR` + `POSCAR` | relax, md, static, phonon or neb from the INCAR; `energy` and `ionic steps` from OSZICAR | XDATCAR for MD, else vasprun.xml, OUTCAR, XDATCAR, CONTCAR, POSCAR |
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

## For the AI

The AI tools `list_workflows` (source `folders`: the folders opened lately) and
`describe_workflow` (any folder path, local or `host:/path`) read the same tree, with each step's
path, state and numbers, so "open the NEB of this study and plot the barrier" needs no file
names. Other workflow sources (AiiDA, sim-agent) come from plugins
([Workflow sources](../plugins/workflows.md)).

Plugins can also offer **analyses** of a finished workflow: the notebook a sim-agent
`vasp_band_offset` workflow opens ends with a "Potential lineup" section that reads the three
statics' planar-averaged potentials and band edges and puts the potential, the macroscopic
average, VBM, CBM and the interface Fermi level on one energy axis, with the band offset or
Schottky barrier in the result. `describe_workflow` lists the analyses, with the reason when one
cannot run (a missing or failed step), and `add_analysis` adds one to the notebook and runs it,
so asking the AI to "draw the potential with VBM, CBM and E_F" runs these cells instead of the
AI aligning energies itself.
