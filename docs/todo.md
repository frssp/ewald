# TODO

The one list of open work. What already exists is in the [feature inventory](features.md),
generated from the code; the milestones and decisions are in the [roadmap](ROADMAP.md). Every
pull request that adds or changes a feature regenerates the inventory and updates this page:
it removes the items it finishes, adds what it leaves open, and moves its own line out of
"In progress" when it merges.

## In progress

- Per-atom vectors drawn as arrows (force, displacement, velocity, magnetic moment) in the
  viewport, ray tracer and vector renderer.
- VASP/LAMMPS output readers: energies, forces, stress and magnetic moments from vasprun.xml,
  OUTCAR and OSZICAR, thermo columns from log.lammps, as frame attributes and per-atom properties.
- PLUMED plugin and collective variables.
- Ask Claude about picked atoms ([#26](https://github.com/frssp/ewald/pull/26)).

## Next

GUI

- Quad views (top, front, left, perspective), as in OVITO.
- Undo/redo for pipeline edits.
- Compute frames in a background thread so the GUI stays responsive.
- Scene updates without a full rebuild when only colors or a few properties change.

Analysis

- ConstructSurfaceMesh (alpha-shape surface), drawn with the existing `SurfaceMesh`.
- FindRings keeps every shortest-path ring; OVITO drops some near-maximum-size rings that differ
  from a same-size ring by a smaller one (closed-source, order-dependent). Decide whether to
  offer an OVITO-compatible mode once its rule is known.
- FindRings in numba for networks above ~100k atoms (8000 a-Si atoms, rings up to 10: 1 s).
- Spatial correlation function and Voronoi-weighted Steinhardt (OVITO / freud have them).

Volumetric data

- Section caps where an isosurface meets the cell faces.
- Color an isosurface by a second grid (e.g. electrostatic potential on the density).
- BXSF (Fermi surface) reader.

Formats

- GROMACS `.gro` with velocities.
- LAMMPS binary dump.

Figures

- GPU (CUDA) ray tracer.
- Ray tracer: shadows of the cell edges show as arcs on atoms.

Plugins and Claude

- LLM plugin ideas: Run doctor (diagnose a VASP/LAMMPS run from its outputs), phase
  identification, campaign report over many runs.

Docs

- Host the manual (the repository is private; CI uploads the built site as the `docs-site`
  artifact).

## Ideas backlog

From the feature-ideas review (2026-10-19), after the four in progress above.

- P0: Symmetry and diffraction: space group, Wyckoff positions and symmetry-equivalent atoms
  via spglib (optional extra) shown in the inspector; hkl planes in the viewport; powder XRD and
  neutron patterns and S(q) for MD, overlaid on an experimental pattern.
- P1: NEB viewer: open `00/ ... 0N/` image folders as one trajectory, energy along the reaction
  coordinate synced with the timeline, saddle point marked.
- P1: Jupyter widget: `pipe.show()` with a rotatable viewport and frame slider in a notebook
  (PyVista trame backend).
- P1: Linked selection: brushing a histogram or scatter plot selects atoms in the viewport, and
  picking atoms plots their property time series; the selection is stored as an
  `ExpressionSelection`.
- P1: Phonon modes: read phonopy `band.yaml`/`mesh.yaml` and VASP IBRION=5-8 OUTCAR eigenvectors,
  animate a mode or draw it as arrows.
- P1: DOS and band structure plots (vasprun.xml, DOSCAR, EIGENVAL, PROCAR), with the PDOS of the
  picked atoms.
- P2: glTF, interactive HTML and Blender export.
- P2: Batch render CLI: apply one `.ewald` session to many run folders.
- P2: Native GSD reader (H5MD and XTC/TRR/DCD are done).

## Verify outside the cloud container

- CUDA paths on a real NVIDIA GPU: CuPy kernels, neighbor lists, cluster analysis, the
  `--device cuda` GUI switch. CI and the cloud container have no GPU.
- SSH remote files against a real HPC login node (ProxyJump, OTP, large dumps); tests use an
  in-process SFTP server.
- Claude features with a real API key: `ewald ask`, `ewald narrate`, `ewald forge`, `ewald mcp`.
- The Windows release workflow (wheels, PyInstaller app, installer) on a real `v*` tag.
- `DislocationAnalysis` and `OvitoModifier` with the `ovito` package installed.

## Infrastructure

- GitHub Actions minutes ran out on 2026-10-19; CI does not run until that is fixed, so pull
  requests are verified locally (`xvfb-run -a python -m pytest -q`, `mkdocs build --strict`).

## Deferred until the user asks

- Structure editing: supercells, slabs, defects. Do not start it without an explicit request.
