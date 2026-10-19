# TODO

The one list of open work. What already exists is in the [feature inventory](features.md),
generated from the code; the milestones and decisions are in the [roadmap](ROADMAP.md). Every
pull request that adds or changes a feature regenerates the inventory and updates this page:
it removes the items it finishes, adds what it leaves open, and moves its own line out of
"In progress" when it merges.

## In progress

Nothing at the moment.

## Next

GUI

- Quad views (top, front, left, perspective), as in OVITO.
- Undo/redo for pipeline edits.
- Compute frames in a background thread so the GUI stays responsive.
- Scene updates without a full rebuild when only colors or a few properties change.

Analysis

- PLUMED: draw the bias forces as arrows, take MOLINFO residue names from the loaded topology,
  and plot `Plumed.*` attributes directly on the timeline with a current-frame marker.
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
- LAMMPS logs written with `thermo_style multi` or `thermo_modify line yaml` (only one-line
  thermo output is read now).

Figures

- GPU (CUDA) ray tracer.
- Ray tracer: shadows of the cell edges show as arcs on atoms.
- Vector arrows: their own color legend when a `ColorCoding` already owns the legend; arrows in
  the coarse (moving) viewport scene, as lines; in SVG/PDF, hide the part of an arrow inside its
  atom exactly (now the shaft is split at the atom and painted by depth).

Visualization plugins (later phases of the Scene IR plan)

- GUI: the built-in elements (Particles, Bonds, Cell, ...) as rows with forms generated from
  their `VisualElement` parameters, replacing the hand-written style controls.
- OpenGL and the ray tracer ignore `Cylinders.alpha`; see-through bonds need it.
- Ray tracer: trajectory lines and other `Lines` without a `radius` are not drawn.
- Phase 1: color map and theme registries, overlay plugins (`@overlay`), material presets.
- Phase 2: renderer backend registry, so a plugin can add a renderer that consumes the Scene IR.
- Phase 3 and later: glTF/HTML export from the IR, animated visuals, Claude-written visual
  elements (a Visual Forge, like Modifier Forge).

Plugins and Claude

- LLM plugin ideas: Paper-to-modifier (Forge from a paper's definition, checked against its
  numbers), Run doctor (diagnose a VASP/LAMMPS run from its outputs), phase identification,
  campaign report over many runs.

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
- P1: Linked selection, rest: picking atoms plots their property time series over all frames;
  scatter plots of two properties (brushing a box selects atoms); keep a pair, bond or angle
  range as a modifier (only property ranges become an `ExpressionSelection` now).
- P1: Phonon modes: read phonopy `band.yaml`/`mesh.yaml` and VASP IBRION=5-8 OUTCAR eigenvectors,
  animate a mode or draw it as arrows.
- P1: DOS and band structure plots (vasprun.xml, DOSCAR, EIGENVAL, PROCAR), with the PDOS of the
  picked atoms.
- P2: glTF, interactive HTML and Blender export.
- P2: Batch render CLI: apply one `.ewald` session to many run folders.
- P2: Native GSD reader (H5MD and XTC/TRR/DCD are done).
- P2: More HDF5 output: Quantum ESPRESSO `charge-density.hdf5` (G-space, needs an FFT to the
  real-space grid), ABINIT netCDF-4 `*_HIST.nc` trajectories and `*_DEN.nc` densities, VASP
  `vaspout.h5` DOS/bands for the DOS and band plots, `/results/partial_charges` (PARCHG).

## Verify outside the cloud container

- CUDA paths on a real NVIDIA GPU: CuPy kernels, neighbor lists, cluster analysis, the
  `--device cuda` GUI switch. CI and the cloud container have no GPU.
- SSH remote files against a real HPC login node (ProxyJump, OTP, large dumps); tests use an
  in-process SFTP server.
- Claude features with a real API key: `ewald ask` (also `--select`), the GUI Ask Claude tab,
  `ewald narrate`, `ewald forge`, `ewald mcp`.
- The Windows release workflow (wheels, PyInstaller app, installer) on a real `v*` tag.
- `DislocationAnalysis` and `OvitoModifier` with the `ovito` package installed.
- The native OUTCAR and vasprun.xml readers on real VASP 5/6 output (relaxations with ISIF=3,
  non-collinear and MD runs, multi-GB files); the test fixtures are hand-written in VASP 6.4
  format and checked against ASE.
- The vaspout.h5/vaspwave.h5 readers on real VASP 6.4/6.5 output (MD, relaxation, ISPIN=2,
  non-collinear, `LCHARGH5`); the fixtures follow py4vasp's schema and are checked against
  py4vasp, and the charge density is assumed to be stored as rho * V like a CHGCAR. The H5MD
  reader on GROMACS 2026 `mdrun` output.
- PLUMED from conda-forge on macOS and Windows (tested here with a 2.10 source build on Linux;
  CI has no PLUMED, so those tests skip).

## Infrastructure

- GitHub Actions minutes ran out on 2026-10-19; CI does not run until that is fixed, so pull
  requests are verified locally (`xvfb-run -a python -m pytest -q`, `mkdocs build --strict`).

## Deferred until the user asks

- Structure editing: supercells, slabs, defects. Do not start it without an explicit request.
