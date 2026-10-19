# TODO

The one list of open work. What already exists is in the [feature inventory](features.md),
generated from the code; the milestones and decisions are in the [roadmap](ROADMAP.md). Every
pull request that adds or changes a feature regenerates the inventory and updates this page:
it removes the items it finishes, adds what it leaves open, and moves its own line out of
"In progress" when it merges.

## In progress


## Next

GUI

- Notebook layout: the viewport could follow the current notebook even when it has no
  pipeline yet (now it keeps the last one); 3D snapshot views (`show(pipe, frame=...)` drawing
  the structure) and image/figure cells; drag cells to reorder; undo for cell edits; inline
  Ask Claude suggestions as diffs; per-notebook render settings; a Jupyter display hook so
  `pipe` shows a rendered image in Jupyter; a `.py` header with the view state and data links;
  File › Trust notebook and a restricted banner in the notebook panel; pip requirements of
  package plugins an `.ewd` names (now only recorded).
- Notebook kernels: run a notebook's cells in another Python environment (a per-tab kernel via
  jupyter_client with a small `ewald-kernel` that streams computed frames back), so a notebook
  can use a conda env without ewald's GUI dependencies.
- Analysis verbs for notebooks (`rdf(pipe, cutoff=6)`, `msd(pipe)`, ... returning tables and
  adding the step), next to `load` and `pipe.add`.
- Plugins side bar: an "updates available" badge on the activity bar, disable per notebook,
  "Recommended" plugins required by the open notebook, a Contributions tab and a full Forge
  view instead of the Ask Claude prompt; uninstalling a pip plugin still needs a restart.
- Spectra, Figure and Forge views from the redesign mock-ups, built as view cells
  rather than separate panels.
- Quad views (top, front, left, perspective), as in OVITO.
- Undo/redo for pipeline edits.
- Compute frames in a background thread so the GUI stays responsive.
- Scene updates without a full rebuild when only colors or a few properties change, and frame
  changes with the same atoms that replace only the point coordinates.
- Picking without a render: click picks still use a `vtkPropPicker` selection render; the hover
  readout already does a CPU ray-sphere test (`camera_ray`, `ray_spheres` in `gui/viewport.py`)
  that `pick_at` could reuse.
- App icon outside the running window: a Linux `.desktop` entry and hicolor icons installed
  with the package, the Windows installer/shortcut (`ewald.ico`) and a macOS app bundle
  (`ewald.icns`); the EWALD mark in the main window header (the Notebook redesign) and the docs.
- Icons for the remaining menu bar entries (View, Pipeline, Tools) and the Add modification
  categories; the view-bar rotation arrows are still text glyphs.
- Context-dependent menus beyond the viewport right click: a "Suggested" group at the top of
  Add modification for the loaded data, view-along buttons hidden without a cell, the timeline
  folded for a single structure, a right-click menu on pipeline items.
- Shortcuts for the viewport tools (orbit, pick, distance, angle, bond) and view-along axes.
- Impostor spheres: exact ray-sphere depth in perspective views (now approximate near the
  screen edges) and VTK scene lights instead of the fixed head light; bond impostors.
- Flat (ASE) style in the ray tracer and the vector renderer (now OpenGL only), and a
  per-element bond table like VESTA's (cation-anion only, measured lengths) instead of the
  covalent-radius sum with metal-metal pairs left out.
- Boundary (VESTA): repeat isosurfaces and grid slices over the boundary range too (now only
  atoms, bonds and coordination polyhedra are repeated), VESTA's recursive "search molecules"
  mode, and keep the images while the coarse viewport level of detail is shown (they disappear
  during rotation of systems above the interactive atom limit).
- Bonds above the automatic-bonds limit (100 000 atoms): a fast path (bond impostors or lines)
  so large systems can show bonds by default too.

Analysis

- PLUMED: draw the bias forces as arrows, take MOLINFO residue names from the loaded topology,
  and plot `Plumed.*` attributes directly on the timeline with a current-frame marker.
- ConstructSurfaceMesh (alpha-shape surface), drawn with the existing `SurfaceMesh`.
- FindRings keeps every shortest-path ring; OVITO drops some near-maximum-size rings that differ
  from a same-size ring by a smaller one (closed-source, order-dependent). Decide whether to
  offer an OVITO-compatible mode once its rule is known.
- FindRings in numba for networks above ~100k atoms (8000 a-Si atoms, rings up to 10: 1 s).
- Spatial correlation function and Voronoi-weighted Steinhardt (OVITO / freud have them).
- The rest of OVITO's everyday modifiers: manual selection (box or lasso in the viewport, kept as
  a modifier) and Smooth trajectory (interpolated frames). FreezeProperty, UnwrapTrajectories,
  ExpandSelection, LoadTrajectory and CombineDatasets are done.
- Ion transport: error bars for the conductivity and Haven ratio (block averaging over
  independent windows; the collective charge MSD is one vector and noisy), Onsager
  coefficients L_ij per species pair, and the conductivity checked against a published
  Li-conductor trajectory (tests use synthetic walks with known answers).
- Site hopping: find sites without a reference structure (density maxima of the mobile ions
  over the trajectory), draw sites and hop paths in the viewport, and a hop-correlation
  (back-jump) statistic.
- Plots tab: G_s(r, t) / F_s(q, t) as a 2D map over r or q and t.
- Grain segmentation: the center of mass of a grain that percolates through the periodic
  boundaries depends on bond order (as in OVITO, so values differ from OVITO's for those grains);
  a grain-boundary surface mesh and misorientation-colored boundary bonds in the viewport; a
  Plots-tab view of `grains-log` with the chosen threshold to pick a manual one.
- Elastic strain and grain segmentation run single-threaded on the CPU (cluster building is
  sequential in OVITO too); parallel neighbor matching and a CuPy path if large systems need it.

Symmetry and diffraction

- hkl planes in the viewport (pick a reflection in the Plots tab, draw its lattice planes).
- Powder profile: 2-theta-dependent widths (Caglioti U, V, W), the K-alpha2 doublet, background,
  preferred orientation and absorption; a Rietveld-style scale/zero-shift fit to the measured
  pattern instead of the plain Rwp.
- Partial S_ab(q) columns from `StructureFactor`, and S(q) to G(r)/D(r) (PDF) for comparison
  with PDFgui-style data.
- Large cells: `PowderDiffraction` and `StructureFactor(method="direct")` cost atoms x
  reflections; above ~50k atoms sample reciprocal space instead of using every lattice point.
- Show the symmetry summary (space group, Wyckoff sites) in the inspector header rather than only
  as attributes.

Phonons

- Phonon DOS / projected DOS from phonopy `total_dos.dat`/`projected_dos.dat` in the Plots tab.
- Non-diagonal supercells for a mode (`supercell` as a 3x3 matrix), and several modes
  superposed (phonopy `--modulation` takes a list).
- Export a modulated structure (frame 0 as POSCAR) from the GUI; irreducible-representation
  labels of the modes (phonopy `--irreps`).
- Fast reading of large `mesh.yaml` files (the parser holds the whole file in memory).
- OUTCAR modes checked against real VASP IBRION=5-8 output; the fixture is written in VASP's
  layout from phonopy results.

Electronic structure

- DOSCAR, EIGENVAL and PROCAR readers (VASP 5 runs without vasprun.xml), and Quantum ESPRESSO /
  other codes' DOS and bands.
- vasprun.xml: the KPOINTS_OPT sections (`<eigenvalues_kpoints_opt>`); vaspout.h5 reads them.
- Plots tab: adjustable energy window (fixed at ±6 eV now), Gaussian smearing of the DOS,
  orbital choice for fat bands independent of the DOS column, spin texture of non-collinear runs.
- Drag an energy window on the DOS to select the atoms with the largest PDOS there, and the
  integrated PDOS in a window as a particle property to color by.
- Band projections of large runs (`<projected>` of a few GB) are read whole into memory on the
  first selection; read them per atom on demand.

Volumetric data

- Section caps where an isosurface meets the cell faces.
- Color an isosurface by a second grid (e.g. electrostatic potential on the density).
- BXSF (Fermi surface) reader.

Formats

- NEB: VTST's `neb.dat`/`spline.dat` and the `NEB: projections` lines of VTST OUTCARs (the
  profile is computed from positions, energies and forces now); NEB folders over SSH; an
  `images=` option on the reader for ASE NEB trajectories, so the band history needs no
  modifier parameter; dimer and string-method outputs.
- NEB: interpolate between images for a smoother movie; `OverlayFrames` copies are extra
  particles, so picking one in the viewport selects a copy, not the real atom.
- Ray tracer: see-through spheres are tessellated glass (`OverlayFrames` copies, Transparency);
  fine for small sets, slow for thousands.
- GROMACS `.gro` with velocities.
- LAMMPS binary dump.
- Large-frame reading against OVITO (docs/developer/readers.md): a 1M-atom dump frame takes
  0.45 s and 367 MB in ewald vs 0.23 s and 188 MB in OVITO (per-column copies, id sort); a
  1M-atom extxyz frame 0.85 s vs 0.33 s (ASE `Atoms` assembly).
- LAMMPS logs written with `thermo_modify line yaml`, and logs whose `thermo_style` has no `step`
  column (only `elapsed`/`time`); one-line and `multi` output are read.
- OUTCAR with interleaved output of several runs (pymatgen's `OUTCAR.electrostaticerror`: ionic
  blocks without their `Iteration` line) is read as fewer steps; ASE fails on it too.
- XDATCAR without `Direct configuration=` lines (frames separated by blank lines, pymatgen's
  `XDATCAR_4`).
- Hybrid band runs in vasprun.xml: breaks between lines of the zero-weight path are not found
  (no labels there; vaspout.h5 has them).

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
- Material presets (`@material`: roughness, metallic, glass, outline/toon) mapped onto each
  renderer (VTK PBR, ray tracer shading, vector flat/gradient/outline), the rest of phase 1.
- Overlay elements show in rendered images and the Render page preview, not in the interactive
  viewport.
- Phase 2: renderer backend registry, so a plugin can add a renderer that consumes the Scene IR.
- Phase 3 and later: glTF/HTML export from the IR, animated visuals, Claude-written visual
  elements (a Visual Forge, like Modifier Forge).

Plugins and Claude

- Plugin manifests, phase 2 (from the VS Code comparison): contributions declared in
  `ewald-plugin.toml` (`[[readers]]`, `[[modifiers]]`, ... with `python = "module:object"`)
  registered lazily and imported on first use; `settings` and `commands` contribution points
  (plugin `Option`s on a generated Preferences page; commands in the palette, as notebook
  functions and as Claude tools).
- Plugin manifests, phase 3: a curated plugin index (JSON in a GitHub repository), install and
  update with `pip` into the running interpreter, notebooks recording the plugins they use and
  offering to install missing ones.
- Plugin manifests, phase 4: notebook/session trust (Restricted mode for files from others),
  index curation rules, signatures if the index opens to outside publishers.
- Per-notebook plugin enable/disable; `plugins.json` holds only the global switch.

- LLM plugin ideas: Paper-to-modifier (Forge from a paper's definition, checked against its
  numbers), Run doctor (diagnose a VASP/LAMMPS run from its outputs), phase identification,
  campaign report over many runs.

Docs

- Example files: list the examples (and the downloadable paper datasets) on an empty viewport at
  first start; a GUI entry for `examples.fetch` (File › Open example › From a paper).
- Paper reproductions: StructureFactor `method="rdf"` holds every pair inside the cutoff at once,
  so 100,000 atoms with a 30 A cutoff run out of memory (the a-Si script uses 20 A); histogram in
  chunks. kinisi's covariance-weighted D for the LLZO runs is 0.6x a straight-line fit of the same
  MSD; find out why. Candidates found but not added: TiOF2 PDF against experiment (needs an
  X-ray weighted G(r)), the a-Si 216-atom HSE06 models (vldgroup/papers-defect-free-a-si), LGPS
  site maps (hacortesp/CrySF), MoSe2 phonons (Ionizing/adom.202403069); no license stated on the
  last three. Most other paper data sits on Zenodo or Materials Cloud, unreachable from the cloud
  container, so `examples.fetch` only knows GitHub-hosted files.
- Host the manual (the repository is private; CI uploads the built site as the `docs-site`
  artifact).
- More tutorials in the notebook style (examples/notebooks, tested by tests/test_tutorials.py):
  Ask Claude and Modifier Forge on a real question, a plugin from the Plugins side bar, a volumetric
  file (CHGCAR isosurface), opening a run on a cluster over SSH, and PLUMED/phonopy once they are
  plugins. The remaining manual pages (Python scripting, Rendering figures, Modifiers) still show
  `pipe.modifiers.append` and `import_file`; switch their examples to `load`/`pipe.add` when they
  are next edited.
- GUI screenshots in docs/images are made by docs/images/make_screenshots.py; Help › About and the
  Render workspace have none yet, and a GIF of linked selection (drag on a histogram) would show
  what a still cannot.

## Ideas backlog

From the feature-ideas review (2026-10-19), after the four in progress above.

- P1: Jupyter widget: `pipe.show()` with a rotatable viewport and frame slider in a notebook
  (PyVista trame backend).
- P1: Linked selection, rest: picking atoms plots their property time series over all frames;
  scatter plots of two properties (brushing a box selects atoms); keep a pair, bond or angle
  range as a modifier (only property ranges become an `ExpressionSelection` now).
- P2: glTF, interactive HTML and Blender export.
- P2: Batch render CLI: apply one `.ewald` session to many run folders.
- P2: Native GSD reader (H5MD and XTC/TRR/DCD are done).
- P2: Live reload for vasprun.xml, H5MD, XTC/TRR/DCD and compressed files (dump, XDATCAR,
  OUTCAR and extxyz follow a growing file now); a file-system watcher instead of polling.
- P2: More HDF5 output: Quantum ESPRESSO `charge-density.hdf5` (G-space, needs an FFT to the
  real-space grid), ABINIT netCDF-4 `*_HIST.nc` trajectories and `*_DEN.nc` densities, VASP
  `vaspout.h5` `/results/partial_charges` (PARCHG).

## Verify outside the cloud container

- CUDA paths on a real NVIDIA GPU: CuPy kernels, neighbor lists, cluster analysis, the
  `--device cuda` GUI switch. CI and the cloud container have no GPU.
- SSH remote files against a real HPC login node (ProxyJump, OTP, large dumps); tests use an
  in-process SFTP server.
- Live reload on a real running LAMMPS/VASP job (local and over SSH), where the writer flushes
  in arbitrary pieces; tests grow fixture files byte by byte.
- Claude features with a real API key: `ewald ask` (also `--select`), the GUI Ask Claude tab,
  `ewald narrate`, `ewald forge`, `ewald mcp`, and Preferences › Claude › Test connection.
- The API key in a real keychain (macOS Keychain, Windows Credential Manager, GNOME Keyring /
  KWallet); the container has none, so only the 0600 file fallback runs for real.
- The Windows release workflow (wheels, PyInstaller app, installer) on a real `v*` tag; no
  release exists yet, so Windows users without Build Tools get ewald without `ewald._native`
  until the first one (blocked on GitHub Actions minutes).
- Installing without a compiler on real Windows (MSVC missing raises in `build_ext.run`; tested
  here with `CXX=/bin/false` on Linux). pip shows the "not built" warning only with `-v`.
- `DislocationAnalysis` and `OvitoModifier` with the `ovito` package installed.
- The native OUTCAR and vasprun.xml readers on multi-GB real output. Checked on the real
  pymatgen test files (VASP 5.2-6.3: relaxations, MD, NEB, SOC, DFPT, GW/BSE, HSE, LVEL,
  KPOINTS_OPT; 39 OUTCARs, 52 vasprun.xml) against ASE; a few of them are in tests/data/real.
- NEB folders from more real VTST runs (pymatgen's `neb1` checked: barrier and energies as
  pymatgen's NEBAnalysis; the VASP fixture is an EMT band written in VASP layout; the LAMMPS
  fixture is a real `neb` run).
- DOS, PDOS and band reading on real VASP output (ISPIN = 2, LORBIT = 10/11/12, non-collinear,
  HSE band runs, KPOINTS_OPT); the fixtures are synthetic, checked against pymatgen (vasprun.xml)
  and py4vasp (vaspout.h5).
- The vaspout.h5/vaspwave.h5 readers on real VASP 6.4/6.5 output (MD, relaxation, ISPIN=2,
  non-collinear, `LCHARGH5`); the fixtures follow py4vasp's schema and are checked against
  py4vasp, and the charge density is assumed to be stored as rho * V like a CHGCAR. The H5MD
  reader on GROMACS 2026 `mdrun` output.
- The Legacy GL viewport on a real `ssh -X` session (RHEL 9 crashed with `XIO: fatal IO error`
  under VTK): whether Qt's GLX integration gets an indirect context there, and how fast display
  lists are over the network. The container's X servers crash on indirect GLX, so it is tested on
  direct Mesa only. If Qt's context fails where plain GLX works, draw into a raw GLX child window
  instead of `QOpenGLWindow`. Same for the VTK probe and Mesa fallback.
- GUI startup checks on a Wayland desktop and on Fedora (the xcb library check is skipped when
  `WAYLAND_DISPLAY` is set; the Fedora package names in its message are untested).
- PLUMED from conda-forge on macOS and Windows (tested here with a 2.10 source build on Linux;
  CI has no PLUMED, so those tests skip).

## Code health

Left from the 2026-10-19 review (minimum image, element masses, color parsing and the
dark-background test are shared now; `gui/app.py` and `gui/preferences.py` are split into
mixins and helper modules):

- One camera basis (right/up/forward) for `vector.py`, `overlays.camera_axes` and `raytrace.py`;
  only the overlays one handles `up` parallel to the view direction.
- One element color lookup: `themes.element_colors`, `Settings.element_color` (Preferences
  offers only Jmol and CPK, not VESTA) and `render.type_color`.
- `plugins.py`: the visual-element and overlay registries are copies of each other; the
  renderer-neutral type color/radius tables in `render.py` belong in their own module.
- `Viewport._mic` and the bond-angle code in `gui/viewport.py` can use `Cell.minimum_image`
  and share the angle helper with `llm/selection.py`.

## Security

From the 2026-10-19 review (expression sandbox, session classes, Forge consent and environment,
`export_file` paths, PLUMED `LOAD` and the known_hosts key type are fixed):

- Claude's read tools (`load_structure`, `read_plumed_output`) open any local path and any
  `host:/path` in known_hosts; add an optional folder allow-list and a GUI confirmation before
  connecting to a host the model chose.
- The GUI's "Python code in session" question covers only top-level PythonScript and
  PythonModifier entries; nested `__modifier__` code and a disabled `Plumed` load silently
  disabled, with only a log message.
- The saved API key: the Forge test subprocess gets a null keyring backend and a scratch config
  folder, but plugin code there could still read `~/.config/ewald/credentials.json` (file
  fallback) by absolute path or call the OS keychain tool directly. Run that subprocess as a
  sandboxed user or container to close this.
- `plumed.py` reads numbers like `pi/2` with `eval` (no builtins, only `pi`); replace it with a
  small arithmetic parser.

## Infrastructure

- GitHub Actions minutes ran out on 2026-10-19, so every workflow now runs only on manual
  dispatch and pull requests are verified locally (`xvfb-run -a python -m pytest -q`,
  `mkdocs build --strict`). Restore the `push`/`pull_request`/tag triggers once minutes are
  available again (docs/developer/testing.md shows the block).

- CI installs each extra on its own only through the test suite's `importorskip`s; a job that
  installs every extra in a clean venv and imports its modules would catch broken extras
  (`tests/test_dependencies.py` only checks that every import is declared).

## Deferred until the user asks

- Structure editing: supercells, slabs, defects. Do not start it without an explicit request.
