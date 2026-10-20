# TODO

The one list of open work. What already exists is in the [feature inventory](features.md),
generated from the code; the milestones and decisions are in the [roadmap](ROADMAP.md). Every
pull request that adds or changes a feature regenerates the inventory and updates this page:
it removes the items it finishes, adds what it leaves open, and moves its own line out of
"In progress" when it merges.

## In progress


## Next

Performance (measured 2026-10-20 in the cloud container, 4 cores, llvmpipe GL; numbers from
`benchmarks/analysis.py`, `benchmarks/large_files.py` and the GUI under `xvfb-run`)

- Pipeline output cache: plugin modifiers are never reused (`Modifier.cacheable` is None for
  them); first-party plugins (ewald-phonopy, ewald-plumed) could set `cacheable = True` where
  their output depends only on their settings.
- Structure identification at 108k atoms spends half its time in the periodic kNN query
  (scipy cKDTree plus image padding, ~0.4 s): a numba cell-list kNN like `find_pairs` would
  roughly halve PTM (0.9 s) and Voronoi (0.9 s). The native kernels already use every core.
- Ray tracer: a BVH instead of the uniform grid for large surfaces, adaptive sampling that
  stops on converged pixels, and the CUDA path.
- GUI startup is about 3 s, mostly imports: pyvista/pyvistaqt (~2 s) and matplotlib (0.6 s,
  needed by the Plots tab at construction); the IPython console is built after the first paint.
- UI theme switch takes 0.5-1 s in `QApplication.setStyleSheet` (one app-wide QSS).

GUI

- Panel layout (phase 1 landed: `gui/workbench.py`, `gui/layout_model.py`; plan in project
  files `design/panel-layout.md`): phase 2 opens a notebook view cell's plot large in the
  editor area ("Open large" or drag the cell header, a Bring back placeholder in the cell, the
  layout saved in the notebook metadata) and makes the Render workspace a tab there; phase 3
  makes the Notebook, Display, Ask AI, Plugins and Remotes side bars views too, adds Move
  to new window (multi-monitor), named layouts, drop bands along an area's outer edge for
  full-length splits, a second 3D view (another camera) and a `@view` plugin API. Not yet tried
  on a real `ssh -X` display.
- Interactive plots (zoom, pan, energy window, hover and click-to-frame on every plot exist):
  drag on an axis to zoom that axis only and pinch zoom on touchpads; hover readout of the
  phonon mode (q-point, band, frequency) and band index under the cursor; hovering a per-frame
  plot previews that frame on the timeline; linked zoom between plots (a DOS beside the bands
  sharing the energy axis, once the panel layout can place plots side by side); a scatter view
  of two particle properties where a brushed box selects atoms; the `show()` script path draws
  DOS/bands like the GUI (it plots the raw table columns now).
- AI chat side bar (`gui/ask_panel.py`, activity bar): stream answers as they arrive (now the
  whole answer at once), keep chats per notebook tab and save them in `.ewd`,
  show tool images inline, insert as a step cell (form) when the block is one `pipe.add(...)`,
  edit or regenerate a sent message. The notebook cell tools (`llm/notebook_tools.py`) could
  highlight the cells the AI changed and show a diff to accept or undo; `ewald ask` and the MCP
  server could take a notebook (`--notebook analysis.py`) and save it back.
- `PythonScript` step cells: `Modifier.__repr__` uses `name`, which PythonScript sets to its title,
  so `step_source` writes `pipe.add(Python script(...))`, which does not parse.
- UI style (flat buttons, fields, checkboxes and tab close icons in `gui/theme.py`): a few widgets
  still set inline colors (render preview, pick labels, color swatches, field error borders), so
  they keep the old palette until a restart after a theme switch; move them to QSS object names.
- Notebook layout: the viewport could follow the current notebook even when it has no
  pipeline yet (now it keeps the last one); image/figure cells; inline Ask AI suggestions
  as diffs; a snapshot cell that updates when the camera of its view changes (now it is code).
- 3D view as code (`pipe.view`, the `view(pipe, ...)` cell, Claude's `set_view`): settings the
  view does not have yet (atom labels drawn in the view, light direction
  and strength, picks and distance/angle measurements as
  code, the viewport axes widget; audit in the project files `ewald-review/view-elements-audit.md`);
  the Display panel cards (Atoms, Types, Bonds, Cell, Elements) could take the visual element
  names `pipe.view.elements` uses (Particles, Bonds, Simulation cell, Surfaces, ...) and gain a
  Surfaces card for the `surfaces` setting; the camera reaches the view cell
  only through Insert › View settings with the camera (not on every rotation); the Render page
  keeps its own per-notebook state in the `.py` header next to the view cell's picture settings
  (one place for both); in a kernel notebook a viewport edit reaches the kernel's `pipe.view`
  only when the view cell runs again.
- Render frame (the picture's region on the 3D view, `view(pipe, region=...)`): not drawn on
  the OpenGL 1.x viewport (`ssh -X`); in perspective, a frame off the view center moves the
  camera sideways instead of an off-axis projection (exact at the focal distance only); the
  Overlay elements and text label are not previewed inside the frame; Claude's `set_view` can set
  `region` but no tool tells it the frame on screen.
- Notebook kernels (`ewald/kernel.py`): step forms for steps a kernel cell adds with code the
  app cannot import (a class only the kernel's Python has stays as code).
- Notebook kernel on the cluster (plan, owner's pick 2026-10-20; not started): run a notebook
  where its data is, so a 50 GB dump is analysed on the cluster and only the shown frame and the
  tables come back. Steps: (1) `Kernel.ssh(profile, python=...)` starts `python -m ewald.kernel_bridge`
  over the saved SSH profile (`io/profiles.py`, paramiko channel; jupyter_client's ssh tunnel or a
  plain stdio channel, no open port on the login node), with a check that the remote ewald version
  matches; (2) the bridge already sends frames as `.npz`: add compression and a frame budget so a
  1M-atom frame over a slow link stays interactive, and send tables and plots only; (3) the
  Notebook › Kernel menu lists the Remotes profiles, with an option to start the kernel inside an
  interactive allocation (`salloc`/`srun --pty`, the command kept per profile) so it runs on a
  compute node, GPU nodes giving `device="cuda"`; (4) `load()` paths in a remote kernel are the
  cluster's own paths (no `host:` prefix), and the notebook records the profile so it reopens
  there; (5) trust: a remote kernel runs the notebook's code on the cluster, so the same
  `Notebook.trust()` gate applies, plus a confirmation naming the host. Tests: the in-process SFTP
  server with an exec channel (`tests/helpers/sftp_server.py`). Related: a JupyterLab widget
  (`pipe.show()`, Ideas backlog P1) for groups that use JupyterHub on the cluster.
- Plugins side bar: disable per notebook,
  "Recommended" plugins required by the open notebook, a Contributions tab and a full Forge
  view instead of the Ask AI prompt; uninstalling a pip plugin still needs a restart.
- UI themes: prompts already printed in the Python console keep the colors of the theme they
  were printed in (new input, syntax colors and the background switch at once).
- Remotes side bar: list folders and connect in a background thread (now the GUI waits, with
  a busy cursor), drag a remote file onto the notebook, mirror a run folder to a local folder
  (rsync-like, for offline analysis), remote data links in the `.ewd` manifest (remote paths are
  skipped now; the code keeps `name:/path`), a remote terminal or `squeue` view, profile import
  from PuTTY/MobaXterm/Termius, and "remember this password" on the login prompt.
- Spectra, Figure and Forge views from the redesign mock-ups, built as view cells
  rather than separate panels.
- Quad views (top, front, left, perspective), as in OVITO.
- Undo/redo: display style changes (colors, radii, camera) (only notebook edits are undone now); a View cell's table or column change is its own entry instead of
  joining the next one.
- Background compute: notebook cells still run on the GUI thread (stdout capture is
  process-wide), so a cell that computes many frames itself blocks the window until the
  notebook kernels below exist; view cells and the Plots tab compute other frames on the GUI
  thread too.
- Scene updates without a full rebuild when only colors or a few properties change, and frame
  changes with the same atoms that replace only the point coordinates.
- Picking without a render: click picks still use a `vtkPropPicker` selection render; the hover
  readout already does a CPU ray-sphere test (`camera_ray`, `ray_spheres` in `gui/viewport.py`)
  that `pick_at` could reuse.
- App icon outside the running window: a Linux `.desktop` entry and hicolor icons installed
  with the package (the Windows app and installer use `ewald.ico`, the macOS bundle
  `ewald.icns`); the EWALD mark in the main window header (the Notebook redesign) and the docs.
- Icons for the remaining menu bar entries (View, Notebook, Insert, Tools) and the Insert › Step
  categories. Buttons no longer use text glyphs (panel headers, timeline, cells, view bar use
  line icons from `gui/icons.py`).
- Panel headers (`gui/header.py`): checked under xvfb with Windows-like scale factors (100, 125,
  150 %) only; check on a real Windows machine (Segoe UI, per-monitor DPI) and macOS. The view bar cuts "View along" when the
  3D view is narrow (could fold into a menu).
- Context-dependent menus beyond the viewport right click: a "Suggested" group at the top of
  Insert › Step for the loaded data, view-along buttons hidden without a cell, the timeline
  folded for a single structure.
- Shortcuts for the viewport tools (orbit, pick, distance, angle, bond) and view-along axes.
- Impostor spheres: exact ray-sphere depth in perspective views (now approximate near the
  screen edges) and VTK scene lights instead of the fixed head light; bond impostors.
- Flat (ASE) style in the ray tracer (now OpenGL and vector only), and a
  per-element bond table like VESTA's (cation-anion only, measured lengths) instead of the
  covalent-radius sum with metal-metal pairs left out.
- Replicated cell display: a quick replicate control in the 3D view toolbar or the redesigned
  View tab (now Display panel, View menu, right-click menu and Ctrl+K).
- Boundary (VESTA): repeat isosurfaces and grid slices over the boundary range too (now only
  atoms, bonds and coordination polyhedra are repeated), VESTA's recursive "search molecules"
  mode, and keep the images while the coarse viewport level of detail is shown (they disappear
  during rotation of systems above the interactive atom limit).
- Bonds above the automatic-bonds limit (100 000 atoms): a fast path (bond impostors or lines)
  so large systems can show bonds by default too.

Runs (`ewald/runinfo.py`, `ewald/runs.py`; manual page docs/user/runs.md)

- `run_info` / `check_run` on real outputs: VASP 5/6 MD with `NBLOCK` > 1 (whether XDATCAR numbers
  its configurations by frame or by ionic step; both are handled, only the first is tested),
  MDALGO 3 NPT, real LAMMPS logs with `variable`-substituted commands, `include` files and several
  `run`s with different fixes (the last thermostat wins now), and `thermo_modify line yaml`.
- Quantum ESPRESSO, CP2K and ABINIT inputs (`run_info` knows VASP and LAMMPS only).
- Use `check.equilibrated` as the default `start=` of MSD, time averaging and the transport
  modifiers (now the user passes it); the transport modifiers and the GUI step forms still default
  `time_per_frame` to 0 (frames) instead of the run's value.
- Time units on plot axes: the MSD/VACF `Time` column is in ps when `time_per_frame` is known, but
  the table does not say so, so the axis reads "Time".
- Block-averaged error bars for every per-run number (D, temperature, energy), and an Arrhenius
  fit weighted by them (now unweighted, with the fit's own standard error).
- `show(runs, ...)` as a live GUI view cell: a view cell binds to `pipe`, so the overlaid runs show as
  the `RunTables` picture under the cell instead of an interactive plot; run comparison in the
  Plots tab.
- A run folder index: scan a project folder (local or on a cluster) for runs and list them with
  kind, state and last energy, as the way into `load_runs`.
- The GUI's run picker binds `pipe` only for notebooks running in this app's Python, not in a
  kernel.

Analysis

- Apple GPU compute (Metal): a backend in `backend.py` for MLX or PyTorch MPS, starting with the
  dense kernels (powder diffraction, S(q)); the neighbor list needs data-dependent shapes they
  lack. Macs compute on the CPU until then.
- PLUMED (ewald-plumed plugin): draw the bias forces as arrows, take MOLINFO residue names from the loaded topology,
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

Phonons (ewald-phonopy plugin, frssp/ewald-plugins `packages/ewald-phonopy`)

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
- Plot style for plain `ewald` use in Jupyter (outside the GUI notebook): an opt-in
  `%load_ext ewald` or `ewald.plotstyle.use()` hint in the manual is all there is now; the GUI
  console's interactive figure windows read a few draw-time rcParams outside the swapped style.
- Plot style (`ewald/plotstyle.py`): per-plot overrides in view cells (`show(..., size=,
  labels=)`; `xlim=`/`ylim=` exist), time units on MSD/VACF axes when `time_per_frame` is set (the unit is not
  stored now, so the axis says "Time"), multi-panel figures (DOS beside bands sharing the energy
  axis), and a journal preset list (ACS, APS, Nature widths and fonts).
- Plots tab: Gaussian smearing of the DOS,
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
  profile is computed from positions, energies and forces now); an
  `images=` option on the reader for ASE NEB trajectories, so the band history needs no
  modifier parameter; dimer and string-method outputs. A NEB folder opened from File › Open
  (a directory picker there) instead of File › Load NEB folder or a drop.
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

Plugins and AI

- AI providers (`llm/providers.py`): the OpenAI-compatible provider never sends images (no
  per-server vision switch yet); Effort reaches only Claude and OpenAI reasoning models; one
  `llm.model` setting for all providers (switching the provider in Preferences clears it); no
  streaming for any provider; a native Gemini or Bedrock/Vertex client would need a `Chat`
  subclass (today Gemini goes through its OpenAI-compatible endpoint). API gateway provider:
  untested against a real an API hub; keys are per provider, so two gateway connections with
  different credentials share one keychain entry; no `GATEWAY_<N>_*` multi-gateway blocks or
  `GATEWAY_NO_POOL` (sim-agent has both).
- Plugin manifests, phase 2 (from the VS Code comparison): contributions declared in
  `ewald-plugin.toml` (`[[readers]]`, `[[modifiers]]`, ... with `python = "module:object"`)
  registered lazily and imported on first use; `settings` and `commands` contribution points
  (plugin `Option`s on a generated Preferences page; commands in the palette, as notebook
  functions and as AI tools).
- Plugin manifests, phase 3: the index repository exists (frssp/ewald-plugins: `plugins/*.toml`,
  `index.json`, first-party `packages/ewald-phonopy` and `packages/ewald-plumed` pinned to tags).
  Left: install and update with `pip` from the Plugins panel into the running interpreter,
  notebooks recording the plugins they use and offering to install missing ones, and a CI job in
  ewald-plugins that runs each package's tests against ewald main (run them by hand until then).
- More first-party plugins where core support is tied to one outside code (candidates: the
  `ovito` bridge, MDAnalysis formats); built-in format groups stay in core.
- Plugin manifests, phase 4: `.ewald` session trust through the same trusted-file store as
  notebooks (`ewald.trust`), index curation rules, signatures if the index opens to outside publishers.
- Per-notebook plugin enable/disable; `plugins.json` holds only the global switch.

- AiiDA (ewald-aiida in frssp/ewald-plugins: local profiles, and remote ones over SSH with
  `aiida://user@host/1234`): an AiiDA side bar (profiles and hosts, recent workflows, outputs;
  copy `load("aiida:...")`), a AI tool to find and open workflows, PDOS/projected bands
  from aiida-quantumespresso, NEB and aiida-lammps outputs, streaming very long remote
  trajectories frame by frame instead of one bundle, export of analysis results back as new
  nodes (opt-in, never default). Not yet tried against a real remote AiiDA host.
- Other `schemes=` sources: Materials Project (`mp:mp-149`), OPTIMADE, NOMAD.

- LLM plugin ideas: Paper-to-modifier (Forge from a paper's definition, checked against its
  numbers), phase identification, campaign report over many runs (`check_run` and `load_runs`
  are the deterministic base for both; the AI explains the findings).

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
  Ask AI and Modifier Forge on a real question, a plugin from the Plugins side bar, a volumetric
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
- P2: Live reload for XTC/TRR/DCD and compressed files (dump, XDATCAR, OUTCAR, vasprun.xml,
  extxyz, H5MD and NEB folders follow a growing file now); a file-system watcher instead of
  polling; resume a growing vasprun.xml from its last step instead of indexing it again.
- P2: Unfinished calculations, rest: a persistent "running" badge on the timeline (the note is a
  status-bar message and a log line now); the GUI's AI chat and `ewald ask` could mention
  `reader.incomplete`; QE, CP2K, ABINIT outputs go through ASE, which fails on a cut file.
- P2: More HDF5 output: Quantum ESPRESSO `charge-density.hdf5` (G-space, needs an FFT to the
  real-space grid), ABINIT netCDF-4 `*_HIST.nc` trajectories and `*_DEN.nc` densities, VASP
  `vaspout.h5` `/results/partial_charges` (PARCHG).

## Verify outside the cloud container

- CUDA paths on a real NVIDIA GPU: CuPy kernels, neighbor lists, cluster analysis, the
  `--device cuda` GUI switch. CI and the cloud container have no GPU.
- SSH remote files against a real HPC login node (ProxyJump, OTP, large dumps); tests use an
  in-process SFTP server. Saved remote computers with a real OS keychain (macOS Keychain,
  Windows Credential Manager, GNOME Keyring) and a passphrase-protected key from a real agent.
  The frozen Windows installer and macOS app now carry paramiko and keyring by default: check
  that PyInstaller bundles the platform keyring backend.
- Live reload on a real running LAMMPS/VASP job (local and over SSH), where the writer flushes
  in arbitrary pieces; tests grow fixture files byte by byte.
- A vaspout.h5 from a real running and a real killed VASP 6 job: which groups exist mid-run,
  whether unwritten steps are zeros, and whether HDF5 can open the file of a killed job (tests
  use h5py-written fixtures, a second process holding the write lock, and a cut file).
- AI features with a real API key (Claude; also OpenAI, Gemini and a local Ollama/LM Studio
  server, which are tested only against mocked servers): `ewald ask` (also `--select`), the GUI AI chat side bar
  (notebook code in the prompt, code blocks inserted as cells, the notebook cell tools),
  `ewald narrate`, `ewald forge`, `ewald mcp`, and Preferences › AI › Test connection.
- The API key in a real keychain (macOS Keychain, Windows Credential Manager, GNOME Keyring /
  KWallet); the container has none, so only the 0600 file fallback runs for real.
- The release workflows (`release.yml`: sdist and Linux wheels; `windows-release.yml` and
  `macos-release.yml`: wheels, apps, installers) on a real `v*` tag. v0.1.0 is prepared (CHANGELOG, version,
  `packaging/release_check.py` passes locally) and waits only for `git tag v0.1.0`, a push and
  dispatching the three workflows ([Releases](developer/release.md)); until then Windows users
  without Build Tools get ewald without `ewald._native` (blocked on GitHub Actions minutes).
- Installing without a compiler on real Windows (MSVC missing raises in `build_ext.run`; tested
  here with `CXX=/bin/false` on Linux). pip shows the "not built" warning only with `-v`.
- macOS on a real Mac (Apple Silicon and Intel): the source install and the full test suite, the
  GUI on Retina (the VTK view filling the whole widget, picking and hover at the right atom),
  trackpad orbit/zoom, the EWALD menu bar name, and EWALD.app from `macos-release.yml` with
  Finder Open With and Dock drops. The list is in docs/developer/macos-release.md; the container
  checks clang, the spec and the dependency wheels only.
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
- The ewald-plumed plugin with PLUMED from conda-forge on macOS and Windows, and with a 2.10
  kernel (checked here with a 2.9.2 source build on Linux, where the two tests that need 2.10,
  `SUM` and per-atom vectors, fail).

## Code health

Left from the 2026-10-19 review (minimum image, element masses, color parsing and the
dark-background test are shared now; `gui/app.py` and `gui/preferences.py` are split into
mixins and helper modules):

- One camera basis (right/up/forward) for `vector.py`, `overlays.camera_axes` and `raytrace.py`;
  only the overlays one handles `up` parallel to the view direction.
- One element color lookup: `themes.element_colors`, `Settings.element_color` (Preferences
  offers only Jmol and CPK, not VESTA) and `render.type_color`.
- The renderer-neutral type color/radius tables in `render.py` belong in their own module.
- `Viewport._mic` and the bond-angle code in `gui/viewport.py` can use `Cell.minimum_image`
  and share the angle helper with `llm/selection.py`.
- Names left from the Ask AI rename: `MainWindow.ask_claude` (and the tests calling it) and the
  `"Ask Claude"` dock object name, kept so saved layouts still restore; rename them with the
  internal renames (step 3 of the layout review), keeping the old dock name readable.

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
- `ewald_plumed/plumed.py` reads numbers like `pi/2` with `eval` (no builtins, only `pi`); replace it with a
  small arithmetic parser.

## Infrastructure

- From the 2026-10-20 systems review (/mnt/project-files/ewald-review/senior-systems-review.md;
  its refactors are merged, #129-#135): a self-hosted runner (xvfb + Mesa) to restore the PR
  gate without Actions minutes; render/VTK tests in a subprocess so a VTK abort fails one test,
  not the suite; a lighter core install (`vtk`/`pyvista` and `paramiko`/`keyring` as extras or
  lazy imports) and manylinux wheels so PTM/Voronoi/CSP are not missing without a compiler.
- GitHub Actions minutes ran out on 2026-10-19, so every workflow now runs only on manual
  dispatch and pull requests are verified locally (`xvfb-run -a python -m pytest -q`,
  `mkdocs build --strict`). Restore the `push`/`pull_request`/tag triggers once minutes are
  available again (docs/developer/testing.md and docs/developer/release.md show the blocks).
- Release follow-ups: a Linux `.desktop` entry, a version resource on the Windows executables, aarch64 Linux
  wheels, and an `[Unreleased]` CHANGELOG line from every user-visible pull request.

- CI installs each extra on its own only through the test suite's `importorskip`s; a job that
  installs every extra in a clean venv and imports its modules would catch broken extras
  (`tests/test_dependencies.py` only checks that every import is declared).

- macOS release: a first run of `macos-release.yml` (manual; macOS minutes bill 10x), then
  Developer ID signing and notarization so the downloaded app opens without the Gatekeeper
  override (needs an Apple Developer account and the certificate as a workflow secret).

## Deferred until the user asks

- Structure editing: supercells, slabs, defects. Do not start it without an explicit request.
  The display side exists (`view(pipe, replicate=...)`, 2026-10-20); the user said a real
  supercell step (a general 3x3 matrix like VESTA's lattice transformation, with bonds, grids
  and polyhedra following, analysis and export on the enlarged cell) is "not now".
