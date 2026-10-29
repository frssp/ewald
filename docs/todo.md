# TODO

The one list of open work. What already exists is in the [feature inventory](features.md),
generated from the code; the milestones and decisions are in the [roadmap](ROADMAP.md). Every
pull request that adds or changes a feature regenerates the inventory and updates this page:
it removes the items it finishes, adds what it leaves open, and moves its own line out of
"In progress" when it merges.

## In progress

- 1M atoms in real time in the 3D view (D73, [plan](developer/viewport-performance.md)):
  `benchmarks/viewport.py` is in, the bond search and reading ahead run off the GUI thread
  (phase 1), and bonds are built in numba with alike halves drawn as one stick (phase 2). Phases
  3-4 (an own GPU bond renderer, culling) wait for the numbers from the owner's Intel Arc
  (`python benchmarks/viewport.py -o arc.json`). Left from phase 2: the scene build and stick input
  of a 1M-atom frame with bonds (4.4 s) still run on the GUI thread; the compute worker could
  build the `Scene` too.

- A smaller Windows standalone app (D80 follow-up): setup.exe 204 MB and zip 322 MB (were 294 and
  396) after leaving out unused VTK modules (`packaging/bundle_excludes.py`) and sharing the
  bytecode (`noarchive`). The `Size` step of `windows-release.yml` lists what is still big
  (llvmlite 115 MB, ffmpeg 84 MB, VTK 230 MB, Qt 105 MB). Next candidates: Qt Quick/QML, which
  nothing uses, and making movie export's ffmpeg an installable component like GPU compute.

## Next

- Running-cell marks in a kernel notebook: when a cell rebinds `pipe` and the cells below run
  again inside the kernel, only the first cell is marked as running (the kernel reports the rest
  when they are all done); the kernel bridge could send a start message per cell.
- Trajectory tubes (`trajectory_style="tube"`): the vector renderer paints each piece at the depth
  of its start (painter's algorithm), so a piece that ends inside an atom nearer the camera than its
  start can cover that atom, as lines do there; sorting by the far end or splitting pieces at atom
  surfaces would fix both. Tubes are one cylinder per traced step, so very long paths of many atoms
  make large meshes in the viewport (no LOD for them yet).
- Proteins: a proof-of-concept `ewald-protein` plugin (PDB/mmCIF via gemmi, simplified DSSP,
  cartoon as one `SurfaceMesh`) sits on the ewald-plugins branch `claude/project-thread-v41ert`
  without tests or an index entry; not planned until the owner asks. Open for it: 3-10/pi helices
  and bulges, ligands and nucleic acids, picking a residue on the cartoon, and a lighter vector
  output (a 214-residue cartoon is a 9.7 MB SVG).
- Mesh lighting: the ray tracer shades every triangle, so grid slices (vertex colours, unlit in
  OpenGL, the legacy viewport and vector files) come out shaded there; an unlit triangle flag in
  `raytrace.py` and `raytrace_gpu.py` would make `SurfaceMesh.lit=False` look the same everywhere.
- Simulation codes (`ewald.codes`, Preferences › Codes): only VASP and LAMMPS are built in and the
  Quantum ESPRESSO entry is an example plugin; CP2K, GPAW, Gaussian and QE as real first-party
  plugins are open. NEB and phonopy folders are still recognized outside the registry
  (`workflow_folder._neb_step`, `_phonopy_step`), `read_thermo` falls back to trying the LAMMPS log
  and OSZICAR parsers, the GUI's open filter groups formats by reader category rather than by
  code, the Codes page cannot switch a built-in code off (the Plugins side bar does), and no AI
  tool lists the codes yet.
- Long cell outputs: whether a cell's output is scrolled or shown whole is kept per window only;
  Jupyter stores it as the cell's `scrolled` metadata in `.ipynb` (read and write it once cells
  keep metadata), and so is a hidden output (Jupyter's `collapsed`).
- Library API, rest of the audit: frame selection when loading (`load(path, frames=...)` like ASE's
  `index=":"`) and reader keyword errors that name `load()` instead of the reader class; internal
  state (`Color.rule`) out of `data.attributes`; `data.tables` values as `verbs.Table`.
- Activity bar: buttons cannot be reordered by dragging yet (hiding and the … overflow menu work).
- Menus after the 2026-10-23 review: right-click menus still mix entries with and without icons;
  context menus have no mnemonics; Edit has no Find in notebook; the Keyboard shortcuts window lists
  shortcuts but cannot change them; the palette's recent commands are kept for the session only.
- Profiles along the cell and the 3D view (`linking` "space" links, Align 3D view): the plot's
  range fills the 3D view's width in Angstrom, not pixel for pixel with the plot's axes; the
  OpenGL 1.x viewport draws no hover plane or slab box; picks on replicated copies (`replicate=`) only ring
  the atoms of the first copy; a profile along a tilted cell vector keeps no `Keep as selection`
  (expressions have no fractional coordinates).
- Search box and + Step (D57): the step list of the notebook bar's + Step and the Ctrl+K palette leave
  out steps that cannot work on the current frame, where the menus grey them out with the reason.
- Workflow analyses (`ewald.workflow.Analysis`, `ewald.potential`): a table's `levels`, `regions`
  and `gaps` are drawn in cell outputs, figures, the Plots panel and the `show(<result>)` view cell
  every analysis with a `result` now ends with (`Analysis.view_cell`, also run by `add_analysis`),
  but a table's attributes do not cross from a separate kernel (frames come back
  as `.npz`), and the side bar has no "Analysis" row. Only the
  sim-agent band-offset lineup and the per-surface work function exist; the work-function
  workflows' surface energies and the polycrystalline aggregate, EOS and formation energy are next.
  The lineup matches sim-agent's own `assemble` on interfaces built by sim-agent's builder with
  model potentials (within 0.001 eV, relaxed seams included); not yet checked against a real run.
- Opening without running (`python.run_notebooks`, default "never"): covers notebook files only. A
  folder or workflow (`simagent:`, AiiDA, calculation folders) still runs its generated `load(...)`
  notebook at once, since it is ewald's own code; no per-notebook "always run this one" choice yet.
- Open dialog: `log.lammps` and `OSZICAR` are read only as companions of a trajectory, not
  opened on their own.
- Several files as one trajectory: restart detection (on Timestep, between files and inside one
  appended dump) works for LAMMPS dumps only; extxyz `Timestep=` comments, H5MD steps and XTC
  steps are not used yet. The join dialog shows file names only, not each file's frames and steps
  before opening (that needs a scan of every file).
- View tab: a card has no per-card reset to defaults.
- Atom color rules (`ewald.color_rule`): a step's category colors (structure types) can only be changed
  through the step (`color_by_type=False` plus `paint`), not per category in the Colors card as OVITO's
  type list does; `color_by` an integer property uses a continuous color map, not one color per value;
  the legacy OpenGL (ssh -X) viewport draws no category legend, and the GUI viewport's legend has no
  on/off switch (`Viewport.show_legend`). Plugin modifiers that write "Color" show as "custom" until
  they call `color_rule.record` (not yet in `plugin_api`).

- Workflows (`ewald.workflow`): opening one step's input next to its output (INCAR/KPOINTS as
  text cells). `workflow_table` matches steps across workflows by name or position only (no
  manual pairing in the GUI), its side bar entry needs a notebook cell (no table view of its
  own), and the folder source reads no LAMMPS energy (last `PotEng` of `log.lammps`), no
  Fermi energy and no gap from vasprun.xml or DOSCAR (EIGENVAL only, up to 32 MB). The side bar's step switch (●/○) and
  "Plot <table>" rows exist; a step picker on top of the Plots panel itself does not, and the
  rows list the tables of a step's first frame only.
- Workflow folders (`ewald/workflow_folder.py`, docs/user/workflows.md): a killed VASP or LAMMPS
  step shows as running (no mtime or queue check); Quantum ESPRESSO, CP2K, ABINIT folders are not
  recognized; no "remove from list" in the Workflows side bar; a large tree on an SSH host is
  listed one folder at a time (no `find` in one round trip); `load_runs` over the steps of a
  workflow (several MD temperatures as one `Runs`).

Performance (measured 2026-10-20 in the cloud container, 4 cores, llvmpipe GL; numbers from
`benchmarks/analysis.py`, `benchmarks/large_files.py` and the GUI under `xvfb-run`)

- Pipeline output cache: plugin modifiers are never reused (`Modifier.cacheable` is None for
  them); first-party plugins (ewald-phonopy, ewald-plumed) could set `cacheable = True` where
  their output depends only on their settings.
- Results that survive Run all: `Notebook.run_all` starts a new namespace, so the new Pipeline
  indexes the file again and recomputes every step (cache keys hold `id(source)` and
  `id(modifier)`). Keys by content (source path, size and mtime, step type and settings, the
  mtimes of files a step reads) would keep them; an all-frames step over a long trajectory gains
  the most. Measured 2026-10-25: 108k atoms, CNA + PTM + Coordination, Run all 1.0 s every time.
- Reader index on disk: dump, XDATCAR and extxyz frame offsets are scanned on every open (about
  4-8 s per GB from a warm OS cache; 376 MB dump 1.5 s, 162 MB XDATCAR 1.3 s). Keeping them in
  the cache folder keyed by path, size and mtime would make reopening a large file instant.
- Structure identification at 108k atoms: the periodic kNN is a numba cell list now (0.10-0.22 s
  for k = 12-33, was 0.32-0.71 s with cKDTree + image padding). PTM (0.93 s) and Steinhardt
  (0.95 s) are now dominated by their per-atom kernels (PTM's C++ template matching, the Ylm
  sums), Voronoi (0.72 s) by voro++; those already use every core.
- Ray tracer: fine surface meshes now go through a BVH (isosurfaces 1.4-2.2x
  faster) while atoms, bonds, arrows, glass spheres and polyhedra stay in the uniform grid, where
  a BVH measured 1.2-2x slower; the binned-SAH build is single-threaded (0.23 s for 271k
  triangles). Adaptive sampling (`noise_threshold`, Publication 0.01) saves almost nothing on
  real scenes (2026-10-24, 800 x 600: SrTiO3 with polyhedra 4.0 s full vs 3.96 s adaptive,
  32k-atom Cu 3.2 vs 3.4 s), but not because its error estimate is wrong: an estimate from two
  half-sample images (even vs odd samples) was tried and matches the true error against a
  192-sample render (median 0.010 vs 0.013 at 4 samples), and the foreground really is noisier
  than 0.01 (only 64 % of its pixels are below it at 16 samples; the error falls as 1/sqrt(n),
  0.020 at 8 samples, 0.014 at 16). The background is already cheap, so no stopping rule gains
  much at equal quality (the best variant, 2.45 s, had 9 % more error). What would make
  Publication pictures faster: a denoiser on the ray tracer's output (edge-aware, from its
  normal and depth buffers) or a GPU. The GPU tracers (`ewald.raytrace_gpu`, 2026-10-24: `device="cuda"` and
  `device="opencl"` from one C source) are checked behind a host-compiler shim (CUDA) and on
  pocl's CPU device (OpenCL), both bit-identical to the CPU in double precision, and by
  compiling the CUDA source with NVRTC; their speed on a real GPU is unmeasured (the owner's
  Intel Arc: `python benchmarks/analysis.py --skip-modifiers --device opencl`). The BVH build
  and the uniform grid are still built on the CPU and uploaded.
- GUI startup (2026-10-23, `python benchmarks/startup.py [--cold-jit]`): `MainWindow()` returns
  after 1.2 s, the 3D view is built at 2.2 s and the console at 2.65 s (was 1.65 / 2.7 / 3.15 s).
  The numba warm-up thread now starts 1.5 s after the window shows: started in `MainWindow()`
  it held the GIL while the window built, and with a cold numba cache (fresh install, or after
  an update touched the kernels) that pushed the usable window from 2.9 s to 6 s. The View tab's
  cards are built once instead of twice, and picking and the render frame import single VTK
  modules instead of `import vtk` (0.3 s and ~150 MB on the first click). Left: the pyvistaqt
  import (~0.9 s, blocks the window right after its first paint; on a thread it does not help,
  the GIL serializes it), `import ewald.gui.app` (~0.6 s: numba, scipy, the modifier modules),
  the in-process IPython console (~0.4 s, built when its tab first paints) and peak memory
  (~700 MB with VTK, Qt, numba and IPython). On the owner's Windows laptop (2026-10-26, Intel
  Arc) everything takes 3-4 times as long: import 1.8-2.1 s, `MainWindow()` 5.3-5.8 s, 3D view
  6.0-6.6 s, console 7.7-9.0 s, peak 610-650 MB, the same with an empty numba cache.
  `benchmarks/startup.py --profile` there (2026-10-26): the import is 2.3 s, about 0.6 s of it
  `os.stat` and 0.3 s opening files (many small files on NTFS with Defender scanning), numba
  0.64 s and scipy.spatial 0.54 s; in `MainWindow()` 0.5 s went to folding the view chips (each
  fold width was measured by showing and hiding buttons, 4 ms per `setVisible` there; now
  worked out from size hints, also on every resize), 0.23 s to the WGL OpenGL check, 0.22 s to
  two `QFontDatabase.families()` scans (now one), 0.32 s to the plugin entry-point scan and
  0.25 s to the theme. The 3D view then blocks the window for about 3 s after the first paint
  (pyvistaqt and the first OpenGL context); `--profile` now covers that stretch too.
- Maximizing the window (2026-10-24, `python benchmarks/resize.py`): in the container (llvmpipe,
  cu_void tutorial) it is settled after 0.3-0.7 s, of which one full 3D render is 0.1-0.3 s;
  restoring takes about 0.12 s. The rest is Qt layout and painting plus matplotlib mathtext
  parsing of the plot labels (~40 ms). On the owner's Windows laptop (2026-10-26) the first
  maximize of a session stalls 3.5-4.5 s inside `showMaximized` on the Intel Arc GPU (driver
  31.0.101.5333), later ones 0.2-0.3 s; on the same laptop's GeForce RTX 4050 the first one is
  0.24 s, so the stall is the Intel driver making a QOpenGLWidget window maximized for the first
  time. Other default formats (`--gl-format`) do not change it, and a transparent maximize at
  startup (`--prewarm early`) only moves it there (2.5 s). Kept as is: the owner keeps the
  floating view chips (D62), which need the QOpenGLWidget, and laptops with a discrete GPU
  should run the 3D view on it anyway.
- UI theme switch takes 0.5-1 s in `QApplication.setStyleSheet` (one app-wide QSS).
- Playback: plots whose data does not change are not redrawn; per-frame histograms and
  tables update their bars or lines in place (20k-atom histogram 157 -> 65 ms per frame,
  g(r) 148 -> 47 ms). What is left is mostly matplotlib's Agg draw; blitting would need fixed
  axes. Contour maps, NEB, phonon and DOS/bands plots still redraw fully.

GUI

- UI feedback audit (2026-10-20, project files `ewald-review/ui-feedback-audit.md`): every
  finding is fixed. Left over: cancelling an open stops waiting but not the read itself (readers
  have no cancel hook), so a cancelled SSH connect runs out its timeout in the background.
- Panel layout (phase 1 landed: `gui/workbench.py`, `gui/layout_model.py`; plan in project
  files `design/panel-layout.md`): phase 2 opens a notebook view cell's plot large in the
  editor area ("Open large" or drag the cell header, a Bring back placeholder in the cell, the
  layout saved in the notebook metadata) and makes the Render workspace a tab there; phase 3
  makes the Notebook, Display, Ask AI, Plugins and Remotes side bars views too, adds Move
  to new window (multi-monitor), named layouts, drop bands along an area's outer edge for
  full-length splits, a second 3D view (another camera) and a `@view` plugin API. Not yet tried
  on a real `ssh -X` display.
- Plot style file (`<config dir>/plots/user.mplstyle`, Preferences › Plots form + Open as text, shared
  by the Plots panel, cells, kernels and the console): a notebook's own `plotstyle.use(...)` does not
  reach that notebook's Plots panel tabs and view cells (they follow the file only); more presets
  as a `@plot_style` plugin kind (today: five built-in `plotstyle.PRESETS`, or drop
  `<name>.mplstyle` files into the folder and `plotstyle.use(name)`); an `.ewd` does not carry the style file,
  so a shared notebook draws in the receiver's style (bundling it needs a trust check: matplotlib
  evaluates `axes.prop_cycle`); the form cannot pick installed fonts from a list.
- Interactive plots (zoom, pan, energy window, hover and click-to-frame on every plot exist):
  drag on an axis to zoom that axis only and pinch zoom on touchpads; hover readout of the
  phonon mode (q-point, band, frequency) and band index under the cursor; hovering a per-frame
  plot previews that frame on the timeline; linked zoom between plots (a DOS beside the bands
  sharing the energy axis, once the panel layout can place plots side by side); a scatter view
  of two particle properties where a brushed box selects atoms; the `show()` script path draws
  DOS/bands like the GUI (it plots the raw table columns now).
- AI chat side bar (`gui/ask_panel.py`, activity bar): chats are per notebook tab and saved only in
  `.ewd` (not `.py`/`.ipynb`), and a chat changes no "unsaved" mark, so closing without saving
  loses it; a saved chat reaches a new model as a text transcript (no tool results or images);
  a streamed answer is drawn as Markdown a few times a second, so an unclosed code fence shows as
  text until it closes; `ewald ask` and the MCP server do not stream; show tool images inline, insert as a step cell when the block is one `pipe.add(...)`,
  edit or regenerate a sent message. Cells the AI changed are linked from
  its answer (a click scrolls to and lights up the cell); the cells could also stay marked in
  the notebook and show a diff to accept or undo, and a cell number the model writes in its text
  ("cell 3") is not a link (only the tools' results are); `ewald ask` and the MCP
  server could take a notebook (`--notebook analysis.py`) and save it back.
- AI tools only through cells (`llm/tools.py` `CELL_WRITERS`): after a plugin tool, steps and
  view settings changed without a cell are undone (`Toolset._undo_outside_cells`), but other
  state a plugin touches by hand (a modifier's attributes, `pipe.source`, files) is not checked.
  Left by design (docs/user/ai-permissions.md, "Gaps in the rule") (D40): `render_image(filename=...)`
  replaces a work-dir file, `set_frame` leaves no cell, and a Forge plugin a notebook uses is
  not part of the notebook (an `.ewd` could carry it in plugins/).
  A failed generated cell leaves two undo entries in the GUI (insert, then removal). After a second
  `pipe = load(...)` cell, the step cells above it still hold the old pipeline's modifiers, and
  `Notebook.sync` (the GUI's refresh) deletes them.
- AI edits cells rather than adding copies (D68): `add_modifier`/`insert_cell` refuse a cell that
  repeats one (same modifier step, plot of the same table, view tab, same code) unless
  `again=true`, and an answer's code block replaces the one cell doing the same job. Still open,
  seen with real models on 2026-10-25: one `CoordinationAnalysis` cutoff sets both the
  coordination and the RDF range, so "show the RDF to 6 Å" either changes the coordination
  colors or makes the model write its own RDF cell (a separate RDF range, or an RDF step of its
  own, would remove the conflict); `show()` has no `title`, so "give the plot a title" turns the
  live plot cell into a static `figure()` cell; the guard compares jobs by modifier name and
  `output` only, so plugin tools that add a step each time (`compute_plumed_cvs`) get refused on
  the second call and the model has to edit the cell; the GUI's 3D view coloring and picks are
  not in the cell-numbered notebook context.
- AI and the Python API: the model reads `python_api` (about 60 KB, roughly 15k tokens, since the
  modifier outputs and recipes were added) on demand rather than in the system prompt. The first
  real-model runs (2026-10-23, Claude through `make_ai_demo.py --live`, three scenes,
  three takes) read it every time and got the numbers right (barrier, hopping atom, sp/sp2/sp3
  fractions, D*(Li) 1.3-1.45 × 10⁻⁵ cm²/s against kinisi's 1.23(10)); `ewald eval` with Opus,
  Sonnet and Haiku 5.5 (2026-10-24, docs/developer/ai-eval.md) passed all 53 tasks, and the `study`
  tasks after public PR #73. `ewald ask` and
  the MCP server still do not get the GUI's notebook note (`llm/notebook_prompt.py` has it Qt-free
  now). What the model tripped over in those runs:
  property expressions in OVITO's C syntax (`a ? b : c`, `&&`, `||`) are syntax errors, and
  `A == 1 | B == 2` parses with Python precedence; `suggest_cutoff` picks 0.84 Å on a melt-quench
  first frame with close contacts; `show(pipe, table="time-series")` also plots the Timestep column,
  so the model drew its own matplotlib plot; it adds a second `GenerateTrajectoryLines` instead of
  editing the first; it ignores "keep the answer short" (answers of 300-500 words with tables).
- `PythonScript` step cells: `Modifier.__repr__` uses `name`, which PythonScript sets to its title,
  so `step_source` writes `pipe.add(Python script(...))`, which does not parse.
- UI style (flat buttons, fields, checkboxes and tab close icons in `gui/theme.py`): a few widgets
  still set inline colors (color swatch borders, field error borders), so they keep the old
  palette until a restart after a theme switch; move them to QSS object names. Open windows
  that draw theme colors themselves follow a switch through a `retheme()` method
  (`theme.retheme_windows`; Preferences has one); other dialogs rely on the app style sheet.
- Notebook layout: the viewport could follow the current notebook even when it has no
  pipeline yet (now it keeps the last one); image/figure cells; inline Ask AI suggestions
  as diffs; a snapshot cell that updates when the camera of its view changes (now it is code);
  raw cells; Jupyter's Shift+Ctrl+- (split a cell at the cursor) and Shift+L (line numbers);
  selections are a contiguous range (no Ctrl+click to pick separate cells).
- 3D view as code (`pipe.view`, the `view(pipe, ...)` cell, Claude's `set_view`): settings the
  view does not have yet (picks as code, the viewport axes widget; measurement labels in the
  legacy OpenGL 1.x viewport, and lines that show through atoms; audit in the project files `ewald-review/view-elements-audit.md`);
  the View tab cards (Atoms, Colors, Types, Bonds, Polyhedra, Surfaces, Cell, Labels, Camera, Elements) could take the visual
  element names `pipe.view.elements` uses (Particles, Bonds, Simulation cell, ...); label size,
  color and types and the selection color are code only (the cards show on/off and the text); the camera reaches the view cell
  only through Insert › View settings with the camera (not on every rotation); the Render page
  keeps its own per-notebook state in the `.py` header next to the view cell's picture settings
  (one place for both); in a kernel notebook a viewport edit reaches the kernel's `pipe.view`
  only when the view cell runs again.
- Colors (`ewald.colors`, every form in every color argument): the color swatches of Preferences
  and of the Colors card open only the color dialog (no typed `'tab:blue'` or `'rgb(...)'` there,
  as step arguments take); alpha in a color is read but atoms, bonds and lines ignore it
  (`transparency` is the way), and `"C0"`-style cycle names follow the current plot style.
- View tabs (`view(pipe, tab=...)`, more Structure tabs): picks, mouse measurements, bond editing
  and linked plot selections work in the main Structure view only (a tab's pick shows its label);
  the View tab cards, the Render page and Insert › Snapshot follow the main view; a saved window
  layout forgets where a structure tab was (it opens next to the Structure tab again); a tab's
  camera reaches its cell only through its right click › Write view to notebook; tabs follow the
  one timeline (a tab's `frame` setting only affects `pipe.render(tab=...)` and snapshots).
- Atom labels (`view(pipe, labels=...)`): not drawn on the OpenGL 1.x viewport (`ssh -X`), not on
  boundary or replicated images, and not hidden behind nearer atoms (they always show, as in VESTA).
- Render frame (the picture's region on the 3D view, `view(pipe, region=...)`): not drawn on
  the OpenGL 1.x viewport (`ssh -X`); in perspective, a frame off the view center moves the
  camera sideways instead of an off-axis projection (exact at the focal distance only); the
  Overlay elements and text label are not previewed inside the frame; Claude's `set_view` can set
  `region` but no tool tells it the frame on screen.
- Python steps per frame vs once (decorators and `# %% [each frame]` / `[all frames]` cells
  landed): `all_frames` start/stop/step arguments; Insert menu entries for the two cell kinds; in
  a kernel notebook, switching a cell back to *Once* takes its step out only when the cell runs
  again. A view cell's error output overlaps its plot (seen 2026-10-20).
- Live `show(pipe, ...)` outside a view cell: zooming does not write `xlim=`/`ylim=` back (only a
  cell that is just the call does), and `frame=` is ignored by the live plot (it follows the
  current frame, as view cells do).
- Plots in the Plots panel (`views.in_panel`; linked or per-frame plots of cells get their own tab
  next to *Plots*, a thumbnail stays under the cell): a tab is titled by its table or histogram
  only, so two plots of one table look alike; the tabs are not kept in the saved panel layout
  (they come back when the cells run, but not where they were dragged); the thumbnail is the image
  of the last run, not the frame on screen; a plot that `in_panel` sends to the panel cannot be
  told "keep under the cell" for a whole notebook at once (only per call with `panel=False`).
- Figures drawn by Python steps (`pipeline.StepFigure`, `data.attributes["Figure: <step>"]`):
  drawn in matplotlib's default style, not the notebook's plot style; a PNG per frame, not a
  live interactive plot (no zoom or linked selection); moving one between the panel and the cell
  is not saved; steps that run in a notebook kernel do not send their figures back yet.
- Notebook kernels (`ewald/kernel.py`): step headers (results, checkbox) for steps a kernel cell adds with code the
  app cannot import (a class only the kernel's Python has stays as code). Outputs of a kernel
  cell (text, progress bars) arrive when the cell ends; in-process cells show them while they
  run. Stream them from the kernel's iopub messages instead of the bridge's final JSON.
- IPython cells (`ewald/shell.py`): HTML outputs use Qt's rich text, so JavaScript outputs
  (ipywidgets, plotly, bokeh) show their text; a QtWebEngine view (or an image fallback from
  `_repr_png_`) would draw them. `%matplotlib qt` / `%gui qt` inside a cell (cells run on a
  worker thread) is not guarded yet. `input()` in a cell has no prompt in the app.
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
  view instead of the Ask AI prompt. Updating or uninstalling a pip plugin whose code is
  loaded still needs a restart (the side bar now says so and offers Restart now); a VS
  Code-style "reload window" without a new process is not possible for imported modules.
  Restart now reopens saved notebooks only: untitled ones, the layout of plot tabs and the
  current frame are not carried over.
  Updates compare versions only: an index entry that pins a new commit of a `pip:git+...`
  package without raising its version is not offered as an update (the installed commit is in
  the distribution's `direct_url.json` if that is ever wanted).
- UI themes, VS Code/Obsidian-like switching: no Color theme picker with live preview in the
  Ctrl+K palette, no preview images of themes in the Plugins side bar, no light/dark variants in
  one theme that follow the OS, no icon themes (the side bar's *Use theme* button is there).
- UI themes: prompts already printed in the Python console keep the colors of the theme they
  were printed in (new input, syntax colors and the background switch at once).
- UI theme style keys (`font`, `mono_font`, `radius`, `density`, `icon_stroke`, `qss`): open code
  editors and the console keep their mono font until reopened; `density` scales QSS padding only,
  not layout margins set in code; `qss` object names are not a stable plugin API yet.
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
  changes with the same atoms that replace only the point coordinates. Measured 2026-10-26: the
  bond search of a new frame now runs on the compute worker, and what is left on the GUI thread
  for 1M atoms without bonds is 0.08 s of `set_data` (scene and actors) before the render, so
  reusing actors would save less than that; with bonds the half-bond geometry and stick input are
  numba kernels since phase 2 of the [1M-atom plan](developer/viewport-performance.md).
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
- Floating view chips (D62): the camera fit (`zoom_fit`) still uses the whole view, so in a small
  view a structure can sit under a chip; fit to the area the chips leave.
- Impostor spheres: exact ray-sphere depth in perspective views (now approximate near the
  screen edges). The `light` setting moves the key light; a fill/rim light and
  shadows in the OpenGL view are not settable.
- A per-element bond table like VESTA's (cation-anion only, measured lengths) instead of the
  covalent-radius sum with metal-metal pairs left out.
- Bond rules: VESTA's recursive molecule search per rule (now "+ atoms" adds one shell of
  partners); rules by atom label or a selection, not only by type; the bond count per rule in
  the Bonds card.
- Replicated cell display: a quick replicate control in the 3D view toolbar or the redesigned
  View tab (now Display panel, View menu, right-click menu and Ctrl+K).
- Boundary (VESTA): repeat isosurfaces and grid slices over the boundary range too (now only
  atoms, bonds and coordination polyhedra are repeated), VESTA's recursive "search molecules"
  mode, and keep the images while the coarse viewport level of detail is shown (they disappear
  during rotation of systems above the interactive atom limit).
- Bonds are impostors in the viewport (`render.add_sticks`, VTK's stick mapper), and the
  automatic-bonds limit is 250 000 atoms (was 100 000): at 256k atoms the bond search and sticks
  take 0.9 s per frame, at 500k 3.9 s, most of it the bond search. Left: the flat (ASE) style
  still draws outlined tube meshes, impostor sticks have flat ends (a bond thicker than its atom
  shows them), and the legacy OpenGL 1.x viewport (`ssh -X`) keeps its own bonds; checked under
  llvmpipe only, not yet on a real Windows, macOS or NVIDIA driver.

Runs (`ewald/runinfo.py`, `ewald/runs.py`; manual page docs/user/runs.md)

- `run_info` / `check_run` on real outputs: VASP 5/6 MD with `NBLOCK` > 1 (whether XDATCAR numbers
  its configurations by frame or by ionic step; both are handled, only the first is tested),
  MDALGO 3 NPT, real LAMMPS logs with `variable`-substituted commands, `include` files and several
  `run`s with different fixes (the last thermostat wins now), and `thermo_modify line yaml`.
- Quantum ESPRESSO, CP2K and ABINIT inputs (`run_info` knows VASP and LAMMPS only).
- Use `check.equilibrated` as the default `start=` of MSD, time averaging and the transport
  modifiers (now the user passes it); the transport modifiers still default
  `time_per_frame` to 0 (frames) instead of the run's value.
- Time units on plot axes: the MSD/VACF `Time` column is in ps when `time_per_frame` is known, but
  the table does not say so, so the axis reads "Time".
- Block-averaged error bars for every per-run number (D, temperature, energy), and an Arrhenius
  fit weighted by them (now unweighted, with the fit's own standard error).
- `show(runs, ...)` as a live GUI view cell: a view cell binds to `pipe`, so the overlaid runs show as
  the `RunTables` picture under the cell instead of an interactive plot; run comparison in the
  Plots tab.
- A run folder index as the way into `load_runs`: `ewald.workflow_folder` now scans a project
  folder (kind, state, last energy); `load_runs(workflow)` taking its MD steps is still open.
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

Phonons (built in: `phonon.py`, `io/phonon.py`, `modifiers/phonon.py`; phonopy force constants in the
ewald-phonopy plugin, frssp/ewald-plugins `packages/ewald-phonopy`)

- Phonon DOS / projected DOS from phonopy `total_dos.dat`/`projected_dos.dat` in the Plots tab.
- Non-diagonal supercells for a mode (`supercell` as a 3x3 matrix), and several modes
  superposed (phonopy `--modulation` takes a list).
- Export a modulated structure (frame 0 as POSCAR) from the GUI; irreducible-representation
  labels of the modes (phonopy `--irreps`).
- A trace of each atom's path (an ellipse for a complex q != 0 mode) as an alternative to the
  per-frame arrows of `PhononMode(arrow_mode=...)`.
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
- NEB: warn when an endpoint OUTCAR lists the atoms in another order than the band (the Fe
  vacancy example's 04/OUTCAR does; the reader takes POSCAR atoms, so only other tools are hit),
  and match atoms by position when images carry no identifiers.
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
- Large vector figures (D67): the per-shape costs behind the warning were measured on fcc Cu in
  the cloud container; a mixed figure (atoms over a size limit as one embedded raster layer, text
  and overlays still vector) would keep big structures editable; the 10 s / 10 MB limits could
  become a preference.
- Vector arrows: their own color legend when a `ColorCoding` already owns the legend; arrows in
  the coarse (moving) viewport scene, as lines; in SVG/PDF, hide the part of an arrow inside its
  atom exactly (now the shaft is split at the atom and painted by depth).

Visualization plugins (later phases of the Scene IR plan)

- GUI: the built-in elements (Particles, Bonds, Cell, ...) as rows with forms generated from
  their `VisualElement` parameters, replacing the hand-written style controls.
- OpenGL and the ray tracer ignore `Cylinders.alpha`; see-through bonds need it.
- Vector files draw bond caps (D50) as a flat ellipse or disc without the stick's highlight stripe,
  and a round joint where several sticks meet can show the stick outlines crossing it.
- Renderer consistency left from the D51 audit (project files `ewald-visual-audit/`): vector files
  show seams between the triangles of color-mapped slices and draw isosurfaces faceted; a
  see-through isosurface casts a fully dark shadow in the ray tracer; an impostor atom inside a
  see-through isosurface comes out washed white; the material tone still differs (ray-traced
  atoms lighter, vector highlights as rings); the ray tracer's flat style also flattens surfaces
  and arrows; the OpenGL 1.x viewport blends see-through surfaces unsorted, so they look lighter.
- Color bars: the legacy OpenGL 1.x viewport (ssh -X) draws the color maps but no color bar card
  (`gui/viewport.py` `_draw_colorbars` has the image to draw), so it cannot fold them either; the card can cover an `Attributes`
  overlay placed top right.
- Trajectory lines:
  a trail ends at the last traced frame before the current one, not at the atom's current
  position between traced frames (step > 1); color by a per-frame particle property along the
  path (e.g. coordination at each frame) needs that property in the trajectory-lines table.
- Material presets (`@material`: roughness, metallic, glass, outline/toon) mapped onto each
  renderer (VTK PBR, ray tracer shading, vector flat/gradient/outline), the rest of phase 1.
- Overlay elements show in rendered images and the Render page preview, not in the interactive
  viewport.
- Phase 2: renderer backend registry, so a plugin can add a renderer that consumes the Scene IR.
- Phase 3 and later: glTF/HTML export from the IR, animated visuals, Claude-written visual
  elements (a Visual Forge, like Modifier Forge).

Plugins and AI

- AI eval suite (`llm/evals.py`, `llm/eval_tasks.py`, `llm/eval_data.py`): the 53 single-step
  tasks saturated on 2026-10-24, and the 11 `study` tasks (whole analyses, D70, public PR #73) did
  after the fixes they led to (Opus 22/22, Sonnet and Haiku 21/22 over two runs). Harder ones are
  open: several traps in one series, a run that passes check_run but is too short to give a D, the
  paragraph's writing and the honesty of its error bars judged by a second model (an LLM judge),
  conductivity and Haven ratio, other systems than Li hopping (oxygen vacancies, proton transport,
  melting), studies that start from a folder rather than from one opened run, and from the GUI side
  (picks, Ask about selection, the 3D view's coloring the GUI sends with each question). Also open:
  more than two repeats per model, reproduction in a separate kernel (`ewald.kernel`) rather than in
  process, plugin tasks (`@eval_task` from plugins such as ewald-phonopy) and a results page that
  tracks scores over commits.

- AI providers (`llm/providers.py`): the OpenAI-compatible provider never sends images (no
  per-server vision switch yet); Effort reaches only Claude and OpenAI reasoning models; one
  `llm.model` setting for all providers (switching the provider in Preferences clears it);
  streaming falls back by itself only for Chat Completions servers (a Claude gateway that cannot
  stream needs Preferences › AI › Stream answers off); a native Gemini or Bedrock/Vertex client would need a `Chat`
  subclass (today Gemini goes through its OpenAI-compatible endpoint). Keys are per provider, so two
  saved connections of one provider with different keys share one keychain entry. Preferences › AI › Extra headers is a
  one-line field, so a long header list is hard to read and easy to cut short (it happened with a
  real gateway, 2026-10-23); a multi-line editor would help.
- Plugin manifests, phase 2 (from the VS Code comparison): contributions declared in
  `ewald-plugin.toml` (`[[readers]]`, `[[modifiers]]`, ... with `python = "module:object"`)
  registered lazily and imported on first use (`@panel` widgets are already built on first open,
  but the plugin module is still imported at start); a `commands` contribution point (commands in
  the palette, as notebook functions and as AI tools). Settings pages (`@settings_page`) and side
  bars (`@panel`) are registered in code, not declared in the manifest.
- GUI plugins (`@panel`, `@menu_item`, `gui/window_plugins.py`; `@settings_page` pages in
  Preferences): reloading plugins rebuilds every open plugin panel, losing its state; no entries
  in the Data inspector, notebook tab or Display panel menus yet, and a plugin cannot change or
  remove built-in entries; a bottom tab cannot be placed in the editor area beside the 3D view
  from the plugin (the user can drag it there). A plugin's
  Preferences page has no changed-dot or per-row Reset, and Edit › Plugin settings and the
  Plugins side bar's Settings still open the separate dialog rather than that page.
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
  copy `load("aiida:...")`), PDOS/projected bands
  from aiida-quantumespresso, NEB and aiida-lammps outputs, streaming very long remote
  trajectories frame by frame instead of one bundle, export of analysis results back as new
  nodes (opt-in, never default). Not yet tried against a real remote AiiDA host.
- OPTIMADE (built in since 2026-10-28, `optimade:DB/ID`): its database table was written without reaching
  the servers (the cloud container has no route to them); finding every database from the
  providers.optimade.org index instead of the fixed table; more pages in the side bar; the
  OPTIMADE `assemblies` (alternative disorder configurations) are drawn as all sites.
- Materials Project (ewald-mp, `mp:mp-149/...`): relaxation trajectories without mp-api (MP
  serves them only as Delta/parquet tables); an mp-api button on its settings page for the
  standalone app, where there is no shell to `pip install mp-api` in; phonon bands and DOS (`get_phonon_bandstructure_by_material_id`
  into `PhononMode`), elastic tensors and other summary fields as attributes, projected band
  structures (`load_projections=True`), more pages in the side bar; other `schemes=` sources.

- LLM plugin ideas: Paper-to-modifier (Forge from a paper's definition, checked against its
  numbers), phase identification, campaign report over many runs (`check_run` and `load_runs`
  are the deterministic base for both; the AI explains the findings).

Docs

- Example files: list the examples (and the downloadable paper datasets) on an empty viewport at
  first start; a GUI entry for `examples.fetch` (File › Open example › From a paper).
- Examples run (2026-10-20, /mnt/project-files/ewald-review/examples-test.md; rerun 2026-10-23:
  every tutorial, paper script and demo.py passes): `llzo_arrhenius.py` needs the `formats` extra
  (MDAnalysis) that the README's `[gui]` install leaves out (it says so); spglib is core now.
  Plugin example files (`ewald_phonopy.EXAMPLES`, Cu3Au_band.yaml) are not under File › Open
  example. The Plugins side bar's default catalog (raw.githubusercontent.com of the private
  frssp/ewald-plugins) answers 404 without a GitHub token, so first-party plugins install only by pip.
- Paper reproductions: StructureFactor `method="rdf"` holds every pair inside the cutoff at once,
  so 100,000 atoms with a 30 A cutoff run out of memory (the a-Si script uses 20 A); histogram in
  chunks. kinisi's covariance-weighted D for the LLZO runs is 0.6x a straight-line fit of the same
  MSD; find out why. Candidates found but not added: TiOF2 PDF against experiment (needs an
  X-ray weighted G(r)), the a-Si 216-atom HSE06 models (vldgroup/papers-defect-free-a-si), LGPS
  site maps (hacortesp/CrySF), MoSe2 phonons (Ionizing/adom.202403069); no license stated on the
  last three. Also checked and not added (2026-10-20): VTST NEB runs in
  drinwater/Nudged-Elastic-Band-Tutorial (14 MB OUTCAR per image) and umichyujl/data_n_type_doping_AlN
  (HSE NEB of AlN donors, images without OUTCARs), and the LAMMPS NEB logs of shortlab/2016Cr2O3
  (Cr2O3 vacancy paths, no per-replica dumps); none states a license. Still missing: a clearly
  licensed LAMMPS dataset with multi-file dumps or a temperature series for `load_runs` +
  `arrhenius` (the LLZO one is GROMACS XTC), and a real `ssh` NEB folder for the remote example.
  Most other paper data sits on Zenodo or Materials Cloud, unreachable from the cloud
  container, so `examples.fetch` only knows GitHub-hosted files.
- Host the manual (the repository is private; CI uploads the built site as the `docs-site`
  artifact).
- More tutorials in the notebook style (examples/notebooks, tested by tests/test_tutorials.py):
  Ask AI and Modifier Forge on a real question, a plugin from the Plugins side bar, a volumetric
  file (CHGCAR isosurface), opening a run on a cluster over SSH, and PLUMED/phonopy once they are
  plugins. Features no tutorial shows yet (review 2026-10-23): trajectory lines with
  `trajectory_trail` and their color bar (the Cu_md example moves too little to show them; needs
  a diffusing example file), a NEB folder (`NebEnergyProfile`, `OverlayFrames`; only the
  fe_vacancy_neb paper script), `load([...], join=...)` restart segments, `workflow_table`,
  `run_info`/`check_run`/`load_runs`, `[each frame]`/`[all frames]` cell tags (the tutorials use
  the decorators), VASP DOS/bands. The remaining manual pages (Python scripting, Rendering figures, Modifiers) still show
  `pipe.modifiers.append` and `import_file`; switch their examples to `load`/`pipe.add` when they
  are next edited.
- GUI screenshots in docs/images are made by docs/images/make_screenshots.py; Help › About and the
  Render workspace have none yet, and a GIF of linked selection (drag on a histogram) would show
  what a still cannot. docs/images/ai-demo.gif (Ask AI on Li6PS5Cl AIMD, amorphous carbon MD and a Fe NEB) is recorded by
  docs/images/make_ai_demo.py, with a scripted model by default or a real one with `--live`
  (key from `EWALD_DEMO_ANTHROPIC_KEY`); re-record it when the chat side bar changes visibly. A
  live take differs every time, so look at it before committing it.
- Cell links in the AI chat count cells by position ("cell 3", markdown cells included), while a
  cell's gutter shows its run count as in Jupyter; after the AI edits and reruns a cell the two
  numbers differ (seen in the ai-demo.gif recording). Consider showing positions in the gutter or
  wording the links differently.

## Ideas backlog

From the feature-ideas review (2026-10-19), after the four in progress above.

- P1: Jupyter widget: `pipe.show()` with a rotatable viewport and frame slider in a notebook
  (PyVista trame backend).
- P1: Linked selection, rest: picking atoms plots their property time series over all frames;
  scatter plots of two properties (brushing a box selects atoms); keep a pair, bond or angle
  range as a modifier (only property ranges become an `ExpressionSelection` now).
- P2: glTF, interactive HTML and Blender export.
- P3: A Metal viewport on the Mac through VTK's WebGPU backend (Dawn maps it to Metal, Vulkan
  or Direct3D 12). The VTK viewport draws through OpenGL 4.1 on a Mac now, which macOS runs on
  top of Metal; the Metal option in Compute › Device is only for the ray tracer. Wait until
  VTK's WebGPU backend leaves experimental and supports what the viewport uses (impostors,
  depth peeling, picking), then add it as a `viewport.backend` choice.
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

- The frozen Windows app from `windows-release.yml` (setup.exe, portable zip) on the owner's
  Windows PC: install, open a VASP/LAMMPS file, render, quit without a crash dialog. Then GPU
  compute from Preferences › Compute › Optional components › Install (D80): pip finishes in the
  Plugins side bar, Compute › Device lists the RTX under CUDA without a restart, and
  `device="cuda"` runs. CI installs it on a GPU-less runner only (`smoke_gpu.py`) and runs the app
  on Mesa software OpenGL, where the CLI smoke script crashes on process exit (ignored there).
- The online installer (D79) on the owner's Windows PC behind the company network: setup downloads
  ewald and its packages (progress lines under the bar), the Start menu entry opens the app without
  a console window, the Plugins side bar installs ewald-mp into that Python, `ewald command prompt`
  has `python`/`pip` on PATH, uninstall removes the folder; once with *NVIDIA GPU support*
  (`device="cuda"` on the RTX through `cupy-cuda12x[ctk]`, unverified in CI).
- Plugins side bar in the Windows exe (D76): install ewald-mp and ewald-plumed, restart, use
  them; once with git on PATH and once without (GitHub zip); update and remove.
- Simple (ASE) atoms on the owner's Windows PC (200 %): round discs in the OpenGL 1.x viewport
  (36-segment meshes now) and in the normal viewport (impostors); View › Atoms › *Sphere detail*
  changes the legacy viewport's discs; applying Preferences keeps the style picked in the view.
- The running-cell marks on the owner's Windows PC at 200 % scaling: the elapsed time fits
  under the stop button, the ring turns smoothly, the tab's play icon shows.
- Plugins side bar on the owner's Windows environment (2026-10-26): installing, updating and
  removing a pip plugin (ewald-mp) keeps the window responsive (it froze for seconds when the
  package metadata was read on the GUI thread), no console window flashes, and *Update to 0.1.x*
  shows once a newer version is in the index.
- The large vector figure estimate (D67) on the owner's Windows PC and Mac: render a few thousand
  bonded atoms to SVG and PDF and compare with `python benchmarks/vector_size.py`; the *Large
  vector figure* dialog at 200 % scaling.

- The floating view chips (D62) on macOS (fine on the owner's Windows laptop at 200 %,
  2026-10-26): the see-through chips composite over the VTK view without flicker, and fold before
  they overlap when the Structure view is split or narrow.
- OPTIMADE against the real servers (the container reaches none of them): `ewald optimade
  search Si --db <name>` for every name in `ewald optimade databases`,
  then `load("optimade:mp/mp-149")`, `load("optimade:cod/1000041")` and the OPTIMADE side bar,
  on Windows behind the owner's network.
- ewald-mp 0.2 with a real API key (the tests use a local stand-in for MP's REST API; the
  container reaches MP's open data bucket but not the API): in the Windows app install
  Materials Project from the Plugins side bar and check that only ewald-mp gets installed, set the
  key in Preferences › Materials Project, then `python -m ewald_mp search Si`, the side bar,
  `load("mp:mp-149")`, `conventional=True`, `/bands`, `/dos`, `/chgcar`, and that `/relax` says
  to install mp-api; with mp-api installed, `/relax` once. Also that the keychain keeps the key.
- Laptops with two GPUs (owner's Windows laptop: Intel Arc + GeForce RTX 4050, 2026-10-26): the
  Windows high-performance choice moves the 3D view to the RTX (first maximize 0.2 s instead of
  ~4 s on Arc) and the CUDA ray tracer runs there (numbers in docs/user/gpu.md; CuPy needs
  `cupy-cuda12x[ctk]`). Modifiers on CUDA measured there too (gpu.md). Still to check: Preferences › GPU
  for the 3D view / `ewald gpu --prefer` themselves (the owner used Windows Settings),
  `benchmarks/neighbors.py`, and Linux PRIME offload on a real hybrid laptop.
- CommonNeighborAnalysis is 3.5x slower on CUDA than on the CPU at 108k atoms (402 vs 115 ms on
  an RTX 4050, 2026-10-26), ClusterAnalysis slightly slower (89 vs 68 ms): profile the xp path
  (many small kernel launches or host round trips) and give CNA one CUDA kernel or keep it on the
  CPU under `device="auto"`.
- The OpenCL ray tracer runs on the owner's Intel Arc on Windows (2026-10-26,
  `benchmarks/analysis.py --skip-modifiers --device opencl`): SrTiO3 + TiO6 polyhedra 1600x1200
  draft 0.17 s, standard 0.30 s, Publication 1.13 s (4.80 s on the CPU); trace only at 800x600
  Publication, gyroid 0.73 s, fcc Cu 108k 0.20 s, Cu slab 0.07 s. pyopencl prints a
  CompilerWarning (non-empty build log) on the first build there; see what it says with
  `PYOPENCL_COMPILER_OUTPUT=1`. Still unchecked: AMD GPUs and the GUI Render page on Arc (the CUDA
  tracer runs on the owner's RTX 4050, see above).
- The Metal ray tracer runs on the owner's M2 MacBook Air (2026-10-24; first picture of a session
  +4 s for the kernel build). `benchmarks/analysis.py --skip-modifiers`, Metal vs CPU: SrTiO3 +
  TiO6 polyhedra 1600x1200 Publication 7.14 vs 7.12 s, gyroid 4.76 vs 4.34 s, fcc Cu 108k
  0.85 vs 1.64 s, Cu slab 0.20 vs 0.34 s. Atoms gain about 2x; see-through polyhedra and the
  fine mesh do not (divergent transparency layers, the BVH stack). Apple's OpenCL took 2.4 s
  for a 400x300 Publication picture the CPU draws in 0.4-0.5 s (19.5 s before the 32-bit item
  codes), and Metal took 0.7 s, then 1.6 s and 8.7 s for the same picture in later runs. Plugged in it took a steady 0.6 s: the stalls were the GPU's
  battery power state, and the tracers now queue all launches of a pass and wait once (a GPU
  idle between launches clocks down). At 1600x1200 Publication Metal took 2.6 s against 9.0 s
  for 108k fcc Cu atoms and 2.5 s against 11.0 s for 500k, so `auto` on a Mac is Metal again. Still slow for a GPU: divergent transparency layers and
  AO rays in one kernel per pixel, the per-thread BVH stack (a wavefront tracer would fix both). Retina, the
  GUI Render page and movies on Metal are unchecked.
- `no display: DISPLAY is not set` on a real Linux machine: the message should name the cause
  (SSH login without `-X`, tmux/screen or `sudo` on the desktop, a Wayland session) and the
  command that fixes it (`gui.no_display_hint`); tested with simulated environments only.
- Notebook cells on Windows at 125-200 % display scaling and on macOS: the code box ends below
  its last line and the output area (rule on the left, monospaced text) starts under it, also
  after changing the font size in Preferences; a traceback or a long `print` wraps inside the
  cell. Checked with `QT_SCALE_FACTOR=2` on Linux only. A print longer than 50 lines scrolls in
  a 20-line box that ends on a whole line; a click on the rule beside it (8 px hit area) or
  Shift+O shows every line, a double-click hides it.
- Preferences › Plots › Open as text on Windows and macOS: a `.mplstyle` file has no program
  associated on a fresh system, so Windows falls back to Notepad and macOS uses `open -t` (the
  default text editor); EWALD must re-read the file when the editor saves (watched with
  QFileSystemWatcher, folder included for editors that replace the file). Tested on Linux only.
- Structure view right-click menu on a real Mac (two-finger click, Control+click) and Windows:
  after closing the menu, moving the mouse must not zoom (fixed for the VTK and OpenGL 1.x views,
  tested with simulated events under xvfb only).
- Notebook scroll while the AI chat writes cells, on Windows and macOS: with the keyboard on a
  notebook cell, an AI edit must not scroll the notebook (the fix moves the focus out of the cell
  widgets before they are rebuilt; xvfb has no window manager, so focus there is not the real
  thing), and Follow the AI's cells (follow, end, never) and the "AI is on cell N" pill should
  behave as on Linux.
- AI workflow search on a real sim-agent server with the site's model: "open the TiN
  calculation that finished yesterday" should call `list_workflows` with `formula`, `status` and
  `since`/`until`, then `open_workflow`; days are the PC's local days against sim-agent's
  creation time.
- GUI startup on the owner's Windows machine: measured and profiled (see Performance); next
  `python benchmarks/startup.py --profile` once more, for the 3 s the 3D view blocks the window
  after its first paint, on the Arc and on the RTX. Windows Defender scanning freshly installed packages can make the
  first start after an update much slower than later ones (the "Preparing EWALD" first-run
  notice, `gui/first_run.py`, shows during that start).
- First-run notice on a real Mac and Windows machine: after `uv pip install -e .` the next
  `ewald gui` should show "Preparing EWALD" within about a second (it needs only PySide6), keep
  it on screen until the main window appears, and not show it on the start after.
- Simple (ASE) style on a real GPU (the owner's Windows machine): rotating a few thousand bonded
  atoms should now be as fluid as Ball & stick. The bond outline became a static inverted hull
  (front faces culled) instead of `vtkPolyDataSilhouette`, which re-ran on the CPU on every camera
  move (about 100 ms per frame at 4000 atoms); the container only has llvmpipe.
- IPython cells on a real Windows and Mac: `!dir` / `!ls` (IPython runs `!commands` with
  cmd.exe on Windows, a pty on macOS/Linux), `%time`, progress bars while a cell runs, and the
  console sharing the notebook's namespace; in the frozen Windows app and EWALD.app too, where
  `!pip` has no Python at `sys.executable`.
- Windows checks of 2026-10-23: the OpenGL check in a VM and a Remote Desktop session without
  GPU (the Legacy GL viewport with `opengl32sw.dll` should come up, with the note on stderr);
  saving a notebook open in Notepad++ or synced by OneDrive (the message names the program); the
  `icacls` ACL on `credentials.json` (only your account listed); the bundled fonts in the UI.
- macOS start-up log (2026-10-23): no "Populating font family aliases" (Qt) or "findfont ...
  Apple Chancery" (matplotlib) lines when the GUI opens a notebook with plots. The
  `TSMSendMessageToUIServer ... com.apple.tsm.uiserver` lines are AppKit's input-method client
  logging from the Python process and are harmless; they are not silenced.
- Windows parity after the portability audit ([Windows, macOS and Linux](developer/portability.md)):
  open and export a POSCAR, XDATCAR, LAMMPS dump and an `.ewd` with data on another drive (D:);
  check exported files have LF ends; the open dialog lists lower-case `poscar`/`*.CIF`; a save
  dialog with a name typed without extension; settings with non-ASCII text on Korean Windows;
  per-monitor DPI at 125/150 %; Windows Credential Manager; `ewald api > llms.txt` in cmd and
  PowerShell. The same list on a real Mac.
- Resize handles between panels (a 1 px rule thickening into a pill grip, accent on hover, 7 px hit area)
  on Windows at 125/150 % and on a Retina Mac: the grip uses the `@2x` PNG and stays sharp, and
  the split cursor shows over the whole handle.
- Plugins side bar on Windows and macOS: install ewald-phonopy from the default index (uv and
  pip), check the side bar keeps its width while pip runs and can be narrowed again, and that
  no *Update to …* button is left on default-index plugins afterwards.
- MD playback in Ball & stick on Windows with a real GPU: bonds stay on through playback and
  scrubbing (no coarse/full switching between frames), and the numpy bond tubes look like the
  old `vtkTubeFilter` ones. Bonds whose length sits at the cutoff still appear and vanish
  between frames, as in OVITO; a cutoff hysteresis was left out because a frame would then
  depend on the frames played before it.
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
- AI eval suite with other providers (`ewald eval --provider ...`): the Claude baselines are in
  benchmarks/ai_eval/ (2026-10-24); OpenAI, Gemini and a local model (Ollama, LM Studio) have not
  been run, and no run used `--repeat` for the spread.
- AI features with a real API key (Claude; also OpenAI, Gemini and a local Ollama/LM Studio
  server, which are tested only against mocked servers): `ewald ask` (also `--select`), the GUI AI chat side bar
  (notebook code in the prompt, code blocks inserted as cells, the notebook cell tools; with
  Let the AI run cells on, that the model runs and fixes its cells, also over an OpenAI-compatible gateway),
  `ewald narrate`, `ewald forge`, `ewald mcp`, and Preferences › AI › Test connection.
- The API key in a real keychain (macOS Keychain, Windows Credential Manager, GNOME Keyring /
  KWallet); the container has none, so only the 0600 file fallback runs for real.
- The release workflows (`release.yml`: sdist and Linux wheels; `windows-release.yml` and
  `macos-release.yml`: wheels, apps, installers) on a real `v*` tag. `windows-release.yml` ran green
  on a branch (2026-10-27) and the owner ran its exe; v0.2.0 is the first tag
  ([Releases](developer/release.md)). macOS has not been built yet.
- Installing without a compiler on real Windows (MSVC missing raises in `build_ext.run`; tested
  here with `CXX=/bin/false` on Linux). pip shows the "not built" warning only with `-v`.
- macOS on a real Mac (Apple Silicon and Intel): the source install and the full test suite, the
  3D view drawing atoms at all (it was black before the 3.2 core default format; if it still is,
  `EWALD_VIEWPORT=legacy ewald gui` is the fallback), NEB overlays and other see-through atoms
  (sphere meshes since the impostor shader draws nothing under depth peeling there; still impostors,
  so likely invisible, are see-through atoms beyond `mesh_limit`, the 'flat' style and the coarse
  drag scene), the GUI on Retina (the VTK view filling the whole widget, picking and hover at the right atom),
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

Portability ([Windows, macOS and Linux](developer/portability.md), audit of 2026-10-21):

- Frozen apps have no Python at `sys.executable` (it is `ewald.exe` / the app binary): the Forge
  check subprocess, plugin `pip install` and the default notebook kernel need a bundled
  interpreter entry (`ewald.exe python -I ...`) or a clear "not in the app" message.
- Windows without OpenGL 3.2 gets the Legacy GL viewport with Qt's software OpenGL, but VTK's
  offscreen pictures (OpenGL renderer, `snapshot`, the notebook's frame pictures when they use
  OpenGL) still go to the system OpenGL 1.1 there; fall back to the ray tracer in that case.
- `plugin_catalog` installs move folders with `shutil.move`, not `fileops.replace`, so a plugin
  folder held open on Windows fails without naming the program.

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

- What the AI may read (`ewald.llm.access`, Preferences › AI › Folders the AI may read, Ask before
  the AI connects to a host): `ewald ask` and the MCP server have no flag for the folders and never
  ask about hosts; the allowed hosts are forgotten with each new chat.
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

- Going public (files prepared 2026-10-20: CITATION.cff, FUNDING.yml, CONTRIBUTING.md,
  SECURITY.md, issue templates; history scrubbed and moved to a fresh repository 2026-10-23,
  D32): the owner switches the repository to public; then enable private vulnerability reporting and GitHub Sponsors, archive
  a release on Zenodo and add its DOI to CITATION.cff, and restore the CI triggers
  (public repositories get free Actions minutes).
- JOSS paper (draft in `paper/`, checked by `tests/test_paper.py`): JOSS (2026 rules) wants at least
  six months of public history with releases, issues and pull requests, so the earliest submission
  is six months after the repository goes public. Before then: fill in the author list,
  affiliations, funding and conflict-of-interest statement (TODOs in `paper/paper.md`), tag
  releases, collect research use (papers, groups, courses) for the research impact statement, and
  archive the submitted version on Zenodo.
- From the literature (docs/developer/references.md, 2026-10-24): precomputed per-atom ambient
  occlusion for the impostor path (Tarini et al. 2006) next to SSAO; image-space outlines from the
  depth and normal buffers for every style and renderer (Saito and Takahashi 1990); occlusion culling for
  multi-million-atom frames (Grottel et al. 2010); harder `ewald eval` tasks modelled on MDCrow's
  graded MD tasks and ChatVis's visualization tasks; an AI tool that fetches reference structures
  and properties (LLaMP). Confirm the DOIs the page marks as not confirmed before citing them.

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
  installs `.[all]` (and the plain core) in a clean venv and imports its modules would catch broken
  extras (`tests/test_dependencies.py` only checks that every import is declared). The frozen apps
  now carry spglib, h5py, imageio-ffmpeg, anthropic/openai/mcp and ipykernel as core packages:
  check their size and that PyInstaller bundles them.
- A step that failed for a missing package retries only when its settings change or the pipeline
  is invalidated; installing the package while the app runs needs a Run of that cell (a GUI
  "Retry" action on the failed step is open).

- macOS release: a first run of `macos-release.yml` (manual; macOS minutes bill 10x), then
  Developer ID signing and notarization so the downloaded app opens without the Gatekeeper
  override (needs an Apple Developer account and the certificate as a workflow secret).

## Deferred until the user asks

- Structure editing: supercells, slabs, defects. Do not start it without an explicit request.
  The display side exists (`view(pipe, replicate=...)`, 2026-10-20); the user said a real
  supercell step (a general 3x3 matrix like VESTA's lattice transformation, with bonds, grids
  and polyhedra following, analysis and export on the enlarged cell) is "not now".
