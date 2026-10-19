# Desktop GUI

```bash
pip install -e ".[gui]"
ewald gui XDATCAR                        # or: ewald-gui dump.lammpstrj --type-map 1=Cu
ewald gui run.lammpstrj --device cuda    # start with GPU compute
```

From Python, `ewald.gui.show(pipe)` opens the viewer on a pipeline you built in a script. In
Jupyter, run `import ewald.gui` and then `%gui qt` (in that order, so IPython picks PySide6 even
when PyQt is installed); `show` then returns at once and the window stays live next to the
notebook. Linux system libraries and startup errors: [Install](install.md#linux-gui-system-libraries).

The window has the analysis on the left as a [notebook](notebooks.md): Python cells, steps and
linked plots, saved as an `.ewd`, `.ipynb` or `.py` file, with one tab per open notebook. The
[tutorials](tutorials/first-notebook.md) walk through it. The viewport and timeline are in the middle. Display settings and Ask Claude are on the
right, and a bottom dock holds the Python console, the data inspector and plots.

![The notebook layout](../images/notebook.png)
 The notebook
replaces the pipeline panel, but the sections below apply to both. Preferences › General ›
*Window layout* = `classic` brings back the OVITO 3 style layout, with the pipeline and the
selected item's parameters on the right. VESTA's crystal-viewing controls are in both.

Buttons use one line-icon set. By default mode switches and less frequent buttons (Open,
Analyze, Render) show an icon with text, and the tool rail and frequent buttons (Open remote,
Save, Fit) show the icon alone, with the name and shortcut in the tooltip. Preferences ›
General › Toolbar buttons switches to icons only or to text only (which also drops the icons
from menus).

## Analyze workspace

**Viewport.** Drag to orbit and scroll to zoom (VTK's standard mouse controls). View › View
along a, b, c, a*, b*, c* aligns the view with a cell vector or reciprocal vector, as in
VESTA, and View › Zoom to fit (Ctrl+0) frames the cell. View › Theme applies a theme (Default,
VESTA, CPK, Journal, ASE, Dark slide, or a [theme plugin](../plugins/styles.md#themes)): element
colors, background, cell color and radius preset in one step.

**Styles.** The Style box on the view toolbar switches between Ball & stick (the default: small
balls and bonds found automatically), Space-filling, Simple (ASE) and Points; Preferences ›
Viewports › *Default style* picks the one a new window starts with. Simple (ASE) looks like ASE's
GUI: flat discs in the element color with a dark outline, unshaded bonds with outlined edges, and
the ASE theme (white background). It is drawn with impostors, so it stays fast for any number of
atoms; leaving it restores the colors and background you had. Particles › Rendering = flat gives
the same look with any radius. Automatic bonds come from a VESTA-style table (Bonds), one row per
pair of types up to 1.15 × the sum of covalent radii; in compounds metal-metal pairs are left
out, so SrTiO3 gets Ti-O and Sr-O bonds but not Sr-Ti. Opening a structure with more than
Preferences › Elements › *Automatic bonds up to* atoms (100 000) turns bonds off, because the
bond search and the sticks would slow the viewport down; turn them on under Bonds if you want them. The tool rail on the left switches
between orbit, pick (click an atom to see its id, type, position and every property;
Shift+click adds atoms to the pick or removes them),
distance (click two atoms) and angle (click three atoms, vertex in the middle). Resting the
cursor on an atom with any tool shows its element, index (and id), Cartesian position in Å and,
when there is a cell, fractional coordinates, in a tooltip and on the right of the status bar.
The ray cast runs once, after the cursor rests (Viewports › Hover › *Hover delay*, 0.25 s), and
never while the view moves, so it costs nothing during rotation; *Atom info on hover* turns it
off. Atoms drawn only as periodic images (VESTA boundary) are not reported. The viewport
draws atoms as GPU impostors: each sphere is one point shaded per pixel, with the depth of the
sphere surface, so spheres stay round at any zoom, overlapping spheres and bonds intersect as
with meshes, and 100 000 atoms rotate smoothly. Viewports › Performance › *Sphere drawing* =
Mesh brings back triangle spheres (*Sphere resolution*, up to the *Impostor threshold*), and
*Smooth edges* (on) applies FXAA to outlines.
A right click (without dragging, which zooms) opens the viewport menu, whose entries follow
what is loaded and picked. With nothing loaded it offers Load file, Load remote file and the
examples. With picked atoms it starts with their count and Center view on picked, Select picked
atoms (an `ExpressionSelection` on their identifiers, so the same atoms stay selected on every
frame), Delete picked atoms (the selection plus `DeleteSelected`) and Clear picks. Phonon modes
add Phonon band structure, an NEB band adds NEB energy profile (both open the Plots tab), and a
volumetric grid adds Add isosurface (or Edit isosurface once there is one). A trajectory adds
Play/Pause and Overlay other frames, which draws every other frame as see-through copies behind
the current one (`OverlayFrames`; on by default for NEB folders), with its transparency (50 to
95 %) and whether only the atoms that move get copies. Show (bonds, simulation cell), View
along, Orthographic, Zoom to fit and Ask Claude are always there.

**Ask Claude.** The Ask Claude tab next to the Display panel (next to the pipeline in the classic layout) (Tools › Ask Claude about selection,
Ctrl+J) answers questions about the picked atoms, such as "What are these atoms?" or "Why is the
coordination low here?". Without picks it uses the particles a selection modifier selected, and
with neither the question is about the whole frame. Claude receives the description from
[Asking about a selection](claude.md#asking-about-a-selection) and a close-up render, can run the
usual tools, and modifiers it adds appear in the pipeline. It needs the `llm` extra and an API key
(Preferences › Claude, or `ANTHROPIC_API_KEY`).

**Timeline.** Play, step and scrub through frames. Every frame is computed through the whole
pipeline when you move to it. While a simulation is still writing the file, new frames are
added to the timeline as they appear ([live reload](formats.md#runs-that-are-still-writing)).

**Pipeline panel.** The list shows the data source at the bottom and modifiers above it, the
last-applied modifier on top, as in OVITO. "Add modification" (or the Pipeline menu) inserts a
modifier; the eye toggle disables one without removing it; the arrow buttons move it up or down. Each row shows the
time the modifier took and, if it failed, its error. A failing modifier is skipped so the view
never goes blank. Selecting a row opens its parameters below the list; the form is generated
from the modifier's constructor, so every argument you can pass in Python can be edited here.

**Visual elements.** The top of the list holds what is drawn: Particles, Bonds and Simulation
cell, plus any [visual element plugins](../plugins/visuals.md) added with the "+ Visual" button.
Simulation cell has VESTA's boundary range: tick Boundary and set the fractional range (e.g.
-0.1 to 1.1) to draw the periodic images of atoms on the faces, edges and corners, with bonds and
polyhedra completed across the cell; [display only](rendering.md#boundary-vesta).
A plugin element has a checkbox to hide it, a parameter form generated from its constructor
(Apply or Ctrl+Enter redraws), and Del removes it. Plugin elements are saved with the session
and drawn in rendered images too.

**Python script modifier.** Pipeline › Python › Python script adds a modifier whose code is
edited in place. The code defines `modify(frame, data)` and runs on every frame like any other
modifier. See [Python scripting](scripting.md).

**Python console.** An IPython console where `pipeline` (also `pipe`), `data` (the current
frame), `viewer` (the window), `vp` (the viewport), `mods` (`ewald.modifiers`) and `np` are
defined. Changes made in the console, such as `pipe.modifiers.append(...)`, update the
pipeline panel and redraw the view.

**Data inspector.** The particle table with a filter expression (the same syntax as
`ExpressionSelection`), plus global attributes and tables.

**Plots.** Any table in the current frame (g(r), MSD, histograms, time series) can be plotted
and saved. The `Thermo` table (log.lammps, OSZICAR, OUTCAR, vasprun.xml; see
[File formats](formats.md#simulation-results)) is drawn one column at a time, picked under
"y", with a dashed line at the current frame's step that follows the timeline. The `NEB`
table (a NEB folder, or `NebEnergyProfile`) is drawn as the energy profile: the spline, the
images as points, the current image highlighted and the barriers in the corner. Clicking an
image jumps to its frame. Diffraction
reflections (`diffraction-peaks`) are drawn as sticks, and a measured pattern loaded through a
modifier's `experimental` file (the "…" button picks it) as points under the calculated curve.
The `DOS` and `Bands` tables of a VASP run are drawn as a density of states and a band
structure; atoms selected in the viewport show as their projected DOS or as fat bands (see
[DOS and band structure](electronic.md)).

*Save plot…* (and *Save plot…* in a view cell's menu) writes a publication figure, not a
screenshot of the panel: the plot is redrawn at a journal column width (Preferences ›
Rendering › Plots: 1-column 3.5 in, 1.5-column or 2-column 7.2 in) with 8 pt sans-serif type,
inward ticks on all four sides, labels with units (*r* (Å), 2*θ* (°), *q* (Å⁻¹), MSD (Å²),
*E* − *E*<sub>F</sub> (eV)), a colorblind-safe color cycle and a white background,
whatever the UI theme. PDF, SVG and EPS keep the text as text (TrueType, editable in
Illustrator or Inkscape); PNG uses the DPI of the image quality (Draft 150, Standard 300,
Publication 600). Hints such as the current-frame marker are left out. On screen the same
style is drawn in the Paper or Graphite colors.

**Linked selection.** Plots and the viewport share one selection, in both directions:

- Drag an x range on a histogram (the "histogram of" box, `Histogram`, CSP or PTM RMSD), g(r),
  bond length or bond angle plot to select the particles behind it: particles with that
  property value, particles in a pair at that distance (only A-B pairs on a partial `g(r) A-B`
  column), both ends of the bonds with that length, or the vertex atoms of those bond angles.
  They are picked in the viewport, so Ask Claude about selection uses them, and the range is
  re-applied on every frame. **Keep as selection** turns a property range into an
  `ExpressionSelection` modifier.
- Drag on a per-frame plot (`Thermo`, `time-series`, `plumed`, `colvar`, MSD) to select the
  frames in that x range; Shift+drag a box to select by value as well (for example the frames
  where a CV lies in a window). Drag a box on a 2D free energy surface to select the frames whose
  CV pair (from `LoadColvar` or `Plumed`) falls in it. The frames are marked on the timeline,
  playback steps through them only, and the ✕ on the timeline clears them. A click on a
  per-frame plot jumps to the nearest frame.
- The other way, atoms picked in the viewport (or the `Selection` property) are drawn on the
  plot: as a highlighted sub-histogram on count plots, or as a second axis ("selected") on g(r).
  Selected frames are marked as points on per-frame plots.

The same mapping is available without the GUI in `ewald.linking` (`link_for`,
`atoms_in_range`, `frames_in_box`, `selection_expression`).

**Command palette.** Window › Command palette searches every menu action and every modifier by
name.

## Render workspace

Window › Render workspace (Ctrl+2), or the Render button, opens the figure settings: renderer
(the built-in ray tracer or OpenGL), anti-aliasing samples, ambient occlusion and shadows,
journal size presets (1-column 3.5 in, 2-column 7.2 in, square, 16:9 slide) at a chosen dpi,
background (white, transparent, black, or the viewport's), overlays, and an output file.
Under Overlays, "+ Overlay" adds an overlay element (Coordinate tripod, Attributes, or an
[overlay plugin](../plugins/overlays.md)) with a checkbox and a parameter form. The preview
shows the final aspect ratio. The "Python for this figure" panel shows the script that
produces the same image with `pipe.render(...)`, updated as you change settings, so a figure
made in the GUI can be reproduced in a batch script. See [Rendering figures](rendering.md).

## Files

File › Open example opens one of the [example files](quickstart.md#example-files) shipped
with ewald. File › Open (File › Load file in the classic layout) opens any format ewald can read ([File formats](formats.md)), including
formats added by [reader plugins](../plugins/readers.md). LAMMPS dumps ask for a
type-to-element map unless you set a default in Preferences. File › Load NEB folder opens a
VASP NEB folder (`00 .. NN`) or a LAMMPS NEB run folder with one frame per image
([NEB runs](formats.md#neb-runs)). In the notebook layout, File › Open also opens `.py`
notebooks, and File › Save notebook writes one ([Notebooks](notebooks.md)). File › Export data writes the
current frame or the whole trajectory through `export_file`, and File › Export movie writes an
MP4 or GIF (needs the `movie` extra). File › Live reload turns following a growing trajectory
on or off for the session; the default and the interval are in Preferences › General.

File › Load remote file (SSH) browses a folder on another machine once. For machines you use
often, the Remotes side bar (the monitor icon in the activity bar, or File › Remote computers,
Ctrl+Alt+R) keeps saved remote computers with their login, jump host, start folder and
bookmarks, and browses their files; passwords and key passphrases go to the OS keychain
([Saved remote computers](formats.md#saved-remote-computers)).

## Preferences

Edit › Preferences (Ctrl+, ; the application menu on macOS) has seven pages: General, Viewports,
Elements (per-element color and radius, Jmol or CPK scheme, radius preset, VESTA `elements.ini` import),
Rendering, Compute (default device, CPU threads, installed components), Python (console
startup script, extra module paths) and Claude (API key, model and effort for Ask Claude; see
[API key](claude.md#api-key)). Only values that differ from the defaults are saved, as
JSON in the ewald config folder: `~/.config/ewald` on Linux,
`~/Library/Application Support/ewald` on macOS, `%APPDATA%\ewald` on Windows, or
`$EWALD_CONFIG_DIR` when set. Preferences affect the GUI only; scripts that call `compute` or
`render` are not changed by them.

The radius preset (covalent, atomic, van der Waals or Shannon ionic radii; see
[Atom radii](rendering.md#atom-radii)) is the one new sessions start with. The Particles visual
element in the pipeline panel switches it for the current session and takes oxidation states
for the ionic presets (`Sr=2, Ti=4, O=-2`; types left out are guessed). The log says which
types fell back to covalent radii.

### Large systems

Interaction comes first in the viewport, and quality comes first in rendered output. Under
Viewports › Performance, *Detail while moving* (Auto by default) controls what happens while
you rotate, zoom, drag the timeline slider or play. Auto switches to a coarse scene only when a
full-detail frame takes longer than 1/*Target frame rate* (30 fps). The coarse scene uses
impostors, leaves out bonds and draws every k-th atom, so at most *Atoms while moving*
(1,000,000) remain. Full detail returns after *Refine after* (0.3 s) of idle. *Always coarse*
and *Always full* fix the choice. When frames are requested faster than they can be computed,
the queued requests collapse into the latest one. The pipeline runs on a worker thread, so the
window stays responsive while a long analysis computes: the view keeps the previous result and
the status bar says *Computing frame n…* until the new one arrives, and changing the frame or a
parameter meanwhile cancels the running computation (trajectory modifiers stop at their next
frame). Results that take less than 50 ms appear at once, as before. Compute › Device ›
*Compute in the background* turns this off. Rendered images and movies always use full
detail. Compute › Memory › *Frame cache memory* (2048 MB) caps how much the trajectory frame
cache holds; the current frame is always kept. Scripts set the same cap with
`Pipeline(..., cache_memory=bytes)` or `pipe.cache_memory`.
`python benchmarks/large_files.py` times opening, frame switching and memory on generated
multi-million-atom files.

The viewport draws a change once, in the next repaint: several changes made before the
repaint (a new frame, picks, a camera move) cost one render, a mouse move while rotating costs
one render rather than two, and nothing is redrawn while the window is idle or the viewport is
hidden. The Data inspector and Plots tabs follow the current frame only while they are open; a
plot range linked to the viewport keeps following it either way.

*3D view* picks the viewport itself. Auto uses VTK, except on an SSH-forwarded display
(`ssh -X`), where it uses Legacy GL: OpenGL 1.x display lists drawn like VESTA, with the frame text
above the view instead of on it, mesh spheres (points above 400,000 atoms) and no smooth edges or
color bar. See [Troubleshooting the GUI](install.md#troubleshooting-the-gui). It takes effect at
the next start; `EWALD_VIEWPORT=vtk|legacy-gl` overrides it.

The Compute switch in the toolbar changes the device for the current session only; the
preference sets the device the GUI starts with: Auto by default, which uses the CUDA GPU when
CuPy finds one (the status bar then reads `AUTO (CUDA)`) and the CPU otherwise.

## Help

The Help menu opens this manual, the plugin guide and the API reference. By default it opens
the `docs/` folder of the repository on GitHub; set Preferences › General › Documentation URL
to point it at a hosted copy of the site instead.
