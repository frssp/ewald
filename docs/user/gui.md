# Desktop GUI

```bash
pip install -e ".[gui]"
ewald gui XDATCAR                        # or: ewald-gui dump.lammpstrj --type-map 1=Cu
ewald gui run.lammpstrj --device cuda    # start with GPU compute
```

From Python, `ewald.gui.show(pipe)` opens the viewer on a pipeline you built in a script. In
Jupyter, run `%gui qt` first; `show` then returns at once and the window stays live
next to the notebook.

The layout follows OVITO 3, with VESTA's crystal-viewing controls added: the viewport is in the
middle, the pipeline and the selected item's parameters are on the right, the timeline is under
the viewport, and a bottom dock holds the Python console, the data inspector and plots.

## Analyze workspace

**Viewport.** Drag to orbit and scroll to zoom (VTK's standard mouse controls). View › View
along a, b, c, a*, b*, c* aligns the view with a cell vector or reciprocal vector, as in
VESTA, and View › Zoom to fit (Ctrl+0) frames the cell. View › Theme applies a theme (Default,
VESTA, CPK, Journal, Dark slide, or a [theme plugin](../plugins/styles.md#themes)): element
colors, background, cell color and radius preset in one step. The tool rail on the left switches
between orbit, pick (click an atom to see its id, type, position and every property;
Shift+click adds atoms to the pick or removes them),
distance (click two atoms) and angle (click three atoms, vertex in the middle). Above 150 000
atoms the viewport draws GPU impostors instead of sphere meshes; the threshold is a preference.

**Ask Claude.** The Ask Claude tab next to the pipeline (Tools › Ask Claude about selection,
Ctrl+J) answers questions about the picked atoms, such as "What are these atoms?" or "Why is the
coordination low here?". Without picks it uses the particles a selection modifier selected, and
with neither the question is about the whole frame. Claude receives the description from
[Asking about a selection](claude.md#asking-about-a-selection) and a close-up render, can run the
usual tools, and modifiers it adds appear in the pipeline. It needs the `llm` extra and an API key.

**Timeline.** Play, step and scrub through frames. Every frame is computed through the whole
pipeline when you move to it.

**Pipeline panel.** The list shows the data source at the bottom and modifiers above it, the
last-applied modifier on top, as in OVITO. "Add modification" (or the Pipeline menu) inserts a
modifier; the eye toggle disables one without removing it; the arrow buttons move it up or down. Each row shows the
time the modifier took and, if it failed, its error. A failing modifier is skipped so the view
never goes blank. Selecting a row opens its parameters below the list; the form is generated
from the modifier's constructor, so every argument you can pass in Python can be edited here.

**Visual elements.** The top of the list holds what is drawn: Particles, Bonds and Simulation
cell, plus any [visual element plugins](../plugins/visuals.md) added with the "+ Visual" button.
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
"y", with a dashed line at the current frame's step that follows the timeline.

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

File › Load file opens any format ewald can read ([File formats](formats.md)), including
formats added by [reader plugins](../plugins/readers.md). LAMMPS dumps ask for a
type-to-element map unless you set a default in Preferences. File › Export data writes the
current frame or the whole trajectory through `export_file`, and File › Export movie writes an
MP4 or GIF (needs the `movie` extra).

## Preferences

Edit › Preferences (Ctrl+, ; the application menu on macOS) has six pages: General, Viewports,
Elements (per-element color and radius, Jmol or CPK scheme, radius preset, VESTA `elements.ini` import),
Rendering, Compute (default device, CPU threads, installed components) and Python (console
startup script, extra module paths). Only values that differ from the defaults are saved, as
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
the queued requests collapse into the latest one. Rendered images and movies always use full
detail. Compute › Memory › *Frame cache memory* (2048 MB) caps how much the trajectory frame
cache holds; the current frame is always kept. Scripts set the same cap with
`Pipeline(..., cache_memory=bytes)` or `pipe.cache_memory`.
`python benchmarks/large_files.py` times opening, frame switching and memory on generated
multi-million-atom files.

The Compute switch in the toolbar changes the device for the current session only; the
preference sets the device the GUI starts with.

## Help

The Help menu opens this manual, the plugin guide and the API reference. By default it opens
the `docs/` folder of the repository on GitHub; set Preferences › General › Documentation URL
to point it at a hosted copy of the site instead.
