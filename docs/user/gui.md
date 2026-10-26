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
[tutorials](tutorials/first-notebook.md) walk through it. The viewport and timeline take the rest of the width, the AI chat opens left of the notebook from the activity bar (side bars you work in on top: Notebook, AI chat, Workflows and other plugin side bars, Remotes; Plugins and Preferences at the bottom; when the window is too short, the last top button becomes … with the rest, and a right-click on the bar hides buttons you do not use, which stay in the Window menu), and the panel
below the viewport holds the Python console, the View tab (display settings), the data inspector, plots and the log. Tabs in the
middle can be rearranged as in VS Code (see [Panel layout](#panel-layout)).

![The notebook layout](../images/notebook.png)

The notebook is the pipeline: each step cell is one `pipe.add(...)`, so there is no separate
pipeline list. (Earlier versions had an OVITO-style "classic" layout with a pipeline panel; it
was removed, and a saved *Window layout* setting is ignored.)

Buttons use one line-icon set. By default mode switches and less frequent buttons (Open,
Notebook, Render) show an icon with text, and the 3D view's tools and frequent buttons (Save, Fit)
show the icon alone, with the name and shortcut in the tooltip. Preferences ›
General › Toolbar buttons switches to icons only or to text only (which also drops the icons
from menus).

## Panel layout

The viewport and the panel below it are tab groups. Drag a tab onto another group to move it:
the middle of the group adds it as a tab, an edge splits the group in that direction, and a
tab bar inserts it at that place. Esc cancels the drag. To look at the pair correlation
function large, for example, drag the Plots tab onto the Structure tab: the plot fills the main area
and the Structure view waits behind its tab, without redrawing, until you click it again. Drop it on
the right edge of the Structure view instead to see structure and plot side by side; both follow the
timeline and linked selections.

Right click a tab (or use the group's menu button) for the same moves without dragging: Split
right, Split down, Move to editor area or panel, Maximize group, New plot tab (a second plot,
for example g(r) next to the MSD), New structure tab (a second 3D view, below) and Close or
Return to panel. Double-click a tab to maximize
its group. View › Layout lists the moves with their shortcuts, and the command palette (Ctrl+K)
finds them too:

| Action | Shortcut |
| --- | --- |
| Split the current view to the right | Ctrl+Alt+\ |
| Move the current view to the next group (or split that way) | Ctrl+Alt+arrow |
| Maximize or restore the group (an editor group also hides the panel below) | Ctrl+Alt+M |
| Return the view to its home area, or close an extra plot view | Ctrl+Alt+W |
| Fold or unfold the panel | Ctrl+Alt+J |
| Back to the default layout | View › Layout › Reset layout |

The layout is saved with the window when Preferences › General › *Restore window layout* is
on (the default), extra plot views included. The Structure view itself never moves between windows:
it can be placed anywhere in the middle but not closed, and the timeline stays under the
editor area.

### More structure tabs

`view(pipe, tab="Top", direction="z")` in a notebook opens a second 3D view, the
*Structure · Top* tab, next to the Structure tab. It shows the same pipeline and frame with the
main view's settings, except the ones the tab sets: here the camera looks down z, and colors,
radii and bonds follow the main view. Running the cell again updates the same tab instead of
opening another; a tab without `tab=` changes the main view as before. Window › New structure
tab (also in the tab menu and the viewport's right click) writes an empty `view(pipe, tab="View 2")`
cell and opens it. A style picked in a tab's viewport goes into that tab's cell, and its right
click › Write view to notebook adds the tab's camera. Closing the tab deletes its cell (Edit ›
Undo brings the cell back; run it to reopen the tab). Split the tab to the right to see two
directions side by side. Picks, measurements with the mouse and linked plot selections stay in
the main Structure view.

## Notebook workspace

**Viewport.** Drag to orbit and scroll to zoom (VTK's standard mouse controls). View › Look
along › a, b, c, a*, b*, c* aligns the view with a cell vector or reciprocal vector, as in
VESTA, and View › Zoom to fit (Ctrl+0) frames the cell. View › Structure theme applies a theme (Default,
VESTA, CPK, Journal, ASE, Dark slide, or a [theme plugin](../plugins/styles.md#themes)): element
colors, background, cell color and radius preset in one step; its Dark background and White
background entries change only the background.

**Styles.** The Style box on the view toolbar switches between Ball & stick (the default: small
balls and bonds found automatically), Space-filling, Simple (ASE) and Points; Preferences ›
3D view › *Default style* picks the one a new window starts with. Simple (ASE) looks like ASE's
GUI: flat discs in the element color with a dark outline, bonds (on, like Ball & stick) as unshaded
sticks with outlined edges, and the ASE theme (white background). It is drawn with impostors, so it stays fast for any number of
atoms; leaving it restores the colors and background you had. Particles › Rendering = flat gives
the same look with any radius. Automatic bonds come from a VESTA-style table (Bonds), one row per
pair of types up to 1.15 × the sum of covalent radii; in compounds metal-metal pairs are left
out, so SrTiO3 gets Ti-O and Sr-O bonds but not Sr-Ti. Opening a structure with more than
Preferences › Atoms & bonds › *Automatic bonds up to* atoms (250 000) turns bonds off, because the
bond search and the sticks would slow the viewport down; turn them on under Bonds if you want them.

The 3D view's controls float over it as three chips: view directions (look along a, b, c, a\*,
b\*, c\*) and one-step rotations at the top left, Ortho, Style and Fit at the top right, and the
tools on the left. When the view gets small (a split, another Structure tab, a narrow window) they
fold instead of overlapping: a\*/b\*/c\* go under a chevron and the rotations under one button,
then the six directions become one *a* menu and Style moves into *⋯*, and at the smallest
size the rotations and Ortho go there too; Fit always stays. A short view moves the last tools
and Clear into a *⋯* at the bottom of the tool chip, which is highlighted while it holds the
current tool. Color bars and the color legend are drawn below the chips. The 3D view shows no
frame text: the timeline under it shows the frame's timestep and time beside its number when the
file has them (Preferences › 3D view › *Show timestep and time*). The
OpenGL 1.x viewport used over `ssh -X` cannot draw widgets over its GL window, so there the same
chips sit in a strip above the view and a column left of it.

The tool chip on the left switches
between orbit, pick (click an atom to see its id, type, position and every property;
Shift+click adds atoms to the pick or removes them),
distance (click two atoms) and angle (click three atoms, vertex in the middle); right-click ›
*Keep measurement in the view* keeps a finished one in the notebook's view cell (`measurements`),
so it stays drawn and appears in pictures (*Clear measurements* drops them). Double-click
empty space or press Esc to clear the picks, measurements and selection (with any tool,
orbit included). Resting the
cursor on an atom with any tool shows its element, index (and id), Cartesian position in Å and,
when there is a cell, fractional coordinates, in a tooltip and on the right of the status bar.
The ray cast runs once, after the cursor rests (3D view › Hover › *Hover delay*, 0.25 s), and
never while the view moves, so it costs nothing during rotation; *Atom info on hover* turns it
off. Atoms drawn only as periodic images (VESTA boundary) are not reported. The viewport
draws atoms as GPU impostors: each sphere is one point shaded per pixel, with the depth of the
sphere surface, so spheres stay round at any zoom, overlapping spheres and bonds intersect as
with meshes, and 100 000 atoms rotate smoothly. Bonds are impostors too: each half-bond is one
point ray-cast as a cylinder, so a 100 000-atom ball-and-stick frame (over a million half-bonds)
is built in a fraction of a second instead of as millions of triangles. 3D view › Performance › *Sphere drawing* =
Mesh brings back triangle spheres (*Sphere resolution*, up to the *Impostor threshold*), and
*Smooth edges* (on) applies FXAA to outlines.
A right click (without dragging, which zooms; on a Mac also a two-finger click or Control+click)
opens the viewport menu, whose entries follow
what is loaded and picked. With nothing loaded it offers Open file, Open remote file and the
examples. With picked atoms it starts with their count and Center view on picked, Select picked
atoms (an `ExpressionSelection` on their identifiers, so the same atoms stay selected on every
frame), Delete picked atoms (the selection plus `DeleteSelected`) and Clear picks. Phonon modes
add Phonon band structure, an NEB band adds NEB energy profile (both open the Plots tab), and a
volumetric grid adds Add isosurface (or Edit isosurface once there is one). A trajectory adds
Play/Pause and Overlay other frames, which draws every other frame as see-through copies behind
the current one (`OverlayFrames`; on by default for NEB folders), with its transparency (50 to
95 %) and whether only the atoms that move get copies. Show (bonds, simulation cell), Look
along, Orthographic, Zoom to fit and Ask AI are always there.

**Plugin side bars.** A plugin can bring its own side bar (a sim-agent panel, a database
browser, ...): installing it adds a button to the activity bar (after AI chat, before Remotes) and an entry
in the Window menu (or a tab under the 3D view instead), it may add toolbar buttons
and entries to the menus and to the right-click menus of the 3D view, notebook cells and plots, and a plugin with settings adds its own page to
Preferences. Plugin side bars and tabs that were open come back at the next start where they
were. Switching the plugin off in the Plugins side bar removes all of it; its settings are kept
for when it comes back ([Side bar plugins](../plugins/gui.md)).

**Workflow side bars.** The built-in **Workflows** side bar reads folders of calculations
(relax, NEB, MD restarts, phonopy; File › Open from › Workflow folder… or dropping a folder does the
same, see [Workflows](workflows.md)). Plugins that read finished calculations (sim-agent, AiiDA)
share its layout: the account, a search box that also takes an id, and the workflows newest
first. Expanding a workflow lists its steps (restarts of one calculation joined, sub-workflows
nested) with their files and structures. A double click on a step or output opens it; on a
workflow it opens the whole workflow as one notebook whose load cell names every step, a line
each (`relax = load(...)`, `scf = load(...)`), and a last cell that sets `pipe` to the main result
(`result` when the workflow's own result is none of its steps). While that notebook is on screen,
the side bar marks every step it loads: ● is the one `pipe` is now (the 3D view and the main
Plots tab), ○ the others. A click on ○, or a double click on the step, makes it `pipe` by
rewriting that last cell and running it. Under each loaded step its tables are listed
(**Plot Bands**, **Plot Steps**, ...); a click adds a view cell such as
`show(bands, table="Bands", panel=True)`, which opens that step's table as its own Plots tab
whichever step `pipe` is, so the bands stay in view while you switch the 3D view between relax and
scf ([Workflow sources](../plugins/workflows.md)).

**AI chat (Ask AI).** The AI button (a chat bubble, second from the top of the activity bar on the far left; also
Window › AI chat, Ctrl+J, and Ask AI in the Structure view's right-click menu) opens a chat side bar left of the
notebook. It reads like a
chat app: your messages in bubbles on the right, the answers under the assistant's name (Claude, Gemini, ...) on the left,
the tools it ran as small chips (*Running …*, then *Used …*), and code blocks as framed code with
*Copy* and *+ Insert as cell*. Enter sends, Shift+Enter starts a new line, and the suggestions
on an empty chat send a first question. Answers appear while they arrive; text the model wrote
before running a tool stays, muted, above that tool's chip (Preferences › AI › **Stream answers**
off shows each answer once it is complete, for a server that cannot stream). While an answer is on
its way the send button turns into Stop (also Esc): the chat is free again at once, what arrived so
far stays readable, and the stopped question is forgotten.
A tool that wrote, edited or ran a cell shows a link in place of its chip (*Edited and ran cell 3*,
*Changed view in cell 5*; cells are counted from 1, top to bottom, as they were then): clicking it
switches to that notebook's tab, scrolls to the cell and lights it up for a moment. The links come
from the tools' results, not from the model's text, and they are saved with the chat in an
`.ewd`; a deleted cell's link is greyed out, and one whose cell is gone since says so in the
status bar.
Each notebook tab has its own chat: switching tabs switches the chat, an answer still on its way
finishes in its own tab's chat (and its cell tools act on that notebook), and *New chat* starts
over in the current tab only. Saving the notebook as `.ewd` saves its chat; opening the file shows
it again, and the next question sends it to the model as an earlier conversation (for a restricted
notebook, only after you trust it).
Each question also sends the current notebook (its code, with each cell's number, state and a
short line about its outputs) and the errors of cells that failed when they last ran (each only
when it changed since the last question). Python blocks of an answer go into the notebook at
once, without running them (Shift+Enter runs a cell, Ctrl+Z takes one out again): as new code
cells after the current one, or, when a block does what one cell already does (the same step,
a plot of the same table, the view cell), in place of that cell's code, so asking for another
option does not leave the old cell behind. Their button then reads *✓ Inserted*. With Preferences › AI › **Code in answers** set to `button`, nothing is inserted
until you press *+ Insert as cell*. The AI can also read and edit cells itself
([Notebook cells](claude.md#notebook-cells)).
The chat answers questions about the picked atoms, such as "What are these atoms?" or "Why is the
coordination low here?". Without picks it uses the particles a selection modifier selected, and
with neither the question is about the whole frame. The model receives the description from
[Asking about a selection](claude.md#asking-about-a-selection) and a close-up render, can run the
usual tools, and steps it adds appear in the notebook. Claude answers by default; Preferences › AI
switches to OpenAI, Gemini or a model on your own machine ([Providers](claude.md#providers)). It needs
the `llm` extra and, except for a local server, an API key (Preferences › AI, or the provider's
environment variable such as `ANTHROPIC_API_KEY`).

**Timeline.** Play, step and scrub through frames. Every frame is computed through the whole
pipeline when you move to it. While a simulation is still writing the file, new frames are
added to the timeline as they appear ([live reload](formats.md#runs-that-are-still-writing)). A file of a
calculation that has not finished (VASP still running or killed) opens with what is complete,
and the status bar says what is missing ([details](formats.md#calculations-that-have-not-finished)).

**Steps.** **+ Step** next to the notebook's Run all (a list of steps to search), Insert › Step, **+ Step**
under the cells and the command palette add a
modifier as a step cell. The cell's checkbox disables it without removing it, Move cell up/down
reorders it, and its header shows the time it took and, if it failed, its error. A failing step
is skipped so the view never goes blank. The step's parameter form is generated from the
modifier's constructor, so every argument you can pass in Python can be edited there. See
[Notebooks](notebooks.md).

**View tab (display settings).** The View tab, next to the Python console below the Structure view
(Window › View settings), shows how the structure is drawn as cards in equal-width columns:
**Atoms** (style preset, radius table, radius scale, oxidation states, how atoms are drawn, *Sphere
detail* (`sphere_resolution`) for mesh spheres, *Draw atoms* off for a stick model), **Colors** (what colors the atoms: the steps' colors, the type colors
or a property through a color map with its range, and the rule drawn now, one row per structure type
or painted condition with its count; see [What colors the atoms](rendering.md#what-colors-the-atoms)),
**Types** (color, radius, transparency and count per type, VESTA
elements.ini import), **Bonds** (sticks, the bond rules A-B / A-* / *-* with a *PBC* choice per rule: on, off, + atoms, *As step* for a Create bonds step),
**Surfaces** (polyhedra, isosurfaces and slices that steps made, by name: shown, color,
transparency, edges; *Reset* gives back the step's look), **Cell** (line width, *Replicate cell* and
VESTA's *Boundary* range), **Labels** (text on each atom such as `{type}{index}`, *Highlight
selection*), **Camera** (what the view cell says about the camera and frame, the key *Light*, *Save
camera*) and **Elements** ([visual element plugins](../plugins/visuals.md), added with *Add
element*). Every form shares one label column, so fields line up across cards; sub-sections such as
Replicate cell, Boundary and Light show their rows only while ticked. Click a card's title to fold
it; right-click a card, or View › Layout › Fold all View cards / Unfold all View cards, for every card.
Folded cards stay folded the next time the app starts (with the window layout). Every value is a setting of
the notebook's `view(pipe, ...)` cell: the rows the cell sets are highlighted, the card says *in
notebook*, and the tooltip shows the value. A change made here is written back into that cell, and
a change made in code (the cell, the console, Ask AI) shows up here. When the tab is narrow (dragged
beside the 3D view), the cards stack in one column.

The boundary range: tick Boundary and set the fractional range (e.g. -0.1 to 1.1) to draw the
periodic images of atoms on the faces, edges and corners, with bonds and polyhedra completed across
the cell; [display only](rendering.md#boundary-vesta). Replicate cell draws a na x nb x nc block of
cells (OVITO's Replicate as a display setting; also under View › Replicate cell and the 3D view's
right-click menu); the data and analysis keep the original cell, see
[Replicated cell](rendering.md#replicated-cell). A plugin element has a checkbox to hide it, a
parameter form generated from its constructor (Apply or Ctrl+Enter redraws) and a remove button.
Plugin elements are saved with the session and drawn in rendered images too. *Save camera* writes
the current camera and frame into the view cell. The Camera card's Light box sets the key light
(azimuth and elevation from the viewer, intensity) for the viewport and every renderer; unchecked,
the default light is used.

**Python steps.** A code cell can define a function and add it with `pipe.add(fn)`; it runs on
every frame like any other modifier. Insert › Step › Python › Python script adds the
`PythonScript` modifier as a step. See [Python scripting](scripting.md).

**Python console.** An IPython console where `pipeline` (also `pipe`), `data` (the current
frame), `viewer` (the window), `vp` (the viewport), `mods` (`ewald.modifiers`) and `np` are
defined. Changes made in the console, such as `pipe.modifiers.append(...)`, update the
notebook and redraw the view.

**Data inspector.** The particle table with a filter expression (the same syntax as
`ExpressionSelection`), plus global attributes, tables, the cell and the data source (file,
reader, number of frames, particle properties).

**Plots.** Any table in the current frame (g(r), MSD, histograms, time series) can be plotted
and saved. The `Steps` table (log.lammps, OSZICAR, OUTCAR, vasprun.xml; see
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

**Zoom, pan and hover.** Every plot (the Plots tab, the plots cells put in the Plots panel and view cells) moves the same way. The
wheel zooms the x axis around the cursor and Ctrl+wheel the y axis (on a band structure the
wheel zooms the energy). A right or middle drag pans, and a double-click goes back to the full
range. A right click without dragging opens the plot's menu (*Save plot…*, *Reset view*, for a
cell's plot *Show in Plots panel* or *Show under the cell*, and any plugin entries; see
[where a plot appears](notebooks.md)). On a plot that selects nothing (DOS, bands, phonon bands, maps) a left drag zooms to the
dragged range; elsewhere a left drag still selects. The zoomed range stays while the frame
changes, and *Save plot…* writes the range you see. The DOS and the band structure share one
energy window, shown in the **E** boxes, where you can also type it. Hovering shows the values
under the cursor (*E* − *E*<sub>F</sub> on electronic plots), and on a per-frame or NEB plot
also the frame or image a click goes to, with its value.

*Save plot…* (and *Save plot…* in a view cell's menu) writes a publication figure, not a
screenshot of the panel: the plot is redrawn at a journal column width (Preferences ›
Plots: 1-column 3.5 in, 1.5-column or 2-column 7.2 in) with 8 pt sans-serif type,
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
  They are picked in the viewport, so Ask AI about selection uses them, and the range is
  re-applied on every frame. **Keep as selection** turns a property range into an
  `ExpressionSelection` modifier.
- Drag on a per-frame plot (`Steps`, `time-series`, `plumed`, `colvar`, MSD) to select the
  frames in that x range; Shift+drag a box to select by value as well (for example the frames
  where a CV lies in a window). Drag a box on a 2D free energy surface to select the frames whose
  CV pair (from `LoadColvar` or `Plumed`) falls in it. The frames are marked on the timeline,
  playback steps through them only, and the ✕ on the timeline clears them. A click on a
  per-frame plot (or an NEB profile) jumps to the nearest frame, and a click on a free energy
  surface jumps to the frame whose CV pair is nearest to it.
- The other way, atoms picked in the viewport (or the `Selection` property) are drawn on the
  plot: as a highlighted sub-histogram on count plots, or as a second axis ("selected") on g(r).
  Selected frames are marked as points on per-frame plots.
- **Profiles along the cell** (a 1D `SpatialBinning`, a planar average, a potential lineup or
  work function profile; any table whose `attributes["axis"]` is `"a"`, `"b"`, `"c"` or
  `"x"`, `"y"`, `"z"`) are tied to the structure in the 3D view, also when the table is an
  analysis result shown on its own and the 3D view shows its structure:
  - Dragging a range selects the slab of atoms between those two positions and draws that slab
    as a see-through box in the 3D view (VTK viewport); **Keep as
    selection** keeps it as a `Position.Z` range when that cell vector lies along x, y or z.
  - Hovering shows a see-through plane at that position in the 3D view (VTK viewport), and atoms
    picked in the view show as ticks on the x axis at their positions.
  - **Align 3D view** turns the view side-on, that direction running left to right, with the
    plot's range across its width. Zooming or panning either one moves the other. Rotating the
    view away from the profile's direction lets go of the plot (a status message says so).

The same mapping is available without the GUI in `ewald.linking` (`link_for`,
`atoms_in_range`, `frames_in_box`, `selection_expression`).

**Command palette.** The search box in the middle of the toolbar, or Tools › Command palette (Ctrl+K),
opens the palette right below the box. It searches every menu action and every modifier by name
(initials work too: `msd` finds Mean squared displacement) and shows each one's shortcut; with
nothing typed, the commands you ran from it last come first.

**Menu bar.** Each menu has one job. File opens (Open recent, Open example, Open from › Remote file,
Location, NEB folder, Workflow folder), saves and exports (Export › Image, Data, Movie, Session).
Edit undoes and edits cells (cut, copy, paste, delete, merge; Jupyter's Esc keys are in the
tooltips). View is the camera and what is drawn: Look along a, b, c, a*, b*, c* and the standard
directions, Orthographic, Zoom to fit, Show (bonds, cell, render frame), Replicate cell, Structure
theme, Layout, Focus on the structure view, Minimal mode and Full screen (F11; Ctrl+Cmd+F on macOS).
Insert adds cells (code, text, Step, Plot, Quick analysis, a snapshot), Notebook runs them and holds
the kernel, Tools are the structure view's tools and the command palette, Window shows the side bars
(ticked while shown; plugin side bars sit with the built-in ones) and the tabs, and Help has the
tutorials, the manual and Help › Keyboard shortcuts, a searchable list of every shortcut including
the notebook's command-mode keys. Shortcuts in menus and tooltips are written the platform's way
(⌘ on macOS).

## Render workspace

Window › Render workspace (Ctrl+2), or the Render button, opens the figure settings: renderer
(the built-in ray tracer or OpenGL), anti-aliasing samples, ambient occlusion, the noise
threshold of adaptive sampling (Off gives every pixel all its rays) and shadows,
journal size presets (1-column 3.5 in, 2-column 7.2 in, square, 16:9 slide) at a chosen dpi,
background (white, transparent, black, or the viewport's), overlays, and an output file.
Under Overlays, "+ Overlay" adds an overlay element (Coordinate tripod, Attributes, or an
[overlay plugin](../plugins/overlays.md)) with a checkbox and a parameter form.
The ray tracer renders in the background: the window stays usable, a spinner beside the picture
size reads "Ray tracing… 40%" until the picture appears under *Last render*, and *Render* turns
into *Cancel*, which stops the trace after the rows being traced (no file is written). In Python,
`render(..., renderer="raytrace", progress=fn)` calls `fn(done_rows, total_rows)` the same way
(with adaptive sampling the count can jump to the end when the last pixels converge);
returning `False` stops it with `RenderCancelled`. OpenGL and vector output are quick and render right away.
A vector figure estimated to take more than about 10 s or 10 MB (many atoms, bonds or polyhedra;
see [vector output](rendering.md)) first opens a *Large vector figure* dialog with the shape count,
expected file size and time: *Render PNG instead* writes a PNG of the same view with OpenGL (and
switches the Render page to it), *Render SVG anyway* draws the vector file, *Cancel* writes nothing.

The left side is the live 3D view with the picture's frame on it: the part outside the frame
is dimmed, and the frame keeps the output's aspect ratio and its place on screen, so zooming or
turning the view changes what is inside it and the picture is exactly what the frame covers.
Drag a corner to resize the frame, a side's middle handle to change the picture's aspect (the
pixel size becomes Custom), or the frame's edge to move it; inside the frame the mouse turns and
zooms the view as usual. *Fit to structure* zooms and centers the view so the structure fills
the frame, *Reset frame* goes back to the largest centered frame. After Render, *Last render*
shows the picture; *3D view* goes back. The frame is the `region` of the notebook's view
(`view(pipe, region=[x, y, height])`, see [Rendering](rendering.md#options)), so it round-trips
with the notebook, and Insert › Write view settings to notebook stores it with the camera. Preferences ›
3D view › Render frame in the 3D view, or right-click › Show › Render frame, shows the frame
outside the Render page too. The frame is not drawn yet on the OpenGL 1.x viewport used over
`ssh -X`. The "Python for this figure" panel shows the script that
produces the same image with `pipe.render(...)`, updated as you change settings, so a figure
made in the GUI can be reproduced in a batch script. See [Rendering figures](rendering.md).

## Files

File › Open example opens one of the [example files](quickstart.md#example-files) shipped
with ewald. File › Open opens any format ewald can read ([File formats](formats.md)), including
formats added by [reader plugins](../plugins/readers.md). Its file filter starts at All supported files,
which lists every format at once, extensionless VASP names (`POSCAR`, `CONTCAR`, `XDATCAR`, `OUTCAR`,
`CHGCAR`, ...) and notebooks and sessions included; the groups, one filter per format and All files
follow it. LAMMPS dumps with numeric types ask for an
optional type-to-element map (Preferences › Codes › LAMMPS › *Type names* turns the question off). File › Open from › NEB folder opens a
VASP NEB folder (`00 .. NN`) or a LAMMPS NEB run folder with one frame per image
([NEB runs](formats.md#neb-runs)); opening a single image file of such a run asks whether to
open the whole band. Files, notebooks and NEB folders can also be dropped on the window. In the notebook layout, File › Open also opens `.py`
notebooks, and File › Save notebook writes one ([Notebooks](notebooks.md)). File › Export › Data writes the
current frame or the whole trajectory through `export_file`, and File › Export › Movie writes an
MP4 or GIF (needs the `movie` extra). File › Live reload turns following a growing trajectory
on or off for the session; the default and the interval are in Preferences › General.

Opening a file, connecting to a remote computer, listing its folders and downloading run off the
window's thread: the status line says what is being waited for, and anything slower than a
moment brings up a small dialog with a busy bar and *Cancel* (or Esc), which gives the window
back at once. A cancelled SSH connect still runs out its timeout in the background, and its
result is dropped.

File › Open from › Remote file (SSH) browses a folder on another machine once. For machines you use
often, the Remotes side bar (the server icon in the activity bar, or Window › Remotes,
Ctrl+Alt+R) keeps saved remote computers with their login, jump host, start folder and
bookmarks, and browses their files; passwords and key passphrases go to the OS keychain
([Saved remote computers](formats.md#saved-remote-computers)).

File › Open from › Location (Ctrl+L) takes a typed path or address, the same text `load()` takes in a
notebook cell: `host:/path` over SSH, or a scheme a reader plugin serves, such as `aiida:1234`
or `simagent:1234`; the dialog lists the schemes that are installed. A plugin that needs setting
up (a server address, an access key) has its own page in Preferences (after the Plugins page) and a
Settings button in the Plugins side bar, so it can be set up without Python.

## Preferences

Edit › Preferences (Ctrl+, ; the application menu on macOS) has eleven pages: General, Codes
(one section per simulation code: VASP, LAMMPS and codes added by [plugins](../plugins/codes.md),
each with the files and formats it covers and its own settings, such as whether LAMMPS dumps ask for a
type map), 3D view,
Atoms & bonds (per-element color and radius, Jmol or CPK scheme, radius preset, bond search, VESTA
`elements.ini` import), Rendering, Plots (the style of your own plots, saved plot size and font, and the form and *Open as text* button of your
matplotlib style file, see [Notebooks](notebooks.md)),
Notebooks (run when opened, figures and data saved with a notebook), Compute (device, CPU threads,
installed components), Python (console startup script, extra module paths), AI (provider, API key,
server URL, model and effort for Ask AI; see [API key](claude.md#api-key)) and Plugins (plugin
catalogs), followed by a page for each installed plugin that has settings (its fields, its buttons
such as Test connection; Apply and OK save them through the plugin). Only values that differ from the defaults are saved, as
JSON in the ewald config folder: `~/.config/ewald` on Linux,
`~/Library/Application Support/ewald` on macOS, `%APPDATA%\ewald` on Windows, or
`$EWALD_CONFIG_DIR` when set. Preferences affect the GUI only; scripts that call `compute` or
`render` are not changed by them.

The radius preset (covalent, atomic, van der Waals or Shannon ionic radii; see
[Atom radii](rendering.md#atom-radii)) is the one new sessions start with. The Atoms card of the
View tab switches it for the current session and takes oxidation states
for the ionic presets (`Sr=2, Ti=4, O=-2`; types left out are guessed). The log says which
types fell back to covalent radii.

### Large systems

Interaction comes first in the viewport, and quality comes first in rendered output. Under
3D view › Performance, *Detail while moving* (Auto by default) controls what happens while
you rotate, zoom, drag the timeline slider or play. Auto switches to a coarse scene only when a
full-detail frame takes longer than 1/*Target frame rate* (30 fps). The coarse scene uses
impostors and leaves out labels and measurements; above *Atoms while moving* (1,000,000) it
draws every k-th atom and then leaves out bonds and arrows too. Full detail returns after
*Refine after* (0.3 s) of idle. Playback keeps the level of detail its first frame got and
refines only when it stops, so bonds do not blink between frames. *Always coarse*
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
multi-million-atom files. `python benchmarks/analysis.py` times the analysis modifiers on a
108k-atom crystal and the ray tracer at each quality preset.

The viewport draws a change once, in the next repaint: several changes made before the
repaint (a new frame, picks, a camera move) cost one render, a mouse move while rotating costs
one render rather than two, and nothing is redrawn while the window is idle or the viewport is
hidden, which includes the Structure view sitting behind another tab (it draws the last frame when you
return to it). The Data inspector and plot views follow the current frame only while they are
on screen; a plot range linked to the viewport keeps following it either way.

*Viewport renderer* picks the viewport itself. Auto uses VTK, except on an SSH-forwarded display
(`ssh -X`), where it uses Legacy GL: OpenGL 1.x display lists drawn like VESTA, with the frame text
above the view instead of on it, mesh spheres (points above 400,000 atoms) and no smooth edges or
color bar. See [Troubleshooting the GUI](install.md#troubleshooting-the-gui). It takes effect at
the next start; `EWALD_VIEWPORT=vtk|legacy-gl` overrides it.

*Device* is where analysis runs, for the window and for the notebook's `pipe` alike. It is the
CPU by default, the safe choice; CUDA uses the NVIDIA GPU through CuPy, and Auto uses the GPU
when CuPy finds one (the status bar then reads `AUTO (CUDA)`) and the CPU otherwise. A change
applies at once.

## Help

The Help menu opens this manual, the plugin guide and the API reference. By default it opens
the `docs/` folder of the repository on GitHub; set Preferences › General › Documentation URL
to point it at a hosted copy of the site instead.
