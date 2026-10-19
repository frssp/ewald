# Notebooks

In the desktop GUI, your analysis is a notebook: Python cells that build the pipeline, plots of
its results, and notes, saved as one `.py` file. It works like Jupyter, and the file stays a
plain script, so `python analysis.py` runs it without the GUI. VS Code, JupyterLab with
jupytext and Spyder open it as a notebook too.

```python
# %% [markdown]
# # Cu void annealing

# %%
from ewald import load
from ewald.modifiers import *
from ewald.views import show
from ewald.paint import paint, hide
pipe = load("cu_void.lammpstrj")

# %%
pipe.add(CoordinationAnalysis(cutoff=3.08))

# %%
pipe.add(ExpressionSelection(expression="Coordination < 12"))

# %%
show(pipe, table="coordination-rdf")
```

`# %%` starts a cell and `# %% [markdown]` a text cell (the percent format). Text after `# %%`
is the cell's title.

## Cells

All cells run in one Python namespace. The variable `pipe` is the notebook's pipeline, and the
viewport shows it. Run a cell with ▶ or Ctrl+Enter, or with Shift+Enter to run it and move to
the next one. Notebook › Run all (Ctrl+Shift+Enter) restarts the namespace and runs every cell
from the top. If a cell fails, its traceback appears under it, and Run all stops there and
marks the cells below as stale.

Running a cell again replaces the modifiers it added last time, in the same place, so you can
edit a cell and re-run it without stacking duplicates. A cell that assigns `pipe` again (a new
`load`) re-runs the cells below it on the new pipeline.

There are four kinds of cell.

**Step cells** are a single `pipe.add(Modifier(...))`, optionally preceded by imports
(`pipe.modifiers.append(...)` works too; `load` is another name for `import_file`). The notebook shows them as a step: the modifier's name, its parameters, a checkbox to
switch it off, the time it took and what it produced (new properties, selected atoms, tables,
removed atoms). A small histogram or curve of the new property or table sits next to the
results. *Form* opens the parameter form, and Apply writes the new values back into the cell's
code. *Python* shows the code. Insert › Step (or **+ Step** under the cells) adds a step with
default parameters after the current cell.

**View cells** are a single `show(pipe, ...)` call. Each is a live plot of a data table or a
particle-property histogram:

```python
show(pipe, table="coordination-rdf")         # g(r), XRD, S(q), DOS, Bands, Thermo, NEB, ...
show(pipe, table="DOS", y="Ga d")            # one column
show(pipe, histogram="Coordination")         # histogram of a particle property
```

In the GUI the plot is linked to the 3D view, like the Plots tab. Dragging an x range selects
those atoms, or those frames on a per-frame plot. The atoms you pick appear on the plot. The DOS
and the band structure follow the picked atoms as projected DOS or fat bands, and clicking a
phonon band plays that mode. Choosing another table, column or histogram in the plot's bar
rewrites the cell. Run as a script, `show` draws the same plot with matplotlib. Add one with
**+ View** or Insert › View, which list the tables and properties of the current frame.

Spectra, diffraction patterns and time series are not separate panels. They are views in the
notebook, next to the steps that compute them.

**Code cells** are any other Python. Printed output appears under them, as does the value of the
last line (as in Jupyter). Matplotlib figures are shown as images, without `plt.show()`
opening a window.

**Styling cells** color, resize or hide atoms by any property. A Python step runs on every
frame, so the style follows the trajectory:

```python
@pipe.add
def style(frame, data):
    cn = data["Coordination"]
    paint(data, cn < 12, "crimson")             # any matplotlib color or RGB
    hide(data, data["Position"][:, 0] > 9)      # cut the block open
    resize(data, "Coordination < 9", 1.3)       # a condition can be an expression string
```

`paint`, `hide`, `isolate` (hide the rest), `fade` and `resize` write new Color, Transparency
and Radius properties, which every renderer draws. Import them with
`from ewald.paint import paint, hide, isolate, fade, resize` (new notebooks already do).

**Text cells** are Markdown. Double-click one to edit it.

The ⋯ menu of a cell has Run, Run this and below, Move up/down, Delete and, for steps and views,
Show as code cell. The `.py` button at the top shows the whole file.

## Notebook tabs

Each open notebook has a tab at the top of the notebook panel. **+** or
File › New notebook (Ctrl+T) opens an empty one. Right-click a tab to rename or close it, or double-click it to rename.
A dot marks unsaved changes. Each notebook has its own namespace and pipeline, and switching to
a notebook that has a pipeline shows it in the viewport.

Opening a data file (File › Open, or `ewald gui file`) makes a notebook for it: the imports, a
`load` cell and one step per modifier the file needs. It replaces the empty notebook you
start with. File › Open on a `.py` file opens it as a notebook. The cells only run after you
choose to (Notebook › Run all), because a notebook is code. Preferences › Python ›
*Run notebooks when opened* can change that to always run them or never run them.

File › Save notebook (Ctrl+S) writes the `.py` file. Sessions (`.ewald`) still load and save
from the File menu.

## Plugins and changes from elsewhere

The puzzle icon on the far left opens the **Plugins** side bar, next to the notebook (Window ›
Plugins, Ctrl+Shift+X). It lists the installed plugins, with where each comes from (plugin
folder, pip package or this session) and any error it raised, and the plugins you can install:
ewald's example plugins and every catalog in Preferences › Python › *Plugin catalogs* (see
[Packaging and sharing](../plugins/packaging.md#plugin-catalogs)). Type to search, or filter with
`@installed`, `@available` or a kind such as `@modifier`. *Install* copies a plugin file into the
plugin folder and loads it at once; a `pip:` entry runs pip in the Python ewald runs in, after
you confirm the command. *Disable* switches a plugin off without removing it
([Switching a plugin off](../plugins/packaging.md#switching-a-plugin-off)); disabled,
incompatible and failed plugins say so in their row. *Remove* moves a file to the plugin
folder's `.removed` folder (its steps stay available until ewald restarts). Selecting a modifier plugin shows its documentation
and *Add as a step in the notebook*. *Create…* asks Claude to write a new plugin.

Plugin modifiers ([Writing plugins](../plugins/index.md)) are steps like the built-in ones.
Insert › Plugin step (or **+ Plugin**) lists the installed ones and writes the import line for
you, for example `from ewald.modifiers import ShiftZ`. *Load plugin file…* adds a cell that
registers the modifiers of a `.py` file:

```python
from ewald.plugins import load_plugin_file
load_plugin_file("~/plugins/local_strain.py")
```

Modifiers added outside the cells become step cells in the right place. That covers the Python
console, Ask Claude, a plot's *Keep as selection*, the bond tool and the phonon band plot. Steps
whose modifier was removed disappear. The console sees the notebook's variables, and `notebook`
is the notebook object itself.

## From Python

```python
from ewald.notebook import Notebook

nb = Notebook.load("analysis.py")
nb.run_all()                      # False if a cell failed; errors are in cell.outputs
data = nb.pipeline.compute(0)
nb.add_view(histogram="Coordination")
nb.save()
```

`Notebook.from_pipeline(pipe)` makes a notebook from a pipeline built in a script.

## Layout

Preferences › General › *Window layout* switches between `notebook` (the default) and `classic`,
the OVITO-style pipeline list and properties panel, after a restart. In the notebook layout,
display settings (particle radii and colors, bonds, cell and visual plugins) are in the
**Display** panel on the right, tabbed with Ask Claude. View › Focus on the viewport (Ctrl+\\)
hides the panels, and pressing it again brings them back. View › *Minimal mode* (Ctrl+Shift+M)
hides the menu bar, toolbars and status bar, leaving the notebook and the viewport. Every
command stays one search away in the command palette (Ctrl+K), menu shortcuts keep working,
and Ctrl+Shift+M brings the menus back. *UI theme* picks Paper (light, the
default) or Graphite (dark).
