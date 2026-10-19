# Notebooks

In the desktop GUI, your analysis is a notebook: Python cells that build the pipeline, plots of
its results, and notes, saved as one `.py` file. It works like Jupyter, and the file stays a
plain script, so `python analysis.py` runs it without the GUI. VS Code, JupyterLab with
jupytext and Spyder open it as a notebook too.

```python
# %% [markdown]
# # Cu void annealing

# %%
from ewald import import_file
from ewald.modifiers import *
from ewald.views import show
pipe = import_file("cu_void.lammpstrj")

# %%
pipe.modifiers.append(CoordinationAnalysis(cutoff=3.08))

# %%
pipe.modifiers.append(ExpressionSelection(expression="Coordination < 12"))

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
`import_file`) re-runs the cells below it on the new pipeline.

There are four kinds of cell.

**Step cells** are a single `pipe.modifiers.append(Modifier(...))`, optionally preceded by
imports. The notebook shows them as a step: the modifier's name, its parameters, a checkbox to
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

**Text cells** are Markdown. Double-click one to edit it.

The ⋯ menu of a cell has Run, Run this and below, Move up/down, Delete and, for steps and views,
Show as code cell. The `.py` button at the top shows the whole file.

## Notebooks on the left

The rail on the left of the notebook has one button per open notebook, like tabs. **+** or
File › New notebook (Ctrl+T) opens an empty one. Right-click a notebook to rename or close it.
A dot marks unsaved changes. Each notebook has its own namespace and pipeline, and switching to
a notebook that has a pipeline shows it in the viewport.

Opening a data file (File › Open, or `ewald gui file`) makes a notebook for it: the imports, an
`import_file` cell and one step per modifier the file needs. It replaces the empty notebook you
start with. File › Open on a `.py` file opens it as a notebook. The cells only run after you
choose to (Notebook › Run all), because a notebook is code. Preferences › Python ›
*Run notebooks when opened* can change that to always run them or never run them.

File › Save notebook (Ctrl+S) writes the `.py` file. Sessions (`.ewald`) still load and save
from the File menu.

## Plugins and changes from elsewhere

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
hides the panels, and pressing it again brings them back. *UI theme* picks Paper (light, the
default) or Graphite (dark).
