# Tutorial 1: your first notebook

A block of fcc copper with a spherical void in it, from a short LAMMPS run. We find the atoms on the
surface of the void from their coordination number, color them, cut the block open to see them,
plot the coordination histogram and the radial distribution function, and save the notebook. It
takes about ten minutes. The finished notebook is
[`examples/notebooks/cu_void.py`](https://github.com/frssp/ewald/blob/main/examples/notebooks/cu_void.py).

![The finished notebook](../../images/notebook.png)

## 1. Open the file

Start the GUI with `ewald gui` and choose File › Open example › `Cu_void.lammpstrj`. The notebook on
the left gets its first cell, which imports what notebooks use and loads the file (the GUI writes
the file's full path; the tutorial notebook uses `examples.path` so it runs anywhere):

```python
import numpy as np

from ewald import all_frames, each_frame, examples, load
from ewald.modifiers import *
from ewald.paint import fade, hide, isolate, paint, resize
from ewald.views import show

pipe = load(examples.path("Cu_void.lammpstrj"))
```

`pipe` is the notebook's pipeline, and the Structure view in the middle always shows it. The timeline
under the view has the run's 5 frames. Drag in the view to rotate it, scroll to zoom, and rest the
mouse on an atom to see its index and position. Your own file works the same way: File › Open, or
`ewald gui dump.lammpstrj`.

## 2. Coordination

Click **+ Step**, type `coord` and pick Coordination analysis. Set the cutoff to 3.0 Å (between the
first and second neighbor shells of Cu, 2.55 and 3.61 Å) and press Apply. The step cell is

```python
pipe.add(CoordinationAnalysis(cutoff=3.0))
```

and under it the notebook lists what the step made: a particle property `Coordination` and a table
`coordination-rdf`, with a small histogram. Atoms in the bulk have 12 neighbors; the 78 atoms on
the void surface have 8 to 11.

## 3. Style the surface atoms

Add a code cell (**+ Code**) and type:

```python
@pipe.add
@each_frame
def style(frame, data):
    paint(data, "Coordination < 12", "#C8551F")   # surface atoms
    paint(data, "Coordination < 9", "gold")       # void edges and corners
    hide(data, "Position.X > 8")                  # cut the block open
```

Run it with Ctrl+Enter. `@pipe.add` makes the function a step after the coordination step, and
`@each_frame` says it runs again on every frame (`frame` is the frame number, `data` that frame's
atoms), so the colors follow the atoms through the trajectory: press play under the view to check. `paint` takes any matplotlib color and `hide` removes atoms from the picture (not
from the data). The condition is an expression on the particle properties (`Position.X` is the x
component), so the legend in the corner of the view and of rendered figures says which condition
each color stands for. A boolean array (`data["Coordination"] < 12`) works too, but the legend can
then only call it "a mask in code". `isolate`, `fade` and `resize` work the same way.

## 4. Plot

Click **+ View** and choose the histogram of `Coordination`, then add a second view of the table
`coordination-rdf`:

```python
show(pipe, histogram="Coordination")
show(pipe, table="coordination-rdf")
```

Views are live. Drag over the bars from 8 to 11 on the histogram and those atoms are selected in
the Structure view; *Keep as selection* turns that range into a step (`ExpressionSelection`) so the
selection is part of the notebook. Shift+click atoms in the view and they show up on the plot.

## 5. Over the whole run

The void changes as the block anneals. A step marked `@all_frames` reads every frame once and
leaves a result that every frame shares, here a table of the surface atoms against time:

```python
@pipe.add
@all_frames
def surface(frames, data):
    data.tables["surface"] = {
        "Timestep": np.array([d.attributes["Timestep"] for d in frames]),
        "Surface atoms": np.array([np.count_nonzero(d["Coordination"] < 12) for d in frames]),
    }

show(pipe, table="surface")
```

`frames` gives each frame as it enters the step, with the coordination step above already
applied (`len(frames)`, `frames[k]`, `for d in frames`). The surface grows from 78 to 107 atoms in
the last two frames. A table whose first column is `Timestep` is a time series: a marker follows
the current frame, a click on the curve goes to that frame and dragging over a range selects
frames. The step runs again only when a step above it or the file changes, not during playback.

Last, a code cell for a number to quote:

```python
data = pipe.compute(0)
surface = data["Coordination"] < 12
print(f"{int(surface.sum())} surface atoms of {data.count}")
```

which prints `78 surface atoms of 462`.

## 6. Save

File › Save notebook (Ctrl+S). Save it as `cu_void.py` to keep it in git: `python cu_void.py` runs
it without the GUI and draws the plots with matplotlib. Save it as `cu_void.ewd` to send it to
someone: the file also carries the outputs, the plots as SVG, PNG and CSV, and the trajectory
itself, since it is small. See [Notebooks](../notebooks.md#notebook-files).

## Going further

- Ask AI (the AI button in the activity bar opens the chat left of the notebook, or Ctrl+J with atoms picked) "what are these atoms?" about the gold
  ones; its answer can add steps, which appear as cells ([AI integration](../claude.md)).
- Replace the coordination step with `PolyhedralTemplateMatching()` or
  `CommonNeighborAnalysis()` to classify the local structure instead ([Modifiers](../modifiers.md)).
- View › Minimal mode (Ctrl+Shift+M) leaves only the notebook and the view; Ctrl+K finds any
  command, and Ctrl+Shift+M again (or **Menus** in the bottom left corner) brings the menus back.

Next: [Tutorial 2: a crystal structure figure](structure-figure.md).
