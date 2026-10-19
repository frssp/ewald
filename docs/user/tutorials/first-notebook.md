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
from ewald import examples, load
from ewald.modifiers import *
from ewald.paint import fade, hide, isolate, paint, resize
from ewald.views import show

pipe = load(examples.path("Cu_void.lammpstrj"))
```

`pipe` is the notebook's pipeline, and the 3D view in the middle always shows it. The timeline
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
def style(frame, data):
    cn = data["Coordination"]
    paint(data, cn < 12, "#C8551F")           # surface atoms
    paint(data, cn < 9, "gold")               # edges and corners of the void
    hide(data, data["Position"][:, 0] > 8)    # cut the block open
```

Run it with Ctrl+Enter. `@pipe.add` makes the function a step that runs on every frame, after the
coordination step, so the colors follow the atoms through the trajectory: press play under the
view to check. `paint` takes any matplotlib color, `hide` removes atoms from the picture (not from
the data), and the condition can also be a string such as `"Coordination < 12"`. `isolate`,
`fade` and `resize` work the same way.

## 4. Plot

Click **+ View** and choose the histogram of `Coordination`, then add a second view of the table
`coordination-rdf`:

```python
show(pipe, histogram="Coordination")
show(pipe, table="coordination-rdf")
```

Views are live. Drag over the bars from 8 to 11 on the histogram and those atoms are selected in
the 3D view; *Keep as selection* turns that range into a step (`ExpressionSelection`) so the
selection is part of the notebook. Shift+click atoms in the view and they show up on the plot.

Last, a code cell for a number to quote:

```python
data = pipe.compute(0)
surface = data["Coordination"] < 12
print(f"{int(surface.sum())} surface atoms of {data.count}")
```

which prints `78 surface atoms of 462`.

## 5. Save

File › Save notebook (Ctrl+S). Save it as `cu_void.py` to keep it in git: `python cu_void.py` runs
it without the GUI and draws the two plots with matplotlib. Save it as `cu_void.ewd` to send it to
someone: the file also carries the outputs, the plots as SVG, PNG and CSV, and the trajectory
itself, since it is small. See [Notebooks](../notebooks.md#notebook-files).

## Going further

- Ask Claude (the AI button in the activity bar opens the chat left of the notebook, or Ctrl+J with atoms picked) "what are these atoms?" about the gold
  ones; its answer can add steps, which appear as cells ([Claude integration](../claude.md)).
- Replace the coordination step with `PolyhedralTemplateMatching()` or
  `CommonNeighborAnalysis()` to classify the local structure instead ([Modifiers](../modifiers.md)).
- View › Minimal mode (Ctrl+Shift+M) leaves only the notebook and the view; Ctrl+K finds any
  command.

Next: [Tutorial 2: a crystal structure figure](structure-figure.md).
