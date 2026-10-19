# %% [markdown]
# # Cu void: surface atoms
#
# Tutorial 1 of the EWALD manual (docs/user/tutorials/first-notebook.md). A copper block with a
# spherical void: find the atoms on the void surface by their coordination, color them, cut the
# block open and look at the coordination histogram and g(r).

# %%
from ewald import examples, load
from ewald.modifiers import *
from ewald.paint import fade, hide, isolate, paint, resize
from ewald.views import show

pipe = load(examples.path("Cu_void.lammpstrj"))

# %% coordination
pipe.add(CoordinationAnalysis(cutoff=3.0))

# %% style: surface atoms, every frame
@pipe.add
def style(frame, data):
    cn = data["Coordination"]
    paint(data, cn < 12, "#C8551F")           # surface atoms
    paint(data, cn < 9, "gold")               # edges and corners of the void
    hide(data, data["Position"][:, 0] > 8)    # cut the block open

# %%
show(pipe, histogram="Coordination")

# %%
show(pipe, table="coordination-rdf")

# %%
data = pipe.compute(0)
surface = data["Coordination"] < 12
print(f"{int(surface.sum())} surface atoms of {data.count}")
