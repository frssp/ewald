# %% [markdown]
# # Cu MD: displacements over a trajectory
#
# Tutorial 3 of the EWALD manual (docs/user/tutorials/trajectory.md). A short MD run of fcc
# copper with forces: how far atoms move from the first frame, the mean squared displacement,
# and force arrows.

# %%
from ewald import examples, load
from ewald.modifiers import *
from ewald.paint import paint, resize
from ewald.views import show

pipe = load(examples.path("Cu_md.extxyz"))
print(pipe.num_frames, "frames")

# %% displacements from frame 0
pipe.add(CalculateDisplacements(reference_frame=0))

# %% color by how far each atom moved
pipe.add(ColorCoding("Displacement Magnitude", cmap="viridis"))

# %% mean squared displacement
pipe.add(MeanSquaredDisplacement(time_per_frame=0.01))

# %% force arrows (eV/Å × 4 = Å on screen)
pipe.add(VectorVis(property="Force", scaling=4.0))

# %%
show(pipe, table="msd")

# %%
show(pipe, histogram="Displacement Magnitude")
