# %% [markdown]
# # SrTiO3: octahedra and a figure
#
# Tutorial 2 of the EWALD manual (docs/user/tutorials/structure-figure.md). The space group of a
# perovskite, its TiO6 octahedra, and a publication figure rendered from the notebook.
# SymmetryAnalysis needs spglib: pip install "ewald[symmetry]" (or pip install spglib).

# %%
from ewald import examples, load
from ewald.modifiers import *
from ewald.views import show

pipe = load(examples.path("POSCAR.SrTiO3"))

# %% space group
pipe.add(SymmetryAnalysis(symprec=0.01))

# %% Ti-O bonds up to 2.2 Å
pipe.add(CreateBonds(mode="pairwise", pairs={("Ti", "O"): 2.2}))

# %% TiO6 octahedra
pipe.add(CoordinationPolyhedra(center_types=["Ti"], transparency=0.25))

# %% bond lengths
pipe.add(BondLengthDistribution())

# %%
show(pipe, table="bond-length-distr")

# %%
data = pipe.compute(0)
a = data.attributes
print(a["SymmetryAnalysis.space_group"], a["SymmetryAnalysis.space_group_number"], a["SymmetryAnalysis.wyckoff_sites"])

# %% figure (quality="publication", the default, for the final version)
pipe.render("srtio3.png", size=(1200, 900), direction=(3, 1.4, 1), theme="VESTA", radius_scale=0.5,
            boundary=(-0.05, 1.05), type_legend=True, renderer="raytrace", quality="draft")
