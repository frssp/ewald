# DOS and band structure

A VASP run that wrote its density of states and eigenvalues opens with them attached: the
vasprun.xml and vaspout.h5 readers put the total and projected DOS, the band structure and the
band projections into every frame, and the GUI's Plots tab draws them next to the structure.
Atoms picked in the viewport show up as their projected DOS, or as fat bands.

```python
from ewald import import_file

data = import_file("vasprun.xml").compute(-1)     # or vaspout.h5
es = data.attributes["Electronic Structure"]
print(data.attributes["Fermi Energy"], data.attributes["Band Gap"])   # eV
print(es.band_gap())               # {'gap': 0.9, 'vbm': ..., 'cbm': ..., 'direct': False}

d_of_ti = es.pdos_of(data.types == "Ti", "d")    # (nspin, nE), states/eV
weights = es.band_weights([0, 1], "p")           # (nspin, nk, nbands), fat-band weights
```

## What is read

| Quantity | vasprun.xml | vaspout.h5 |
|---|---|---|
| Fermi energy | `<dos>/<i name="efermi">` | `results/electron_dos/efermi` |
| Total DOS | `<dos>/<total>` | `results/electron_dos/dos` |
| Projected DOS (LORBIT >= 10) | `<dos>/<partial>` | `results/electron_dos/dospar`, orbital names from `results/projectors/lchar` |
| Eigenvalues, occupations | `<eigenvalues>` | `results/electron_eigenvalues` (`_kpoints_opt` when KPOINTS_OPT was used) |
| k-points, path | `<kpoints>`; labels from a line-mode `KPOINTS` next to the file | `input/kpoints` (mode, points per line, labels) |
| Band projections (LORBIT >= 10) | `<projected>`, read when first used | `results/projectors/par`, read when first used |

`ElectronicStructure` (in `ewald.electronic`) keeps the arrays as VASP has them: `dos` is
`(nspin, nE)`, `pdos` `(natoms, nspin, norbitals, nE)`, `eigenvalues` and `occupations`
`(nspin, nk, nbands)`, `projections` `(nspin, nk, nbands, natoms, norbitals)`, energies in eV and
absolute (subtract `efermi` for plots). `nspin` is 2 for ISPIN = 2; a non-collinear run keeps
only the total (the first of its four components). Without these sections, as in a plain MD
vasprun.xml, nothing is added.

Orbitals are picked by name (`"dxy"`), by angular momentum (`"s"`, `"p"`, `"d"`, `"f"`, which
covers the lm-decomposed orbitals of LORBIT = 11 and the plain `s p d` of LORBIT = 10), or as a
list. Atoms are indices or a boolean mask.

The band path follows VASP's line mode: the distance (1/Å, without 2π, as in phonopy and
py4vasp) does not grow between the end of one line and the start of the next, `es.breaks` lists
where the path jumps (W|K), and `es.ticks` holds the high-symmetry labels (`\Gamma` and `G`
become Γ). Without a KPOINTS file next to vasprun.xml the line ends are labelled by their
reduced coordinates. In a hybrid-functional band run (zero-weight k-points added to the SCF
mesh) the zero-weight points are taken as the band path.

## Tables

Two tables are added, so the DOS and bands can also be saved or plotted like any other table:

- `DOS`: `Energy` (E − E_F), `Total`, one column per element and one per element and l
  (`Ti d`). Spin-polarized runs have `(up)` and `(down)` columns, with the down channel negated
  so that it plots below the axis.
- `Bands`: `Distance` along the path and one column per band (E − E_F), `Band n (up)` and
  `Band n (down)` for ISPIN = 2.

## In the GUI

The Plots tab opens the `DOS` table first. With "y" set to "(all)" it draws the total DOS
shaded with the per-element PDOS on top; picking a column draws that one. The `Bands` table is
drawn as a band structure with its high-symmetry labels, spin-down bands dashed. Both show
6 eV around the Fermi energy (dashed line) and the band gap in the corner. The energy window
is shared by both plots: type it in the **E** boxes, drag an energy range on the DOS (or on
the bands, vertically) to zoom to it, or use the wheel; a double-click goes back to ±6 eV. In
a notebook the window is part of the view cell, `show(pipe, table="DOS", xlim=(-3, 2))` (the
bands take `ylim=`).

Select atoms in the viewport and the plot follows: on the DOS their summed PDOS is filled in the
selection color, and on the bands each state gets a marker whose size is the selected atoms'
weight in it (fat bands; squares for spin down). When a column like `Ti d (up)` is picked, the
selection is restricted to that orbital (d). "Save plot…" writes PDF, PNG or SVG.
