# Reproducing papers

Some computational papers publish their trajectories and structures on GitHub. ewald knows a
few of them: `examples.fetch(name)` downloads one dataset into the cache, checks every file
against its SHA-256 and returns the folder, and a script in
[`examples/papers/`](https://github.com/frssp/ewald/blob/main/examples/papers/) reruns the paper's analysis with ewald modifiers and
compares the numbers. A download shows a progress bar with its size, speed and time left (under
the cell in the app, on stderr in a terminal), and Interrupt stops it without leaving a partial
file. The data is not shipped with ewald; each file is pinned to a commit of the
authors' repository, so a script always sees the bytes it was written for.

```bash
ewald examples --papers                 # list the datasets, sizes, licenses and scripts
ewald examples --fetch li6ps5cl-aimd    # download one (printed folder)
python examples/papers/li6ps5cl_diffusion.py out/   # download if needed, analyze, write out/*.png
```

The scripts are also notebooks (`# %%` cells): open one with **File › Open** in the app and run
its cells to see the structures in the 3D view while the numbers come out under the cells.

```python
from ewald import examples, import_file

folder = examples.fetch("li6ps5cl-aimd")     # ~/.cache/ewald/papers/li6ps5cl-aimd/
pipe = import_file(folder / "XDATCAR.gz")
```

Downloads land in `<cache dir>/papers/<name>/` (`~/.cache/ewald` on Linux,
`~/Library/Caches/ewald` on macOS, `%LOCALAPPDATA%\ewald\cache` on Windows; `EWALD_CACHE_DIR`
moves it). A file already there with the right checksum is not downloaded again; a download
whose checksum differs is deleted and raises `ValueError`.

## Datasets

| Name | Paper | Data | What the script reproduces |
|---|---|---|---|
| `li6ps5cl-aimd` | Morgan, *Chem. Mater.* 33, 2004 (2021); McCluskey, Coles, Morgan, *JCTC* 21, 79 (2025) | VASP XDATCAR, 416 atoms, 140 frames, 0.8 MB, [bjmorgan/kinisi](https://github.com/bjmorgan/kinisi) (MIT) | Li MSD and D* with `MeanSquaredDisplacement`, against kinisi: [`li6ps5cl_diffusion.py`](https://github.com/frssp/ewald/blob/main/examples/papers/li6ps5cl_diffusion.py) |
| `llzo-arrhenius` | McCluskey, Coles, Morgan, *JCTC* 21, 79 (2025) | XTC + GRO at 500, 600, 700, 800 K, 1536 atoms, 29 MB, same repository (MIT); needs MDAnalysis | Arrhenius plot and activation energy with `MeanSquaredDisplacement`, Nernst-Einstein conductivity with `IonicConductivity`: [`llzo_arrhenius.py`](https://github.com/frssp/ewald/blob/main/examples/papers/llzo_arrhenius.py) |
| `carbon-synth` | Gardner, Beaulieu, Deringer, *Digital Discovery* (2023) | 6 of 546 C-GAP-17 melt-quench-anneal runs (200 atoms, 211 frames, local energies), 31 MB, [jla-gardner/carbon-data](https://github.com/jla-gardner/carbon-data) (MIT per load-atoms) | Fig. 2a, local energy by sp/sp2/sp3 coordination (1.85 A cutoff), and sp3 fraction against density with `CoordinationAnalysis`: [`carbon_coordination.py`](https://github.com/frssp/ewald/blob/main/examples/papers/carbon_coordination.py) |
| `asi-paracrystalline` | Rosset, Drabold, Deringer, *Nat. Commun.* 16, 2360 (2025) | 100,000-atom paracrystalline and polycrystalline a-Si, the paper's S(q) and Laaziri's experiment, 11 MB, [lamr18/aSi-data](https://github.com/lamr18/aSi-data) (no license stated) | Fig. 5, S(q) with `StructureFactor` against the paper and experiment; g(r), coordination and diamond-like fraction with `CoordinationAnalysis` and `IdentifyDiamond`: [`asi_paracrystalline.py`](https://github.com/frssp/ewald/blob/main/examples/papers/asi_paracrystalline.py) |
| `argyrodite-disorder` | Morgan, *Chem. Mater.* 33, 2004 (2021) | 37 VASP XDATCAR.gz of Li6PS5I and Li6PS5Cl with 0/50/100 % anion site disorder (416 atoms, 700 frames each, as consecutive segments), 25 MB, [bjmorgan/data_argyrodite_disorder](https://github.com/bjmorgan/data_argyrodite_disorder) (MIT) | the repository's `msd.pdf`, Li MSD against disorder, with `MeanSquaredDisplacement` on a wildcard trajectory: [`argyrodite_disorder.py`](https://github.com/frssp/ewald/blob/main/examples/papers/argyrodite_disorder.py) |
| `tiof2` | Morgan et al., *J. Am. Chem. Soc.* (2024), doi:10.1021/jacs.4c06304 | four CP2K-relaxed 4x4x4 cells (CIF) and the 6x6x6 GA anion arrangement (POSCAR), 0.1 MB, [bjmorgan/data_TiOF2](https://github.com/bjmorgan/data_TiOF2) (CC BY 4.0) | the Ti-F/Ti-O bond-length table and the TiO(6-n)Fn octahedra counts with `CreateBonds`, partial RDFs with `CoordinationAnalysis`: [`tiof2_octahedra.py`](https://github.com/frssp/ewald/blob/main/examples/papers/tiof2_octahedra.py) |
| `ammonia-borane-metad` | Rizzi, Polino, Sicilia, Russo, Parrinello, *Angew. Chem. Int. Ed.* (2019); PLUMED-NEST 19.082 | HILLS of 4 metadynamics walkers and the authors' `sum_hills` fes.dat, 4.6 MB, [gilgalad33/Ammonia-Borane-Dehydrogenation](https://github.com/gilgalad33/Ammonia-Borane-Dehydrogenation) (no license stated) | the AB -> AaDB free energy surface with `fes_from_hills` of the ewald-plumed plugin, point by point against fes.dat: [`ammonia_borane_fes.py`](https://github.com/frssp/ewald/blob/main/examples/papers/ammonia_borane_fes.py) |
| `alanine-dipeptide-metad` | Trapl, Spiwok, *The R Journal* 14(3) (2022); PLUMED-NEST 20.023 | phi/psi HILLS (30,000 hills) and plumed.dat, 4.9 MB, [spiwokv/metadynminerdata](https://github.com/spiwokv/metadynminerdata) (no license stated) | the Ramachandran free energy surface and its basins with `fes_from_hills` (ewald-plumed plugin): [`alanine_dipeptide_fes.py`](https://github.com/frssp/ewald/blob/main/examples/papers/alanine_dipeptide_fes.py) |
| `nbo2f-anion-order` | Morgan, arXiv:2608.04841 (2026) | 92 VASP OUTCARs of NbO2F anion orderings in a 3x3x3 supercell (108 atoms, full relaxations), 19 MB, [bjmorgan/data_nbo2f_chirality](https://github.com/bjmorgan/data_nbo2f_chirality) (data CC BY 4.0) | the DFT ground state as the P3_121 / P3_221 enantiomer pair with `load_runs`, `check_run` and `SymmetryAnalysis`, and cis-NbO4F2 octahedra from the F-Nb-F angles: [`nbo2f_anion_order.py`](https://github.com/frssp/ewald/blob/main/examples/papers/nbo2f_anion_order.py) |
| `fe-vacancy-neb` | Ong et al., *Comput. Mater. Sci.* 68, 314 (2013) (pymatgen) | a VASP climbing-image NEB folder `00`-`04` of a vacancy hop in fcc Fe (Fe31), 34 MB, from pymatgen's NEBAnalysis tests, [materialsproject/pymatgen](https://github.com/materialsproject/pymatgen) (MIT) | the 325 meV barrier pymatgen's test asserts, with `load` on the folder (`UnwrapTrajectories`, `NebEnergyProfile`, `OverlayFrames`) and `CommonNeighborAnalysis`: [`fe_vacancy_neb.py`](https://github.com/frssp/ewald/blob/main/examples/papers/fe_vacancy_neb.py) |
| `mgb2-phonons` | Togo, *J. Phys. Soc. Jpn.* 92, 012001 (2023); band path of Bohnen, Heid, Renker, *PRL* 86, 5771 (2001) | phonopy_disp.yaml and FORCE_SETS of a 3x3x2 MgB2 supercell (VASP forces), 15 kB, [phonopy/phonopy](https://github.com/phonopy/phonopy) (BSD-3-Clause); needs the ewald-phonopy plugin | the band structure along Bohnen's path, the Gamma modes named from their eigenvectors and the E2g mode played with `PhononMode`: [`mgb2_phonons.py`](https://github.com/frssp/ewald/blob/main/examples/papers/mgb2_phonons.py) |

## What the scripts find

- **Li6PS5Cl**: the Li MSD matches kinisi to three digits at every lag
  (0.445/0.707/0.892 A^2 at 0.1/0.2/0.3 ps), and D* = 1.25e-5 cm^2/s from 3 ps on agrees with
  kinisi's 1.23(10)e-5 cm^2/s.
- **LLZO**: Ea = 0.118 eV, against 0.124 eV from kinisi's D values. The MSDs agree to about 2 %,
  but kinisi's covariance-weighted D is about 0.6x a straight-line fit of the same MSD; compare
  activation energies rather than single D values.
- **Amorphous carbon**: sp atoms sit about 1.3 eV above sp2 and sp3 (whose distributions overlap),
  as in Fig. 2a; the final structures go from sp2 to 92 % sp3 between 2.5 and 3.5 g/cm^3.
- **a-Si**: ewald's S(q) (Fourier transform of g(r), 20 A, Lorch window) correlates at 0.995 with
  the paper's Debye S(q) for the paracrystalline model and follows the experiment; the
  polycrystalline model's Bragg peaks come out smoothed (0.92). IdentifyDiamond finds 11 % and
  94 % diamond-like atoms.
- **Argyrodite disorder**: the six MSD curves match the repository's `msd.pdf`; at 60 ps
  Li6PS5I reaches 8.6/22.4/10.5 A^2 and Li6PS5Cl 11.3/43.1/13.7 A^2 at 0/50/100 % disorder, so
  partial anion disorder is what makes Li fast. (The kinisi example XDATCAR is run 1 of
  Li6PS5Cl at 50 %.)
- **TiOF2**: max, min, mean and mean absolute deviation of the Ti-F and Ti-O bonds in all four
  relaxed structures agree with the repository's notebook to the printed 0.001 A, and the 6x6x6
  structure has the same 194 TiO2F4, 11 TiOF5 and 11 TiO3F3 octahedra.
- **Ammonia borane**: `fes_from_hills` reproduces `plumed sum_hills`' fes.dat to RMS 0.031 and
  max 0.18 kJ/mol over 201 x 201 points, without PLUMED installed.
- **Alanine dipeptide**: C7eq/PII is the global minimum, beta/C5 1.6, alpha_R 2.7 and alpha_L
  5.3 kJ/mol above it.
- **NbO2F**: all 92 relaxations reached EDIFFG (`check_run`). The two lowest, 9 ueV/atom apart,
  are P3_221 and P3_121 (the enantiomer pair the paper calls orbit 11); the next orderings lie
  0.5-3 meV/atom higher and are also cis everywhere, period-three orderings with mixed cis/trans
  octahedra 5-20 meV/atom and random ones 20-270 meV/atom above.
- **Fe vacancy NEB**: barrier 325.2 meV (spline maximum 325.2 meV), saddle at image 2 at 0.497 of
  the 3.00 A path, as pymatgen's NEBAnalysis test asserts (325.2 meV at 0.500). Atom 7 hops
  2.45 A into the vacancy; CNA finds 19 FCC atoms of 31, the other 12 being the vacancy's neighbours.
- **MgB2**: E1u 332, A2u 399, E2g 576 and B1g 686 cm^-1 at Gamma, the frequencies phonopy itself
  gives for these force constants; the E2g branch stays between 540 and 580 cm^-1 along Gamma-A.

## Adding a dataset

Add a `Paper` to `PAPERS` in `src/ewald/examples/papers.py` with each file's raw GitHub URL at a
commit (`https://raw.githubusercontent.com/<owner>/<repo>/<40-hex sha>/<path>`), its size and
SHA-256, the license as the repository states it, and a script in `examples/papers/`; list it in
the table above and in `examples/README.md` (`tests/test_papers.py` checks all of this). Prefer
permissively licensed data under ~30 MB per dataset, or a subset of a larger one. A file name
may include folders (`00/OUTCAR`, `config_000/OUTCAR`) when the data is a NEB folder or a series
of runs that `load` and `load_runs` should see as such.
