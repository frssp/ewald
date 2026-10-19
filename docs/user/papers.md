# Reproducing papers

Some computational papers publish their trajectories and structures on GitHub. ewald knows a
few of them: `examples.fetch(name)` downloads one dataset into the cache, checks every file
against its SHA-256 and returns the folder, and a script in
[`examples/papers/`](https://github.com/frssp/ewald/blob/main/examples/papers/) reruns the paper's analysis with ewald modifiers and
compares the numbers. The data is not shipped with ewald; each file is pinned to a commit of the
authors' repository, so a script always sees the bytes it was written for.

```bash
ewald examples --papers                 # list the datasets, sizes, licenses and scripts
ewald examples --fetch li6ps5cl-aimd    # download one (printed folder)
python examples/papers/li6ps5cl_diffusion.py out/   # download if needed, analyze, write out/*.png
```

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
| `carbon-synth` | Gardner, Beaulieu, Deringer, *Digital Discovery* 2, 651 (2023) | 6 of 546 C-GAP-17 melt-quench-anneal runs (200 atoms, 211 frames, local energies), 31 MB, [jla-gardner/carbon-data](https://github.com/jla-gardner/carbon-data) (MIT per load-atoms) | Fig. 2a, local energy by sp/sp2/sp3 coordination (1.85 A cutoff), and sp3 fraction against density with `CoordinationAnalysis`: [`carbon_coordination.py`](https://github.com/frssp/ewald/blob/main/examples/papers/carbon_coordination.py) |
| `asi-paracrystalline` | Rosset, Drabold, Deringer, *Nat. Commun.* 16, 2360 (2025) | 100,000-atom paracrystalline and polycrystalline a-Si, the paper's S(q) and Laaziri's experiment, 11 MB, [lamr18/aSi-data](https://github.com/lamr18/aSi-data) (no license stated) | Fig. 5, S(q) with `StructureFactor` against the paper and experiment; g(r), coordination and diamond-like fraction with `CoordinationAnalysis` and `IdentifyDiamond`: [`asi_paracrystalline.py`](https://github.com/frssp/ewald/blob/main/examples/papers/asi_paracrystalline.py) |

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

## Adding a dataset

Add a `Paper` to `PAPERS` in `src/ewald/examples/papers.py` with each file's raw GitHub URL at a
commit (`https://raw.githubusercontent.com/<owner>/<repo>/<40-hex sha>/<path>`), its size and
SHA-256, the license as the repository states it, and a script in `examples/papers/`; list it in
the table above and in `examples/README.md` (`tests/test_papers.py` checks all of this). Prefer
permissively licensed data under ~30 MB per dataset, or a subset of a larger one.
