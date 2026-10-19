# Real output files

Unmodified (except where noted) output of real VASP and LAMMPS runs, used by
`tests/test_real_outputs.py` to keep the readers working on what the programs actually write.

| File | Source | What it exercises |
|---|---|---|
| `vasprun.static_silicon.xml.gz` | pymatgen `test-files/io/vasp/fixtures/static_silicon/vasprun.xml.gz` | VASP 5.2.12: `e_0_energy` = 0 after the last `<scstep>`; projections with fewer orbitals than the PDOS |
| `vasprun.etest3.xml.gz` | pymatgen `test-files/io/vasp/outputs/vasprun.etest3.xml.gz` | VASP 5.4.1, 9 ionic steps; energy correction outside the SCF loop |
| `vasprun.GW0.xml.gz` | pymatgen `test-files/io/vasp/outputs/vasprun.GW0.xml.gz` | GW0 run: no `<structure>` inside `<calculation>` |
| `XDATCAR.bad_fmt.gz` | pymatgen `test-files/io/vasp/outputs/XDATCAR.bad_fmt.gz` | coordinate columns run together (`-0.63265286-0.11227753`) |
| `vaspout.line_mode_band_structure.h5` | pymatgen `test-files/io/vasp/outputs/vaspout.line_mode_band_structure.h5.gz`, **trimmed** | VASP 6 band path as zero-weight k-points; `lchar` with 16 orbitals, PDOS with 9 |
| `log.gauss-diel` | LAMMPS `examples/PACKAGES/gauss_diel/log.gauss-diel` | `thermo_style multi` |

`vaspout.line_mode_band_structure.h5` was decompressed and copied without the datasets
`results/projectors/par`, `results/projectors/celtot` (size) and `input/potcar/content` (the POTCAR
itself), with gzip-compressed datasets; everything else is as VASP wrote it.

Sources and licenses:

- pymatgen, https://github.com/materialsproject/pymatgen, commit
  `52e69c26e45b0205e4f5206e30c94a0619c0d64a`, MIT license.
- LAMMPS, https://github.com/lammps/lammps, commit `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`,
  GPL-2.0 license.
