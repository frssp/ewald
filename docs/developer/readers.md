# File readers: which ones ewald owns

ewald writes its own reader only where it measurably beats the libraries it could delegate to,
which in practice means large text trajectories. Opening such a file in the GUI needs a frame
count at once and any frame on demand. ASE's `read(path, index=k)` re-parses the whole file, or
everything up to frame k, on every call. Everything else is delegated: XTC/TRR/DCD/NetCDF to
MDAnalysis, and CIF, POSCAR and every other `ase.io.read` format to ASE.

Native readers:

- **Big text trajectories** (LAMMPS dump, extxyz, XDATCAR): index frame offsets once and parse
  one frame on demand with the parallel numba table parser in `io/_textparse.py`.
- **Data ASE drops or cannot read:**
    - OUTCAR/vasprun forces and magnetic moments per ionic step, plus XSF grids.
    - vaspout.h5, H5MD and MDTraj HDF5.
    - NEB folders.
    - log.lammps/OSZICAR thermo tables.

A new native reader needs the same kind of reason; otherwise register ASE or MDAnalysis for the
format.

## Measurements (2026-10-19)

The numbers come from `python benchmarks/readers.py` in the cloud container:

- **Machine:** 4 cores, with the files already in the page cache, so disk speed is not part of the result.
- **Versions:** ASE 3.29, MDAnalysis 2.10, OVITO pip 3.16.1, chemfiles 0.10.4, extxyz (libAtoms C parser) 0.4.6, oxyz 1.2.1.
- **Columns:**
    - **open:** time until the frame count is known. ASE has no index, so for ASE it is a full parse.
    - **random:** mean time for an uncached frame from the back half of the file.
    - **full:** time to read every frame in order.
    - **RSS:** peak memory growth.

LAMMPS dump, 2,000 atoms x 2,000 frames (169 MB):

| library | open | random | full | RSS |
|---|---|---|---|---|
| ewald | 0.60 s | 0.8 ms | 1.3 s | 81 MB |
| OVITO | 0.31 s | 1.3 ms | 1.9 s | 12 MB |
| chemfiles | 0.09 s | 0.9 ms | 1.8 s | 3 MB |
| MDAnalysis | 0.82 s | 5.0 ms | 13.5 s | 2 MB |
| ASE | 3.9 s | 1,309 ms | 3.6 s | 218 MB |

LAMMPS dump, 1,000,000 atoms x 5 frames, ids shuffled (235 MB):

| library | open | random | full | RSS |
|---|---|---|---|---|
| OVITO | 0.62 s | 235 ms | 1.3 s | 188 MB |
| ewald | 0.59 s | 451 ms | 1.5 s | 367 MB |
| ASE | 5.8 s | 2,841 ms | 6.8 s | 489 MB |
| MDAnalysis | 6.1 s | 2,888 ms | 18.0 s | 98 MB |

chemfiles 0.10.4 crashed with a segfault on this file.

extxyz, 1,000 atoms x 2,000 frames with forces (138 MB):

| library | open | random | full | RSS |
|---|---|---|---|---|
| ewald | 0.45 s | 1.0 ms | 1.7 s | 100 MB |
| ewald before the numba parser | 0.39 s | 4.4 ms | 8.1 s | 50 MB |
| OVITO | 0.23 s | 0.7 ms | 1.4 s | 12 MB |
| chemfiles | 0.09 s | 1.0 ms | 1.1 s | 3 MB |
| MDAnalysis (drops cell and forces) | 0.46 s | 1.2 ms | 2.2 s | 1 MB |
| oxyz (Rust, 4 threads) | 0.04 s | 414 ms | 0.42 s | 259 MB |
| extxyz (libAtoms C) | 0.46 s | 409 ms | 0.63 s | 158 MB |
| ASE | 6.8 s | 307 ms | 6.9 s | 258 MB |

The extxyz file with 1,000,000 atoms x 3 frames (225 MB) is not part of the script. A random frame
took 0.85 s with ewald, 0.33 s with OVITO and 3.9 s with ASE.

VASP XDATCAR, 256 atoms x 20,000 frames (162 MB):

| library | open | random | full | RSS |
|---|---|---|---|---|
| ewald | 1.3 s | 0.6 ms | 6.1 s | 4 MB |
| OVITO | 2.8 s | 0.6 ms | 8.0 s | 14 MB |
| ASE | 13.8 s | 12,920 ms | 12.4 s | 197 MB |

## Why not delegate these too

- **chemfiles** is as fast on small frames. However, it has had no release since 2023 and
  crashed with a segfault on the 1M-atom dump.
- **MDAnalysis** parses LAMMPS dumps line by line, which takes 2.9 s for a 1M-atom frame. Its
  XYZ reader drops the extxyz cell and per-atom properties.
- **oxyz and libAtoms extxyz** are the fastest sequential extxyz parsers. Neither keeps an index,
  so `read(path, index=k)` parses from the start each time. The C parser also refuses `index` on
  an open file, so ewald cannot hand it a byte range.
- **OVITO's pip module** is fast. However, it returns OVITO pipeline objects and comes with its
  own license terms.

## Open gaps

- **1M-atom dump frame:** OVITO reads it in about half the time and memory. The difference is
  ewald's per-column copies and the id sort, not the parse.
- **1M-atom extxyz frame:** OVITO is 2.5 times faster. The time goes to assembling ASE's `Atoms`
  and calculator, which keeps names identical to `ase.io.read`.
