# Writer plugins

A writer plugin adds an output format to `export_file`, the GUI's File › Export data and the
Claude `export_file` tool. It is the export-side twin of a [reader plugin](readers.md): the same
plugin folder, the same kind of entry point, the same registry, and `ewald plugins` lists both.

## A writer in one file

```python
# <config dir>/plugins/positions_csv.py
from ewald.plugin_api import writer


@writer("positions-csv", patterns=("*.pcsv",), description="positions as CSV")
def write_positions_csv(path, frames, precision=6):
    with open(path, "w") as fh:
        for k, data in enumerate(frames):          # frames is an iterator of DataCollection
            for t, p in zip(data.types, data.positions):
                fh.write(f"{k},{t}," + ",".join(f"{x:.{precision}f}" for x in p) + "\n")
```

Then `export_file(pipe, "run.pcsv", precision=3)` uses it, and so does File › Export data.
Keyword arguments of `export_file` are passed to the writer.

`frames` is an iterator: when a pipeline is exported, each frame is computed just before the
writer asks for it, so a writer that writes one frame at a time never holds the trajectory in
memory. Do not call `len(frames)`; write as you go, or collect into a list only when the format
needs a frame count up front.

## One-frame formats

A format that holds one structure (POSCAR, a LAMMPS data file, a cube file) registers with
`multi_frame=False` and receives one `DataCollection` instead of an iterator:

```python
@writer("qe-input", patterns=("*.pwi",), description="Quantum ESPRESSO pw.x input", multi_frame=False)
def write_pwi(path, data, pseudopotentials=None):
    import ase.io

    ase.io.write(path, data.to_ase(), format="espresso-in", pseudopotentials=pseudopotentials or {})
```

Exporting several frames to such a writer raises an error that asks for `frames=[i]`; the GUI
and the AI tool write the current frame.

## How the writer is chosen

1. `export_file(..., format="positions-csv")` uses the writer with that name or alias
   (`aliases=(...)`); a name no writer has is passed to `ase.io.write` as an ASE format.
2. Without `format=`, writers are tried by `priority` (lowest first; plugins default to 50)
   and the first whose `patterns` match the output file name wins. Matching ignores case and a
   `.gz`/`.bz2`/`.xz` suffix. Nothing matching means ASE, which guesses from the extension. For a compressed name
   (`run.extxyz.gz`) the writer writes an uncompressed temporary file, which ewald then
   compresses, so writers never deal with compression.
3. Registering an existing name replaces that writer, built-in ones included.
4. `priority=None` registers a writer that is only used with `format=`.

The built-in writers (`poscar`, `lammps-data`, `lammps-dump`, `extxyz`, `xsf`, `h5md`,
`md-trajectory`, `ase-db`, `ase-traj`, `cif`, `cube`, `vasp-volumetric`, `ase`) are listed with
their patterns on [File formats](../user/formats.md#writing-files).

## What a writer should handle

- Read `Position`, `Particle Type` and `data.cell` (rows of `cell.matrix` are the cell vectors,
  `cell.origin` the corner, `cell.pbc` the periodic axes); an invalid cell (`not cell.is_valid`)
  means a molecule.
- Write other properties under the file format's own names (`Force` as `forces`, ...), and
  per-frame values from `data.attributes` (`Timestep`, `Time`, `Energy`).
- Never modify the arrays of `data`; they are shared with the pipeline cache.
- Raise `ValueError` with a clear message for data the format cannot hold (a changing atom
  count in a fixed-size format, a grid with components where one scalar is needed).

## Registering without the decorator, and in a package

`ewald.plugins.register_writer(name, factory, patterns=..., description=..., priority=...,
aliases=..., multi_frame=..., category=...)` does the same as `@writer`. An installed package
declares writers as entry points in the `ewald.writers` group:

```toml
[project.entry-points."ewald.writers"]
positions-csv = "my_pkg.io:write_positions_csv"
```

An entry-point function can carry `patterns`, `description`, `priority`, `aliases` and
`multi_frame` as attributes. See [Packaging and sharing](packaging.md).
