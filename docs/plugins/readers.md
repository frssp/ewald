# Reader plugins

A reader plugin teaches ewald a file format. After it is registered, `import_file`, the GUI's
File › Load file and the Claude `load_structure` tool all read that format.

## The smallest reader

A function that takes the path and returns the frames is enough:

```python
# <config dir>/plugins/cp2k_xyz.py        ("ewald plugins" prints the folder)
import numpy as np
from ewald.plugin_api import Cell, DataCollection, reader


@reader("cp2k-xyz", patterns=("*-pos-*.xyz",), description="CP2K position trajectory")
def read_cp2k_xyz(path, cell=None, **kwargs):
    frames = []
    with open(path) as fh:
        while header := fh.readline().strip():
            n = int(header)
            fh.readline()                                          # comment line
            rows = [fh.readline().split() for _ in range(n)]
            types = [r[0] for r in rows]
            positions = np.array([r[1:4] for r in rows], float)
            data = DataCollection(positions, types)
            if cell is not None:
                data.cell = Cell(cell)
            frames.append(data)
    return frames
```

```python
pipe = import_file("water-pos-1.xyz", cell=np.diag([12.4, 12.4, 12.4]))
```

The function may return a list of `DataCollection` or `ase.Atoms`, a single frame, or a
`FrameReader`. A list is held in memory, which is fine for files up to a few hundred MB.
Keyword arguments given to `import_file` are passed through, so a reader can take options such
as `cell=` or `type_map=`.

## Lazy readers for large trajectories

For large files, subclass `FrameReader`: index the file once in `__init__`, then read one frame
in `read(i)`. Only the frames that are looked at are read, and the pipeline keeps a few of them
in its cache.

```python
from ewald.plugin_api import DataCollection, FrameReader, open_binary, reader


@reader("myformat", patterns=("*.myf", "*.myf.gz"), description="MyCode trajectory")
class MyFormatReader(FrameReader):
    def __init__(self, path, type_map=None):
        self.path = path
        self.type_map = type_map or {}
        self.offsets = []                         # byte offset of each frame
        with open_binary(path) as fh:             # handles .gz/.bz2/.xz
            pos = 0
            for line in fh:
                if line.startswith(b"FRAME"):
                    self.offsets.append(pos)
                pos += len(line)

    def __len__(self):
        return len(self.offsets)

    def read(self, index):
        with open_binary(self.path) as fh:
            fh.seek(self.offsets[index])
            ...                                    # parse one frame
        return DataCollection(positions, types)
```

A class is constructed as `cls(path, **kwargs)`. `ewald.io.base` also has `head_lines(path, n)`
for sniffing and `strip_compression(path)` for name checks. The native LAMMPS dump and XDATCAR
readers in `src/ewald/io/` are complete examples of this pattern.

A file the program is still writing (or that a killed run left behind) should give the frames
that are complete. Leave out the half-written last frame and record why with
`self._set_incomplete("12 frame(s) read, the partial last one skipped")`: that sets
`reader.incomplete`, which the GUI shows, and warns with `IncompleteFileWarning`. Raise
`IncompleteFileError` (from `ewald.io.base`, a `ValueError`) when nothing is complete yet. A
`refresh()` method that returns the first frame that may have changed (or None) lets live
reload follow the file.

## How the format is chosen

Every format, built-in or plugin, is a `ReaderPlugin` in one registry (`ewald plugins` lists
them in the order below), and readers are tried by `priority`, lowest first:

| Priority | Readers |
|---|---|
| 5, 6 | built-in volumetric: Gaussian cube, VASP CHGCAR/CHG/PARCHG/AECCAR/LOCPOT/ELFCAR |
| 7, 8 | built-in H5MD, XSF |
| 10-12 | built-in LAMMPS dump, XDATCAR, LAMMPS data |
| 13-15 | built-in ASE database, ASE trajectory, XTC/TRR/DCD/NetCDF (MDAnalysis) |
| 20 | built-in XYZ / extended XYZ |
| 50 | plugins (default; pass `priority=` to change it) |
| 90 | built-in CIF (through ASE) |
| 100 | ASE, for everything else |

1. `import_file(path, format="myformat")` uses the reader with that name or alias
   (`aliases=("my-format",)`); a name no reader has is passed to ASE as an ASE format.
2. Without `format=`, each reader in priority order is asked: its `sniff(path)` function if it
   has one, else its `patterns` matched against the file name (without a `.gz`/`.bz2`/`.xz`
   suffix). The first match wins, so a plugin at the default priority cannot take over files
   the built-in readers recognise (pass `priority=15` to read `*.xyz` files of your own code
before the built-in extxyz reader).
3. Registering a name that already exists replaces that reader, built-in ones included
   (`@reader("xdatcar", sniff=..., priority=11)` swaps in your own XDATCAR reader).
4. `priority=None` registers a reader that is only used with `format=`.

### Sources that are not files

A reader can serve a URI scheme instead of files: `@reader("aiida", schemes=("aiida",))` (or a
`schemes` class attribute) receives every path that starts with `aiida:`, such as
`load("aiida:1234")`, before any other reader is asked. Such a path is passed to the reader as
it is: it is not looked up next to the notebook, globbed, or opened over SSH (otherwise
`aiida:1234` would read as the SSH host `aiida`), and sessions and notebooks store the string
unchanged. Database, archive and web sources (AiiDA, Materials Project, OPTIMADE) fit here; the
[ewald-aiida](https://github.com/frssp/ewald-plugins/tree/main/packages/ewald-aiida) plugin is
the reference. In the GUI such a path opens with File › Open location.

Use `sniff` when the file name says little (`OUTPUT`, `traj.dat`):

```python
from ewald.plugin_api import head_lines

def looks_like_myformat(path):
    lines = head_lines(path, 2)
    return bool(lines) and lines[0].startswith(b"# MyCode")

@reader("myformat", sniff=looks_like_myformat, patterns=("*.myf",))
class MyFormatReader(FrameReader): ...
```

A sniffer must be cheap (read a few lines, never the whole file) and must return `False`
rather than raise for files it does not understand; an exception is logged and treated as
no match.

## What a frame should contain

- `Position` (N×3, Å) and `Particle Type` as element symbols when the file has them. For
  numeric types, accept `type_map={1: "Cu"}` as the built-in LAMMPS readers do.
- `Particle Identifier` when the file has ids. MSD, displacements and Wigner-Seitz match
  particles across frames by it.
- `data.cell = Cell(matrix, origin, pbc)` with the cell vectors as the **rows** of `matrix`.
  Leave the cell empty for molecules without one.
- Other per-atom columns under OVITO names: `Velocity`, `Force`, `Charge`, `Mass`,
  `Potential Energy`. Vector quantities are one N×3 property, not three columns.
- Per-frame values in `data.attributes`: `Timestep`, `Time`, `Energy`, and anything else.

Run `data.summary()` on a frame of your reader to check what it produced.

## Registering without the decorator

`ewald.plugins.register_reader(name, factory, patterns=..., sniff=..., description=..., priority=...,
aliases=..., category=..., schemes=...)` does
the same as `@reader` and can be called from a script or the GUI console. In an installed
package, an entry point registers the reader without any code running at import time; see
[Packaging and sharing](packaging.md). An entry-point class can carry `patterns`, `sniff`, `description`,
`priority`, `aliases`, `schemes` and `category` (`"structure"` or `"volumetric"`, which groups
the GUI file filter) as class attributes.

## Example

`examples/plugins/npz_trajectory.py` is a complete reader for NumPy `.npz`
trajectories with positions, types, cell, forces and energies. Copy it into the plugin folder
and `import_file("run.npz")` works.
