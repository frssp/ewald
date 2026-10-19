# Reader plugins

> **State:** the `ewald.plugins` module described here is in review (PR #14) and not yet on
> `main`. Until it is merged, a reader can be used by wrapping it in a pipeline source
> yourself; see "Without the registry" at the end.

A reader plugin teaches ewald a file format. After it is registered, `import_file`, the GUI's
File › Load file and the Claude `load_structure` tool all read that format.

## The smallest reader

A function that takes the path and returns the frames is enough:

```python
# <config dir>/plugins/cp2k_xyz.py        ("ewald plugins" prints the folder)
import numpy as np
from ewald import Cell, DataCollection
from ewald.plugins import reader


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
from ewald import DataCollection
from ewald.io.base import FrameReader, open_binary
from ewald.plugins import reader


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

## How the format is chosen

1. `import_file(path, format="myformat")` always uses the plugin with that name.
2. Without `format=`, ewald's native readers (LAMMPS dump, LAMMPS data, XDATCAR) are tried
   first, so a plugin cannot take over those files by accident.
3. Then each plugin is asked: its `sniff(path)` function, if it has one, else its `patterns`
   matched against the file name (without a `.gz`/`.bz2`/`.xz` suffix).
4. Everything else goes to ASE.

Use `sniff` when the file name says little (`OUTPUT`, `traj.dat`):

```python
from ewald.io.base import head_lines

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

`ewald.plugins.register_reader(name, factory, patterns=..., sniff=..., description=...)` does
the same as `@reader` and can be called from a script or the GUI console. In an installed
package, an entry point registers the reader without any code running at import time; see
[Packaging and sharing](packaging.md). An entry-point class can carry `patterns`, `sniff` and
`description` as class attributes.

## Example

`examples/plugins/npz_trajectory.py` (in PR #14) is a complete reader for NumPy `.npz`
trajectories with positions, types, cell, forces and energies. Copy it into the plugin folder
and `import_file("run.npz")` works.

## Without the registry

Until the registry is on `main`, wrap a reader in a source yourself:

```python
from ewald import Pipeline


class ReaderSource:
    def __init__(self, reader):
        self.reader = reader

    @property
    def num_frames(self):
        return len(self.reader)

    def compute(self, frame):
        return self.reader.read(frame)


pipe = Pipeline(ReaderSource(MyFormatReader("run.myf")))
```

For a reader that returns a list of frames, `ewald.StaticSource(frames)` is the same thing.
