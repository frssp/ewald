# Packaging and sharing

## One file

The simplest plugin is one `.py` file in the plugin folder (`ewald plugins` prints where it
is). Send the file; the recipient copies it into their own plugin folder. A folder with an
`__init__.py` works the same way when a plugin needs several modules. Files and folders whose
names start with `_` or `.` are skipped, which is a convenient way to switch one off.

Plugin files are imported as modules named `ewald_plugin_<file name>`, so two plugin files
should not import each other by file name; put shared code in a package folder instead.

## A pip package

For anything with dependencies, tests or more than one user, make a package. Name it
`ewald-<what>` so it is easy to find, and declare what it provides as entry points:

```text
ewald-cp2k/
  pyproject.toml
  src/ewald_cp2k/__init__.py
  src/ewald_cp2k/readers.py
  tests/test_readers.py
  tests/data/water-pos-1.xyz
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ewald-cp2k"
version = "0.1.0"
description = "CP2K trajectory readers for ewald"
requires-python = ">=3.10"
dependencies = ["ewald>=0.1", "numpy"]

[project.entry-points."ewald.readers"]
cp2k-xyz = "ewald_cp2k.readers:Cp2kXyzReader"
```

The entry point's name is the format name (`format="cp2k-xyz"`), and its value points at a
`FrameReader` subclass or a reader function. Give the class `patterns`, `sniff` and
`description` attributes so the format is detected without `format=`:

```python
class Cp2kXyzReader(FrameReader):
    """CP2K position trajectory."""
    patterns = ("*-pos-*.xyz",)
    ...
```

After `pip install ewald-cp2k` (or `pip install -e .` while developing), `ewald plugins` lists
the reader with its origin. Entry points are read once per process, so restart the GUI after
installing.

Modifiers use the group `ewald.modifiers`; the entry-point name becomes the modifier's name
(class attribute `category` picks its GUI submenu):

```toml
[project.entry-points."ewald.modifiers"]
LocalComposition = "ewald_sro.modifiers:LocalComposition"
```

## Plugin catalogs

The GUI's Plugins side bar lists plugins to install from catalogs: JSON indexes at the URLs or
paths in Preferences › Python › *Plugin catalogs* (`python.plugin_catalogs`, separated by `;`).
A lab or a project can publish its own list:

```json
{"plugins": [
  {"name": "Local strain", "summary": "Atomic strain against a reference frame",
   "kinds": ["modifier"], "source": "https://example.org/plugins/local_strain.py",
   "version": "0.2", "author": "Kim lab", "homepage": "https://example.org"},
  {"name": "ewald-cp2k", "summary": "CP2K readers", "kinds": ["reader"], "source": "pip:ewald-cp2k"}
]}
```

`source` is a `.py` file (a URL, an absolute path, or a path relative to the index), which
*Install* copies into the plugin folder, or `pip:<requirement>`, which it installs with pip.
The same functions work from Python:

```python
from ewald import plugin_catalog as pc

entries, errors = pc.catalog(["https://example.org/ewald-plugins.json"])
pc.install_file(pc.search(entries, "strain")[0])   # copy + load; returns the registered names
[i.name for i in pc.installed()]
```

## Dependencies

Depend on `ewald` and what the plugin imports at module level, nothing more. Import optional
or heavy packages (CuPy, PyTorch, a simulation code's Python bindings) inside the function that
needs them, with an error that says what to install. Do not depend on `PySide6`; the GUI
integration works without importing Qt in your plugin.

## Compatibility

ewald is at 0.x and its API may still change between minor versions. Test your plugin against
the ewald version you declare, keep to the documented API (`ewald`, `ewald.modifiers`,
`ewald.io.base`, `ewald.plugins`, `ewald.neighbors`, `ewald.kernels`), and avoid names that
start with `_`. The [roadmap](../ROADMAP.md) lists changes that are coming.

## Licensing

State a license in your package. If you port code from another program (as ewald ports parts of
OVITO under its MIT option), keep the original copyright notice and license text with the
ported code.
