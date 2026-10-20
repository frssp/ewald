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

AI tools use `ewald.tools` ([AI tool plugins](tools.md)). The first-party packages in
[frssp/ewald-plugins](https://github.com/frssp/ewald-plugins) (`packages/ewald-phonopy`,
`packages/ewald-plumed`) are complete examples with readers, modifiers, tools, a manifest and
tests.

## Manifest

A plugin can describe itself in a manifest, which ewald reads without importing the plugin's
code (as VS Code reads an extension's `package.json`). `ewald plugins`, `plugins.installed()`
and the GUI's plugin list show what it says, and ewald checks `requires-ewald` before it
imports anything: a plugin that needs another ewald version is listed as incompatible and left
alone instead of failing halfway through an import.

In a package, put `ewald-plugin.toml` inside the package and name the package in the
`ewald.plugins` entry point group. The entry-point name is the plugin id:

```toml
# src/ewald_cp2k/ewald-plugin.toml (add it to package-data)
id = "cp2k"
display-name = "CP2K formats"
version = "0.1.0"            # default: the distribution version
publisher = "your-github-name"
description = "CP2K trajectory readers"
requires-ewald = ">=0.1, <1"
```

```toml
# pyproject.toml
[project.entry-points."ewald.plugins"]
cp2k = "ewald_cp2k"
```

ewald locates the package with `importlib.util.find_spec`, which does not run it, so name a
top-level package there. A package without this entry point still works; its id is the
distribution name (`ewald-cp2k`) and its version and description come from the package metadata.

In a single plugin file, assign a dict of plain literals to `__ewald__` anywhere at the top
level; ewald reads it with `ast`. The end of the file is a good place, after a docstring or
`from __future__` imports:

```python
__ewald__ = {"id": "lab-tools", "display-name": "Lab tools", "version": "1.2.0",
             "requires-ewald": ">=0.1"}
```

A plugin folder with an `__init__.py` can use either form. Without a manifest the id is the
file or folder name. Modifier Forge writes `__ewald__` with `"source": "forge"` into the
plugins it saves, so they are listed as Forge plugins. `requires-ewald` takes comma-separated
`>=`, `>`, `<=`, `<`, `==`, `!=` and `~=` clauses on release numbers.

## Switching a plugin off

`ewald plugins --disable ID` (or `ewald.plugins.set_enabled(ID, False)`) keeps the plugin
installed but stops ewald from importing it; `--enable ID` switches it back on. The choice is
stored in `<config dir>/plugins.json` and applies to every later session. In a running
process, readers, modifiers and other contributions of a disabled plugin leave the registries
at once; a package that was already imported stays in memory until restart.

`plugins.installed()` returns every plugin with its id, version, kind (`package`, `folder`,
`forge`), state (`loaded`, `disabled`, `incompatible`, `failed`), errors and contributions (the
readers, modifiers, writers, visuals, overlays, color maps and themes it registered).
`ewald plugins --json` prints the same under `"plugins"`.

## Plugin catalogs

The GUI's Plugins side bar lists plugins to install from catalogs: the default index
(`https://raw.githubusercontent.com/frssp/ewald-plugins/main/index.json`) and JSON indexes at
the URLs or paths in Preferences › Python › *Plugin catalogs* (`python.plugin_catalogs`,
separated by `;`). The environment variable `EWALD_PLUGIN_INDEX` replaces the default index,
and an empty value turns it off. An index that cannot be found (or, on GitHub, is private and
there is no token) shows as one muted line. A lab
or a project can publish its own list:

```json
{"format": 1, "plugins": [
  {"id": "local_strain", "name": "Local strain", "summary": "Atomic strain against a reference frame",
   "kinds": ["modifier"], "source": "https://example.org/plugins/local_strain.py",
   "version": "0.2", "requires_ewald": ">=0.1", "author": "Kim lab", "homepage": "https://example.org"},
  {"id": "ewald-cp2k", "name": "ewald-cp2k", "summary": "CP2K readers", "kinds": ["reader"],
   "source": "pip:ewald-cp2k"}
]}
```

`source` is a `.py` file (a URL, an absolute path, or a path relative to the index), which
*Install* copies into the plugin folder, or `pip:<requirement>`, which it installs with pip.
`id` is the plugin's id once installed (the file stem, or the package's `ewald-plugin.toml`
name); the side bar matches an entry to an installed plugin by `id`, then by name, so an
installed plugin is not offered twice. When an entry has a newer `version` than the installed
plugin, the installed row gets an *Update to …* button (a newer file replaces the old one; a
package is upgraded with `pip install --upgrade`). An entry whose `requires_ewald` this ewald
does not meet is greyed out with the version it needs. An index with a `format` above 1 is
refused with a message, so an older ewald does not misread a newer index.

Private indexes and plugins on GitHub work too: requests to `raw.githubusercontent.com` and
`api.github.com` carry a token from `GITHUB_TOKEN` or, without it, `gh auth token`. The token is
read when needed, sent to those two hosts only (and not on to a redirect), and never stored.
For a `pip:git+https://github.com/...` source it reaches the pip process through git's
environment (`GIT_CONFIG_*`), so it does not appear in the command or in the pip log.

The same functions work from Python:

```python
from ewald import plugin_catalog as pc

entries, errors = pc.catalog(pc.catalog_urls("https://example.org/ewald-plugins.json"))
pc.install_file(pc.search(entries, "strain")[0])   # copy + load; returns the registered names
[(i.name, i.version, pc.update_for(i, entries)) for i in pc.installed()]
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
