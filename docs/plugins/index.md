# Plugins

A plugin extends ewald without changing it: a file format ewald cannot read, an analysis it
does not have, a modifier tuned for your own simulations. Plugins are ordinary Python, written
against the same API the built-in readers and modifiers use, so everything in the
[user manual](../user/scripting.md) applies to them.

## Extension points

| Extension | What you write | How ewald finds it | State |
|---|---|---|---|
| [Reader](readers.md) | a `FrameReader` class or a function returning frames | `@reader(...)` in the plugin folder, or an `ewald.readers` entry point | available |
| [Writer](writers.md) | a function `f(path, frames, **kwargs)` | `@writer(...)` in the plugin folder, or an `ewald.writers` entry point | available |
| [Modifier](modifiers.md) | a `Modifier` subclass (or a function `f(frame, data)` in a script) | `@modifier` in the plugin folder, or an `ewald.modifiers` entry point | available |
| [Visual element](visuals.md) | a `VisualElement` subclass returning scene primitives | `@visual` in the plugin folder, or an `ewald.visuals` entry point | available |
| [Overlay](overlays.md) | an `OverlayElement` subclass returning 2D items (text, shapes, images) | `@overlay` in the plugin folder, or an `ewald.overlays` entry point | available |
| [Color map](styles.md#color-maps) | a function returning colors or a matplotlib `Colormap` | `@colormap` in the plugin folder, or an `ewald.colormaps` entry point | available |
| [Theme](styles.md#themes) | a function returning a `Theme` (element colors, background, cell color, radius preset) | `@theme` in the plugin folder, or an `ewald.themes` entry point | available |
| [AI tool](tools.md) | a function `f(toolset, **arguments)` returning JSON-ready data | `@ai_tool` (or `@claude_tool`) in the plugin folder, or an `ewald.tools` entry point | available |
| [AI provider](tools.md#ai-providers) | an `ewald.llm.providers.Provider` (another AI service: Mistral, a company gateway, ...) | `@ai_provider` in the plugin folder, or an `ewald.ai_providers` entry point | available |
| [UI theme](styles.md#ui-themes) | a function returning design tokens (`{"accent": "#7aa2f7", ...}`) on top of Paper or Graphite | `@ui_theme` in the plugin folder, or an `ewald.ui_themes` entry point | available |
| [Side bar](gui.md) | a function returning a Qt widget, opened from its own activity-bar button | `@panel` in the plugin folder, or an `ewald.panels` entry point | available |
| [Settings page](settings.md) | fields (`SettingsField`), plus `load()` / `save(values)` or nothing (ewald stores them); a page of Preferences and a dialog | `@settings_page(...)` in the plugin folder, or an `ewald.settings_pages` entry point | available |
| [Simulation code](codes.md) | a `SimulationCode`: its files, formats, settings (`codes.<name>.*`) and hooks for run info, checks, workflow folders, thermo files and opening files; VASP and LAMMPS are built-in entries | `@simulation_code` in the plugin folder, or an `ewald.codes` entry point | available |
| [Workflow source](workflows.md) | `list()` / `describe()` returning `Workflow`s (steps, outputs, paths); paired with `workflow_panel` for a side bar | `@workflow_source(...)` in the plugin folder, or an `ewald.workflows` entry point | available |

[Modifier Forge](forge.md) writes a modifier plugin from a description: the AI writes the
module and its tests, and ewald saves it to the plugin folder once its checks pass.

ewald's own formats (LAMMPS, VASP, extxyz, XSF, H5MD, XTC, cube, ASE) are registered as built-in
readers and writers in the same registries, so a plugin is a first-class format: `ewald plugins`
lists both.

## Built-in and first-party plugins

As in VS Code, ewald's own formats are listed as **built-in plugins** that ship with ewald and
can be switched off but not uninstalled: `vasp`, `lammps`, `cube`, `md-formats` (H5MD, XTC, DCD,
MDTraj HDF5), `xsf` and `ase-formats` (cif, ASE trajectory and database). extxyz and the ASE
fallback reader are always on. `ewald plugins --disable vasp` removes the VASP readers and
writers from `import_file`, the GUI and Claude until `--enable vasp`.

Support tied to one outside code lives in **first-party plugins** in
[frssp/ewald-plugins](https://github.com/frssp/ewald-plugins) (`packages/ewald-<name>`), installed
from the Plugins panel or with pip:

| Plugin | Package | Adds |
|---|---|---|
| `phonopy` | `ewald-phonopy` | phonopy's input with forces or force constants (`phonopy-fc`), `from_phonopy` for `phonopy.Phonopy` objects; band.yaml, VASP IBRION=5-8 OUTCARs and `PhononMode` are built in ([Phonon modes](../user/phonons.md)) |
| `plumed` | `ewald-plumed` | `Plumed`, `LoadColvar`, `FreeEnergySurface`, the `ewald_plumed.plumed` library and two AI tools ([PLUMED](../user/plumed.md)) |

```bash
pip install 'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-phonopy'
```

PLUMED support was part of ewald before; scripts and sessions written then keep working once the
plugin is installed (`from ewald.modifiers import Plumed`, `import ewald.plumed`), and without it
the import fails with an `ImportError` that says how to install it. Phonon modes went the other
way: they moved back into ewald on 2026-10-23, and the plugin keeps only what needs phonopy.

## Two ways to ship a plugin

**A single file in the plugin folder.** Drop `my_plugin.py` (or a package folder with an
`__init__.py`) into `<config dir>/plugins/`. ewald imports every file there once per process.
The config folder is `~/.config/ewald` on Linux, `~/Library/Application Support/ewald` on
macOS, `%APPDATA%\ewald` on Windows, or `$EWALD_CONFIG_DIR` when that is set; `ewald plugins`
prints the exact path. This is the quickest way to add something for yourself or to send one
file to a colleague.

**An installed package.** A pip-installable package declares its readers (and later its
modifiers) as entry points in `pyproject.toml`, so `pip install ewald-myformat` is all a user
does. See [Packaging and sharing](packaging.md).

A plugin can carry a [manifest](packaging.md#manifest) (`ewald-plugin.toml` in a package,
`__ewald__ = {...}` in a file) with its id, version, publisher and the ewald versions it works
with; ewald reads it without importing the plugin. `ewald plugins --disable ID` switches a
plugin off without uninstalling it ([details](packaging.md#switching-a-plugin-off)).

## The plugin API

Import what a plugin needs from `ewald.plugin_api`:

```python
from ewald.plugin_api import DataCollection, Modifier, find_pairs, modifier
```

It holds the data model (`DataCollection`, `Cell`, `VoxelGrid`, `Bonds`, `SurfaceMesh`, `Table`),
the pipeline (`Modifier`, `ComputeContext`, `Pipeline`, `StaticSource`, `load`), reader helpers
(`FrameReader`, `open_binary`, `head_lines`, ...), drawing (`VisualElement`, `Spheres`,
`OverlayElement`, `Text`, `Theme`, ...), AI tool helpers and every registration decorator.
These names keep their meaning while `ewald.plugin_api.PLUGIN_API` (now 1) stays the same; a
change that would break plugins raises it. Declare the version in the manifest
(`"plugin-api": 1`, see [Manifest](packaging.md#manifest)) so an incompatible ewald refuses the
plugin with a clear message instead of failing in the middle of a computation. Other
`ewald.*` modules work as well, but they are internal and can change in any release.

## Where a plugin shows up

Once registered, a plugin is used everywhere ewald reads files or runs modifiers:

| Place | Reader plugin | Modifier plugin |
|---|---|---|
| `import_file` / scripts | picked by `format=`, `sniff` or file-name pattern | `from ewald.modifiers import MyModifier` |
| GUI | File › Load file filter lists its patterns | Add modification menu and command palette; the parameter form is built from the constructor |
| AI tools | `load_structure` lists installed formats | `list_modifiers` / `add_modifier` |
| `ewald plugins` | listed with patterns and origin, next to the built-in formats | listed |
| `llms.txt` | not listed (it documents core ewald only) | not listed |

`llms.txt` and `ewald api` stay a reference for core ewald, so the same file is valid on every
installation (`ewald api --with-plugins` appends yours). The AI tools, which run on your
installation, see your plugins.

## Rules every plugin follows

The built-in code follows these, and plugins that break them give wrong results in ways that
are hard to trace:

1. **Never write into an array of a `DataCollection` in place.** Arrays are shared with cached
   frames, which the pipeline makes read-only, so `data.positions[0] += 1` raises
   `ValueError: assignment destination is read-only`. Build a new array and assign it (`data["X"] = new`).
2. **Use OVITO's names** for properties and attributes ("Position", "Particle Type",
   "Particle Identifier", "Structure Type", "Force"), so built-in modifiers, expressions and
   the renderer understand your output.
3. **Store every constructor argument as an attribute of the same name** (modifiers). The GUI
   form, `repr`, caching and the generated scripts are built from them.
4. **Write compute code against `ctx.xp`** (modifiers), not `numpy` directly, so it also runs
   on the GPU. Convert to NumPy only when storing results.
5. **Import heavy or optional dependencies inside functions**, so that a missing package
   makes your plugin fail when used, with a clear message, rather than when ewald starts.
6. **Fail with an exception, never by printing.** A plugin that fails to import is listed by
   `ewald plugins` with its error and does not stop ewald; a modifier that raises shows its
   error in the GUI pipeline row.

## Troubleshooting

`ewald plugins` lists what is installed, where each plugin came from, and every plugin that
failed to import with its error (exit code 1 when any failed). `EWALD_NO_PLUGINS=1` starts
ewald without loading any plugin, which tells you whether a problem comes from a plugin.
