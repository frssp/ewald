# Plugins

A plugin extends ewald without changing it: a file format ewald cannot read, an analysis it
does not have, a modifier tuned for your own simulations. Plugins are ordinary Python, written
against the same API the built-in readers and modifiers use, so everything in the
[user manual](../user/scripting.md) applies to them.

## Extension points

| Extension | What you write | How ewald finds it | State |
|---|---|---|---|
| [Reader](readers.md) | a `FrameReader` class or a function returning frames | `@reader(...)` in the plugin folder, or an `ewald.readers` entry point | available |
| [Modifier](modifiers.md) | a `Modifier` subclass (or a function `f(frame, data)` in a script) | `@modifier` in the plugin folder, or an `ewald.modifiers` entry point | available |

[Modifier Forge](forge.md) writes a modifier plugin from a description: Claude writes the
module and its tests, and ewald saves it to the plugin folder once its checks pass.

ewald's own formats (LAMMPS, VASP, cube, ASE) are registered as built-in readers in the same
registry, so a plugin is a first-class format: `ewald plugins` lists both.

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

## Where a plugin shows up

Once registered, a plugin is used everywhere ewald reads files or runs modifiers:

| Place | Reader plugin | Modifier plugin |
|---|---|---|
| `import_file` / scripts | picked by `format=`, `sniff` or file-name pattern | `from ewald.modifiers import MyModifier` |
| GUI | File › Load file filter lists its patterns | Add modification menu and command palette; the parameter form is built from the constructor |
| Claude tools | `load_structure` lists installed formats | `list_modifiers` / `add_modifier` |
| `ewald plugins` | listed with patterns and origin, next to the built-in formats | listed |
| `llms.txt` | not listed (it documents core ewald only) | not listed |

`llms.txt` and `ewald api` stay a reference for core ewald, so the same file is valid on every
installation (`ewald api --with-plugins` appends yours). The Claude tools, which run on your
installation, see your plugins.

## Rules every plugin follows

The built-in code follows these, and plugins that break them give wrong results in ways that
are hard to trace:

1. **Never write into an array of a `DataCollection` in place.** Arrays are shared with cached
   frames. Build a new array and assign it (`data["X"] = new`).
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
