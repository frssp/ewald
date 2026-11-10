# API stability

Before 1.0 a minor version may change the API ([Releases](release.md)), but it must not change
without anyone noticing. Two text snapshots of the public API and one test file make sure of
that (decision D104).

## What the public API is

`python -m ewald api --signatures` prints it, one name per line (`src/ewald/api_surface.py`):

- every name in `ewald.__all__`: functions with their signature, classes with their constructor
  and their public methods, properties and attributes (`ewald.Pipeline.compute(...)`,
  `ewald.Pipeline.modifiers`, `ewald.DataCollection.tables`), and `ewald.verbs.Table`;
- the functions `llms.txt` documents (`ewald.views.show`, `ewald.paint.hide`, ...);
- the constructor of every modifier in `ewald.modifiers.__all__`;
- everything in `ewald.plugin_api`, the surface plugins import from;
- the keywords of `render()`, the settings of the 3D view (`View.bonds = 'auto'`) and the
  options of `ewald.display.options`.

Signatures are written without type annotations, and a default that is not a plain value is
written as its type (`<factory>`, `<ndarray>`), so the text is the same on every Python
version, NumPy version and platform.

Other `ewald.*` modules can be imported, but they are internal and may change in any release.

## The two snapshots

`tests/data/api/current.txt` is the API of the checked-out code.
`tests/test_api_stability.py::test_snapshot_is_current` fails when the code and the file differ
and lists what was removed, changed and added. If the change is intended, regenerate the file:

```bash
python -m ewald api --signatures > tests/data/api/current.txt
```

The diff of that file in the pull request then shows the reviewer exactly how the public API
changed, including names that became public by accident.

`tests/data/api/released.txt` is the API of the last release. Code written for that release
must keep working, or `CHANGELOG.md` must say what it has to change:
`test_what_the_release_promised_is_kept_or_in_the_changelog` compares the release with the
code and fails for every breaking change that the changelog sections above that release do not
name. A changelog line names a change when it contains the last part of the name (`rdf`,
`CoordinationAnalysis`, `frames`) and, for a parameter, the parameter (`cutoff`), each as a
word of its own.

Breaking means:

- a name, a parameter, a `render()` keyword or a view setting is gone;
- a default changed, or a parameter lost its default;
- a positional parameter moved, became keyword-only, or a new parameter was put in front of
  existing positional ones;
- a new parameter without a default;
- a constant changed (`ewald.plugin_api.PLUGIN_API`, a view setting's default).

Adding a name, or a parameter with a default at the end or as keyword-only, breaks nothing
and needs no changelog line for this test (the usual rule still asks for one when a user would
notice).

The preferred fix for a failure is to keep the old spelling working: put a new parameter
last, keep the old name as an alias. Only when the change is the point, write the changelog
line under `[Unreleased]` › Changed.

## At a release

Step 1 of [cutting a release](release.md#cutting-a-release) writes the API of the new version
into both files; `packaging/release_check.py` refuses a release whose `released.txt` is of
another version. From then on the new release is what must be kept.

## Limits

The snapshot holds names and signatures. It does not see a function that keeps its signature
and changes what it returns, the names a step writes into a frame (`"Coordination"`,
`"CoordinationAnalysis.mean"`, the table `"coordination"`), setting names in `settings.json`,
or command-line options (`docs/features.md` lists those and is checked the same way).
Old names that still work (`XxxModifier` for `Xxx`, the names in `Modifier.ALIASES`) do so
without a `DeprecationWarning`; whether and when to warn is left for the 1.0 cleanup (D104).

## Notebooks say which version wrote them

Every saved notebook records the ewald version that wrote it: `metadata.ewald.version` in an
`.ipynb`, `ewald_version` in the manifest of an `.ewd`, and the first line of the `ewald:`
block in the header of a `.py` notebook. `Notebook.ewald_version` returns it. Opening a notebook
saved by a newer ewald than the one running says so (`Notebook.warnings`, shown in the
status bar), since its cells may use names this version does not have. A later ewald that
renames something can use the recorded version to read old notebooks correctly.
