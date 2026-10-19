# Releases

How a version of ewald is cut and what it ships. Windows-specific packaging (the standalone app,
the installer, CUDA bundling) is in [Windows release](windows-release.md).

## What a release ships

| File | Built by | Notes |
|---|---|---|
| `ewald-X.Y.Z.tar.gz` (sdist) | `release.yml` | builds `ewald._native` on install when a C++17 compiler is present |
| `ewald-X.Y.Z-cp3XX-*-manylinux_*_x86_64.whl` | `release.yml` | Python 3.10 to 3.13 |
| `ewald-X.Y.Z-cp3XX-*-macosx_*_{arm64,x86_64}.whl` | `release.yml` | macOS 11 or later; optional input `macos` |
| `ewald-X.Y.Z-cp3XX-*-win_amd64.whl` | `windows-release.yml` | Python 3.10 to 3.13 |
| `ewald-X.Y.Z-win64-{cpu,gpu}-setup.exe`, `-portable.zip` | `windows-release.yml` | standalone app with the EWALD icon |

All wheels carry the compiled `ewald._native` (`EWALD_REQUIRE_NATIVE=1` turns a failed C++ build
into an error), so wheel users never need a compiler. A macOS app bundle (`ewald.app` with
`ewald.icns`, as a `.dmg`) has a slot in `release.yml` and is not built yet.

Both workflows attach their files to one GitHub Release for the tag, with the notes taken from the
version's section of `CHANGELOG.md`. ewald is not on PyPI: the repository is private, and the
`Private :: Do Not Upload` classifier in `pyproject.toml` makes PyPI reject an accidental upload.
Colleagues install a wheel from the release page, or straight from the repository:

```bash
pip install "ewald[gui] @ git+https://github.com/frssp/ewald@v0.1.0"
```

## Version

The version is written in one place, `__version__` in `src/ewald/__init__.py`. `pyproject.toml`
reads it (`dynamic = ["version"]`), the Windows workflow checks the tag against it, and
`ewald --version` and Help › About print it. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 a minor version may change the API. Pre-releases use `0.2.0rc1`, tagged `v0.2.0rc1`.

`CHANGELOG.md` keeps an `## [Unreleased]` section at the top. Pull requests that change behavior a
user would notice add a line there (what changed, in user terms, with the PR number).

## Cutting a release

1. On a branch, move the `[Unreleased]` lines into a new `## [X.Y.Z] - YYYY-MM-DD` section, add the
   `[X.Y.Z]:` link at the bottom, and set `__version__ = "X.Y.Z"`.
2. Run the release check, which needs `pip install build twine` and a C++ compiler:

   ```bash
   python packaging/release_check.py
   ```

   It checks that the version, `pyproject.toml` and `CHANGELOG.md` agree, builds the sdist and
   wheel into `dist/`, runs `twine check`, checks the wheel contains the native extension, icons,
   tables and example files, then installs the wheel into a fresh virtual environment and runs
   `ewald --version` plus a small analysis (CNA, PTM) on the shipped examples.
3. Run the full test suite as in CLAUDE.md (`xvfb-run -a python -m pytest -q`), merge the branch.
4. Tag main and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "ewald X.Y.Z"
   git push origin vX.Y.Z
   ```

5. Run both release workflows on the tag. Pushing a tag starts nothing while the workflows are
   manual-only (below):

   ```bash
   gh workflow run release.yml --ref vX.Y.Z            # add -f macos=false to skip macOS wheels
   gh workflow run windows-release.yml --ref vX.Y.Z
   ```

   or Actions › Release › Run workflow and Actions › Windows release › Run workflow, choosing the
   tag under "Use workflow from". Each workflow checks that the tag matches `__version__` and
   fails before building anything if it does not.
6. Before announcing a GPU build, run a script with `device="cuda"` on an NVIDIA machine with the
   GPU app (the runners have no GPU).

Fixing a bad release: delete the GitHub Release and the tag (`git push origin :refs/tags/vX.Y.Z`),
fix on main, and tag again. Once anyone has installed it, release `X.Y.(Z+1)` instead.

## Cost and turning automatic runs back on

Every workflow in `.github/workflows/` is `workflow_dispatch` only while Actions minutes are short
(the repository is private; Windows minutes bill 2x and macOS 10x). A full release run is roughly
3 Linux, 2 macOS (x10) and 3 Windows (x2) jobs; the GPU Windows app is the longest at about 30
minutes. Dispatch with `-f macos=false` to save most of the cost when macOS wheels are not needed.

To release on tag push again, add the trigger under `on:` in both release workflows:

```yaml
on:
  push:
    tags: ["v*"]
  workflow_dispatch:
```

A tag push builds the macOS wheels too (the `macos` input applies to manual runs only). CI triggers
are restored the same way, as [Testing](testing.md) describes.
