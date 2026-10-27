# macOS binary distribution

How ewald ships on macOS, next to the [Windows release](windows-release.md). The build is
`.github/workflows/macos-release.yml` (manual only: macOS runners bill ten times the Linux
minutes); the packaging files are in `packaging/macos/`.

## What ships

| File | For | Contents |
|---|---|---|
| `ewald-X.Y.Z-cp3XX-cp3XX-macosx_11_0_arm64.whl` and `..._x86_64.whl` (3.10 to 3.13) | people with their own Python | the package with the compiled `ewald._native`; `pip install <wheel>[gui]` |
| `ewald-X.Y.Z-macos-arm64.dmg` | Apple Silicon without Python | `EWALD.app`: the viewer, plus the `ewald` CLI inside it |
| `ewald-X.Y.Z-macos-x86_64.dmg` | Intel Macs without Python | the same, built on an Intel runner |

There is no CUDA build (no CUDA on macOS) and no universal2 app: numba and llvmlite wheels are per
architecture, so each architecture gets its own build on its own runner (`macos-14` for arm64,
`macos-15-intel` for x86_64).

## The app bundle

`packaging/macos/ewald.spec` is the Windows spec's PyInstaller one-folder build wrapped in a
`BUNDLE`, with the same entry points (`packaging/windows/launch_gui.py`, `launch_cli.py`):

- `EWALD.app/Contents/MacOS/ewald-gui` is the bundle executable (Finder, Dock, `open -a EWALD file`);
- `EWALD.app/Contents/MacOS/ewald` is the CLI, including `ewald run script.py` and `ewald python`
  with the bundled interpreter. Put it on the PATH with
  `ln -s /Applications/EWALD.app/Contents/MacOS/ewald /usr/local/bin/ewald`.

`Info.plist` sets the name, version, `NSHighResolutionCapable` (Retina), dark mode, macOS 12 as the
minimum and the document types: Finder offers EWALD under Open With for trajectory and structure
extensions, and any file (VASP's POSCAR, XDATCAR and OUTCAR have no extension) can be dropped on the
Dock icon. Finder delivers those files as `QFileOpenEvent`s, not as arguments, so
`gui/app.py:EwaldApplication` queues them until the main window exists and opens them there;
`argv_emulation` stays off.

The runtime hook (`rthook_macos.py`) does what a terminal would have done:

- `NUMBA_CACHE_DIR=~/Library/Caches/ewald/numba`, because numba otherwise writes its cache next to
  the sources inside the bundle, which changes a signed app when `/Applications` is writable;
- adds `/opt/homebrew/bin` and `/usr/local/bin` to the minimal PATH Finder gives apps, so a
  Homebrew `ffmpeg` (movies) or `plumed` is found;
- puts `~/Library/Application Support/ewald/python/lib/python3.X/site-packages` on `sys.path`: the
  Plugins side bar installs pip plugins there with the bundled pip (`ewald -m pip install --prefix ...`,
  as on Windows, D76); the older `~/Library/Application Support/ewald/site-packages` stays on it too.

Run from a terminal (not the bundle), `gui/macos.py:set_app_name` puts EWALD instead of Python in
the menu bar through the Objective-C runtime (ctypes, no pyobjc).

`make_dmg.sh` copies the app next to an `/Applications` link into a compressed disk image
(`hdiutil`).

## Signing

PyInstaller signs the app ad hoc, which is enough to run on Apple Silicon. It is not signed with a
Developer ID or notarized, so the first launch of a downloaded copy is blocked by Gatekeeper:
Control-click the app › Open, or System Settings › Privacy & Security › Open Anyway, or
`xattr -dr com.apple.quarantine /Applications/EWALD.app`. Notarization needs an Apple Developer
account ($99 a year), a Developer ID certificate in the workflow secrets, and `codesign` +
`xcrun notarytool submit --wait` + `xcrun stapler staple` after PyInstaller; it is listed in the
TODO.

## Release

Follow [Releases](release.md): bump `ewald.__version__`, add the CHANGELOG section, push a `v*`
tag, then dispatch `release.yml`, `windows-release.yml` and `macos-release.yml` on that tag; each
attaches its files to the same GitHub Release, with the notes from CHANGELOG.md.

## Checked without a Mac

The cloud container is Linux, so this was checked there:

- `ewald._native` builds with clang (`CC=clang CXX=clang++`), without warnings that matter;
- `pyinstaller packaging/macos/ewald.spec` runs on Linux (it skips `BUNDLE` there) and the frozen
  CLI passes `packaging/windows/smoke.py`;
- `tests/test_macos.py`: the Info.plist values serialize, the workflow is manual only, the runtime
  hook, the file-open queue, `-psn_` arguments, and the `metal`/`mps` device error;
- every dependency resolves to macOS arm64 and x86_64 wheels (`pip download --platform`).

## To test on a real Mac

1. `xcode-select --install`, `pip install -e ".[dev,gui]"`, then `python -c "import ewald._native"`.
2. `python -m pytest -q` (all of it, including `tests/test_gui.py` and `tests/test_render.py`,
   which need a display; OpenGL visual references skip on non-Mesa drivers).
3. `ewald gui tests/data/vasp/XDATCAR`: the menu bar says EWALD, Preferences is under EWALD › Settings
   (Cmd+,), Cmd+K opens the palette.
4. The 3D view draws the atoms at all (black before the OpenGL 3.2 core default format; if it is
   black again, compare with `EWALD_VIEWPORT=legacy ewald gui`). Retina: the structure fills the whole viewport (not one quarter), picking and the hover
   readout hit the atom under the cursor, and screenshots are at full resolution.
5. Trackpad: two-finger scroll zooms, a click-drag orbits, and the right-click menu opens with a
   two-finger click.
6. `pyinstaller --noconfirm packaging/macos/ewald.spec && open dist/EWALD.app`, then drop a POSCAR
   on the Dock icon and use Open With on a `.lammpstrj`.
7. `python benchmarks/neighbors.py` on Apple Silicon, to compare with the CUDA numbers.
