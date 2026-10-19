# Install

ewald needs Python 3.10 or newer and a C++17 compiler, because `pip` builds the native
extension `ewald._native` (CNA, PTM, Voronoi and centrosymmetry kernels ported from OVITO)
during installation. On Linux that is gcc or clang, on macOS the Xcode command line tools
(`xcode-select --install`), on Windows the MSVC Build Tools.

On Windows you can skip the compiler: each release on GitHub carries prebuilt wheels
(`pip install ewald-X.Y.Z-cp312-cp312-win_amd64.whl[gui]`) and a standalone installer that needs no
Python at all, in a CPU and a CUDA build ([Windows release](../developer/windows-release.md)).

```bash
git clone https://github.com/frssp/ewald
cd ewald
pip install -e .              # core: readers, modifiers, offscreen rendering, CLI
```

Optional parts are extras, installed only when you need them:

| Extra | Adds | Install |
|---|---|---|
| `gui` | desktop viewer (PySide6, pyvistaqt, qtconsole) | `pip install -e ".[gui]"` |
| `gpu` | `device="cuda"` through CuPy (NVIDIA, CUDA 12) | `pip install -e ".[gpu]"` |
| `movie` | MP4/AVI/MOV export with the ffmpeg binary from imageio-ffmpeg (GIF and PNG need no extra) | `pip install -e ".[movie]"` |
| `formats` | H5MD, vaspout.h5, MDTraj HDF5 (h5py) and GROMACS XTC/TRR, DCD, NetCDF (MDAnalysis) | `pip install -e ".[formats]"` |
| `ovito` | `DislocationAnalysis` (DXA) and `OvitoModifier`; the `ovito` wheel pins its own PySide6, so install it in the same command as `gui` | `pip install -e ".[gui,ovito]"` |
| `plumed` | PLUMED Python binding for `Plumed` (the PLUMED library itself: conda-forge or your build, see [PLUMED](plumed.md)) | `pip install -e ".[plumed]"` |
| `remote` | files on SSH hosts (`import_file("host:/path")`) through paramiko | `pip install -e ".[remote]"` |
| `llm` | Claude agent and MCP server | `pip install -e ".[llm]"` |
| `symmetry` | spglib, for `SymmetryAnalysis` (space group, Wyckoff positions) | `pip install -e ".[symmetry]"` |
| `dev` | pytest, paramiko, spglib | `pip install -e ".[dev]"` |

Extras combine: `pip install -e ".[gui,gpu,movie]"`. pymatgen is used when it happens to be
installed (bond-valence oxidation states for the Shannon ionic radii) but is not needed.

The GUI uses PySide6 only. PyQt5 or PyQt6 in the same environment is harmless as long as nothing
imports them before the GUI starts; if something does (matplotlib's Qt backend, IPython's
`%gui qt`), `ewald gui` stops with a message saying so. `import ewald.gui` sets `QT_API=pyside6`,
so importing it first makes matplotlib, QtPy and IPython pick PySide6 too.

`pip install ovito` on its own, after the GUI, downgrades only `PySide6-Essentials` to the version
ovito pins and leaves the rest of PySide6 newer (`pip check` reports it). Install both extras in one
command so pip picks versions that agree, or rerun `pip install -e ".[gui,ovito]"` to repair it.

If your CUDA is not 12.x, install the matching CuPy wheel yourself (for example
`pip install cupy-cuda11x`) instead of the `gpu` extra. [GPU compute](gpu.md) explains how
to check that the GPU is used.

## Linux: GUI system libraries

The PySide6 wheel bundles Qt itself, but Qt's X11 platform plugin loads a few system libraries
that minimal installs lack. On Debian/Ubuntu:

```bash
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \
    libxcb-render-util0 libxcb-image0 libxcb-randr0 libegl1 libgl1 libfontconfig1 libdbus-1-3
```

On Fedora: `sudo dnf install xcb-util-cursor xcb-util-wm xcb-util-keysyms xcb-util-image
xcb-util-renderutil libxkbcommon-x11 mesa-libEGL`. `ewald gui` checks for the xcb libraries
before it creates the window and names the missing packages, instead of Qt's
"xcb-cursor0 or libxcb-cursor0 is needed" abort.

## Troubleshooting the GUI

**`undefined symbol: ..., version Qt_6_PRIVATE_API` (or `Qt_6`) when PySide6 is imported.** Two
Qt versions got mixed: a system Qt (for example Qt 6.4 from apt, `/lib/x86_64-linux-gnu/libQt6DBus.so.6`)
was loaded next to the newer Qt bundled with the PySide6 wheel. The usual cause is an
`LD_LIBRARY_PATH` that lists system directories such as `/lib`, `/usr/lib`, `/usr/lib64` or
`/lib/x86_64-linux-gnu`: directories in `LD_LIBRARY_PATH` are searched before the wheel's own
library path. Remove those entries (the loader searches them anyway, after the wheel's) and keep
only the extra directories you need:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64      # no /lib, /usr/lib, /lib/x86_64-linux-gnu
ewald gui
```

`ewald gui` reports this case and names the offending `LD_LIBRARY_PATH` entries.

**`qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed`.** Install the packages
listed under [Linux: GUI system libraries](#linux-gui-system-libraries).

**`libEGL.so.1: cannot open shared object file`.** `sudo apt install libegl1 libgl1`.

**No window on a server.** Without `DISPLAY` there is nothing to open a window on: use
`ssh -X`, a remote desktop, or `xvfb-run -a ewald gui` for screenshots and tests.

## Headless machines

Rendering uses VTK's OpenGL. On a cluster node or a server without a display, install an
offscreen OpenGL such as OSMesa (`apt install libosmesa6`) or EGL, or run under a virtual
display with `xvfb-run -a python script.py`. The built-in ray tracer
(`renderer="raytrace"`) needs no OpenGL for the image itself.

## After updating

`git pull` followed by `pip install -e .` rebuilds the native extension. The extension is not
rebuilt automatically, so skipping the reinstall after an update that touched the C++ code
gives import errors or old results.

## Checking the install

```bash
ewald info tests/data/vasp/POSCAR      # any structure file works
```

Edit › Preferences › Compute in the GUI lists every optional component, whether it is
installed and which version.
