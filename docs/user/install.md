# Install

ewald needs Python 3.10 or newer. A C++17 compiler is recommended: `pip` uses it to build the
native extension `ewald._native` (CNA, PTM, Voronoi, centrosymmetry, elastic strain and grain segmentation kernels ported
from OVITO)
during installation. On Linux that is gcc or clang, on macOS the Xcode command line tools
(`xcode-select --install`), on Windows the Microsoft C++ Build Tools.

```bash
git clone https://github.com/frssp/ewald
cd ewald
pip install -e .              # core: readers, modifiers, offscreen rendering, CLI
```

## Without a C++ compiler

When no compiler is found (on Windows: `error: Microsoft Visual C++ 14.0 or greater is required`),
the install still finishes, without `ewald._native`, and pip prints a warning box when run with
`-v`. Everything works except these modifiers, which stop with an error that says how to fix it:

- `PolyhedralTemplateMatching`
- `VoronoiAnalysis`
- `CentroSymmetry(mode="matching")` (the default `conventional` mode works)
- `CommonNeighborAnalysis(mode="interval")` (adaptive and fixed-cutoff CNA fall back to Python)
- `ElasticStrain`, `GrainSegmentation` and `IdentifyFCCPlanarFaults`

Edit › Preferences › Compute lists `ewald._native` as missing, and the GUI's Log tab says so at
startup. To add it later, install a compiler and run `pip install -e .` again. On Windows: install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), tick the
workload *Desktop development with C++* (MSVC and the Windows SDK), then open a new terminal so
pip finds it. `EWALD_REQUIRE_NATIVE=1 pip install -e .` makes a failed C++ build an error instead
of a warning; CI and the release wheels use it.

## From a release

Each GitHub Release carries prebuilt wheels with `ewald._native` included, so no compiler is
needed: Linux (x86_64), macOS (arm64 and Intel) and Windows, for Python 3.10 to 3.13. Download
the one for your platform and Python from the release page, then

```bash
pip install "ewald-0.1.0-cp312-cp312-win_amd64.whl[gui]"
```

or install a tagged version straight from the repository (this builds from source):

```bash
pip install "ewald[gui] @ git+https://github.com/frssp/ewald@v0.1.0"
```

Windows releases also carry a standalone installer that needs no Python at all, in a CPU and a
CUDA build ([Windows release](../developer/windows-release.md)). `ewald --version` prints the
installed version; what changed between versions is in
[CHANGELOG.md](https://github.com/frssp/ewald/blob/main/CHANGELOG.md).

## Extras

Optional parts are extras, installed only when you need them:

| Extra | Adds | Install |
|---|---|---|
| `gui` | desktop viewer (PySide6, pyvistaqt, qtconsole) | `pip install -e ".[gui]"` |
| `gpu` | `device="cuda"` through CuPy (NVIDIA, CUDA 12) | `pip install -e ".[gpu]"` |
| `movie` | MP4/AVI/MOV export with the ffmpeg binary from imageio-ffmpeg (GIF and PNG need no extra) | `pip install -e ".[movie]"` |
| `formats` | H5MD, vaspout.h5, MDTraj HDF5 (h5py) and GROMACS XTC/TRR, DCD, NetCDF (MDAnalysis) | `pip install -e ".[formats]"` |
| `ovito` | `DislocationAnalysis` (DXA) and `OvitoModifier`; the `ovito` wheel pins its own PySide6, so install it in the same command as `gui` | `pip install -e ".[gui,ovito]"` |
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

## macOS

ewald runs on Apple Silicon (arm64) and Intel (x86_64) Macs with macOS 12 or newer. Use a Python
from python.org, Homebrew (`brew install python@3.12`) or Miniforge, not the system
`/usr/bin/python3`, and install the Xcode command line tools once so pip can build `ewald._native`
with Apple clang:

```bash
xcode-select --install
python3.12 -m venv ~/ewald-env && source ~/ewald-env/bin/activate
git clone https://github.com/frssp/ewald && cd ewald
pip install -e ".[gui]"
ewald gui
```

The C++ extension needs no OpenMP (it uses `std::thread`), so Homebrew's `libomp` is not needed.
Without the command line tools the install finishes without `ewald._native`, as described
[above](#without-a-c-compiler). Every dependency has arm64 and x86_64 wheels; on Intel Macs pip
picks numba 0.62, the last release with x86_64 macOS wheels.

In the GUI, shortcuts written with Ctrl in this manual use Cmd (Cmd+K for the command palette,
Cmd+O to open), Preferences is in the EWALD menu (Cmd+,), and the menu bar shows EWALD instead
of Python. The viewport draws with Apple's OpenGL 4.1 (on top of Metal on Apple Silicon).
Analysis computes on
the CPU: the `gpu` extra installs nothing on macOS, and `device="auto"` picks the CPU
([GPU compute](gpu.md#macos)).

A standalone EWALD.app in a .dmg, which needs no Python and opens files from Finder (Open With,
or a drop on its Dock icon), is planned ([macOS release](../developer/macos-release.md)); none
has been published yet.

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

**`ssh -X` (X11 forwarding).** With `DISPLAY=localhost:10.0` and up the window goes through SSH X11
forwarding, whose OpenGL (indirect GLX) is version 1.x. VTK needs 3.2 and used to take the X
connection down with it (`The X11 connection broke`, `XIO: fatal IO error`, `malloc_consolidate()`,
core dump). On a forwarded display `ewald gui` therefore uses the **Legacy GL** viewport, which
draws like VESTA: OpenGL 1.x display lists, so the structure crosses the network once, your own
computer's GPU draws it, and rotating only sends a new camera matrix. Picking, measuring, the view
buttons, styles and sessions work as usual; impostor spheres, smooth edges, the color bar and 3D
text don't (spheres are meshes, and above 400,000 atoms points), and Render still uses the full
renderers offscreen. Preferences › Viewports › 3D view (`auto`, `vtk`, `legacy-gl`) or
`EWALD_VIEWPORT=vtk|legacy-gl` choose it by hand; the setting takes effect at the next start.

If you choose `vtk` on a forwarded display, `ewald gui` first opens a small test window in a child
process; if that fails it retries with Mesa's software OpenGL (`LIBGL_ALWAYS_SOFTWARE=1`, then also
`QT_XCB_GL_INTEGRATION=none`), keeps the first setting that draws and says so, and otherwise stops
with a message instead of crashing. If neither viewport works, or for large systems, the options
in order are:

1. Run ewald on your own computer and open the cluster's files over SSH:
   `ewald gui user@cluster:/scratch/run/XDATCAR`, or File › Load remote file (SSH). Only the
   frames you look at are transferred.
2. A remote desktop that renders with the cluster's GPU: TurboVNC + VirtualGL (`vglrun ewald gui`),
   X2Go, NoMachine, or an Open OnDemand desktop.
3. Mesa software OpenGL on the cluster, if the automatic fallback finds none
   (RHEL/Rocky 9: `sudo dnf install mesa-dri-drivers mesa-libGL`; Debian/Ubuntu: `libgl1-mesa-dri`),
   then `EWALD_VIEWPORT=vtk LIBGL_ALWAYS_SOFTWARE=1 ewald gui`.
4. No window at all: `ewald render FILE out.png` or the Python API (headless rendering below).

`EWALD_SKIP_GL_PROBE=1` skips the test window (for example when it misjudges a working setup).

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
