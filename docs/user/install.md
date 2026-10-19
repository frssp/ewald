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
| `movie` | MP4/GIF export through imageio and ffmpeg | `pip install -e ".[movie]"` |
| `ovito` | `DislocationAnalysis` (DXA) and `OvitoModifier` | `pip install -e ".[ovito]"` |
| `remote` | files on SSH hosts (`import_file("host:/path")`) through paramiko | `pip install -e ".[remote]"` |
| `llm` | Claude agent and MCP server | `pip install -e ".[llm]"` |
| `analysis` | freud, for cross-checks | `pip install -e ".[analysis]"` |
| `dev` | pytest, paramiko | `pip install -e ".[dev]"` |

Extras combine: `pip install -e ".[gui,gpu,movie]"`.

If your CUDA is not 12.x, install the matching CuPy wheel yourself (for example
`pip install cupy-cuda11x`) instead of the `gpu` extra. [GPU compute](gpu.md) explains how
to check that the GPU is used.

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
