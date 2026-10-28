# Windows binary distribution

Plan and how-to for shipping ewald on Windows. The build is
`.github/workflows/windows-release.yml`; the packaging files are in `packaging/windows/`.

## What ships

Each release produces three kinds of files, all attached to a GitHub Release when a `v*` tag is pushed.

| File | For | Contents |
|---|---|---|
| `ewald-X.Y.Z-cp3XX-cp3XX-win_amd64.whl` (3.10 to 3.13) | people with their own Python | the package with the compiled `ewald._native`; `pip install <wheel>[gui,gpu]` |
| `ewald-X.Y.Z-win64-cpu-setup.exe` and `-portable.zip` | people without Python | standalone app: `ewald-gui.exe` (viewer) and `ewald.exe` (CLI and script runner) |
| `ewald-X.Y.Z-win64-gpu-setup.exe` and `-portable.zip` | NVIDIA users without Python | the same app with CuPy and the CUDA 12 runtime libraries bundled |
| `ewald-X.Y.Z-win64-online-setup.exe` | everyone else without Python (the default) | a private Python; pip downloads ewald and its dependencies during setup (see below) |

The wheels are the main channel for scripting work: they behave exactly like the source install, and
`pip install cupy-cuda12x`, `ovito`, `anthropic` and so on just work next to them. The standalone app
is for opening files and making figures on a machine without a Python setup.

## Packaging tool: PyInstaller (one folder)

PyInstaller in one-folder mode, with two executables sharing one `_internal/` folder
(`packaging/windows/ewald.spec`):

- `ewald-gui.exe`, windowed, opens the main window (`ewald-gui.exe file.lammpstrj` works, so file
  associations can point at it);
- `ewald.exe`, console, the normal CLI (`ewald render`, `ewald info`, ...) plus `ewald run script.py`
  and `ewald python`, which run user scripts with the bundled interpreter, like OVITO's `ovitos`.

Why PyInstaller: its hooks (and pyinstaller-hooks-contrib) already cover every heavy dependency we
have: PySide6, VTK/PyVista, numba/llvmlite, SciPy, matplotlib, IPython/qtconsole, CuPy. It builds in
about two minutes. One-folder starts fast (one-file unpacks 1 GB to a temp dir on every launch) and
triggers fewer antivirus false positives.

Alternatives considered:

- Nuitka: compiles modules to C, which breaks numba (`@njit` needs real Python functions and source)
  unless ewald is excluded, and builds take tens of minutes with SciPy/VTK. No gain for us.
- Briefcase (MSI via WiX): fine for plain apps, but less control over VTK/numba collection and no
  console + windowed pair from one bundle.
- conda `constructor`: ships a real conda environment, so users could `conda install` more packages,
  but needs an ewald conda package and a conda-forge build of everything first. Worth revisiting
  only if the standalone app has to grow a package manager.

Frozen-app specifics, already handled in the spec and runtime hook:

- `ewald` is collected as `.py` plus bytecode (`module_collection_mode="pyz+py"`), because numba's
  `cache=True` finds functions through their source files. Numba then caches compiled kernels in the
  user cache dir (`%LOCALAPPDATA%\numba`), so the install folder can stay read-only.
- `debugpy` is excluded: ipykernel imports it optionally, and its vendored pydevd tree breaks frozen.
  Without this the IPython console silently falls back to the plain console.
- The app bundles pip and setuptools (D76). `ewald.exe` takes python's command lines (`-m`, `-c`,
  a script path), so the Plugins side bar installs pip plugins with
  `ewald.exe -m pip install --prefix "%LOCALAPPDATA%\ewald\python" --no-build-isolation --constraint <pins> ...`
  (`plugin_catalog.frozen_pip_command`). The pins are every bundled distribution at its version,
  so a plugin's dependencies never replace the app's numpy or Qt (pip says so if a plugin needs a
  newer one). The runtime hook puts `%LOCALAPPDATA%\ewald\python\Lib\site-packages` on `sys.path`.
  Without git on the PC, GitHub `git+https` sources are fetched as GitHub's zip of the pinned commit.
  Any package can be added the same way from a terminal: `ewald.exe -m pip install --prefix ... <pkg>`.
  The CI smoke step `packaging/windows/smoke_pip.py` installs ewald-plumed both ways.
- Older add-ons in `%LOCALAPPDATA%\ewald\site-packages` (installed with `pip install --target`
  from a separate Python 3.12) are still on `sys.path`, as is *Extra module paths* (`python.extra_paths`).

## Online installer (D79)

`packaging/windows/ewald-online.iss`, built by the `online` job. It ships only Python 3.12 (the
`tools\` folder of python.org's NuGet package `python`, a full CPython with pip) and the icon. Setup
copies it to `{app}\python` (per user: `%LOCALAPPDATA%\Programs\ewald-python`), then runs
`python -m pip install "ewald[gui] @ <this release's cp312 wheel on GitHub>"`; everything else
comes from PyPI. The *NVIDIA GPU support* task adds `cupy-cuda12x[ctk]` (CuPy with the CUDA 12
libraries as wheels). pip's output scrolls under the progress bar and goes to `{app}\install.log`;
a failed install says to check the network or use the offline installer, and exits with code 1.

The shortcuts and file association run `python\pythonw.exe -m ewald gui`. The app is not frozen, so
the Plugins side bar uses the normal `python -m pip install` into that Python (no `--prefix`, no
constraints), and `pymatgen`, `mp-api` or anything else a notebook needs installs the same way
(`ewald command prompt` in the Start menu has that Python on `PATH`). Uninstall removes the whole
`python` folder, including what pip put there. It installs next to the offline app (different
AppId and folder).

Why both: the online setup is small and gives a real Python environment, but needs the network
during setup and resolves the dependencies on the user's PC; the offline app works anywhere
(company PCs without internet, firewalls that block PyPI). pip 24.2 and later trusts the Windows
certificate store (truststore) and requests reads the Windows proxy settings, so networks that
re-sign HTTPS work, but a network that blocks pypi.org or files.pythonhosted.org does not.

`/SOURCE=<wheel path or URL>` installs another wheel (the CI smoke step installs the wheel it just
built, silently: `/VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /DIR=...`), then runs `smoke.py`
with that Python.

## CUDA

The CPU build excludes CuPy; `device="auto"` falls back to NumPy as today.

The GPU build bundles `cupy-cuda12x` and the CUDA 12 runtime wheels it loads at run time
(`nvidia-cuda-runtime-cu12`, `-cuda-nvrtc-`, `-cufft-`, `-cublas-`, `-curand-`, `-cusolver-`,
`-cusparse-`). NVRTC is what compiles our `RawKernel` neighbor-list source; cuFFT backs the MSD FFT;
cuBLAS/cuSOLVER back `linalg` calls in the local-order modifiers. The runtime hook adds each bundled
`nvidia/*/bin` folder to the DLL search path. The user needs only an NVIDIA driver new enough for
CUDA 12 (R525 or later); no CUDA Toolkit install. Cost: measured on the first CI build, the CPU
installer and zip are about 240 MB each, the GPU ones about 1.7 GB each (cuBLAS, cuSOLVER and
cuSPARSE dominate), close to GitHub's 2 GiB limit per release file. Dropping libraries we do not
call (cuSPARSE is only needed for `cupyx.scipy.sparse`) is the first size fix.

Kept as a separate installer rather than one build because most machines that open trajectory files
have no NVIDIA GPU, and the CUDA libraries roughly double the download. The GitHub runners have no
GPU, so CI only checks that the GPU build starts and falls back cleanly; a real GPU check is a manual
step before tagging (`ewald run` on a script with `device="cuda"`).

DXA (`ovito`) is not bundled: the `ovito` wheel pins its own PySide6 and is redistributed under its
own terms. Users who need DXA install it into the add-on folder, or use the wheel install.

## Installer: Inno Setup

`packaging/windows/ewald.iss`, built with `iscc` in CI. Per-user install by default (no admin,
`%LOCALAPPDATA%\Programs\ewald`; the dialog still offers all-users), Start-menu entries for the
viewer and an "ewald command prompt", and three optional tasks: desktop shortcut, "Open with" for
`.lammpstrj` / `.dump`, and adding `ewald.exe` to PATH (removed again on uninstall). The portable zip
is the same folder for people who do not want an installer.

Inno Setup over MSI/WiX: one readable script, free, no admin needed, and LZMA2 makes the installer
noticeably smaller. MSI only matters for fleet deployment through Group Policy, which we do not need.

## Code signing

Not now. The repo is private and the audience is the owner and a few colleagues; unsigned builds show
a SmartScreen "Windows protected your PC" prompt once ("More info" then "Run anyway").

If the app goes to more people: Azure Trusted Signing (about USD 10 a month, identity validation,
signing from GitHub Actions with `azure/trusted-signing-action`, no hardware token) is the cheapest
route that builds SmartScreen reputation. Classic OV certificates now require an HSM or cloud key, so
they cost more for the same result, and EV no longer skips SmartScreen. Signing would sign both
executables in `dist/ewald` before `iscc`, then the installer itself.

## Release workflow

`.github/workflows/windows-release.yml`:

1. `wheels`: cibuildwheel builds cp310 to cp313 win_amd64, `delvewheel` vendors the C++ runtime
   (`msvcp140.dll`) that the pybind11 extension needs, and each wheel is test-imported.
2. `app` (matrix `cpu`, `gpu`, Python 3.12): install `.[gui]` (+ CuPy and CUDA wheels for `gpu`),
   PyInstaller, then smoke tests against the frozen app: `ewald.exe info`, `ewald.exe run
   packaging/windows/smoke.py` (LAMMPS/VASP/ASE readers, the C++ extension, numba neighbor list,
   CNA, OpenGL render under Mesa, the numba ray tracer, the main window with the IPython console),
   and `ewald-gui.exe` launched with `EWALD_SMOKE_TEST=1`, which quits after the window is up. Then
   `iscc` and the portable zip.
3. `release` (tags only): attaches all wheels, installers and zips to a GitHub Release with
   generated notes.

It runs only on manual dispatch (Actions › Windows release › Run workflow, or `gh workflow run
windows-release.yml --ref <branch or tag>`), to save Actions minutes and storage (the repo is
private, so both count against the account quota). Dispatched on a `v*` tag it also runs `release`.
Non-release artifacts are kept 3 days. To re-enable automatic runs, add `push: tags: ["v*"]` and a
`pull_request:` trigger with the packaging `paths` back under `on:`; pull-request runs build only
the CPU app. A tag
must match `ewald.__version__`, or the build fails.

To cut a release, follow [Releases](release.md): bump `__version__` in `src/ewald/__init__.py`
(`pyproject.toml` reads it), add the CHANGELOG section, run `packaging/release_check.py`, merge,
push a `v*` tag and dispatch this workflow and `release.yml` on it.

## Later

- Size: VTK is about half the folder. Excluding VTK modules we never import (IO formats, charts,
  web, parallel) with an explicit list could cut a few hundred MB; do it with the smoke test as the
  guard.
- A version resource for the executables (the app icon, `src/ewald/assets/icon/ewald.ico`, is
  already used by the spec and the installer; `EWALD_ICON=path.ico` overrides it).
- Software OpenGL fallback for machines without OpenGL 3.2 (remote desktop, some VMs): ship Mesa's
  `opengl32.dll` in a subfolder and copy it next to the executables behind a `--software-gl` switch.
- In-app update check against the latest GitHub Release.
