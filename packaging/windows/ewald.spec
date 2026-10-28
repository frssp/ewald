# PyInstaller spec for the Windows app: one folder with ewald.exe (console CLI) and
# ewald-gui.exe (windowed), sharing one _internal/ directory.
#
#   pip install ".[gui]" pyinstaller setuptools
#   pyinstaller --noconfirm packaging/windows/ewald.spec
#
# CuPy is never bundled: GPU compute is installed from the app (Preferences › Compute › Optional
# components › Install, or ewald.exe gpu --install) into the add-on folder (D80).
import importlib.metadata
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

HERE = os.path.dirname(os.path.abspath(SPEC))
# the app icon rendered by src/ewald/assets/icon/make_icons.py; EWALD_ICON overrides it
ICON = os.environ.get("EWALD_ICON") or os.path.join(HERE, "..", "..", "src", "ewald", "assets", "icon", "ewald.ico")

hiddenimports = (collect_submodules("ewald")
                 # ASE picks readers by format name at runtime
                 + collect_submodules("ase.io") + collect_submodules("ase.calculators")
                 # pymatgen is a core dependency (D79) that notebooks and plugins import freely
                 + collect_submodules("pymatgen", filter=lambda m: ".tests" not in m)
                 + ["qtconsole.inprocess", "ipykernel.inprocess", "pyvistaqt"])
datas = collect_data_files("ewald") + collect_data_files("ase") + collect_data_files("pymatgen") + collect_data_files("pyvista") + collect_data_files("qtconsole")
# pip and setuptools for the Plugins side bar (ewald.exe -m pip, ewald.plugin_catalog.frozen_pip_command),
# plus every bundled distribution's metadata, so pip sees ewald, numpy, ... as installed and a
# plugin's requirements on them are met
hiddenimports += collect_submodules("pip") + collect_submodules("setuptools") + ["_distutils_hack"]
datas += collect_data_files("pip") + collect_data_files("setuptools")
BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes", "macholib", "delvewheel"}
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    if name and name.lower().replace("_", "-") not in BUILD_ONLY:
        try:
            datas += copy_metadata(name)
        except Exception as exc:  # noqa: BLE001  a broken dist-info in the build env
            print("no metadata for", name, exc)
# the whole standard library (but GUI toolkits and tests): packages pip installs later (CuPy from
# the GPU Install button, plugins) import modules that nothing bundled imports
STDLIB_SKIP = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3", "ensurepip", "venv", "curses",
               "dbm", "msilib", "this", "antigravity", "__phello__", "_pyrepl"}
for name in sorted(sys.stdlib_module_names - STDLIB_SKIP):
    if name.startswith("_") or importlib.util.find_spec(name) is None:
        continue
    hiddenimports += collect_submodules(name, filter=lambda m: ".test" not in m and ".idle" not in m)
binaries = []
# debugpy: ipykernel imports it optionally, but its vendored pydevd tree breaks when frozen
excludes = ["tkinter", "ovito", "torch", "tensorflow", "PyQt5", "PyQt6", "PySide2", "debugpy",
            "cupy", "cupyx", "cupy_backends", "nvidia"]
sys.path.insert(0, os.path.join(HERE, ".."))
from bundle_excludes import VTK_UNUSED  # noqa: E402  (packaging/bundle_excludes.py)

excludes += VTK_UNUSED

a = Analysis(
    [os.path.join(HERE, "launch_cli.py"), os.path.join(HERE, "launch_gui.py")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(HERE, "rthook_ewald.py")],
    excludes=excludes,
    # keep ewald's .py sources next to the bytecode: numba cache=True locates functions by source file
    # pip runs its own files as scripts (build backend hooks), so pip and setuptools stay plain .py
    module_collection_mode={"ewald": "pyz+py", "pip": "py", "setuptools": "py", "_distutils_hack": "py"},
    # bytecode as files in _internal/, shared by both executables: a PYZ archive would be embedded
    # in each of them (62 MB apiece on Windows with pymatgen and the standard library)
    noarchive=True,
)
pyz = PYZ(a.pure)

common = dict(debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
              icon=ICON if os.path.exists(ICON) else None)
cli = EXE(pyz, [s for s in a.scripts if s[0] != "launch_gui"], [], exclude_binaries=True,
          name="ewald", console=True, **common)
gui = EXE(pyz, [s for s in a.scripts if s[0] != "launch_cli"], [], exclude_binaries=True,
          name="ewald-gui", console=False, **common)

coll = COLLECT(cli, gui, a.binaries, a.datas, strip=False, upx=False, name="ewald")
