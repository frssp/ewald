# PyInstaller spec for the Windows app: one folder with ewald.exe (console CLI) and
# ewald-gui.exe (windowed), sharing one _internal/ directory.
#
#   pip install ".[gui]" pyinstaller setuptools            (+ cupy-cuda12x and nvidia-*-cu12 for the GPU build)
#   pyinstaller --noconfirm packaging/windows/ewald.spec
#
# EWALD_VARIANT=gpu bundles CuPy and the CUDA runtime wheels; otherwise CuPy is excluded.
import os

import importlib.metadata

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

HERE = os.path.dirname(os.path.abspath(SPEC))
# the app icon rendered by src/ewald/assets/icon/make_icons.py; EWALD_ICON overrides it
ICON = os.environ.get("EWALD_ICON") or os.path.join(HERE, "..", "..", "src", "ewald", "assets", "icon", "ewald.ico")
GPU = os.environ.get("EWALD_VARIANT", "cpu") == "gpu"

hiddenimports = (collect_submodules("ewald")
                 # ASE picks readers by format name at runtime
                 + collect_submodules("ase.io") + collect_submodules("ase.calculators")
                 + ["qtconsole.inprocess", "ipykernel.inprocess", "pyvistaqt"])
datas = collect_data_files("ewald") + collect_data_files("ase") + collect_data_files("pyvista") + collect_data_files("qtconsole")
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
binaries = []
# debugpy: ipykernel imports it optionally, but its vendored pydevd tree breaks when frozen
excludes = ["tkinter", "ovito", "torch", "tensorflow", "PyQt5", "PyQt6", "PySide2", "debugpy"]
if GPU:
    hiddenimports += collect_submodules("cupy") + collect_submodules("cupyx") + collect_submodules("cupy_backends")
    datas += collect_data_files("cupy", include_py_files=False)
    # nvidia-*-cu12 wheels: DLLs under nvidia/<lib>/bin, kept at the same relative path
    import importlib.util

    if importlib.util.find_spec("nvidia") is not None:
        binaries += collect_dynamic_libs("nvidia")
else:
    excludes += ["cupy", "cupyx", "cupy_backends", "nvidia"]

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
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
              icon=ICON if os.path.exists(ICON) else None)
cli = EXE(pyz, [s for s in a.scripts if s[0] != "launch_gui"], [], exclude_binaries=True,
          name="ewald", console=True, **common)
gui = EXE(pyz, [s for s in a.scripts if s[0] != "launch_cli"], [], exclude_binaries=True,
          name="ewald-gui", console=False, **common)

coll = COLLECT(cli, gui, a.binaries, a.datas, strip=False, upx=False, name="ewald")
