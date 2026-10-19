# PyInstaller spec for the macOS app: EWALD.app with ewald-gui (the bundle's executable) and the
# ewald CLI side by side in Contents/MacOS, sharing one set of libraries.
#
#   pip install ".[gui]" pyinstaller
#   pyinstaller --noconfirm packaging/macos/ewald.spec          # -> dist/EWALD.app
#   packaging/macos/make_dmg.sh                                  # -> dist/ewald-X.Y.Z-macos-<arch>.dmg
#
# Build on the architecture you ship (arm64 on Apple Silicon, x86_64 on Intel): numba/llvmlite
# wheels are per architecture, so there is no universal2 app. CuPy is never bundled (no CUDA on
# macOS). On Linux the spec still runs and builds the plain folder, which tests use to check it.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = os.path.dirname(os.path.abspath(SPEC))
WIN = os.path.join(HERE, "..", "windows")  # the frozen entry points are shared with Windows
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ewald import __version__  # noqa: E402

hiddenimports = (collect_submodules("ewald")
                 # ASE picks readers by format name at runtime
                 + collect_submodules("ase.io") + collect_submodules("ase.calculators")
                 + ["qtconsole.inprocess", "ipykernel.inprocess", "pyvistaqt"])
datas = collect_data_files("ewald") + collect_data_files("ase") + collect_data_files("pyvista") + collect_data_files("qtconsole")
# debugpy: ipykernel imports it optionally, but its vendored pydevd tree breaks when frozen
excludes = ["tkinter", "ovito", "torch", "tensorflow", "PyQt5", "PyQt6", "PySide2", "debugpy",
            "cupy", "cupyx", "cupy_backends", "nvidia"]

a = Analysis(
    [os.path.join(WIN, "launch_gui.py"), os.path.join(WIN, "launch_cli.py")],
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(HERE, "rthook_macos.py")],
    excludes=excludes,
    # keep ewald's .py sources next to the bytecode: numba cache=True locates functions by source file
    module_collection_mode={"ewald": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(debug=False, bootloader_ignore_signals=False, strip=False, upx=False)
# argv_emulation stays off: the GUI takes Finder's files as QFileOpenEvents (gui/app.py EwaldApplication)
gui = EXE(pyz, [s for s in a.scripts if s[0] != "launch_cli"], [], exclude_binaries=True,
          name="ewald-gui", console=False, argv_emulation=False, **common)
cli = EXE(pyz, [s for s in a.scripts if s[0] != "launch_gui"], [], exclude_binaries=True,
          name="ewald", console=True, **common)
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="ewald")

# Finder offers EWALD under Open With for these; VASP files (POSCAR, XDATCAR, OUTCAR) have no
# extension, so any file can also be dropped on the Dock icon ("public.data", rank Alternate).
EXTENSIONS = ["lammpstrj", "dump", "data", "lmp", "xyz", "extxyz", "cif", "xsf", "axsf", "vasp", "poscar",
              "contcar", "h5md", "h5", "xtc", "trr", "dcd", "pdb", "gro", "traj", "cube", "ewald"]
INFO = {
    "CFBundleName": "EWALD",
    "CFBundleDisplayName": "EWALD",
    "CFBundleExecutable": "ewald-gui",
    "CFBundleShortVersionString": __version__,
    "CFBundleVersion": __version__,
    "LSMinimumSystemVersion": "12.0",   # oldest macOS the PySide6 and VTK wheels support
    "NSHighResolutionCapable": True,    # Retina: Qt and VTK render at device pixels
    "NSRequiresAquaSystemAppearance": False,  # follow the system dark mode
    "CFBundleDocumentTypes": [
        {"CFBundleTypeName": "Atomistic structure or trajectory", "CFBundleTypeRole": "Viewer",
         "LSHandlerRank": "Alternate", "CFBundleTypeExtensions": EXTENSIONS},
        {"CFBundleTypeName": "Data file", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
         "LSItemContentTypes": ["public.data", "public.plain-text"]},
    ],
}

if sys.platform == "darwin":
    app = BUNDLE(coll, name="EWALD.app", icon=os.path.join(ROOT, "src", "ewald", "assets", "icon", "ewald.icns"),
                 bundle_identifier="io.github.frssp.ewald", version=__version__, info_plist=INFO)
