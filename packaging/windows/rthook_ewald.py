# PyInstaller runtime hook: runs before ewald is imported in the frozen app.
import os
import sys
from pathlib import Path

local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "ewald"
# add-on packages (e.g. ovito, anthropic) installed with
#   pip install --target %LOCALAPPDATA%\ewald\site-packages --python-version 3.12 --only-binary=:all: PKG
addons = local / "site-packages"
if addons.is_dir():
    sys.path.append(str(addons))
# GPU build: CUDA DLLs from the bundled nvidia-*-cu12 wheels
nvidia = Path(sys._MEIPASS) / "nvidia"
if nvidia.is_dir() and hasattr(os, "add_dll_directory"):
    for b in nvidia.glob("*/bin"):
        os.add_dll_directory(str(b))
        os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
