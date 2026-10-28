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
# plugins installed from the Plugins side bar: pip --prefix local/python
# (ewald.plugin_catalog.addons_prefix and frozen_pip_command)
import sysconfig

prefix = str(local / "python")
plugin_site = sysconfig.get_path("purelib", sysconfig.get_preferred_scheme("prefix"), vars={"base": prefix, "platbase": prefix})
os.makedirs(plugin_site, exist_ok=True)   # on sys.path from the start, so a plugin pip installs imports at once
sys.path.append(plugin_site)
# GPU compute installed from the app (cupy-cuda12x[ctk] into plugin_site): ewald.backend.cuda_dll_dirs
# puts the CUDA wheels' DLL folders on the search path before CuPy loads
