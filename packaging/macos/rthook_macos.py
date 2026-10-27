# PyInstaller runtime hook: runs before ewald is imported in the frozen macOS app.
import os
import sys
from pathlib import Path

support = Path.home() / "Library" / "Application Support" / "ewald"
# add-on packages (e.g. ovito, anthropic) installed with
#   pip install --target ~/Library/Application\ Support/ewald/site-packages --python-version 3.12 --only-binary=:all: PKG
addons = support / "site-packages"
if addons.is_dir():
    sys.path.append(str(addons))
# plugins installed from the Plugins side bar: pip --prefix support/python
# (ewald.plugin_catalog.addons_prefix and frozen_pip_command)
import sysconfig

prefix = str(support / "python")
plugin_site = sysconfig.get_path("purelib", sysconfig.get_preferred_scheme("prefix"), vars={"base": prefix, "platbase": prefix})
os.makedirs(plugin_site, exist_ok=True)   # on sys.path from the start, so a plugin pip installs imports at once
sys.path.append(plugin_site)
# numba caches compiled kernels next to their source by default, which would write into the signed
# bundle (and break its signature) when /Applications is writable; keep the cache in ~/Library/Caches
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / "Library" / "Caches" / "ewald" / "numba"))
# apps started from Finder get PATH=/usr/bin:/bin:/usr/sbin:/sbin; add Homebrew so ffmpeg (movies),
# plumed and ssh helpers are found as in a terminal
path = os.environ.get("PATH", "").split(os.pathsep)
for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if extra not in path and os.path.isdir(extra):
        path.append(extra)
os.environ["PATH"] = os.pathsep.join(p for p in path if p)
