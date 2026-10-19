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
