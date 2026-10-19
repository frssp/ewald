"""Moved: the PLUMED engine, COLVAR/HILLS parsing and fes_from_hills live in the ewald-plumed plugin.

``import ewald.plumed`` keeps working when the plugin is installed (this module is replaced by
``ewald_plumed.plumed``); otherwise it raises ImportError saying how to install it.
"""
import importlib
import sys

try:
    sys.modules[__name__] = importlib.import_module("ewald_plumed.plumed")
except ImportError as exc:
    raise ImportError("ewald.plumed moved to the ewald-plumed plugin: install 'plumed' from the Plugins panel, or "
                      "pip install 'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-plumed'",
                      name=__name__) from exc
