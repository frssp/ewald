"""Moved: the PhononModes and the phonopy/VASP phonon readers live in the ewald-phonopy plugin.

``import ewald.phonon`` keeps working when the plugin is installed (this module is replaced by
``ewald_phonopy.phonon``); otherwise it raises ImportError saying how to install it.
"""
import importlib
import sys

try:
    sys.modules[__name__] = importlib.import_module("ewald_phonopy.phonon")
except ImportError as exc:
    raise ImportError("ewald.phonon moved to the ewald-phonopy plugin: install 'phonopy' from the Plugins panel, or "
                      "pip install 'git+https://github.com/frssp/ewald-plugins#subdirectory=packages/ewald-phonopy'",
                      name=__name__) from exc
