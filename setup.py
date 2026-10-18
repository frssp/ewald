"""Build the optional native extension (PTM + Voronoi kernels ported from OVITO)."""
import os

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pybind11 missing: install pure-Python package only
    ext_modules, cmdclass = [], {}
else:
    native = os.path.join("src", "ewald", "_csrc")
    sources = [os.path.join(native, "native.cpp"), os.path.join(native, "cna.cpp"),
               os.path.join(native, "voro++", "cell.cc"),
               os.path.join(native, "voro++", "common.cc"),
               os.path.join(native, "mwm_csp", "mwm_csp.cpp"),
               os.path.join(native, "mwm_csp", "matching.cpp")]
    sources += sorted(os.path.join(native, "ptm", f) for f in os.listdir(os.path.join(native, "ptm"))
                      if f.endswith(".cpp"))
    ext_modules = [Pybind11Extension("ewald._native", sources, include_dirs=[native], cxx_std=17,
                                     optional=True)]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
