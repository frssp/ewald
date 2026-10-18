"""Build the native extension (CNA, PTM, Voronoi and CSP kernels ported from OVITO)."""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

native = os.path.join("src", "ewald", "_csrc")
sources = [os.path.join(native, "native.cpp"), os.path.join(native, "cna.cpp"),
           os.path.join(native, "voro++", "cell.cc"),
           os.path.join(native, "voro++", "common.cc"),
           os.path.join(native, "mwm_csp", "mwm_csp.cpp"),
           os.path.join(native, "mwm_csp", "matching.cpp")]
sources += sorted(os.path.join(native, "ptm", f) for f in os.listdir(os.path.join(native, "ptm"))
                  if f.endswith(".cpp"))
ext_modules = [Pybind11Extension("ewald._native", sources, include_dirs=[native], cxx_std=17)]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
