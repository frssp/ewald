"""Build the native extension (CNA, PTM, Voronoi and CSP kernels ported from OVITO).

Without a C++17 compiler (on Windows: Microsoft C++ Build Tools) the build is skipped with a
warning and ewald installs without ``ewald._native``: everything works except the modifiers
listed in ``NEEDS_NATIVE``, which then raise an ImportError saying how to get it.
``EWALD_REQUIRE_NATIVE=1`` turns the warning back into a build error (CI, release wheels).
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

NEEDS_NATIVE = ("PolyhedralTemplateMatching", "VoronoiAnalysis", "CentroSymmetry(mode='matching')",
                "CommonNeighborAnalysis(mode='interval')", "ElasticStrain", "GrainSegmentation",
                "IdentifyFCCPlanarFaults")
HELP = {
    "win32": "install 'Microsoft C++ Build Tools' (https://visualstudio.microsoft.com/visual-cpp-build-tools/, "
             "workload 'Desktop development with C++'), open a new terminal",
    "darwin": "run 'xcode-select --install'",
}.get(sys.platform, "install g++ (Debian/Ubuntu: sudo apt install build-essential; RHEL: sudo dnf install gcc-c++)")


class optional_build_ext(build_ext):
    """build_ext that warns instead of failing when no C++ compiler is available."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # no compiler: DistutilsPlatformError, CompileError, ...
            self._skip(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._skip(exc)

    def _skip(self, exc):
        if os.environ.get("EWALD_REQUIRE_NATIVE"):
            raise exc
        bar = "*" * 78
        sys.stderr.write(
            f"\n{bar}\newald: the C++ extension ewald._native was NOT built ({type(exc).__name__}: {exc}).\n"
            f"ewald works without it, except {', '.join(NEEDS_NATIVE)}.\n"
            f"To add it later: {HELP}, then reinstall (pip install -e . or pip install --force-reinstall ewald).\n"
            f"{bar}\n\n")


native = os.path.join("src", "ewald", "_csrc")
sources = [os.path.join(native, "native.cpp"), os.path.join(native, "cna.cpp"), os.path.join(native, "crystal.cpp"),
           os.path.join(native, "voro++", "cell.cc"),
           os.path.join(native, "voro++", "common.cc"),
           os.path.join(native, "mwm_csp", "mwm_csp.cpp"),
           os.path.join(native, "mwm_csp", "matching.cpp")]
sources += sorted(os.path.join(native, "ptm", f) for f in os.listdir(os.path.join(native, "ptm"))
                  if f.endswith(".cpp"))
ext_modules = [Pybind11Extension("ewald._native", sources, include_dirs=[native], cxx_std=17)]

setup(ext_modules=ext_modules, cmdclass={"build_ext": optional_build_ext})
