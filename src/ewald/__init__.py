"""EWALD (Extensible Workbench for Atomic Lattices & Dynamics): atomistic structure & trajectory
visualization and analysis.

Typical use::

    from ewald import import_file
    from ewald.modifiers import CoordinationAnalysis, ColorCoding

    pipe = import_file("XDATCAR")
    pipe.modifiers.append(CoordinationAnalysis(cutoff=3.0))
    pipe.modifiers.append(ColorCoding("Coordination"))
    pipe.render("frame0.png", frame=0)
"""
from .backend import gpu_available, get_backend
from .data import Cell, DataCollection
from .io import StaticSource, export_file, import_file
from .pipeline import ComputeContext, Modifier, Pipeline, PythonModifier

__version__ = "0.1.0"
__title__ = "EWALD: Extensible Workbench for Atomic Lattices & Dynamics"

__all__ = ["Cell", "DataCollection", "Pipeline", "Modifier", "PythonModifier", "ComputeContext",
           "StaticSource", "import_file", "export_file", "get_backend", "gpu_available"]
