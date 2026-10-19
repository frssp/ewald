"""Built-in modifiers."""
from ..pipeline import Modifier, PythonModifier
from .analysis import CalculateDisplacements, CoordinationAnalysis
from .bonds import CoordinationPolyhedra, CreateBonds, EditBonds
from .cluster import ClusterAnalysis
from .coloring import AssignColor, ColorCoding
from .structure import CommonNeighborAnalysis, IdentifyDiamond, PolyhedralTemplateMatching
from .voronoi import VoronoiAnalysis
from .wigner_seitz import WignerSeitzAnalysis
from .local_order import AcklandJones, AtomicStrain, CentroSymmetry, ChillPlus
from .ovito_bridge import DislocationAnalysis, OvitoModifier
from .selection import (ClearSelection, DeleteSelected, ExpressionSelection, InvertSelection,
                        SelectType, Slice)
from .script import PythonScript
from .statistics import Histogram, ReduceProperty, SpatialBinning
from .steinhardt import SteinhardtOrderParameters
from .topology import BondAngleDistribution, BondLengthDistribution, FindRings
from .trajectory import (GenerateTrajectoryLines, MeanSquaredDisplacement, TimeAveraging, TimeSeries,
                         VelocityAutocorrelation)
from .transform import AffineTransformation, ComputeProperty, Replicate, WrapAtoms
from .vectors import VectorVis
from .volume import CreateIsosurface, GridDifference, GridSlice

__all__ = ["Modifier", "PythonModifier", "CalculateDisplacements", "CoordinationAnalysis", "ClusterAnalysis",
           "AssignColor", "ColorCoding", "ClearSelection", "DeleteSelected", "ExpressionSelection",
           "InvertSelection", "SelectType", "Slice", "AffineTransformation", "ComputeProperty",
           "Replicate", "WrapAtoms", "PythonScript", "CommonNeighborAnalysis", "PolyhedralTemplateMatching",
           "VoronoiAnalysis", "WignerSeitzAnalysis", "DislocationAnalysis", "OvitoModifier",
           "AcklandJones", "AtomicStrain", "CentroSymmetry", "ChillPlus", "IdentifyDiamond",
           "MeanSquaredDisplacement", "VelocityAutocorrelation", "TimeAveraging", "TimeSeries",
           "GenerateTrajectoryLines", "CreateBonds", "EditBonds", "CoordinationPolyhedra",
           "CreateIsosurface", "GridSlice", "GridDifference", "Histogram", "SpatialBinning", "ReduceProperty",
           "SteinhardtOrderParameters", "BondAngleDistribution", "BondLengthDistribution", "FindRings", "VectorVis"]

# OVITO spellings (``CommonNeighborAnalysisModifier``, ``WrapPeriodicImagesModifier``, ...) resolve to the
# ewald class, so OVITO scripts - and code written by LLMs that learned OVITO - run unchanged.
_OVITO_NAMES = {"WrapPeriodicImagesModifier": "WrapAtoms", "PythonScriptModifier": "PythonScript"}


def __getattr__(name):
    target = _OVITO_NAMES.get(name) or (name[:-len("Modifier")] if name.endswith("Modifier") else None)
    if target and target in __all__:
        return globals()[target]
    if not name.startswith("_"):  # modifier plugins (ewald.plugins) import like built-ins
        from .. import plugins

        found = plugins.modifiers().get(name) or (plugins.modifiers().get(target) if target else None)
        if found is not None:
            return found.cls
    raise AttributeError(f"module 'ewald.modifiers' has no attribute {name!r}")
