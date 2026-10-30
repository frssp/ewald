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
from .pipeline import (AllFramesModifier, ComputeCancelled, ComputeContext, Frames, Modifier, Pipeline,
                       PythonModifier, all_frames, each_frame)
from .view3d import View, view
from .views import show
from .verbs import bond_angles, clusters, cna, load_table, msd, ptm, rdf, sensitivity, sq, vacf, voronoi, xrd
from .runinfo import check_run, run_info
from .runs import arrhenius, load_runs
from .workflow_table import workflow_table



def load(path, *args, **kwargs) -> Pipeline:
    """:func:`import_file` under the short name notebooks use (``pipe = load("XDATCAR")``).
    In a notebook a relative path is also looked up next to the notebook file and in the data
    an ``.ewd`` notebook carries, so a shared notebook finds its files."""
    from .notebook import resolve_data_path

    path = resolve_data_path(path)
    try:
        return import_file(path, *args, **kwargs)
    except Exception as exc:
        hint = _table_hint(path)
        if hint:
            raise ValueError(f"{path} holds no structure, it is a table ({hint}): "
                             f"read it with load_table({str(path)!r})") from exc
        raise


def _table_hint(path) -> str:
    """The columns of ``path`` when it reads as a table and not as a structure ('' otherwise)."""
    import os

    if not isinstance(path, (str, os.PathLike)):
        return ""
    try:
        from .verbs import load_table

        columns = list(load_table(path))
    except Exception:
        return ""
    return "columns " + ", ".join(columns[:6]) + (", ..." if len(columns) > 6 else "")


__version__ = "0.2.0"
__title__ = "EWALD: Extensible Workbench for Atomic Lattices & Dynamics"

__all__ = ["Cell", "DataCollection", "Pipeline", "Modifier", "PythonModifier", "AllFramesModifier", "Frames", "each_frame", "all_frames",
           "ComputeContext", "ComputeCancelled",
           "StaticSource", "import_file", "load", "load_table", "View", "view", "show", "export_file", "get_backend", "gpu_available",
           "rdf", "msd", "vacf", "sq", "xrd", "bond_angles", "cna", "ptm", "voronoi", "clusters", "sensitivity",
           "run_info", "check_run", "load_runs", "arrhenius", "workflow_table"]
