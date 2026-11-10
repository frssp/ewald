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
    an ``.ewd`` notebook carries, so a shared notebook finds its files. A file that holds a table
    and no structure (a LAMMPS log, a CV file) raises an error that names :func:`load_table`."""
    from .io import IncompleteFileError
    from .notebook import resolve_data_path

    path = resolve_data_path(path)
    try:
        return import_file(path, *args, **kwargs)
    except Exception as exc:
        if isinstance(exc, IncompleteFileError) or getattr(exc, "reader_keyword", None):
            # A run that is still writing the file, or a keyword its reader does not take
            # (io.open_reader marks that TypeError): true as they are.
            raise
        if _access_error(exc):
            # Nothing was read, so there is nothing to say about what the file holds; looking for a
            # table would connect (and ask for a password) a second time.
            raise
        # Nearly every text file with a row of numbers reads as a table (an extxyz, a POSCAR), so
        # the hint is only true when the file fails as a structure by itself. With format= or
        # reader options the failure may be theirs (a keyword ASE rejects, the wrong format):
        # a file that opens without them keeps the error of the call.
        if (args or kwargs) and _reads_as_structure(path):
            raise
        hint = _table_hint(path)
        if hint:
            if type(exc).__name__ == "UnknownFileTypeError":   # ASE's: no reader knew the file at all
                why = "is not a structure file any reader recognizes"
            else:   # a reader took it and failed on what it found: its reason helps
                why = " ".join(str(exc).split())
                why = f"was not read as a structure ({why if len(why) <= 120 else why[:117] + '...'})"
            raise ValueError(f"{path} {why}, but it reads as a table ({hint}): "
                             f"use load_table({str(path)!r})") from exc
        raise


def _access_error(exc) -> bool:
    """Whether ``exc`` says the file was never reached (missing, no permission, no connection),
    not that a reader opened it and could not use what it found. ASE reports bad content with
    IOError subclasses too, so OSError alone does not tell: one from the system carries an errno."""
    from .io.remote import RemoteError

    return (isinstance(exc, (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, RemoteError))
            or (isinstance(exc, OSError) and exc.errno is not None))


def _reads_as_structure(path) -> bool:
    """Whether ``path`` opens as a structure with the format detected and no reader options."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the caller reports its own error, not this probe's warnings
        try:
            import_file(path)
        except Exception:
            return False
    return True


def _table_hint(path) -> str:
    """The columns of ``path`` when it reads as a table ('' when it does not)."""
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
