"""Histogram, spatial binning and property reduction (OVITO's Histogram, Spatial Binning and Reduce
Property modifiers).

Results match OVITO 3.16 (tests/data/make_analysis_reference.py). The Histogram binning rule follows
HistogramModifier.cpp (Copyright OVITO GmbH, used under the MIT option of its GPLv3-or-MIT license;
see ewald/_csrc/LICENSE.OVITO-MIT.txt).
"""
from __future__ import annotations

import numpy as np

from ..data import Cell, VoxelGrid
from ..pipeline import Modifier

_LETTERS = {"Color": "RGB", "Topology": "AB"}   # component names; "XYZW" otherwise


def _container(data, operate_on: str) -> dict:
    if operate_on == "particles":
        return data.particles
    if operate_on == "bonds":
        if data.bonds is None:
            raise ValueError("operate_on='bonds' needs bonds; add CreateBonds first")
        return data.bonds.properties
    raise ValueError("operate_on must be 'particles' or 'bonds'")


def property_values(props: dict, name: str):
    """``(values, label)`` for 'Mass', 'Position.Z', 'Color.R' or 'Position.2' (1-based) in ``props``;
    a vector property without a component comes back 2D."""
    if name in props:
        return np.asarray(props[name]), name
    base, _, comp = name.rpartition(".")
    if base in props:
        arr = np.asarray(props[base])
        letters = _LETTERS.get(base, "XYZW")
        k = int(comp) - 1 if comp.isdigit() else letters.find(comp.upper()) if comp else None
        if arr.ndim == 2 and k is not None and 0 <= k < arr.shape[1]:
            return arr[:, k], name
    raise KeyError(f"no property {name!r}; available: {', '.join(props)}")


def _components(values, label):
    """Split a vector property into ``[(label.X, column), ...]``."""
    if values.ndim == 1:
        return [(label, values)]
    letters = _LETTERS.get(label, "XYZW")
    return [(f"{label}.{letters[k] if k < len(letters) else k + 1}", values[:, k]) for k in range(values.shape[1])]


def _selected(props: dict, only_selected: bool):
    if not only_selected:
        return None
    if "Selection" not in props:
        raise ValueError("only_selected=True but there is no 'Selection' property")
    return np.asarray(props["Selection"]) != 0


class Histogram(Modifier):
    """Histogram of a particle or bond property ('Mass', 'Position.Z', ...).

    The range is the min..max of the values unless ``fix_xrange`` (then ``xrange_start``..
    ``xrange_end``; values outside are skipped). ``normalization``: "count", "frequency"
    (relative frequency) or "density" (probability density). ``select_elements`` selects the
    elements with values in ``selection_start``..``selection_end`` and sets
    ``Histogram.num_selected``. Writes ``tables['histogram[<property>]']`` with the bin centers and
    'Count' / 'Relative Frequency' / 'Probability Density'.
    """

    def __init__(self, property: str = "Position.Z", bin_count: int = 200, fix_xrange: bool = False,
                 xrange_start: float = 0.0, xrange_end: float = 0.0, only_selected: bool = False,
                 normalization: str = "count", select_elements: bool = False, selection_start: float = 0.0,
                 selection_end: float = 0.0, operate_on: str = "particles"):
        self.property = property
        self.bin_count = bin_count
        self.fix_xrange = fix_xrange
        self.xrange_start = xrange_start
        self.xrange_end = xrange_end
        self.only_selected = only_selected
        self.normalization = normalization
        self.select_elements = select_elements
        self.selection_start = selection_start
        self.selection_end = selection_end
        self.operate_on = operate_on

    def modify(self, data, ctx):
        props = _container(data, self.operate_on)
        values, label = property_values(props, self.property)
        if values.ndim != 1:
            raise ValueError(f"{self.property!r} is a vector property; pick a component like '{label}.X'")
        norm = {"count": "count", "absolutecount": "count", "frequency": "frequency",
                "relativefrequency": "frequency", "density": "density",
                "probabilitydensity": "density"}.get(str(self.normalization).replace(" ", "").lower())
        if norm is None:
            raise ValueError("normalization must be 'count', 'frequency' or 'density'")
        sel = _selected(props, self.only_selected)
        nbins = max(1, int(self.bin_count))
        counts = np.zeros(nbins, np.int64)
        v = values if sel is None else values[sel]
        v = v.astype(float)
        if len(values):
            if self.fix_xrange:
                start, end = float(self.xrange_start), float(self.xrange_end)
            else:
                start, end = (float(v.min()), float(v.max())) if len(v) else (0.0, 0.0)
            size = (end - start) / nbins
            if end > start:
                v = v[(v >= start) & (v <= end)]
                idx = np.clip(((v - start) / size).astype(np.int64), 0, nbins - 1)
                counts = np.bincount(idx, minlength=nbins).astype(np.int64)
            else:
                counts[0] = len(v)
        else:
            start = end = size = 0.0
        x = start + (np.arange(nbins) + 0.5) * size
        if norm == "frequency":
            y, yname = counts / max(1, counts.sum()), "Relative Frequency"
        elif norm == "density":
            y = counts / (max(1, counts.sum()) * size) if size != 0 else np.zeros(nbins)
            yname = "Probability Density"
        else:
            y, yname = counts, "Count"
        data.tables[f"histogram[{label}]"] = {label: x, yname: y}
        if self.select_elements:
            lo, hi = sorted((float(self.selection_start), float(self.selection_end)))
            chosen = (values >= lo) & (values <= hi)
            if sel is not None:
                chosen &= sel
            props["Selection"] = chosen.astype(np.int8)
            data.attributes["Histogram.num_selected"] = int(chosen.sum())


_AXES = {"X": (0,), "Y": (1,), "Z": (2,), "XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2), "XYZ": (0, 1, 2)}
_OPS = ("mean", "sum", "sumvol", "min", "max")


class SpatialBinning(Modifier):
    """Coarse-grain a particle property on a 1D, 2D or 3D grid of bins along the cell vectors.

    ``direction`` is "X", "Y", "Z" (1D), "XY", "XZ", "YZ" (2D) or "XYZ" (3D), meaning the first,
    second and third cell vector; ``bin_count`` is an int or one count per binned direction.
    ``reduction_operation``: "mean", "sum", "sumvol" (sum divided by the bin volume), "min" or
    "max"; empty bins are 0. ``property=""`` bins the value 1 (counts particles; with "sumvol" the
    number density). Particles are wrapped into the cell along periodic directions and skipped
    outside it along open ones. ``first_derivative`` (1D) writes the central finite-difference
    derivative instead. 1D writes ``tables['binning']`` (bin centers along the cell vector and one
    column per component; its ``attributes["axis"]`` names the vector); 2D/3D write ``grids['binning']`` (shape ``(n1, n2, 1)`` or
    ``(n1, n2, n3)``, cell-centered bins).
    """

    def __init__(self, property: str = "", reduction_operation: str = "mean", direction: str = "Z",
                 bin_count=(50, 50, 20), first_derivative: bool = False, only_selected: bool = False):
        self.property = property
        self.reduction_operation = reduction_operation
        self.direction = direction
        self.bin_count = bin_count
        self.first_derivative = first_derivative
        self.only_selected = only_selected

    def modify(self, data, ctx):
        axes = _AXES.get(str(self.direction).upper())
        if axes is None:
            raise ValueError(f"direction must be one of {', '.join(_AXES)}")
        op = str(self.reduction_operation).lower()
        if op not in _OPS:
            raise ValueError(f"reduction_operation must be one of {', '.join(_OPS)}")
        if not data.cell.is_valid:
            raise ValueError("SpatialBinning needs a simulation cell")
        counts = np.atleast_1d(np.asarray(self.bin_count, int))
        shape = tuple(max(1, int(counts[k] if k < len(counts) else counts[-1])) for k in range(len(axes)))
        if self.property:
            values, label = property_values(data.particles, self.property)
        else:
            values, label = np.ones(data.count), "Unity"
        columns = _components(values.astype(float), label)

        keep = np.ones(data.count, bool)
        sel = _selected(data.particles, self.only_selected)
        if sel is not None:
            keep &= sel
        frac = data.cell.to_fractional(data.positions)
        flat = np.zeros(data.count, np.int64)
        for a, n in zip(axes, shape):
            f = frac[:, a]
            if data.cell.pbc[a]:
                f = f - np.floor(f)
            else:
                keep &= (f >= 0) & (f < 1)
            idx = np.clip(np.floor(f * n), 0, n - 1).astype(np.int64)
            flat = flat * n + idx
        flat = flat[keep]
        nbins = int(np.prod(shape))
        bin_volume = data.cell.volume / nbins
        out = []
        for name, col in columns:
            v = col[keep]
            if op in ("mean", "sum", "sumvol"):
                res = np.bincount(flat, weights=v, minlength=nbins)
                if op == "mean":
                    num = np.bincount(flat, minlength=nbins)
                    res = np.divide(res, num, out=np.zeros(nbins), where=num > 0)
                elif op == "sumvol":
                    res = res / bin_volume
            else:
                fill = np.inf if op == "min" else -np.inf
                res = np.full(nbins, fill)
                (np.minimum if op == "min" else np.maximum).at(res, flat, v)
                res[~np.isfinite(res)] = 0.0
            out.append((name, res.reshape(shape)))

        if len(axes) == 1:
            a = axes[0]
            n = shape[0]
            length = float(np.linalg.norm(data.cell.matrix[a]))
            width = length / n
            table = {"Position": (np.arange(n) + 0.5) * width}
            for name, res in out:
                if self.first_derivative:
                    res = _derivative(res, width, bool(data.cell.pbc[a]))
                    name = f"d({name})/d(Position)"
                table[name] = res
            from ..verbs import Table

            # "axis": the 3D view and the Plots panel link the profile to its slab of atoms (ewald.linking)
            data.tables["binning"] = Table("binning", table, attributes={"axis": "abc"[a]})
        else:
            m = data.cell.matrix
            vectors = [m[a] for a in axes]
            if len(axes) == 2:  # OVITO spans the 2D grid with the two binned cell vectors and +z
                vectors.append(np.array([0.0, 0.0, 1.0]))
            pbc = [bool(data.cell.pbc[a]) for a in axes] + [False] * (3 - len(axes))
            domain = Cell(np.array(vectors), data.cell.origin.copy(), np.array(pbc))
            grid = VoxelGrid(domain, shape=shape + (1,) * (3 - len(shape)))
            for name, res in out:
                grid[name] = res.reshape(grid.shape)
            data.grids["binning"] = grid


def _derivative(y, h, periodic):
    """Central differences; one-sided at the ends of an open direction."""
    n = len(y)
    if n < 2:
        return np.zeros(n)
    if periodic:
        return (np.roll(y, -1) - np.roll(y, 1)) / (2 * h)
    d = np.empty(n)
    d[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    d[0] = (y[1] - y[0]) / h
    d[-1] = (y[-1] - y[-2]) / h
    return d


_REDUCE = {"min": "Min", "max": "Max", "sum": "Sum", "nonzero": "Nonzero", "mean": "Mean", "median": "Median",
           "variance": "Variance", "stddev": "StdDev", "argmin": "ArgMin", "argmax": "ArgMax"}


class ReduceProperty(Modifier):
    """Reduce a particle or bond property to one number per component.

    ``reduction_operation``: "min", "max", "sum", "nonzero" (count of nonzero values), "mean",
    "median", "variance" and "stddev" (sample, N-1), "argmin" / "argmax" (index of the element).
    Writes ``ReduceProperty.<operate_on>.<Op>.<property>`` (e.g.
    ``ReduceProperty.particles.Mean.Position.X`` for each component of a vector property).
    """

    def __init__(self, property: str = "Position.Z", reduction_operation: str = "mean",
                 only_selected: bool = False, operate_on: str = "particles"):
        self.property = property
        self.reduction_operation = reduction_operation
        self.only_selected = only_selected
        self.operate_on = operate_on

    def modify(self, data, ctx):
        op = _REDUCE.get(str(self.reduction_operation).lower())
        if op is None:
            raise ValueError(f"reduction_operation must be one of {', '.join(_REDUCE)}")
        props = _container(data, self.operate_on)
        values, label = property_values(props, self.property)
        sel = _selected(props, self.only_selected)
        index = np.arange(len(values)) if sel is None else np.nonzero(sel)[0]
        for name, col in _components(values, label):
            v = col[index]
            data.attributes[f"ReduceProperty.{self.operate_on}.{op}.{name}"] = _reduce(op, v, index)


def _reduce(op, v, index):
    if op == "Nonzero":
        return int(np.count_nonzero(v))
    if not len(v):
        return 0.0 if op not in ("ArgMin", "ArgMax") else -1
    if op in ("ArgMin", "ArgMax"):
        return int(index[np.argmin(v) if op == "ArgMin" else np.argmax(v)])
    if op == "Variance":
        return float(np.var(v, ddof=1)) if len(v) > 1 else 0.0
    if op == "StdDev":
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    fn = {"Min": np.min, "Max": np.max, "Sum": np.sum, "Mean": np.mean, "Median": np.median}[op]
    out = fn(v)
    return out.item() if v.dtype.kind in "iub" and op in ("Min", "Max", "Sum") else float(out)
