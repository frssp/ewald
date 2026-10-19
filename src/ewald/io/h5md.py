"""H5MD (HDF5 molecular data, https://www.nongnu.org/h5md/) trajectories, lazy read and streaming write.

Needs ``h5py`` (``pip install ewald[formats]``). Written by LAMMPS ``dump h5md``, ESPResSo,
pyiron, MDAnalysis and ewald itself. Frames are slices of ``/particles/<group>/position/value``
read on demand, so a trajectory of any length opens instantly. Element mapping to ewald:
``position`` -> "Position", ``velocity`` -> "Velocity", ``force`` -> "Force", ``image`` ->
"Periodic Image", ``id`` -> "Particle Identifier", ``mass``/``charge`` -> "Mass"/"Charge",
``species`` -> "Particle Type"; other elements keep their name. Species names come from
string species, ``/parameters/vmd_structure`` (``indexOfSpecies`` + ``name``, what VMD and
ewald write) or ``type_map={1: "Cu"}``; otherwise the species number is the type name, as
for LAMMPS files. Lengths in ``nm`` (``unit`` attribute) are converted to Angstrom.
Scalar ``/observables`` sampled at the frame's step become attributes.
"""
from __future__ import annotations

import os

import numpy as np

from ..data import Cell, DataCollection
from .base import FrameReader

_ELEMENTS = {"position": "Position", "velocity": "Velocity", "force": "Force", "image": "Periodic Image",
             "id": "Particle Identifier", "mass": "Mass", "charge": "Charge", "species": "Particle Type"}
_NAMES = {v: k for k, v in _ELEMENTS.items()}
_LENGTH = {"nm": 10.0, "angstrom": 1.0, "a": 1.0, "pm": 0.01}  # to Angstrom


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError("H5MD files need h5py: pip install h5py (or ewald[formats])") from None
    return h5py


def is_h5md(path) -> bool:
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\x89HDF\r\n\x1a\n":
                return False
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            return "h5md" in f and "particles" in f
    except (OSError, ImportError):
        return False


def _element(group, name):
    """(value dataset or array, step array or None) of a time-dependent or fixed element."""
    if name not in group:
        return None, None
    item = group[name]
    if hasattr(item, "keys") and "value" in item:
        step = item.get("step")
        if step is not None:
            step = np.atleast_1d(step[()])
            if step.ndim == 1 and step.size == 1 and item["value"].shape[0] > 1:  # fixed-interval form
                offset = item["step"].attrs.get("offset", 0)
                step = offset + step[0] * np.arange(item["value"].shape[0])
        return item["value"], step
    return item, None  # time-independent


def _scale(item) -> float:
    unit = item.attrs.get("unit", b"") if hasattr(item, "attrs") else b""
    unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    parts = unit.split()
    return _LENGTH.get(parts[0].lower(), 1.0) if parts else 1.0


class H5mdReader(FrameReader):
    """H5MD particles group (the first, or ``group=``), lazy."""

    def __init__(self, path, group: str | None = None, type_map: dict | None = None):
        h5py = _h5py()
        self.path = str(path)
        self.type_map = {str(k): v for k, v in (type_map or {}).items()}
        with h5py.File(self.path, "r") as f:
            groups = list(f["particles"])
            if not groups:
                raise ValueError(f"{path}: no particle groups")
            self.group = group or groups[0]
            if self.group not in f["particles"]:
                raise KeyError(f"{path}: no particle group {self.group!r} (has {groups})")
            g = f["particles"][self.group]
            value, step = _element(g, "position")
            if value is None:
                raise ValueError(f"{path}: particle group {self.group!r} has no position")
            self._n = value.shape[0] if step is not None else 1
            self._steps = step if step is not None else np.zeros(1, int)
            self.elements = [k for k in g if k != "box"]
            self._names = self._species_names(f)

    def _species_names(self, f):
        vmd = f.get("parameters/vmd_structure")
        if vmd is not None and "name" in vmd:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in vmd["name"][()]]
            index = vmd["indexOfSpecies"][()] if "indexOfSpecies" in vmd else np.arange(len(names))
            return {str(int(i)): n for i, n in zip(index, names)}
        return {}

    def __len__(self):
        return self._n

    def _frame_index(self, steps, index):
        if steps is None:
            return None  # time-independent
        if len(steps) == self._n:
            return index
        hit = np.flatnonzero(steps == self._steps[index])
        return int(hit[0]) if hit.size else -1

    def read(self, index: int) -> DataCollection:
        h5py = _h5py()
        index = range(self._n)[index]
        with h5py.File(self.path, "r") as f:
            g = f["particles"][self.group]
            value, step = _element(g, "position")
            pos = np.asarray(value[index] if step is not None else value[()], float) * _scale(value)
            data = DataCollection(pos.reshape(-1, 3))
            data.cell = self._cell(g, index)
            for name in self.elements:
                if name == "position":
                    continue
                value, step = _element(g, name)
                if value is None or not hasattr(value, "shape"):
                    continue
                k = self._frame_index(step, index)
                if k == -1:
                    continue
                arr = np.asarray(value[k] if k is not None else value[()])
                if arr.ndim == 0 or len(arr) != len(pos):
                    continue
                prop = _ELEMENTS.get(name, name)
                if prop == "Particle Type":
                    arr = np.array([self._type_name(t) for t in arr])
                elif prop == "Velocity":
                    arr = arr * _scale(value)
                data.particles[prop] = arr
            data.attributes["Timestep"] = int(self._steps[index])
            time = g["position"].get("time") if hasattr(g["position"], "keys") else None
            if time is not None and time.shape and time.shape[0] == self._n:
                data.attributes["Time"] = float(time[index])
            self._observables(f, data, index)
        return data

    def _type_name(self, t):
        if isinstance(t, bytes):
            return t.decode()
        key = str(int(t)) if np.issubdtype(type(t), np.integer) else str(t)
        return self.type_map.get(key) or self._names.get(key) or key

    def _cell(self, g, index):
        box = g.get("box")
        if box is None:
            return Cell()
        boundary = box.attrs.get("boundary")
        pbc = np.array([(b.decode() if isinstance(b, bytes) else str(b)) == "periodic" for b in boundary]) \
            if boundary is not None else np.ones(3, bool)
        value, step = _element(box, "edges")
        if value is None:
            return Cell(np.zeros((3, 3)), np.zeros(3), pbc)
        k = self._frame_index(step, index)
        edges = np.asarray(value[max(k, 0)] if k is not None else value[()], float) * _scale(value)
        matrix = np.diag(edges) if edges.ndim == 1 else edges
        origin = np.zeros(3)
        offset, ostep = _element(box, "offset")
        if offset is not None:
            k = self._frame_index(ostep, index)
            origin = np.asarray(offset[max(k, 0)] if k is not None else offset[()], float) * _scale(offset)
        return Cell(matrix, origin, pbc)

    def _observables(self, f, data, index):
        obs = f.get("observables")
        if obs is None:
            return
        for name, item in obs.items():
            if not hasattr(item, "keys") or "value" not in item:
                continue
            value, step = _element(obs, name)
            k = self._frame_index(step, index)
            if k is None or k == -1 or value.ndim != 1:
                continue
            data.attributes[name] = value[k].item()


def write_h5md(path, frames, group: str = "all", author: str = "ewald", compression: str | None = "gzip"):
    """Stream ``frames`` into an H5MD file (one particle group, time-dependent box).

    Writes position, species (names in ``/parameters/vmd_structure``, so VMD and ewald
    read element names back), every other numeric per-particle property (velocity,
    force, id, mass, charge, image, and the rest under their ewald names) and scalar
    attributes as ``/observables``. The particle count must stay constant.
    """
    h5py = _h5py()
    from .. import __version__ as version

    with h5py.File(os.fspath(path), "w") as f:
        h = f.create_group("h5md")
        h.attrs["version"] = np.array([1, 1])
        h.create_group("author").attrs["name"] = author
        creator = h.create_group("creator")
        creator.attrs["name"] = "ewald"
        creator.attrs["version"] = version
        g = f.create_group("particles").create_group(group)
        box = g.create_group("box")
        box.attrs["dimension"] = 3
        series: dict[str, h5py.Group] = {}
        species = None
        n = None
        for t, data in enumerate(frames):
            if n is None:
                n = data.count
                box.attrs["boundary"] = np.array(["periodic" if p else "none" for p in data.cell.pbc], dtype="S8")
                names = list(dict.fromkeys(str(x) for x in data.types))
                index = {s: i for i, s in enumerate(names)}
                species = np.array([index[str(x)] for x in data.types], dtype=np.int32)
                g.create_dataset("species", data=species)
                vmd = f.create_group("parameters").create_group("vmd_structure")
                vmd.create_dataset("indexOfSpecies", data=np.arange(len(names), dtype=np.int32))
                vmd.create_dataset("name", data=np.array(names, dtype="S"))
            elif data.count != n:
                raise ValueError(f"H5MD needs a constant particle count (frame {t} has {data.count}, not {n})")
            elif any(str(a) != b for a, b in zip(data.types, (names[i] for i in species))):
                raise ValueError(f"frame {t}: particle types changed; H5MD species are written once")
            step = int(data.attributes.get("Timestep", t))
            time = float(data.attributes.get("Time", step))
            _append(series, box, "edges", data.cell.matrix if data.cell.is_valid else np.zeros((3, 3)),
                    step, time, compression, unit="Angstrom")
            _append(series, box, "offset", data.cell.origin, step, time, compression, unit="Angstrom")
            for prop, arr in data.particles.items():
                arr = np.asarray(arr)
                if prop == "Particle Type" or arr.dtype.kind not in "biuf":
                    continue
                name = _NAMES.get(prop, prop)
                unit = "Angstrom" if name == "position" else None
                _append(series, g, name, arr, step, time, compression, unit=unit)
            for key, value in data.attributes.items():
                if key in ("Timestep", "Time", "SourceFrame") or not np.isscalar(value) or isinstance(value, str):
                    continue
                if t == 0 or f"observables/{key}" in series:
                    obs = f.require_group("observables")
                    _append(series, obs, key, np.asarray(value, float), step, time, compression,
                            key=f"observables/{key}")


def _append(series, parent, name, value, step, time, compression, unit=None, key=None):
    key = key or f"{parent.name}/{name}"
    value = np.asarray(value)
    if key not in series:
        el = parent.create_group(name)
        el.create_dataset("value", shape=(0,) + value.shape, maxshape=(None,) + value.shape, dtype=value.dtype,
                          chunks=(1,) + value.shape if value.ndim else (1024,),
                          compression=compression if value.ndim else None)
        el.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        el.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        if unit:
            el["value"].attrs["unit"] = unit
        series[key] = el
    el = series[key]
    k = el["value"].shape[0]
    if value.shape != el["value"].shape[1:]:
        raise ValueError(f"H5MD element {name!r} changed shape from {el['value'].shape[1:]} to {value.shape}")
    for ds, v in (("value", value), ("step", step), ("time", time)):
        el[ds].resize(k + 1, axis=0)
        el[ds][k] = v
