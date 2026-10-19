"""Modifiers that bring in a second file: a trajectory for a topology, or more particles."""
from __future__ import annotations

import numpy as np

from ..pipeline import Modifier
from .trajectory import _match_identifiers


class _FileModifier(Modifier):
    """Opens ``file`` lazily (any format import_file reads) and keeps it open until a
    parameter changes."""

    file_parameters = ("file",)
    file_filter = "All files (*)"
    file: str = ""
    format: str | None = None
    type_map: dict | None = None

    def _source(self):
        if not self.file:
            raise ValueError(f"{self.name}: set 'file'")
        key = (str(self.file), self.format, repr(self.type_map))
        cache = self.__dict__.get("_opened")
        if cache is None or cache[0] != key:
            from ..io import FileSource

            kwargs = {} if self.type_map is None else {"type_map": self.type_map}
            cache = (key, FileSource(str(self.file), self.format, thermo=None, **kwargs))
            self._opened = cache
        return cache[1]

    def _frame(self, frame: int):
        src = self._source()
        return src.compute(frame % src.num_frames)


def _min_image_bond_shifts(data):
    """Bond periodic images recomputed under the minimum-image convention after positions moved."""
    bonds = data.bonds
    if bonds is None or not bonds.count or not data.cell.is_valid:
        return
    i, j = bonds.topology.T
    f = (data.positions[j] - data.positions[i]) @ np.linalg.inv(data.cell.matrix)
    bonds.properties["Periodic Image"] = (-np.round(f) * data.cell.pbc).astype(np.int32)


class LoadTrajectory(_FileModifier):
    """Positions (and other per-particle properties) of every frame from a trajectory file,
    on top of a topology from the pipeline source (OVITO's LoadTrajectoryModifier).

    Typical use: a LAMMPS data file with bonds, charges and molecule ids as the source, the
    LAMMPS dump of the run as ``file``. The pipeline then has as many frames as the trajectory.
    Particles are matched by 'Particle Identifier' when both files have it, else by index (the
    counts must agree). Every trajectory property replaces the topology's except
    'Particle Type' (the topology's names win); the cell and attributes such as the timestep
    come from the trajectory, and bond periodic images follow the minimum-image convention.
    ``type_map`` and ``format`` go to the trajectory reader.
    """

    def __init__(self, file: str = "", format: str | None = None, type_map: dict | None = None):
        self.file = file
        self.format = format
        self.type_map = type_map

    def frame_count(self) -> int | None:
        """Number of frames the pipeline gets (the trajectory's)."""
        try:
            return self._source().num_frames
        except Exception:
            return None

    def modify(self, data, ctx):
        traj = self._frame(ctx.frame)
        if "Particle Identifier" in data and "Particle Identifier" in traj:
            index = _match_identifiers(np.asarray(traj["Particle Identifier"]), np.asarray(data["Particle Identifier"]))
            if (index < 0).any():
                raise ValueError(f"{int((index < 0).sum())} topology particles are missing from the trajectory frame")
        elif traj.count == data.count:
            index = np.arange(data.count)
        else:
            raise ValueError(f"the trajectory frame has {traj.count} particles, the topology {data.count}, "
                             "and there are no identifiers to match them")
        for name, values in traj.particles.items():
            if name == "Particle Type" and name in data:
                continue
            data.particles[name] = np.asarray(values)[index]
        if traj.cell.is_valid:
            data.cell = traj.cell.copy()
        for name, value in traj.attributes.items():
            data.attributes[name] = value
        if data.bonds is not None:
            data.bonds = data.bonds.copy()
            _min_image_bond_shifts(data)
        data.attributes["LoadTrajectory.frame"] = ctx.frame % self._source().num_frames


class CombineDatasets(_FileModifier):
    """Add the particles (and bonds) of another file to the current frame
    (OVITO's CombineDatasetsModifier).

    If ``file`` has several frames, the frame with the same index is used (wrapping around).
    Properties present on only one side are zero-filled on the other; 'Particle Type' names
    are kept as they are. When both sides have 'Particle Identifier', the added particles'
    identifiers are shifted past the largest existing one if any of them collide (and numbered
    that way when the file has none). The cell
    and attributes of the pipeline win; the other file's attributes are added only where the
    name is new. Writes ``CombineDatasets.num_merged``.
    """

    def __init__(self, file: str = "", format: str | None = None, type_map: dict | None = None):
        self.file = file
        self.format = format
        self.type_map = type_map

    def modify(self, data, ctx):
        other = self._frame(ctx.frame)
        n0, n1 = data.count, other.count
        add = dict(other.particles)
        if "Particle Identifier" in data and n0 and n1:
            mine = np.asarray(data["Particle Identifier"])
            theirs = np.asarray(add.get("Particle Identifier", np.arange(1, n1 + 1)))
            if "Particle Identifier" not in add or np.isin(theirs, mine).any():
                add["Particle Identifier"] = theirs - theirs.min() + mine.max() + 1
        out = {}
        for name in dict.fromkeys([*data.particles, *add]):
            parts = []
            for props, n, ref in ((data.particles, n0, add), (add, n1, data.particles)):
                if name in props:
                    parts.append(np.asarray(props[name]))
                else:
                    r = np.asarray(ref[name])
                    fill = "X" if name == "Particle Type" else (0 if r.dtype.kind != "U" else "")
                    parts.append(np.full((n,) + r.shape[1:], fill, r.dtype if r.dtype.kind != "U" else object))
            if any(p.dtype.kind in "UO" for p in parts):
                parts = [p.astype(str) for p in parts]
            out[name] = np.concatenate(parts)
        data.particles = out
        if other.bonds is not None and other.bonds.count:
            shifted = other.bonds.copy()
            shifted.properties["Topology"] = other.bonds.topology + n0
            data.bonds = shifted if data.bonds is None else data.bonds.extend(shifted)
        for name, value in other.attributes.items():
            data.attributes.setdefault(name, value)
        data.attributes["CombineDatasets.num_merged"] = n1
