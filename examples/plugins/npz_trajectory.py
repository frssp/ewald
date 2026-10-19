"""Example reader plugin: trajectories stored as NumPy ``.npz`` archives.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is) and
``import_file("run.npz")`` works in scripts, the GUI and the Claude tools.

Expected arrays (the layout many MLIP/MD scripts dump):

- ``positions``  (frames, atoms, 3) in Angstrom
- ``types``      (atoms,) element symbols or integers
- ``cell``       (3, 3) or (frames, 3, 3), optional; lattice vectors as rows
- any other (frames, atoms) or (frames, atoms, k) array becomes a particle property,
  e.g. ``forces`` -> "Force", ``energies`` (frames,) -> attribute "Energy".
"""
import numpy as np

from ewald import Cell, DataCollection
from ewald.io.base import FrameReader
from ewald.plugins import reader

_NAMES = {"forces": "Force", "velocities": "Velocity", "charges": "Charge", "energies": "Energy"}


@reader("npz-trajectory", patterns=("*.npz",), description="NumPy archive with positions (frames, atoms, 3)")
class NpzTrajectoryReader(FrameReader):
    def __init__(self, path, type_map=None):
        with np.load(path) as archive:      # .npz members are decompressed once, here
            arrays = {key: archive[key] for key in archive.files}
        if "positions" not in arrays:
            raise ValueError(f"{path}: no 'positions' array (found {sorted(arrays)})")
        self.positions = arrays.pop("positions")
        if self.positions.ndim == 2:
            self.positions = self.positions[None]
        n_frames, n_atoms = self.positions.shape[:2]
        types = arrays.pop("types", np.full(n_atoms, "X"))
        self.types = np.array([(type_map or {}).get(t, t) for t in types.tolist()]).astype(str)
        self.cell = arrays.pop("cell", None)
        self.per_atom = {_name(k): v for k, v in arrays.items() if v.shape[:2] == (n_frames, n_atoms)}
        self.per_frame = {_name(k): v for k, v in arrays.items() if v.shape == (n_frames,)}

    def __len__(self):
        return len(self.positions)

    def read(self, index):
        data = DataCollection(self.positions[index], self.types)
        if self.cell is not None:
            data.cell = Cell(self.cell[index] if self.cell.ndim == 3 else self.cell)
        for name, values in self.per_atom.items():
            data.particles[name] = np.array(values[index])
        for name, values in self.per_frame.items():
            data.attributes[name] = float(values[index])
        return data


def _name(key):
    return _NAMES.get(key, key[:1].upper() + key[1:])
