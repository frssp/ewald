"""Phonon mode files as a looping trajectory.

- ``phonopy-yaml``: phonopy band.yaml / mesh.yaml / qpoints.yaml with eigenvectors;
- ``vasp-phonon``: VASP OUTCARs of IBRION=5-8 runs (Gamma modes).

Every frame is the primitive cell with the modes in ``data.attributes["Phonon Modes"]`` (a
:class:`ewald.phonon.PhononModes`) and the frequencies in the table "Phonon Bands"; the
:class:`ewald.modifiers.PhononMode` modifier displaces the atoms along one mode, one period
over the ``frames`` frames. phonopy's input with forces or force constants (``phonopy.yaml``
with ``FORCE_SETS``), which needs phonopy itself, is read by the ewald-phonopy plugin.
"""
from __future__ import annotations

from ..phonon import PhononModes, is_outcar_phonon, is_phonopy_yaml, read_modes
from .base import FrameReader


class PhononReader(FrameReader):
    """Phonon modes of ``path``; ``frames`` frames make one vibration period."""

    def __init__(self, path, frames: int = 24, structure=None):
        self.path = str(path)
        self.frames = int(frames)
        if self.frames < 1:
            raise ValueError("frames must be at least 1")
        self.modes: PhononModes = read_modes(self.path, structure=structure)
        self._table = self.modes.table()  # shared by every frame, so the Plots tab draws it once

    @classmethod
    def from_modes(cls, modes: PhononModes, frames: int = 24, path: str = "") -> PhononReader:
        """A reader over phonon modes made elsewhere (phonopy force constants, AiiDA, ...)."""
        reader = cls.__new__(cls)
        reader.path = path or modes.source or "phonon modes"
        reader.frames = int(frames)
        if reader.frames < 1:
            raise ValueError("frames must be at least 1")
        reader.modes = modes
        reader._table = modes.table()
        return reader

    def __len__(self) -> int:
        return self.frames

    def read(self, index: int):
        from ..data import DataCollection

        m = self.modes
        data = DataCollection(m.positions.copy(), m.types, m.cell.copy())
        data.particles["Mass"] = m.masses
        data.attributes["Phonon Modes"] = m
        data.attributes["Timestep"] = int(index)
        data.tables["Phonon Bands"] = dict(self._table)
        return data


class PhonopyYamlReader(PhononReader):
    """phonopy band.yaml/mesh.yaml/qpoints.yaml with eigenvectors: phonon modes (PhononMode animates one); frames= per period, structure= for qpoints.yaml"""


class VaspPhononReader(PhononReader):
    """VASP OUTCAR of an IBRION=5-8 run: Gamma phonon modes (PhononMode animates one)"""


__all__ = ["PhononReader", "PhonopyYamlReader", "VaspPhononReader", "is_outcar_phonon", "is_phonopy_yaml"]
