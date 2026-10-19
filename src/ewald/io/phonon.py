"""Phonon mode files as a looping trajectory: phonopy band.yaml / mesh.yaml / qpoints.yaml
(with eigenvectors) and VASP OUTCARs of IBRION=5-8 runs.

Every frame is the primitive cell with the modes in ``data.attributes["Phonon Modes"]`` (a
:class:`ewald.phonon.PhononModes`) and the frequencies in the table "Phonon Bands"; the
:class:`ewald.modifiers.PhononMode` modifier displaces the atoms along one mode, one period
over the ``frames`` frames.
"""
from __future__ import annotations

from ..phonon import PhononModes, read_modes
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
