"""Lazy, random-access trajectory readers."""
from __future__ import annotations

from ..data import DataCollection


class FrameReader:
    """Interface: index a file once, then load any frame on demand."""

    def __len__(self) -> int:  # number of frames
        raise NotImplementedError

    def read(self, index: int) -> DataCollection:
        raise NotImplementedError


def _scan_lines(path, predicate):
    """Return byte offsets of lines for which ``predicate(line_bytes)`` is true."""
    offsets = []
    pos = 0
    with open(path, "rb") as fh:
        for line in fh:
            if predicate(line):
                offsets.append(pos)
            pos += len(line)
    return offsets
