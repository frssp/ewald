"""Lazy, random-access trajectory readers."""
from __future__ import annotations

import bz2
import gzip
import lzma

from ..data import DataCollection
from . import remote

_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


class FrameReader:
    """Interface: index a file once, then load any frame on demand."""

    def __len__(self) -> int:  # number of frames
        raise NotImplementedError

    def read(self, index: int) -> DataCollection:
        raise NotImplementedError


def open_binary(path):
    """Open ``path`` for binary reading, decompressing ``.gz``/``.bz2``/``.xz`` transparently.

    Compressed streams support ``seek`` by re-reading from the start, so random
    access into a compressed trajectory costs O(offset); fine for typical sizes.
    Remote paths (``host:/path``, ``ssh://...``) are read in place over SFTP.
    """
    path = str(path)
    if remote.is_remote(path):
        return _open_remote(path)
    for ext, opener in _OPENERS.items():
        if path.endswith(ext):
            return opener(path, "rb")
    return open(path, "rb")


def _open_remote(path):
    raw = remote.open_remote(path)
    for ext, opener in _OPENERS.items():
        if path.endswith(ext):
            fh = opener(raw, "rb")
            close = fh.close

            def close_both():  # the decompressor does not close a file object it was given
                try:
                    close()
                finally:
                    raw.close()

            fh.close = close_both
            return fh
    return raw


def strip_compression(path) -> str:
    """File name without a trailing compression suffix (for name-based format sniffing)."""
    path = str(path)
    for ext in _OPENERS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def head_lines(path, n: int) -> list[bytes]:
    """First ``n`` lines of a (possibly compressed) file; empty list if unreadable."""
    try:
        with open_binary(path) as fh:
            return [line for _, line in zip(range(n), fh)]
    except (OSError, EOFError, lzma.LZMAError):
        return []


def _scan_lines(path, predicate):
    """Return byte offsets of lines for which ``predicate(line_bytes)`` is true."""
    offsets = []
    pos = 0
    with open_binary(path) as fh:
        for line in fh:
            if predicate(line):
                offsets.append(pos)
            pos += len(line)
    return offsets
