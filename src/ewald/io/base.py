"""Lazy, random-access trajectory readers."""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
import re

from ..data import DataCollection
from . import remote

_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


def natural_key(path: str):
    """Sort key putting numbers in numeric order: ``dump.200`` before ``dump.1000``."""
    return [(1, int(t), t) if t.isdigit() else (0, 0, t) for t in re.split(r"(\d+)", str(path))]


class IncompleteFileWarning(UserWarning):
    """The file belongs to a calculation that is still running or was stopped: everything
    complete was read, the part being written was left out (see :attr:`FrameReader.incomplete`)."""


class IncompleteFileError(ValueError):
    """The file belongs to a calculation that is still running or was stopped, and nothing in it
    is complete yet (no finished step). Try again later, or open another output of the run."""


class FrameReader:
    """Interface: index a file once, then load any frame on demand."""

    #: why the file is incomplete (a calculation still running or stopped), or None when it is
    #: complete or that cannot be told; set by the readers, shown by the GUI
    incomplete: str | None = None

    def _set_incomplete(self, note: str | None, warn: bool = True, stacklevel: int = 3):
        """Record (and by default warn about, as :class:`IncompleteFileWarning`) an incomplete file."""
        self.incomplete = note
        if note and warn:
            import warnings

            warnings.warn(f"{getattr(self, 'path', '')}: calculation incomplete: {note}", IncompleteFileWarning,
                          stacklevel=stacklevel)

    def __len__(self) -> int:  # number of frames
        raise NotImplementedError

    def read(self, index: int) -> DataCollection:
        raise NotImplementedError

    def refresh(self) -> int | None:
        """Pick up frames appended since the file was indexed (a run still writing it).

        Returns the first frame whose data may have changed (the old last frame, which
        may have been re-read, or the first new one), or None when nothing changed.
        Readers that cannot follow a growing file keep the default: None.
        """
        return None

    def run_starts(self) -> list[int]:
        """Frames where the step goes back inside this one file: a restarted simulation that
        appended to it, or a reset step counter. :class:`~ewald.io.FileSource` splits the file
        there and joins the parts like separate files. Empty when the format cannot tell cheaply."""
        return []

    def timestep(self, index: int) -> int | None:
        """The simulation step of frame ``index`` without reading the whole frame, or None when
        the format has no step that counts across restarts (VASP numbers each run from 1).
        :class:`~ewald.io.FileSource` joins restart segments with it."""
        return None


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


def file_size(path) -> int | None:
    """Current size in bytes of a local or remote file; None for compressed files, whose
    size says nothing about how much has been written, and for missing ones."""
    path = str(path)
    if path.endswith(tuple(_OPENERS)):
        return None
    try:
        return remote.stat(path).st_size if remote.is_remote(path) else os.path.getsize(path)
    except OSError:
        return None


def complete_size(path, size: int, chunk: int = 1 << 16) -> int:
    """Offset just past the last newline within the first ``size`` bytes: a file being written
    may end in half a line, and everything up to here holds whole lines only."""
    with open_binary(path) as fh:
        end = size
        while end > 0:
            start = max(0, end - chunk)
            fh.seek(start)
            j = fh.read(end - start).rfind(b"\n")
            if j >= 0:
                return start + j + 1
            end = start
    return 0


def head_lines(path, n: int) -> list[bytes]:
    """First ``n`` lines of a (possibly compressed) file; empty list if unreadable."""
    try:
        with open_binary(path) as fh:
            return [line for _, line in zip(range(n), fh)]
    except (OSError, EOFError, lzma.LZMAError):
        return []


def grep_lines(path, pattern: str, literals=(), chunk_size: int = 1 << 24,
               start: int = 0) -> list[tuple[int, bytes]]:
    """Byte offset and text (without the newline) of every line matching ``pattern``.

    ``pattern`` must be valid both as a Python regex and as a POSIX extended regex
    (use ``[0-9]``, not ``\\d``): a remote file is searched by ``grep -b`` on the host,
    so only the matching lines cross the network. Locally the file is read in large
    chunks and searched with ``re``. ``literals``, byte strings at least one of which every
    matching line contains, speed that up several times: lines are found with ``bytes.find``
    and only those are tried against ``pattern``. Compressed files are decompressed on the fly.
    ``start`` (a line start) skips the bytes before it, for following a growing file.
    """
    import io
    import re

    hits = remote.line_offsets(path, pattern, start) if remote.is_remote(path) else None
    if hits is not None:
        return [(off, text.rstrip(b"\r")) for off, text in hits]
    regex = re.compile(pattern.encode(), re.M)
    out: list[tuple[int, bytes]] = []
    pos, carry = start, b""
    with open_binary(path) as fh:
        if start:
            fh.seek(start)
        read = fh.read if isinstance(fh, io.BufferedReader) else getattr(fh, "read1", fh.read)
        while True:
            try:
                chunk = read(chunk_size)
            except EOFError:  # truncated compressed stream: keep what decompressed
                chunk = b""
            buf = carry + chunk
            if chunk:
                end = buf.rfind(b"\n") + 1
                if end == 0:
                    carry = buf
                    continue
            else:
                end = len(buf)
            seg, carry = buf[:end], buf[end:]
            if literals:
                starts = set()
                for lit in literals:
                    j = seg.find(lit)
                    while j >= 0:
                        starts.add(seg.rfind(b"\n", 0, j) + 1)
                        j = seg.find(lit, j + len(lit))
                for start in sorted(starts):
                    stop = seg.find(b"\n", start)
                    line = seg[start:stop if stop >= 0 else end].rstrip(b"\r")
                    if regex.search(line):
                        out.append((pos + start, line))
            else:
                last = -1
                for m in regex.finditer(seg):
                    start = seg.rfind(b"\n", 0, m.start()) + 1
                    if start == last:
                        continue
                    last = start
                    stop = seg.find(b"\n", m.start())
                    out.append((pos + start, seg[start:stop if stop >= 0 else end].rstrip(b"\r")))
            pos += end
            if not chunk:
                break
    return out


_FLOAT = None


def floats(line: bytes) -> list[float]:
    """Every number in a line of fixed-format Fortran output, also when columns run together
    (``-10.123456789-0.123``); ``*****`` overflow fields are skipped."""
    global _FLOAT
    if _FLOAT is None:
        import re

        _FLOAT = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")
    return [float(x.replace(b"D", b"E").replace(b"d", b"e")) for x in _FLOAT.findall(line)]
