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


def grep_lines(path, pattern: str, literals=(), chunk_size: int = 1 << 24) -> list[tuple[int, bytes]]:
    """Byte offset and text (without the newline) of every line matching ``pattern``.

    ``pattern`` must be valid both as a Python regex and as a POSIX extended regex
    (use ``[0-9]``, not ``\\d``): a remote file is searched by ``grep -b`` on the host,
    so only the matching lines cross the network. Locally the file is read in large
    chunks and searched with ``re``. ``literals``, byte strings at least one of which every
    matching line contains, speed that up several times: lines are found with ``bytes.find``
    and only those are tried against ``pattern``. Compressed files are decompressed on the fly.
    """
    import io
    import re

    hits = remote.line_offsets(path, pattern) if remote.is_remote(path) else None
    if hits is not None:
        return [(off, text.rstrip(b"\r")) for off, text in hits]
    regex = re.compile(pattern.encode(), re.M)
    out: list[tuple[int, bytes]] = []
    pos, carry = 0, b""
    with open_binary(path) as fh:
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
