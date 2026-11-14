"""Frame indexes of big trajectory files, kept on disk between sessions.

Opening a LAMMPS dump, XDATCAR or extended XYZ file scans it once for where each frame starts
(about 4-8 s per GB from a warm OS cache). :func:`lookup` and :func:`store` keep that index in
the cache folder (next to the remote-file cache, ``EWALD_CACHE_DIR`` overrides it), one ``.npz``
per file and reader, so opening the same file again skips the scan. An entry counts only while
the file has the same size and modification time (nanoseconds) as when it was indexed; any change,
a run appending frames included, makes :func:`lookup` miss. A file that only grew (a running
simulation appended frames since) is not scanned from the start: :func:`lookup_prefix` gives the
kept index when the bytes it covered are still there (a digest of the first and the last
:data:`PROBE` bytes it covered), and the reader scans only from its last frame on. Files on
another machine (``host:/path``) are kept too, under their SFTP size and modification time
(whole seconds), so reopening a big remote dump skips the ``grep`` on the host. Only files of at
least :data:`MIN_BYTES` are kept (small ones scan in milliseconds), compressed ones without the
prefix check, and only complete scans (no partial last frame). ``EWALD_INDEX_CACHE=0`` turns it off.

The files hold integer arrays only and are read with ``allow_pickle=False``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

VERSION = 1
MIN_BYTES = 32 << 20
PROBE = 1 << 16   # bytes read at each end of the indexed part to tell that a grown file kept it


def enabled() -> bool:
    return os.environ.get("EWALD_INDEX_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def cache_dir() -> Path:
    from .remote import cache_dir as remote_cache

    return remote_cache().parent / "index"


def _name(path: str) -> str:
    """The file's name in the cache: a local path made absolute (case folded where the file
    system ignores case), a remote one as ``user@host:port/path``."""
    from .remote import parse_remote

    rp = parse_remote(path)
    if rp is not None:
        return f"remote\0{rp.user or ''}@{rp.host}:{rp.port or ''}\0{rp.path}"
    return os.path.normcase(os.path.abspath(path))


def _entry(path: str, kind: str) -> Path:
    name = _name(path)
    digest = hashlib.sha1(f"{kind}\0{name}".encode("utf-8")).hexdigest()
    return cache_dir() / f"{kind}-{digest[:24]}.npz"


def _stamp(path: str):
    """(size, mtime in ns) of a file worth caching, else None."""
    from . import remote

    if not enabled():
        return None
    try:
        if remote.is_remote(path):
            st = remote.stat(path)
            size, mtime = int(st.st_size or 0), int(st.st_mtime or 0) * 10**9
        else:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime_ns
    except Exception:  # noqa: BLE001 - missing, or a host that cannot be reached: scan as before
        return None
    if size < MIN_BYTES:
        return None
    return size, mtime


def _compressed(path: str) -> bool:
    from .base import _OPENERS

    return path.endswith(tuple(_OPENERS))


def _probe(path: str, size: int) -> np.ndarray | None:
    """Digest of the first and the last :data:`PROBE` of the first ``size`` bytes of ``path``."""
    import hashlib

    from .base import open_binary

    try:
        with open_binary(path) as fh:
            head = fh.read(min(PROBE, size))
            start = max(size - PROBE, 0)
            fh.seek(start)
            tail = fh.read(size - start)
    except (OSError, ValueError):
        return None
    if len(tail) != size - start:
        return None
    digest = hashlib.blake2b(head + b"\0" + tail, digest_size=16).digest()
    return np.frombuffer(digest, dtype=np.int64).copy()


def lookup(path, kind: str) -> dict | None:
    """The arrays :func:`store` kept for ``path`` and this reader ``kind``, if the file is unchanged."""
    path = str(path)
    stamp = _stamp(path)
    if stamp is None:
        return None
    kept = _read(path, kind)
    if kept is None or kept["_meta"].tolist() != [VERSION, *stamp]:
        return None
    return {k: v for k, v in kept.items() if not k.startswith("_")}


def lookup_prefix(path, kind: str) -> dict | None:
    """The index :func:`store` kept for ``path`` when the file has grown since and still starts
    with the bytes that index covered (frames appended, nothing rewritten): the reader scans on
    from its last frame. None otherwise, or when the entry was not kept with a probe."""
    path = str(path)
    stamp = _stamp(path)
    if stamp is None or _compressed(path):
        return None
    kept = _read(path, kind)
    if kept is None or "_probe" not in kept:
        return None
    version, size, _ = kept["_meta"].tolist()
    if version != VERSION or not size < stamp[0]:
        return None
    probe = _probe(path, size)
    if probe is None or not np.array_equal(probe, kept["_probe"]):
        return None
    return {k: v for k, v in kept.items() if not k.startswith("_")}


def _read(path: str, kind: str) -> dict | None:
    try:
        with np.load(_entry(path, kind), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # missing, half written by another process, or from an older EWALD
        return None


def store(path, kind: str, **arrays) -> bool:
    """Keep ``arrays`` (integer arrays) as the index of ``path``; False when not kept."""
    path = str(path)
    stamp = _stamp(path)
    if stamp is None:
        return False
    target = _entry(path, kind)
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp.npz")
    extra = {}
    if not _compressed(path):
        probe = _probe(path, stamp[0])
        if probe is not None:
            extra["_probe"] = probe
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            np.savez(fh, _meta=np.array([VERSION, *stamp], dtype=np.int64), **extra,
                     **{k: np.asarray(v, dtype=np.int64) for k, v in arrays.items()})
        from ..fileops import replace

        replace(tmp, target, delays=())   # another EWALD reading it: keep the scan, do not wait
        return True
    except Exception:  # a read-only or full cache folder: the next open scans again
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def clear() -> None:
    import shutil

    shutil.rmtree(cache_dir(), ignore_errors=True)
