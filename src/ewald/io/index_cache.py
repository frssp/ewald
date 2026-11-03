"""Frame indexes of big trajectory files, kept on disk between sessions.

Opening a LAMMPS dump, XDATCAR or extended XYZ file scans it once for where each frame starts
(about 4-8 s per GB from a warm OS cache). :func:`lookup` and :func:`store` keep that index in
the cache folder (next to the remote-file cache, ``EWALD_CACHE_DIR`` overrides it), one ``.npz``
per file and reader, so opening the same file again skips the scan. An entry counts only while
the file has the same size and modification time (nanoseconds) as when it was indexed; any change,
a run appending frames included, makes the reader scan again and replace the entry. Only local
files of at least :data:`MIN_BYTES` are kept (small ones scan in milliseconds) and only complete
scans (no partial last frame). ``EWALD_INDEX_CACHE=0`` turns it off.

The files hold integer arrays only and are read with ``allow_pickle=False``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

VERSION = 1
MIN_BYTES = 32 << 20


def enabled() -> bool:
    return os.environ.get("EWALD_INDEX_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def cache_dir() -> Path:
    from .remote import cache_dir as remote_cache

    return remote_cache().parent / "index"


def _entry(path: str, kind: str) -> Path:
    name = os.path.normcase(os.path.abspath(path))
    digest = hashlib.sha1(f"{kind}\0{name}".encode("utf-8")).hexdigest()
    return cache_dir() / f"{kind}-{digest[:24]}.npz"


def _stamp(path: str):
    """(size, mtime in ns) of a local file worth caching, else None."""
    from .remote import is_remote

    if not enabled() or is_remote(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size < MIN_BYTES:
        return None
    return st.st_size, st.st_mtime_ns


def lookup(path, kind: str) -> dict | None:
    """The arrays :func:`store` kept for ``path`` and this reader ``kind``, if the file is unchanged."""
    path = str(path)
    stamp = _stamp(path)
    if stamp is None:
        return None
    try:
        with np.load(_entry(path, kind), allow_pickle=False) as z:
            meta = z["_meta"]
            if meta.tolist() != [VERSION, *stamp]:
                return None
            return {k: z[k] for k in z.files if k != "_meta"}
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
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            np.savez(fh, _meta=np.array([VERSION, *stamp], dtype=np.int64),
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
