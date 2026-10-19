"""Fast parser for whitespace-separated tables (LAMMPS dump and extxyz atom blocks).

``np.loadtxt`` reads about 2M lines/s on one core; this numba kernel splits the
block at newlines and converts each row in parallel. Numbers are converted with
Clinger's fast path (at most 15 significant digits and |exponent| <= 22), which is
exactly as rounded as ``strtod``; rows with anything else (17-digit values, inf/nan,
text) are flagged and re-read with ``np.loadtxt``, so the result never differs.
Text columns (extxyz species, logicals) are not converted; the kernel records where
each of their tokens starts and ends.
"""
from __future__ import annotations

import io

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a core dependency
    HAVE_NUMBA = False

_POW10 = np.array([10.0 ** k for k in range(23)])


def _parse_rows(buf, starts, ends, ncols, colmap, tokmap, pow10, out, tok_s, tok_e, bad):  # pragma: no cover - compiled
    for r in prange(len(starts)):
        p = starts[r]
        e = ends[r]
        for c in range(ncols):
            while p < e and (buf[p] == 32 or buf[p] == 9 or buf[p] == 13):
                p += 1
            if p >= e:
                bad[r] = True
                break
            t = tokmap[c]
            if t >= 0:  # text column: record the token, do not convert
                tok_s[r, t] = p
                while p < e and not (buf[p] == 32 or buf[p] == 9 or buf[p] == 13):
                    p += 1
                tok_e[r, t] = p
                continue
            neg = False
            if buf[p] == 45:
                neg = True
                p += 1
            elif buf[p] == 43:
                p += 1
            m = 0
            nd = 0  # significant digits
            exp10 = 0
            digits = 0
            while p < e and 48 <= buf[p] <= 57:
                if m > 0 or buf[p] != 48:
                    nd += 1
                if nd <= 18:
                    m = m * 10 + (buf[p] - 48)
                else:
                    exp10 += 1
                digits += 1
                p += 1
            if p < e and buf[p] == 46:
                p += 1
                while p < e and 48 <= buf[p] <= 57:
                    if m > 0 or buf[p] != 48:
                        nd += 1
                    if nd <= 18:
                        m = m * 10 + (buf[p] - 48)
                        exp10 -= 1
                    digits += 1
                    p += 1
            if digits == 0:
                bad[r] = True
                break
            if p < e and (buf[p] == 101 or buf[p] == 69):
                p += 1
                eneg = False
                if p < e and buf[p] == 45:
                    eneg = True
                    p += 1
                elif p < e and buf[p] == 43:
                    p += 1
                ev = 0
                edigits = 0
                while p < e and 48 <= buf[p] <= 57:
                    if ev < 10000:
                        ev = ev * 10 + (buf[p] - 48)
                    edigits += 1
                    p += 1
                if edigits == 0:
                    bad[r] = True
                    break
                exp10 += -ev if eneg else ev
            if p < e and not (buf[p] == 32 or buf[p] == 9 or buf[p] == 13):
                bad[r] = True  # not a plain number (text, inf, nan, ...)
                break
            if m == 0:
                v = 0.0
            elif nd > 15:
                bad[r] = True
                break
            elif exp10 == 0:
                v = float(m)
            elif 0 < exp10 <= 22:
                v = float(m) * pow10[exp10]
            elif -22 <= exp10 < 0:
                v = float(m) / pow10[-exp10]
            else:
                bad[r] = True
                break
            k = colmap[c]
            if k >= 0:
                out[r, k] = -v if neg else v


if HAVE_NUMBA:
    _parse_rows = njit(parallel=True, cache=True, nogil=True)(_parse_rows)


def _rows(block: bytes, n: int):
    buf = np.frombuffer(block, dtype=np.uint8)
    newlines = np.flatnonzero(buf == 10)
    starts = np.concatenate(([0], newlines + 1))
    ends = np.concatenate((newlines, [len(buf)]))
    keep = ends > starts  # drop empty lines (and the empty tail after a final newline)
    starts, ends = starts[keep], ends[keep]
    if len(starts) < n:
        raise ValueError(f"atom table has {len(starts)} rows, expected {n}")
    return buf, starts[:n], ends[:n]


def _run(buf, starts, ends, ncols, usecols, textcols):
    n = len(starts)
    colmap = np.full(ncols, -1, dtype=np.int64)
    colmap[list(usecols)] = np.arange(len(usecols))
    tokmap = np.full(ncols, -1, dtype=np.int64)
    tokmap[list(textcols)] = np.arange(len(textcols))
    out = np.empty((n, len(usecols)))
    tok_s = np.zeros((n, len(textcols)), dtype=np.int64)
    tok_e = np.zeros((n, len(textcols)), dtype=np.int64)
    bad = np.zeros(n, dtype=np.bool_)
    _parse_rows(buf, starts, ends, ncols, colmap, tokmap, _POW10, out, tok_s, tok_e, bad)
    return out, tok_s, tok_e, bad


def parse_table(block: bytes, n: int, ncols: int, usecols) -> np.ndarray:
    """First ``n`` rows of ``ncols`` numbers each -> float array of the ``usecols`` columns.

    Raises ValueError when the block has fewer than ``n`` rows or a row cannot be read.
    """
    usecols = list(usecols)
    if not HAVE_NUMBA or n == 0:
        return _loadtxt(block, n, usecols)
    buf, starts, ends = _rows(block, n)
    out, _, _, bad = _run(buf, starts, ends, ncols, usecols, [])
    if bad.any():
        rows = np.flatnonzero(bad)
        if len(rows) > n // 4:  # mostly full-precision numbers: one loadtxt pass is simpler
            return _loadtxt(block, n, usecols)
        lines = b"\n".join(block[starts[r]:ends[r]] for r in rows)
        out[rows] = _loadtxt(lines, len(rows), usecols)
    return out


def parse_mixed(block: bytes, n: int, ncols: int, usecols, textcols) -> tuple[np.ndarray, list[np.ndarray]]:
    """First ``n`` rows of ``ncols`` tokens -> (floats of ``usecols``, one bytes array per ``textcols``).

    Columns in neither list are converted and dropped. Rows the kernel cannot read exactly
    (too many digits, inf/nan) are split in Python with ``float()``. Raises ValueError for a
    short row or a numeric column that is not a number.
    """
    usecols, textcols = list(usecols), list(textcols)
    if n == 0:
        return np.empty((0, len(usecols))), [np.empty(0, dtype="S1") for _ in textcols]
    buf, starts, ends = _rows(block, n)
    if HAVE_NUMBA:
        out, tok_s, tok_e, bad = _run(buf, starts, ends, ncols, usecols, textcols)
    else:  # pragma: no cover - numba is a core dependency
        out, tok_s, tok_e = np.empty((n, len(usecols))), None, None
        bad = np.ones(n, dtype=bool)
    rows = np.flatnonzero(bad)
    if len(rows) == n:
        tok_s = tok_e = None  # every row goes through Python below
    text = [_tokens(buf, tok_s[:, j], tok_e[:, j]) if tok_s is not None else np.empty(n, dtype=object)
            for j in range(len(textcols))]
    if len(rows):  # object arrays, so a token from a re-read row is never cut to the kernel's width
        text = [t.astype(object) for t in text]
    for r in rows:
        fields = block[starts[r]:ends[r]].split()
        if len(fields) < ncols:
            raise ValueError(f"atom line {r + 1} has {len(fields)} values, expected {ncols}")
        out[r] = [float(fields[k]) for k in usecols]
        for j, k in enumerate(textcols):
            text[j][r] = fields[k]
    return out, [t if t.dtype.kind == "S" else t.astype(bytes) for t in text]


def _tokens(buf, s, e) -> np.ndarray:
    """Byte ranges [s, e) of ``buf`` -> fixed-width bytes array, without a Python loop."""
    lengths = e - s
    width = max(int(lengths.max()), 1)
    chars = np.zeros((len(s), width), dtype=np.uint8)
    for k in range(width):
        m = lengths > k
        chars[m, k] = buf[s[m] + k]
    return chars.view(f"S{width}").ravel().copy()


def _loadtxt(block, n, usecols):
    values = np.loadtxt(io.BytesIO(block), dtype=float, usecols=usecols, ndmin=2, comments=None,
                        max_rows=n)
    if values.shape != (n, len(usecols)):
        raise ValueError(f"atom table has {len(values)} rows, expected {n}")
    return values
