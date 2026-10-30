"""Plain text tables: columns of numbers with an optional header (whitespace or comma separated).

Comment lines start with ``#``, ``@`` or ``!``; the column names are the last comment line with as
many words as the data has columns (``#! FIELDS a b`` and ``# a b`` both name ``a``, ``b``), or a
first line that is not numbers (``time,energy``). Used by :func:`ewald.load_table` for files that
hold no structure: CV and energy logs, analysis output, a table of runs.
"""
from __future__ import annotations

import re

import numpy as np

from .base import open_binary

_COMMENT = ("#", "@", "!")


def _number(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def _numeric(tokens) -> bool:
    try:
        [float(t) for t in tokens]
        return True
    except ValueError:
        return False


def _split(line: str, comma: bool) -> list[str]:
    return [t.strip() for t in line.split(",")] if comma else line.split()


def read_text_table(path) -> dict:
    """``{column: array}`` from a text table; rows with another number of fields are skipped.
    Raises ValueError when no row of numbers is found."""
    with open_binary(path) as fh:
        text = fh.read().decode("utf-8", errors="replace")
    comments, rows, header = [], [], None
    comma = None
    width = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_COMMENT):
            if not rows:
                comments.append(line)
            continue
        if comma is None:
            comma = "," in line
        tokens = _split(line, comma)
        if not rows and header is None and not _numeric(tokens):
            header = tokens            # 'time,energy' or 'Step Temp PotEng'
            continue
        if width is None:
            width = len(tokens)
        if len(tokens) == width:
            rows.append([_number(t) for t in tokens])
    if not rows:
        raise ValueError(f"no rows of numbers in {path}")
    if header is None or len(header) != width:
        header = None
        for line in reversed(comments):
            words = re.sub(r"^[#@!]+", "", line).split()
            if words[:1] == ["FIELDS"]:
                words = words[1:]
            if len(words) == width:
                header = words
                break
    names = header or [f"column {i + 1}" for i in range(width)]
    names = [n or f"column {i + 1}" for i, n in enumerate(names)]
    names = [n if names.index(n) == i else f"{n} {i + 1}" for i, n in enumerate(names)]   # repeated names
    data = np.asarray(rows, float)
    return {name: data[:, i] for i, name in enumerate(names)}


__all__ = ["read_text_table"]
