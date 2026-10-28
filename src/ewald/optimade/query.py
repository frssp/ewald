"""What a person types into a search box, as an OPTIMADE filter.

``LiFePO4``, ``Ca(OH)2``  a formula: ``chemical_formula_reduced="CaH2O2"``
``Li-P-S``               a chemical system, exactly those elements
``Li P S``, ``Li,P,S``   structures that contain all of them (and maybe more)
anything with ``=``, ``<``, ``>`` or an OPTIMADE keyword (``HAS``, ``AND``, ...) is passed on as
a filter, e.g. ``elements HAS ALL "Si","O" AND nsites<=12``.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ase.data import chemical_symbols

_SYMBOLS = frozenset(chemical_symbols[1:])
_KEYWORDS = re.compile(r"[=<>!]|\b(HAS|AND|OR|NOT|CONTAINS|STARTS|ENDS|KNOWN|LENGTH)\b")
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\))(\d*)")


def _quoted(symbols) -> str:
    return ",".join(f'"{s}"' for s in symbols)


def _check(symbols):
    bad = [s for s in symbols if s not in _SYMBOLS]
    if bad:
        raise ValueError(f"not element symbols: {', '.join(bad)}")


def parse_formula(text: str) -> Counter:
    """``Ca(OH)2`` -> Counter(Ca=1, O=2, H=2); whole numbers only."""
    stack = [Counter()]
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise ValueError(f"not a formula: {text!r}")
        pos = m.end()
        if m.group(1):
            n = m.group(2) or "1"
            if not n.isdigit():
                raise ValueError(f"{text!r}: only whole numbers in a formula")
            stack[-1][m.group(1)] += int(n)
        elif m.group(3):
            stack.append(Counter())
        else:
            if len(stack) == 1:
                raise ValueError(f"unbalanced ')' in {text!r}")
            group, n = stack.pop(), int(m.group(5) or 1)
            for k, v in group.items():
                stack[-1][k] += v * n
    if pos != len(text) or len(stack) != 1 or not stack[0]:
        raise ValueError(f"not a formula: {text!r}")
    _check(stack[0])
    return stack[0]


def reduced_formula(counts) -> str:
    """OPTIMADE ``chemical_formula_reduced``: elements in alphabetical order, numbers divided by
    their greatest common divisor, 1 left out."""
    g = 0
    for v in counts.values():
        g = math.gcd(g, int(v))
    return "".join(f"{s}{'' if counts[s] // g == 1 else counts[s] // g}" for s in sorted(counts))


def to_filter(text: str) -> str:
    """The OPTIMADE filter for a search box text (see the module docstring); "" for all."""
    text = (text or "").strip()
    if not text:
        return ""
    if _KEYWORDS.search(text) or '"' in text:
        return text
    if "-" in text and " " not in text:
        symbols = [s for s in text.split("-") if s]
        _check(symbols)
        return f"elements HAS ALL {_quoted(sorted(set(symbols)))} AND nelements={len(set(symbols))}"
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    if len(parts) > 1:
        _check(parts)
        return f"elements HAS ALL {_quoted(sorted(set(parts)))}"
    return f'chemical_formula_reduced="{reduced_formula(parse_formula(text))}"'
