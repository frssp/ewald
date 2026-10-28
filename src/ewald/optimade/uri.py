"""``optimade:DB/ID`` (one structure) and ``optimade:DB?QUERY`` (the structures a search finds).

``DB`` is a short database name (:mod:`ewald.optimade.providers`); ``ID`` is the database's own
id and may hold slashes (``optimade:mp/mp-149``, ``optimade:cod/1000041``); ``QUERY`` is what the
search box takes (``optimade:cod?Si O``, ``optimade:mp?Li-P-S``, or an OPTIMADE filter).
``optimade:ID`` with no database uses the default one (Preferences › OPTIMADE). These are
addresses, not file paths, so the parts are split on "/" on every platform.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import providers as P


@dataclass(frozen=True)
class OptimadeURI:
    db: str
    id: str = ""          # "" with a query
    query: str = ""

    def __str__(self) -> str:
        return f"optimade:{self.db}" + (f"/{self.id}" if self.id else "") + (f"?{self.query}" if self.query else "")


def make(db: str, entry_id: str) -> str:
    return str(OptimadeURI(db, entry_id))


def parse(text: str, values: dict | None = None) -> OptimadeURI:
    if not text.lower().startswith("optimade:"):
        raise ValueError(f"not an optimade: path: {text!r}")
    rest = text[len("optimade:"):].lstrip("/")
    rest, sep, query = rest.partition("?")
    query = query.strip()
    head, _, tail = rest.partition("/")      # an address, never a local path
    known = P.providers(values)
    if head.lower() in known:
        db, entry = head.lower(), tail.strip("/")
    else:
        db, entry = P.get(None, values).name, rest.strip("/")
    if not entry and not sep:
        ex = known[db].example or "ID"
        raise ValueError(f"{text!r}: expected optimade:{db}/{ex} or optimade:{db}?QUERY")
    if entry and sep:
        raise ValueError(f"{text!r}: give an id or a query, not both")
    return OptimadeURI(db, entry, query)
