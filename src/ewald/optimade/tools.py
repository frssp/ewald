"""The AI's read-only search tool (Ask AI, ``ewald ask``, the MCP server).

It only lists what a database has. To open a structure the AI writes a notebook cell
``pipe = load("optimade:mp/mp-149")`` like any other change (EWALD's rule that the AI changes
things only through cells). It reaches only the databases the person set up (built-in table
plus Preferences › OPTIMADE), never an address the model makes up.
"""
from __future__ import annotations

from . import providers as P
from .search import search


NAME = "optimade_search"
DESCRIPTION = (
    "Search a public crystal-structure database through OPTIMADE (read-only; no key). db: one of the "
    "known short names (mp = Materials Project, cod, oqmd, alexandria, nomad, mc3d, jarvis, ... ; "
    "default from the user's settings). query: a formula ('LiFePO4'), a chemical system exactly "
    "('Li-P-S'), elements it must contain ('Li P S'), or an OPTIMADE filter ('elements HAS ALL \"Si\",\"O\" "
    "AND nsites<=12'). Returns uri, id, formula, nsites per hit and how many match. To open one, write and "
    "run a notebook cell: pipe = load(\"<uri>\") (several: load(\"optimade:<db>?<query>\") gives one frame "
    "per structure).")
SCHEMA = {"type": "object", "properties": {
        "query": {"type": "string"},
        "db": {"type": "string", "description": "database short name"},
        "limit": {"type": "integer", "description": "at most this many hits (default 20, max 100)"},
        "list_databases": {"type": "boolean", "description": "only list the known databases"}}}


def optimade_search(toolset, query="", db=None, limit=20, list_databases=False):
    """The ``optimade_search`` tool (registered by :func:`ewald.optimade.register`)."""
    known = P.providers()
    if list_databases:
        return {"databases": {p.name: {"title": p.title, "example_id": p.example or None} for p in known.values()},
                "default": P.get(None).name}
    res = search(query, db=db, limit=max(1, min(int(limit or 20), 100)))
    return {"db": res.db, "filter": res.filter, "total": res.total, "hits": [h.as_dict() for h in res]}
