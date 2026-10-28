"""``search("LiFePO4", db="mp")``: what an OPTIMADE database has, as a short list to pick from."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import client
from . import providers as P
from .query import to_filter
from .uri import make


@dataclass
class Hit:
    db: str
    id: str
    formula: str = ""
    nsites: int | None = None
    elements: list = field(default_factory=list)
    last_modified: str = ""

    @property
    def uri(self) -> str:
        """The path that opens it: ``load(hit.uri)``."""
        return make(self.db, self.id)

    def as_dict(self) -> dict:
        return {"uri": self.uri, "id": self.id, "formula": self.formula, "nsites": self.nsites,
                "elements": self.elements}


class Results(list):
    """A list of :class:`Hit` with ``total`` (how many the server says match, None: unknown)
    and the ``filter`` that was sent."""

    def __init__(self, hits=(), total=None, filter="", db=""):
        super().__init__(hits)
        self.total, self.filter, self.db = total, filter, db

    def __repr__(self) -> str:
        head = f"{len(self)} of {self.total if self.total is not None else '?'} in {self.db} ({self.filter})"
        rows = [f"  {h.uri:<32} {h.formula:<16} {h.nsites if h.nsites is not None else '':>5}" for h in self]
        return "\n".join([head, *rows])


def hit(db: str, resource: dict) -> Hit:
    a = resource.get("attributes") or {}
    return Hit(db, str(resource.get("id")),
               a.get("chemical_formula_reduced") or a.get("chemical_formula_descriptive") or "",
               a.get("nsites"), list(a.get("elements") or []), str(a.get("last_modified") or ""))


def search(query: str = "", db: str | None = None, limit: int | None = None, filter: str | None = None,
           values: dict | None = None) -> Results:
    """Structures of database ``db`` (default: Preferences › OPTIMADE) matching ``query``, a
    formula (``LiFePO4``), a chemical system (``Li-P-S``), elements it must contain (``Li P S``)
    or an OPTIMADE filter; ``filter`` passes a filter as it is. Open one with ``load(hit.uri)``."""
    provider = P.get(db, values)
    flt = filter if filter is not None else to_filter(query)
    limit = limit or P.page_size(values)
    entries, total = client.structures(provider.structures_url, flt, limit, client.SEARCH_FIELDS)
    return Results([hit(provider.name, r) for r in entries], total, flt, provider.name)
