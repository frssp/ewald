"""The ``optimade:`` reader: ``load("optimade:mp/mp-149")``, ``load("optimade:cod?Si O")``."""
from __future__ import annotations

from ..io.base import FrameReader

from . import client
from . import providers as P
from .convert import to_data
from .query import to_filter
from .uri import make, parse


class OptimadeReader(FrameReader):
    """Structures of an OPTIMADE database: ``optimade:DB/ID`` is one frame,
    ``optimade:DB?QUERY`` one frame per structure found (up to ``limit``), with
    ``attributes["OPTIMADE"]`` (database, id, formula, ...). ``filter`` gives an OPTIMADE filter
    instead of a query. Read once, then kept in memory; read-only."""

    schemes = ("optimade",)
    description = ("OPTIMADE databases (optimade:mp/mp-149, optimade:cod/1000041, optimade:oqmd?Li-P-S): "
                   "Materials Project, COD, OQMD, Alexandria, NOMAD, Materials Cloud, JARVIS, ...")

    def __init__(self, path, filter: str | None = None, limit: int = 50):
        self.path = str(path)
        self.filter, self.limit = filter, limit
        self.uri = parse(self.path)
        self.provider = P.get(self.uri.db)
        url = self.provider.structures_url
        if self.uri.id and filter is None:
            resources = [client.structure(url, self.uri.id)]
        else:
            flt = filter if filter is not None else to_filter(self.uri.query)
            resources, self.total = client.structures(url, flt, limit)
            if not resources:
                raise ValueError(f"{self.path}: nothing in {self.provider.title} matches {flt!r}")
        self._resources = resources
        self._frames: dict = {}

    def __len__(self) -> int:
        return len(self._resources)

    def read(self, index: int):
        if index not in self._frames:
            r = self._resources[index]
            self._frames[index] = to_data(r, self.provider.name, make(self.provider.name, str(r.get("id"))),
                                          client.entry_url(self.provider.structures_url, str(r.get("id"))))
        return self._frames[index]
