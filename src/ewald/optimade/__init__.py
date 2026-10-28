"""Search and open structures of OPTIMADE databases (no account or key needed).

```python
from ewald import load
from ewald.optimade import search

pipe = load("optimade:mp/mp-149")          # Materials Project, Si
pipe = load("optimade:cod/1000041")        # Crystallography Open Database
pipe = load("optimade:oqmd?Li-P-S")        # every Li-P-S structure found (one frame each)
search("LiFePO4", db="mp")                 # hits: uri, formula, nsites
search('elements HAS ALL "Si","O" AND nsites<=6', db="cod")
```

Databases go by short names (:data:`providers.BUILTIN`: mp, cod, tcod, oqmd, alexandria, nomad,
mc3d, mc2d, jarvis, odbx); more under Preferences › OPTIMADE. Every frame carries
``attributes["OPTIMADE"]`` (database, id, formula, address). The OPTIMADE side bar searches and
opens; the AI gets a read-only ``optimade_search`` tool. Built in since 2026-10-28 (it was the
``ewald-optimade`` plugin); the built-in plugin group "optimade" switches all of it off.
"""
from .providers import BUILTIN, Provider  # noqa: E402
from .providers import providers as databases  # noqa: E402
from .query import to_filter  # noqa: E402
from .search import Hit, search  # noqa: E402
from .uri import OptimadeURI, parse  # noqa: E402

__all__ = ["BUILTIN", "Hit", "OptimadeURI", "Provider", "databases", "parse", "search", "to_filter"]


def register(origin: str = "built-in") -> dict:
    """Put the AI tool, the settings page and the side bar into the plugin registries (the
    reader is in :func:`ewald.io.builtin_readers`); returns what was added, by kind."""
    from .. import plugins
    from .panel import optimade
    from .providers import OptimadeSettings
    from .tools import DESCRIPTION, NAME, SCHEMA, optimade_search

    plugins.register_tool(NAME, optimade_search, DESCRIPTION, SCHEMA, origin=origin)
    plugins.register_settings_page("optimade", OptimadeSettings, origin=origin)
    plugins.register_panel("optimade", optimade, origin=origin)
    return {"tools": [NAME], "settings_pages": ["optimade"], "panels": ["optimade"]}
