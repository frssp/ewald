"""The OPTIMADE databases EWALD knows by a short name, and the settings page that adds more.

A database is reached at its *base URL*; EWALD appends ``/v1/structures``. The table holds
public databases that need no account. More come from the settings page (Preferences ›
OPTIMADE › Extra databases: one ``name = base URL`` per line), so a lab's own server or a new
provider of https://www.optimade.org/providers-dashboard works without a new release.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..plugin_kinds import SettingsField


@dataclass(frozen=True)
class Provider:
    name: str        # short name used in optimade:NAME/ID
    title: str
    base_url: str    # without /v1
    example: str = ""   # an id that exists there, for help texts

    @property
    def structures_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/structures"


#: Public OPTIMADE databases (from the OPTIMADE providers list, 2026-10).
BUILTIN = {p.name: p for p in (
    Provider("mp", "Materials Project", "https://optimade.materialsproject.org", "mp-149"),
    Provider("cod", "Crystallography Open Database", "https://www.crystallography.net/cod/optimade", "1000041"),
    Provider("tcod", "Theoretical Crystallography Open Database", "https://www.crystallography.net/tcod/optimade"),
    Provider("oqmd", "Open Quantum Materials Database", "https://oqmd.org/optimade"),
    Provider("alexandria", "Alexandria (PBE)", "https://alexandria.icams.rub.de/pbe"),
    Provider("alexandria-pbesol", "Alexandria (PBEsol)", "https://alexandria.icams.rub.de/pbesol"),
    Provider("nomad", "NOMAD", "https://nomad-lab.eu/prod/v1/optimade"),
    Provider("mc3d", "Materials Cloud MC3D", "https://aiida.materialscloud.org/mc3d/optimade"),
    Provider("mc2d", "Materials Cloud MC2D", "https://aiida.materialscloud.org/mc2d/optimade"),
    Provider("jarvis", "JARVIS-DFT", "https://jarvis.nist.gov/optimade/jarvisdft"),
    Provider("odbx", "open database of xtals", "https://optimade.odbx.science"),
)}

DEFAULT = "mp"
_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


class OptimadeSettings:
    """Which database the search side bar starts with, and databases beyond the built-in ones."""

    title = "OPTIMADE"
    fields = (
        SettingsField("default", "Default database", placeholder=DEFAULT,
                      help="Short name used by search() and the side bar: " + ", ".join(BUILTIN)),
        SettingsField("page_size", "Results per search", kind="int", default=25),
        SettingsField("extra", "Extra databases", kind="multiline", advanced=True,
                      placeholder="mylab = https://optimade.mylab.org",
                      help="One 'name = base URL' per line (the URL without /v1); "
                           "the name is used as optimade:name/ID"),
    )


def settings() -> dict:
    try:
        from ..plugins import plugin_settings

        return plugin_settings("optimade")
    except KeyError:      # the built-in OPTIMADE group is switched off
        return {}


def parse_extra(text: str) -> dict:
    """``name = URL`` lines (``#`` comments) -> providers; bad lines raise ValueError."""
    out = {}
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, url = (s.strip() for s in line.partition("="))
        name = name.lower()
        if not sep or not _NAME.match(name) or not url.lower().startswith(("https://", "http://")):
            raise ValueError(f"OPTIMADE extra databases, line {n}: expected 'name = https://...', got {line!r}")
        out[name] = Provider(name, name, url.rstrip("/").removesuffix("/v1"))
    return out


def providers(values: dict | None = None) -> dict:
    """Every known database by short name: the built-in table, then the settings' extras
    (which may replace a built-in one, e.g. a mirror)."""
    values = settings() if values is None else values
    return {**BUILTIN, **parse_extra(values.get("extra") or "")}


def get(name: str | None, values: dict | None = None) -> Provider:
    values = settings() if values is None else values
    name = (name or values.get("default") or DEFAULT).strip().lower()
    known = providers(values)
    if name not in known:
        raise KeyError(f"unknown OPTIMADE database {name!r}; known: {', '.join(known)} "
                       "(add more under Preferences › OPTIMADE › Extra databases)")
    return known[name]


def page_size(values: dict | None = None) -> int:
    values = settings() if values is None else values
    return max(1, min(int(values.get("page_size") or 25), 500))
