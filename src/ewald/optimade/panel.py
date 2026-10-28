"""The OPTIMADE side bar's factory; the Qt code (:mod:`ewald.gui.optimade_panel`) is imported
when the side bar first opens."""
from __future__ import annotations

#: a crystal (cube in perspective) with a magnifier, drawn with currentColor for the UI theme
ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 7.5 9 4l6 3.5v7L9 18l-6-3.5z"/><path d="M3 7.5 9 11l6-3.5M9 11v7"/>'
        '<circle cx="17" cy="16" r="3"/><path d="m19.2 18.2 2.3 2.3"/></svg>')


def optimade(context):
    """OPTIMADE databases: search and open structures (Materials Project, COD, OQMD, ...)."""
    from ..gui.optimade_panel import OptimadePanel

    return OptimadePanel(context)


optimade.title = "OPTIMADE"
optimade.icon = ICON
optimade.tooltip = "OPTIMADE: search structure databases (Materials Project, COD, OQMD, ...)"
