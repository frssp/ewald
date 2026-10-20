"""Example UI color themes for the desktop app: "Nord" (dark, frost-blue accents) and "Sepia"
(light, warm paper). See docs/plugins/styles.md#ui-themes.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is), then pick the
theme under Preferences > General > UI theme, or from Python::

    from ewald import settings

    s = settings.load()
    s["general.ui_theme"] = "Nord"
    s.save()
"""
from ewald.plugin_api import ui_theme


@ui_theme(name="Nord", base="graphite")
def nord():
    """Polar-night panels with frost-blue accents."""
    return {
        "bg": "#2E3440", "panel": "#323946", "raised": "#3B4252", "border": "#4C566A",
        "text": "#ECEFF4", "secondary": "#D8DEE9", "muted": "#A3ACBD", "accent": "#88C0D0",
        "on_accent": "#2E3440", "accent_soft": "#3E5563", "data": "#D08770", "data_soft": "#4A3F40",
        "viewport": "#2E3440",
    }


@ui_theme(name="Sepia", base="paper")
def sepia():
    """Warm paper panels, brown text, deep red accents."""
    return {
        "bg": "#FBF5EA", "panel": "#F3EBDC", "raised": "#FFFAF1", "border": "#D9CCB4",
        "text": "#2B2118", "secondary": "#4E4033", "muted": "#6B5B4B", "accent": "#9C3B2E",
        "accent_soft": "#F1DDD5", "data": "#2F6B8A", "data_soft": "#DCE8EE", "viewport": "#F3EBDC",
    }


__ewald__ = {"plugin_api": 1}   # plugin manifest: written for ewald.plugin_api version 1
