"""Example UI color themes for the desktop app: "Midnight" (dark, blue accents) and "High contrast"
(light, black text and borders). See docs/plugins/styles.md#ui-themes.

Copy this file into ewald's plugin folder (``ewald plugins`` prints where it is), then pick the
theme under Preferences > General > UI theme and restart, or from Python::

    from ewald import settings

    s = settings.load()
    s["general.ui_theme"] = "Midnight"
    s.save()
"""
from ewald.plugins import ui_theme


@ui_theme(name="Midnight", base="graphite")
def midnight():
    """Dark blue-gray panels with periwinkle accents."""
    return {
        "bg": "#11131C", "panel": "#171A26", "raised": "#1F2333", "border": "#2C3147",
        "accent": "#7AA2F7", "on_accent": "#0B1020", "accent_soft": "#1E2A4A",
        "data": "#FF9E64", "data_soft": "#3A2A22", "viewport": "#11131C",
    }


@ui_theme(name="High contrast", base="paper")
def high_contrast():
    """White panels, black text and borders, dark teal accents."""
    return {
        "bg": "#FFFFFF", "panel": "#FFFFFF", "border": "#000000", "text": "#000000",
        "secondary": "#000000", "muted": "#333333", "accent": "#005A50", "data": "#A23B00",
        "viewport": "#FFFFFF",
    }
