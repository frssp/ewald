"""Design tokens of the UI themes: the built-in palettes and :func:`derive`.

Qt-free data shared by the desktop app (:mod:`ewald.gui.theme` styles Qt with it), the plugin
registry (``@ui_theme``) and plot styles; the core never imports :mod:`ewald.gui` for it.
"""
from __future__ import annotations

PALETTES = {
    "paper": {
        "bg": "#FBFBF9", "panel": "#F3F4F1", "raised": "#FFFFFF", "border": "#D5D8D1",
        "text": "#15181C", "secondary": "#3D434A", "muted": "#5C636B", "accent": "#1A66CC",
        "on_accent": "#FFFFFF", "accent_soft": "#E1EBF8", "data": "#C8551F", "data_soft": "#F7E4D9",
        "selection": "#C99A06", "pick": "#C8551F", "error": "#C62828", "ok": "#2E7D32",
        "viewport": "#ECEDE9", "hover": "#E9EBE6", "pressed": "#DFE2DC", "accent_hover": "#195AB2",
        "accent_pressed": "#19509B",
    },
    "graphite": {
        "bg": "#14161A", "panel": "#1B1E23", "raised": "#23272E", "border": "#30353D",
        "text": "#E7E9EC", "secondary": "#AEB4BE", "muted": "#8C939E", "accent": "#5CA0F2",
        "on_accent": "#0B1220", "accent_soft": "#25374E", "data": "#F08A4B", "data_soft": "#3A2A20",
        "selection": "#E8C547", "pick": "#FFD84D", "error": "#FF6B6B", "ok": "#5BD08A",
        "viewport": "#14161A", "hover": "#2A2F37", "pressed": "#333943", "accent_hover": "#74AEF4",
        "accent_pressed": "#4F87CB",
    },
    "midnight": {
        "bg": "#11131C", "panel": "#171A26", "raised": "#1F2333", "border": "#2E3449", "text": "#E6E9F2",
        "secondary": "#B3BAD0", "muted": "#8D96B0", "accent": "#7AA2F7", "on_accent": "#0B1020",
        "accent_soft": "#22305A", "data": "#FF9E64", "data_soft": "#3A2A22", "selection": "#E0AF68",
        "pick": "#FFD84D", "error": "#F7768E", "ok": "#9ECE6A", "viewport": "#11131C", "hover": "#272C40",
        "pressed": "#30364D", "accent_hover": "#97B7F9", "accent_pressed": "#5F8AE8",
    },
    "solarized light": {
        "bg": "#FDF6E3", "panel": "#F3ECD7", "raised": "#FFFBF0", "border": "#D8D0B8", "text": "#073642",
        "secondary": "#3F555C", "muted": "#5B6E73", "accent": "#1D6FA8", "on_accent": "#FFFFFF",
        "accent_soft": "#DCE6E3", "data": "#B3410F", "data_soft": "#F5DFCB", "selection": "#9A7400",
        "pick": "#B3410F", "error": "#C42B28", "ok": "#5F6E00", "viewport": "#EEE8D5", "hover": "#EAE3CC",
        "pressed": "#E0D8BF", "accent_hover": "#195F90", "accent_pressed": "#154F78",
    },
    "solarized dark": {
        "bg": "#002B36", "panel": "#073642", "raised": "#0C404D", "border": "#1D505C", "text": "#EEE8D5",
        "secondary": "#A9B6B6", "muted": "#8A9C9E", "accent": "#4BA3E3", "on_accent": "#002B36",
        "accent_soft": "#0B4255", "data": "#E8733A", "data_soft": "#3A3424", "selection": "#D9A92B",
        "pick": "#F2C744", "error": "#F0605B", "ok": "#A3B82A", "viewport": "#002B36", "hover": "#114755",
        "pressed": "#185160", "accent_hover": "#6CB5EA", "accent_pressed": "#3489C8",
    },
    "high contrast": {
        "bg": "#FFFFFF", "panel": "#FFFFFF", "raised": "#FFFFFF", "border": "#000000", "text": "#000000",
        "secondary": "#000000", "muted": "#333333", "accent": "#0040C0", "on_accent": "#FFFFFF",
        "accent_soft": "#D6E2FA", "data": "#A23B00", "data_soft": "#F6DFCF", "selection": "#7A5C00",
        "pick": "#A23B00", "error": "#B00020", "ok": "#1B5E20", "viewport": "#FFFFFF", "hover": "#E6E6E6",
        "pressed": "#CCCCCC", "accent_hover": "#00349C", "accent_pressed": "#002878",
    },
}
DEFAULT = "graphite"
DESCRIPTIONS = {"paper": "light", "graphite": "dark (default)", "midnight": "dark blue",
                "solarized light": "Solarized, light", "solarized dark": "Solarized, dark",
                "high contrast": "white, black text and borders"}


# Interaction shades: (token, from, toward, amount). Built-in palettes pick them by hand;
# ``@ui_theme`` plugins that leave them out get them mixed from their own colors (:func:`derive`).
DERIVED = (("hover", "raised", "text", 0.07), ("pressed", "raised", "text", 0.12),
           ("accent_hover", "accent", "text", 0.15), ("accent_pressed", "accent", "text", 0.28))


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


def derive(tokens: dict, own=()) -> dict:
    """``tokens`` with the interaction shades not in ``own`` mixed from its base colors."""
    out = dict(tokens)
    for key, a, b, t in DERIVED:
        if key not in own:
            out[key] = _mix(out[a], out[b], t)
    return out
