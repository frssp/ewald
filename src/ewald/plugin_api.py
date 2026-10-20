"""The supported surface for plugins, versioned by :data:`PLUGIN_API`.

Plugins import what they need from here::

    from ewald.plugin_api import DataCollection, Modifier, modifier

Every name below keeps its meaning (constructor, attributes, call signature) for as long as
:data:`PLUGIN_API` stays the same; a change that would break a plugin raises it. A plugin
says which version it was written for in its manifest (``plugin-api = 1`` in
``ewald-plugin.toml``, ``"plugin_api": 1`` in a file's ``__ewald__``), and ewald refuses to load
one whose version it does not support (:data:`SUPPORTED`), with a message, instead of letting
it fail somewhere in the middle of a computation. Other ``ewald.*`` modules work too, but they
are internal and may change in any release.

Names are imported on first use, so ``import ewald.plugin_api`` stays cheap.
"""
from __future__ import annotations

import importlib

#: The version of this surface. Raised when a name here changes incompatibly.
PLUGIN_API = 1
#: Versions this ewald loads plugins for.
SUPPORTED = frozenset({1})

# name -> module it lives in
_NAMES = {
    # data model
    "DataCollection": "ewald.data", "Cell": "ewald.data", "VoxelGrid": "ewald.data",
    "Bonds": "ewald.bonds", "SurfaceMesh": "ewald.mesh", "Table": "ewald.verbs",
    "ElectronicStructure": "ewald.electronic",
    # pipeline
    "Pipeline": "ewald.pipeline", "Modifier": "ewald.pipeline", "ComputeContext": "ewald.pipeline",
    "ComputeCancelled": "ewald.pipeline", "StaticSource": "ewald.io", "load": "ewald",
    # readers
    "FrameReader": "ewald.io.base", "open_binary": "ewald.io.base", "head_lines": "ewald.io.base",
    "strip_compression": "ewald.io.base", "floats": "ewald.io.base",
    # compute helpers
    "find_pairs": "ewald.neighbors", "element_masses": "ewald.radii",
    # drawing
    "VisualElement": "ewald.scene", "Spheres": "ewald.scene", "Cylinders": "ewald.scene",
    "Lines": "ewald.scene", "make_arrows": "ewald.scene",
    "OverlayElement": "ewald.overlays", "Text": "ewald.overlays", "Rect": "ewald.overlays",
    "Image": "ewald.overlays", "OverlayContext": "ewald.overlays", "Theme": "ewald.themes",
    # AI tools
    "ToolResult": "ewald.llm.tools", "downsample": "ewald.llm.tools",
    # registration
    "modifier": "ewald.plugins", "reader": "ewald.plugins", "writer": "ewald.plugins",
    "visual": "ewald.plugins", "overlay": "ewald.plugins", "colormap": "ewald.plugins",
    "theme": "ewald.plugins", "ui_theme": "ewald.plugins", "ai_tool": "ewald.plugins",
    "claude_tool": "ewald.plugins", "ai_provider": "ewald.plugins",
    "panel": "ewald.plugins",
}

__all__ = ["PLUGIN_API", "SUPPORTED", *_NAMES]


def __getattr__(name):
    module = _NAMES.get(name)
    if module is None:
        raise AttributeError(f"ewald.plugin_api has no {name!r} (the plugin API is: {', '.join(_NAMES)})")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)


def check(version) -> str:
    """"" when a plugin written for API ``version`` (None: not declared) loads here, else why not."""
    if version in (None, ""):
        return ""
    try:
        v = int(version)
    except (TypeError, ValueError):
        return f"plugin-api {version!r} is not a whole number"
    if v in SUPPORTED:
        return ""
    newer = v > max(SUPPORTED)
    return (f"written for plugin API {v}; this ewald supports {', '.join(map(str, sorted(SUPPORTED)))}"
            + (" (update ewald)" if newer else " (update the plugin)"))
