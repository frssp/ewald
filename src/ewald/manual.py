"""Links into the manual (``docs/`` in the repository), used by the GUI's Help menu.

The manual is plain Markdown that GitHub renders, and also an MkDocs site (``mkdocs.yml``).
``base`` is where it lives: the docs folder on GitHub by default, or a built or hosted site
(Preferences > General > Documentation URL).
"""
from __future__ import annotations

DEFAULT_BASE = "https://github.com/frssp/ewald/blob/main/docs/"

# Help menu: label -> page under docs/
PAGES = {
    "Tutorials": "user/tutorials/first-notebook.md",
    "User manual": "index.md",
    "Desktop GUI": "user/gui.md",
    "Modifiers": "user/modifiers.md",
    "Python scripting": "user/scripting.md",
    "Writing plugins": "plugins/index.md",
    "Python API reference": "reference/api.md",
}


def url(page: str = "index.md", base: str | None = None) -> str:
    """URL of ``page`` (a path under docs/, e.g. ``"user/gui.md"``) in the manual at ``base``."""
    base = (base or DEFAULT_BASE).rstrip("/") + "/"
    on_github = "github.com" in base and "/blob/" in base
    if on_github:
        if page == "reference/api.md":  # mkdocstrings directives do not render on GitHub
            return base.rsplit("docs/", 1)[0] + "llms.txt"
        return base + page
    stem = page[:-3] if page.endswith(".md") else page
    stem = "" if stem == "index" else stem[:-len("index")] if stem.endswith("/index") else stem + "/"
    if base.startswith("file:"):  # a local `mkdocs build`: directory URLs need index.html
        return base + stem + "index.html"
    return base + stem
