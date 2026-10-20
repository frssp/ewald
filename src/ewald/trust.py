"""Whether code from a given source may run: the one place those checks live.

ewald runs Python from several places, and each has a gate:

- notebook cells: an ``.ewd`` someone sent opens restricted until :meth:`Notebook.trust`;
- sessions: ``PythonScript``/``PythonModifier`` code runs only with ``load_session(allow_code=True)``;
- Ask AI and the MCP server: ``run_python`` exists only with ``allow_code=True``;
- the AI's notebook tools (:mod:`ewald.llm.notebook_tools`) insert and edit cells without running
  them; ``run_cell`` is offered only with ``allow_code`` or Preferences › AI › Let the AI run cells,
  and step parameters of code modifiers (``PythonScript``) cannot be set through them;
- plugins: files in the plugin folder and installed packages are trusted like ewald itself;
  a catalog download is checked against the entry's ``sha256`` when it has one
  (:func:`verify_sha256`);
- expressions (``ExpressionSelection``, ``ComputeProperty``) are parsed and evaluated without
  builtins, so they never need trust.

Every ``exec``/``eval``/import-by-name in ewald is listed, with its gate, in
``tests/test_security.py::EXEC_SITES``; adding one fails that test until it is listed there.
"""
from __future__ import annotations

import hashlib


class UntrustedCode(PermissionError):
    """Code from a source that is not trusted (yet) was about to run."""


def require(trusted: bool, source: str, how: str) -> None:
    """Raise :class:`UntrustedCode` unless ``trusted``: "``source`` is not trusted yet: ``how``"."""
    if not trusted:
        raise UntrustedCode(f"{source} is not trusted yet: {how}")


def sha256(data: bytes | str) -> str:
    """Hex SHA-256 of ``data`` (text is hashed as UTF-8)."""
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def verify_sha256(data: bytes | str, expected: str, source: str) -> None:
    """Raise :class:`UntrustedCode` when ``expected`` (hex, case-insensitive) is set and does not
    match ``data``; an empty ``expected`` checks nothing."""
    expected = (expected or "").strip().lower()
    if expected and sha256(data) != expected:
        raise UntrustedCode(f"{source}: the download does not match the catalog's sha256 "
                            f"(expected {expected[:12]}…, got {sha256(data)[:12]}…); not installed")
