"""Whether code from a given source may run: the one place those checks live.

ewald runs Python from several places, and each has a gate:

- notebook cells: a notebook file (``.py``, ``.ipynb``, ``.ewd``) you did not save or trust
  yourself opens restricted in the app until :meth:`Notebook.trust`, as in Jupyter. Notebooks you
  save or trust are remembered by content hash (:func:`remember_file`, :func:`is_trusted_file`);
- sessions: ``PythonScript``/``PythonModifier`` code runs only with ``load_session(allow_code=True)``;
- Ask AI and the MCP server: the AI changes results only by running notebook cells. Its own code
  goes into cells it inserts or edits, which run only through ``run_cell``, offered only with
  ``allow_code`` or Preferences › AI › Let the AI run cells. The cells ewald writes for a tool
  (a ``pipe.add(Name(...))`` step, the view cell, ``pipe = load(...)``, ``export_file(...)``) hold
  a validated name and JSON literals only, so they run without that, but never in a restricted
  notebook; step parameters of code modifiers (``PythonScript``) cannot be set through them;
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
import json
import os
import time

#: How many trusted notebook files are remembered (the oldest are forgotten first).
TRUST_LIMIT = 5000


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


# --------------------------------------------------------------- trusted notebook files
# Like Jupyter's notebook signatures: the SHA-256 of every notebook file this user saved or
# trusted, in <config dir>/trusted_notebooks.json. A file whose bytes match one of them opens
# trusted wherever it is; a changed or unknown file opens restricted. No secret key is needed:
# whoever can write the config folder can already put plugins there.

def _trust_store():
    from .settings import config_dir

    return config_dir() / "trusted_notebooks.json"


def _read_store() -> dict:
    try:
        with open(_trust_store(), encoding="utf-8") as fh:
            doc = json.load(fh)
        return dict(doc.get("sha256") or {}) if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _file_hash(path) -> str | None:
    try:
        with open(path, "rb") as fh:
            return sha256(fh.read())
    except OSError:
        return None


def is_trusted_file(path) -> bool:
    """Whether the notebook file at ``path`` is one this user saved or trusted, byte for byte."""
    h = _file_hash(path)
    return h is not None and h in _read_store()


def remember_file(path) -> bool:
    """Remember the notebook file at ``path`` (as it is on disk now) as trusted; False when it
    cannot be read or the list cannot be written."""
    h = _file_hash(path)
    if h is None:
        return False
    entries = _read_store()
    entries[h] = {"path": os.path.abspath(os.fspath(path)), "time": time.time()}
    if len(entries) > TRUST_LIMIT:
        keep = sorted(entries.items(), key=lambda kv: kv[1].get("time", 0) if isinstance(kv[1], dict) else 0)
        entries = dict(keep[-TRUST_LIMIT:])
    store = _trust_store()
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.with_name(store.name + ".tmp")
        tmp.write_text(json.dumps({"format": 1, "sha256": entries}, indent=0), encoding="utf-8")
        os.replace(tmp, store)
    except OSError:
        return False
    return True
