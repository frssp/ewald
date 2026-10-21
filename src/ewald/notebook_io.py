"""Notebook files: Jupyter ``.ipynb`` and ewald's own ``.ewd``.

``.ipynb`` is the standard notebook format (nbformat 4): code, Markdown and the outputs under
each cell, readable in Jupyter, VS Code and on GitHub. ewald keeps what Jupyter does not know
(cell titles, the viewer state) in metadata, which Jupyter preserves.

``.ewd`` is one zip file with everything needed to read and re-run an analysis::

    analysis.ewd
    ├── notebook.ipynb   code, outputs and metadata (opens in Jupyter once unzipped)
    ├── manifest.json    ewald version, data files (path, size, fingerprint), plugins, viewer state
    ├── figures/         every view cell as SVG and PNG, with the plotted data as CSV
    ├── data/            data files up to ``data_limit_mb`` (larger ones stay links)
    └── plugins/         plugin files the notebook uses, from the plugin folder

Large trajectories stay where they are: the manifest records their path relative to the
``.ewd`` file, their size and a fingerprint, so opening warns when a file moved or changed.

A received ``.ewd`` (one this user did not save or trust) is code, so it opens *restricted*: nothing runs and its plugins are not
imported until :meth:`ewald.notebook.Notebook.trust` (the GUI asks first). Saved outputs and
figures are readable without running anything.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import warnings
import zipfile
from pathlib import Path, PureWindowsPath

FORMAT = 1
NOTEBOOK_NAME = "notebook.ipynb"
MANIFEST_NAME = "manifest.json"
_KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


# ---------------------------------------------------------------- .ipynb
def _lines(text: str) -> list[str]:
    return text.splitlines(True)


def _output_to_json(out) -> dict:
    if out.kind == "text":
        return {"output_type": "stream", "name": "stdout", "text": _lines(out.data)}
    if out.kind == "error":
        text = str(out.data)
        first = text.strip().splitlines()[-1] if text.strip() else "Error"
        name, _, value = first.partition(": ")
        return {"output_type": "error", "ename": name or "Error", "evalue": value, "traceback": text.splitlines()}
    if out.kind == "image":
        return {"output_type": "display_data", "metadata": {"ewald": {"view": out.view}} if out.view else {},
                "data": {"image/png": base64.b64encode(out.data).decode("ascii"), "text/plain": ["<Figure>"]}}
    return {"output_type": "execute_result", "execution_count": None, "metadata": {},
            "data": {"text/plain": _lines(str(out.data))}}


def _output_from_json(raw: dict):
    from .notebook import Output

    kind = raw.get("output_type")

    def text(v):
        return "".join(v) if isinstance(v, list) else str(v or "")
    if kind == "stream":
        return Output("text", text(raw.get("text")))
    if kind == "error":
        return Output("error", "\n".join(raw.get("traceback") or [f"{raw.get('ename')}: {raw.get('evalue')}"]))
    data = raw.get("data") or {}
    if "image/png" in data:
        view = ((raw.get("metadata") or {}).get("ewald") or {}).get("view")
        return Output("image", base64.b64decode(text(data["image/png"])), view if isinstance(view, dict) else None)
    if "text/plain" in data:
        return Output("value", text(data["text/plain"]))
    return None


def to_ipynb(nb, outputs: bool = True) -> dict:
    """The notebook as an nbformat 4 dict (``outputs=False`` leaves the outputs out)."""
    cells = []
    for cell in nb.cells:
        meta = {"title": cell.title} if cell.title else {}
        if cell.kind == "code" and cell.frames:
            meta["ewald"] = {"frames": cell.frames}
        if cell.collapsed:  # JupyterLab's collapsed input and output
            meta["jupyter"] = {"source_hidden": True, "outputs_hidden": True}
        if cell.kind == "markdown":
            cells.append({"id": cell.id, "cell_type": "markdown", "metadata": meta, "source": _lines(cell.source)})
            continue
        cells.append({"id": cell.id, "cell_type": "code", "metadata": meta, "source": _lines(cell.source),
                      "execution_count": cell.count if outputs else None,
                      "outputs": [_output_to_json(o) for o in cell.outputs] if outputs else []})
    meta = dict(nb.metadata)
    meta.setdefault("kernelspec", _KERNEL)
    meta.setdefault("language_info", {"name": "python"})
    ewald = dict(meta.get("ewald") or {})
    ewald.update(version=_version(), pipeline=nb.pipeline_name)
    if nb.state:
        ewald["state"] = _jsonable(nb.state)
    meta["ewald"] = ewald
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def to_ipynb_text(nb, outputs: bool = True) -> str:
    return json.dumps(to_ipynb(nb, outputs), indent=1, ensure_ascii=False) + "\n"


def from_ipynb(doc, cls=None, **kwargs):
    """A Notebook from an nbformat 4 dict or JSON text; outputs and metadata are kept."""
    from .notebook import Cell, Notebook

    cls = cls or Notebook
    if isinstance(doc, (str, bytes)):
        doc = json.loads(doc)
    if int(doc.get("nbformat", 4)) < 4:
        raise ValueError("only nbformat 4 notebooks can be read (save it again with Jupyter)")
    cells = []
    for raw in doc.get("cells", []):
        kind = raw.get("cell_type")
        if kind not in ("code", "markdown"):
            continue  # raw cells
        source = raw.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        raw_meta = raw.get("metadata") or {}
        frames = (raw_meta.get("ewald") or {}).get("frames", "") if kind == "code" else ""
        cell = Cell(source.rstrip("\n"), kind=kind, title=raw_meta.get("title", ""),
                    frames=frames if frames in ("each", "all") else "",
                    collapsed=bool((raw_meta.get("jupyter") or {}).get("source_hidden")))
        if raw.get("id"):
            cell.id = str(raw["id"])
        if kind == "code":
            cell.outputs = [o for o in map(_output_from_json, raw.get("outputs", [])) if o is not None]
            cell.count = raw.get("execution_count")
        cells.append(cell)
    meta = dict(doc.get("metadata") or {})
    ewald = meta.get("ewald") or {}
    kwargs.setdefault("pipeline_name", ewald.get("pipeline", "pipe"))
    nb = cls(cells, **kwargs)
    nb.metadata = meta
    nb.state = dict(ewald.get("state") or {})
    return nb


# ---------------------------------------------------------------- data files
def fingerprint(path, chunk: int = 1 << 20) -> str:
    """sha256 of the size and the first and last MiB: quick on multi-GB trajectories and
    enough to notice a file that was replaced or rewritten."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(chunk, size - chunk))
            h.update(fh.read(chunk))
    return "sha256-ends:" + h.hexdigest()


def data_files(nb) -> list[str]:
    """Local files the notebook's pipelines read (absolute paths)."""
    from .pipeline import Pipeline

    found = []
    for value in list(nb.namespace.values()):
        if not isinstance(value, Pipeline):
            continue
        src = value.source
        paths = getattr(src, "paths", None) or getattr(src, "path", None)
        for p in paths if isinstance(paths, (list, tuple)) else [paths]:
            if isinstance(p, (str, os.PathLike)) and os.path.isfile(p):
                p = os.path.abspath(p)
                if p not in found:
                    found.append(p)
    return found


def _relative(path: str, base: str) -> str:
    """``path`` relative to ``base`` with ``/`` separators (absolute on another drive)."""
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:  # another drive on Windows
        return path


def _bundle_name(path: str) -> str:
    """The zip member for a data file: ``data/`` + its relative path with ``/`` separators and
    no drive, root or ``..`` parts, so a file on another Windows drive (kept absolute,
    ``D:\\runs\\XDATCAR``) gets a valid name too. The same on every platform."""
    pure = PureWindowsPath(path)   # splits on / and on backslashes
    keep = [part for part in pure.parts if part not in (pure.anchor, ".", "..")]
    return "data/" + "/".join(keep)


# ---------------------------------------------------------------- .ewd
def _slug(text: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "view"
    name, k = base, 2
    while name in used:
        name, k = f"{base}-{k}", k + 1
    used.add(name)
    return name


def save_ewd(nb, path, *, figures: bool = True, data_limit_mb: float = 50.0, plugins: bool = True,
             outputs: bool = True) -> dict:
    """Write ``nb`` as an ``.ewd`` file; returns the manifest. ``data_limit_mb`` is the size up
    to which a data file is copied into the file (0: links only)."""
    from . import plugins as plugin_registry

    path = os.path.abspath(os.fspath(path))
    base = os.path.dirname(path)
    manifest = {"ewald_notebook": FORMAT, "ewald_version": _version(), "name": nb.name,
                "data": [], "figures": [], "plugins": [], "state": _jsonable(nb.state)}
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(NOTEBOOK_NAME, to_ipynb_text(nb, outputs))
        for f in data_files(nb):
            size = os.path.getsize(f)
            entry = {"path": _relative(f, base), "size": size, "fingerprint": fingerprint(f), "bundled": None}
            if size <= data_limit_mb * 1e6:
                entry["bundled"] = _bundle_name(entry["path"])
                z.write(f, entry["bundled"])
            manifest["data"].append(entry)
        if figures and nb.pipeline is not None:
            manifest["figures"] = _write_figures(nb, z)
        if plugins:
            manifest["plugins"] = _write_plugins(nb, z, plugin_registry)
        z.writestr(MANIFEST_NAME, json.dumps(manifest, indent=1, ensure_ascii=False))
    os.replace(tmp, path)
    return manifest


def _write_figures(nb, z) -> list:
    import io

    from . import plotstyle
    from .views import DRAW_ARGS, figure, view_data

    out, used = [], set()
    for k, cell in enumerate(nb.cells):
        args = cell.view
        if not args:
            continue
        source = nb.namespace.get(args.pop("source"))
        if source is None:
            continue
        name = _slug(cell.title or args.get("table") or args.get("histogram") or f"view-{k + 1}", used)
        try:
            title, columns = view_data(source, **{k: v for k, v in args.items() if k in DRAW_ARGS})
            fig = figure(source, **{k: v for k, v in args.items() if k != "panel"})
        except Exception as exc:  # noqa: BLE001 - a view that cannot be drawn now is skipped
            warnings.warn(f"view cell {k + 1}: {exc}", stacklevel=3)
            continue
        for ext in ("svg", "png"):
            buf = io.BytesIO()
            plotstyle.save(fig, buf, format=ext, dpi=300)
            z.writestr(f"figures/{name}.{ext}", buf.getvalue())
        buf = io.StringIO()
        names = list(columns)
        buf.write(",".join(names) + "\n")
        for row in zip(*(columns[n] for n in names)):
            buf.write(",".join(repr(float(v)) for v in row) + "\n")
        z.writestr(f"figures/{name}.csv", buf.getvalue())
        out.append({"cell": k, "name": name, "title": title, "files": [f"figures/{name}.{e}" for e in ("svg", "png", "csv")]})
    return out


def _write_plugins(nb, z, registry) -> list:
    """Plugin-folder plugins whose modifiers the notebook's pipelines use (files are copied)."""
    used = set()
    for value in nb.namespace.values():
        for m in getattr(value, "modifiers", []) or []:
            used.add(type(m).__name__)
    folder = registry.plugin_dir().resolve()
    out = []
    for pid, p in registry.installed().items():
        names = set(p.contributions.get("modifiers", []))
        if not names & used:
            continue
        entry = {"id": pid, "name": p.name, "version": p.version, "kind": p.kind, "modifiers": sorted(names & used),
                 "file": None}
        src = Path(p.origin)
        if p.kind != "package" and src.is_file() and src.resolve().parent == folder:
            entry["file"] = f"plugins/{src.name}"
            z.write(src, entry["file"])
        elif p.kind == "package":
            entry["requirement"] = f"{p.origin}=={p.version}" if p.version else p.origin
        out.append(entry)
    return out


def _safe_members(z: zipfile.ZipFile) -> list[str]:
    names = []
    for info in z.infolist():
        name = info.filename
        parts = Path(name).parts
        if name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
            raise ValueError(f"unsafe path in .ewd file: {name!r}")
        names.append(name)
    return names


def read_manifest(path) -> dict:
    with zipfile.ZipFile(path) as z:
        if MANIFEST_NAME not in z.namelist():
            raise ValueError(f"{path} is not an ewald notebook (.ewd): no {MANIFEST_NAME}")
        return json.loads(z.read(MANIFEST_NAME))


def bundle_dir_for(path) -> Path:
    """Where an ``.ewd`` file is unpacked: the config cache, one folder per file content."""
    from .settings import config_dir

    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return config_dir() / "cache" / "ewd" / f"{Path(path).stem}-{h}"


def load_ewd(path, cls=None, trusted: bool | None = None, **kwargs):
    """Open an ``.ewd`` file. The notebook keeps its saved outputs and is ``restricted`` (its
    cells do not run, its plugins are not imported) unless ``trusted`` (None: when this user
    saved or trusted this exact file, :func:`ewald.trust.is_trusted_file`). Data files are found
    where the manifest says, next to the ``.ewd`` file, or in its bundled copy; a changed or
    missing file is listed in ``nb.warnings``."""
    path = os.path.abspath(os.fspath(path))
    if trusted is None:
        from .trust import is_trusted_file

        trusted = is_trusted_file(path)
    with zipfile.ZipFile(path) as z:
        names = _safe_members(z)
        if MANIFEST_NAME not in names or NOTEBOOK_NAME not in names:
            raise ValueError(f"{path} is not an ewald notebook (.ewd)")
        manifest = json.loads(z.read(MANIFEST_NAME))
        if int(manifest.get("ewald_notebook", 0)) > FORMAT:
            raise ValueError(f"{path} was written by a newer ewald (notebook format {manifest['ewald_notebook']})")
        nb = from_ipynb(z.read(NOTEBOOK_NAME).decode("utf-8"), cls=cls, path=path, **kwargs)
        bundle = bundle_dir_for(path)
        if not bundle.exists():
            tmp = bundle.with_name(bundle.name + ".part")
            shutil.rmtree(tmp, ignore_errors=True)
            for name in names:
                if name.startswith(("data/", "plugins/", "figures/")) and not name.endswith("/"):
                    target = tmp / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
            tmp.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, bundle)
    nb.bundle_dir = str(bundle)
    nb.manifest = manifest
    nb.state = dict(manifest.get("state") or nb.state)
    nb.warnings = check_data(manifest, os.path.dirname(path), bundle)
    nb.warnings += [f"needs the pip package {e['requirement']} for {', '.join(e.get('modifiers', [])) or e['id']} "
                    f"(pip install {e['requirement']})" for e in missing_packages(manifest)]
    nb.data_dirs = [str(bundle / "data")]
    nb.restricted = not trusted
    if trusted:
        load_bundled_plugins(nb)
    return nb


def check_data(manifest: dict, base: str, bundle: Path) -> list[str]:
    """Messages for data files that moved, changed or are only in the bundled copy."""
    found = []
    for entry in manifest.get("data", []):
        rel = entry["path"]
        local = rel if os.path.isabs(rel) else os.path.join(base, rel)
        if os.path.isfile(local):
            if fingerprint(local) != entry.get("fingerprint"):
                found.append(f"{rel} changed since the notebook was saved")
        elif entry.get("bundled") and (bundle / entry["bundled"]).is_file():
            found.append(f"{rel} not found next to the notebook: using the copy saved in it")
        else:
            found.append(f"{rel} not found ({entry.get('size', 0) / 1e6:.1f} MB, not saved in the notebook)")
    return found


def missing_packages(manifest: dict) -> list[dict]:
    """Plugin entries of an ``.ewd`` manifest that come from a pip package this Python lacks
    (not installed under that id, and their modifiers are not registered either): each has
    ``requirement`` (``name==version``), ``id``, ``name`` and ``modifiers``."""
    from . import plugins as registry

    have = registry.installed()
    known = set(registry.modifiers())
    out = []
    for entry in (manifest or {}).get("plugins", []):
        if entry.get("kind") != "package" or not entry.get("requirement") or entry.get("id") in have:
            continue
        mods = entry.get("modifiers") or []
        if mods and set(mods) <= known:
            continue
        out.append(entry)
    return out


def load_bundled_plugins(nb) -> list[str]:
    """Import the plugin files an ``.ewd`` carries (only after the notebook is trusted). A
    plugin already installed under the same id is used as installed."""
    from . import plugins as registry

    if not nb.bundle_dir:
        return []
    have = registry.installed()
    names = []
    for entry in getattr(nb, "manifest", {}).get("plugins", []):
        f = entry.get("file")
        if not f or entry.get("id") in have:
            continue
        names += registry.load_plugin_file(Path(nb.bundle_dir) / f)
    return names


# ---------------------------------------------------------------- helpers
def _version() -> str:
    from . import __version__

    return __version__


def _jsonable(value):
    """Viewer state as plain JSON (numpy arrays and tuples become lists)."""
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
