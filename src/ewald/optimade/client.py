"""A small OPTIMADE client: GET JSON:API pages of ``/v1/structures`` (stdlib only).

Uses the operating system's certificate store (:mod:`ewald._tls`), so a
proxy that re-signs HTTPS works the same on Windows, macOS and Linux; ``HTTPS_PROXY`` is
honoured by :mod:`urllib`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .._tls import ssl_context

TIMEOUT = 60
#: fields asked for in searches; a single structure is read whole
SEARCH_FIELDS = ("chemical_formula_reduced", "chemical_formula_descriptive", "nsites", "nelements",
                 "elements", "last_modified")


class OptimadeError(RuntimeError):
    """The server answered with an error, or could not be reached."""


def _version() -> str:
    from ..plugin_manifest import ewald_version

    return ewald_version()


def get_json(url: str) -> dict:
    if not url.lower().startswith(("https://", "http://")):   # a server's next link must not reach file://
        raise OptimadeError(f"not an http(s) address: {url!r}")
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.api+json, application/json",
        "User-Agent": f"ewald/{_version()}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT,
                                    context=ssl_context() if url.lower().startswith("https:") else None) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        raise OptimadeError(f"{url}: {exc.code} {_errors(exc.read()) or exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise OptimadeError(f"{url}: {getattr(exc, 'reason', exc)}") from None
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise OptimadeError(f"{url}: the answer is not JSON (is this an OPTIMADE base URL?)") from None
    if not isinstance(doc, dict):
        raise OptimadeError(f"{url}: unexpected answer")
    if doc.get("errors") and "data" not in doc:
        raise OptimadeError(f"{url}: {_errors(doc)}")
    return doc


def _errors(doc) -> str:
    if isinstance(doc, (bytes, str)):
        try:
            doc = json.loads(doc)
        except ValueError:
            return ""
    errors = doc.get("errors") if isinstance(doc, dict) else None
    if not errors:
        return ""
    return "; ".join(" ".join(str(e.get(k)) for k in ("title", "detail") if e.get(k)) for e in errors
                     if isinstance(e, dict))


def entry_url(structures_url: str, entry_id: str) -> str:
    return structures_url.rstrip("/") + "/" + urllib.parse.quote(entry_id, safe="")


def query_url(structures_url: str, filter: str = "", page_limit: int = 25, fields=None) -> str:
    params = {"page_limit": str(page_limit)}
    if filter:
        params["filter"] = filter
    if fields:
        params["response_fields"] = ",".join(fields)
    return structures_url.rstrip("/") + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def next_link(doc: dict) -> str | None:
    link = (doc.get("links") or {}).get("next")
    if isinstance(link, dict):
        link = link.get("href")
    return link or None


def structure(structures_url: str, entry_id: str) -> dict:
    """The JSON:API resource object of one structure."""
    data = get_json(entry_url(structures_url, entry_id)).get("data")
    if isinstance(data, list):      # some servers answer a single entry as a list
        data = data[0] if data else None
    if not data:
        raise OptimadeError(f"{structures_url}: no structure {entry_id!r}")
    return data


def structures(structures_url: str, filter: str = "", limit: int = 25, fields=None, page_limit=None):
    """Up to ``limit`` resource objects matching ``filter``, following ``links.next``; returns
    ``(entries, number the server says match or None)``."""
    url = query_url(structures_url, filter, page_limit or min(limit, 100), fields)
    out, total = [], None
    while url and len(out) < limit:
        doc = get_json(url)
        if total is None:
            total = (doc.get("meta") or {}).get("data_returned")
        data = doc.get("data") or []
        if not data:
            break
        out += data
        url = next_link(doc)
    return out[:limit], total
