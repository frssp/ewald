"""A small OPTIMADE server on 127.0.0.1 (JSON:API answers as in the OPTIMADE v1.2 spec), set up
used as the database ``test`` (tests/test_optimade.py), so the tests run the real HTTP code without
the internet."""
import json
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = 5.43

STRUCTURES = {
    "mp-149": {   # Si, diamond (primitive cell)
        "lattice_vectors": [[0, A / 2, A / 2], [A / 2, 0, A / 2], [A / 2, A / 2, 0]],
        "dimension_types": [1, 1, 1], "nperiodic_dimensions": 3,
        "cartesian_site_positions": [[0, 0, 0], [A / 4, A / 4, A / 4]],
        "species_at_sites": ["Si", "Si"],
        "species": [{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
        "elements": ["Si"], "nelements": 1, "nsites": 2,
        "chemical_formula_reduced": "Si", "chemical_formula_anonymous": "A",
        "last_modified": "2026-01-01T00:00:00Z", "structure_features": []},
    "1000041": {  # NaCl, rock salt (conventional), Cl site partly vacant
        "lattice_vectors": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        "dimension_types": [1, 1, 1],
        "cartesian_site_positions": [[0, 0, 0], [2.82, 2.82, 0], [2.82, 0, 2.82], [0, 2.82, 2.82],
                                     [2.82, 0, 0], [0, 2.82, 0], [0, 0, 2.82], [2.82, 2.82, 2.82]],
        "species_at_sites": ["Na", "Na", "Na", "Na", "Cl1", "Cl1", "Cl1", "Cl1"],
        "species": [{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
                    {"name": "Cl1", "chemical_symbols": ["Cl", "vacancy"], "concentration": [0.9, 0.1]}],
        "elements": ["Cl", "Na"], "nelements": 2, "nsites": 8, "chemical_formula_reduced": "ClNa",
        "structure_features": ["disorder"]},
    "water": {    # a molecule: no lattice
        "lattice_vectors": [[None, None, None]] * 3, "dimension_types": [0, 0, 0],
        "cartesian_site_positions": [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
        "species_at_sites": ["O", "H", "H"],
        "species": [{"name": "O", "chemical_symbols": ["O"], "concentration": [1]},
                    {"name": "H", "chemical_symbols": ["H"], "concentration": [1]}],
        "elements": ["H", "O"], "nelements": 2, "nsites": 3, "chemical_formula_reduced": "H2O"},
    "mp-2/slab": {   # an id with a slash, periodic in two directions
        "lattice_vectors": [[2.5, 0, 0], [-1.25, 2.165, 0], [None, None, None]], "dimension_types": [1, 1, 0],
        "cartesian_site_positions": [[0, 0, 0], [1.25, 0.72, 0]], "species_at_sites": ["C", "C"],
        "species": [{"name": "C", "chemical_symbols": ["C"], "concentration": [1]}],
        "elements": ["C"], "nelements": 1, "nsites": 2, "chemical_formula_reduced": "C"},
}


def _resource(i):
    return {"id": i, "type": "structures", "attributes": STRUCTURES[i]}


def _matches(flt: str, attrs: dict) -> bool:
    """The few filters the tests send."""
    if not flt:
        return True
    if flt.startswith("bad"):
        raise ValueError(flt)
    m = re.fullmatch(r'chemical_formula_reduced="(\w+)"', flt)
    if m:
        return attrs["chemical_formula_reduced"] == m[1]
    m = re.fullmatch(r'elements HAS ALL ((?:"\w+",?)+)(?: AND nelements=(\d+))?', flt)
    if m:
        need = set(re.findall(r'"(\w+)"', m[1]))
        return need <= set(attrs["elements"]) and (m[2] is None or attrs["nelements"] == int(m[2]))
    raise ValueError(flt)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, doc):
        body = json.dumps(doc).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/vnd.api+json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        self.server.requests.append(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        path = urllib.parse.unquote(url.path)
        if path.startswith("/v1/structures/"):
            i = path[len("/v1/structures/"):]
            if i not in STRUCTURES:
                return self._send(404, {"errors": [{"status": "404", "title": "Not Found",
                                                    "detail": f"no structure {i}"}]})
            return self._send(200, {"data": _resource(i), "meta": {"api_version": "1.2.0"}})
        if path == "/v1/structures":
            try:
                ids = [i for i in STRUCTURES if _matches(q.get("filter", ""), STRUCTURES[i])]
            except ValueError as exc:
                return self._send(400, {"errors": [{"status": "400", "title": "Bad Request",
                                                    "detail": f"cannot parse filter {exc}"}]})
            limit = min(int(q.get("page_limit", 10)), self.server.page_max)
            offset = int(q.get("page_offset", 0))
            page = [_resource(i) for i in ids[offset:offset + limit]]
            fields = q.get("response_fields")
            if fields:
                keep = fields.split(",")
                page = [{**r, "attributes": {k: v for k, v in r["attributes"].items() if k in keep}} for r in page]
            nxt = None
            if offset + limit < len(ids):
                nq = {**q, "page_offset": str(offset + limit)}
                nxt = {"href": f"http://127.0.0.1:{self.server.server_port}/v1/structures?"
                               + urllib.parse.urlencode(nq)}
            return self._send(200, {"data": page, "links": {"next": nxt},
                                    "meta": {"data_returned": len(ids), "more_data_available": nxt is not None}})
        self._send(404, {"errors": [{"status": "404", "title": "Not Found"}]})


def start():
    """A running server on a free port (``srv.requests``: the paths asked for); stop it with
    ``srv.shutdown(); srv.server_close()``."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    srv.page_max = 2          # small pages, so that searches follow links.next
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
