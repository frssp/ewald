"""``ewald optimade search LiFePO4 --db mp`` / ``ewald optimade databases``."""
from __future__ import annotations

import json
import sys


def run(args) -> int:
    from . import providers as P
    from .search import search

    if args.optimade_cmd == "databases":
        for p in P.providers().values():
            print(f"{p.name:<20} {p.title:<45} {p.base_url}")
        return 0
    res = search(" ".join(args.query), db=args.db, limit=args.limit)
    if args.json:
        json.dump({"filter": res.filter, "total": res.total, "hits": [h.as_dict() for h in res]}, sys.stdout,
                  indent=1)
        print()
    else:
        print(repr(res))
    return 0
