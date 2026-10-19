"""Panel layout of the main window as plain data (no Qt): areas of tab groups in splits.

The window has two areas, ``editor`` (the big middle, where the 3D view starts) and ``panel``
(below it: console, inspector, plots, log). Each area is a tree: a :class:`Split` holds
children side by side (``"h"``) or stacked (``"v"``) with relative sizes, and a :class:`Group`
is a leaf, a tab bar over one shown view. Views are named by string ids (``"viewport"``,
``"plot:1"``, ``"console"``, ...). :meth:`Layout.move` is the one edit every drag, menu entry
and shortcut goes through; ``gui/workbench.py`` draws the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

AREAS = ("editor", "panel")
ZONES = ("center", "left", "right", "top", "bottom")
VERSION = 2
_ids = count(1)


@dataclass(eq=False)
class Group:
    views: list[str] = field(default_factory=list)
    current: int = 0
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def current_view(self) -> str | None:
        if not self.views:
            return None
        return self.views[min(max(self.current, 0), len(self.views) - 1)]


@dataclass(eq=False)
class Split:
    orient: str  # "h": children left to right, "v": top to bottom
    children: list = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)


def zone_at(width: float, height: float, x: float, y: float, edge: float = 0.25) -> str:
    """The drop zone of point (x, y) in a ``width`` x ``height`` group body: the nearest edge
    when within ``edge`` of it (as a fraction of that side), else the centre."""
    if width <= 0 or height <= 0:
        return "center"
    fx, fy = x / width, y / height
    near = {"left": fx, "right": 1 - fx, "top": fy, "bottom": 1 - fy}
    name = min(near, key=near.get)
    return name if near[name] < edge else "center"


class Layout:
    def __init__(self, areas: dict | None = None):
        self.areas = areas or {a: Group() for a in AREAS}

    # ------------------------------------------------------------------ queries
    def groups(self, area: str | None = None) -> list[Group]:
        out = []
        for name in ([area] if area else AREAS):
            _walk(self.areas[name], out)
        return out

    def views(self) -> list[str]:
        return [v for g in self.groups() for v in g.views]

    def find(self, view: str) -> Group | None:
        return next((g for g in self.groups() if view in g.views), None)

    def area_of(self, group: Group) -> str:
        return next(a for a in AREAS if group in self.groups(a))

    def group(self, gid: int) -> Group | None:
        return next((g for g in self.groups() if g.id == gid), None)

    def parent(self, node) -> tuple[Split | None, int]:
        for root in self.areas.values():
            found = _parent(root, node)
            if found:
                return found
        return None, -1

    def neighbour(self, group: Group, direction: str) -> Group | None:
        """The group next to ``group`` towards ``direction`` (left/right/top/bottom) in its area."""
        orient = "h" if direction in ("left", "right") else "v"
        step = -1 if direction in ("left", "top") else 1
        node = group
        while True:
            split, i = self.parent(node)
            if split is None:
                return None
            if split.orient == orient and 0 <= i + step < len(split.children):
                return _edge_group(split.children[i + step], orient, first=step > 0)
            node = split

    # ------------------------------------------------------------------ edits
    def show(self, view: str) -> Group | None:
        g = self.find(view)
        if g is not None:
            g.current = g.views.index(view)
        return g

    def add(self, view: str, group: Group, index: int | None = None):
        if view in self.views():
            self.remove(view)
        index = len(group.views) if index is None else max(0, min(index, len(group.views)))
        group.views.insert(index, view)
        group.current = index

    def remove(self, view: str):
        g = self.find(view)
        if g is None:
            return
        i = g.views.index(view)
        g.views.remove(view)
        if g.current >= i and g.current > 0:
            g.current -= 1
        self.normalize()

    def move(self, view: str, target: Group, zone: str = "center", index: int | None = None) -> Group:
        """Put ``view`` into ``target`` (zone ``center``, at ``index`` or the end) or into a new
        group split off ``target`` on the ``zone`` side. Returns the group that now shows it."""
        if zone not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}, not {zone!r}")
        src = self.find(view)
        if zone == "center":
            if src is target and index is not None:  # reorder within the bar
                old = target.views.index(view)
                target.views.remove(view)
                index = index - 1 if index > old else index
            elif src is not None:
                self._take(view, src)
            index = len(target.views) if index is None else max(0, min(index, len(target.views)))
            target.views.insert(index, view)
            target.current = index
            self.normalize()
            return target
        if src is target and len(target.views) == 1:
            return target  # splitting a group off itself: nothing to do
        if src is not None:
            self._take(view, src)
        new = Group([view])
        orient = "h" if zone in ("left", "right") else "v"
        first = zone in ("left", "top")
        split, i = self.parent(target)
        if split is not None and split.orient == orient:
            share = split.sizes[i] / 2 if split.sizes else 1
            if split.sizes:
                split.sizes[i] = share
            split.children.insert(i if first else i + 1, new)
            if split.sizes:
                split.sizes.insert(i if first else i + 1, share)
        else:
            pair = Split(orient, [new, target] if first else [target, new], [1, 1])
            self._replace(target, pair)
        self.normalize()
        return new

    def _take(self, view: str, group: Group):
        i = group.views.index(view)
        group.views.remove(view)
        if group.current > i or group.current >= len(group.views):
            group.current = max(group.current - 1, 0)

    def _replace(self, node, new):
        split, i = self.parent(node)
        if split is None:
            for a, root in self.areas.items():
                if root is node:
                    self.areas[a] = new
            return
        split.children[i] = new

    def normalize(self):
        """Drop empty groups (an area keeps one, possibly empty), lift single children out of
        their split and merge a split into a parent of the same orientation."""
        for a, root in self.areas.items():
            root = _prune(root)
            self.areas[a] = root if root is not None else Group()

    # ------------------------------------------------------------------ files
    def to_json(self) -> dict:
        return {"version": VERSION, **{a: _dump(self.areas[a]) for a in AREAS}}

    @classmethod
    def from_json(cls, data: dict, known=None, homes: dict | None = None) -> "Layout":
        """A layout from :meth:`to_json`. Views not accepted by ``known(view)`` (a removed plugin,
        a deleted cell) are dropped, each id is kept once, and views of ``homes`` ({view: area})
        that are missing go back to the first group of their area."""
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise ValueError("not a version-2 layout")
        seen: set[str] = set()
        areas = {}
        for a in AREAS:
            node = _load(data.get(a), known, seen)
            areas[a] = node if node is not None else Group()
        lay = cls(areas)
        lay.normalize()
        for view, area in (homes or {}).items():
            if lay.find(view) is None:
                g = lay.groups(area)[0]
                g.views.append(view)
        return lay


def _walk(node, out):
    if isinstance(node, Group):
        out.append(node)
    else:
        for c in node.children:
            _walk(c, out)


def _parent(node, target):
    if isinstance(node, Split):
        for i, c in enumerate(node.children):
            if c is target:
                return node, i
            found = _parent(c, target)
            if found:
                return found
    return None


def _edge_group(node, orient, first):
    while isinstance(node, Split):
        node = node.children[0 if first or node.orient != orient else -1]
    return node


def _prune(node):
    if isinstance(node, Group):
        return node if node.views else None
    kids, sizes = [], []
    for c, s in zip(node.children, node.sizes or [1] * len(node.children)):
        c = _prune(c)
        if c is None:
            continue
        if isinstance(c, Split) and c.orient == node.orient:  # flatten same-direction nesting
            total = sum(c.sizes) or 1
            kids += c.children
            sizes += [s * x / total for x in c.sizes]
        else:
            kids.append(c)
            sizes.append(s)
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    node.children, node.sizes = kids, sizes
    return node


def _dump(node):
    if isinstance(node, Group):
        return {"group": list(node.views), "current": node.current}
    return {"split": node.orient, "sizes": [round(float(s), 4) for s in node.sizes],
            "children": [_dump(c) for c in node.children]}


def _load(d, known, seen):
    if not isinstance(d, dict):
        return None
    if "group" in d:
        views = []
        for v in d["group"]:
            if isinstance(v, str) and v not in seen and (known is None or known(v)):
                seen.add(v)
                views.append(v)
        cur = d.get("current", 0)
        return Group(views, min(cur, len(views) - 1) if isinstance(cur, int) and views else 0)
    if d.get("split") in ("h", "v"):
        pairs = [(_load(c, known, seen), s) for c, s in zip(d.get("children", []), d.get("sizes") or [1] * 99)]
        pairs = [(c, float(s) if isinstance(s, (int, float)) and s > 0 else 1.0) for c, s in pairs if c is not None]
        if not pairs:
            return None
        return Split(d["split"], [c for c, _ in pairs], [s for _, s in pairs])
    return None
