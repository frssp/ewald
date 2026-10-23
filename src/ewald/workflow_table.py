"""The numbers of a workflow's steps as one table, and of several workflows side by side.

Every :class:`~ewald.workflow.WorkflowStep` carries ``scalars`` (energy, energy/atom, max force,
gap, barrier, ...: whatever its source read). :func:`workflow_table` lines them up, a row per
step and a column per number, so a study reads at a glance and two studies (another functional,
a bigger cell, a different dopant) compare step by step::

    from ewald import workflow_table
    workflow_table("~/calc/li_pbe")                       # one workflow: its steps, their numbers
    t = workflow_table("~/calc/li_pbe", "~/calc/li_r2scan", labels=["PBE", "r2SCAN"])
    t                                                     # a column per number and workflow
    t.difference("PBE")                                   # every workflow minus PBE
    t["energy/atom"]                                      # {"PBE": array, "r2SCAN": array}, one value per step
    t.plot("energy/atom")                                 # the steps on x, a line per workflow
    t.to_csv("compare.csv")                               # pandas.read_csv reads it back

Arguments are :class:`~ewald.workflow.Workflow` objects or anything
:func:`ewald.workflow.describe` takes (a folder of calculations, ``host:/path``, ``aiida:1234``).
Rows are the steps that did a calculation (a group step stands for its steps, a restarted run
is one row), named by their path in the tree (``bands / scf``); several workflows are matched
by those names, or by position with ``align="position"`` when the folders are named differently.
"""
from __future__ import annotations

import html
import math
import re

import numpy as np

from .workflow import Workflow, WorkflowStep

__all__ = ["WorkflowTable", "workflow_table", "step_rows"]

#: numbers shown first, in this order; the others follow in the order the steps have them
ORDER = ("energy", "energy/atom", "barrier", "gap", "max force", "fermi energy", "atoms", "ionic steps",
         "images", "displacements")
#: words shown in a table of one workflow (several workflows show the numbers only)
WORDS = ("code", "kind")
#: units of the numbers ewald's own sources write, for column headers
UNITS = {"energy": "eV", "energy/atom": "eV", "barrier": "eV", "gap": "eV", "max force": "eV/Å",
         "fermi energy": "eV"}
_ORDER_PREFIX = re.compile(r"^\d+[_.-]+(?=\D)")


def _number(value):
    """``value`` as a float when it is a number (bools are not), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def step_rows(wf: WorkflowStep, prefix: str = "") -> list:
    """``(name, step)`` for the steps of ``wf`` that are rows of its table: steps without
    sub-steps and restarted runs (one row), in tree order; group steps add their name in front
    (``bands / scf``). A workflow without steps is its own one row."""
    out = []
    for s in wf.steps:
        name = f"{prefix} / {s.name}" if prefix else s.name
        if s.steps and s.runs <= 1:
            out += step_rows(s, name)
        else:
            out.append((name, s))
    if not out and not prefix and isinstance(wf, Workflow):
        out.append((wf.name or wf.label or "result", wf))
    return out


def _key(name: str) -> str:
    """What matches a step across workflows: its name without case and leading order numbers
    (``01_relax`` and ``1-Relax`` are the same step)."""
    return " / ".join(_ORDER_PREFIX.sub("", part.strip()).lower() for part in name.split(" / "))


def _format(value) -> str:
    v = _number(value)
    if v is None:
        return "" if value is None else " ".join(str(value).split())
    if not math.isfinite(v):
        return ""
    if float(v).is_integer() and abs(v) < 1e15 and not isinstance(value, (float, np.floating)):
        return str(int(v))
    if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-3):
        return f"{v:.4e}"
    return f"{v:.4f}"


class WorkflowTable:
    """Rows of steps, columns of numbers, one block per workflow (:func:`workflow_table`).

    ``labels`` name the workflows, ``rows`` the steps, ``columns`` the quantities; ``cell(row,
    column, label)`` is one value (None where that step has none), ``state(row, label)`` the
    step's mark (✓ ✗ …). Indexing by a quantity gives ``{label: array}`` with NaN for missing."""

    def __init__(self, labels, rows, columns, values, states, paths, words=(), title: str = ""):
        self.labels, self.rows, self.columns = list(labels), list(rows), list(columns)
        self.words, self.title = list(words), title
        self._values, self._states, self._paths = values, states, paths

    # ------------------------------------------------------------ access
    def cell(self, row: str, column: str, label: str | None = None):
        label = self.labels[0] if label is None else label
        return self._values.get(label, {}).get(row, {}).get(column)

    def state(self, row: str, label: str | None = None) -> str:
        label = self.labels[0] if label is None else label
        return self._states.get(label, {}).get(row, "")

    def path(self, row: str, label: str | None = None) -> str:
        """What ``load()`` opens for that step ("" when it has no result)."""
        label = self.labels[0] if label is None else label
        return self._paths.get(label, {}).get(row, "")

    def __getitem__(self, column: str) -> dict:
        if column not in self.columns and column not in self.words:
            raise KeyError(f"{column!r}: the table has {', '.join(self.columns + self.words) or 'no columns'}")
        out = {}
        for label in self.labels:
            got = [self.cell(r, column, label) for r in self.rows]
            nums = [_number(v) for v in got]
            out[label] = (np.array([np.nan if n is None else n for n in nums])
                          if column in self.columns else np.array([v or "" for v in got], dtype=object))
        return out

    def __len__(self):
        return len(self.rows)

    def __contains__(self, column):
        return column in self.columns or column in self.words

    def difference(self, reference: str | int = 0) -> WorkflowTable:
        """Every workflow's numbers minus those of ``reference`` (a label or a position), step by
        step; the reference itself is left out. Counts (atoms, ionic steps) are differences too."""
        ref = self.labels[reference] if isinstance(reference, int) else reference
        if ref not in self.labels:
            raise KeyError(f"{reference!r}: the workflows are {', '.join(self.labels)}")
        if len(self.labels) < 2:
            raise ValueError("a difference needs at least two workflows")
        labels = [lab for lab in self.labels if lab != ref]
        values = {}
        for lab in labels:
            values[f"{lab} − {ref}"] = {r: self._minus(r, lab, ref) for r in self.rows}
        names = list(values)
        return WorkflowTable(names, self.rows, self.columns, values,
                             {n: self._states.get(lab, {}) for n, lab in zip(names, labels)},
                             {n: self._paths.get(lab, {}) for n, lab in zip(names, labels)},
                             title=f"Difference to {ref}")

    def _minus(self, row: str, label: str, ref: str) -> dict:
        out = {}
        for c in self.columns:
            a, b = self.cell(row, c, label), self.cell(row, c, ref)
            if _number(a) is None or _number(b) is None:
                continue
            both_int = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (a, b))
            out[c] = int(a) - int(b) if both_int else _number(a) - _number(b)
        return out

    # ------------------------------------------------------------ out
    def header(self, column: str) -> str:
        unit = UNITS.get(column)
        return f"{column} ({unit})" if unit else column

    def _layout(self):
        """``(heads, body)``: the column titles and the text rows of the table."""
        several = len(self.labels) > 1
        heads, getters = ["step"], []
        if several:
            for lab in self.labels:
                heads.append(f"state {lab}")
                getters.append(lambda r, lab=lab: self.state(r, lab))
            for col in self.columns:
                for lab in self.labels:
                    heads.append(f"{self.header(col)} {lab}")
                    getters.append(lambda r, col=col, lab=lab: _format(self.cell(r, col, lab)))
        else:
            lab = self.labels[0] if self.labels else None
            heads.append("state")
            getters.append(lambda r: self.state(r, lab))
            for col in self.words + self.columns:
                heads.append(self.header(col))
                getters.append(lambda r, col=col: _format(self.cell(r, col, lab)))
        body = [[" ".join(str(r).split())] + [g(r) for g in getters] for r in self.rows]
        return [" ".join(h.split()) for h in heads], body

    def as_dict(self) -> dict:
        """Plain data (what the AI reads): workflows, steps, quantities and the values."""
        return {"workflows": self.labels, "steps": self.rows, "columns": self.words + self.columns,
                "values": {lab: {r: dict(self._values.get(lab, {}).get(r, {})) for r in self.rows}
                           for lab in self.labels},
                "states": {lab: dict(self._states.get(lab, {})) for lab in self.labels}}

    def to_csv(self, path) -> None:
        """The table as CSV (UTF-8), the same columns as the cell shows."""
        import csv

        heads, body = self._layout()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(heads)
            w.writerows(body)

    def _repr_html_(self):
        """A table; with several workflows the quantity heads span a column per workflow."""
        heads, body = self._layout()
        esc = html.escape
        n = len(self.labels)
        words = 0 if n > 1 else 1 + len(self.words)          # left-aligned columns after "step"
        if n > 1:
            top = ("<th rowspan='2' align='left'>step</th>"
                   + f"<th colspan='{n}'>state</th>"
                   + "".join(f"<th colspan='{n}'>{esc(self.header(c))}</th>" for c in self.columns))
            sub = "".join(f"<th>{esc(lab)}</th>" for lab in self.labels) * (1 + len(self.columns))
            thead = f"<tr>{top}</tr><tr>{sub}</tr>"
            words = n
        else:
            thead = "<tr>" + "".join(f"<th align='{'left' if k <= words else 'right'}'>{esc(h)}</th>"
                                     for k, h in enumerate(heads)) + "</tr>"
        rows = "".join("<tr>" + "".join(f"<td align='{'left' if k <= words else 'right'}'>{esc(v)}</td>"
                                        for k, v in enumerate(row)) + "</tr>" for row in body)
        caption = f"<caption>{esc(' '.join(self.title.split()))}</caption>" if self.title else ""
        return (f"<table border='1' cellspacing='0' cellpadding='3'>{caption}<thead>{thead}</thead>"
                f"<tbody>{rows}</tbody></table>")

    def _repr_markdown_(self):
        heads, body = self._layout()

        def esc(text):
            return str(text).replace("|", "\\|")

        lines = ["| " + " | ".join(esc(h) for h in heads) + " |", "|" + "---|" * len(heads)]
        lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in body]
        if self.title:
            lines = [f"**{esc(' '.join(self.title.split()))}**", ""] + lines
        return "\n".join(lines)

    def __repr__(self):
        heads, body = self._layout()
        widths = [max(len(str(x)) for x in col) for col in zip(heads, *body)]
        lines = ["  ".join(str(h).ljust(w) for h, w in zip(heads, widths)).rstrip()]
        lines += ["  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip() for row in body]
        return "\n".join(([self.title] if self.title else []) + lines)

    def figure(self, column: str | None = None, size=None):
        """``column`` (default: the first number) over the steps, a line per workflow."""
        from . import plotstyle

        column = column or (self.columns[0] if self.columns else None)
        if column not in self.columns:
            raise KeyError(f"{column!r} is not a number of this table ({', '.join(self.columns) or 'none'})")
        fig = plotstyle.figure(size)
        with plotstyle.context(*fig._ewald_style):
            ax = fig.add_subplot()
            x = np.arange(len(self.rows))
            for lab, y in self[column].items():
                ok = np.isfinite(y)
                ax.plot(x[ok], y[ok], "o-", label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels(self.rows, rotation=30, ha="right")
            ax.set_ylabel(self.header(column))
            if len(self.labels) > 1:
                ax.legend(loc="best")
        return fig

    def plot(self, column: str | None = None):
        """Draw :meth:`figure` (``plt.show()``)."""
        import matplotlib.pyplot as plt

        self.figure(column)
        plt.show()


def _label(wf: Workflow, taken: set) -> str:
    base = " ".join(str(wf.label or wf.name or wf.id or "workflow").split())
    out, n = base, 2
    while out in taken:
        out, n = f"{base} ({n})", n + 1
    return out


def workflow_table(*workflows, labels=None, columns=None, align: str = "name") -> WorkflowTable:
    """The steps of one or more workflows with their numbers (see the module docstring).

    ``workflows``: :class:`~ewald.workflow.Workflow` objects or paths :func:`ewald.workflow.describe`
    reads (a list works too); ``labels`` names them (default: their labels, else names);
    ``columns`` picks and orders the quantities (default: every number any step has, the usual
    ones first); ``align`` matches steps across workflows by "name" (case and leading ``01_``
    ignored) or by "position"."""
    from . import workflow as _workflow

    if len(workflows) == 1 and isinstance(workflows[0], (list, tuple)):
        workflows = tuple(workflows[0])
    if not workflows:
        raise ValueError("workflow_table needs at least one workflow")
    if align not in ("name", "position"):
        raise ValueError(f"align must be 'name' or 'position', not {align!r}")
    wfs = [w if isinstance(w, WorkflowStep) else _workflow.describe(str(w)) for w in workflows]
    if labels is None:
        taken: set = set()
        labels = []
        for w in wfs:
            labels.append(_label(w, taken) if isinstance(w, Workflow) else w.name)
            taken.add(labels[-1])
    labels = [str(x) for x in labels]
    if len(labels) != len(wfs) or len(set(labels)) != len(labels):
        raise ValueError(f"{len(labels)} labels for {len(wfs)} workflows, and they must differ")

    rows: list = []          # shown names, in order of first appearance
    keys: dict = {}          # match key -> shown name
    values, states, paths = {}, {}, {}
    seen_numbers, seen_words = [], []
    for lab, wf in zip(labels, wfs):
        values[lab], states[lab], paths[lab] = {}, {}, {}
        for k, (name, step) in enumerate(step_rows(wf)):
            key = str(k) if align == "position" else _key(name)
            if key not in keys:
                keys[key] = name
                rows.append(name)
            row = keys[key]
            if row in values[lab]:                   # two steps of one workflow with one name
                row = f"{name} ({k + 1})"
                rows.append(row)
            values[lab][row] = dict(step.scalars)
            states[lab][row] = step.mark()
            paths[lab][row] = step.path
            for q, v in step.scalars.items():
                target = seen_numbers if _number(v) is not None else seen_words
                if q not in target:
                    target.append(q)
    if columns is None:
        numbers = [q for q in ORDER if q in seen_numbers] + [q for q in seen_numbers if q not in ORDER]
        words = [q for q in WORDS if q in seen_words] if len(wfs) == 1 else []
    else:
        numbers = [str(c) for c in columns if str(c) not in seen_words]
        words = [str(c) for c in columns if str(c) in seen_words]
    words = [w for w in words if w not in numbers]
    return WorkflowTable(labels, rows, numbers, values, states, paths, words)
