"""Trajectory narrator: find what happens in a run and say it in words.

Three steps, only the last needs an LLM:

1. :func:`~ewald.llm.describe.describe_trajectory` flags frames where a
   descriptor shifts abruptly (events).
2. :func:`event_details` measures each event: which atoms moved, how far, how
   their coordination changed, and whether the change is local or global.
3. :func:`narrate` hands the numbers (and optionally renders of the frames
   around each event) to Claude, which returns a title and a short description
   per event plus a one-paragraph summary of the run. Without a client it
   writes plain rule-based titles instead.

>>> timeline = narrate(import_file("dump.lammpstrj", type_map={1: "Cu"}))
>>> print(timeline.to_markdown())
>>> timeline.bookmarks()          # [(frame, title), ...] for a GUI timeline
"""
from __future__ import annotations

import base64
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..neighbors import find_pairs
from .describe import _round, describe_trajectory, suggest_cutoff

NARRATOR_PROMPT = """\
You narrate atomistic simulation trajectories for a computational materials \
scientist. You get automatically detected events: frames where a descriptor \
jumped, with measured evidence (which atoms moved, displacement, coordination \
before/after, whether the change is local or system-wide) and possibly images \
of the frames before and after.

For each event, give a short title (under 12 words) and two or three sentences \
saying what physically happens, using only the numbers given. Name the likely \
process when the evidence supports it (melting, vacancy or interstitial hop, \
surface diffusion, dislocation or twin nucleation, phase transition, bond \
breaking, an unphysical jump or overlap pointing to a simulation problem) and \
say how confident you are. Then write a summary of the whole run in at most \
four sentences. Do not invent events or frame numbers that are not in the input."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {"frame": {"type": "integer"}, "title": {"type": "string"},
                           "description": {"type": "string"},
                           "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
            "required": ["frame", "title", "description", "confidence"],
            "additionalProperties": False}},
    },
    "required": ["summary", "events"],
    "additionalProperties": False,
}


@dataclass
class NarratedEvent:
    frame: int
    title: str
    description: str = ""
    confidence: str = "low"
    evidence: dict = field(default_factory=dict)


@dataclass
class Timeline:
    summary: str
    events: list[NarratedEvent]
    trajectory: dict
    narrated_by: str = "rules"

    def bookmarks(self) -> list[tuple[int, str]]:
        return [(e.frame, e.title) for e in self.events]

    def to_markdown(self) -> str:
        lines = ["# Trajectory timeline", "", self.summary, ""]
        for e in self.events:
            lines += [f"## Frame {e.frame}: {e.title}", ""]
            if e.description:
                lines += [e.description, ""]
            ev = e.evidence
            if ev:
                lines.append(f"*Evidence: frames {ev['frames'][0]} -> {ev['frames'][1]}, "
                             f"{ev['moved']['count']} atoms moved > {ev['moved']['threshold']} A "
                             f"({ev['extent']}); confidence {e.confidence}.*")
                lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "narrated_by": self.narrated_by,
                "events": [vars(e) for e in self.events], "trajectory": self.trajectory}


# ------------------------------------------------------------------ evidence

def _matched_positions(data):
    pos = np.asarray(data.positions, float)
    types = np.asarray(data.types)
    ids = np.asarray(data["Particle Identifier"]) if "Particle Identifier" in data else np.arange(data.count)
    order = np.argsort(ids, kind="stable")
    return pos[order], types[order], ids[order], order


def _coordination(data, cutoff):
    i, j, _, _ = find_pairs(data.positions, data.cell, cutoff, half=True)
    return np.bincount(np.asarray(i), minlength=data.count) + np.bincount(np.asarray(j), minlength=data.count)


def event_details(pipeline, frame: int, window: int = 5, cutoff: float | None = None,
                  threshold: float | None = None, top: int = 8) -> dict:
    """Measure what changes across an event at ``frame``.

    Compares source frames ``frame - window`` and ``frame + window - 1``:
    displacements (minimum image, matched by identifier), the atoms that moved
    more than ``threshold`` (default: half the coordination cutoff) by type,
    their centroid and spread (local vs global), and coordination changes.
    """
    n = pipeline.num_frames
    fb, fa = max(0, frame - window), min(n - 1, frame + window - 1)
    before, after = pipeline.source_frame(fb), pipeline.source_frame(fa)
    if cutoff is None:
        cutoff = suggest_cutoff(before) or 3.0
    threshold = 0.5 * cutoff if threshold is None else threshold
    pb, tb, ids, ob = _matched_positions(before)
    pa, _, ids_a, oa = _matched_positions(after)
    if len(pb) != len(pa) or np.any(ids != ids_a):
        return {"frames": [fb, fa], "error": "particle count or identifiers change across the event"}
    cell = after.cell
    dr = cell.minimum_image(pa - pb)
    d = np.linalg.norm(dr, axis=1)
    cnb, cna = _coordination(before, cutoff)[ob], _coordination(after, cutoff)[oa]
    moved = d > threshold
    out = {"frames": [fb, fa], "cutoff": _round(cutoff), "displacement": {
        "mean": _round(d.mean()), "max": _round(d.max()), "p95": _round(np.percentile(d, 95))},
        "moved": {"threshold": _round(threshold), "count": int(moved.sum()), "fraction": _round(moved.mean()),
                  "by_type": dict(Counter(str(t) for t in tb[moved]))}}
    if moved.any():
        centre = pb[moved].mean(0)
        rel = cell.minimum_image(pb[moved] - centre)
        rg = float(np.sqrt((rel ** 2).sum(1).mean()))
        box = float(np.min(cell.perpendicular_widths())) if cell.is_valid else float(np.ptp(pb, 0).min())
        out["moved"]["centroid"] = _round(centre.tolist())
        out["moved"]["radius_of_gyration"] = _round(rg)
        mean_dir = dr[moved].mean(0)
        out["moved"]["mean_displacement_vector"] = _round(mean_dir.tolist())
        local = moved.mean() < 0.1 and rg < 0.25 * box
        out["extent"] = "local" if local else "system-wide"
    else:
        out["extent"] = "no atom moved beyond threshold (collective or thermal change)"
    out["coordination"] = {"mean_before": _round(cnb.mean()), "mean_after": _round(cna.mean()),
                           "changed": int((cnb != cna).sum()),
                           "transitions": {f"{a}->{b}": c for (a, b), c in
                                           Counter(zip(cnb[cnb != cna].tolist(), cna[cnb != cna].tolist())).most_common(6)}}
    k = np.argsort(-d)[:top]
    out["top_movers"] = [{"id": int(ids[i]), "type": str(tb[i]), "displacement": _round(d[i]),
                          "from": _round(pb[i].tolist(), 3), "to": _round(pa[i].tolist(), 3),
                          "cn": [int(cnb[i]), int(cna[i])]} for i in k]
    return out


def _rule_title(event: dict, ev: dict) -> str:
    parts = []
    ch = event["changes"]
    for key, label, unit in (("max_displacement", "max displacement", " A"), ("mean_coordination", "mean CN", ""),
                             ("min_distance", "min distance", " A"), ("volume", "volume", " A^3")):
        if key in ch:
            parts.append(f"{label} {ch[key]['before']} -> {ch[key]['after']}{unit}")
    if ev.get("moved", {}).get("count"):
        parts.append(f"{ev['moved']['count']} atoms moved ({ev['extent']})")
    return "; ".join(parts[:3]) or "descriptor change"


# ------------------------------------------------------------------ narrate

def narrate(pipeline, client=None, *, model: str | None = None, effort: str = "medium",
            render: bool = False, max_events: int = 8, window: int | None = None,
            fallbacks: bool = True, **describe_kwargs) -> Timeline:
    """Detect, measure and (with ``client``) describe the events of a trajectory.

    ``client`` is an ``anthropic.Anthropic()`` (or a stand-in with the same
    ``beta.messages.create``); without one the titles are rule-based. With
    ``render=True`` the frames before/after the first events are rendered and
    shown to the model. Frame numbers in the model's answer are checked
    against the detected events; anything else is dropped.
    """
    from .agent import DEFAULT_MODEL

    traj = describe_trajectory(pipeline, **describe_kwargs)
    events = traj["events"][:max_events]
    step = traj["frame_step"]
    w = window or max(1, 3 * step)
    cutoff = traj.get("coordination_cutoff")
    details = {e["frame"]: event_details(pipeline, e["frame"], w, cutoff) for e in events}
    narrated = [NarratedEvent(e["frame"], _rule_title(e, details[e["frame"]]), evidence=details[e["frame"]])
                for e in events]
    n = traj["num_frames"]
    summary = (f"{n} frames; {len(narrated)} event(s) detected"
               + (f" at frame(s) {', '.join(str(e.frame) for e in narrated)}." if narrated else "."))
    timeline = Timeline(summary, narrated, traj)
    if client is None or not events:
        return timeline

    content: list[dict] = []
    if render:
        content += _render_blocks(pipeline, narrated[:3])
    payload = {"trajectory": {k: traj[k] for k in ("num_frames", "frame_step", "coordination_cutoff",
                                                   "series", "first_frame_warnings")},
               "events": [{"frame": e["frame"], "descriptor_changes": e["changes"], "evidence": details[e["frame"]]}
                          for e in events]}
    content.append({"type": "text", "text": json.dumps(payload)})
    params = {"model": model or DEFAULT_MODEL, "max_tokens": 16000, "system": NARRATOR_PROMPT,
              "messages": [{"role": "user", "content": content}],
              "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": _SCHEMA}}}
    if fallbacks:
        params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    response = client.beta.messages.create(**params)
    if response.stop_reason == "refusal":
        return timeline
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return timeline
    by_frame = {e.frame: e for e in narrated}
    for item in answer.get("events", []):
        e = by_frame.get(int(item.get("frame", -1)))
        if e is None:  # not a detected event: ignore rather than trust it
            continue
        e.title = item.get("title") or e.title
        e.description = item.get("description", "")
        e.confidence = item.get("confidence", "low")
    if answer.get("summary"):
        timeline.summary = answer["summary"]
    timeline.narrated_by = params["model"]
    return timeline


def _render_blocks(pipeline, events) -> list[dict]:
    import os
    import tempfile

    from . import tools

    blocks = []
    with tempfile.TemporaryDirectory() as tmp:
        for e in events:
            for label, f in zip(("before", "after"), e.evidence.get("frames", [])):
                path = os.path.join(tmp, f"{f}.png")
                try:
                    tools.render_png(pipeline.compute(f), path, size=(800, 600), direction="iso")
                    with open(path, "rb") as fh:
                        data = base64.standard_b64encode(fh.read()).decode()
                except Exception:
                    continue
                blocks += [{"type": "text", "text": f"Event at frame {e.frame}, {label}: frame {f}"},
                           {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]
    return blocks
