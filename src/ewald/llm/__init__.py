"""LLM layer: the pipeline API as tools for Claude, plus compact structure descriptors.

* :mod:`ewald.llm.describe` - structure / trajectory descriptors and event detection (no LLM needed)
* :class:`Toolset` - stateful tools over one pipeline (load, describe, modify, analyze, render, export script)
* :func:`narrate` - trajectory narrator: detected events, measured evidence, Claude's account
* :class:`Agent` - Claude tool-use loop over a Toolset (``pip install anthropic``)
* :mod:`ewald.llm.mcp_server` - the same tools over MCP (``ewald mcp``; ``pip install mcp``)
"""
from .agent import DEFAULT_MODEL, Agent
from .describe import describe_frame, describe_trajectory, detect_change_points, health_check, suggest_cutoff
from .narrate import NarratedEvent, Timeline, event_details, narrate
from .tools import Tool, ToolResult, Toolset

__all__ = ["Agent", "DEFAULT_MODEL", "Toolset", "Tool", "ToolResult", "describe_frame",
           "describe_trajectory", "detect_change_points", "health_check", "suggest_cutoff", "narrate",
           "event_details", "Timeline", "NarratedEvent"]
