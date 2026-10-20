"""LLM layer: the pipeline API as tools for an AI model, plus compact structure descriptors.

* :mod:`ewald.llm.describe` - structure / trajectory descriptors and event detection (no LLM needed)
* :class:`Toolset` - stateful tools over one pipeline (load, describe, modify, analyze, render, export script)
* :func:`narrate` - trajectory narrator: detected events, measured evidence, the model's account
* :class:`Agent` - tool-use loop over a Toolset with any provider
* :mod:`ewald.llm.providers` - AI providers (Claude by default, OpenAI, Gemini, OpenAI-compatible servers,
  ``@ai_provider`` plugins); :mod:`ewald.llm.chat` - one conversation in a provider's format
* :func:`api_reference` - compact Python API reference for LLMs writing ewald code (``ewald api``, llms.txt)
* :mod:`ewald.llm.mcp_server` - the same tools over MCP (``ewald mcp``; ``pip install mcp``)
"""
from .agent import DEFAULT_MODEL, Agent
from .describe import describe_frame, describe_trajectory, detect_change_points, health_check, suggest_cutoff
from .narrate import NarratedEvent, Timeline, event_details, narrate
from .reference import api_reference
from .tools import Tool, ToolResult, Toolset

__all__ = ["Agent", "DEFAULT_MODEL", "Toolset", "Tool", "ToolResult", "describe_frame",
           "describe_trajectory", "detect_change_points", "health_check", "suggest_cutoff", "narrate",
           "event_details", "Timeline", "NarratedEvent", "api_reference"]
