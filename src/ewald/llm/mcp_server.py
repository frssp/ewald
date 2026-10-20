"""Expose an ewald :class:`~ewald.llm.Toolset` as an MCP server (stdio).

Lets Claude Desktop, Claude Code or any MCP client drive ewald directly::

    ewald mcp                      # or: python -m ewald.llm.mcp_server
    claude mcp add ewald -- ewald mcp
    claude mcp add ewald -- ewald mcp --allow-code   # also let it run cells it writes (run_cell)

Needs ``pip install mcp``.
"""
from __future__ import annotations

import base64

from .tools import Toolset


def list_tools(toolset: Toolset):
    from mcp import types

    return [types.Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
            for t in toolset.tools.values()]


def call_tool(toolset: Toolset, name: str, arguments: dict | None, result=None):
    from mcp import types

    result = result if result is not None else toolset.call(name, arguments or {})
    text = result.text()
    out = [types.TextContent(type="text", text=("ERROR: " + text) if result.is_error else text)]
    out += [types.ImageContent(type="image", data=base64.standard_b64encode(png).decode(), mimeType="image/png")
            for png in result.images]
    return out


def build_server(toolset: Toolset | None = None):
    """Low-level MCP server over ``toolset``; works with mcp 1.x (decorators) and 2.x (handlers)."""
    from mcp import types
    from mcp.server.lowlevel import Server

    toolset = toolset or Toolset()
    if hasattr(Server, "list_tools"):  # mcp 1.x
        server = Server("ewald")

        @server.list_tools()
        async def _list_tools():
            return list_tools(toolset)

        @server.call_tool()
        async def _call_tool(name, arguments):
            return call_tool(toolset, name, arguments)
    else:  # mcp 2.x
        async def _list_tools(ctx, params):
            return types.ListToolsResult(tools=list_tools(toolset))

        async def _call_tool(ctx, params):
            result = toolset.call(params.name, params.arguments or {})
            return types.CallToolResult(content=call_tool(toolset, params.name, None, result),
                                        is_error=result.is_error)

        server = Server("ewald", on_list_tools=_list_tools, on_call_tool=_call_tool)
    server.toolset = toolset
    return server


def serve(allow_code: bool = False, path: str | None = None):
    import anyio
    from mcp.server.stdio import stdio_server

    toolset = Toolset(allow_code=allow_code)
    if path:
        toolset.call("load_structure", {"path": path})
    server = build_server(toolset)

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ewald MCP server (stdio)")
    ap.add_argument("path", nargs="?", help="structure/trajectory to load at start")
    ap.add_argument("--allow-code", action="store_true", help="offer run_cell: the AI runs notebook cells it wrote")
    a = ap.parse_args()
    serve(a.allow_code, a.path)
