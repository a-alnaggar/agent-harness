"""Connects to each configured MCP server and registers its discovered
tools into the harness's ToolRegistry, alongside the builtin tools."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..permissions import PermissionTier
from ..registry import ToolRegistry
from .client import McpClient, McpConnectionError
from .proxy_tool import McpToolProxy

if TYPE_CHECKING:
    from agent.config import McpServerConfig


def register_mcp_tools(registry: ToolRegistry, servers: list["McpServerConfig"]) -> list[McpClient]:
    """Connect to every configured MCP server and register its tools. A
    server that fails to connect is skipped with a warning — one broken
    server must not prevent the agent from starting."""
    clients: list[McpClient] = []
    for server in servers:
        client = McpClient(server.name, server.command, server.args, env=server.env)
        try:
            tools = client.connect()
        except McpConnectionError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            continue

        allowed = set(server.tools) if server.tools is not None else None
        auto_tools = set(server.auto_tools) if server.auto_tools else set()
        confirm_tools = set(server.confirm_tools) if server.confirm_tools else set()
        for tool in tools:
            if allowed is not None and tool["name"] not in allowed:
                continue
            if tool["name"] in auto_tools:
                tier = PermissionTier.REGULAR
            elif tool["name"] in confirm_tools:
                tier = PermissionTier.CONFIRM
            else:
                tier = PermissionTier.STRICT
            registry.register(McpToolProxy(
                client, tool["name"], tool["description"], tool["input_schema"], permission_tier=tier,
            ))
        clients.append(client)
    return clients
