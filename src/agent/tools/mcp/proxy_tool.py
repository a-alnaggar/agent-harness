"""Wraps one remote MCP tool as a local Tool, so ToolRegistry treats a
discovered MCP tool identically to a builtin one."""

from __future__ import annotations

from typing import Any

from ..base import Tool
from ..permissions import PermissionTier
from .client import McpClient


# Workaround for vLLM backends:
# Third-party MCP servers (like Notion) may emit non-standard JSON schema fields like
# 'format: json'. vLLM's grammar parser strictly checks schema formats and raises
# 'Grammar error: Unknown format: json' (HTTP 400). Stripping it keeps schemas valid.
def _sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _sanitize_schema(v) for k, v in schema.items() if not (k == "format" and v == "json")}
    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]
    return schema


class McpToolProxy(Tool):
    origin = "mcp"

    def __init__(
        self,
        client: McpClient,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        permission_tier: PermissionTier = PermissionTier.STRICT,
    ):
        self.name = name
        self.description = description
        self.parameters = _sanitize_schema(input_schema)
        self.permission_tier = permission_tier
        self._client = client

    def run(self, **kwargs: Any) -> str:
        return self._client.call_tool(self.name, kwargs)

