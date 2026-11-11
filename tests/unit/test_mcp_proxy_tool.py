import pytest

pytest.importorskip("mcp")

from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.mcp.proxy_tool import McpToolProxy  # noqa: E402
from agent.tools.permissions import PermissionTier  # noqa: E402


class FakeMcpClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"called {name} with {arguments}"


def test_proxy_exposes_schema_from_discovery():
    client = FakeMcpClient()
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    tool = McpToolProxy(client, "search", "search something", schema)

    assert tool.to_schema() == {
        "type": "function",
        "function": {"name": "search", "description": "search something", "parameters": schema},
    }


def test_proxy_run_delegates_to_client():
    client = FakeMcpClient()
    tool = McpToolProxy(client, "search", "search something", {})

    result = tool.run(q="passau")

    assert result == "called search with {'q': 'passau'}"
    assert client.calls == [("search", {"q": "passau"})]


def test_proxy_registers_into_tool_registry_like_a_builtin_tool():
    client = FakeMcpClient()
    registry = ToolRegistry()
    registry.register(McpToolProxy(client, "search", "search something", {}))

    result = registry.call("search", q="passau")

    assert result == "called search with {'q': 'passau'}"


def test_proxy_defaults_to_strict_permission_tier():
    """Third-party MCP tools never earn a lighter tier just by being
    present - only via explicit config (see registration.py)."""
    client = FakeMcpClient()
    tool = McpToolProxy(client, "search", "search something", {})

    assert tool.permission_tier == PermissionTier.STRICT


def test_proxy_accepts_an_explicit_permission_tier():
    client = FakeMcpClient()
    tool = McpToolProxy(client, "search", "search something", {}, permission_tier=PermissionTier.REGULAR)

    assert tool.permission_tier == PermissionTier.REGULAR


def test_proxy_sanitizes_non_standard_json_format():
    client = FakeMcpClient()
    raw_schema = {
        "type": "object",
        "properties": {
            "icon": {"type": "string", "format": "json"},
            "title": {"type": "string"},
        },
    }
    tool = McpToolProxy(client, "create_page", "create a page", raw_schema)

    assert tool.parameters == {
        "type": "object",
        "properties": {
            "icon": {"type": "string"},
            "title": {"type": "string"},
        },
    }

