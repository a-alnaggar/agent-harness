"""End-to-end: runtime MCP tool discovery, invocation, and result handling,
through the full harness stack (ToolRegistry -> ReactLoop), against the
real Gmail MCP server subprocess. Stays offline: discovery needs no
credentials, and the controlled failure path (no credentials configured)
is itself a legitimate result to hand back to the LLM."""

import sys

import pytest

pytest.importorskip("mcp")

from agent.config import McpServerConfig  # noqa: E402
from agent.engine.react_loop import ReactLoop  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.mcp import register_mcp_tools  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def test_mcp_tool_discovered_invoked_and_result_fed_back_to_the_loop():
    registry = ToolRegistry()
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"])]
    clients = register_mcp_tools(registry, servers)

    try:
        # discovery: the loop's tool schema for the LLM includes the MCP tool
        schema_names = {s["function"]["name"] for s in registry.schemas()}
        assert "search_emails" in schema_names

        # invocation + result handling: the loop calls it and continues
        llm = FakeLLM([
            {"tool_call": {"id": "call_1", "name": "search_emails", "arguments": {"query": "is:unread"}}},
            {"final_answer": "done"},
        ])
        loop = ReactLoop(llm, tool_registry=registry, max_steps=5)

        result = loop.run("search my unread emails", confirm_callback=lambda *_: "once")

        assert result == "done"
        tool_message = next(m for m in loop.messages if m.get("role") == "tool")
        assert "Gmail authorization unavailable" in tool_message["content"]
    finally:
        for client in clients:
            client.close()
