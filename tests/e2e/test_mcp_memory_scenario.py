"""End-to-end scenario combining both capabilities: an MCP-discovered
tool (Gmail, over a real stdio subprocess) and long-term memory that
survives an application restart — simulated here by building a second
ReactLoop from a fresh SqliteMemoryStore pointed at the same file, the same
way __main__.py would after a real restart."""

import sys

import pytest

pytest.importorskip("mcp")

from agent.config import McpServerConfig  # noqa: E402
from agent.engine.react_loop import ReactLoop  # noqa: E402
from agent.memory import SqliteMemoryStore  # noqa: E402
from agent.memory.tools import ForgetMemoryTool, RecallMemoryTool, SaveMemoryTool  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.mcp import register_mcp_tools  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def approve(tool_name, arguments, description):
    """These scenarios exercise real tools that need a yes; this is the user saying yes."""
    return "once"


def build_registry(memory_store):
    registry = ToolRegistry()
    registry.register(SaveMemoryTool(memory_store))
    registry.register(RecallMemoryTool(memory_store))
    registry.register(ForgetMemoryTool(memory_store))
    servers = [McpServerConfig(name="gmail", command=sys.executable, args=["-m", "agent.mcp_servers.gmail"])]
    clients = register_mcp_tools(registry, servers)
    return registry, clients


def test_mcp_tool_and_memory_together_survive_a_restart(tmp_path):
    memory_path = tmp_path / "memory.db"

    # --- "Session 1": search email (MCP), then remember the outcome ---
    store_1 = SqliteMemoryStore(memory_path)
    registry_1, clients_1 = build_registry(store_1)
    try:
        llm = FakeLLM([
            {"tool_call": {"id": "call_1", "name": "search_emails", "arguments": {"query": "is:unread"}}},
            {"tool_call": {"id": "call_2", "name": "save_memory",
                           "arguments": {"key": "gmail_status", "value": "not configured yet"}}},
            {"final_answer": "noted"},
        ])
        loop_1 = ReactLoop(llm, tool_registry=registry_1, max_steps=5, memory_store=store_1)
        result_1 = loop_1.run("check my email and remember the result", confirm_callback=approve)

        assert result_1 == "noted"
        tool_messages = [m for m in loop_1.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        assert "Gmail authorization unavailable" in tool_messages[0]["content"]
        assert tool_messages[1]["content"] == "Saved to memory: gmail_status = not configured yet"
        assert store_1.get("gmail_status") == "not configured yet"
    finally:
        for client in clients_1:
            client.close()

    # --- "Session 2": a fresh store retrieves the persisted general memory
    # through the SQLite-backed recall tool ---
    store_2 = SqliteMemoryStore(memory_path)
    registry_2, clients_2 = build_registry(store_2)
    try:
        llm = FakeLLM([
            {"tool_call": {"id": "call_recall", "name": "recall_memory", "arguments": {"query": "gmail_status"}}},
            {"final_answer": "I remember: gmail_status = not configured yet"},
        ])
        loop_2 = ReactLoop(llm, tool_registry=registry_2, max_steps=3, memory_store=store_2)
        result_2 = loop_2.run("what do you remember about gmail?", confirm_callback=approve)

        recall_messages = [message["content"] for message in loop_2.messages if message.get("role") == "tool"]
        assert any("gmail_status" in message and "not configured yet" in message for message in recall_messages)
        assert result_2 == "I remember: gmail_status = not configured yet"
    finally:
        for client in clients_2:
            client.close()
