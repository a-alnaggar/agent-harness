"""End-to-end: a genuinely broken MCP server (bad command, real subprocess
spawn failure - not mocked) must not take down the rest of the agent.
Builtin tools must still work normally in the same registry."""

from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent.config import McpServerConfig  # noqa: E402
from agent.engine.react_loop import ReactLoop  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from agent.tools.builtin import register_builtin_tools  # noqa: E402
from agent.tools.mcp import register_mcp_tools  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def test_broken_mcp_server_is_skipped_and_builtin_tools_still_work(tmp_path, capsys):
    registry = ToolRegistry()
    register_builtin_tools(registry, tmp_path)

    broken_server = McpServerConfig(name="broken", command="this-command-does-not-exist", args=[])
    clients = register_mcp_tools(registry, [broken_server])

    # the broken server produced no client and no registered tools
    assert clients == []
    assert not registry.has_tool("search_emails")

    # a warning was printed rather than an exception propagating
    assert "broken" in capsys.readouterr().err

    # builtin tools registered before the broken server are untouched
    assert registry.has_tool("create_file")

    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "create_file",
                       "arguments": {"path": "note.txt", "content": "still works"}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry, max_steps=3)

    result = loop.run("create note.txt")

    assert result == "done"
    assert (tmp_path / "note.txt").read_text() == "still works"
