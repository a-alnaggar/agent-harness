"""Exercises the same wiring __main__.py uses (config -> tools -> engine ->
cli_interface) end to end, without hitting the network, to catch interface
mismatches between independently-built modules."""

from pathlib import Path

from agent.cli_interface import start_cli
from agent.cli_interface.repl import Prompt
from agent.config import AgentConfig
from agent.engine.react_loop import ReactLoop
from agent.orchestrator import CLIConfigView, ReactEngineAdapter
from agent.tools import ToolRegistry
from agent.tools.builtin import register_builtin_tools


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def test_demo_wiring_runs_a_full_task_then_exits(tmp_path, monkeypatch):
    inputs = iter([
        "create a file",
        "/tools",
        "/config",
        "/exit",
    ])
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: next(inputs))

    registry = ToolRegistry()
    register_builtin_tools(registry, tmp_path)

    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "create_file", "arguments": {"path": "a.txt", "content": "hi"}}},
        {"final_answer": "created it"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)
    engine = ReactEngineAdapter(loop)
    config = AgentConfig(model="m", temperature=0.2, max_tokens=100, timeout=30)
    cli_config = CLIConfigView(config, workspace_root=tmp_path, max_iterations=10)

    start_cli(config=cli_config, engine=engine, registry=registry, context_manager=None)

    assert (tmp_path / "a.txt").read_text() == "hi"
