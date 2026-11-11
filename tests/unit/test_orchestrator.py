from pathlib import Path

from agent.config import AgentConfig
from agent.engine.react_loop import ReactLoop
from agent.orchestrator import CLIConfigView, ReactEngineAdapter
from agent.tools import ToolRegistry


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def make_config(**overrides):
    defaults = dict(model="m", temperature=0.2, max_tokens=100, timeout=30, base_url=None, access_token=None)
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_engine_adapter_wraps_final_answer():
    llm = FakeLLM([{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=ToolRegistry())
    adapter = ReactEngineAdapter(loop)

    result = adapter.run(user_prompt="hello")

    assert result.final_answer == "done"
    assert result.is_success is True
    assert result.total_iterations == 1
    assert result.total_duration_seconds >= 0


def test_engine_adapter_counts_tool_iterations():
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "noop", "arguments": {}}},
        {"final_answer": "done"},
    ])
    registry = ToolRegistry()
    from agent.tools import Tool

    class NoopTool(Tool):
        name = "noop"
        description = "does nothing"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self) -> str:
            return "ok"

    registry.register(NoopTool())
    loop = ReactLoop(llm, tool_registry=registry)
    adapter = ReactEngineAdapter(loop)

    result = adapter.run(user_prompt="do the thing")

    assert result.total_iterations == 2


def test_engine_adapter_marks_error_as_failure():
    # No registry at all is one of the few states that still ends a run with
    # an "Error..." answer (a merely unknown tool no longer does - the model
    # is told and can retry).
    llm = FakeLLM([{"tool_call": {"id": "call_1", "name": "anything", "arguments": {}}}])
    loop = ReactLoop(llm, tool_registry=None)
    adapter = ReactEngineAdapter(loop)

    result = adapter.run(user_prompt="do the thing")

    assert result.is_success is False
    assert result.final_answer.startswith("Error")


def test_engine_adapter_marks_max_steps_as_failure():
    llm = FakeLLM([{"tool_call": {"id": "call_1", "name": "missing", "arguments": {}}}] * 3)
    loop = ReactLoop(llm, tool_registry=ToolRegistry(), max_steps=1)
    adapter = ReactEngineAdapter(loop)

    result = adapter.run(user_prompt="loop")

    assert result.is_success is False


def test_engine_adapter_context_percent_reads_from_the_llm():
    llm = FakeLLM([{"final_answer": "done"}])
    llm.context_usage_percent = lambda: 42.5
    loop = ReactLoop(llm, tool_registry=ToolRegistry())
    adapter = ReactEngineAdapter(loop)

    assert adapter.context_percent() == 42.5


def test_engine_adapter_context_percent_defaults_to_zero_without_support():
    llm = FakeLLM([{"final_answer": "done"}])  # no context_usage_percent method
    loop = ReactLoop(llm, tool_registry=ToolRegistry())
    adapter = ReactEngineAdapter(loop)

    assert adapter.context_percent() == 0.0


def test_engine_adapter_forwards_on_compaction_to_the_loop():
    llm = FakeLLM([{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=ToolRegistry())
    adapter = ReactEngineAdapter(loop)

    received = []
    adapter.run(user_prompt="hello", on_compaction=lambda *a: received.append(a))

    # no compaction happened (fresh FakeLLM has no usage tracking), but the
    # callback must reach ReactLoop.run() without erroring
    assert received == []


def test_cli_config_view_exposes_nested_attributes(tmp_path):
    config = make_config(model="my-model", temperature=0.4, timeout=45)
    view = CLIConfigView(config, workspace_root=tmp_path, max_iterations=7)

    assert view.llm.model == "my-model"
    assert view.llm.temperature == 0.4
    assert view.llm.timeout_seconds == 45
    assert view.agent.max_iterations == 7
    assert view.workspace.absolute_root == tmp_path
