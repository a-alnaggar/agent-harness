import pytest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

from agent.observability import LifecycleEvent, RunContext
from agent.tools.mcp.proxy_tool import McpToolProxy

from agent.tools import Tool, ToolRegistry
from agent.tools.permissions import PermissionTier


class EchoTool(Tool):
    name = "echo"
    description = "Echoes back its input."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, *, text: str) -> str:
        return text


class BoomTool(Tool):
    name = "boom"
    description = "Always fails."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs) -> str:
        from agent.tools import ToolError

        raise ToolError("boom failed")


def test_register_and_get():
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)

    assert registry.get("echo") is tool


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError):
        registry.register(EchoTool())


def test_get_missing_tool_raises():
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_list_tools():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(BoomTool())

    names = {tool.name for tool in registry.list_tools()}
    assert names == {"echo", "boom"}


def test_schemas_shape():
    registry = ToolRegistry()
    registry.register(EchoTool())

    schemas = registry.schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echoes back its input.",
                "parameters": EchoTool.parameters,
            },
        }
    ]


def test_call_runs_registered_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())

    assert registry.call("echo", text="hi") == "hi"


def test_call_missing_tool_returns_error_string_instead_of_raising():
    registry = ToolRegistry()

    result = registry.call("nope")
    assert result.startswith("Error:")


def test_call_tool_error_returns_error_string_instead_of_raising():
    registry = ToolRegistry()
    registry.register(BoomTool())

    result = registry.call("boom")
    assert result == "Error: boom failed"


def test_unregister():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.unregister("echo")

    with pytest.raises(KeyError):
        registry.get("echo")


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


@pytest.mark.parametrize("kind", ["builtin", "mcp", "controlled", "unexpected", "missing"])
def test_tool_lifecycle_is_single_correlated_and_content_free(kind: str) -> None:
    hook = RecordingHook()
    ticks = iter([10.0, 12.0])
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span",
                            clock=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
                            monotonic_clock=lambda: next(ticks))
    client = Mock()
    client.call_tool.return_value = "private-result"
    error = RuntimeError("private-error")
    class UnexpectedTool(EchoTool):
        def run(self, **kwargs: Any) -> str:
            raise error
    tool = McpToolProxy(client, "echo", "private-description", {}) if kind == "mcp" else (
        BoomTool() if kind == "controlled" else UnexpectedTool() if kind == "unexpected" else EchoTool())
    registry.register(tool)
    name = "missing" if kind == "missing" else tool.name
    parent = RunContext("run", "agent-span", None, "agent-a")
    if kind == "unexpected":
        with pytest.raises(RuntimeError) as caught:
            registry.call_with_context(name, {"text": "private-argument"}, context=parent)
        assert caught.value is error
    else:
        result = registry.call_with_context(name, {"text": "private-argument"}, context=parent)
        assert result == ("private-result" if kind == "mcp" else "Error: boom failed" if kind == "controlled" else
                          "Error: \"no tool registered under 'missing'\"" if kind == "missing" else "private-argument")
    start, end = hook.events
    assert start.event_name == "tool.call.start"
    assert end.event_name == ("tool.call.finish" if kind in {"builtin", "mcp"} else "tool.call.error")
    assert end.outcome == ("success" if kind in {"builtin", "mcp"} else "failure")
    assert end.duration_seconds == 2.0
    assert {(e.run_id, e.span_id, e.parent_span_id, e.agent_id) for e in hook.events} == {
        ("run", "tool-span", "agent-span", "agent-a")}
    if kind == "missing":
        assert "origin" not in end.attributes
    else:
        assert end.attributes["origin"] == ("mcp" if kind == "mcp" else "builtin")
    assert "private" not in repr(hook.events)
    if kind == "mcp":
        client.call_tool.assert_called_once_with("echo", {"text": "private-argument"})


def test_tool_context_and_hook_argument_names_are_not_reserved() -> None:
    class ArgumentsTool(EchoTool):
        def run(self, **kwargs: Any) -> str:
            assert kwargs == {"context": "private-context", "hook": "private-hook"}
            return "ok"
    registry = ToolRegistry(hook=RecordingHook())
    registry.register(ArgumentsTool())
    assert registry.call("echo", context="private-context", hook="private-hook") == "ok"


@pytest.mark.parametrize("site", ["hook", "id_provider", "clock", "monotonic_clock"])
@pytest.mark.parametrize("failure", [False, True])
def test_tool_monitoring_fails_open(site: str, failure: bool) -> None:
    def broken(*args: Any) -> Any:
        raise RuntimeError("private-monitoring-error")
    hook = Mock()
    hook.emit.side_effect = broken if site == "hook" else None
    diagnostics: list[str] = []
    registry = ToolRegistry(hook=hook, diagnostic=diagnostics.append,
                            **({site: broken} if site != "hook" else {}))
    registry.register(BoomTool() if failure else EchoTool())
    assert registry.call("boom" if failure else "echo", text="ok") == ("Error: boom failed" if failure else "ok")
    assert diagnostics and set(diagnostics) == {"Monitoring failure; agent execution continues."}


def test_unsafe_tool_name_uses_fixed_metadata_fallback() -> None:
    hook = RecordingHook()
    tool = EchoTool()
    tool.name = "private name / path"
    registry = ToolRegistry(hook=hook)
    registry.register(tool)
    assert registry.call(tool.name, text="ok") == "ok"
    assert {e.attributes["tool_name"] for e in hook.events} == {"unknown_tool"}


def test_tool_event_construction_and_diagnostic_fail_open() -> None:
    def diagnostic(_: str) -> None:
        raise RuntimeError("private-diagnostic")
    registry = ToolRegistry(hook=RecordingHook(), diagnostic=diagnostic)
    tool = EchoTool()
    tool.origin = "invalid-origin"
    registry.register(tool)
    assert registry.call("echo", text="ok") == "ok"


def test_no_hook_registry_ignores_monitoring_providers() -> None:
    def broken() -> Any:
        raise AssertionError("provider must not run")
    registry = ToolRegistry(id_provider=broken, clock=broken, monotonic_clock=broken)
    registry.register(EchoTool())
    assert registry.call("echo", text="ok") == "ok"


def test_react_preserves_registry_subclass_call_override() -> None:
    from agent.engine.react_loop import ReactLoop
    class LegacyRegistry(ToolRegistry):
        def call(self, name: str, **kwargs: Any) -> str:
            return "legacy override"
    class LLM:
        def __init__(self) -> None:
            self.responses = iter([{"tool_call": {"name": "boom", "arguments": {}}}, {"final_answer": "done"}])
        def chat(self, *args: Any, **kwargs: Any) -> dict:
            return next(self.responses)
    hook = RecordingHook()
    registry = LegacyRegistry(hook=hook)
    boom = BoomTool()
    boom.permission_tier = PermissionTier.REGULAR  # this test is about call overrides, not permissions
    registry.register(boom)
    loop = ReactLoop(LLM(), registry, hook=hook)
    assert loop.run("private-prompt") == "done"
    assert next(m for m in loop.messages if m["role"] == "tool")["content"] == "legacy override"
    assert [e.event_name for e in hook.events] == ["agent.run.start", "agent.run.finish"]


def test_an_unavailable_tool_error_is_raised_so_the_loop_can_withdraw_the_tool():
    from agent.tools.base import ToolUnavailableError

    class Down(Tool):
        name = "down"
        description = "d"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, **kwargs):
            raise ToolUnavailableError("host did not answer")

    registry = ToolRegistry()
    registry.register(Down())

    with pytest.raises(ToolUnavailableError, match="host did not answer"):
        registry.call("down")
