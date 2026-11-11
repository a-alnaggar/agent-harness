import json
from datetime import datetime, timezone

import pytest

from agent.engine.react_loop import ReactLoop
from agent.monitoring.jsonl import JsonlHook
from agent.monitoring.prometheus import PrometheusHook
from agent.observability import CompositeHook, LifecycleEvent
from agent.subagents import DelegateTaskTool, SubAgentConfig, SubAgentRunner
from agent.tools import Tool, ToolRegistry
from agent.tools.permissions import PermissionTier
from agent.tools.sandbox.exec_tool import SandboxExecTool
from agent.tools.sandbox.ssh_client import SshCommandResult


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict], tools: object = None, tool_choice: str = "auto") -> dict:
        return self.responses.pop(0)


class ConfirmTool(Tool):
    name = "confirm_tool"
    description = "private-description"
    parameters = {"type": "object", "properties": {"secret": {"type": "string"}}}
    permission_tier = PermissionTier.CONFIRM

    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs: object) -> str:
        self.calls += 1
        return "private-result"


class StrictTool(ConfirmTool):
    name = "strict_tool"
    permission_tier = PermissionTier.STRICT


class SecretTool(ConfirmTool):
    name = "secret_tool"


class FakeSshClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_command(self, command: str) -> SshCommandResult:
        self.calls.append(command)
        return SshCommandResult(0, "ok", "")


def monitored_loop(responses: list[dict], registry: ToolRegistry, hook: RecordingHook) -> ReactLoop:
    identifiers = iter(["run", "agent-span", "permission-1", "permission-2", "permission-3"])
    return ReactLoop(
        FakeLLM(responses), registry, hook=hook, id_provider=lambda: next(identifiers),
        clock=lambda: datetime(2026, 9, 17, tzinfo=timezone.utc), monotonic_clock=lambda: 1.0,
    )


def permission_events(hook: RecordingHook) -> list[LifecycleEvent]:
    return [event for event in hook.events if event.component == "permission"]


@pytest.mark.parametrize(
    ("tool", "decisions", "expected"),
    [
        (ConfirmTool(), ["once"], ["once"]),
        (ConfirmTool(), ["always"], ["always", "cached"]),
        (StrictTool(), ["always", "always"], ["always", "always"]),
    ],
)
def test_permission_events_cover_once_always_cached_and_strict(
    tool: ConfirmTool, decisions: list[str], expected: list[str],
) -> None:
    hook = RecordingHook()
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span", monotonic_clock=lambda: 1.0)
    registry.register(tool)
    calls = iter(decisions)
    responses = [
        {"tool_call": {"name": tool.name, "arguments": {}}}
        for _ in expected
    ] + [{"final_answer": "done"}]

    assert monitored_loop(responses, registry, hook).run("private-prompt", confirm_callback=lambda *_: next(calls)) == "done"

    events = permission_events(hook)
    assert [event.event_name for event in events] == [
        name for _ in expected for name in ("permission.request.start", "permission.request.finish")
    ]
    assert [event.attributes["decision"] for event in events if event.event_name.endswith(".finish")] == expected
    assert all(event.run_id == "run" and event.parent_span_id == "agent-span" for event in events)
    assert all(event.duration_seconds == 0.0 for event in events if event.outcome is not None)
    assert tool.calls == len(expected)


def test_denial_is_observed_without_a_tool_call_or_private_data() -> None:
    hook = RecordingHook()
    tool = ConfirmTool()
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span", monotonic_clock=lambda: 1.0)
    registry.register(tool)

    assert monitored_loop([
        {"tool_call": {"name": tool.name, "arguments": {"secret": "private-email-data"}}},
        {"final_answer": "done"},
    ], registry, hook).run("private-prompt", confirm_callback=lambda *_: "deny") == "done"

    events = permission_events(hook)
    assert [event.event_name for event in events] == ["permission.request.start", "permission.request.finish"]
    assert events[-1].attributes["decision"] == "deny"
    assert not [event for event in hook.events if event.component == "tool"]
    assert tool.calls == 0
    assert "private" not in repr(events)


def test_callback_failure_is_observed_and_preserves_agent_failure() -> None:
    hook = RecordingHook()
    tool = ConfirmTool()
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span")
    registry.register(tool)

    def broken_callback(*_: object) -> str:
        raise RuntimeError("private callback failure")

    with pytest.raises(RuntimeError, match="private callback failure"):
        monitored_loop([
            {"tool_call": {"name": tool.name, "arguments": {"secret": "private"}}},
        ], registry, hook).run("private", confirm_callback=broken_callback)

    events = permission_events(hook)
    assert [event.event_name for event in events] == ["permission.request.start", "permission.request.error"]
    assert events[-1].outcome == "failure"
    assert events[-1].attributes == {"tool_name": "confirm_tool", "origin": "builtin", "tier": "confirm", "error_type": "RuntimeError"}
    assert not [event for event in hook.events if event.component == "tool"]
    assert "private" not in repr(events)


def test_permission_metrics_and_jsonl_are_correlated_and_content_free() -> None:
    records: list[str] = []
    metrics = PrometheusHook()
    logs = JsonlHook(records.append)
    hook = CompositeHook([metrics, logs])
    tool = ConfirmTool()
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span", monotonic_clock=lambda: 1.0)
    registry.register(tool)
    identifiers = iter(["run", "agent-span", "permission-span"])
    loop = ReactLoop(
        FakeLLM([{"tool_call": {"name": tool.name, "arguments": {"secret": "private-email-data"}}}, {"final_answer": "done"}]),
        registry, hook=hook, id_provider=lambda: next(identifiers),
        clock=lambda: datetime(2026, 9, 17, tzinfo=timezone.utc), monotonic_clock=lambda: 1.0,
    )
    try:
        assert loop.run("private-prompt", confirm_callback=lambda *_: "once") == "done"
    finally:
        logs.close()

    labels = {"tool": "confirm_tool", "origin": "builtin", "tier": "confirm", "decision": "once", "outcome": "success"}
    assert metrics.registry.get_sample_value("agent_permission_requests_total", labels) == 1
    values = [json.loads(line) for line in records]
    permissions = [value for value in values if value["component"] == "permission"]
    assert {value["run_id"] for value in permissions} == {"run"}
    assert {value["parent_span_id"] for value in permissions} == {"agent-span"}
    assert "private" not in "".join(records)


def test_child_tool_permission_uses_the_shared_hook_and_parent_context() -> None:
    hook = RecordingHook()
    secret = SecretTool()
    master = ToolRegistry(hook=hook, id_provider=lambda: "unused")
    master.register(secret)
    runner = SubAgentRunner(
        FakeLLM([
            {"tool_call": {"name": "delegate_task", "arguments": {"subagent_type": "child", "task": "private"}}},
            {"tool_call": {"name": "secret_tool", "arguments": {}}},
            {"final_answer": "child complete"},
            {"final_answer": "parent complete"},
        ]), master, hook=hook,
        configs={"child": SubAgentConfig(name="child", role="Use the tool.", allowed_tools=["secret_tool"], max_steps=3)},
    )
    parent = ToolRegistry(hook=hook, id_provider=lambda: "delegate-span", monotonic_clock=lambda: 1.0)
    parent.register(DelegateTaskTool(runner))
    loop = ReactLoop(runner.llm, parent, hook=hook, id_provider=iter(["run", "parent-span", "permission-span"]).__next__, monotonic_clock=lambda: 1.0)

    assert loop.run("delegate", confirm_callback=lambda *_: "once") == "parent complete"

    delegate_event = next(event for event in hook.events if event.component == "tool" and event.attributes.get("tool_name") == "delegate_task")
    child_start = next(event for event in hook.events if event.event_name == "agent.run.start" and event.agent_id == "child")
    permission = next(event for event in hook.events if event.component == "permission")
    assert child_start.run_id == permission.run_id == "run"
    assert child_start.parent_span_id == delegate_event.span_id
    assert permission.agent_id == "child" and permission.parent_span_id == child_start.span_id


def test_sandbox_permission_approval_and_command_refusal_are_separate_events() -> None:
    hook = RecordingHook()
    client = FakeSshClient()
    registry = ToolRegistry(hook=hook, id_provider=lambda: "tool-span", monotonic_clock=lambda: 1.0)
    registry.register(SandboxExecTool(client))

    assert monitored_loop([
        {"tool_call": {"name": "sandbox_exec", "arguments": {"command": "rm -rf /private"}}},
        {"final_answer": "done"},
    ], registry, hook).run("private", confirm_callback=lambda *_: "once") == "done"

    assert [event.event_name for event in permission_events(hook)] == ["permission.request.start", "permission.request.finish"]
    tool_events = [event for event in hook.events if event.component == "tool"]
    assert [event.event_name for event in tool_events] == ["tool.call.start", "tool.call.error"]
    assert tool_events[-1].outcome == "failure" and client.calls == []
    assert "private" not in repr(hook.events)
