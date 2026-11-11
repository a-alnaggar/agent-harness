from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from agent.observability import CompositeHook, LifecycleEvent, NoOpHook, RunContext, context_aware, contextual_method


def make_event(**overrides: object) -> LifecycleEvent:
    values = {
        "timestamp": datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc),
        "event_name": "agent.run.finish",
        "run_id": "run-1",
        "span_id": "span-1",
        "agent_id": "parent",
        "component": "agent",
        "operation": "run",
        "outcome": "success",
        "duration_seconds": 1.25,
        "attributes": {"max_steps": 10},
    }
    values.update(overrides)
    return LifecycleEvent(**values)


def test_lifecycle_event_is_immutable_and_attributes_are_bounded() -> None:
    event = make_event()

    assert event.event_name == "agent.run.finish"
    assert event.attributes == {"max_steps": 10}
    with pytest.raises(FrozenInstanceError):
        event.outcome = "failure"
    with pytest.raises(TypeError):
        event.attributes["extra"] = True


@pytest.mark.parametrize("key", ["prompt", "tool_arguments", "memory_value", "access_token"])
def test_lifecycle_event_rejects_sensitive_attribute_keys(key: str) -> None:
    with pytest.raises(ValueError, match="not allowed for this event"):
        make_event(attributes={key: "redacted"})


def test_lifecycle_event_rejects_too_many_attributes() -> None:
    with pytest.raises(ValueError, match="at most 16"):
        make_event(attributes={f"field_{index}": index for index in range(17)})


def test_noop_and_composite_hooks_dispatch_in_order() -> None:
    event = make_event()
    received: list[tuple[str, LifecycleEvent]] = []

    class RecordingHook:
        def __init__(self, name: str) -> None:
            self.name = name

        def emit(self, emitted: LifecycleEvent) -> None:
            received.append((self.name, emitted))

    NoOpHook().emit(event)
    CompositeHook([RecordingHook("first"), RecordingHook("second")]).emit(event)

    assert received == [("first", event), ("second", event)]


def test_composite_hook_isolates_a_failing_hook() -> None:
    received: list[LifecycleEvent] = []
    event = make_event()

    class FailingHook:
        def emit(self, emitted: LifecycleEvent) -> None:
            raise RuntimeError("collector unavailable")

    class RecordingHook:
        def emit(self, emitted: LifecycleEvent) -> None:
            received.append(emitted)

    diagnostics: list[str] = []
    CompositeHook([FailingHook(), RecordingHook()], diagnostic=diagnostics.append).emit(event)

    assert received == [event]
    assert diagnostics == ["Monitoring failure; agent execution continues."]


@pytest.mark.parametrize("attributes", [
    {"label": "private data"}, {"error_type": "RuntimeError"},
    {"input_tokens": 10}, {"output_tokens": 20},
])
def test_attributes_require_an_explicit_event_policy(attributes: dict) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        make_event(attributes=attributes)


@pytest.mark.parametrize("value", [True, "10", 1.5])
def test_max_steps_requires_exact_integer_type(value: object) -> None:
    with pytest.raises(TypeError, match="wrong type"):
        make_event(attributes={"max_steps": value})


@pytest.mark.parametrize("value", [-1, 2**63])
def test_max_steps_is_bounded(value: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        make_event(attributes={"max_steps": value})


@pytest.mark.parametrize("value", ["private value with spaces", "X" * 129])
def test_error_type_is_a_bounded_identifier(value: str) -> None:
    with pytest.raises(ValueError, match="bounded class identifier"):
        make_event(event_name="agent.run.error", attributes={"error_type": value})


def test_unknown_event_requires_deliberate_policy_extension() -> None:
    with pytest.raises(ValueError, match="no safe attribute policy"):
        make_event(event_name="future.event", attributes={"input_tokens": 10})


@pytest.mark.parametrize("key", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [True, "10", -1, 2**63])
def test_llm_token_policy_rejects_unsafe_values(key: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_event(event_name="llm.call.finish", attributes={key: value})


@pytest.mark.parametrize("event_name", ["llm.call.start", "llm.call.error"])
def test_llm_token_counts_are_finish_only(event_name: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        make_event(event_name=event_name, outcome=None if event_name.endswith(".start") else "failure", attributes={"input_tokens": 10})


def test_event_copies_attributes_and_normalizes_timestamp_to_utc() -> None:
    from datetime import timedelta

    attributes = {"max_steps": 10}
    event = make_event(
        timestamp=datetime(2026, 9, 15, 11, tzinfo=timezone(timedelta(hours=2))),
        attributes=attributes,
    )
    attributes["max_steps"] = 20
    assert event.attributes == {"max_steps": 10}
    assert event.timestamp == datetime(2026, 9, 15, 9, tzinfo=timezone.utc)


def test_run_context_is_immutable_and_event_retains_parent_span() -> None:
    context = RunContext("run-1", "child-1", "parent-1", "parent")
    with pytest.raises(FrozenInstanceError):
        context.run_id = "changed"
    assert make_event(parent_span_id=context.parent_span_id).parent_span_id == "parent-1"


@pytest.mark.parametrize("field,value", [("run_id", ""), ("span_id", 1),
                                        ("parent_span_id", "X" * 129), ("agent_id", None)])
def test_run_context_validates_identifiers(field: str, value: object) -> None:
    values = dict(run_id="run-1", span_id="span-1", parent_span_id=None, agent_id="parent")
    values[field] = value
    with pytest.raises(ValueError, match="bounded"):
        RunContext(**values)


@pytest.mark.parametrize("attributes", [{"origin": "remote"}, {"tool_name": "private path"},
                                        {"arguments": "private"}, {"result": "private"}])
def test_tool_attribute_policy_rejects_unsafe_metadata(attributes: dict) -> None:
    with pytest.raises(ValueError):
        make_event(event_name="tool.call.finish", attributes=attributes)


@pytest.mark.parametrize("key", ["path", "key", "value", "email", "result"])
def test_memory_attribute_policy_is_content_free(key: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        make_event(event_name="memory.operation.finish", operation="read", attributes={key: "private"})


def test_memory_operation_identity_is_explicitly_bounded() -> None:
    with pytest.raises(ValueError, match="memory operation"):
        make_event(event_name="memory.operation.finish", operation="private-value", attributes={})


@pytest.mark.parametrize("field", ["run_id", "span_id", "parent_span_id", "agent_id"])
@pytest.mark.parametrize("value", ["private data", "x\ny", "x/y", "x@y", "é", "x" * 129])
def test_context_identifiers_reject_unsafe_syntax(field: str, value: str) -> None:
    values = dict(run_id="run", span_id="span", parent_span_id=None, agent_id="agent")
    values[field] = value
    with pytest.raises(ValueError):
        RunContext(**values)


@pytest.mark.parametrize("field", ["event_name", "component", "operation"])
@pytest.mark.parametrize("value", [None, "private data", "x\ny", "x/y", "é", "x" * 129])
def test_event_labels_reject_unsafe_syntax(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_event(**{field: value})


@pytest.mark.parametrize("outcome", [None, "error", "private data", True])
def test_terminal_outcome_contract(outcome: object) -> None:
    with pytest.raises(ValueError, match="terminal outcome"):
        make_event(outcome=outcome)


def test_context_capability_requires_declaration_and_preserves_overrides() -> None:
    @context_aware("chat", "chat_with_context")
    class Client:
        def chat(self) -> str:
            return "legacy"
        def chat_with_context(self) -> str:
            return "contextual"
    class Unchanged(Client):
        pass
    class Override(Client):
        def chat(self) -> str:
            return "override"
    @context_aware("chat", "chat_with_context")
    class OptIn(Override):
        def chat_with_context(self) -> str:
            return self.chat()
    class Undeclared:
        def chat(self) -> str:
            return "legacy"
        def chat_with_context(self) -> str:
            return "contextual"
    assert contextual_method(Unchanged(), "chat")() == "contextual"
    assert contextual_method(Override(), "chat") is None
    assert contextual_method(OptIn(), "chat")() == "override"
    assert contextual_method(Undeclared(), "chat") is None
    instance = Client()
    instance.chat = lambda: "instance override"
    assert contextual_method(instance, "chat") is None


def permission_finish(**attributes):
    return make_event(
        event_name="permission.request.finish", component="permission", operation="request",
        outcome="success", duration_seconds=0.1, attributes=attributes,
    )


def test_permission_events_accept_the_deny_tier_and_the_policy_deny_decision():
    event = permission_finish(tool_name="send_email", origin="mcp", tier="deny", decision="policy_deny")

    assert event.attributes["decision"] == "policy_deny"
    assert event.attributes["tier"] == "deny"


def test_permission_events_still_reject_an_unknown_decision_or_tier():
    with pytest.raises(ValueError, match="decision"):
        permission_finish(tool_name="t", origin="mcp", tier="strict", decision="maybe")
    with pytest.raises(ValueError, match="tier"):
        permission_finish(tool_name="t", origin="mcp", tier="lenient", decision="once")


def tool_finish(**attributes):
    return make_event(
        event_name="tool.call.finish", component="tool", operation="call", attributes=attributes,
    )


@pytest.mark.parametrize("origin", ["builtin", "mcp", "subagent"])
def test_tool_events_accept_the_three_real_origins(origin: str) -> None:
    assert tool_finish(tool_name="delegate_task", origin=origin).attributes["origin"] == origin


def test_tool_events_still_reject_an_unknown_origin() -> None:
    with pytest.raises(ValueError, match="origin"):
        tool_finish(tool_name="t", origin="remote")
