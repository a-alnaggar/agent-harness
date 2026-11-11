from datetime import datetime, timezone
from urllib.request import urlopen

from prometheus_client import generate_latest

from agent.monitoring.prometheus import PrometheusHook
from agent.observability import LifecycleEvent


def event(name: str, component: str, operation: str = "call", **changes: object) -> LifecycleEvent:
    values = dict(timestamp=datetime(2026, 9, 16, tzinfo=timezone.utc), event_name=name,
                  run_id="run", span_id="span", agent_id="parent", component=component,
                  operation=operation, outcome=None if name.endswith(".start") else "success",
                  duration_seconds=None if name.endswith(".start") else 1.0)
    values.update(changes)
    return LifecycleEvent(**values)


def test_metrics_terminal_counting_usage_and_registry_isolation() -> None:
    hook = PrometheusHook()
    hook.emit(event("agent.run.start", "agent", "run"))
    hook.emit(event("llm.call.start", "llm"))
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "parent"}) == 1
    hook.emit(event("llm.call.finish", "llm", attributes={"input_tokens": 7, "output_tokens": 0}))
    hook.emit(event("llm.call.error", "llm", outcome="failure", attributes={"error_type": "RuntimeError"}))
    hook.emit(event("agent.run.finish", "agent", "run"))
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "parent"}) == 0
    assert hook.registry.get_sample_value("agent_runs_completed_total", {"outcome": "success", "agent_id": "parent"}) == 1
    assert hook.registry.get_sample_value("agent_llm_calls_total", {"outcome": "success"}) == 1
    assert hook.registry.get_sample_value("agent_llm_calls_total", {"outcome": "failure"}) == 1
    assert hook.registry.get_sample_value("agent_llm_tokens_total", {"direction": "input"}) == 7
    assert hook.registry.get_sample_value("agent_llm_tokens_total", {"direction": "output"}) == 0
    assert hook.registry.get_sample_value("agent_llm_duration_seconds_sum", {"outcome": "success"}) == 1
    assert PrometheusHook().registry.get_sample_value("agent_runs_started_total", {"agent_id": "parent"}) is None
    text = generate_latest(hook.registry).decode()
    assert "run_id=" not in text and "span_id=" not in text and "error_type=" not in text


def test_tool_labels_are_bounded_and_missing_origin_is_unknown() -> None:
    hook = PrometheusHook()
    for index in range(130):
        hook.emit(event("tool.call.finish", "tool", attributes={"tool_name": f"tool_{index}"}))
    assert len(hook._tools) == 128
    labels = {"tool": "other", "origin": "unknown", "outcome": "success"}
    assert hook.registry.get_sample_value("agent_tool_calls_total", labels) == 2
    hook.emit(event("memory.operation.finish", "memory", "read"))
    assert hook.registry.get_sample_value("agent_memory_operations_total", {"operation": "read", "outcome": "success"}) == 1


def test_active_set_is_bounded_and_duplicate_terminal_does_not_go_negative() -> None:
    hook = PrometheusHook()
    for index in range(1025):
        hook.emit(event("agent.run.start", "agent", "run", span_id=f"span-{index}"))
    assert len(hook._active) == 1024
    for index in range(1025):
        hook.emit(event("agent.run.finish", "agent", "run", span_id=f"span-{index}"))
    hook.emit(event("agent.run.finish", "agent", "run"))
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "parent"}) == 0


def test_agent_id_distinguishes_subagent_runs_from_the_parent() -> None:
    """Regression test for the 'Sub-agents activity' dashboard panel: a
    sub-agent's completed runs must be countable separately from the
    parent's, not merged into one undifferentiated total."""
    hook = PrometheusHook()
    hook.emit(event("agent.run.start", "agent", "run", agent_id="parent"))
    hook.emit(event("agent.run.start", "agent", "run", agent_id="course_knowledge", span_id="sub-span"))
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "parent"}) == 1
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "course_knowledge"}) == 1
    hook.emit(event("agent.run.finish", "agent", "run", agent_id="course_knowledge", span_id="sub-span"))
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "course_knowledge"}) == 0
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "parent"}) == 1
    assert hook.registry.get_sample_value(
        "agent_runs_completed_total", {"outcome": "success", "agent_id": "course_knowledge"}
    ) == 1
    assert hook.registry.get_sample_value(
        "agent_runs_completed_total", {"outcome": "success", "agent_id": "parent"}
    ) is None


def test_agent_id_label_is_bounded() -> None:
    hook = PrometheusHook()
    for index in range(130):
        hook.emit(event("agent.run.start", "agent", "run", agent_id=f"agent_{index}", span_id=f"span-{index}"))
    assert len(hook._agents) == 128
    assert hook.registry.get_sample_value("agent_runs_active", {"agent_id": "other"}) == 2


def test_metrics_endpoint_and_clean_repeated_shutdown() -> None:
    hook = PrometheusHook()
    hook.start_endpoint(port=0)
    port = hook.server.server_port
    try:
        hook.emit(event("agent.run.start", "agent", "run"))
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
            assert response.status == 200
            assert b"agent_runs_started_total" in response.read()
    finally:
        hook.close()
        hook.close()
    assert not hook.thread.is_alive()


def test_the_subagent_origin_reaches_the_tool_metrics_as_its_own_label_value() -> None:
    from datetime import datetime, timezone

    from prometheus_client import CollectorRegistry

    from agent.monitoring.prometheus import PrometheusHook
    from agent.observability import LifecycleEvent

    registry = CollectorRegistry()
    hook = PrometheusHook(registry)
    for tool_name, origin, seconds in (("read_file", "builtin", 0.01), ("delegate_task", "subagent", 12.0)):
        hook.emit(LifecycleEvent(
            timestamp=datetime(2026, 9, 25, tzinfo=timezone.utc), event_name="tool.call.finish",
            run_id="r", span_id=f"s-{tool_name}", agent_id="parent", component="tool", operation="call",
            outcome="success", duration_seconds=seconds, attributes={"tool_name": tool_name, "origin": origin},
        ))

    def total(name: str, origin: str) -> float:
        return registry.get_sample_value(name, {"tool": "delegate_task" if origin == "subagent" else "read_file",
                                                "origin": origin, "outcome": "success"})

    assert total("agent_tool_calls_total", "subagent") == 1.0
    assert total("agent_tool_calls_total", "builtin") == 1.0
    # the two are separable, which is what lets the aggregate latency panels leave delegation out
    assert registry.get_sample_value("agent_tool_duration_seconds_sum",
                                     {"tool": "delegate_task", "origin": "subagent", "outcome": "success"}) == 12.0
