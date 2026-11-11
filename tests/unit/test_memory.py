"""Lifecycle, persistence, and tool-compatibility tests for hierarchical memory."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agent.memory import CoreProfileManager, ForgetMemoryTool, SaveMemoryTool, SqliteMemoryStore
from agent.observability import LifecycleEvent, RunContext
from agent.tools import ToolRegistry


class MemoryRecordingHook:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


def _parent_context() -> RunContext:
    return RunContext("run", "parent-span", None, "agent-a")


def _assert_correlated_pairs(events: list[LifecycleEvent], operations: list[str]) -> None:
    assert [event.event_name for event in events] == [
        name
        for operation in operations
        for name in ("memory.operation.start", "memory.operation.finish")
    ]
    assert [event.operation for event in events] == [
        operation for operation in operations for _ in range(2)
    ]
    assert {(event.run_id, event.span_id, event.parent_span_id, event.agent_id) for event in events} == {
        ("run", "memory-span", "parent-span", "agent-a")
    }
    for event in events:
        assert event.attributes == {}
        if event.event_name.endswith(".finish"):
            assert event.outcome == "success"
            assert event.duration_seconds == 2.0


def test_sqlite_general_and_faq_operations_emit_safe_correlated_spans(tmp_path: Path) -> None:
    hook = MemoryRecordingHook()
    ticks = iter(float(value) for value in range(0, 20, 2))
    store = SqliteMemoryStore(
        tmp_path / "private-memory.db",
        hook=hook,
        id_provider=lambda: "memory-span",
        clock=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
        monotonic_clock=lambda: next(ticks),
    )
    context = _parent_context()

    store.save_general(key="private-key", content="private-value", context=context)
    assert store.get_general_value("private-key", context=context) == "private-value"
    assert store.delete_general("private-key", context=context)
    store.save_faq(
        "course", "When is class?", "Friday", "pdf", "private-source", 1.0, context=context,
    )
    assert len(store.search_faq("course", "class", context=context)) == 1

    _assert_correlated_pairs(hook.events, ["save", "read", "forget", "save", "read"])
    assert "private-key" not in repr(hook.events)
    assert "private-value" not in repr(hook.events)
    assert "private-source" not in repr(hook.events)


def test_core_profile_operations_emit_safe_correlated_spans(tmp_path: Path) -> None:
    hook = MemoryRecordingHook()
    ticks = iter(float(value) for value in range(0, 16, 2))
    profile = CoreProfileManager(
        tmp_path / "private-core.md",
        hook=hook,
        id_provider=lambda: "memory-span",
        clock=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
        monotonic_clock=lambda: next(ticks),
    )
    context = _parent_context()

    profile.update_preference("user_name", "private-name", context=context)
    assert profile.get_preference("user_name", context=context) == "private-name"
    assert "private-name" in profile.format_for_prompt_with_context(context=context)
    assert profile.remove_preference("user_name", context=context)

    _assert_correlated_pairs(hook.events, ["save", "read", "read", "forget"])
    assert "private-name" not in repr(hook.events)


def test_hierarchical_memory_persists_across_new_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    core_path = tmp_path / "core.md"

    first_store = SqliteMemoryStore(db_path)
    first_store.save_general(key="preferred_language", content="Python")
    first_store.save_faq("course", "When?", "Friday", "pdf", "syllabus", 1.0)
    first_profile = CoreProfileManager(core_path)
    first_profile.update_preference("user_name", "Ahmed")

    second_store = SqliteMemoryStore(db_path)
    second_profile = CoreProfileManager(core_path)
    assert second_store.get_general_value("preferred_language") == "Python"
    assert [entry.answer for entry in second_store.search_faq("course", "When")] == ["Friday"]
    assert second_profile.get_preference("user_name") == "Ahmed"


@pytest.mark.parametrize("component", ["sqlite", "core"])
def test_memory_operation_failures_preserve_exception_and_emit_safe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str,
) -> None:
    hook = MemoryRecordingHook()
    error = OSError("private-storage-failure")
    if component == "sqlite":
        store = SqliteMemoryStore(tmp_path / "memory.db", hook=hook)
        monkeypatch.setattr(store._general_repo, "save_general", lambda **_: (_ for _ in ()).throw(error))
        action = lambda: store.save_general(key="key", content="value")
    else:
        profile = CoreProfileManager(tmp_path / "core.md", hook=hook)
        monkeypatch.setattr(profile, "_atomic_write", lambda _: (_ for _ in ()).throw(error))
        action = lambda: profile.update_preference("user_name", "Ahmed")

    with pytest.raises(OSError) as caught:
        action()

    assert caught.value is error
    assert [event.event_name for event in hook.events] == [
        "memory.operation.start", "memory.operation.error",
    ]
    assert hook.events[-1].outcome == "failure"
    assert hook.events[-1].attributes == {"error_type": "OSError"}
    assert "private" not in repr(hook.events)


@pytest.mark.parametrize("component", ["sqlite", "core"])
def test_memory_monitoring_failures_are_fail_open(tmp_path: Path, component: str) -> None:
    def broken(*_: Any) -> Any:
        raise RuntimeError("private-monitoring-failure")

    class BrokenHook:
        def emit(self, event: LifecycleEvent) -> None:
            broken(event)

    diagnostics: list[str] = []
    if component == "sqlite":
        store = SqliteMemoryStore(tmp_path / "memory.db", hook=BrokenHook(), diagnostic=diagnostics.append)
        store.save_general(key="key", content="value")
        assert SqliteMemoryStore(tmp_path / "memory.db").get_general_value("key") == "value"
    else:
        profile = CoreProfileManager(tmp_path / "core.md", hook=BrokenHook(), diagnostic=diagnostics.append)
        profile.update_preference("user_name", "Ahmed")
        assert CoreProfileManager(tmp_path / "core.md").get_preference("user_name") == "Ahmed"

    assert diagnostics and set(diagnostics) == {"Monitoring failure; agent execution continues."}


def test_no_hook_does_not_call_monitoring_providers(tmp_path: Path) -> None:
    def broken() -> Any:
        raise AssertionError("provider must not run")

    store = SqliteMemoryStore(
        tmp_path / "memory.db", id_provider=broken, clock=broken, monotonic_clock=broken,
    )
    profile = CoreProfileManager(
        tmp_path / "core.md", id_provider=broken, clock=broken, monotonic_clock=broken,
    )
    store.save_general(key="key", content="value")
    profile.update_preference("user_name", "Ahmed")
    assert store.get_general_value("key") == "value"
    assert profile.get_preference("user_name") == "Ahmed"


@pytest.mark.parametrize("operation", ["save", "forget"])
def test_registry_preserves_hierarchical_memory_tool_run_overrides(tmp_path: Path, operation: str) -> None:
    class LegacySave(SaveMemoryTool):
        def run(self, *, key: str, value: str) -> str:
            return "legacy save"

    class LegacyForget(ForgetMemoryTool):
        def run(self, *, key: str) -> str:
            return "legacy forget"

    hook = MemoryRecordingHook()
    store = SqliteMemoryStore(tmp_path / "memory.db", hook=hook)
    registry = ToolRegistry(hook=hook)
    tool = LegacySave(store) if operation == "save" else LegacyForget(store)
    registry.register(tool)
    arguments = {"key": "key", **({"value": "new"} if operation == "save" else {})}

    result = registry.call_with_context(tool.name, arguments, context=_parent_context())

    assert result == f"legacy {operation}"
    tool_events = list(hook.events)
    assert [event.event_name for event in tool_events] == ["tool.call.start", "tool.call.finish"]
    assert store.get_general_value("key") is None
