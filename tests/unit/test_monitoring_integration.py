import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import urlopen

import pytest

from agent.config import MonitoringConfig, MetricsConfig
from agent.engine.llm_client import LLMClient
from agent.engine.react_loop import ReactLoop
from agent.memory import CoreProfileManager, SqliteMemoryStore
from agent.monitoring.runtime import MonitoringRuntime
from agent.observability import CompositeHook
from agent.tools import ToolRegistry


def test_offline_run_reaches_metrics_and_correlated_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.monitoring.prometheus as metrics_module
    original = metrics_module.PrometheusHook.start_endpoint
    monkeypatch.setattr(metrics_module.PrometheusHook, "start_endpoint",
                        lambda self, host, port: original(self, host, 0))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="private-answer", tool_calls=None))],
                               usage=SimpleNamespace(total_tokens=9, prompt_tokens=7, completion_tokens=2))
    sdk = Mock()
    sdk.chat.completions.create.return_value = response
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: sdk)
    runtime = MonitoringRuntime(MonitoringConfig(enabled=True, metrics=MetricsConfig()), tmp_path)
    hook = CompositeHook(runtime.hooks)
    timestamp = lambda: datetime(2026, 9, 16, tzinfo=timezone.utc)
    ticks = iter([1.0, 2.0])
    llm = LLMClient(api_key="test-placeholder", hook=hook, id_provider=lambda: "llm-span",
                    clock=timestamp, monotonic_clock=lambda: next(ticks))
    identifiers = iter(["parent-run", "parent-span"])
    parent_ticks = iter([0.0, 3.0])
    memory_store = SqliteMemoryStore(tmp_path / "memory.db")
    core_profile = CoreProfileManager(tmp_path / "core.md")
    loop = ReactLoop(llm, ToolRegistry(hook=hook), memory_store=memory_store,
                     system_prompt_provider=core_profile.format_for_prompt, hook=hook,
                     id_provider=lambda: next(identifiers), clock=timestamp,
                     monotonic_clock=lambda: next(parent_ticks))
    try:
        assert loop.run("private-prompt") == "private-answer"
        metrics = runtime.hooks[0]
        with urlopen(f"http://127.0.0.1:{metrics.server.server_port}/metrics", timeout=2) as response:
            text = response.read().decode()
        assert 'agent_runs_completed_total{agent_id="parent",outcome="success"} 1.0' in text
        assert 'agent_llm_tokens_total{direction="input"} 7.0' in text
    finally:
        runtime.close()
    text = (tmp_path / ".agent_monitoring/events.jsonl").read_text()
    assert "private-prompt" not in text and "private-answer" not in text and "test-placeholder" not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert {event["run_id"] for event in events} == {"parent-run"}
    llm_events = [event for event in events if event["component"] == "llm"]
    assert len(llm_events) == 2
    assert all(event["span_id"] == "llm-span" and event["parent_span_id"] == "parent-span" for event in llm_events)


# --- a run's trace has to contain everything the run did -----------------------

class _RecordingHook:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _OneAnswerLLM:
    def chat(self, messages, tools=None, tool_choice="auto"):
        return {"final_answer": "ok"}


def _sequential_ids():
    counter = iter(range(1, 1000))
    return lambda: f"id{next(counter)}"


def test_prompt_time_memory_reads_are_recorded_inside_the_run_that_caused_them(tmp_path: Path) -> None:
    """Seen live: the parent re-reads the core profile on every step, and those
    reads were emitted with no run context - each became its own one-event
    "run" (25 distinct run_ids in a trace of 6 real runs), so a run's trace
    was missing its own memory reads."""
    hook = _RecordingHook()
    core_profile = CoreProfileManager(tmp_path / "core.md", hook=hook)
    loop = ReactLoop(
        _OneAnswerLLM(), ToolRegistry(),
        system_prompt_provider=lambda context=None: core_profile.format_for_prompt_with_context(context=context),
        hook=hook, id_provider=_sequential_ids(),
    )

    loop.run("hello")

    reads = [e for e in hook.events if e.component == "memory"]
    (run_finish,) = [e for e in hook.events if e.event_name == "agent.run.finish"]
    assert reads, "the profile read should have been observed"
    assert {e.run_id for e in reads} == {run_finish.run_id}
    assert all(e.parent_span_id == run_finish.span_id for e in reads)
    assert {e.run_id for e in hook.events} == {run_finish.run_id}  # nothing outside the one run


def test_a_provider_without_a_context_parameter_is_still_called_plainly(tmp_path: Path) -> None:
    hook = _RecordingHook()
    core_profile = CoreProfileManager(tmp_path / "core.md", hook=hook)
    loop = ReactLoop(_OneAnswerLLM(), ToolRegistry(), system_prompt_provider=core_profile.format_for_prompt, hook=hook)

    assert loop.run("hello") == "ok"
    assert loop._provider_takes_context is False
