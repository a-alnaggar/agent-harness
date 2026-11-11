"""Unit tests for LLMClient: API key/base URL resolution priority, and
correct translation between the OpenAI-shaped completion response and the
tool_call / final_answer dict ReactLoop expects."""

import inspect
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx2
import pytest
from openai import BadRequestError

from agent.engine.llm_client import ContextWindowExceededError, LLMClient
from agent.observability import LifecycleEvent, NoOpHook, RunContext
from agent.engine.react_loop import ReactLoop


def fake_bad_request_error(message: str) -> BadRequestError:
    response = httpx2.Response(400, request=httpx2.Request("POST", "http://example.test"))
    return BadRequestError(message, response=response, body=None)


def fake_completion(*, tool_calls=None, content=None, total_tokens=None, prompt_tokens=None):
    """Builds a minimal object matching the shape of the real OpenAI SDK's
    ChatCompletion, with only the fields LLMClient actually reads."""
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    choice = SimpleNamespace(message=message)
    usage = (
        SimpleNamespace(total_tokens=total_tokens, prompt_tokens=prompt_tokens)
        if total_tokens is not None or prompt_tokens is not None
        else None
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def fake_tool_call(name, arguments_json, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=arguments_json)
    return SimpleNamespace(id=call_id, function=function)


@pytest.fixture
def mock_openai(monkeypatch):
    """Patches the OpenAI class LLMClient constructs, returning the mock
    client instance so tests can control chat.completions.create()."""
    mock_client = MagicMock()
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: mock_client)
    return mock_client


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="No API key found"):
        LLMClient()


@pytest.mark.parametrize(
    "env_key,env_value",
    [("LLM_API_KEY", "from-llm-key"), ("OPENAI_API_KEY", "from-openai"), ("ACCESS_TOKEN", "from-access-token")],
)
def test_api_key_resolved_from_env_when_not_passed(monkeypatch, mock_openai, env_key, env_value):
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_key, env_value)

    captured = {}
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: captured.update(kwargs) or mock_openai)

    LLMClient()

    assert captured["api_key"] == env_value


def test_explicit_api_key_wins_over_env(monkeypatch, mock_openai):
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    captured = {}
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: captured.update(kwargs) or mock_openai)

    LLMClient(api_key="explicit-key")

    assert captured["api_key"] == "explicit-key"


def test_base_url_defaults_to_openai_when_nothing_else_set(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured = {}
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: captured.update(kwargs) or mock_openai)

    LLMClient()

    assert captured["base_url"] == "https://api.openai.com/v1"


def test_explicit_base_url_wins_over_env_and_default(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    monkeypatch.setenv("LLM_BASE_URL", "https://from-env.example")
    captured = {}
    monkeypatch.setattr("agent.engine.llm_client.OpenAI", lambda **kwargs: captured.update(kwargs) or mock_openai)

    LLMClient(base_url="https://explicit.example")

    assert captured["base_url"] == "https://explicit.example"


def test_chat_without_tools_omits_tools_kwargs(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content="hi")
    llm = LLMClient()

    llm.chat([{"role": "user", "content": "hi"}])

    _, kwargs = mock_openai.chat.completions.create.call_args
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_chat_with_tools_passes_tool_choice_auto_by_default(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content="hi")
    llm = LLMClient()

    llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "x"}}])

    _, kwargs = mock_openai.chat.completions.create.call_args
    assert kwargs["tool_choice"] == "auto"


def test_chat_passes_through_explicit_tool_choice(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content="hi")
    llm = LLMClient()

    llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], tool_choice="required")

    _, kwargs = mock_openai.chat.completions.create.call_args
    assert kwargs["tool_choice"] == "required"


def test_chat_parses_a_real_tool_call_correctly(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    tool_call = fake_tool_call("search_emails", '{"query": "is:unread", "max_results": 5}', call_id="call_42")
    mock_openai.chat.completions.create.return_value = fake_completion(
        tool_calls=[tool_call], content="thinking..."
    )
    llm = LLMClient()

    response = llm.chat([{"role": "user", "content": "search"}], tools=[{}])

    assert response == {
        "thought": "thinking...",
        "tool_call": {
            "name": "search_emails",
            "arguments": {"query": "is:unread", "max_results": 5},
            "id": "call_42",
        },
    }


def test_chat_falls_back_to_empty_args_on_malformed_json(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    tool_call = fake_tool_call("search_emails", "{not valid json")
    mock_openai.chat.completions.create.return_value = fake_completion(tool_calls=[tool_call])
    llm = LLMClient()

    response = llm.chat([{"role": "user", "content": "search"}], tools=[{}])

    assert response["tool_call"]["arguments"] == {}


def test_chat_returns_final_answer_when_no_tool_call(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content="Here's your answer.")
    llm = LLMClient()

    response = llm.chat([{"role": "user", "content": "hi"}])

    assert response == {"final_answer": "Here's your answer."}


def test_chat_returns_empty_string_final_answer_when_content_is_none(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content=None)
    llm = LLMClient()

    response = llm.chat([{"role": "user", "content": "hi"}])

    assert response == {"final_answer": ""}


def test_total_tokens_accumulates_across_calls(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    llm = LLMClient()
    assert llm.total_tokens == 0

    mock_openai.chat.completions.create.return_value = fake_completion(content="a", total_tokens=100)
    llm.chat([{"role": "user", "content": "hi"}])
    assert llm.total_tokens == 100

    mock_openai.chat.completions.create.return_value = fake_completion(content="b", total_tokens=50)
    llm.chat([{"role": "user", "content": "hi again"}])
    assert llm.total_tokens == 150


def test_total_tokens_unaffected_when_usage_missing(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.return_value = fake_completion(content="a", total_tokens=None)
    llm = LLMClient()

    llm.chat([{"role": "user", "content": "hi"}])

    assert llm.total_tokens == 0


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


def monitored_client(mock_openai: MagicMock, **overrides: object) -> tuple[LLMClient, RecordingHook]:
    hook = RecordingHook()
    ids = iter(["run-1", "span-1", "run-2", "span-2"])
    ticks = iter([10.0, 12.5, 20.0, 21.0])
    times = iter([
        datetime(2026, 9, 15, 10, tzinfo=timezone.utc),
        datetime(2026, 9, 15, 9, tzinfo=timezone.utc),
    ])
    options = dict(api_key="private-key", hook=hook, id_provider=lambda: next(ids),
                   monotonic_clock=lambda: next(ticks), clock=lambda: next(times))
    options.update(overrides)
    return LLMClient(**options), hook


def test_llm_lifecycle_success_is_safe_and_uses_monotonic_duration(mock_openai: MagicMock) -> None:
    completion = fake_completion(content="private-response", total_tokens=9)
    completion.usage.prompt_tokens = 7
    completion.usage.completion_tokens = 2
    mock_openai.chat.completions.create.return_value = completion
    llm, hook = monitored_client(mock_openai)
    assert llm.chat([{"role": "user", "content": "private-prompt"}]) == {"final_answer": "private-response"}
    start, finish = hook.events
    assert [e.event_name for e in hook.events] == ["llm.call.start", "llm.call.finish"]
    assert (start.run_id, start.span_id) == (finish.run_id, finish.span_id) == ("run-1", "span-1")
    assert start.component == "llm" and start.operation == "call" and start.agent_id == "parent"
    assert start.outcome is None and start.duration_seconds is None and start.attributes == {}
    assert finish.outcome == "success" and finish.duration_seconds == 2.5
    assert finish.timestamp < start.timestamp
    assert finish.attributes == {"input_tokens": 7, "output_tokens": 2}
    assert llm.total_tokens == 9
    assert "private" not in repr(hook.events)


@pytest.mark.parametrize("value", [None, True, "7", 1.5, -1, 2**63, 0, 7])
def test_llm_usage_accepts_only_safe_provider_integers(mock_openai: MagicMock, value: object) -> None:
    completion = fake_completion(content="ok")
    completion.usage = SimpleNamespace(prompt_tokens=value)
    mock_openai.chat.completions.create.return_value = completion
    llm, hook = monitored_client(mock_openai)
    assert llm.chat([]) == {"final_answer": "ok"}
    expected = {"input_tokens": value} if type(value) is int and 0 <= value <= 2**63 - 1 else {}
    assert hook.events[-1].attributes == expected


def test_llm_missing_usage_and_tool_response_preserve_behavior(mock_openai: MagicMock) -> None:
    mock_openai.chat.completions.create.return_value = fake_completion(
        tool_calls=[fake_tool_call("x", "{bad")], content="private-thought")
    llm, hook = monitored_client(mock_openai)
    assert llm.chat([], tools=[{}]) == {
        "thought": "private-thought", "tool_call": {"name": "x", "arguments": {}, "id": "call_1"}}
    assert hook.events[-1].attributes == {}
    assert "private" not in repr(hook.events)


@pytest.mark.parametrize("translation_failure", [False, True])
def test_llm_errors_are_reraised_without_sensitive_details(mock_openai: MagicMock, translation_failure: bool) -> None:
    error = RuntimeError("private-provider-details")
    if translation_failure:
        completion = fake_completion()
        completion.choices = []
        mock_openai.chat.completions.create.return_value = completion
        expected_type = IndexError
    else:
        mock_openai.chat.completions.create.side_effect = error
        expected_type = RuntimeError
    llm, hook = monitored_client(mock_openai)
    with pytest.raises(expected_type) as caught:
        llm.chat([])
    if not translation_failure:
        assert caught.value is error
    assert [e.event_name for e in hook.events] == ["llm.call.start", "llm.call.error"]
    assert hook.events[-1].duration_seconds == 2.5
    assert hook.events[-1].attributes == {"error_type": expected_type.__name__}
    assert hook.events[-1].outcome == "failure"
    assert "private" not in repr(hook.events)


@pytest.mark.parametrize("site", ["id_provider", "clock", "monotonic_clock", "hook", "construction"])
@pytest.mark.parametrize("agent_failure", [False, True])
def test_llm_monitoring_failures_preserve_execution(mock_openai: MagicMock, site: str, agent_failure: bool) -> None:
    def broken() -> object:
        raise RuntimeError("private-monitoring-details")

    class BrokenHook:
        def emit(self, event: LifecycleEvent) -> None:
            broken()

    diagnostics: list[str] = []
    overrides = {site: broken} if site in {"id_provider", "clock", "monotonic_clock"} else (
        {"hook": BrokenHook()} if site == "hook" else {"agent_id": ""})
    llm, _ = monitored_client(mock_openai, diagnostic=diagnostics.append, **overrides)
    error = RuntimeError("private-provider-details")
    if agent_failure:
        mock_openai.chat.completions.create.side_effect = error
        with pytest.raises(RuntimeError) as caught:
            llm.chat([])
        assert caught.value is error
    else:
        mock_openai.chat.completions.create.return_value = fake_completion(content="ok")
        assert llm.chat([]) == {"final_answer": "ok"}
    assert diagnostics and set(diagnostics) == {"Monitoring failure; agent execution continues."}


@pytest.mark.parametrize("hook", [None, NoOpHook()])
def test_llm_optional_and_noop_hooks(mock_openai: MagicMock, hook: object) -> None:
    def broken() -> object:
        raise RuntimeError("private-details")

    mock_openai.chat.completions.create.return_value = fake_completion(content="ok")
    llm = LLMClient(api_key="private-key", hook=hook, diagnostic=lambda _: broken(),
                    id_provider=broken if hook is None else None)
    assert llm.chat([]) == {"final_answer": "ok"}


def test_llm_unavailable_token_property_does_not_drop_finish(mock_openai: MagicMock) -> None:
    class Usage:
        total_tokens = 3
        completion_tokens = 3

        @property
        def prompt_tokens(self) -> int:
            raise RuntimeError("private-details")

    completion = fake_completion(content="ok")
    completion.usage = Usage()
    mock_openai.chat.completions.create.return_value = completion
    diagnostics: list[str] = []
    llm, hook = monitored_client(mock_openai, diagnostic=diagnostics.append)
    assert llm.chat([]) == {"final_answer": "ok"}
    assert hook.events[-1].attributes == {"output_tokens": 3}
    assert diagnostics == ["Monitoring failure; agent execution continues."]


def test_llm_terminal_clock_and_diagnostic_failures_preserve_provider_error(mock_openai: MagicMock) -> None:
    times = iter([datetime(2026, 9, 15, tzinfo=timezone.utc)])

    def diagnostic(_: str) -> None:
        raise RuntimeError("private-diagnostic-details")

    error = RuntimeError("private-provider-details")
    mock_openai.chat.completions.create.side_effect = error
    llm, hook = monitored_client(mock_openai, clock=lambda: next(times), diagnostic=diagnostic)
    with pytest.raises(RuntimeError) as caught:
        llm.chat([])
    assert caught.value is error
    assert [e.event_name for e in hook.events] == ["llm.call.start"]


def test_explicit_parent_context_and_standalone_calls_do_not_leak(mock_openai: MagicMock) -> None:
    mock_openai.chat.completions.create.return_value = fake_completion(content="ok")
    ids = iter(["child-1", "standalone-run", "standalone-span"])
    hook = RecordingHook()
    llm = LLMClient(api_key="private", hook=hook, id_provider=lambda: next(ids))
    parent = RunContext("parent-run", "parent-span", None, "agent-a")
    assert llm.chat_with_context([], context=parent) == {"final_answer": "ok"}
    assert llm.chat([]) == {"final_answer": "ok"}
    start, finish, standalone_start, standalone_finish = hook.events
    for event in (start, finish):
        assert (event.run_id, event.span_id, event.parent_span_id, event.agent_id) == (
            "parent-run", "child-1", "parent-span", "agent-a")
    for event in (standalone_start, standalone_finish):
        assert (event.run_id, event.span_id, event.parent_span_id, event.agent_id) == (
            "standalone-run", "standalone-span", None, "parent")
    assert "context" not in mock_openai.chat.completions.create.call_args.kwargs


def test_react_retry_llm_spans_share_parent_run(mock_openai: MagicMock) -> None:
    class Registry:
        def schemas(self) -> list[dict]:
            return [{"type": "function"}]

        def list_tools(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="example_tool")]

    mock_openai.chat.completions.create.side_effect = [
        fake_completion(content="example_tool()"), fake_completion(content="done")]
    hook = RecordingHook()
    parent_ids = iter(["run-a", "parent-a"])
    child_ids = iter(["llm-a", "llm-b"])
    parent_ticks = iter([0.0, 5.0])
    child_ticks = iter([1.0, 2.0, 3.0, 4.0])
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    llm = LLMClient(api_key="private", hook=hook, id_provider=lambda: next(child_ids),
                    clock=lambda: now, monotonic_clock=lambda: next(child_ticks))
    loop = ReactLoop(llm, Registry(), hook=hook, agent_id="agent-a",
                     id_provider=lambda: next(parent_ids), clock=lambda: now,
                     monotonic_clock=lambda: next(parent_ticks))
    assert loop.run("private-message") == "done"
    assert [e.event_name for e in hook.events] == [
        "agent.run.start", "llm.call.start", "llm.call.finish",
        "llm.call.start", "llm.call.finish", "agent.run.finish"]
    assert {e.run_id for e in hook.events} == {"run-a"}
    assert {e.agent_id for e in hook.events} == {"agent-a"}
    assert [e.span_id for e in hook.events] == ["parent-a", "llm-a", "llm-a", "llm-b", "llm-b", "parent-a"]
    assert [e.parent_span_id for e in hook.events] == [None, "parent-a", "parent-a", "parent-a", "parent-a", None]
    assert hook.events[-1].duration_seconds == 5.0
    assert [e.duration_seconds for e in hook.events if e.event_name == "llm.call.finish"] == [1.0, 1.0]
    assert mock_openai.chat.completions.create.call_args.kwargs["tool_choice"] == "required"


def test_correlated_provider_error_keeps_both_terminal_events(mock_openai: MagicMock) -> None:
    hook = RecordingHook()
    error = RuntimeError("private-details")
    mock_openai.chat.completions.create.side_effect = error
    llm = LLMClient(api_key="private", hook=hook, id_provider=lambda: "child")
    ids = iter(["run", "parent-span"])
    loop = ReactLoop(llm, hook=hook, id_provider=lambda: next(ids))
    with pytest.raises(RuntimeError) as caught:
        loop.run("private-message")
    assert caught.value is error
    assert [e.event_name for e in hook.events] == [
        "agent.run.start", "llm.call.start", "llm.call.error", "agent.run.error"]
    assert {e.run_id for e in hook.events} == {"run"}
    assert hook.events[2].parent_span_id == "parent-span"


def test_invalid_supplied_context_fails_open(mock_openai: MagicMock) -> None:
    mock_openai.chat.completions.create.return_value = fake_completion(content="ok")
    diagnostics: list[str] = []
    llm, hook = monitored_client(mock_openai, diagnostic=diagnostics.append)
    assert llm.chat_with_context([], context=object()) == {"final_answer": "ok"}
    assert hook.events == []
    assert diagnostics == ["Monitoring failure; agent execution continues."]


def test_react_preserves_llm_subclass_chat_override(mock_openai: MagicMock) -> None:
    class LegacyClient(LLMClient):
        def chat(self, messages: list[dict], tools: object = None, tool_choice: str = "auto") -> dict:
            return {"final_answer": "legacy override"}
    hook = RecordingHook()
    llm = LegacyClient(api_key="private", hook=hook)
    assert ReactLoop(llm, hook=hook).run("private-prompt") == "legacy override"
    mock_openai.chat.completions.create.assert_not_called()
    assert [e.event_name for e in hook.events] == ["agent.run.start", "agent.run.finish"]

def test_context_usage_percent_is_zero_before_any_call(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    llm = LLMClient(context_window_tokens=1000)

    assert llm.context_usage_percent() == 0.0


def test_context_usage_percent_reflects_the_most_recent_calls_prompt_size(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    llm = LLMClient(context_window_tokens=1000)

    mock_openai.chat.completions.create.return_value = fake_completion(content="a", prompt_tokens=250)
    llm.chat([{"role": "user", "content": "hi"}])

    assert llm.context_usage_percent() == 25.0


def test_context_usage_percent_is_not_cumulative_across_calls(monkeypatch, mock_openai):
    """Unlike total_tokens, this must reflect the LAST call's prompt size,
    not a running sum - it answers "how full is my context right now"."""
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    llm = LLMClient(context_window_tokens=1000)

    mock_openai.chat.completions.create.return_value = fake_completion(content="a", prompt_tokens=100)
    llm.chat([{"role": "user", "content": "hi"}])
    mock_openai.chat.completions.create.return_value = fake_completion(content="b", prompt_tokens=120)
    llm.chat([{"role": "user", "content": "hi again"}])

    assert llm.context_usage_percent() == 12.0


def test_default_context_window_matches_the_deployed_model():
    """Regression guard: this number came from a live empirical test
    against a vLLM backend (see docs/architecture.md) - if it silently drifts,
    compaction would trigger at the wrong point."""
    assert inspect.signature(LLMClient).parameters["context_window_tokens"].default == 262144


def test_context_window_exceeded_is_translated_to_our_own_error_type(monkeypatch, mock_openai):
    """The real error from the provider, verified live: 'litellm.ContextWindow
    ExceededError: ... This model's maximum context length is 262144
    tokens. However, you requested ... for a total of at least ... input
    tokens.' ReactLoop needs a stable exception type to catch, not a raw
    BadRequestError it would have to string-match itself."""
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.side_effect = fake_bad_request_error(
        "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
        "Hosted_vllmException - {\"error\":{\"message\":\"This model's maximum context length is "
        "262144 tokens. However, you requested 5 output tokens and your prompt contains at least "
        "262140 input tokens, for a total of at least 262145 tokens.\"}}"
    )
    llm = LLMClient()

    with pytest.raises(ContextWindowExceededError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_other_bad_request_errors_are_not_mistaken_for_context_overflow(monkeypatch, mock_openai):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    mock_openai.chat.completions.create.side_effect = fake_bad_request_error("invalid 'temperature': must be between 0 and 2")
    llm = LLMClient()

    with pytest.raises(BadRequestError):
        llm.chat([{"role": "user", "content": "hi"}])
