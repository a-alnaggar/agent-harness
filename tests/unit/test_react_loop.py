from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agent.engine.llm_client import ContextWindowExceededError
from agent.engine.react_loop import ReactLoop
from agent.observability import LifecycleEvent
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
        return f"echoed: {text}"


class BoomTool(Tool):
    """Raises a plain (non-ToolError) exception when called with the wrong
    arguments — simulates the LLM passing a bad argument name."""

    name = "boom"
    description = "Requires 'value', used to test bad-argument handling."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def run(self, *, value: str) -> str:
        return value


class FakeLLM:
    """Returns canned responses in order, one per call() invocation."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice})
        return self._responses.pop(0)


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


class FailingLLM:
    def chat(
        self,
        messages: object,
        tools: object = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        raise RuntimeError("provider unavailable")

class FakeLLMWithContext(FakeLLM):
    """FakeLLM with a directly-settable context_usage_percent(), for
    testing compaction triggers without needing a real usage-tracking
    LLMClient."""

    def __init__(self, responses, percent=0.0):
        super().__init__(responses)
        self.percent = percent

    def context_usage_percent(self):
        return self.percent


class FakeLLMThatOverflowsOnce(FakeLLM):
    """Raises ContextWindowExceededError on its first call (simulating the
    provider rejecting an over-budget request), then returns the scripted
    responses normally from the retry onward."""

    def __init__(self, responses):
        super().__init__(responses)
        self._raised = False

    def chat(self, messages, tools=None, tool_choice="auto"):
        if not self._raised:
            self._raised = True
            raise ContextWindowExceededError("maximum context length is 262144 tokens")
        return super().chat(messages, tools=tools, tool_choice=tool_choice)


def seeded_history(system_content, n_user_turns):
    """A plausible-looking message history with N complete user/assistant
    turns, for tests that need pre-existing history to compact."""
    messages = [{"role": "system", "content": system_content}]
    for i in range(n_user_turns):
        messages.append({"role": "user", "content": f"turn {i} question"})
        messages.append({"role": "assistant", "content": f"turn {i} answer"})
    return messages


def make_registry():
    registry = ToolRegistry()
    for tool in (EchoTool(), BoomTool()):
        # These two exist to exercise loop mechanics, not permissions (those
        # have their own fixtures below), so they are marked harmless here -
        # an unmarked tool defaults to CONFIRM and would be refused with no
        # way to ask.
        tool.permission_tier = PermissionTier.REGULAR
        registry.register(tool)
    return registry


class RegularTool(Tool):
    name = "regular_tool"
    description = "A REGULAR-tier tool, auto-executes."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_tier = PermissionTier.REGULAR

    def __init__(self):
        self.call_count = 0

    def run(self, **kwargs) -> str:
        self.call_count += 1
        return "regular ran"


class ConfirmTool(Tool):
    name = "confirm_tool"
    description = "A CONFIRM-tier tool: asks once, 'always' persists for the session."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_tier = PermissionTier.CONFIRM

    def __init__(self):
        self.call_count = 0

    def run(self, **kwargs) -> str:
        self.call_count += 1
        return "confirm ran"


class StrictTool(Tool):
    name = "strict_tool"
    description = "A STRICT-tier tool: asks every single call, no 'always' shortcut."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_tier = PermissionTier.STRICT

    def __init__(self):
        self.call_count = 0

    def run(self, **kwargs) -> str:
        self.call_count += 1
        return "strict ran"


def make_permission_registry():
    registry = ToolRegistry()
    registry.register(RegularTool())
    registry.register(ConfirmTool())
    registry.register(StrictTool())
    return registry


class ScriptedConfirm:
    """Records every call it receives and returns scripted decisions in
    order, so tests can assert exactly what the gate asked and answer it
    deterministically."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls: list[tuple[str, dict, str]] = []

    def __call__(self, tool_name, arguments, description):
        self.calls.append((tool_name, arguments, description))
        return self._decisions.pop(0)


def test_final_answer_immediately():
    llm = FakeLLM([{"final_answer": "hi there"}])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("hello")

    assert result == "hi there"
    assert loop.messages[-1] == {"role": "assistant", "content": "hi there"}


def test_agent_run_emits_deterministic_start_and_success_events() -> None:
    hook = RecordingHook()
    started_at = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
    times = iter([started_at, started_at + timedelta(seconds=1.5)])
    ids = iter(["run-123", "span-456"])
    loop = ReactLoop(
        FakeLLM([{"final_answer": "hi there"}]),
        tool_registry=make_registry(),
        hook=hook,
        agent_id="parent-agent",
        id_provider=lambda: next(ids),
        clock=lambda: next(times),
        monotonic_clock=iter([10.0, 11.5]).__next__,
    )

    result = loop.run("private user prompt")

    assert result == "hi there"
    assert [event.event_name for event in hook.events] == ["agent.run.start", "agent.run.finish"]
    start, finish = hook.events
    assert start.run_id == finish.run_id == "run-123"
    assert start.span_id == finish.span_id == "span-456"
    assert start.agent_id == finish.agent_id == "parent-agent"
    assert finish.outcome == "success"
    assert finish.duration_seconds == 1.5
    assert start.attributes == finish.attributes == {"max_steps": 10}
    assert "private user prompt" not in repr(hook.events)


def test_agent_run_emits_finish_with_failure_for_controlled_error() -> None:
    hook = RecordingHook()
    started_at = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
    times = iter([started_at, started_at + timedelta(seconds=0.25)])
    loop = ReactLoop(
        FakeLLM([{}]),
        tool_registry=make_registry(),
        hook=hook,
        id_provider=iter(["run-1", "span-1"]).__next__,
        clock=lambda: next(times),
        monotonic_clock=iter([10.0, 10.25]).__next__,
    )

    result = loop.run("do something")

    assert result == "Error: invalid LLM response."
    assert [event.event_name for event in hook.events] == ["agent.run.start", "agent.run.finish"]
    assert hook.events[-1].outcome == "failure"
    assert hook.events[-1].duration_seconds == 0.25


def test_agent_run_emits_error_and_preserves_exception_behavior() -> None:
    hook = RecordingHook()
    started_at = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)
    times = iter([started_at, started_at + timedelta(seconds=2)])
    loop = ReactLoop(
        FailingLLM(),
        hook=hook,
        id_provider=iter(["run-1", "span-1"]).__next__,
        clock=lambda: next(times),
        monotonic_clock=iter([10.0, 12.0]).__next__,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        loop.run("private user prompt")

    assert [event.event_name for event in hook.events] == ["agent.run.start", "agent.run.error"]
    error = hook.events[-1]
    assert error.outcome == "failure"
    assert error.duration_seconds == 2
    assert error.attributes == {"error_type": "RuntimeError"}
    assert "private user prompt" not in repr(hook.events)


@pytest.mark.parametrize("answer", ["Error handling is useful.", "Stopped: is a label.", ""])
def test_final_answer_status_does_not_depend_on_text(answer: str) -> None:
    hook = RecordingHook()
    loop = ReactLoop(FakeLLM([{"final_answer": answer}]), hook=hook)
    assert loop.run("hello") == answer
    assert hook.events[-1].outcome == "success"


def test_monotonic_duration_survives_backward_wall_clock_change() -> None:
    hook = RecordingHook()
    start = datetime(2026, 9, 15, 9, tzinfo=timezone.utc)
    loop = ReactLoop(
        FakeLLM([{"final_answer": "done"}]), hook=hook,
        clock=iter([start, start - timedelta(hours=1)]).__next__,
        monotonic_clock=iter([100.0, 102.5]).__next__,
    )
    assert loop.run("hello") == "done"
    assert hook.events[-1].timestamp == start - timedelta(hours=1)
    assert hook.events[-1].duration_seconds == 2.5


@pytest.mark.parametrize("site", ["id_provider", "clock", "monotonic_clock", "hook", "construction"])
@pytest.mark.parametrize("agent_failure", [False, True])
def test_monitoring_failures_cannot_alter_agent_execution(site: str, agent_failure: bool) -> None:
    original = RuntimeError("original agent failure")
    diagnostics: list[str] = []

    def broken_provider() -> Any:
        raise RuntimeError("private monitoring exception")

    class BrokenHook:
        def emit(self, event: LifecycleEvent) -> None:
            broken_provider()

    class AgentLLM:
        def chat(self, *args: Any, **kwargs: Any) -> dict[str, str]:
            if agent_failure:
                raise original
            return {"final_answer": "unchanged answer"}

    options: dict[str, Any] = {"hook": RecordingHook(), "diagnostic": diagnostics.append}
    if site == "hook":
        options["hook"] = BrokenHook()
    elif site == "construction":
        options["agent_id"] = ""
    else:
        options[site] = broken_provider
    loop = ReactLoop(AgentLLM(), **options)
    if agent_failure:
        with pytest.raises(RuntimeError) as captured:
            loop.run("private prompt")
        assert captured.value is original
    else:
        assert loop.run("private prompt") == "unchanged answer"
    assert diagnostics
    assert set(diagnostics) == {"Monitoring failure; agent execution continues."}


def test_terminal_clock_and_diagnostic_failures_preserve_original_exception() -> None:
    times = iter([datetime(2026, 9, 15, tzinfo=timezone.utc)])

    def broken_diagnostic(message: str) -> None:
        raise RuntimeError("diagnostic failed")

    loop = ReactLoop(
        FailingLLM(), hook=RecordingHook(), clock=times.__next__, diagnostic=broken_diagnostic,
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        loop.run("hello")


def test_no_hook_does_not_call_monitoring_providers() -> None:
    def broken_provider() -> Any:
        raise AssertionError("monitoring provider should not run")

    loop = ReactLoop(
        FakeLLM([{"final_answer": "done"}]), id_provider=broken_provider,
        clock=broken_provider, monotonic_clock=broken_provider,
    )
    assert loop.run("hello") == "done"


def test_no_tool_registry_does_not_crash_on_plain_conversation():
    llm = FakeLLM([{"final_answer": "sure"}])
    loop = ReactLoop(llm)  # no tool_registry, matches the constructor default

    result = loop.run("hello")

    assert result == "sure"
    assert llm.calls[0]["tools"] is None


def test_tool_call_then_final_answer():
    llm = FakeLLM([
        {"thought": "let's echo", "tool_call": {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("please echo hi")

    assert result == "done"

    # the tool actually ran
    tool_message = next(m for m in loop.messages if m["role"] == "tool")
    assert tool_message["content"] == "echoed: hi"

    # message history is a valid OpenAI-style tool-call/tool-result pair
    assistant_message = next(m for m in loop.messages if m.get("tool_calls"))
    call_id = assistant_message["tool_calls"][0]["id"]
    assert tool_message["tool_call_id"] == call_id
    assert assistant_message["tool_calls"][0]["function"]["name"] == "echo"


def test_second_llm_call_receives_valid_tool_call_history():
    """The exact regression case for the original str(response)/no-id bug:
    the second chat() call must see a well-formed tool_call/tool message
    pair, not a stringified dict with no tool_call_id."""
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())
    loop.run("please echo hi")

    second_call_messages = llm.calls[1]["messages"]
    assistant_msg = next(m for m in second_call_messages if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")

    assert tool_msg["tool_call_id"] == assistant_msg["tool_calls"][0]["id"]
    assert isinstance(assistant_msg["tool_calls"][0]["function"]["arguments"], str)


def test_unknown_tool_is_reported_to_the_model_which_can_recover():
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "does_not_exist", "arguments": {}}},
        {"tool_call": {"id": "call_2", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "recovered"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("do the thing")

    assert result == "recovered"
    first_observation = next(m for m in loop.messages if m.get("role") == "tool")["content"]
    assert first_observation.startswith("Error: unknown tool 'does_not_exist'")
    assert "echo" in first_observation  # names the tools that do exist


def test_tool_call_without_registry_returns_error():
    llm = FakeLLM([{"tool_call": {"id": "call_1", "name": "echo", "arguments": {}}}])
    loop = ReactLoop(llm, tool_registry=None)

    result = loop.run("do the thing")

    assert result == "Error: no tool registry available."


def test_bad_tool_arguments_are_reported_to_the_model_which_can_retry():
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "boom", "arguments": {"wrong_key": "x"}}},
        {"tool_call": {"id": "call_2", "name": "boom", "arguments": {"value": "x"}}},
        {"final_answer": "fixed the call"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("trigger boom")

    assert result == "fixed the call"
    first_observation = next(m for m in loop.messages if m.get("role") == "tool")["content"]
    assert first_observation.startswith("Error while executing 'boom'")


def test_a_model_that_never_fixes_its_calls_is_still_bounded_by_max_steps():
    # a different wrong name each time (the same one twice ends the run earlier)
    llm = FakeLLM([{"tool_call": {"id": f"c{i}", "name": f"does_not_exist_{i}", "arguments": {}}} for i in range(5)])
    loop = ReactLoop(llm, tool_registry=make_registry(), max_steps=3)

    assert loop.run("go").startswith("Stopped:")
    assert len(llm.calls) == 3


def test_max_steps_reached():
    responses = [{"tool_call": {"id": f"call_{i}", "name": "echo", "arguments": {"text": "x"}}} for i in range(5)]
    llm = FakeLLM(responses)
    loop = ReactLoop(llm, tool_registry=make_registry(), max_steps=3)

    result = loop.run("loop forever")

    assert result == "Stopped: maximum number of steps reached."
    assert len(llm.calls) == 3


def test_invalid_llm_response_returns_error():
    llm = FakeLLM([{}])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("confuse the agent")

    assert result == "Error: invalid LLM response."


def test_empty_final_answer_is_not_treated_as_invalid():
    llm = FakeLLM([{"final_answer": ""}])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("say nothing")

    assert result == ""


def test_narrated_tool_call_triggers_a_forced_retry():
    """The model sometimes writes out a tool call as text (e.g.
    "echo(text='hi')") instead of invoking it. One retry with
    tool_choice="required" should recover a real call."""
    llm = FakeLLM([
        {"final_answer": "Action: echo(text='hi')"},
        {"tool_call": {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("say hi")

    assert result == "done"
    assert llm.calls[0]["tool_choice"] == "auto"
    assert llm.calls[1]["tool_choice"] == "required"
    tool_messages = [m for m in loop.messages if m.get("role") == "tool"]
    assert tool_messages == [{"role": "tool", "tool_call_id": "call_1", "content": "echoed: hi"}]


def test_plain_final_answer_does_not_trigger_a_retry():
    llm = FakeLLM([{"final_answer": "Sure, happy to help with that."}])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("hello")

    assert result == "Sure, happy to help with that."
    assert len(llm.calls) == 1


def test_step_callback_emits_intermediate_steps():
    """Verify that intermediate ReAct steps (thought, tool calls, and observations)
    are emitted to the step_callback during execution."""
    llm = FakeLLM([
        {
            "thought": "I need to echo the message first",
            "tool_call": {"id": "call_1", "name": "echo", "arguments": {"text": "hello"}},
        },
        {"final_answer": "echoed successfully"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry())

    steps = []
    result = loop.run("test step callback", step_callback=steps.append)

    assert result == "echoed successfully"
    assert len(steps) == 1
    step = steps[0]
    assert step.step_number == 1
    assert step.thought == "I need to echo the message first"
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].name == "echo"
    assert len(step.observations) == 1
    assert step.observations[0].content == "echoed: hello"
    assert not step.observations[0].is_error


def test_tool_router_forces_real_tool_call_before_llm_answer():
    llm = FakeLLM([{"final_answer": "done after delegation"}])
    loop = ReactLoop(
        llm,
        tool_registry=make_registry(),
        tool_router=lambda _: {
            "name": "echo",
            "arguments": {"text": "persist this"},
        },
    )

    result = loop.run("remember this")

    assert result == "done after delegation"
    assert loop.messages[-2]["content"] == "echoed: persist this"
    assert len(llm.calls) == 1


def test_required_successful_tool_rejects_false_success():
    llm = FakeLLM([
        {"final_answer": "I saved it."},
        {"final_answer": "I definitely saved it."},
    ])
    loop = ReactLoop(
        llm,
        tool_registry=make_registry(),
        require_successful_tool=True,
    )

    result = loop.run("save this")

    assert result == "Error: this task requires a successful tool call before completion."
    assert llm.calls[1]["tool_choice"] == "required"


def test_system_prompt_provider_refreshes_between_turns():
    current = {"prompt": "profile: Ahmed"}
    llm = FakeLLM([
        {"final_answer": "first"},
        {"final_answer": "second"},
    ])
    loop = ReactLoop(
        llm,
        tool_registry=make_registry(),
        system_prompt_provider=lambda: current["prompt"],
    )

    loop.run("first turn")
    current["prompt"] = "profile: Mohammed"
    loop.run("second turn")

    assert llm.calls[0]["messages"][0]["content"] == "profile: Ahmed"
    assert llm.calls[1]["messages"][0]["content"] == "profile: Mohammed"


def test_compaction_triggers_at_threshold_and_summarizes_older_history():
    llm = FakeLLMWithContext(
        responses=[
            {"final_answer": "SUMMARY: earlier turns discussed X, Y, Z"},  # the compaction call
            {"final_answer": "done"},  # the real step, after compaction
        ],
        percent=85.0,  # >= default 80% threshold
    )
    loop = ReactLoop(llm, tool_registry=make_registry(), compaction_keep_last_user_turns=2)
    # 3 prior user turns already in history + the new one about to be
    # added by run() = 4 total, more than enough to have something to
    # compact ahead of the last 2 kept turns.
    loop.messages = seeded_history("base system prompt", n_user_turns=3)

    result = loop.run("new task")

    assert result == "done"
    assert len(llm.calls) == 2
    # first call was the compaction request: system + one user message
    # asking for a summary, no tools
    compaction_call = llm.calls[0]
    assert compaction_call["tools"] is None
    assert compaction_call["messages"][0] == {"role": "system", "content": "base system prompt"}
    assert "turn 0 question" in compaction_call["messages"][1]["content"]

    # messages were actually replaced: system message first, then a
    # summary-carrying message, then the preserved recent turns
    assert loop.messages[0] == {"role": "system", "content": "base system prompt"}
    assert "SUMMARY: earlier turns discussed X, Y, Z" in loop.messages[1]["content"]
    assert "[Earlier conversation summarized" in loop.messages[1]["content"]


def test_compaction_keeps_the_configured_number_of_recent_user_turns_verbatim():
    llm = FakeLLMWithContext(
        responses=[{"final_answer": "summary"}, {"final_answer": "done"}],
        percent=90.0,
    )
    # keep=2: the newly-appended "new task" turn counts as the most recent
    # of the kept turns, so turn 2 (the previous most recent) should also
    # survive alongside it; turns 0 and 1 should not.
    loop = ReactLoop(llm, tool_registry=make_registry(), compaction_keep_last_user_turns=2)
    loop.messages = seeded_history("sys", n_user_turns=3)

    loop.run("new task")

    contents = [m.get("content") for m in loop.messages]
    assert "turn 0 question" not in contents
    assert "turn 1 question" not in contents
    assert "turn 2 question" in contents
    assert "turn 2 answer" in contents
    assert "new task" in contents


def test_compaction_does_not_trigger_below_threshold():
    llm = FakeLLMWithContext(responses=[{"final_answer": "done"}], percent=50.0)
    loop = ReactLoop(llm, tool_registry=make_registry())
    loop.messages = seeded_history("sys", n_user_turns=3)

    result = loop.run("new task")

    assert result == "done"
    assert len(llm.calls) == 1  # no extra compaction call


def test_compaction_is_a_noop_without_enough_history_to_compact():
    """Even at/above threshold, there's nothing safe to summarize when
    there aren't more user turns than the configured keep-count yet -
    e.g. right at session start."""
    llm = FakeLLMWithContext(responses=[{"final_answer": "done"}], percent=95.0)
    loop = ReactLoop(llm, tool_registry=make_registry(), compaction_keep_last_user_turns=2)
    # only 1 prior turn, fewer than the 2 turns compaction is configured to keep

    result = loop.run("first ever task")

    assert result == "done"
    assert len(llm.calls) == 1  # compaction was skipped, no extra call


def test_on_compaction_callback_receives_the_triggering_percent_and_counts():
    llm = FakeLLMWithContext(
        responses=[{"final_answer": "summary"}, {"final_answer": "done"}],
        percent=88.0,
    )
    loop = ReactLoop(llm, tool_registry=make_registry(), compaction_keep_last_user_turns=1)
    loop.messages = seeded_history("sys", n_user_turns=3)

    events = []
    loop.run("new task", on_compaction=lambda *args: events.append(args))

    assert len(events) == 1
    percent, before_count, after_count = events[0]
    assert percent == 88.0
    assert after_count < before_count


def test_on_compaction_callback_not_called_when_compaction_is_a_noop():
    llm = FakeLLMWithContext(responses=[{"final_answer": "done"}], percent=95.0)
    loop = ReactLoop(llm, tool_registry=make_registry())

    events = []
    loop.run("first ever task", on_compaction=lambda *args: events.append(args))

    assert events == []


def test_a_broken_on_compaction_callback_does_not_crash_the_loop():
    llm = FakeLLMWithContext(
        responses=[{"final_answer": "summary"}, {"final_answer": "done"}],
        percent=85.0,
    )
    loop = ReactLoop(llm, tool_registry=make_registry(), compaction_keep_last_user_turns=1)
    loop.messages = seeded_history("sys", n_user_turns=3)

    def broken_callback(*args):
        raise RuntimeError("renderer exploded")

    result = loop.run("new task", on_compaction=broken_callback)

    assert result == "done"


def test_context_window_exceeded_triggers_emergency_compaction_and_retries():
    """The hard bound: even if the proactive percentage-based check
    (percent=0 here, i.e. it wouldn't have fired) missed it, a real
    rejection from the provider must still recover instead of bricking
    the session."""
    llm = FakeLLMThatOverflowsOnce(responses=[{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=make_registry())
    loop.messages = seeded_history("sys", n_user_turns=3)

    result = loop.run("new task")

    assert result == "done"
    # emergency compaction dropped everything except system + current turn
    assert loop.messages[0] == {"role": "system", "content": "sys"}
    assert "context limit was hit" in loop.messages[1]["content"]
    contents = [m.get("content") for m in loop.messages]
    assert "turn 0 question" not in contents
    assert "turn 2 question" not in contents
    assert "new task" in contents


def test_emergency_compaction_never_calls_the_llm_to_summarize():
    """Unlike normal _compact(), this must not risk a second failure from
    the same cause - verified by checking only ONE real chat() call
    happened (the retry), no extra summarization call in between."""
    llm = FakeLLMThatOverflowsOnce(responses=[{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=make_registry())
    loop.messages = seeded_history("sys", n_user_turns=3)

    loop.run("new task")

    assert len(llm.calls) == 1  # only the successful retry was recorded


def test_on_compaction_is_notified_on_emergency_compaction_too():
    llm = FakeLLMThatOverflowsOnce(responses=[{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=make_registry())
    loop.messages = seeded_history("sys", n_user_turns=3)

    events = []
    loop.run("new task", on_compaction=lambda *args: events.append(args))

    assert len(events) == 1
    percent, before_count, after_count = events[0]
    assert percent == 100.0
    assert after_count < before_count


def test_context_window_exceeded_recovers_even_on_the_very_first_task():
    """Overflow on the very first call ever (fixed tool-schema overhead
    alone already too big for the configured window) must still recover,
    not brick the session before it even starts."""
    llm = FakeLLMThatOverflowsOnce(responses=[{"final_answer": "done"}])
    loop = ReactLoop(llm, tool_registry=make_registry())

    result = loop.run("first ever task")

    assert result == "done"
    assert loop.messages[0]["role"] == "system"
    assert any(m.get("content") == "first ever task" for m in loop.messages)


def test_regular_tier_tool_never_asks_for_confirmation():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "regular_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)
    confirm = ScriptedConfirm([])  # would raise IndexError if ever called

    result = loop.run("do it", confirm_callback=confirm)

    assert result == "done"
    assert confirm.calls == []
    assert registry.get("regular_tool").call_count == 1


@pytest.mark.parametrize("tool_name", ["confirm_tool", "strict_tool"])
def test_no_confirm_callback_refuses_tools_that_need_confirmation(tool_name):
    """A missing UI must never quietly lift a permission: with nobody to
    ask, a tool that requires confirmation is refused, not run."""
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": tool_name, "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)

    result = loop.run("do it")

    assert result == "done"
    assert registry.get(tool_name).call_count == 0
    observation = next(m for m in loop.messages if m.get("role") == "tool")["content"]
    assert observation.startswith("Error:") and "no way to ask" in observation


def test_no_confirm_callback_still_runs_regular_tools():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "regular_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])

    ReactLoop(llm, tool_registry=registry).run("do it")

    assert registry.get("regular_tool").call_count == 1


def test_confirm_callback_receives_tool_name_arguments_and_description():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "confirm_tool", "arguments": {"x": 1}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)
    confirm = ScriptedConfirm(["once"])

    loop.run("do it", confirm_callback=confirm)

    assert confirm.calls == [("confirm_tool", {"x": 1}, ConfirmTool.description)]


def test_confirm_tier_once_does_not_persist_for_the_next_call():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "confirm_tool", "arguments": {}}},
        {"tool_call": {"id": "call_2", "name": "confirm_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry, max_steps=5)
    confirm = ScriptedConfirm(["once", "once"])

    result = loop.run("do it twice", confirm_callback=confirm)

    assert result == "done"
    assert len(confirm.calls) == 2  # asked both times
    assert registry.get("confirm_tool").call_count == 2


def test_confirm_tier_always_persists_for_the_rest_of_the_session():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "confirm_tool", "arguments": {}}},
        {"tool_call": {"id": "call_2", "name": "confirm_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry, max_steps=5)
    confirm = ScriptedConfirm(["always"])  # would raise IndexError if asked a 2nd time

    result = loop.run("do it twice", confirm_callback=confirm)

    assert result == "done"
    assert len(confirm.calls) == 1  # only asked once
    assert registry.get("confirm_tool").call_count == 2  # but ran both times


def test_confirm_tier_always_persists_across_separate_run_calls():
    """'the rest of the session' means the ReactLoop instance's lifetime,
    not just one run() call - a second, later task must not re-ask."""
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "confirm_tool", "arguments": {}}},
        {"final_answer": "done 1"},
        {"tool_call": {"id": "call_2", "name": "confirm_tool", "arguments": {}}},
        {"final_answer": "done 2"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)
    confirm = ScriptedConfirm(["always"])

    loop.run("first task", confirm_callback=confirm)
    result_2 = loop.run("second task", confirm_callback=confirm)

    assert result_2 == "done 2"
    assert len(confirm.calls) == 1


def test_strict_tier_asks_every_call_even_after_always():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "strict_tool", "arguments": {}}},
        {"tool_call": {"id": "call_2", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry, max_steps=5)
    confirm = ScriptedConfirm(["always", "always"])

    result = loop.run("do it twice", confirm_callback=confirm)

    assert result == "done"
    assert len(confirm.calls) == 2  # STRICT never skips, regardless of "always"
    assert registry.get("strict_tool").call_count == 2


def test_denying_confirmation_produces_an_error_observation_without_running_the_tool():
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "confirm_tool", "arguments": {}}},
        {"final_answer": "done"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)
    confirm = ScriptedConfirm(["deny"])

    result = loop.run("do it", confirm_callback=confirm)

    assert result == "done"
    assert registry.get("confirm_tool").call_count == 0  # never actually ran
    tool_message = next(m for m in loop.messages if m["role"] == "tool")
    assert tool_message["content"].startswith("Error: user denied permission to run 'confirm_tool'.")
    assert "Do not retry" in tool_message["content"]


def test_default_tool_permission_tier_is_confirm():
    """The base Tool class defaults to CONFIRM, not REGULAR - a tool that
    forgets to declare its tier is treated as needing approval, not
    trusted by omission."""
    assert EchoTool.permission_tier == PermissionTier.CONFIRM



# --- DENY tier and the shared, fail-closed policy ---------------------------

from agent.tools.permissions import PermissionPolicy  # noqa: E402


def one_call_then_done(tool_name):
    return FakeLLM([
        {"tool_call": {"id": "call_1", "name": tool_name, "arguments": {}}},
        {"final_answer": "done"},
    ])


@pytest.mark.parametrize("confirm", [None, ScriptedConfirm(["once"])], ids=["no-callback", "with-callback"])
def test_a_denied_tool_never_runs_and_is_never_asked_about(confirm):
    registry = make_permission_registry()
    loop = ReactLoop(
        one_call_then_done("regular_tool"), tool_registry=registry,
        permission_policy=PermissionPolicy(deny_tools=["regular_tool"]),
    )

    result = loop.run("do it", confirm_callback=confirm)

    assert result == "done"  # the run carries on; only the call was refused
    assert registry.get("regular_tool").call_count == 0
    if confirm is not None:
        assert confirm.calls == []  # policy decides - nobody is asked
    observation = next(m for m in loop.messages if m.get("role") == "tool")["content"]
    assert observation.startswith("Error:")
    assert "blocked by this deployment's permission policy" in observation
    assert "Do not retry" in observation


def test_deny_applies_to_strict_and_confirm_tools_as_well_as_regular_ones():
    for tool_name in ("confirm_tool", "strict_tool"):
        registry = make_permission_registry()
        confirm = ScriptedConfirm(["once"])
        loop = ReactLoop(
            one_call_then_done(tool_name), tool_registry=registry,
            permission_policy=PermissionPolicy(deny_tools=[tool_name]),
        )

        loop.run("do it", confirm_callback=confirm)

        assert registry.get(tool_name).call_count == 0
        assert confirm.calls == []


def test_tools_outside_the_deny_list_are_unaffected():
    registry = make_permission_registry()
    loop = ReactLoop(
        one_call_then_done("regular_tool"), tool_registry=registry,
        permission_policy=PermissionPolicy(deny_tools=["strict_tool"]),
    )

    loop.run("do it")

    assert registry.get("regular_tool").call_count == 1


def permission_finishes(hook):
    return [e for e in hook.events if e.event_name == "permission.request.finish"]


def test_a_policy_denial_is_recorded_as_a_permission_decision():
    hook = RecordingHook()
    loop = ReactLoop(
        one_call_then_done("regular_tool"), tool_registry=make_permission_registry(), hook=hook,
        permission_policy=PermissionPolicy(deny_tools=["regular_tool"]),
    )

    loop.run("do it")

    (finish,) = permission_finishes(hook)
    assert finish.attributes["decision"] == "policy_deny"
    assert finish.attributes["tier"] == "deny"
    assert finish.attributes["tool_name"] == "regular_tool"
    assert finish.outcome == "success"  # the permission machinery worked; the answer was "no"


def test_refusing_for_lack_of_a_way_to_ask_is_recorded_as_a_policy_decision_too():
    hook = RecordingHook()
    loop = ReactLoop(one_call_then_done("strict_tool"), tool_registry=make_permission_registry(), hook=hook)

    loop.run("do it")  # no confirm_callback

    (finish,) = permission_finishes(hook)
    assert finish.attributes["decision"] == "policy_deny"
    assert finish.attributes["tier"] == "strict"


def test_a_users_no_is_still_recorded_as_deny_not_policy_deny():
    hook = RecordingHook()
    loop = ReactLoop(one_call_then_done("strict_tool"), tool_registry=make_permission_registry(), hook=hook)

    loop.run("do it", confirm_callback=ScriptedConfirm(["deny"]))

    (finish,) = permission_finishes(hook)
    assert finish.attributes["decision"] == "deny"


def test_always_holds_across_loops_that_share_a_policy():
    registry = make_permission_registry()
    policy = PermissionPolicy()
    confirm = ScriptedConfirm(["always"])

    for _ in range(2):
        ReactLoop(one_call_then_done("confirm_tool"), tool_registry=registry, permission_policy=policy).run(
            "do it", confirm_callback=confirm
        )

    assert len(confirm.calls) == 1  # asked once; the second loop reused the approval
    assert registry.get("confirm_tool").call_count == 2


def test_loops_with_their_own_policies_do_not_share_approvals():
    registry = make_permission_registry()
    confirm = ScriptedConfirm(["always", "always"])

    for _ in range(2):
        ReactLoop(one_call_then_done("confirm_tool"), tool_registry=registry).run("do it", confirm_callback=confirm)

    assert len(confirm.calls) == 2


def test_always_never_carries_over_to_a_strict_tool_through_the_shared_policy():
    registry = make_permission_registry()
    policy = PermissionPolicy()
    confirm = ScriptedConfirm(["always", "once"])

    for _ in range(2):
        ReactLoop(one_call_then_done("strict_tool"), tool_registry=registry, permission_policy=policy).run(
            "do it", confirm_callback=confirm
        )

    assert len(confirm.calls) == 2


# --- a refused tool stops being offered for the rest of the run --------------

def offered_names(call):
    return {schema["function"]["name"] for schema in (call["tools"] or [])}


def test_a_refused_tool_is_no_longer_offered_to_the_model_for_the_rest_of_the_run():
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "done without it"},
    ])
    loop = ReactLoop(llm, tool_registry=make_permission_registry())

    loop.run("do it", confirm_callback=ScriptedConfirm(["deny"]))

    assert "strict_tool" in offered_names(llm.calls[0])
    assert "strict_tool" not in offered_names(llm.calls[1])
    assert "regular_tool" in offered_names(llm.calls[1])  # only the refused one goes away


@pytest.mark.parametrize("how", ["policy", "no_callback", "user_no"])
def test_every_kind_of_refusal_removes_the_tool_and_a_repeat_call_is_not_asked_again(how):
    registry = make_permission_registry()
    policy = PermissionPolicy(deny_tools=["strict_tool"]) if how == "policy" else PermissionPolicy()
    confirm = None if how == "no_callback" else ScriptedConfirm(["deny", "once"])
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "strict_tool", "arguments": {}}},
        {"tool_call": {"id": "c2", "name": "strict_tool", "arguments": {}}},  # model ignores the refusal
        {"final_answer": "gave up on it"},
    ])
    loop = ReactLoop(llm, tool_registry=registry, permission_policy=policy)

    result = loop.run("do it", confirm_callback=confirm)

    assert result == "gave up on it"
    assert registry.get("strict_tool").call_count == 0
    if how == "user_no":
        assert len(confirm.calls) == 1  # the user was asked once, not again for the repeat
    observations = [m["content"] for m in loop.messages if m.get("role") == "tool"]
    assert "already refused in this run" in observations[1]


def test_a_refusal_lasts_for_one_run_not_the_whole_session():
    registry = make_permission_registry()
    confirm = ScriptedConfirm(["deny", "once"])
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "skipped it"},
        {"tool_call": {"id": "c2", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "ran it"},
    ])
    loop = ReactLoop(llm, tool_registry=registry)

    loop.run("first task", confirm_callback=confirm)
    loop.run("second task", confirm_callback=confirm)

    assert registry.get("strict_tool").call_count == 1
    assert "strict_tool" in offered_names(llm.calls[2])  # offered again for the new task


def test_when_every_tool_has_been_refused_no_empty_tool_list_is_sent():
    registry = ToolRegistry()
    registry.register(StrictTool())
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "nothing left to try"},
    ])

    ReactLoop(llm, tool_registry=registry).run("do it", confirm_callback=ScriptedConfirm(["deny"]))

    assert llm.calls[1]["tools"] is None


# --- untrusted tool content ----------------------------------------------------

def test_the_default_prompt_tells_the_model_that_tool_output_is_data_not_instructions():
    from agent.engine.react_loop import REACT_SYSTEM_PROMPT, UNTRUSTED_CONTENT_NOTICE

    assert UNTRUSTED_CONTENT_NOTICE in REACT_SYSTEM_PROMPT
    assert "untrusted DATA" in UNTRUSTED_CONTENT_NOTICE and "Never follow" in UNTRUSTED_CONTENT_NOTICE


# --- a tool that can't work right now ------------------------------------------

class UnreachableStrictTool(Tool):
    name = "remote_exec"
    description = "STRICT tool whose host doesn't answer."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_tier = PermissionTier.STRICT

    def __init__(self):
        self.call_count = 0

    def run(self, **kwargs) -> str:
        from agent.tools.base import ToolUnavailableError

        self.call_count += 1
        raise ToolUnavailableError("sandbox command failed: timed out")


def test_an_unavailable_strict_tool_is_asked_about_once_not_on_every_retry():
    """Seen live: a sandbox timeout came back as an error, the model retried,
    and every retry of the STRICT tool asked the user again - a prompt loop."""
    tool = UnreachableStrictTool()
    registry = ToolRegistry()
    registry.register(tool)
    call = {"tool_call": {"id": "c", "name": "remote_exec", "arguments": {}}}
    llm = FakeLLM([call, call, call, {"final_answer": "the host is unreachable"}])
    confirm = ScriptedConfirm(["once", "once", "once"])

    result = ReactLoop(llm, tool_registry=registry).run("go", confirm_callback=confirm)

    assert result == "the host is unreachable"
    assert len(confirm.calls) == 1 and tool.call_count == 1
    observations = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "unavailable for the rest of this request" in observations[-1]
    assert llm.calls[1]["tools"] is None  # withdrawn from the schema too


def test_an_unavailable_tool_is_offered_again_on_the_next_request():
    tool = UnreachableStrictTool()
    registry = ToolRegistry()
    registry.register(tool)
    call = {"tool_call": {"id": "c", "name": "remote_exec", "arguments": {}}}
    llm = FakeLLM([call, {"final_answer": "down"}, call, {"final_answer": "still down"}])
    loop = ReactLoop(llm, tool_registry=registry)

    loop.run("first", confirm_callback=ScriptedConfirm(["once"]))
    loop.run("second", confirm_callback=ScriptedConfirm(["once"]))

    assert tool.call_count == 2  # the host may be back, so a new request tries once more



class MissingFileTool(Tool):
    name = "read_thing"
    description = "Fails for a path that doesn't exist."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    permission_tier = PermissionTier.REGULAR

    def __init__(self):
        self.call_count = 0
        self.exists: set[str] = set()

    def run(self, path: str) -> str:
        from agent.tools.base import ToolError

        self.call_count += 1
        if path not in self.exists:
            raise ToolError(f"'{path}' is not a file")
        return "contents"


def read(path, call_id="r"):
    return {"tool_call": {"id": call_id, "name": "read_thing", "arguments": {"path": path}}}


def test_the_same_failing_call_is_not_run_again_and_the_loop_ends():
    """Seen live: the model re-sent an identical failing read_file five times
    until max_steps."""
    tool = MissingFileTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM([read("x.txt"), read("x.txt"), {"final_answer": "x.txt does not exist"}])

    result = ReactLoop(llm, tool_registry=registry).run("read x.txt")

    assert result == "x.txt does not exist"
    assert tool.call_count == 1
    last = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "tool"][-1]
    assert "already failed in this request" in last and "not run again" in last


def test_a_different_argument_or_a_success_in_between_still_runs():
    tool = MissingFileTool()
    registry = ToolRegistry()
    registry.register(RegularTool())
    registry.register(tool)
    regular = {"tool_call": {"id": "g", "name": "regular_tool", "arguments": {}}}
    llm = FakeLLM([read("x.txt"), read("y.txt"), regular, read("x.txt"), {"final_answer": "done"}])

    ReactLoop(llm, tool_registry=registry).run("go")

    assert tool.call_count == 3  # other path runs; after a success the same call may work now



def test_after_a_denial_an_answer_that_names_the_tool_is_accepted_not_forced_into_another_call():
    """Seen live: after "n" to sandbox_exec the answer said "sandbox_exec was
    not approved", the narrated-call guard forced a tool call, and the model
    went on to send_email, delegate_task and read_file loops."""
    registry = make_permission_registry()
    strict = registry.get("strict_tool")
    llm = FakeLLM([
        {"tool_call": {"id": "s", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "strict_tool was not approved, so I did not run it."},
        {"tool_call": {"id": "x", "name": "confirm_tool", "arguments": {}}},  # must never be requested
    ])
    confirm = ScriptedConfirm(["deny"])

    result = ReactLoop(llm, tool_registry=registry).run("run it", confirm_callback=confirm)

    assert result == "strict_tool was not approved, so I did not run it."
    assert len(llm.calls) == 2 and all(c["tool_choice"] == "auto" for c in llm.calls)
    assert len(confirm.calls) == 1 and strict.call_count == 0


def test_a_summary_that_names_the_tools_it_used_is_not_treated_as_a_narrated_call():
    """After a real call, the final answer naturally says which tool was used;
    forcing another call there made the model run extra tools."""
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "I used echo to say hi."},
        {"tool_call": {"id": "c2", "name": "echo", "arguments": {"text": "again"}}},  # must never be requested
    ])

    result = ReactLoop(llm, tool_registry=make_registry()).run("say hi")

    assert result == "I used echo to say hi."
    assert len(llm.calls) == 2 and all(c["tool_choice"] == "auto" for c in llm.calls)


def test_a_call_that_raised_counts_as_a_failure_not_a_success():
    """A bad call ("Error while executing ...") used to count as success: it
    satisfied require_successful_tool and cleared the repeated-failure record."""
    llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "echo", "arguments": {}}},  # missing 'text' -> TypeError
        {"final_answer": "done"},
        {"tool_call": {"id": "c2", "name": "echo", "arguments": {"text": "hi"}}},
        {"final_answer": "done after a real call"},
    ])
    loop = ReactLoop(llm, tool_registry=make_registry(), require_successful_tool=True)

    result = loop.run("go")

    assert result == "done after a real call"
    assert llm.calls[2]["tool_choice"] == "required"


def test_require_successful_tool_accepts_an_answer_after_a_refusal():
    """The needed tool was denied; forcing a call would push the model to
    some other tool instead of saying so."""
    registry = make_permission_registry()
    llm = FakeLLM([
        {"tool_call": {"id": "s", "name": "strict_tool", "arguments": {}}},
        {"final_answer": "I could not do it: it was not approved."},
        {"tool_call": {"id": "x", "name": "regular_tool", "arguments": {}}},  # must never be requested
    ])
    loop = ReactLoop(llm, tool_registry=registry, require_successful_tool=True)

    result = loop.run("go", confirm_callback=ScriptedConfirm(["deny"]))

    assert result == "I could not do it: it was not approved."
    assert len(llm.calls) == 2


@pytest.mark.parametrize("answer, narrated", [
    ("I can use echo, strict_tool and regular_tool for that.", False),  # a listing
    ("Available tools: echo, confirm_tool, strict_tool.", False),
    ("I'll call echo now.", True),  # one tool named
    ("Action: echo(text='hi') and then strict_tool()", True),  # call syntax
    ('{"name": "echo", "arguments": {"text": "hi"}} then strict_tool', True),
    ("Tool call: echo Args: {\"text\": \"hi\"}; strict_tool next", True),
])
def test_narration_detection_tells_a_listing_from_a_written_out_call(answer, narrated):
    from agent.engine.react_loop import _looks_like_a_narrated_tool_call

    assert _looks_like_a_narrated_tool_call(answer, ["echo", "strict_tool", "regular_tool", "confirm_tool"]) is narrated


def test_the_same_unknown_tool_twice_ends_the_run_instead_of_looping():
    llm = FakeLLM([
        {"tool_call": {"id": "u1", "name": "teleport", "arguments": {}}},
        {"tool_call": {"id": "u2", "name": "teleport", "arguments": {}}},
        {"final_answer": "never reached"},
    ])

    result = ReactLoop(llm, tool_registry=make_registry(), max_steps=10).run("go")

    assert result.startswith("Stopped: the model kept calling 'teleport'")
    assert len(llm.calls) == 2
