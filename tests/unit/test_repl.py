"""Unit tests for CLIHandler's REPL loop: command dispatch, the engine.run
signature fallback, and that interrupts/unexpected errors end or continue
the loop correctly instead of crashing it."""

from unittest.mock import MagicMock

import pytest

import agent.cli_interface.repl as repl_module
from agent.cli_interface.commands import CommandAction
from agent.cli_interface.repl import CLIHandler
from agent.models import ReActResult


def make_handler(monkeypatch, prompt_side_effect):
    renderer = MagicMock()
    dispatcher = MagicMock()
    engine = MagicMock()
    context_manager = MagicMock()
    # cumulative, not per-task - first call is "before", second is "after"
    context_manager.count_tokens.side_effect = [100, 350]
    context_manager.context_percent.return_value = 12.5

    handler = CLIHandler(renderer=renderer, dispatcher=dispatcher, engine=engine, context_manager=context_manager)

    monkeypatch.setattr(repl_module.Prompt, "ask", MagicMock(side_effect=prompt_side_effect))

    return handler, renderer, dispatcher, engine, context_manager


def test_exit_command_stops_the_loop_without_calling_the_engine(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, ["/exit"])
    dispatcher.dispatch.return_value = CommandAction.EXIT

    handler.run_loop()

    engine.run.assert_not_called()


def test_handled_command_does_not_call_the_engine_and_loop_continues(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, ["/help", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.HANDLED, CommandAction.EXIT]

    handler.run_loop()

    engine.run.assert_not_called()
    assert dispatcher.dispatch.call_count == 2


def test_empty_input_is_skipped_without_dispatching(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, ["   ", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.EXIT]

    handler.run_loop()

    dispatcher.dispatch.assert_called_once_with("/exit")


def test_a_real_task_calls_the_engine_and_renders_the_final_answer(monkeypatch):
    handler, renderer, dispatcher, engine, context_manager = make_handler(monkeypatch, ["do the thing", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.NOT_A_COMMAND, CommandAction.EXIT]
    result = ReActResult(final_answer="done", total_iterations=1, total_duration_seconds=0.1)
    engine.run.return_value = result

    handler.run_loop()

    _, kwargs = engine.run.call_args
    assert kwargs["user_prompt"] == "do the thing"
    assert kwargs["step_callback"] == renderer.render_step
    assert kwargs["on_compaction"] == renderer.render_compaction_notice
    assert kwargs["confirm_callback"] == handler._confirm_tool
    renderer.render_final_answer.assert_called_once_with(result, token_count=250, context_percent=12.5)


def test_confirm_tool_shows_the_request_then_maps_the_users_choice(monkeypatch):
    renderer = MagicMock()
    handler = CLIHandler(renderer=renderer, dispatcher=MagicMock(), engine=MagicMock(), context_manager=MagicMock())
    monkeypatch.setattr(repl_module.Prompt, "ask", MagicMock(return_value="a"))

    decision = handler._confirm_tool("send_email", {"to": "x@example.com"}, "sends an email")

    renderer.render_permission_request.assert_called_once_with(
        "send_email", {"to": "x@example.com"}, "sends an email"
    )
    assert decision == "always"


def test_confirm_tool_maps_each_choice_correctly(monkeypatch):
    renderer = MagicMock()
    handler = CLIHandler(renderer=renderer, dispatcher=MagicMock(), engine=MagicMock(), context_manager=MagicMock())

    for raw_choice, expected in [("y", "once"), ("a", "always"), ("n", "deny")]:
        monkeypatch.setattr(repl_module.Prompt, "ask", MagicMock(return_value=raw_choice))
        assert handler._confirm_tool("tool", {}, "desc") == expected


def test_confirm_tool_pauses_the_thinking_spinner_around_the_blocking_prompt(monkeypatch):
    """Regression test: the "Agent is thinking..." spinner's own redraw loop
    was fighting Prompt.ask() for the terminal, so a user's keystroke could
    be eaten before it registered. _confirm_tool must stop() the active
    status before blocking on input and start() it again after, and the
    call order matters - stop before the prompt runs, start only after."""
    renderer = MagicMock()
    handler = CLIHandler(renderer=renderer, dispatcher=MagicMock(), engine=MagicMock(), context_manager=MagicMock())
    status = MagicMock()
    handler._active_status = status

    calls = []
    status.stop.side_effect = lambda: calls.append("stop")
    status.start.side_effect = lambda: calls.append("start")
    monkeypatch.setattr(
        repl_module.Prompt, "ask", MagicMock(side_effect=lambda *a, **k: calls.append("prompt") or "y")
    )

    decision = handler._confirm_tool("sandbox_exec", {"command": "ls"}, "runs a shell command")

    assert decision == "once"
    assert calls == ["stop", "prompt", "start"]


def test_confirm_tool_resumes_the_spinner_even_if_prompt_raises(monkeypatch):
    renderer = MagicMock()
    handler = CLIHandler(renderer=renderer, dispatcher=MagicMock(), engine=MagicMock(), context_manager=MagicMock())
    status = MagicMock()
    handler._active_status = status
    monkeypatch.setattr(repl_module.Prompt, "ask", MagicMock(side_effect=KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        handler._confirm_tool("sandbox_exec", {"command": "ls"}, "runs a shell command")

    status.start.assert_called_once()


def test_confirm_tool_is_a_no_op_on_the_spinner_when_none_is_active(monkeypatch):
    """_confirm_tool can also be exercised outside run_loop's spinner
    context (e.g. by tests, or a future non-interactive caller) - must not
    blow up when there's no status to pause."""
    renderer = MagicMock()
    handler = CLIHandler(renderer=renderer, dispatcher=MagicMock(), engine=MagicMock(), context_manager=MagicMock())
    monkeypatch.setattr(repl_module.Prompt, "ask", MagicMock(return_value="n"))

    assert handler._confirm_tool("tool", {}, "desc") == "deny"


def test_token_count_is_this_tasks_usage_not_the_session_cumulative_total(monkeypatch):
    """count_tokens() is cumulative for the whole session, so the CLI must
    subtract the before/after snapshots itself - otherwise every task after
    the first would display an ever-growing number that was never actually
    spent on that task."""
    handler, renderer, dispatcher, engine, context_manager = make_handler(monkeypatch, ["do the thing", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.NOT_A_COMMAND, CommandAction.EXIT]
    context_manager.count_tokens.side_effect = [7000, 14200]  # already 7000 in from earlier turns
    result = ReActResult(final_answer="done", total_iterations=1, total_duration_seconds=0.1)
    engine.run.return_value = result

    handler.run_loop()

    renderer.render_final_answer.assert_called_once_with(result, token_count=7200, context_percent=12.5)


def test_engine_run_falls_back_to_positional_call_on_type_error(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, ["do the thing", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.NOT_A_COMMAND, CommandAction.EXIT]
    result = ReActResult(final_answer="done", total_iterations=1, total_duration_seconds=0.1)

    call_log = []

    def run_side_effect(*args, **kwargs):
        if "user_prompt" in kwargs:
            call_log.append(("keyword-style call", args, kwargs))
            raise TypeError("unexpected keyword argument 'user_prompt'")
        call_log.append(("positional-style call", args, kwargs))
        return result

    engine.run.side_effect = run_side_effect

    handler.run_loop()

    assert call_log[0][0] == "keyword-style call"
    assert call_log[1][0] == "positional-style call"
    assert call_log[1][1] == ("do the thing",)
    renderer.render_final_answer.assert_called_once_with(result, token_count=250, context_percent=12.5)


def test_keyboard_interrupt_ends_the_loop_gracefully(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, KeyboardInterrupt())

    handler.run_loop()

    engine.run.assert_not_called()
    renderer.console.print.assert_called_once()
    assert "Interrupted" in renderer.console.print.call_args[0][0]


def test_unexpected_exception_is_rendered_as_an_error_and_loop_continues(monkeypatch):
    handler, renderer, dispatcher, engine, _ = make_handler(monkeypatch, ["do the thing", "/exit"])
    dispatcher.dispatch.side_effect = [CommandAction.NOT_A_COMMAND, CommandAction.EXIT]
    engine.run.side_effect = ValueError("boom")

    handler.run_loop()

    renderer.render_error.assert_called_once()
    args, kwargs = renderer.render_error.call_args
    assert "ValueError: boom" in args[0]
    assert kwargs["title"] == "Unexpected Error"
    # the loop kept going after the error and reached /exit on the next prompt
    assert dispatcher.dispatch.call_count == 2
