"""Unit tests for CommandDispatcher: every slash command actually routes
to the right renderer call with the right data, not just that dispatch()
returns without raising."""

from unittest.mock import MagicMock

import pytest

from agent.cli_interface.commands import CommandAction, CommandDispatcher


def make_dispatcher(registry=None, context_manager=None, config=None):
    renderer = MagicMock()
    dispatcher = CommandDispatcher(
        renderer=renderer,
        registry=registry if registry is not None else MagicMock(),
        context_manager=context_manager if context_manager is not None else MagicMock(),
        config=config,
    )
    return dispatcher, renderer


def test_non_command_input_is_passed_through():
    dispatcher, renderer = make_dispatcher()

    action = dispatcher.dispatch("create a file called notes.txt")

    assert action == CommandAction.NOT_A_COMMAND
    renderer.assert_not_called()


@pytest.mark.parametrize("cmd", ["/exit", "/quit", "/EXIT"])
def test_exit_commands_print_goodbye_and_return_exit(cmd):
    dispatcher, renderer = make_dispatcher()

    action = dispatcher.dispatch(cmd)

    assert action == CommandAction.EXIT
    renderer.console.print.assert_called_once()
    assert "Goodbye" in renderer.console.print.call_args[0][0]


def test_help_command_calls_print_help():
    dispatcher, renderer = make_dispatcher()

    action = dispatcher.dispatch("/help")

    assert action == CommandAction.HANDLED
    renderer.print_help.assert_called_once()


def test_tools_command_passes_schemas_from_registry_schemas_method():
    registry = MagicMock()
    registry.schemas.return_value = [{"function": {"name": "read_file"}}]
    dispatcher, renderer = make_dispatcher(registry=registry)

    dispatcher.dispatch("/tools")

    renderer.print_tools.assert_called_once_with([{"function": {"name": "read_file"}}])


def test_tools_command_falls_back_to_get_all_schemas_when_schemas_missing():
    registry = MagicMock(spec=["get_all_schemas"])
    registry.get_all_schemas.return_value = [{"function": {"name": "legacy_tool"}}]
    dispatcher, renderer = make_dispatcher(registry=registry)

    dispatcher.dispatch("/tools")

    renderer.print_tools.assert_called_once_with([{"function": {"name": "legacy_tool"}}])


def test_tools_command_passes_empty_list_when_registry_has_neither_method():
    registry = MagicMock(spec=[])
    dispatcher, renderer = make_dispatcher(registry=registry)

    dispatcher.dispatch("/tools")

    renderer.print_tools.assert_called_once_with([])


def test_tokens_command_passes_real_stats_when_context_manager_supports_it():
    context_manager = MagicMock()
    context_manager.get_stats.return_value = {"total_messages": 4, "estimated_tokens": 200}
    dispatcher, renderer = make_dispatcher(context_manager=context_manager)

    dispatcher.dispatch("/tokens")

    renderer.print_tokens.assert_called_once_with({"total_messages": 4, "estimated_tokens": 200})


def test_tokens_command_falls_back_to_zeroed_stats_without_get_stats():
    context_manager = MagicMock(spec=[])
    dispatcher, renderer = make_dispatcher(context_manager=context_manager)

    dispatcher.dispatch("/tokens")

    renderer.print_tokens.assert_called_once_with({"total_messages": 0, "estimated_tokens": 0})


def test_reset_command_calls_context_manager_reset_and_confirms():
    context_manager = MagicMock()
    dispatcher, renderer = make_dispatcher(context_manager=context_manager)

    action = dispatcher.dispatch("/reset")

    assert action == CommandAction.HANDLED
    context_manager.reset.assert_called_once()
    assert "reset" in renderer.console.print.call_args[0][0].lower()


def test_reset_command_does_not_crash_without_reset_method():
    context_manager = MagicMock(spec=[])
    dispatcher, renderer = make_dispatcher(context_manager=context_manager)

    action = dispatcher.dispatch("/reset")

    assert action == CommandAction.HANDLED
    renderer.console.print.assert_called_once()


def test_config_command_passes_through_nested_config_values():
    config = MagicMock()
    config.llm.model = "qwen-agentworld-35b-a3b"
    config.llm.temperature = 0.2
    config.llm.timeout_seconds = 60
    config.agent.max_iterations = 10
    config.workspace.absolute_root = "/workspace"
    dispatcher, renderer = make_dispatcher(config=config)

    dispatcher.dispatch("/config")

    renderer.print_config.assert_called_once_with(
        model="qwen-agentworld-35b-a3b",
        temperature=0.2,
        max_iterations=10,
        timeout=60,
        workspace="/workspace",
    )


def test_config_command_uses_defaults_when_config_is_none():
    dispatcher, renderer = make_dispatcher(config=None)

    dispatcher.dispatch("/config")

    renderer.print_config.assert_called_once_with(
        model="default",
        temperature=0.0,
        max_iterations=10,
        timeout=60,
        workspace=".",
    )


def test_unknown_command_reports_an_error():
    dispatcher, renderer = make_dispatcher()

    action = dispatcher.dispatch("/bogus")

    assert action == CommandAction.HANDLED
    renderer.render_error.assert_called_once()
    args, kwargs = renderer.render_error.call_args
    assert "/bogus" in args[0]
    assert kwargs["title"] == "Invalid Command"
