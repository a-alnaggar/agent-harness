"""CLI Interface Package.

Provides the presentation and interactive user terminal experience for Agent Harness.
"""

from __future__ import annotations

from typing import Any

from agent.cli_interface.commands import CommandAction, CommandDispatcher
from agent.cli_interface.renderer import CLIRenderer
from agent.cli_interface.repl import CLIHandler


def start_cli(
    config: Any,
    engine: Any,
    registry: Any,
    context_manager: Any,
    llm_client: Any = None,
) -> None:
    """Bootstrap and start the interactive CLI session."""
    renderer = CLIRenderer()

    if hasattr(registry, "list_tools"):
        tools_count = len(registry.list_tools())
    elif hasattr(registry, "list_tool_names"):
        tools_count = len(registry.list_tool_names())
    else:
        tools_count = 0

    model_name = getattr(getattr(config, "llm", None), "model", "default")
    renderer.print_banner(
        model_name=model_name,
        tools_count=tools_count,
    )

    dispatcher = CommandDispatcher(
        renderer=renderer,
        registry=registry,
        context_manager=context_manager,
        config=config,
    )

    handler = CLIHandler(
        renderer=renderer,
        dispatcher=dispatcher,
        engine=engine,
        context_manager=context_manager,
    )

    handler.run_loop()


__all__ = [
    "CLIRenderer",
    "CommandDispatcher",
    "CommandAction",
    "CLIHandler",
    "start_cli",
]
