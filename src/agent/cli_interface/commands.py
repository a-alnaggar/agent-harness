"""CLI Slash Commands Dispatcher Module.

Handles built-in slash commands (/help, /tools, /tokens, /reset, /config, /exit).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol

from agent.cli_interface.renderer import CLIRenderer


class CommandAction(Enum):
    """Result of evaluating a potential slash command."""
    NOT_A_COMMAND = auto()
    HANDLED = auto()
    EXIT = auto()


class ContextProtocol(Protocol):
    def get_stats(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...


class RegistryProtocol(Protocol):
    """Protocol accepting both schemas() and get_all_schemas()."""
    pass


class CommandDispatcher:
    """Dispatches interactive slash commands to appropriate handlers."""

    def __init__(
        self,
        renderer: CLIRenderer,
        registry: Any,
        context_manager: Any,
        config: Any,
    ) -> None:
        self.renderer = renderer
        self.registry = registry
        self.context_manager = context_manager
        self.config = config

    def _get_schemas(self) -> list[dict[str, Any]]:
        """Retrieve schemas regardless of whether registry uses schemas() or get_all_schemas()."""
        if hasattr(self.registry, "schemas") and callable(self.registry.schemas):
            return self.registry.schemas()
        if hasattr(self.registry, "get_all_schemas") and callable(self.registry.get_all_schemas):
            return self.registry.get_all_schemas()
        return []

    def dispatch(self, user_input: str) -> CommandAction:
        """Check if input is a slash command and execute it."""
        clean_input = user_input.strip()
        if not clean_input.startswith("/"):
            return CommandAction.NOT_A_COMMAND

        cmd = clean_input.lower()

        if cmd in ["/exit", "/quit"]:
            self.renderer.console.print("[bold cyan]Goodbye! Exiting Agent Harness.[/bold cyan]")
            return CommandAction.EXIT

        elif cmd == "/help":
            self.renderer.print_help()
            return CommandAction.HANDLED

        elif cmd == "/tools":
            self.renderer.print_tools(self._get_schemas())
            return CommandAction.HANDLED

        elif cmd == "/tokens":
            stats = self.context_manager.get_stats() if hasattr(self.context_manager, "get_stats") else {"total_messages": 0, "estimated_tokens": 0}
            self.renderer.print_tokens(stats)
            return CommandAction.HANDLED

        elif cmd == "/reset":
            if hasattr(self.context_manager, "reset"):
                self.context_manager.reset()
            self.renderer.console.print("[bold green]Context reset successfully. Fresh session started.[/bold green]")
            return CommandAction.HANDLED

        elif cmd == "/config":
            model = getattr(getattr(self.config, "llm", None), "model", "default")
            temp = getattr(getattr(self.config, "llm", None), "temperature", 0.0)
            max_iter = getattr(getattr(self.config, "agent", None), "max_iterations", 10)
            timeout = getattr(getattr(self.config, "llm", None), "timeout_seconds", 60)
            workspace = str(getattr(getattr(self.config, "workspace", None), "absolute_root", "."))

            self.renderer.print_config(
                model=model,
                temperature=temp,
                max_iterations=max_iter,
                timeout=timeout,
                workspace=workspace,
            )
            return CommandAction.HANDLED

        else:
            self.renderer.render_error(
                f"Unknown command '{clean_input}'. Type /help for available commands.",
                title="Invalid Command",
            )
            return CommandAction.HANDLED
