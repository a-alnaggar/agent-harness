"""Interactive REPL Module.

Provides the main Read-Eval-Print Loop for the user-facing CLI session.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from rich.prompt import Prompt

from agent.cli_interface.commands import CommandAction, CommandDispatcher
from agent.cli_interface.renderer import CLIRenderer
from agent.models import ReActResult, ReActStep


class ReActEngineProtocol(Protocol):
    def run(self, user_prompt: str, **kwargs: Any) -> ReActResult: ...


class ContextManagerProtocol(Protocol):
    def count_tokens(self) -> int: ...
    def reset(self) -> None: ...


class CLIHandler:
    """Manages the interactive terminal REPL session."""

    def __init__(
        self,
        renderer: CLIRenderer,
        dispatcher: CommandDispatcher,
        engine: Any,
        context_manager: Any,
    ) -> None:
        self.renderer = renderer
        self.dispatcher = dispatcher
        self.engine = engine
        self.context_manager = context_manager
        # Set while run_loop's "thinking" spinner is active, so _confirm_tool
        # can pause it - the spinner's own redraw loop otherwise fights the
        # blocking Prompt.ask() for the terminal and eats the user's keystroke.
        self._active_status: Any = None

    def _confirm_tool(self, tool_name: str, arguments: dict[str, Any], description: str) -> str:
        """Blocking approval prompt for a CONFIRM/STRICT-tier tool call.
        Returns "once", "always", or "deny" - see ReactLoop.run's gating
        logic for exactly what each means per tier."""
        self.renderer.render_permission_request(tool_name, arguments, description)
        if self._active_status is not None:
            self._active_status.stop()
        try:
            choice = Prompt.ask(
                "[bold yellow]Allow this action?[/bold yellow] (y = once, a = always this session, n = deny)",
                choices=["y", "a", "n"],
                default="n",
            )
        finally:
            if self._active_status is not None:
                self._active_status.start()
        return {"y": "once", "a": "always", "n": "deny"}[choice]

    def run_loop(self) -> None:
        """Start and run the continuous interactive REPL loop."""
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]agent[/bold cyan] [bold green]>[/bold green]").strip()

                if not user_input:
                    continue

                # 1. Handle potential slash command
                action = self.dispatcher.dispatch(user_input)
                if action == CommandAction.EXIT:
                    break
                elif action == CommandAction.HANDLED:
                    continue

                # 2. Execute user task via ReAct Engine (prints steps live)
                has_count_tokens = hasattr(self.context_manager, "count_tokens")
                tokens_before = self.context_manager.count_tokens() if has_count_tokens else 0

                status = self.renderer.console.status("[bold cyan]Agent is thinking and executing...[/bold cyan]", spinner="dots")
                with status:
                    self._active_status = status
                    try:
                        result = self.engine.run(
                            user_prompt=user_input,
                            step_callback=self.renderer.render_step,
                            on_compaction=self.renderer.render_compaction_notice,
                            confirm_callback=self._confirm_tool,
                        )
                    except TypeError:
                        result = self.engine.run(
                            user_input,
                            step_callback=self.renderer.render_step,
                        )
                    finally:
                        self._active_status = None

                # 3. Render final answer — tokens used by THIS task, not the
                # session's running total (count_tokens() is cumulative).
                # Context %, unlike tokens, is always the live snapshot.
                tokens_after = self.context_manager.count_tokens() if has_count_tokens else 0
                context_percent = (
                    self.context_manager.context_percent() if hasattr(self.context_manager, "context_percent") else 0.0
                )
                self.renderer.render_final_answer(
                    result, token_count=tokens_after - tokens_before, context_percent=context_percent
                )

            except (KeyboardInterrupt, EOFError):
                self.renderer.console.print("\n[dim]Interrupted by user. Exiting...[/dim]")
                break
            except Exception as e:
                self.renderer.render_error(f"{type(e).__name__}: {e}", title="Unexpected Error")
