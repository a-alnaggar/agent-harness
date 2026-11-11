"""Presentation & Rendering Module.

Handles terminal formatting, color themes, step animations, and panels using Rich.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from agent.models import ReActResult, ReActStep

# Characters that can act on the terminal or on how the text reads instead of
# just being shown: C0/C1 controls (ESC sequences clear the screen, move the
# cursor, hide or rewrite what's already printed), Unicode direction
# overrides (reorder text visually) and tag characters (invisible payloads).
# Tool arguments and results carry text from emails, web pages and other
# people, so none of it is trusted to be inert.
_ACTIVE_CHARACTERS = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\U000e0000-\U000e007f]")


def neutralize(text: Any) -> str:
    """Show control/direction/tag characters as visible \\uXXXX escapes instead of letting them act."""
    return _ACTIVE_CHARACTERS.sub(
        lambda match: f"\\u{ord(match.group()):04x}" if ord(match.group()) <= 0xFFFF else f"\\U{ord(match.group()):08x}",
        str(text),
    )


def safe(text: Any) -> str:
    """Untrusted text for a Rich markup string: made inert, and markup like
    [/INST] or [conceal] rendered literally rather than interpreted (an
    unmatched closing tag raises MarkupError; some tags hide text)."""
    return escape(neutralize(text))


class CLIRenderer:
    """Renders styled CLI components: banners, thoughts, tool executions, and markdown answers."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def print_banner(self, model_name: str, tools_count: int) -> None:
        """Print clean, minimalist welcoming header."""
        self.console.print("[dim]Type your task, or /help for commands, /exit to quit.[/dim]")
        self.console.print()

    def render_step(self, step: ReActStep) -> None:
        """Render a single ReAct step (thought, tool calls, and observations)."""
        header = f"Iteration {step.step_number} [dim]({step.duration_seconds:.2f}s)[/dim]"
        scope = f"Sub-agent: {step.agent_name} - " if step.agent_name != "parent" else ""

        # 1. Display Thinking / Reasoning
        if step.thought:
            self.console.print(
                Panel(
                    Markdown(neutralize(step.thought.strip())),
                    title=f"[bold blue]{scope}Thought - {header}[/bold blue]",
                    border_style="blue",
                    padding=(0, 1),
                )
            )

        # 2. Display Tool Executions and Observations
        for tc, obs in zip(step.tool_calls, step.observations):
            tool_content = (
                f"[bold yellow]Tool:[/bold yellow] [cyan]{safe(tc.name)}[/cyan]\n"
                f"[bold yellow]Args:[/bold yellow] [white]{safe(tc.arguments)}[/white]\n"
                f"[bold yellow]Observation:[/bold yellow] [dim white]{safe(obs.content)}[/dim white]"
            )
            border = "red" if obs.is_error else "yellow"
            self.console.print(
                Panel(
                    tool_content,
                    title=f"[{border}]{scope}Action & Observation[/{border}]",
                    border_style=border,
                    padding=(0, 1),
                )
            )

    def render_final_answer(self, result: ReActResult, token_count: int, context_percent: float = 0.0) -> None:
        """Render the final conclusion returned by the agent."""
        box_style = "bold green" if result.is_success else "bold red"
        header = "Final Answer" if result.is_success else "Task Incomplete"

        self.console.print(
            Panel(
                Markdown(neutralize(result.final_answer)),
                title=f"[{box_style}]{header}[/{box_style}]",
                subtitle=(
                    f"[dim]Iterations: {result.total_iterations} | "
                    f"Time: {result.total_duration_seconds:.2f}s | "
                    f"Tokens: {token_count} | "
                    f"Context: {context_percent:.0f}%[/dim]"
                ),
                border_style=box_style,
                padding=(1, 2),
            )
        )

    def render_compaction_notice(self, percent_before: float, messages_before: int, messages_after: int) -> None:
        """Shown the moment the conversation gets compacted, so it's never
        a silent thing happening to the user's context."""
        self.console.print(
            Panel(
                f"Context hit {percent_before:.0f}% of the window — summarized the older part of the "
                f"conversation ({messages_before} messages -> {messages_after}). Recent turns are untouched.",
                title="[bold magenta]Context Compacted[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def render_permission_request(self, tool_name: str, arguments: dict[str, Any], description: str) -> None:
        """Shown right before asking the user to approve a CONFIRM/STRICT-
        tier tool call - the pending action, in full, before the yes/no
        prompt, so the decision isn't a blind guess."""
        # Every value here is shown exactly as it will be executed - which is
        # only true if nothing in it can restyle, hide or rewrite the panel.
        args_text = "\n".join(f"  {safe(k)}: {safe(v)}" for k, v in arguments.items()) or "  (no arguments)"
        self.console.print(
            Panel(
                f"[bold]Tool:[/bold] {safe(tool_name)}\n"
                f"[bold]Description:[/bold] {safe(description)}\n"
                f"[bold]Arguments:[/bold]\n{args_text}",
                title="[bold yellow]Permission Requested[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def render_error(self, message: str, title: str = "Error") -> None:
        """Render an error message panel."""
        self.console.print(
            Panel(
                f"[bold red]{safe(message)}[/bold red]",
                title=f"[bold red]{title}[/bold red]",
                border_style="red",
            )
        )

    def print_help(self) -> None:
        """Print help table showing interactive commands."""
        table = Table(title="Commands", border_style="cyan", show_header=True)
        table.add_column("Command", style="bold green")
        table.add_column("Description", style="white")

        table.add_row("/help", "Display this help menu.")
        table.add_row("/tools", "List all available tools and their parameter schemas.")
        table.add_row("/tokens", "Show current context window token usage and statistics.")
        table.add_row("/reset", "Clear conversation history and reset context back to system prompt.")
        table.add_row("/config", "Show active configuration settings.")
        table.add_row("/exit, /quit", "Exit the application.")

        self.console.print(table)

    def print_tools(self, schemas: list[dict[str, Any]]) -> None:
        """Print detailed list of registered tools and schemas."""
        table = Table(title="Registered Tools", border_style="yellow", show_header=True)
        table.add_column("Tool Name", style="bold yellow")
        table.add_column("Description", style="white")
        table.add_column("Required Args", style="cyan")

        for schema in schemas:
            # Handle both OpenAI-style and direct dict formats
            if "function" in schema:
                fn = schema["function"]
                name = fn.get("name", "unknown")
                desc = fn.get("description", "")
                req = ", ".join(fn.get("parameters", {}).get("required", [])) or "None"
            else:
                name = schema.get("name", "unknown")
                desc = schema.get("description", "")
                req = ", ".join(schema.get("parameters", {}).get("required", [])) or "None"
            table.add_row(safe(name), safe(desc), safe(req))

        self.console.print(table)

    def print_tokens(self, stats: dict[str, Any]) -> None:
        """Print context token usage metrics."""
        self.console.print(
            Panel(
                f"- [bold white]Total Messages:[/bold white] {stats.get('total_messages', 0)}\n"
                f"- [bold white]Estimated Tokens:[/bold white] [bold green]{stats.get('estimated_tokens', 0)}[/bold green]\n"
                f"- [bold white]Message Distribution:[/bold white] {stats.get('role_distribution', {})}",
                title="Context Token Stats",
                border_style="blue",
            )
        )

    def print_config(self, model: str, temperature: float, max_iterations: int, timeout: int, workspace: str) -> None:
        """Print active configuration summary."""
        self.console.print(
            Panel(
                f"[bold white]Model:[/bold white] {model}\n"
                f"[bold white]Temperature:[/bold white] {temperature}\n"
                f"[bold white]Max Iterations:[/bold white] {max_iterations}\n"
                f"[bold white]Timeout:[/bold white] {timeout}s\n"
                f"[bold white]Workspace:[/bold white] {workspace}",
                title="Active Configuration",
                border_style="cyan",
            )
        )
