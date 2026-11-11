"""Central registry the ReAct engine uses to discover and invoke tools."""

from __future__ import annotations

from typing import Any, Callable
from datetime import datetime

from agent.observability import LifecycleHook, LifecycleObserver, RunContext, safe_tool_name, context_aware, contextual_method

from .base import Tool, ToolError, ToolUnavailableError


@context_aware("call", "call_with_context")
class ToolRegistry:
    """Holds the set of tools available to the agent for a session."""

    def __init__(
        self, hook: LifecycleHook | None = None, agent_id: str = "parent",
        id_provider: Callable[[], str] | None = None, clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._observer = LifecycleObserver(hook, agent_id, id_provider, clock, monotonic_clock, diagnostic)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"no tool registered under '{name}'") from None

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas in the shape the LLM's tool-calling API expects."""
        return [tool.to_schema() for tool in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> str:
        """Look up and run a tool, turning both a missing tool and a tool
        failure into an observation string instead of raising, since this
        is normally called from inside the ReAct loop."""
        return self.call_with_context(name, kwargs)

    def call_with_context(
        self, name: str, arguments: dict[str, Any], *, context: RunContext | None = None,
    ) -> str:
        observer = self._observer
        span = observer.start(context)
        attributes = observer.protect(lambda: {"tool_name": safe_tool_name(name),
            **({"origin": self._tools[name].origin} if name in self._tools else {})}) or {}
        observer.emit("tool.call.start", span, "tool", "call", attributes)
        try:
            tool = self.get(name)
        except KeyError as exc:
            observer.emit("tool.call.error", span, "tool", "call", attributes, type(exc).__name__)
            return f"Error: {exc}"
        try:
            contextual_run = observer.protect(lambda: contextual_method(tool, "run")) if span else None
            result = contextual_run(arguments, context=span[0]) if callable(contextual_run) else tool.run(**arguments)
        except ToolUnavailableError as exc:
            # Raised on, not turned into a string: the loop has to know the
            # tool is gone for this request (see ToolUnavailableError).
            observer.emit("tool.call.error", span, "tool", "call", attributes, type(exc).__name__)
            raise
        except ToolError as exc:
            observer.emit("tool.call.error", span, "tool", "call", attributes, type(exc).__name__)
            return f"Error: {exc}"
        except BaseException as exc:
            observer.emit("tool.call.error", span, "tool", "call", attributes, type(exc).__name__)
            raise
        observer.emit("tool.call.finish", span, "tool", "call", attributes)
        return result
