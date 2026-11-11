from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from agent.config import AgentConfig
from agent.engine.react_loop import ReactLoop
from agent.models import ReActResult


class ReactEngineAdapter:
    """Makes ReactLoop satisfy cli_interface's ReActEngineProtocol
    (`run(user_prompt, **kwargs) -> ReActResult`), since ReactLoop.run
    takes `user_input` and returns a plain string."""

    def __init__(self, loop: ReactLoop):
        self._loop = loop

    def count_tokens(self) -> int:
        return getattr(self._loop.llm, "total_tokens", 0)

    def context_percent(self) -> float:
        """How full the context window is right now, as a percentage."""
        return getattr(self._loop.llm, "context_usage_percent", lambda: 0.0)()

    def reset(self) -> None:
        self._loop.messages.clear()
        if hasattr(self._loop.llm, "total_tokens"):
            self._loop.llm.total_tokens = 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_messages": len(self._loop.messages),
            "estimated_tokens": self.count_tokens(),
            "role_distribution": {
                role: sum(1 for m in self._loop.messages if m.get("role") == role)
                for role in set(m.get("role") for m in self._loop.messages)
            },
        }

    def run(
        self,
        user_prompt: str,
        step_callback: Optional[Callable] = None,
        on_compaction: Optional[Callable] = None,
        confirm_callback: Optional[Callable] = None,
    ) -> ReActResult:
        start = time.monotonic()
        answer = self._loop.run(
            user_prompt,
            step_callback=step_callback,
            on_compaction=on_compaction,
            confirm_callback=confirm_callback,
        )
        duration = time.monotonic() - start

        iterations = sum(1 for m in self._loop.messages if m.get("role") == "tool") + 1
        is_success = not (answer.startswith("Error") or answer.startswith("Stopped:"))

        return ReActResult(
            final_answer=answer,
            total_iterations=iterations,
            total_duration_seconds=duration,
            is_success=is_success,
        )


class _LLMConfigView:
    def __init__(self, config: AgentConfig):
        self.model = config.model
        self.temperature = config.temperature
        self.timeout_seconds = config.timeout


class _AgentConfigView:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations


class _WorkspaceConfigView:
    def __init__(self, root: Path):
        self.absolute_root = root


class CLIConfigView:
    """Adapts the flat AgentConfig to the nested `.llm.*` / `.agent.*` /
    `.workspace.*` shape cli_interface's CommandDispatcher reads."""

    def __init__(self, config: AgentConfig, workspace_root: Path, max_iterations: int):
        self.llm = _LLMConfigView(config)
        self.agent = _AgentConfigView(max_iterations)
        self.workspace = _WorkspaceConfigView(workspace_root)
