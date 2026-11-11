"""Execution engine for specialized sub-agents."""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent.engine.react_loop import UNTRUSTED_CONTENT_NOTICE, ReactLoop
from agent.models import ReActStep
from agent.observability import LifecycleHook, RunContext
from agent.tools.permissions import PermissionPolicy
from agent.tools.registry import ToolRegistry
from .config import SubAgentConfig

# The exact sentinel ReactLoop._run() returns when a run falls through its
# step loop without the LLM ever producing a final_answer (see
# engine/react_loop.py). Plain text, no "Error:" prefix, so a parent reading
# it as a delegate_task observation has no structural signal that this was
# a failure - it just looks like some other kind of report. Without
# rewriting it, the parent tends to re-delegate the identical task instead
# of accepting or reporting the failure, repeating the same step-exhaustion
# every time.
_STEP_LIMIT_SENTINEL = "Stopped: maximum number of steps reached."


class SubAgentRunner:
    """Manages and executes specialized sub-agents on the same ReactLoop harness."""

    def __init__(
        self,
        llm: Any,
        master_registry: ToolRegistry,
        configs: Optional[dict[str, SubAgentConfig]] = None,
        lifecycle_hooks: Optional[Any] = None,
        hook: LifecycleHook | None = None,
        diagnostic: Callable[[str], None] | None = None,
        permission_policy: PermissionPolicy | None = None,
    ):
        self.llm = llm
        self.master_registry = master_registry
        self.configs: dict[str, SubAgentConfig] = configs or {}
        self.lifecycle_hooks = lifecycle_hooks
        self.hook = hook
        self.diagnostic = diagnostic
        # The same instance the parent loop uses, so deny_tools and a user's
        # "always" apply to a sub-agent's calls exactly as to the parent's.
        self.permission_policy = permission_policy

    def register_subagent(self, config: SubAgentConfig) -> None:
        self.configs[config.name] = config

    def has_subagent(self, name: str) -> bool:
        return name in self.configs

    def list_subagents(self) -> list[SubAgentConfig]:
        return list(self.configs.values())

    def run_subagent(
        self,
        name: str,
        task: str,
        parent_run_id: Optional[str] = None,
        step_callback: Optional[Callable[[ReActStep], None]] = None,
        confirm_callback: Optional[Callable[[str, dict, str], str]] = None,
        parent_context: RunContext | None = None,
    ) -> str:
        if name not in self.configs:
            available = ", ".join(f"'{k}'" for k in self.configs.keys()) or "none"
            return f"Error: Unknown sub-agent type '{name}'. Available sub-agents: {available}"

        config = self.configs[name]

        # 1. Filter tools strictly based on allowed_tools whitelist
        sub_registry = ToolRegistry(hook=self.hook, agent_id=name, diagnostic=self.diagnostic)
        for tool_name in config.allowed_tools:
            if self.master_registry.has_tool(tool_name):
                sub_registry.register(self.master_registry.get(tool_name))

        # 2. Lifecycle hooks notification (if attached)
        if self.lifecycle_hooks and hasattr(self.lifecycle_hooks, "on_subagent_start"):
            try:
                self.lifecycle_hooks.on_subagent_start(
                    subagent_name=name,
                    task=task,
                    parent_run_id=parent_run_id,
                )
            except Exception:
                pass

        # 3. Create isolated ReactLoop instance running on the same harness
        sub_loop = ReactLoop(
            llm=self.llm,
            tool_registry=sub_registry,
            max_steps=config.max_steps,
            system_prompt=f"{config.role}\n\n{UNTRUSTED_CONTENT_NOTICE}",
            require_successful_tool=config.require_successful_tool,
            hook=self.hook,
            agent_id=name,
            diagnostic=self.diagnostic,
            permission_policy=self.permission_policy,
        )

        try:
            def emit_step(step: ReActStep) -> None:
                step.agent_name = name
                if step_callback is not None:
                    step_callback(step)

            result = sub_loop.run(
                task,
                step_callback=emit_step if step_callback else None,
                confirm_callback=confirm_callback,
                parent_context=parent_context,
            )
            if result == _STEP_LIMIT_SENTINEL:
                result = (
                    f"Error: sub-agent '{name}' ran out of its step budget (max_steps={config.max_steps}) "
                    "before returning a final answer. Any tool calls it already made did happen and are "
                    "not undone. Do not immediately retry the identical task - it will likely hit the "
                    "same limit again. Either break the task into a narrower request, tell the user it "
                    "could not be completed, or proceed without this delegation."
                )
        except Exception as exc:
            result = f"Error during sub-agent execution: {exc}"

        if self.lifecycle_hooks and hasattr(self.lifecycle_hooks, "on_subagent_end"):
            try:
                self.lifecycle_hooks.on_subagent_end(
                    subagent_name=name,
                    result=result,
                    parent_run_id=parent_run_id,
                )
            except Exception:
                pass

        return result
