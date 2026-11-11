"""Tool allowing the parent agent to delegate tasks to specialized sub-agents."""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent.models import ReActStep
from agent.observability import RunContext, context_aware
from agent.tools.base import Tool, ToolError
from agent.tools.permissions import PermissionTier
from .runner import SubAgentRunner


@context_aware("run", "run_with_context")
class DelegateTaskTool(Tool):
    name = "delegate_task"
    # Not a leaf tool: its duration is a whole nested sub-agent run, so it is
    # labelled apart from built-in/MCP tools and kept out of tool latency.
    origin = "subagent"
    # Delegation itself isn't the risky part - the sub-agent's own tool
    # calls are gated individually via the same confirm_callback, threaded
    # through below (nested_callback pattern, same as step_callback).
    permission_tier = PermissionTier.REGULAR

    def __init__(self, runner: SubAgentRunner, parent_run_id: str = ""):
        self.runner = runner
        self.parent_run_id = parent_run_id
        self.step_callback: Optional[Callable[[ReActStep], None]] = None
        self.confirm_callback: Optional[Callable[[str, dict, str], str]] = None

    @property
    def description(self) -> str:
        lines = [
            "Delegate a specific task to an isolated, specialized sub-agent.",
            "IMPORTANT: When the user asks any question about courses, syllabi, grading, policies, deadlines, or lectures, ALWAYS delegate directly to 'course_knowledge' instead of listing files or searching directories.",
        ]
        if self.runner and self.runner.configs:
            lines.append("Available specialized sub-agents:")
            for name, sa in self.runner.configs.items():
                desc = sa.description or sa.role[:100]
                lines.append(f"- '{name}': {desc}")
        return "\n".join(lines)

    @property
    def parameters(self) -> dict[str, Any]:
        subagents = list(self.runner.configs.keys()) if self.runner else []
        type_prop: dict[str, Any] = {
            "type": "string",
            "description": "The type of specialized sub-agent to delegate to (e.g. 'course_knowledge', 'pdf_analyst').",
        }
        if subagents:
            type_prop["enum"] = subagents

        return {
            "type": "object",
            "properties": {
                "subagent_type": type_prop,
                "task": {
                    "type": "string",
                    "description": "Clear and detailed instructions for the sub-agent to execute.",
                },
            },
            "required": ["subagent_type", "task"],
        }

    def run(self, *, subagent_type: str, task: str) -> str:
        return self._run(subagent_type=subagent_type, task=task)

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(self, *, subagent_type: str, task: str, context: RunContext | None = None) -> str:
        if not subagent_type or not task:
            raise ToolError("Both 'subagent_type' and 'task' are required.")
        return self.runner.run_subagent(
            name=subagent_type.strip(),
            task=task.strip(),
            parent_run_id=self.parent_run_id,
            step_callback=self.step_callback,
            confirm_callback=self.confirm_callback,
            parent_context=context,
        )
