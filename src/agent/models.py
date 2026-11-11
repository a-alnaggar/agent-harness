"""Shared Domain Models & Data Transfer Objects (DTOs) for Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolObservation:
    """Represents the execution outcome of a tool."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ReActStep:
    """Represents a single iteration/step in the ReAct loop."""
    step_number: int
    thought: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    duration_seconds: float = 0.0
    agent_name: str = "parent"


@dataclass
class ReActResult:
    """Represents the overall outcome of a ReAct agent run."""
    final_answer: str
    total_iterations: int
    total_duration_seconds: float
    is_success: bool = True
    steps: list[ReActStep] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Structured response from the LLM client."""
    content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
