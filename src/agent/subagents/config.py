"""Configuration models for specialized sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubAgentConfig:
    name: str
    role: str
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    description: str = ""
    require_successful_tool: bool = False
