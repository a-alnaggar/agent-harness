"""Base interface every tool implements, plus the schema shape the ReAct
engine hands to the LLM for tool calling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .permissions import PermissionTier


class ToolError(Exception):
    """Raised by a tool when it fails; caught by the engine and fed back
    to the LLM as an observation rather than crashing the loop."""


class ToolUnavailableError(ToolError):
    """The tool can't work right now (e.g. its remote host doesn't answer),
    so calling it again in the same request only fails again. The loop
    withdraws it for the rest of the request; for a STRICT tool that also
    stops the approval prompt from repeating for every retry."""


class Tool(ABC):
    """A single callable tool.

    Subclasses set `name`, `description`, and `parameters` (a JSON Schema
    "object" describing the keyword arguments `run` accepts), and implement
    `run`. `parameters` is what gets sent to the LLM's tool-calling API, so
    it must stay in sync with `run`'s actual signature.

    `permission_tier` defaults to CONFIRM - the safe default for any tool
    that doesn't explicitly declare itself lower-risk. Set it explicitly on
    concrete tools after reviewing what they actually do.
    """

    name: str
    origin: str = "builtin"
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    permission_tier: PermissionTier = PermissionTier.CONFIRM

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string observation for the LLM."""

    def to_schema(self) -> dict[str, Any]:
        """OpenAI-style function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
