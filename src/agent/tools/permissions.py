"""Permission tiers for tool execution.

Builtin/first-party tools get a hardcoded tier below, reviewed by whoever
wrote the tool - not guessed by an LLM at runtime, since a classifier can
be wrong (a misleadingly-named tool, a subtly-worded description) in
exactly the cases a permission system exists to catch. Third-party
MCP-sourced tools default to STRICT and can only be loosened via explicit
config (see McpServerConfig.auto_tools / confirm_tools) - no tool earns
a lighter tier just by being present.

DENY is not a property of a tool but of the deployment: names listed under
`permissions.deny_tools` in config are blocked outright, whatever tier the
tool declares, and without asking anyone. PermissionPolicy is the single
object that applies that list and remembers "always" approvals; the parent
loop and every sub-agent loop share one instance so the rules (and a user's
"always") mean the same thing everywhere.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class PermissionTier(Enum):
    REGULAR = "regular"  # auto-execute, no confirmation
    CONFIRM = "confirm"  # requires approval; "always" persists for the rest of the session
    STRICT = "strict"    # requires approval on every single call, no "always" shortcut
    DENY = "deny"        # blocked by policy: never runs, never asks


class PermissionPolicy:
    """What may run in this session. Shared by every loop (parent and
    sub-agents) so "deny" and "always this session" behave identically
    regardless of which agent makes the call."""

    def __init__(self, deny_tools: Iterable[str] = ()) -> None:
        self.deny_tools = frozenset(deny_tools)
        self._approved: set[str] = set()
        # Tools refused during the current top-level request (by the user, the
        # policy, or because they can't work right now). Shared with the
        # sub-agents that request delegates to, so a denied tool isn't asked
        # about again when the parent delegates a second time.
        self.refused_in_request: set[str] = set()

    def start_request(self) -> None:
        self.refused_in_request.clear()

    def tier_for(self, tool_name: str, tool: Any) -> PermissionTier:
        """The effective tier: a denied name always wins over whatever the
        tool (or an MCP server's config) says about itself. A tool that
        declares no tier falls back to CONFIRM, never to REGULAR."""
        if tool_name in self.deny_tools:
            return PermissionTier.DENY
        return getattr(tool, "permission_tier", PermissionTier.CONFIRM)

    def is_approved(self, tool_name: str, tier: PermissionTier) -> bool:
        # STRICT never accepts a remembered approval.
        return tier == PermissionTier.CONFIRM and tool_name in self._approved

    def remember(self, tool_name: str, tier: PermissionTier) -> None:
        if tier == PermissionTier.CONFIRM:
            self._approved.add(tool_name)
