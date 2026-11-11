"""Shared path resolution for filesystem tools.

Confines every tool to a workspace root so the agent can't read or write
outside the project it was launched on (basic sandboxing, per the lab's
permission-tiers concept).
"""

from __future__ import annotations

from pathlib import Path

from ..base import ToolError


def resolve_in_workspace(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ToolError(
            f"path '{relative_path}' resolves outside the workspace root"
        ) from None
    return candidate
