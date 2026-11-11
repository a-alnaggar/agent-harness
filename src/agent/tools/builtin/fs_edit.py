from __future__ import annotations

from pathlib import Path

from ..base import Tool, ToolError
from ..permissions import PermissionTier
from ._diff import unified_diff_text
from ._paths import resolve_in_workspace


class EditFileTool(Tool):
    """Overwrites or appends to an existing file. Deliberately cannot
    delete files or folders — that capability is out of scope by design."""

    name = "edit_file"
    permission_tier = PermissionTier.REGULAR
    description = "Modify an existing file by overwriting or appending content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the workspace root."},
            "content": {"type": "string", "description": "Content to write."},
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Whether to replace the file's contents or append to them.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, don't modify the file — return a diff of what would change.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str, content: str, mode: str = "overwrite", dry_run: bool = False) -> str:
        target = resolve_in_workspace(self.root, path)
        if not target.is_file():
            raise ToolError(f"'{path}' does not exist; use create_file first")
        if mode not in ("overwrite", "append"):
            raise ToolError(f"unknown mode '{mode}', expected 'overwrite' or 'append'")

        current = target.read_text(encoding="utf-8")
        new_content = current + content if mode == "append" else content

        if dry_run:
            return unified_diff_text(current, new_content, path)

        target.write_text(new_content, encoding="utf-8")
        return f"Updated '{path}' ({mode})"
