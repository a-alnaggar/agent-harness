from __future__ import annotations

from pathlib import Path

from ..base import Tool, ToolError
from ..permissions import PermissionTier
from ._diff import unified_diff_text
from ._paths import resolve_in_workspace


class CreateFolderTool(Tool):
    name = "create_folder"
    permission_tier = PermissionTier.REGULAR
    description = "Create a new folder (and any missing parent folders) in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Folder path, relative to the workspace root."},
        },
        "required": ["path"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str) -> str:
        target = resolve_in_workspace(self.root, path)
        if target.exists() and target.is_file():
            raise ToolError(f"'{path}' already exists and is a file")
        target.mkdir(parents=True, exist_ok=True)
        return f"Created folder '{path}'"


class CreateFileTool(Tool):
    name = "create_file"
    permission_tier = PermissionTier.REGULAR
    description = "Create a new file with the given content. Fails if the file already exists."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the workspace root."},
            "content": {"type": "string", "description": "Initial file content."},
            "dry_run": {
                "type": "boolean",
                "description": "If true, don't create the file — return a diff of what would be written.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str, content: str = "", dry_run: bool = False) -> str:
        target = resolve_in_workspace(self.root, path)
        if target.exists():
            raise ToolError(f"'{path}' already exists")
        if dry_run:
            return unified_diff_text("", content, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Created file '{path}' ({len(content)} bytes)"
