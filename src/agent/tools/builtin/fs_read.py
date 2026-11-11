from __future__ import annotations

from pathlib import Path

from ..base import Tool, ToolError
from ..permissions import PermissionTier
from ._paths import resolve_in_workspace


class ReadFileTool(Tool):
    name = "read_file"
    permission_tier = PermissionTier.REGULAR
    description = "Read a file's contents, returned with 1-based line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the workspace root."},
        },
        "required": ["path"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str) -> str:
        target = resolve_in_workspace(self.root, path)
        if not target.is_file():
            raise ToolError(f"'{path}' is not a file")
        lines = target.read_text(encoding="utf-8").splitlines()
        return "\n".join(f"{i:>4}\t{line}" for i, line in enumerate(lines, start=1))
