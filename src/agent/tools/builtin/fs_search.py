from __future__ import annotations

from pathlib import Path

from ..base import Tool, ToolError
from ..permissions import PermissionTier
from ._paths import resolve_in_workspace


class ListDirectoryTool(Tool):
    name = "list_directory"
    permission_tier = PermissionTier.REGULAR
    description = "List the files and folders directly inside a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path, relative to the workspace root. Use '.' for the root."},
        },
        "required": ["path"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str = ".") -> str:
        target = resolve_in_workspace(self.root, path)
        if not target.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        if not entries:
            return f"'{path}' is empty"
        return "\n".join(f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries)


class SearchInFilesTool(Tool):
    name = "search_in_files"
    permission_tier = PermissionTier.REGULAR
    description = "Search for a text substring across files under a directory, returning matching file:line:text."

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Substring to search for."},
            "path": {"type": "string", "description": "Directory to search under, relative to the workspace root. Defaults to the root."},
        },
        "required": ["query"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, query: str, path: str = ".") -> str:
        base = resolve_in_workspace(self.root, path)
        if not base.is_dir():
            raise ToolError(f"'{path}' is not a directory")

        matches: list[str] = []
        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = file_path.relative_to(self.root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(f"{rel}:{lineno}:{line.strip()}")

        return "\n".join(matches) if matches else f"No matches for '{query}'"
