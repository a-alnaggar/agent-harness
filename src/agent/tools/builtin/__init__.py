"""Built-in filesystem tools and a helper to register all of them at once."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..registry import ToolRegistry
from .fs_create import CreateFileTool, CreateFolderTool
from .fs_edit import EditFileTool
from .fs_read import ReadFileTool
from .fs_search import ListDirectoryTool, SearchInFilesTool
from .web_search import WebSearchTool

__all__ = [
    "CreateFileTool",
    "CreateFolderTool",
    "EditFileTool",
    "ReadFileTool",
    "ListDirectoryTool",
    "SearchInFilesTool",
    "WebSearchTool",
    "register_builtin_tools",
]


def register_builtin_tools(
    registry: ToolRegistry,
    root: Path,
) -> None:
    """Instantiate and register every built-in tool, scoped to `root`."""
    for tool_cls in (
        CreateFolderTool,
        CreateFileTool,
        ReadFileTool,
        EditFileTool,
        ListDirectoryTool,
        SearchInFilesTool,
    ):
        registry.register(tool_cls(root))
    registry.register(WebSearchTool())
