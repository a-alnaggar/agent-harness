"""E2E scenario from the lab spec: create a folder, write a file into it,
then search for text across the workspace."""

from agent.tools import ToolRegistry
from agent.tools.builtin import register_builtin_tools


def test_create_folder_write_file_then_search(tmp_path):
    registry = ToolRegistry()
    register_builtin_tools(registry, tmp_path)

    registry.call("create_folder", path="project")
    registry.call(
        "create_file",
        path="project/notes.txt",
        content="TODO: implement the tool registry\nDone: base Tool class",
    )

    result = registry.call("search_in_files", query="TODO")

    assert "project/notes.txt:1:TODO: implement the tool registry" in result
