"""E2E scenario from the lab spec: list a directory, then read a file back
with its line-numbered content."""

from agent.tools import ToolRegistry
from agent.tools.builtin import register_builtin_tools


def test_list_directory_then_read_file_with_line_numbers(tmp_path):
    registry = ToolRegistry()
    register_builtin_tools(registry, tmp_path)

    registry.call("create_file", path="a.txt", content="one\ntwo\nthree")
    registry.call("create_folder", path="sub")

    listing = registry.call("list_directory", path=".")
    assert "f  a.txt" in listing
    assert "d  sub" in listing

    content = registry.call("read_file", path="a.txt")
    lines = content.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith("1\tone")
    assert lines[2].endswith("3\tthree")
