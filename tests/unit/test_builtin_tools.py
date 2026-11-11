import pytest

from agent.memory import ForgetMemoryTool, SaveMemoryTool, SqliteMemoryStore
from agent.tools import ToolRegistry
from agent.tools.builtin import register_builtin_tools


@pytest.fixture
def registry(tmp_path):
    reg = ToolRegistry()
    register_builtin_tools(reg, tmp_path)
    return reg


def test_create_folder(registry, tmp_path):
    registry.call("create_folder", path="notes")
    assert (tmp_path / "notes").is_dir()


def test_create_folder_nested(registry, tmp_path):
    registry.call("create_folder", path="a/b/c")
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_create_file(registry, tmp_path):
    registry.call("create_file", path="hello.txt", content="hi there")
    assert (tmp_path / "hello.txt").read_text() == "hi there"


def test_create_file_already_exists_errors(registry, tmp_path):
    registry.call("create_file", path="hello.txt", content="hi")
    result = registry.call("create_file", path="hello.txt", content="again")
    assert result.startswith("Error:")


def test_create_file_dry_run_does_not_write(registry, tmp_path):
    result = registry.call("create_file", path="hello.txt", content="hi there", dry_run=True)

    assert not (tmp_path / "hello.txt").exists()
    assert "+hi there" in result
    assert "b/hello.txt" in result


def test_create_file_dry_run_still_errors_if_exists(registry, tmp_path):
    registry.call("create_file", path="hello.txt", content="hi")
    result = registry.call("create_file", path="hello.txt", content="again", dry_run=True)
    assert result.startswith("Error:")


def test_read_file(registry, tmp_path):
    (tmp_path / "hello.txt").write_text("line one\nline two")
    result = registry.call("read_file", path="hello.txt")
    assert "1\tline one" in result
    assert "2\tline two" in result


def test_read_missing_file_errors(registry):
    result = registry.call("read_file", path="nope.txt")
    assert result.startswith("Error:")


def test_edit_file_overwrite(registry, tmp_path):
    (tmp_path / "hello.txt").write_text("old")
    registry.call("edit_file", path="hello.txt", content="new", mode="overwrite")
    assert (tmp_path / "hello.txt").read_text() == "new"


def test_edit_file_append(registry, tmp_path):
    (tmp_path / "hello.txt").write_text("a")
    registry.call("edit_file", path="hello.txt", content="b", mode="append")
    assert (tmp_path / "hello.txt").read_text() == "ab"


def test_edit_file_dry_run_does_not_write(registry, tmp_path):
    (tmp_path / "hello.txt").write_text("old")
    result = registry.call("edit_file", path="hello.txt", content="new", mode="overwrite", dry_run=True)

    assert (tmp_path / "hello.txt").read_text() == "old"
    assert "-old" in result
    assert "+new" in result


def test_edit_file_dry_run_append_shows_addition(registry, tmp_path):
    (tmp_path / "hello.txt").write_text("a")
    result = registry.call("edit_file", path="hello.txt", content="b", mode="append", dry_run=True)

    assert (tmp_path / "hello.txt").read_text() == "a"
    assert "+ab" in result or "+a" in result


def test_edit_missing_file_errors(registry):
    result = registry.call("edit_file", path="nope.txt", content="x")
    assert result.startswith("Error:")


def test_no_delete_tool_registered(registry):
    names = {tool.name for tool in registry.list_tools()}
    assert not any("delete" in name for name in names)


def test_memory_registry_exposes_forget_not_delete(tmp_path):
    registry = ToolRegistry()
    store = SqliteMemoryStore(tmp_path / "memory.db")
    registry.register(SaveMemoryTool(store))
    registry.register(ForgetMemoryTool(store))

    names = {tool.name for tool in registry.list_tools()}
    assert "forget" in names
    assert "delete_memory" not in names


def test_list_directory(registry, tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "sub").mkdir()
    result = registry.call("list_directory", path=".")
    assert "f  a.txt" in result
    assert "d  sub" in result


def test_search_in_files(registry, tmp_path):
    (tmp_path / "a.txt").write_text("hello world\nsecond line")
    (tmp_path / "b.txt").write_text("nothing here")
    result = registry.call("search_in_files", query="hello")
    assert "a.txt:1:hello world" in result
    assert "b.txt" not in result


def test_search_no_matches(registry, tmp_path):
    (tmp_path / "a.txt").write_text("nothing relevant")
    result = registry.call("search_in_files", query="zzz")
    assert result.startswith("No matches")


def test_path_cannot_escape_workspace_root(registry):
    result = registry.call("read_file", path="../../etc/passwd")
    assert result.startswith("Error:")
    assert "outside the workspace root" in result


def test_web_search_is_registered(registry):
    names = {tool.name for tool in registry.list_tools()}
    assert "web_search" in names
