"""End-to-end: a task that chains memory tools and filesystem tools in one
conversation, then verifies both the file on disk and that forgetting a
fact genuinely removes it (not just clears an in-memory cache) across a
restart - semantic correctness of the actual saved state, not just that
the calls didn't raise."""

from agent.engine.react_loop import ReactLoop
from agent.memory import ForgetMemoryTool, RecallMemoryTool, SaveMemoryTool, SqliteMemoryStore
from agent.tools import ToolRegistry
from agent.tools.builtin import register_builtin_tools


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def approve(tool_name, arguments, description):
    """These scenarios exercise real tools that need a yes; this is the user saying yes."""
    return "once"


def build_registry(tmp_path, memory_store):
    registry = ToolRegistry()
    register_builtin_tools(registry, tmp_path)
    registry.register(SaveMemoryTool(memory_store))
    registry.register(RecallMemoryTool(memory_store))
    registry.register(ForgetMemoryTool(memory_store))
    return registry


def test_remember_write_file_then_forget_survives_a_restart(tmp_path):
    memory_path = tmp_path / "memory.db"

    # --- session 1: remember a fact, then use it while writing a file ---
    store_1 = SqliteMemoryStore(memory_path)
    registry_1 = build_registry(tmp_path, store_1)
    llm_1 = FakeLLM([
        {"tool_call": {"id": "call_1", "name": "save_memory",
                       "arguments": {"key": "project_name", "value": "Aurora"}}},
        {"tool_call": {"id": "call_2", "name": "create_file",
                       "arguments": {"path": "readme_note.txt", "content": "Project: Aurora"}}},
        {"final_answer": "saved and written"},
    ])
    loop_1 = ReactLoop(llm_1, tool_registry=registry_1, max_steps=5, memory_store=store_1)

    result_1 = loop_1.run("remember the project name is Aurora, then write it to a file", confirm_callback=approve)

    assert result_1 == "saved and written"
    assert (tmp_path / "readme_note.txt").read_text() == "Project: Aurora"
    assert store_1.get("project_name") == "Aurora"

    # --- session 2 (simulated restart): a fresh store retrieves the fact
    # through the SQLite-backed recall tool ---
    store_2 = SqliteMemoryStore(memory_path)
    registry_2 = build_registry(tmp_path, store_2)
    llm_2 = FakeLLM([
        {"tool_call": {"id": "call_recall", "name": "recall_memory", "arguments": {"query": "project_name"}}},
        {"final_answer": "It's Aurora."},
    ])
    loop_2 = ReactLoop(llm_2, tool_registry=registry_2, max_steps=3, memory_store=store_2)

    result_2 = loop_2.run("what's the project name?", confirm_callback=approve)

    assert result_2 == "It's Aurora."
    recall_messages = [message["content"] for message in loop_2.messages if message.get("role") == "tool"]
    assert any("Aurora" in message for message in recall_messages)

    # --- session 2 continued: forget the fact ---
    llm_2b = FakeLLM([
        {"tool_call": {"id": "call_3", "name": "forget", "arguments": {"key": "project_name"}}},
        {"final_answer": "forgotten"},
    ])
    loop_2.llm = llm_2b
    result_2b = loop_2.run("forget the project name", confirm_callback=approve)

    assert result_2b == "forgotten"
    assert store_2.get("project_name") is None

    # --- session 3 (another restart): the fact is genuinely gone from
    # disk, not just cleared from session 2's in-memory state ---
    store_3 = SqliteMemoryStore(memory_path)
    assert store_3.get("project_name") is None
    assert store_3.format_for_prompt() == "" or "project_name" not in store_3.format_for_prompt()
