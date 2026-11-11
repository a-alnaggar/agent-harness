from pathlib import Path
from typing import Any

from agent.engine.react_loop import ReactLoop
from agent.memory import CoreProfileManager, SqliteMemoryStore, UpdateCorePreferenceTool
from agent.memory.routing import route_memory_mutation
from agent.subagents import DelegateTaskTool, SubAgentConfig, SubAgentRunner
from agent.tools import ToolRegistry


class SequentialLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        return self.responses.pop(0)


def test_name_change_is_routed_to_memory_manager():
    route = route_memory_mutation("okay now my name is Ahmed")

    assert route is not None
    assert route["name"] == "delegate_task"
    assert route["arguments"]["subagent_type"] == "memory_manager"
    assert "update_core_preference" in route["arguments"]["task"]


def test_arabic_name_change_is_routed_to_memory_manager():
    route = route_memory_mutation("اسمي أحمد، تذكره")

    assert route is not None
    assert route["arguments"]["subagent_type"] == "memory_manager"


def test_unrelated_request_is_not_forced_to_memory_manager():
    assert route_memory_mutation("create a Python file") is None


def test_name_update_delegates_persists_and_survives_restart(tmp_path: Path):
    core_path = tmp_path / "core.md"
    core_mgr = CoreProfileManager(core_path)
    store = SqliteMemoryStore(tmp_path / "memory.db")
    master = ToolRegistry()
    master.register(UpdateCorePreferenceTool(core_mgr, store))
    llm = SequentialLLM([
        {
            "tool_call": {
                "name": "update_core_preference",
                "arguments": {"key": "user_name", "value": "Ahmed"},
            }
        },
        {"final_answer": "The core profile was updated successfully."},
        {"final_answer": "Your name is now Ahmed."},
    ])
    runner = SubAgentRunner(
        llm=llm,
        master_registry=master,
        configs={
            "memory_manager": SubAgentConfig(
                name="memory_manager",
                role="Use the memory tool before reporting success.",
                allowed_tools=["update_core_preference"],
                require_successful_tool=True,
            )
        },
    )
    parent_tools = ToolRegistry()
    parent_tools.register(DelegateTaskTool(runner))
    parent = ReactLoop(llm, parent_tools, tool_router=route_memory_mutation)

    answer = parent.run("okay now my name is Ahmed")

    assert answer == "Your name is now Ahmed."
    assert CoreProfileManager(core_path).get_preference("user_name") == "Ahmed"
    assert store.get_general_value("user_name") is None


def test_reminders_about_a_task_are_not_memory_requests():
    for text in ("Remember to run the tests before you finish.",
                 "don't forget to check the sandbox",
                 "Do not forget the attachment",
                 "Please forget to mention it"):
        assert route_memory_mutation(text) is None, text


def test_real_memory_requests_are_still_routed():
    for text in ("Remember that I prefer short answers.",
                 "Please remember my office is room 204",
                 "Forget my email summary preference.",
                 "What do you remember about me?",
                 "forget it"):
        assert route_memory_mutation(text) is not None, text
