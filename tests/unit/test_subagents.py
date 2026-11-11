from pathlib import Path
from typing import Any
import pytest
import fitz

from agent.tools.base import Tool, ToolError
from agent.tools.permissions import PermissionTier
from agent.tools.registry import ToolRegistry
from agent.tools.pdf.reader import ReadPdfTool, SearchPdfTool
from agent.engine.react_loop import ReactLoop
from agent.subagents.config import SubAgentConfig
from agent.subagents.runner import SubAgentRunner
from agent.subagents.delegate_tool import DelegateTaskTool


class FakeLLM:
    """Fake LLM client that can simulate tool calls or return direct answers."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.call_count = 0
        self.history: list[Any] = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.call_count += 1
        self.history.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return {"final_answer": "default response"}


class DummySecretTool(Tool):
    name = "secret_nuclear_launch"
    description = "Super dangerous tool that subagents must not have access to"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self) -> str:
        return "Launched"


@pytest.fixture
def workspace_with_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "lab_policy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "AI Lab Policy: Late submission deadline is Thursday 18:00.")
    doc.save(str(pdf_path))
    doc.close()
    return tmp_path


def test_subagent_config_and_registration():
    master_registry = ToolRegistry()
    fake_llm = FakeLLM([])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry)

    config = SubAgentConfig(
        name="pdf_analyst",
        role="Document specialist",
        allowed_tools=["read_pdf", "search_pdf"],
        max_steps=5,
    )
    runner.register_subagent(config)

    assert runner.has_subagent("pdf_analyst")
    assert not runner.has_subagent("unknown_agent")
    assert len(runner.list_subagents()) == 1


def test_subagent_tool_isolation(workspace_with_pdf: Path):
    """Verifies that subagents only receive tools from their allowed_tools whitelist."""
    master_registry = ToolRegistry()
    master_registry.register(ReadPdfTool(workspace_with_pdf))
    master_registry.register(SearchPdfTool(workspace_with_pdf))
    master_registry.register(DummySecretTool())

    # Sub-agent allowed only search_pdf
    config = SubAgentConfig(
        name="restricted_agent",
        role="Search only",
        allowed_tools=["search_pdf"],
        max_steps=3,
    )

    fake_llm = FakeLLM([
        # Step 1: Subagent calls search_pdf
        {
            "tool_call": {
                "id": "call_1",
                "name": "search_pdf",
                "arguments": {"path": "lab_policy.pdf", "query": "Late submission"},
            }
        },
        # Step 2: Final answer
        {"final_answer": "Late deadline is Thursday 18:00."},
    ])

    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry, configs={"restricted_agent": config})
    result = runner.run_subagent(name="restricted_agent", task="Find late submission policy")

    assert "Thursday 18:00" in result
    # Check that tools passed to subagent LLM only contained search_pdf, NEVER DummySecretTool
    first_call_tools = fake_llm.history[0]["tools"]
    tool_names = [t["function"]["name"] for t in first_call_tools]
    assert "search_pdf" in tool_names
    assert "secret_nuclear_launch" not in tool_names
    assert "read_pdf" not in tool_names


def test_delegate_task_tool(workspace_with_pdf: Path):
    master_registry = ToolRegistry()
    master_registry.register(ReadPdfTool(workspace_with_pdf))

    config = SubAgentConfig(
        name="pdf_analyst",
        role="You are a PDF analyst",
        allowed_tools=["read_pdf"],
        max_steps=3,
    )

    fake_llm = FakeLLM([
        {"final_answer": "Policy extracted successfully: Thursday 18:00."},
    ])

    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry, configs={"pdf_analyst": config})
    delegate_tool = DelegateTaskTool(runner=runner, parent_run_id="parent_123")

    output = delegate_tool.run(subagent_type="pdf_analyst", task="Check lab_policy.pdf")
    assert "Policy extracted successfully" in output


def test_delegate_unknown_subagent():
    master_registry = ToolRegistry()
    fake_llm = FakeLLM([])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry)
    delegate_tool = DelegateTaskTool(runner=runner)

    output = delegate_tool.run(subagent_type="non_existent", task="do something")
    assert "Error: Unknown sub-agent type 'non_existent'" in output


def test_delegate_empty_arguments():
    master_registry = ToolRegistry()
    fake_llm = FakeLLM([])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry)
    delegate_tool = DelegateTaskTool(runner=runner)

    with pytest.raises(ToolError, match="Both 'subagent_type' and 'task' are required"):
        delegate_tool.run(subagent_type="", task="some task")


def test_delegate_streams_subagent_steps_to_parent_callback():
    master_registry = ToolRegistry()
    tool = DummySecretTool()
    tool.permission_tier = PermissionTier.REGULAR  # step streaming, not permissions, is under test
    master_registry.register(tool)
    config = SubAgentConfig(
        name="specialist",
        role="Use the assigned tool.",
        allowed_tools=["secret_nuclear_launch"],
        max_steps=3,
    )
    fake_llm = FakeLLM([
        {
            "thought": "I should delegate this task.",
            "tool_call": {
                "id": "parent_delegate",
                "name": "delegate_task",
                "arguments": {"subagent_type": "specialist", "task": "Run the check"},
            }
        },
        {
            "tool_call": {
                "id": "subagent_tool",
                "name": "secret_nuclear_launch",
                "arguments": {},
            }
        },
        {"final_answer": "Sub-agent finished."},
        {"final_answer": "Parent finished."},
    ])
    runner = SubAgentRunner(
        llm=fake_llm,
        master_registry=master_registry,
        configs={"specialist": config},
    )
    parent_registry = ToolRegistry()
    delegate = DelegateTaskTool(runner)
    parent_registry.register(delegate)
    steps = []

    result = ReactLoop(fake_llm, parent_registry).run("Delegate this", step_callback=steps.append)

    assert result == "Parent finished."
    assert [step.agent_name for step in steps] == ["parent", "specialist", "parent"]
    assert steps[0].thought is not None
    assert steps[1].tool_calls[0].name == "secret_nuclear_launch"
    assert steps[1].observations[0].content == "Launched"
    assert steps[2].tool_calls[0].name == "delegate_task"
    assert delegate.step_callback is None


def test_delegate_threads_confirm_callback_to_subagent_tool_calls():
    """delegate_task itself is REGULAR (delegation isn't the risky part),
    but DummySecretTool defaults to CONFIRM - the permission gate must
    reach the sub-agent's own tool calls, not just the parent's."""
    master_registry = ToolRegistry()
    master_registry.register(DummySecretTool())
    config = SubAgentConfig(
        name="specialist",
        role="Use the assigned tool.",
        allowed_tools=["secret_nuclear_launch"],
        max_steps=3,
    )
    fake_llm = FakeLLM([
        {
            "tool_call": {
                "id": "parent_delegate",
                "name": "delegate_task",
                "arguments": {"subagent_type": "specialist", "task": "Run the check"},
            }
        },
        {"tool_call": {"id": "subagent_tool", "name": "secret_nuclear_launch", "arguments": {}}},
        {"final_answer": "Sub-agent finished."},
        {"final_answer": "Parent finished."},
    ])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry, configs={"specialist": config})
    parent_registry = ToolRegistry()
    delegate = DelegateTaskTool(runner)
    parent_registry.register(delegate)

    confirm_calls = []

    def confirm(tool_name, arguments, description):
        confirm_calls.append(tool_name)
        return "once"

    result = ReactLoop(fake_llm, parent_registry).run("Delegate this", confirm_callback=confirm)

    assert result == "Parent finished."
    assert confirm_calls == ["secret_nuclear_launch"]
    assert delegate.confirm_callback is None  # cleared after, like step_callback


def test_subagent_step_limit_is_rewritten_into_an_unambiguous_error():
    """Regression test: a sub-agent that exhausts its step budget without a
    final_answer used to return the bare string "Stopped: maximum number of
    steps reached." as its delegate_task result. That has no "Error:" prefix,
    so a parent reading it as a tool observation had no structural signal it
    was a failure - it just re-delegated the identical task, over and over,
    hitting the same limit every time. The result must now clearly read as
    an error and tell the caller not to blindly retry."""
    master_registry = ToolRegistry()
    master_registry.register(DummySecretTool())
    config = SubAgentConfig(
        name="looper", role="Keep using the tool.",
        allowed_tools=["secret_nuclear_launch"], max_steps=2,
    )
    # Never returns a final_answer, so the sub-loop always falls through.
    fake_llm = FakeLLM([
        {"tool_call": {"id": "c1", "name": "secret_nuclear_launch", "arguments": {}}},
        {"tool_call": {"id": "c2", "name": "secret_nuclear_launch", "arguments": {}}},
    ])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry, configs={"looper": config})

    result = runner.run_subagent(name="looper", task="do the thing")

    assert result.startswith("Error:")
    assert "looper" in result
    assert "max_steps=2" in result
    assert "not undone" in result
    assert "Do not immediately retry" in result


def test_subagent_normal_result_is_not_mangled():
    master_registry = ToolRegistry()
    config = SubAgentConfig(name="quick", role="Answer directly.", allowed_tools=[], max_steps=3)
    fake_llm = FakeLLM([{"final_answer": "Stopped early on purpose, but this is a real answer."}])
    runner = SubAgentRunner(llm=fake_llm, master_registry=master_registry, configs={"quick": config})

    result = runner.run_subagent(name="quick", task="do the thing")

    assert result == "Stopped early on purpose, but this is a real answer."


# --- one permission policy for parent and sub-agents -------------------------

from agent.tools.permissions import PermissionPolicy  # noqa: E402


def _delegating_setup(policy, tool):
    """Parent delegates to the same sub-agent twice; each time the sub-agent
    calls `tool`. Returns (parent_loop_factory, delegate_calls)."""
    master_registry = ToolRegistry()
    master_registry.register(tool)
    config = SubAgentConfig(name="specialist", role="r", allowed_tools=[tool.name], max_steps=3)
    delegate_args = {"subagent_type": "specialist", "task": "t"}
    llm = FakeLLM([
        {"tool_call": {"id": "p1", "name": "delegate_task", "arguments": delegate_args}},
        {"tool_call": {"id": "s1", "name": tool.name, "arguments": {}}},
        {"final_answer": "first"},
        {"tool_call": {"id": "p2", "name": "delegate_task", "arguments": delegate_args}},
        {"tool_call": {"id": "s2", "name": tool.name, "arguments": {}}},
        {"final_answer": "second"},
        {"final_answer": "all done"},
    ])
    runner = SubAgentRunner(
        llm=llm, master_registry=master_registry, configs={"specialist": config}, permission_policy=policy,
    )
    parent_registry = ToolRegistry()
    parent_registry.register(DelegateTaskTool(runner))
    return ReactLoop(llm, tool_registry=parent_registry, permission_policy=policy)


def test_an_always_answered_in_one_delegation_holds_for_the_next():
    asked = []

    def confirm(tool_name, arguments, description):
        asked.append(tool_name)
        return "always"

    loop = _delegating_setup(PermissionPolicy(), DummySecretTool())  # CONFIRM by default

    assert loop.run("go", confirm_callback=confirm) == "all done"
    assert asked == ["secret_nuclear_launch"]  # once, not once per delegation


def test_deny_tools_blocks_a_sub_agents_tool_call_without_asking():
    asked = []
    tool = DummySecretTool()
    tool.run = lambda: pytest.fail("a denied tool must not run")
    loop = _delegating_setup(PermissionPolicy(deny_tools=["secret_nuclear_launch"]), tool)

    result = loop.run("go", confirm_callback=lambda *args: asked.append(args) or "once")

    assert result == "all done"
    assert asked == []


# --- delegation is not a leaf tool ---------------------------------------------

def test_delegate_task_is_labelled_as_a_sub_agent_origin_not_a_builtin_tool():
    """Its duration is a whole nested sub-agent run, so the dashboards must be
    able to tell it apart from a built-in or MCP tool that ran in milliseconds."""
    runner = SubAgentRunner(llm=FakeLLM([]), master_registry=ToolRegistry())

    assert DelegateTaskTool(runner).origin == "subagent"


def test_a_delegation_emits_its_tool_events_with_the_subagent_origin():
    from agent.observability import LifecycleEvent

    class Recorder:
        def __init__(self):
            self.events: list[LifecycleEvent] = []

        def emit(self, event):
            self.events.append(event)

    hook = Recorder()
    llm = FakeLLM([
        {"tool_call": {"id": "p", "name": "delegate_task",
                       "arguments": {"subagent_type": "specialist", "task": "t"}}},
        {"final_answer": "sub done"},
        {"final_answer": "all done"},
    ])
    config = SubAgentConfig(name="specialist", role="r", allowed_tools=[], max_steps=2)
    runner = SubAgentRunner(llm=llm, master_registry=ToolRegistry(), configs={"specialist": config}, hook=hook)
    registry = ToolRegistry(hook=hook)
    registry.register(DelegateTaskTool(runner))

    ReactLoop(llm, tool_registry=registry, hook=hook).run("go")

    (finish,) = [e for e in hook.events if e.event_name == "tool.call.finish"]
    assert finish.attributes == {"tool_name": "delegate_task", "origin": "subagent"}


def test_a_sub_agent_prompt_carries_its_role_and_the_untrusted_content_notice():
    from agent.engine.react_loop import UNTRUSTED_CONTENT_NOTICE

    llm = FakeLLM([{"final_answer": "done"}])
    config = SubAgentConfig(name="specialist", role="You are the specialist.", allowed_tools=[], max_steps=2)
    runner = SubAgentRunner(llm=llm, master_registry=ToolRegistry(), configs={"specialist": config})

    runner.run_subagent("specialist", "t")

    system = llm.history[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("You are the specialist.")
    assert UNTRUSTED_CONTENT_NOTICE in system["content"]


def test_a_denial_inside_one_delegation_holds_when_the_parent_delegates_again():
    """Otherwise the parent can re-delegate and the user is asked again for
    the tool they just refused, once per delegation."""
    asked = []
    tool = DummySecretTool()
    tool.run = lambda: pytest.fail("a denied tool must not run")
    loop = _delegating_setup(PermissionPolicy(), tool)  # CONFIRM by default

    result = loop.run("go", confirm_callback=lambda *args: asked.append(args[0]) or "deny")

    assert result == "all done"
    assert asked == ["secret_nuclear_launch"]  # asked once, not once per delegation


def test_a_refusal_does_not_carry_over_to_the_next_request():
    policy = PermissionPolicy()
    policy.refused_in_request.add("secret_nuclear_launch")

    policy.start_request()

    assert policy.refused_in_request == set()
