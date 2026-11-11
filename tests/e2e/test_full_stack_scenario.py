"""End-to-end scenario combining every capability in one run: a
parent delegates to a sub-agent, the sub-agent's own tool call is gated by
the same permission mechanism as the parent's, the approved tool actually
executes (a real file gets written), and every step of it - parent run,
sub-agent run, and the permission decision in between - lands in a real
JSONL observability trace, correlated by run_id/parent_span_id."""

from __future__ import annotations

import json

from agent.engine.react_loop import ReactLoop
from agent.monitoring.jsonl import JsonlHook, RotatingWriter
from agent.subagents.config import SubAgentConfig
from agent.subagents.delegate_tool import DelegateTaskTool
from agent.subagents.runner import SubAgentRunner
from agent.tools.base import Tool
from agent.tools.builtin.fs_read import ReadFileTool
from agent.tools.permissions import PermissionTier
from agent.tools.registry import ToolRegistry


class WriteReportTool(Tool):
    """A CONFIRM-tier tool, standing in for any real risky tool - the point
    is that it needs a permission decision, unlike most builtin tools."""

    name = "write_report"
    description = "Writes the final report file."
    permission_tier = PermissionTier.CONFIRM
    parameters = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }

    def __init__(self, path):
        self.path = path

    def run(self, *, content: str) -> str:
        self.path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes"


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, tool_choice="auto"):
        return self._responses.pop(0)


def test_delegation_permission_enforcement_tool_execution_and_observability_together(tmp_path):
    report_path = tmp_path / "report.txt"

    llm = ScriptedLLM([
        # Parent: delegate to the sub-agent.
        {"thought": "Delegating the report to the specialist.", "tool_call": {
            "id": "parent_delegate", "name": "delegate_task",
            "arguments": {"subagent_type": "specialist", "task": "Write the report."},
        }},
        # Sub-agent: calls the CONFIRM-tier tool.
        {"tool_call": {"id": "sub_write", "name": "write_report", "arguments": {"content": "final report"}}},
        # Sub-agent: reports success.
        {"final_answer": "Report written."},
        # Parent: finishes using the delegated result.
        {"final_answer": "Done - the report has been written."},
    ])

    master_registry = ToolRegistry()
    master_registry.register(WriteReportTool(report_path))
    master_registry.register(ReadFileTool(tmp_path))

    jsonl_path = tmp_path / "events.jsonl"
    writer = RotatingWriter(tmp_path, "events.jsonl", max_file_bytes=10_000_000, backup_count=0)
    hook = JsonlHook(writer)

    config = SubAgentConfig(
        name="specialist", role="You write reports.",
        allowed_tools=["write_report"], max_steps=3,
    )
    runner = SubAgentRunner(llm=llm, master_registry=master_registry, configs={"specialist": config}, hook=hook)

    parent_registry = ToolRegistry(hook=hook)
    parent_registry.register(DelegateTaskTool(runner))

    decisions = []

    def confirm(tool_name, arguments, description):
        decisions.append(tool_name)
        return "once"

    loop = ReactLoop(llm, tool_registry=parent_registry, hook=hook)
    result = loop.run("Have the specialist write the report.", confirm_callback=confirm)

    hook.close()

    # --- sub-agent delegation: result reached the parent ---
    assert result == "Done - the report has been written."

    # --- permission enforcement: the sub-agent's own tool call was gated ---
    assert decisions == ["write_report"]

    # --- tool execution: the approved call had a real effect ---
    assert report_path.read_text(encoding="utf-8") == "final report"

    # --- observability: parent run, sub-agent run, and the permission
    # decision are all present and correlated by run_id/parent_span_id ---
    events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

    parent_runs = [e for e in events if e["event_name"] == "agent.run.finish" and e["agent_id"] == "parent"]
    subagent_runs = [e for e in events if e["event_name"] == "agent.run.finish" and e["agent_id"] == "specialist"]
    permission_events = [e for e in events if e["event_name"] == "permission.request.finish"]
    delegate_calls = [
        e for e in events
        if e["event_name"] == "tool.call.finish" and e["attributes"].get("tool_name") == "delegate_task"
    ]

    assert len(parent_runs) == 1
    assert len(subagent_runs) == 1
    assert len(permission_events) == 1
    assert len(delegate_calls) == 1

    parent_run_id = parent_runs[0]["run_id"]

    # every event in the whole delegation - parent run, the delegate_task
    # tool call, the sub-agent's own run, and the permission decision inside
    # it - shares one run_id: this is what "record sub-agent executions as
    # part of the corresponding parent run" means in the trace data.
    assert {e["run_id"] for e in (parent_runs + subagent_runs + permission_events + delegate_calls)} == {parent_run_id}

    # the sub-agent's run is a child of the delegate_task call, which is
    # itself a child of the parent run - not a disconnected, separate trace.
    assert delegate_calls[0]["parent_span_id"] == parent_runs[0]["span_id"]
    assert subagent_runs[0]["parent_span_id"] == delegate_calls[0]["span_id"]

    # the permission decision happened inside the sub-agent's own run
    assert permission_events[0]["parent_span_id"] == subagent_runs[0]["span_id"]
    assert permission_events[0]["attributes"]["decision"] == "once"
    assert permission_events[0]["attributes"]["tool_name"] == "write_report"


def test_a_config_denied_tool_is_blocked_inside_a_sub_agent_and_shows_up_in_the_trace(tmp_path):
    """The `deny` outcome, end to end: the policy blocks the sub-agent's tool
    call without asking anyone, nothing is written, the delegation still
    completes, and the trace records the decision as a policy denial."""
    from agent.tools.permissions import PermissionPolicy

    report_path = tmp_path / "report.txt"
    llm = ScriptedLLM([
        {"tool_call": {"id": "p", "name": "delegate_task",
                       "arguments": {"subagent_type": "specialist", "task": "Write the report."}}},
        {"tool_call": {"id": "s", "name": "write_report", "arguments": {"content": "final report"}}},
        {"final_answer": "The report tool is disabled, so nothing was written."},
        {"final_answer": "Could not write the report: the tool is disabled here."},
    ])
    master_registry = ToolRegistry()
    master_registry.register(WriteReportTool(report_path))

    writer = RotatingWriter(tmp_path, "events.jsonl", max_file_bytes=10_000_000, backup_count=0)
    hook = JsonlHook(writer)
    policy = PermissionPolicy(deny_tools=["write_report"])
    config = SubAgentConfig(name="specialist", role="You write reports.", allowed_tools=["write_report"], max_steps=3)
    runner = SubAgentRunner(
        llm=llm, master_registry=master_registry, configs={"specialist": config}, hook=hook, permission_policy=policy,
    )
    parent_registry = ToolRegistry(hook=hook)
    parent_registry.register(DelegateTaskTool(runner))
    asked = []

    result = ReactLoop(llm, tool_registry=parent_registry, hook=hook, permission_policy=policy).run(
        "Have the specialist write the report.", confirm_callback=lambda *args: asked.append(args) or "once",
    )
    hook.close()

    assert result == "Could not write the report: the tool is disabled here."
    assert asked == []
    assert not report_path.exists()

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    (decision,) = [e for e in events if e["event_name"] == "permission.request.finish"]
    subagent_run = next(e for e in events if e["event_name"] == "agent.run.finish" and e["agent_id"] == "specialist")
    assert decision["attributes"]["decision"] == "policy_deny"
    assert decision["attributes"]["tier"] == "deny"
    assert decision["parent_span_id"] == subagent_run["span_id"]
