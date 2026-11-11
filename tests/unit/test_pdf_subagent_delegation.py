from pathlib import Path
from typing import Any
import pytest
import fitz

from agent.engine.react_loop import ReactLoop
from agent.tools.registry import ToolRegistry
from agent.tools.pdf.reader import ReadPdfTool, SearchPdfTool
from agent.subagents.config import SubAgentConfig
from agent.subagents.runner import SubAgentRunner
from agent.subagents.delegate_tool import DelegateTaskTool


class SequentialMockLLM:
    """Mock LLM that returns pre-scripted responses in order."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.history: list[Any] = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.history.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return {"final_answer": "default fallback response"}


@pytest.fixture
def mock_syllabus_workspace(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "AI_Engineering_Lab_2026.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Sample Course 2026 Course Syllabus")

    p2 = doc.new_page()
    p2.insert_text((50, 72), "Late Submission Policy:\nWeekly reports submitted on Thursday are penalized 10% per day.")

    doc.save(str(pdf_path))
    doc.close()
    return tmp_path


def test_parent_delegates_to_pdf_analyst_end_to_end(mock_syllabus_workspace: Path):
    """Verifies that:
    1. Parent agent has delegate_task, but does NOT have read_pdf or search_pdf.
    2. Parent delegates to pdf_analyst.
    3. pdf_analyst runs in its own loop, calls search_pdf, and returns exact answer to parent.
    4. Parent uses the answer to reply to the user.
    """
    # 1. Setup master tools
    master_registry = ToolRegistry()
    master_registry.register(ReadPdfTool(mock_syllabus_workspace))
    master_registry.register(SearchPdfTool(mock_syllabus_workspace))

    # 2. Configure pdf_analyst
    pdf_analyst_config = SubAgentConfig(
        name="pdf_analyst",
        role="Document Analyst Specialist",
        allowed_tools=["read_pdf", "search_pdf"],
        max_steps=4,
    )

    # 3. Setup scripted LLM for both parent and subagent turns:
    # Turn 1 (Parent): LLM decides to call delegate_task
    # Turn 2 (Subagent): LLM decides to call search_pdf
    # Turn 3 (Subagent): LLM produces final answer with extracted rule
    # Turn 4 (Parent): LLM receives observation and produces final answer to user
    scripted_llm = SequentialMockLLM([
        # Turn 1: Parent calls delegate_task
        {
            "tool_call": {
                "id": "call_parent_1",
                "name": "delegate_task",
                "arguments": {
                    "subagent_type": "pdf_analyst",
                    "task": "Find late submission policy in AI_Engineering_Lab_2026.pdf",
                },
            }
        },
        # Turn 2: Subagent calls search_pdf
        {
            "tool_call": {
                "id": "call_sub_1",
                "name": "search_pdf",
                "arguments": {
                    "path": "AI_Engineering_Lab_2026.pdf",
                    "query": "Late Submission Policy",
                },
            }
        },
        # Turn 3: Subagent finishes with synthesized finding
        {
            "final_answer": "According to page 2: Weekly reports submitted on Thursday are penalized 10% per day.",
        },
        # Turn 4: Parent synthesizes final answer to student
        {
            "final_answer": "Based on the course syllabus (page 2), Thursday submissions incur a 10% penalty.",
        },
    ])

    subagent_runner = SubAgentRunner(
        llm=scripted_llm,
        master_registry=master_registry,
        configs={"pdf_analyst": pdf_analyst_config},
    )

    # 4. Parent agent registry only has delegate_task
    parent_registry = ToolRegistry()
    parent_registry.register(DelegateTaskTool(runner=subagent_runner, parent_run_id="run_001"))

    assert not parent_registry.has_tool("read_pdf")
    assert not parent_registry.has_tool("search_pdf")
    assert parent_registry.has_tool("delegate_task")

    parent_loop = ReactLoop(llm=scripted_llm, tool_registry=parent_registry, max_steps=5)
    final_answer = parent_loop.run("Can I submit my weekly report on Thursday without penalty?")

    assert "10% penalty" in final_answer
    assert "page 2" in final_answer
