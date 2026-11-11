"""Automated capability scenarios (A1-A6): every agent
from config/default.yaml - the parent and each configured sub-agent - doing
basic read and write work, with every permission outcome.

Each scenario boots the real composition root (`agent.__main__.main`) in a
temporary workspace with the committed config, so the tool set, the
sub-agents' roles and allowed tools, and the MCP permission tiers are
exactly what a real session gets. Only three things are replaced: the LLM is
scripted, the CLI is a scripted user who types prompts and answers approval
prompts, and the Gmail server runs without credentials (so nothing is ever
read from or sent to a real mailbox). Notion is left out: it needs npx, a
token and the network (manual scenario M3)."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable

import fitz
import pytest

pytest.importorskip("mcp")

import agent.__main__ as main_module  # noqa: E402
from agent.config import PermissionsConfig, load_config  # noqa: E402
from agent.memory import CoreProfileManager, SqliteMemoryStore  # noqa: E402
from agent.tools.permissions import PermissionTier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_ID = "sample-course"


class ScriptedLLM:
    """Plays the model for the parent and every sub-agent, in call order,
    and records which tools each call was offered."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.offered: list[set[str]] = []
        self.system_prompts: list[str] = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.offered.append({t["function"]["name"] for t in tools or []})
        self.system_prompts.append(messages[0]["content"])
        return self._responses.pop(0)

    def remaining(self) -> int:
        return len(self._responses)


class ScriptedUser:
    """Answers approval prompts with the CLI's keys (y = once, a = always this
    session, n = deny; cli_interface/repl.py), recording what was asked."""

    _DECISIONS = {"y": "once", "a": "always", "n": "deny"}

    def __init__(self, *keys: str):
        self._keys = list(keys)
        self.asked: list[str] = []

    def __call__(self, tool_name: str, arguments: dict, description: str) -> str:
        self.asked.append(tool_name)
        return self._DECISIONS[self._keys.pop(0)]


@dataclasses.dataclass
class Session:
    answers: list[str]
    # per turn: the parent's tool observations
    tool_messages: list[list[str]]
    # per sub-agent run: (agent name, the sub-agent's own tool observations)
    subagent_runs: list[tuple[str, list[str]]]
    registry: Any


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_call": {"id": f"call_{name}", "name": name, "arguments": arguments}}


def answer(text: str) -> dict[str, Any]:
    return {"final_answer": text}


def committed_config(**overrides: Any):
    """config/default.yaml as shipped, with the sandbox and Notion switched
    off and Gmail started with this interpreter (same module, no PATH lookup)."""
    config = load_config(REPO_ROOT / "config" / "default.yaml")
    gmail = next(s for s in config.mcp_servers if s.name == "gmail")
    gmail = dataclasses.replace(gmail, command=sys.executable)
    return dataclasses.replace(config, sandbox=None, mcp_servers=[gmail], **overrides)


def run_session(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    llm: ScriptedLLM,
    turns: list[tuple[str, Callable | None]],
    config: Any = None,
) -> Session:
    """Starts the agent the way `python -m agent` does and types `turns`
    (prompt, approval callback) at it, one run per prompt."""
    import agent.subagents.runner as runner_module

    config = config or committed_config()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(main_module, "LLMClient", lambda **_: llm)
    session = Session(answers=[], tool_messages=[], subagent_runs=[], registry=None)

    sub_agent_loop = runner_module.ReactLoop

    class RecordingSubAgentLoop(sub_agent_loop):
        def run(self, *args: Any, **kwargs: Any) -> str:
            result = super().run(*args, **kwargs)
            session.subagent_runs.append(
                (self.agent_id, [m["content"] for m in self.messages if m.get("role") == "tool"])
            )
            return result

    monkeypatch.setattr(runner_module, "ReactLoop", RecordingSubAgentLoop)

    def scripted_cli(*, config, engine, registry, context_manager) -> None:
        session.registry = registry
        for prompt, confirm in turns:
            seen = len(engine._loop.messages)
            session.answers.append(engine.run(prompt, confirm_callback=confirm).final_answer)
            session.tool_messages.append(
                [m["content"] for m in engine._loop.messages[seen:] if m.get("role") == "tool"]
            )

    monkeypatch.setattr(main_module, "start_cli", scripted_cli)
    main_module.main()
    assert llm.remaining() == 0, "the scenario script was not fully consumed"
    return session


def never_asked(*_: Any) -> str:
    raise AssertionError("a REGULAR tool asked for approval")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    # Never pick up a real key, token or mailbox from the machine running the tests.
    for name in ("TAVILY_API_KEY", "SANDBOX_SSH_KEY", "GMAIL_ALLOWED_RECIPIENTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GMAIL_CREDENTIALS_PATH", str(tmp_path / "no-gmail" / "credentials.json"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "no-gmail" / "gmail-token.json"))
    return root


def write_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()


# --- A1: parent agent, workspace files (REGULAR: no approval) ---------------

def test_a1_parent_creates_edits_reads_lists_and_searches_workspace_files(monkeypatch, workspace):
    user = ScriptedUser()
    llm = ScriptedLLM([
        call("create_folder", path="notes"),
        call("create_file", path="notes/todo.md", content="- write report\n"),
        call("edit_file", path="notes/todo.md", content="- review slides\n", mode="append"),
        call("read_file", path="notes/todo.md"),
        call("list_directory", path="notes"),
        call("search_in_files", query="slides"),
        answer("notes/todo.md has two items."),
        call("create_file", path="../outside.txt", content="should never be written"),
        answer("That path is outside the workspace."),
    ])

    session = run_session(monkeypatch, workspace, llm, [
        ("Create notes/todo.md with '- write report', append '- review slides', then show it.", user),
        ("Write outside.txt one folder above the workspace.", user),
    ])

    assert (workspace / "notes" / "todo.md").read_text() == "- write report\n- review slides\n"
    created, updated, read, listed, searched = session.tool_messages[0][1:]
    assert created == "Created file 'notes/todo.md' (15 bytes)"
    assert updated == "Updated 'notes/todo.md' (append)"
    assert "write report" in read and "review slides" in read
    assert "todo.md" in listed
    assert "todo.md" in searched and "review slides" in searched
    assert not any(m.startswith("Error") for m in session.tool_messages[0])

    (escape,) = session.tool_messages[1]
    assert escape.startswith("Error:")
    assert not (workspace.parent / "outside.txt").exists()
    assert user.asked == []


# --- A2: parent agent, web search (REGULAR; Tavily faked) --------------------

def test_a2_web_search_without_a_key_is_a_controlled_error(monkeypatch, workspace):
    llm = ScriptedLLM([
        call("web_search", query="weather in Passau today"),
        answer("Web search is not configured."),
    ])

    session = run_session(monkeypatch, workspace, llm, [("search the web to find the weather in passau today", never_asked)])

    assert session.tool_messages[0] == [
        "Error: TAVILY_API_KEY is not set — get a free key at https://tavily.com "
        "and set it as an environment variable"
    ]


def test_a2_web_search_with_a_key_returns_the_answer_and_its_sources(monkeypatch, workspace):
    import urllib.request

    def fake_tavily(request, timeout):
        assert request.full_url == "https://api.tavily.com/search"
        assert json.loads(request.data)["query"] == "weather in Passau today"
        body = {"answer": "Cloudy, 14 C.", "results": [{"url": "https://weather.example/passau"}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_tavily)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-placeholder")
    llm = ScriptedLLM([
        call("web_search", query="weather in Passau today"),
        answer("Cloudy, 14 C."),
    ])

    session = run_session(monkeypatch, workspace, llm, [("search the web to find the weather in passau today", never_asked)])

    assert session.tool_messages[0] == ["Cloudy, 14 C. (sources: https://weather.example/passau)"]


# --- A3: parent agent, Gmail MCP (REGULAR / CONFIRM / STRICT / allowlist) ----

def test_a3_gmail_tools_follow_their_configured_tiers_without_touching_a_mailbox(monkeypatch, workspace):
    monkeypatch.setenv("GMAIL_ALLOWED_RECIPIENTS", "@example.com")
    draft = dict(subject="Re: final report", body="The limit is in the syllabus.")
    user = ScriptedUser("a", "n", "y", "y")
    llm = ScriptedLLM([
        call("search_emails", query="is:unread", max_results=1),
        answer("Gmail is not authorized here."),
        call("create_draft", to="alice@example.com", **draft),
        call("create_draft", to="alice@example.com", **draft),
        answer("Tried to draft twice."),
        call("send_email", to="alice@example.com", **draft),
        answer("Not sent: you declined."),
        call("send_email", to="alice@example.com", **draft),
        call("send_email", to="someone@example.org", **draft),
        answer("Neither mail went out."),
    ])

    session = run_session(monkeypatch, workspace, llm, [
        ("Search my inbox for unread emails.", user),
        ("Draft a reply to alice@example.com, twice.", user),
        ("Send that reply to alice@example.com.", user),
        ("Send it to alice@example.com and to someone@example.org.", user),
    ])

    tiers = {name: session.registry.get(name).permission_tier
             for name in ("search_emails", "read_email", "create_draft", "send_email")}
    assert tiers == {
        "search_emails": PermissionTier.REGULAR, "read_email": PermissionTier.REGULAR,
        "create_draft": PermissionTier.CONFIRM, "send_email": PermissionTier.STRICT,
    }
    # create_draft: asked once, "always" covers the second call. send_email:
    # asked on every call, including after a "y".
    assert user.asked == ["create_draft", "send_email", "send_email", "send_email"]

    # No credentials in this workspace: every approved call reaches the
    # server and stops at authorization, so nothing is read, drafted or sent.
    unauthorized = "Gmail authorization unavailable"
    assert unauthorized in session.tool_messages[0][0]
    assert all(unauthorized in m for m in session.tool_messages[1])
    assert session.tool_messages[2][0].startswith("Error: user denied permission to run 'send_email'")
    allowed, blocked = session.tool_messages[3]
    assert unauthorized in allowed
    assert "not on this deployment's allowlist" in blocked


def test_a3_a_tool_listed_in_deny_tools_is_never_offered_a_prompt_or_run(monkeypatch, workspace):
    user = ScriptedUser()
    llm = ScriptedLLM([
        call("send_email", to="alice@example.com", subject="hi", body="hello"),
        answer("Sending mail is disabled in this deployment."),
    ])
    config = committed_config(permissions=PermissionsConfig(deny_tools=["send_email"]))

    session = run_session(monkeypatch, workspace, llm, [("Email alice@example.com.", user)], config=config)

    assert user.asked == []
    assert session.tool_messages[0][0].startswith(
        "Error: 'send_email' is blocked by this deployment's permission policy."
    )


# --- A4: memory_manager (write, read, CONFIRM forget) ------------------------

def test_a4_memory_manager_saves_recalls_and_forgets_with_confirmation(monkeypatch, workspace):
    user = ScriptedUser("y")
    llm = ScriptedLLM([
        # "Remember ...", "My name is ..." and "Forget ..." are routed to
        # memory_manager without asking the model (memory/routing.py).
        call("save_memory", key="summary_style", value="short bullet points", save_reason="explicit"),
        answer("Saved your summary preference."),
        answer("I'll keep summaries to short bullet points."),
        call("update_core_preference", key="user_name", value="Alex Example"),
        answer("Name stored."),
        answer("Nice to meet you, Alex."),
        call("recall_memory", query="summary"),
        answer("You prefer short bullet points."),
        answer("You like short bullet-point summaries."),
        call("forget", key="summary_style"),
        answer("Forgot the summary preference."),
        answer("Done, I no longer have that preference."),
    ])

    session = run_session(monkeypatch, workspace, llm, [
        ("Remember that I prefer my email summaries as short bullet points.", user),
        ("My name is Alex Example.", user),
        ("What do you remember about my summary style?", user),
        ("Forget my summary style preference.", user),
    ])

    assert llm.offered[0] == {
        "save_memory", "recall_memory", "forget", "save_faq", "search_faq", "update_core_preference",
    }
    assert "delegate_task" in llm.offered[2] and "save_memory" not in llm.offered[2]

    (_, saved), (_, core), (_, recalled), (_, forgot) = session.subagent_runs
    assert [name for name, _ in session.subagent_runs] == ["memory_manager"] * 4
    assert saved == ["Saved to memory: summary_style = short bullet points"]
    assert "Alex Example" in core[0]
    assert "short bullet points" in recalled[0]
    assert forgot == ["Forgot from memory: summary_style"]
    assert user.asked == ["forget"]

    # the core profile reaches the parent's system prompt on later turns
    assert "Alex Example" in llm.system_prompts[8]
    memory_dir = workspace / ".agent_memory"
    assert CoreProfileManager(memory_dir / "core.md").get_preference("user_name") == "Alex Example"
    assert SqliteMemoryStore(memory_dir / "memory.db").get("summary_style") is None


# --- A5: course_knowledge (cited answer, FAQ cache write, cache hit) ---------

def test_a5_course_knowledge_answers_with_a_citation_then_reuses_the_cached_faq(monkeypatch, workspace):
    config = committed_config()
    material = config.courses[COURSE_ID].materials[0]
    write_pdf(workspace / material.path, [
        "Sample Course - Syllabus",
        "Final report: at most 10 pages, due at the end of the term.",
    ])
    question = "What is the page limit for the final report?"
    cited = "At most 10 pages (Course Syllabus, page 2)."
    llm = ScriptedLLM([
        call("delegate_task", subagent_type="course_knowledge", task=question),
        call("search_faq", course_id=COURSE_ID, query="final report page limit"),
        call("search_course_material", course_id=COURSE_ID, query="final report pages"),
        call("read_course_page", course_id=COURSE_ID, document_id=material.id, page=2),
        call("save_faq", course_id=COURSE_ID, question=question, answer=cited,
             source_type="pdf", source_ref="Course Syllabus, page 2"),
        answer(cited),
        answer(cited),
        call("delegate_task", subagent_type="course_knowledge", task=question),
        call("search_faq", course_id=COURSE_ID, query="final report page limit"),
        answer(cited),
        answer(cited),
    ])

    session = run_session(monkeypatch, workspace, llm, [(question, never_asked), (question, never_asked)], config=config)

    assert llm.offered[1] == {"search_course_material", "read_course_page", "search_faq", "save_faq"}
    assert session.answers == [cited, cited]

    (_, first), (_, second) = session.subagent_runs
    faq_miss, evidence, page, saved = first
    assert faq_miss.startswith("No cached FAQs found")
    assert "Page: 2" in evidence and "10 pages" in evidence
    assert "10 pages" in page
    assert not saved.startswith("Error")
    # the second question never reaches the PDF: the cached FAQ answers it
    (faq_hit,) = second
    assert "10 pages" in faq_hit

    (faq,) = SqliteMemoryStore(workspace / ".agent_memory" / "memory.db").search_faq(COURSE_ID, "final report page limit")
    assert faq.answer == cited and faq.source_type == "pdf"


# --- A6: pdf_analyst (read-only, confined to its allowed tools) --------------

def test_a6_pdf_analyst_reads_a_workspace_pdf(monkeypatch, workspace):
    write_pdf(workspace / "docs" / "handbook.pdf", [
        "Team handbook",
        "Office hours: Tuesdays 14:00-15:00 in room IM 033.",
    ])
    llm = ScriptedLLM([
        call("delegate_task", subagent_type="pdf_analyst", task="Find the office hours in docs/handbook.pdf."),
        call("list_directory", path="docs"),
        call("search_pdf", path="docs/handbook.pdf", query="office hours"),
        call("read_pdf", path="docs/handbook.pdf", start_page=2, max_pages=1),
        answer("Office hours: Tuesdays 14:00-15:00 (handbook.pdf, page 2)."),
        answer("Tuesdays 14:00-15:00, per page 2 of the handbook."),
    ])

    session = run_session(monkeypatch, workspace, llm, [("When are office hours? See docs/handbook.pdf.", never_asked)])

    assert llm.offered[1] == {"read_pdf", "search_pdf", "read_file", "list_directory"}
    ((name, (listed, found, page)),) = session.subagent_runs
    assert name == "pdf_analyst"
    assert "handbook.pdf" in listed
    assert "Tuesdays 14:00-15:00" in found
    assert "Tuesdays 14:00-15:00" in page
    assert session.answers == ["Tuesdays 14:00-15:00, per page 2 of the handbook."]


def test_a6_a_sub_agent_cannot_call_a_tool_outside_its_allow_list(monkeypatch, workspace):
    llm = ScriptedLLM([
        call("delegate_task", subagent_type="pdf_analyst", task="Write summary.txt."),
        call("create_file", path="summary.txt", content="not allowed"),
        answer("I can only read documents."),
        answer("The PDF analyst cannot write files."),
    ])

    session = run_session(monkeypatch, workspace, llm, [("Have the PDF analyst write summary.txt.", never_asked)])

    ((_, observations),) = session.subagent_runs
    assert observations == [
        "Error: unknown tool 'create_file'. Available tools: list_directory, read_file, read_pdf, search_pdf"
    ]
    assert not (workspace / "summary.txt").exists()
