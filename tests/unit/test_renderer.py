"""Unit tests for CLIRenderer: assert on the actual rendered text content
(via a recording Console), not just that print() was called without
raising - so a renderer that silently drops data would be caught."""

from rich.console import Console

from agent.cli_interface.renderer import CLIRenderer
from agent.models import ReActResult, ReActStep, ToolCall, ToolObservation


def make_renderer():
    console = Console(record=True, width=120, force_terminal=False)
    return CLIRenderer(console=console), console


def rendered_text(console):
    return console.export_text()


def test_render_step_shows_thought_and_tool_observation():
    renderer, console = make_renderer()
    step = ReActStep(
        step_number=2,
        thought="Checking the inbox for unread mail.",
        tool_calls=[ToolCall(id="call_1", name="search_emails", arguments={"query": "is:unread"})],
        observations=[ToolObservation(tool_call_id="call_1", name="search_emails",
                                       content='{"messages": []}', is_error=False)],
        duration_seconds=1.5,
    )

    renderer.render_step(step)
    text = rendered_text(console)

    assert "Checking the inbox for unread mail." in text
    assert "search_emails" in text
    assert "is:unread" in text
    assert '{"messages": []}' in text
    assert "Iteration 2" in text
    assert "1.50s" in text


def test_render_step_marks_a_failed_tool_call():
    renderer, console = make_renderer()
    step = ReActStep(
        step_number=1,
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "missing.txt"})],
        observations=[ToolObservation(tool_call_id="call_1", name="read_file",
                                       content="Error: file not found", is_error=True)],
    )

    renderer.render_step(step)

    assert "Error: file not found" in rendered_text(console)


def test_render_step_labels_subagent_activity():
    renderer, console = make_renderer()
    step = ReActStep(
        step_number=1,
        agent_name="memory_manager",
        tool_calls=[ToolCall(id="call_1", name="update_core_preference", arguments={})],
        observations=[ToolObservation(
            tool_call_id="call_1",
            name="update_core_preference",
            content="Updated user_name",
        )],
    )

    renderer.render_step(step)

    text = rendered_text(console)
    assert "Sub-agent: memory_manager" in text
    assert "update_core_preference" in text


def test_render_step_with_no_thought_prints_no_thought_panel():
    renderer, console = make_renderer()
    step = ReActStep(step_number=1, thought=None, tool_calls=[], observations=[])

    renderer.render_step(step)

    assert "Thought" not in rendered_text(console)


def test_render_final_answer_success_shows_final_answer_title_and_stats():
    renderer, console = make_renderer()
    result = ReActResult(final_answer="All done.", total_iterations=3,
                          total_duration_seconds=2.5, is_success=True)

    renderer.render_final_answer(result, token_count=1234)

    text = rendered_text(console)
    assert "Final Answer" in text
    assert "All done." in text
    assert "Iterations: 3" in text
    assert "2.50s" in text
    assert "1234" in text


def test_render_final_answer_failure_shows_task_incomplete_title():
    renderer, console = make_renderer()
    result = ReActResult(final_answer="Error: something broke", total_iterations=1,
                          total_duration_seconds=0.1, is_success=False)

    renderer.render_final_answer(result, token_count=0)

    text = rendered_text(console)
    assert "Task Incomplete" in text
    assert "Final Answer" not in text


def test_render_final_answer_shows_context_percent():
    renderer, console = make_renderer()
    result = ReActResult(final_answer="All done.", total_iterations=1, total_duration_seconds=0.1)

    renderer.render_final_answer(result, token_count=0, context_percent=42.7)

    assert "Context: 43%" in rendered_text(console)


def test_render_compaction_notice_shows_percent_and_message_counts():
    renderer, console = make_renderer()

    renderer.render_compaction_notice(percent_before=85.3, messages_before=40, messages_after=6)

    text = rendered_text(console)
    assert "85%" in text
    assert "40" in text
    assert "6" in text
    assert "Context Compacted" in text


def test_render_permission_request_shows_tool_name_description_and_arguments():
    renderer, console = make_renderer()

    renderer.render_permission_request(
        "send_email",
        {"to": "someone@example.com", "subject": "hi"},
        "Send an email immediately - cannot be undone.",
    )

    text = rendered_text(console)
    assert "send_email" in text
    assert "cannot be undone" in text
    assert "someone@example.com" in text
    assert "Permission Requested" in text


def test_render_permission_request_handles_no_arguments():
    renderer, console = make_renderer()

    renderer.render_permission_request("noop", {}, "does nothing")

    text = rendered_text(console)
    assert "noop" in text
    assert "no arguments" in text


def test_render_error_shows_message_and_title():
    renderer, console = make_renderer()

    renderer.render_error("could not connect", title="Connection Error")

    text = rendered_text(console)
    assert "could not connect" in text
    assert "Connection Error" in text


def test_print_help_lists_every_slash_command():
    renderer, console = make_renderer()

    renderer.print_help()

    text = rendered_text(console)
    for cmd in ["/help", "/tools", "/tokens", "/reset", "/config", "/exit"]:
        assert cmd in text


def test_print_tools_handles_openai_style_schema_with_required_args():
    renderer, console = make_renderer()
    schemas = [{
        "function": {
            "name": "read_file",
            "description": "Read a file's contents.",
            "parameters": {"required": ["path"]},
        }
    }]

    renderer.print_tools(schemas)

    text = rendered_text(console)
    assert "read_file" in text
    assert "Read a file's contents." in text
    assert "path" in text


def test_print_tools_handles_direct_dict_schema_with_no_required_args():
    renderer, console = make_renderer()
    schemas = [{"name": "ping", "description": "No-op.", "parameters": {}}]

    renderer.print_tools(schemas)

    text = rendered_text(console)
    assert "ping" in text
    assert "None" in text


def test_print_tokens_shows_message_count_and_estimated_tokens():
    renderer, console = make_renderer()

    renderer.print_tokens({"total_messages": 7, "estimated_tokens": 512, "role_distribution": {"user": 3}})

    text = rendered_text(console)
    assert "7" in text
    assert "512" in text


def test_print_config_shows_all_fields():
    renderer, console = make_renderer()

    renderer.print_config(model="qwen-agentworld-35b-a3b", temperature=0.2,
                           max_iterations=10, timeout=60, workspace="/workspace")

    text = rendered_text(console)
    assert "qwen-agentworld-35b-a3b" in text
    assert "0.2" in text
    assert "/workspace" in text


# --- untrusted text must be shown, not obeyed ---------------------------------
# Arguments and tool results carry text from emails, web pages and other
# people. Each of these used to crash a panel (an unmatched closing tag
# raises MarkupError) or restyle/hide/rewrite it (markup tags, escape codes).

from agent.cli_interface.renderer import neutralize  # noqa: E402


def test_permission_panel_shows_markup_lookalikes_literally_instead_of_crashing():
    renderer, console = make_renderer()

    renderer.render_permission_request(
        "send_email", {"body": "please reply [/INST] thanks", "note": "a [conceal]hidden?[/conceal] b"}, "sends mail"
    )

    text = rendered_text(console)
    assert "[/INST]" in text
    assert "[conceal]hidden?[/conceal]" in text  # visible - not consumed as a style that could hide it


def test_permission_panel_shows_terminal_escape_sequences_as_visible_text():
    renderer, console = make_renderer()

    renderer.render_permission_request("send_email", {"body": "x\x1b[2Jcleared"}, "d")

    text = rendered_text(console)
    assert "\x1b" not in text
    assert "\\u001b[2Jcleared" in text  # the approver can see something odd is in there


def test_permission_panel_marks_direction_overrides_and_invisible_tag_characters():
    renderer, console = make_renderer()

    renderer.render_permission_request("t", {"a": "pay \u202eevil", "b": "x" + chr(0xE0041)}, "d")

    text = rendered_text(console)
    assert "\u202e" not in text and chr(0xE0041) not in text
    assert "\\u202e" in text and "\\U000e0041" in text


def test_permission_panel_also_neutralizes_the_tool_name_and_description():
    renderer, console = make_renderer()

    renderer.render_permission_request("evil[/x]\x1b[0m", {}, "desc [/y] \x07")

    text = rendered_text(console)
    assert "evil[/x]" in text and "desc [/y]" in text
    assert "\x1b" not in text and "\x07" not in text


def test_step_panel_survives_markup_lookalikes_in_tool_output_arguments_and_name():
    renderer, console = make_renderer()
    step = ReActStep(
        step_number=1,
        tool_calls=[ToolCall(id="c", name="odd[/name]", arguments={"q": "[/INST]"})],
        observations=[ToolObservation(tool_call_id="c", name="odd[/name]", content="mail body [/INST] end", is_error=False)],
    )

    renderer.render_step(step)

    text = rendered_text(console)
    assert "mail body [/INST] end" in text
    assert "odd[/name]" in text


def test_step_panel_neutralizes_escape_sequences_in_tool_output():
    renderer, console = make_renderer()
    step = ReActStep(
        step_number=1,
        tool_calls=[ToolCall(id="c", name="read_email", arguments={})],
        observations=[ToolObservation(tool_call_id="c", name="read_email", content="hi\x1b[2Jbye", is_error=False)],
    )

    renderer.render_step(step)

    assert "\x1b" not in rendered_text(console)


def test_thought_and_final_answer_neutralize_escape_sequences():
    renderer, console = make_renderer()

    renderer.render_step(ReActStep(step_number=1, thought="thinking\x1b[2J done"))
    renderer.render_final_answer(
        ReActResult(final_answer="answer\x1b[2J done", total_iterations=1, total_duration_seconds=0.1), token_count=0
    )

    assert "\x1b" not in rendered_text(console)


def test_error_panel_survives_a_message_containing_markup_lookalikes():
    renderer, console = make_renderer()

    renderer.render_error("MarkupError: closing tag '[/INST]' doesn't match", title="Unexpected Error")

    assert "[/INST]" in rendered_text(console)


def test_tools_table_survives_markup_lookalikes_in_third_party_descriptions():
    renderer, console = make_renderer()

    renderer.print_tools([{"name": "api[/x]", "description": "does things [/INST] [bold", "parameters": {"required": []}}])

    text = rendered_text(console)
    assert "api[/x]" in text and "[/INST]" in text


def test_neutralize_leaves_ordinary_text_newlines_tabs_and_unicode_alone():
    ordinary = "Grüße, 日本語 and emoji \U0001F600\nsecond line\tindented"

    assert neutralize(ordinary) == ordinary


def test_the_permission_panel_shows_the_whole_email_body_before_a_send_is_approved():
    from rich.console import Console
    from agent.cli_interface.renderer import CLIRenderer

    body = "start " + "x" * 3000 + " END-MARKER"
    console = Console(width=200, record=True)
    CLIRenderer(console).render_permission_request(
        "send_email", {"to": "a@example.com", "subject": "s", "body": body}, "Immediately SEND one email")

    shown = "".join(console.export_text(styles=False).split())
    assert "END-MARKER" in shown and "a@example.com" in shown
