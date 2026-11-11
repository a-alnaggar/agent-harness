import asyncio
import base64
from email import policy
from email.parser import BytesParser
from unittest.mock import MagicMock

import httplib2
import pytest
from googleapiclient.errors import HttpError
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from agent.mcp_servers import gmail


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock()
    monkeypatch.setattr(gmail, "gmail_service", lambda: fake)
    return fake


def test_schema() -> None:
    tools = {tool.name: tool for tool in asyncio.run(gmail.mcp.list_tools())}
    assert set(tools) == {"search_emails", "read_email", "create_draft", "send_email"}
    limit = tools["search_emails"].input_schema["properties"]["max_results"]
    assert (limit["minimum"], limit["maximum"]) == (1, 20)
    assert tools["send_email"].input_schema["required"] == ["to", "subject", "body"]
    assert "SEND" in tools["send_email"].description
    assert all(tool.output_schema for tool in tools.values())


def test_search(service: MagicMock) -> None:
    request = service.users().messages().list
    request.return_value.execute.return_value = {
        "messages": [{"id": "abc", "threadId": "def"}], "nextPageToken": "opaque", "extra": "omit"}
    assert gmail.search_emails("is:unread", 1).model_dump() == {
        "messages": [{"message_id": "abc", "thread_id": "def"}], "has_more": True}
    request.assert_called_once_with(userId="me", q="is:unread", maxResults=1)
    request.return_value.execute.assert_called_once_with(num_retries=0)


def test_empty_search(service: MagicMock) -> None:
    service.users().messages().list.return_value.execute.return_value = {}
    assert gmail.search_emails("in:inbox").messages == []


@pytest.mark.parametrize("query,limit", [("", 1), (" ", 1), ("x" * 501, 1), ("x", 0),
                                        ("x", 21), ("x", True), ("x", "2")])
def test_invalid_search(query: object, limit: object, service: MagicMock) -> None:
    with pytest.raises(ValidationError):
        gmail.search_emails(query, limit)
    service.assert_not_called()
    assert not service.users.called


@pytest.mark.parametrize("message_id", ["", "../foo", "a?format=raw", "x" * 129, None])
def test_invalid_id(message_id: object, service: MagicMock) -> None:
    with pytest.raises(ValidationError):
        gmail.read_email(message_id)
    assert not service.users.called


@pytest.mark.parametrize("tool", [gmail.create_draft, gmail.send_email])
@pytest.mark.parametrize("to,subject,body", [
    ("bad", "subject", "body"), ("a@example.com,b@example.com", "subject", "body"),
    ("Name <a@example.com>", "subject", "body"), ("a@example.com\r\nBcc: b@example.com", "s", "b"),
    ("a@example.com", "s\r\nBcc: b@example.com", "body"),
    ("a@example.com", " ", "body"), ("a@example.com", "s" * 201, "body"),
    ("a@example.com", "s", "b" * 20001), ("a@example.com", "s", " "),
    ("a@example.com", "s", "x\x00y"),
])
def test_invalid_outgoing(tool: object, to: str, subject: str, body: str, service: MagicMock) -> None:
    with pytest.raises((ValidationError, ToolError)):
        tool(to, subject, body)
    assert not service.users.called


@pytest.mark.parametrize("draft", [False, True])
def test_outgoing_mime(draft: bool, service: MagicMock) -> None:
    messages = service.users().messages()
    drafts = service.users().drafts()
    reference = {"id": "abc", "threadId": "def"}
    operation = drafts.create if draft else messages.send
    operation.return_value.execute.return_value = {"id": "draft1", "message": reference} if draft else reference
    tool = gmail.create_draft if draft else gmail.send_email
    result = tool("test@example.com", "Héllo", "Hello\n世界")
    arguments = operation.call_args.kwargs
    assert arguments["userId"] == "me"
    raw = arguments["body"]["message"]["raw"] if draft else arguments["body"]["raw"]
    message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw))
    assert message["To"] == "test@example.com"
    assert message["Subject"] == "Héllo"
    assert message.get_content().strip() == "Hello\n世界"
    assert message["Bcc"] is None
    assert result.status == ("draft_created" if draft else "sent")
    operation.return_value.execute.assert_called_once_with(num_retries=0)
    if draft:
        messages.send.assert_not_called()


def test_read_multipart(service: MagicMock) -> None:
    encoded = base64.urlsafe_b64encode(b"hello").decode().rstrip("=")
    service.users().messages().get.return_value.execute.return_value = {
        "id": "abc", "threadId": "def", "snippet": "preview", "payload": {
            "mimeType": "multipart/mixed", "headers": [{"name": "Subject", "value": "Test"}],
            "parts": [{"mimeType": "multipart/alternative", "parts": [
                {"mimeType": "text/html", "body": {"data": "ignored"}},
                {"mimeType": "text/plain", "body": {"data": encoded}}]},
                {"mimeType": "text/plain", "filename": "secret.txt", "body": {"data": "ignored"}}]}}
    result = gmail.read_email("abc")
    assert result.body == "hello"
    assert result.subject == "Test"
    assert not result.body_truncated
    service.users().messages().get.assert_called_once_with(userId="me", id="abc", format="full")


def test_body_limits_and_html() -> None:
    encoded = base64.urlsafe_b64encode(b"x" * 50000).decode()
    text, truncated = gmail.plain_body({"mimeType": "text/plain", "body": {"data": encoded}})
    assert len(text) == 10000 and truncated
    assert gmail.plain_body({"mimeType": "text/html"}) == ("", False)


@pytest.mark.parametrize("payload", [None, {"mimeType": "text/plain", "body": None},
                                     {"mimeType": "text/plain", "body": {"data": "!!!"}}])
def test_malformed_body(payload: object) -> None:
    with pytest.raises(ToolError, match="invalid message content"):
        gmail.plain_body(payload)


@pytest.mark.parametrize("status,match", [(401, "authorization"), (403, "authorization"),
                                         (404, "not found"), (429, "rate limit"), (500, "failed")])
def test_api_errors(status: int, match: str, service: MagicMock) -> None:
    service.users().messages().list.return_value.execute.side_effect = HttpError(
        httplib2.Response({"status": status}), b"private contents")
    with pytest.raises(ToolError, match=match) as caught:
        gmail.search_emails("x")
    assert "private contents" not in str(caught.value)


def test_send_uncertain_failure(service: MagicMock) -> None:
    service.users().messages().send.return_value.execute.side_effect = TimeoutError("private")
    with pytest.raises(ToolError, match="Outcome is unknown"):
        gmail.send_email("test@example.com", "subject", "body")
    service.users().messages().send.return_value.execute.assert_called_once_with(num_retries=0)


@pytest.mark.parametrize("response", [None, {"messages": None}, {"messages": [{}]}, {"messages": [None]}])
def test_malformed_search(response: object, service: MagicMock) -> None:
    service.users().messages().list.return_value.execute.return_value = response
    with pytest.raises(ToolError):
        gmail.search_emails("x")


def test_mcp_validation(service: MagicMock) -> None:
    with pytest.raises(ToolError):
        asyncio.run(gmail.mcp.call_tool("send_email", {"to": "bad", "subject": "x", "body": "y"}))
    assert not service.users.called


def test_invalid_json_write(service: MagicMock) -> None:
    service.users().messages().send.return_value.execute.side_effect = ValueError("private")
    with pytest.raises(ToolError, match="Outcome is unknown"):
        gmail.send_email("test@example.com", "subject", "body")


@pytest.mark.parametrize("response", [{}, {"id": "message-only"}, {"threadId": "thread-only"}])
def test_malformed_send_metadata_warns_about_ambiguous_outcome(
    response: dict, service: MagicMock
) -> None:
    service.users().messages().send.return_value.execute.return_value = response

    with pytest.raises(ToolError, match="Outcome is unknown; check Gmail before retrying"):
        gmail.send_email("test@example.com", "subject", "body")

    service.users().messages().send.return_value.execute.assert_called_once_with(num_retries=0)


def test_stdio_protocol_without_credentials(tmp_path: object) -> None:
    import os
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise() -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        environment["GMAIL_CREDENTIALS_PATH"] = str(tmp_path / "missing-client.json")
        environment["GMAIL_TOKEN_PATH"] = str(tmp_path / "missing-token.json")
        parameters = StdioServerParameters(command=sys.executable,
                                           args=["-m", "agent.mcp_servers.gmail"], env=environment)
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                assert len((await session.list_tools()).tools) == 4
                result = await session.call_tool("search_emails", {"query": "is:unread"})
                assert result.is_error
                assert "authorization unavailable" in result.content[0].text
                assert len((await session.list_tools()).tools) == 4

    asyncio.run(exercise())


# --- recipient allowlist -----------------------------------------------------

@pytest.mark.parametrize("tool", ["send_email", "create_draft"])
@pytest.mark.parametrize("allowed, recipient, ok", [
    ("", "anyone@example.com", True),  # unset: no restriction
    ("me@example.com", "me@example.com", True),
    ("me@example.com", "ME@Example.com", True),
    ("me@example.com", "other@example.com", False),
    ("@example.edu", "x@example.edu", True),
    ("@example.edu", "x@evil.example.edu.attacker.com", False),
    ("@example.edu", "x@sub.example.edu", False),  # a domain entry means exactly that domain
    ("a@x.org, @example.edu", "b@example.edu", True),
])
def test_recipient_allowlist(monkeypatch: pytest.MonkeyPatch, service: MagicMock, tool: str,
                             allowed: str, recipient: str, ok: bool) -> None:
    monkeypatch.setenv("GMAIL_ALLOWED_RECIPIENTS", allowed)
    service.users().messages().send.return_value.execute.return_value = {"id": "m", "threadId": "t"}
    service.users().drafts().create.return_value.execute.return_value = {
        "id": "d", "message": {"id": "m", "threadId": "t"}}

    if ok:
        getattr(gmail, tool)(recipient, "subject", "body")
    else:
        with pytest.raises(ToolError, match="allowlist"):
            getattr(gmail, tool)(recipient, "subject", "body")
        service.users().messages().send.assert_not_called()
        service.users().drafts().create.assert_not_called()
