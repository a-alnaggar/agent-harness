"""Bounded Gmail API tools for a dedicated OAuth test account."""

import base64
import binascii
import os
from email.message import EmailMessage
from typing import Annotated, Any, Callable

import httplib2
from email_validator import EmailNotValidError, validate_email
from google.auth.exceptions import GoogleAuthError
from googleapiclient.errors import HttpError
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import AfterValidator, BaseModel, Field, validate_call

from .gmail_auth import gmail_service


def single_address(value: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError("Recipient must be one email address without control characters.")
    try:
        return validate_email(value, check_deliverability=False, allow_smtputf8=False).normalized
    except EmailNotValidError:
        raise ValueError("Recipient must be one valid email address, without a display name.") from None


def header_text(value: str) -> str:
    if not value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Use nonblank text without control characters.")
    return value


def allowed_recipients() -> list[str]:
    """GMAIL_ALLOWED_RECIPIENTS: comma-separated addresses and/or @domains.
    Unset or empty means no restriction."""
    return [item.strip().lower() for item in os.environ.get("GMAIL_ALLOWED_RECIPIENTS", "").split(",") if item.strip()]


def require_allowed_recipient(address: str) -> None:
    """Whoever approves a send sees the recipient, but an approval clicked by
    habit is a weak last line: when an allowlist is configured, mail to anyone
    else is refused here, whatever was approved."""
    allowed = allowed_recipients()
    if not allowed:
        return
    address = address.lower()
    domain = address.rpartition("@")[2]
    if address in allowed or f"@{domain}" in allowed:
        return
    raise ToolError("Recipient is not on this deployment's allowlist (GMAIL_ALLOWED_RECIPIENTS); nothing was sent or drafted.")


Recipient = Annotated[str, Field(strict=True, max_length=254, description="One recipient email address."),
                      AfterValidator(single_address)]
Subject = Annotated[str, Field(strict=True, min_length=1, max_length=200), AfterValidator(header_text)]
Body = Annotated[str, Field(strict=True, min_length=1, max_length=20000,
                            description="Plain-text message body, up to 20,000 characters.")]
Query = Annotated[str, Field(strict=True, min_length=1, max_length=500,
                             description="Gmail search syntax; one bounded page of results."),
                  AfterValidator(header_text)]
MessageId = Annotated[str, Field(strict=True, pattern=r"^[a-zA-Z0-9_-]{1,128}$",
                                 description="Gmail message ID returned by search_emails.")]
Limit = Annotated[int, Field(strict=True, ge=1, le=20)]


class MessageRef(BaseModel):
    message_id: str
    thread_id: str


class SearchResult(BaseModel):
    messages: list[MessageRef]
    has_more: bool


class ReadResult(MessageRef):
    sender: str
    to: str
    subject: str
    date: str
    snippet: str
    body: str
    body_truncated: bool
    body_note: str


class WriteResult(MessageRef):
    status: str
    draft_id: str | None = None


def api_call(operation: Callable[[Any], Any], writing: bool = False) -> dict:
    try:
        result = operation(gmail_service()).execute(num_retries=0)
        if not isinstance(result, dict):
            raise ValueError("Invalid response type")
        return result
    except HttpError as error:
        status = error.resp.status
        if status in (401, 403):
            message = "Gmail authorization denied. Check OAuth scopes and test-account access."
        elif status == 404:
            message = "Gmail message was not found."
        elif status == 429:
            message = "Gmail rate limit reached. Try later."
        else:
            message = "Gmail request failed."
        if writing and status >= 500:
            message += " Outcome is unknown; check Gmail before retrying to avoid duplicates."
        raise ToolError(message) from None
    except (OSError, httplib2.HttpLib2Error, GoogleAuthError):
        message = "Gmail connection or authorization failed."
        if writing:
            message += " Outcome is unknown; check Gmail before retrying to avoid duplicates."
        raise ToolError(message) from None
    except ValueError:
        message = "Gmail returned an invalid response."
        if writing:
            message += " Outcome is unknown; check Gmail before retrying to avoid duplicates."
        raise ToolError(message) from None


def message_ref(data: dict) -> MessageRef:
    try:
        if not isinstance(data["id"], str) or not isinstance(data["threadId"], str):
            raise ValueError
        return MessageRef(message_id=data["id"], thread_id=data["threadId"])
    except (KeyError, TypeError, ValueError):
        raise ToolError("Gmail returned invalid message metadata.") from None


def plain_body(payload: dict) -> tuple[str, bool]:
    chunks = []
    pending = [payload]
    visited = 0
    truncated = False
    try:
        while pending and visited < 100:
            part = pending.pop()
            visited += 1
            if part.get("filename"):
                continue
            if part.get("mimeType") == "text/plain":
                encoded = part.get("body", {}).get("data", "")
                if len(encoded) > 60000:
                    encoded = encoded[:60000]
                    truncated = True
                raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
                chunks.append(raw.decode("utf-8", errors="replace"))
            pending.extend(reversed(part.get("parts", [])))
        text = "\n".join(chunks)
        return text[:10000], truncated or bool(pending) or len(text) > 10000
    except (AttributeError, TypeError, ValueError, binascii.Error):
        raise ToolError("Gmail returned invalid message content.") from None


def raw_message(to: str, subject: str, body: str) -> str:
    if not body.strip() or "\x00" in body:
        raise ToolError("Body must be nonblank plain text without NUL characters.")
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


mcp = MCPServer("gmail-test-account")


@mcp.tool()
@validate_call
def search_emails(query: Query, max_results: Limit = 10) -> SearchResult:
    """Search the authorized test mailbox; return at most 20 message IDs without bodies."""
    result = api_call(lambda service: service.users().messages().list(
        userId="me", q=query, maxResults=max_results))
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        raise ToolError("Gmail returned invalid search results.")
    return SearchResult(messages=[message_ref(item) for item in messages[:max_results]],
                        has_more=bool(result.get("nextPageToken")) or len(messages) > max_results)


@mcp.tool()
@validate_call
def read_email(message_id: MessageId) -> ReadResult:
    """Read one email, capped at 10,000 body characters. No attachments or HTML rendering."""
    result = api_call(lambda service: service.users().messages().get(
        userId="me", id=message_id, format="full"))
    reference = message_ref(result)
    try:
        payload = result["payload"]
        headers = {item["name"].lower(): item["value"][:500] for item in payload.get("headers", [])}
        text, truncated = plain_body(payload)
        return ReadResult(**reference.model_dump(), sender=headers.get("from", ""),
                          to=headers.get("to", ""), subject=headers.get("subject", ""),
                          date=headers.get("date", ""), snippet=result.get("snippet", "")[:300],
                          body=text, body_truncated=truncated,
                          body_note="Inline plain text only; HTML and attachments omitted.")
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ToolError("Gmail returned invalid message content.") from None


@mcp.tool()
@validate_call
def create_draft(to: Recipient, subject: Subject, body: Body) -> WriteResult:
    """Create one plain-text draft in the authorized test account; does not send it."""
    require_allowed_recipient(to)
    raw = raw_message(to, subject, body)
    result = api_call(lambda service: service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}), writing=True)
    if not isinstance(result.get("id"), str) or not isinstance(result.get("message"), dict):
        raise ToolError("Gmail returned invalid draft metadata. Check Gmail before retrying.")
    return WriteResult(**message_ref(result["message"]).model_dump(), status="draft_created", draft_id=result["id"])


@mcp.tool()
@validate_call
def send_email(to: Recipient, subject: Subject, body: Body) -> WriteResult:
    """Immediately SEND one plain-text email from the authorized test account to one recipient."""
    require_allowed_recipient(to)
    raw = raw_message(to, subject, body)
    result = api_call(lambda service: service.users().messages().send(
        userId="me", body={"raw": raw}), writing=True)
    try:
        reference = message_ref(result)
    except ToolError:
        raise ToolError(
            "Gmail returned invalid send metadata. Outcome is unknown; "
            "check Gmail before retrying to avoid duplicates."
        ) from None
    return WriteResult(**reference.model_dump(), status="sent")


if __name__ == "__main__":
    mcp.run(transport="stdio")
