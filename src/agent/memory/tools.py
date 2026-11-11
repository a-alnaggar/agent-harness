"""Memory tools for saving, recalling, managing FAQs, and pruning memories."""

from __future__ import annotations

from typing import Any, Optional

from agent.observability import RunContext, context_aware
from agent.tools.base import Tool, ToolError
from agent.tools.permissions import PermissionTier
from .core_profile import CoreProfileManager, is_core_key
from .privacy import validate_safe_content
from .sqlite_store import SqliteMemoryStore

MAX_KEY_LENGTH = 100
MAX_VALUE_LENGTH = 2000


@context_aware("run", "run_with_context")
class SaveMemoryTool(Tool):
    """Allows saving general facts, preferences, and project context to persistent memory."""

    name = "save_memory"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Save a verified fact or persistent user preference to long-term memory. "
        "Supports importance (0.0-1.0), pinned (immune to deletion), and explicit reasons. "
        "Use save_reason='explicit' only when the user directly asks to remember it. "
        "Overwrites an existing key when the fact changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short identifier for the fact (e.g. 'preferred_language', 'coding_style').",
            },
            "value": {
                "type": "string",
                "description": "Content of the fact to remember.",
            },
            "importance": {
                "type": "number",
                "description": "Significance score from 0.0 to 1.0 (default: 0.5).",
            },
            "pinned": {
                "type": "boolean",
                "description": "If true, memory is permanent and immune to deletion (default: false).",
            },
            "save_reason": {
                "type": "string",
                "enum": ["explicit", "inferred", "system"],
                "description": "Origin of memory: 'explicit' (user requested), 'inferred', or 'system'.",
            },
            "source_type": {
                "type": "string",
                "enum": ["user", "gmail", "pdf", "system", "unverified"],
                "description": "Where the fact came from.",
            },
            "source_ref": {
                "type": "string",
                "description": "Optional source identifier such as a Gmail message ID.",
            },
        },
        "required": ["key", "value"],
    }

    def __init__(self, store: Any) -> None:
        self._store = store

    def run(
        self,
        *,
        key: str,
        value: str,
        importance: float = 0.5,
        pinned: bool = False,
        save_reason: str = "inferred",
        source_type: str = "user",
        source_ref: Optional[str] = None,
    ) -> str:
        return self._run(
            key=key, value=value, importance=importance, pinned=pinned,
            save_reason=save_reason, source_type=source_type, source_ref=source_ref,
        )

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(
        self, *, key: str, value: str, importance: float = 0.5, pinned: bool = False,
        save_reason: str = "inferred", source_type: str = "user",
        source_ref: Optional[str] = None, context: RunContext | None = None,
    ) -> str:
        clean_key = key.strip()
        clean_value = value.strip()
        if not clean_key:
            raise ToolError("key must not be empty")
        if "\n" in clean_key or "\r" in clean_key:
            raise ToolError("key must not contain newlines")
        if len(clean_key) > MAX_KEY_LENGTH:
            raise ToolError(f"key exceeds maximum length of {MAX_KEY_LENGTH} characters")
        if not clean_value:
            raise ToolError("value must not be empty")
        if len(clean_value) > MAX_VALUE_LENGTH:
            raise ToolError(f"value exceeds maximum length of {MAX_VALUE_LENGTH} characters")

        validate_safe_content(f"{clean_key}: {clean_value}")

        if is_core_key(clean_key):
            raise ToolError(
                f"'{clean_key}' is a core-profile key; use update_core_preference instead."
            )
        self._store.save_general(
            content=clean_value,
            key=clean_key,
            importance=importance,
            pinned=pinned,
            save_reason=save_reason,
            source_type=source_type,
            source_ref=source_ref,
            context=context,
        )
        return f"Saved to memory: {clean_key} = {clean_value}"


@context_aware("run", "run_with_context")
class RecallMemoryTool(Tool):
    """Searches memory for past facts and preferences using SQLite FTS5."""

    name = "recall_memory"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Search long-term memory for previously saved facts, user preferences, and project context. "
        "Uses full-text search (SQLite FTS5) to find relevant memories."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword or concept to search for in memory.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of memory items to return (default: 5).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: Any) -> None:
        self._store = store

    def run(self, *, query: str, top_k: int = 5) -> str:
        return self._run(query=query, top_k=top_k)

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(self, *, query: str, top_k: int = 5, context: RunContext | None = None) -> str:
        clean_query = query.strip()
        if not clean_query:
            raise ToolError("Search query cannot be empty.")

        try:
            top_k_value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ToolError("top_k must be an integer.") from exc
        if not 1 <= top_k_value <= 20:
            raise ToolError("top_k must be between 1 and 20.")
        entries = self._store.search_general(clean_query, top_k=top_k_value, context=context)
        if not entries:
            return f"No memories found matching query: '{clean_query}'."

        formatted = []
        for entry in entries:
            key_desc = f"Key: {entry.key} | " if entry.key else ""
            pin_desc = " [PINNED]" if entry.pinned else ""
            formatted.append(f"- {key_desc}Content: {entry.content}{pin_desc} (Score: {entry.retention_score})")

        return f"Found {len(entries)} relevant memory item(s):\n" + "\n".join(formatted)


@context_aware("run", "run_with_context")
class ForgetMemoryTool(Tool):
    """Deletes a specific memory entry."""

    name = "forget"
    permission_tier = PermissionTier.CONFIRM
    description = (
        "Forget a previously saved fact from long-term memory. "
        "Use this when the user asks you to forget something."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key of the fact to forget (e.g. 'preferred_language').",
            },
        },
        "required": ["key"],
    }

    def __init__(
        self,
        store: SqliteMemoryStore,
        profile_mgr: CoreProfileManager | None = None,
    ) -> None:
        self._store = store
        self._profile_mgr = profile_mgr

    def run(self, *, key: str) -> str:
        return self._run(key=key)

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(self, *, key: str, context: RunContext | None = None) -> str:
        clean_key = key.strip()
        if not clean_key:
            raise ToolError("key must not be empty")

        core_removed = False
        if self._profile_mgr is not None and is_core_key(clean_key):
            core_removed = self._profile_mgr.remove_preference(clean_key, context=context)
        general_removed = self._store.delete_general(clean_key, context=context)
        existed = core_removed or general_removed

        if existed:
            return f"Forgot from memory: {clean_key}"
        return f"Nothing to forget: '{clean_key}' was not in memory."



@context_aware("run", "run_with_context")
class SaveFaqTool(Tool):
    """Saves a verified Q&A pair linked to a course and factual source."""

    name = "save_faq"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Save a course question and answer pair with verified source attribution. "
        "Deduplicates automatically if the same question is saved."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Identifier of the course (e.g. 'sample-course').",
            },
            "question": {
                "type": "string",
                "description": "The question to store.",
            },
            "answer": {
                "type": "string",
                "description": "The verified answer.",
            },
            "source_type": {
                "type": "string",
                "enum": ["pdf", "gmail", "user", "unverified"],
                "description": "Origin of the answer.",
            },
            "source_ref": {
                "type": "string",
                "description": "Specific reference citation (e.g. 'syllabus.pdf, page 4').",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0 (default: 1.0).",
            },
        },
        "required": ["course_id", "question", "answer"],
    }

    def __init__(self, store: SqliteMemoryStore) -> None:
        self._store = store

    def run(
        self,
        *,
        course_id: str,
        question: str,
        answer: str,
        source_type: str = "unverified",
        source_ref: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        return self._run(
            course_id=course_id, question=question, answer=answer, source_type=source_type,
            source_ref=source_ref, confidence=confidence,
        )

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(
        self, *, course_id: str, question: str, answer: str, source_type: str = "unverified",
        source_ref: Optional[str] = None, confidence: float = 1.0,
        context: RunContext | None = None,
    ) -> str:
        return self._store.save_faq(
            course_id=course_id,
            question=question,
            answer=answer,
            source_type=source_type,
            source_ref=source_ref,
            confidence=confidence,
            context=context,
        )


@context_aware("run", "run_with_context")
class SearchFaqTool(Tool):
    """Searches stored FAQs for a course using FTS5."""

    name = "search_faq"
    permission_tier = PermissionTier.REGULAR
    description = "Search previously verified Q&A entries for a course before performing heavy document reads."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course identifier to search within.",
            },
            "query": {
                "type": "string",
                "description": "Search keyword or question phrasing.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of FAQs to return (default: 5).",
            },
        },
        "required": ["course_id", "query"],
    }

    def __init__(self, store: SqliteMemoryStore) -> None:
        self._store = store

    def run(self, *, course_id: str, query: str, top_k: int = 5) -> str:
        return self._run(course_id=course_id, query=query, top_k=top_k)

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(
        self, *, course_id: str, query: str, top_k: int = 5,
        context: RunContext | None = None,
    ) -> str:
        try:
            top_k_value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ToolError("top_k must be an integer.") from exc
        if not 1 <= top_k_value <= 20:
            raise ToolError("top_k must be between 1 and 20.")
        faqs = self._store.search_faq(
            course_id=course_id, query=query, top_k=top_k_value, context=context,
        )
        if not faqs:
            return f"No cached FAQs found for course '{course_id}' matching '{query}'."

        formatted = []
        for faq in faqs:
            formatted.append(
                f"Q: {faq.question}\n"
                f"A: {faq.answer}\n"
                f"(FAQ ID: {faq.id} | Source: {faq.source_type} | Ref: {faq.source_ref or 'None'} "
                f"| Occurrences: {faq.occurrence_count} | Retrievals: {faq.retrieval_count})"
            )
        return f"Found {len(faqs)} FAQ match(es) for '{course_id}':\n\n" + "\n\n".join(formatted)


@context_aware("run", "run_with_context")
class UpdateCorePreferenceTool(Tool):
    """Updates or adds an entry in the core Markdown profile (.agent_memory/core.md)."""

    name = "update_core_preference"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Update or add a high-priority session preference in .agent_memory/core.md. "
        "Max 12 bullet entries strictly enforced."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Preference name (e.g. 'preferred_reply_language', 'office_hours').",
            },
            "value": {
                "type": "string",
                "description": "Preference value.",
            },
        },
        "required": ["key", "value"],
    }

    def __init__(
        self,
        profile_mgr: CoreProfileManager,
        general_store: SqliteMemoryStore | None = None,
    ) -> None:
        self._mgr = profile_mgr
        self._general_store = general_store

    def run(self, *, key: str, value: str) -> str:
        return self._run(key=key, value=value)

    def run_with_context(self, arguments: dict[str, Any], *, context: RunContext) -> str:
        return self._run(**arguments, context=context)

    def _run(self, *, key: str, value: str, context: RunContext | None = None) -> str:
        result = self._mgr.update_preference(key, value, context=context)
        if self._general_store is not None:
            self._general_store.delete_general(
                key.strip().lower().replace(" ", "_"), context=context,
            )
        return result
