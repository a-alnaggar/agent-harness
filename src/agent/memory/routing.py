"""Deterministic routing for requests that must mutate persistent memory."""

from __future__ import annotations

import re
from typing import Any


_MEMORY_MUTATION_PATTERNS = (
    # "remember to ...", "don't forget to ..." are reminders about a task,
    # not memory requests: routing them forced a memory_manager delegation.
    re.compile(
        r"(?<!don't )(?<!do not )(?<!never )\b(?:remember|forget)\b(?!\s+to\b)"
        r"|\b(?:save this|store this|remove from memory)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:my name is|call me)\b", re.IGNORECASE),
    re.compile(r"(?:تذكر|تذكّر|احفظ|انس|انسى|امسح من الذاكرة|اسمي)"),
)


def route_memory_mutation(user_input: str) -> dict[str, Any] | None:
    if not any(pattern.search(user_input) for pattern in _MEMORY_MUTATION_PATTERNS):
        return None
    return {
        "name": "delegate_task",
        "arguments": {
            "subagent_type": "memory_manager",
            "task": (
                "Handle this persistent-memory request. You MUST use the appropriate memory tool "
                "and may report success only after a successful tool observation. Core profile keys "
                "such as user_name must use update_core_preference, not save_memory.\n\n"
                f"User request: {user_input}"
            ),
        },
    }
