"""Privacy and security filters for memory storage."""

from __future__ import annotations

import re
from agent.tools.base import ToolError

_SENSITIVE_PATTERNS = [
    # OpenAI, Anthropic, HuggingFace, GitHub keys & tokens
    re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b"),
    re.compile(r"\b(ghp_[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(github_pat_[a-zA-Z0-9_]{20,})\b"),
    re.compile(r"\b(hf_[a-zA-Z0-9]{20,})\b"),
    # JWT tokens
    re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"),
    # Private keys
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
    # Generic password / secret assignments
    re.compile(r"(?i)\b(password|passwd|secret_key|access_token|api_key)\s*[:=]\s*['\"][^'\"]{4,}['\"]"),
    re.compile(r"(?i)\b(password|passwd|secret_key|access_token|api_key)\s*[:=]\s*(\S{6,})"),
]


def validate_safe_content(text: str) -> None:
    """Validates that text does not contain API keys, passwords, or credentials.

    Raises:
        ToolError: If any sensitive pattern is detected.
    """
    if not text:
        return

    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            raise ToolError(
                "Privacy violation: Storing credentials, passwords, tokens, "
                "or API keys in long-term memory is strictly prohibited."
            )
