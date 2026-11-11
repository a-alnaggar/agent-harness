"""Blocks destructive commands before they ever reach the SSH connection.

The lab spec's "tools must never be able to delete anything" constraint
predates this tool and was written with local filesystem tools in mind, but
a remote shell is a different risk class — it can run anything. This filter
keeps sandbox_exec inside that same spirit: pattern-match known-destructive
commands and refuse them client-side, rather than relying on the remote box
being disposable.

This is a blocklist, not a sandbox boundary — it catches the obvious cases
(rm, dd, mkfs, force-pushes, fork bombs) but a determined LLM could still
phrase around it (e.g. piping through base64). It's a guardrail against
routine mistakes, not a security boundary; the SSH target should still be a
throwaway machine.
"""

from __future__ import annotations

import re

_DESTRUCTIVE_PATTERNS = [
    r"\brm\b",
    r"\bshred\b",
    r"\bunlink\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b",
    r"\bwipefs\b",
    r"\btruncate\b",
    r">\s*/dev/sd",
    r"\bgit\s+push\b.*(--force|-f)\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b.*-[a-z]*f",
    r":\(\)\s*\{.*:\|:.*\}",  # fork bomb
    r"\bchmod\s+-R\s+000\b",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _DESTRUCTIVE_PATTERNS]


def find_destructive_match(command: str) -> str | None:
    """Returns the matched pattern's source text if `command` looks
    destructive, else None."""
    for compiled in _COMPILED:
        if compiled.search(command):
            return compiled.pattern
    return None
