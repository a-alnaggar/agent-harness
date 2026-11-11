"""Shared diff rendering for tools that support a dry-run preview."""

from __future__ import annotations

import difflib


def unified_diff_text(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    text = "".join(diff)
    return text if text else "(no changes)"
