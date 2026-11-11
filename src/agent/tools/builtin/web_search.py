from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

from ..base import Tool, ToolError
from ..permissions import PermissionTier

_ENDPOINT = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 8
_MAX_RESULT_CHARS = 800
_MAX_SOURCES = 3


def _default_fetch(query: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ToolError(
            "TAVILY_API_KEY is not set — get a free key at https://tavily.com "
            "and set it as an environment variable"
        )

    body = json.dumps({
        "query": query,
        "max_results": _MAX_SOURCES,
        "include_answer": True,
    }).encode("utf-8")

    request = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        raise ToolError(f"web search request failed: {exc}") from None


class WebSearchTool(Tool):
    """Bounded to the Tavily search API: one query in, one synthesized
    answer with sources out. Requires TAVILY_API_KEY, a free key from
    https://tavily.com — never read from a config file, only the
    environment, same as the LLM API token."""

    name = "web_search"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Search the web using the Tavily search API and get a synthesized "
        "answer plus source links. Covers general web content, including "
        "time-sensitive information — not limited to encyclopedia topics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    }

    def __init__(self, fetch: Optional[Callable[[str], str]] = None):
        self._fetch = fetch or _default_fetch

    def run(self, *, query: str) -> str:
        raw = self._fetch(query)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ToolError("web search returned an unparseable response") from None

        results = data.get("results") or []
        answer = (data.get("answer") or "").strip()
        if not answer:
            snippets = [r.get("content", "").strip() for r in results if isinstance(r, dict) and r.get("content")]
            answer = " | ".join(snippets[:2])

        if not answer:
            return f"No results found for '{query}'."

        if len(answer) > _MAX_RESULT_CHARS:
            answer = answer[:_MAX_RESULT_CHARS].rstrip() + "..."

        sources = [r.get("url") for r in results[:_MAX_SOURCES] if isinstance(r, dict) and r.get("url")]
        source_text = f" (sources: {', '.join(sources)})" if sources else ""
        return answer + source_text
