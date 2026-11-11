import json

import pytest

from agent.tools import ToolError
from agent.tools.builtin import web_search
from agent.tools.builtin.web_search import WebSearchTool


def fake_fetch(response_json):
    def _fetch(query):
        return json.dumps(response_json)
    return _fetch


def test_returns_synthesized_answer_with_sources():
    tool = WebSearchTool(fetch=fake_fetch({
        "answer": "Passau is a city in Bavaria.",
        "results": [{"url": "https://example.com", "content": "..."}],
    }))

    result = tool.run(query="Passau")

    assert result == "Passau is a city in Bavaria. (sources: https://example.com)"


def test_falls_back_to_result_snippets_when_no_answer():
    tool = WebSearchTool(fetch=fake_fetch({
        "answer": "",
        "results": [
            {"url": "https://a.com", "content": "First fact."},
            {"url": "https://b.com", "content": "Second fact."},
            {"url": "https://c.com", "content": "Third fact, should be dropped."},
        ],
    }))

    result = tool.run(query="something obscure")

    assert result.startswith("First fact. | Second fact.")
    assert "https://a.com" in result


def test_no_results_found():
    tool = WebSearchTool(fetch=fake_fetch({"answer": "", "results": []}))

    result = tool.run(query="asdkjhaskjdh")

    assert result == "No results found for 'asdkjhaskjdh'."


def test_truncates_long_answer():
    long_text = "x" * 2000
    tool = WebSearchTool(fetch=fake_fetch({"answer": long_text, "results": []}))

    result = tool.run(query="long")

    assert result.endswith("...")
    assert len(result) < 900


def test_malformed_response_raises_tool_error():
    tool = WebSearchTool(fetch=lambda query: "not json")

    with pytest.raises(ToolError, match="unparseable"):
        tool.run(query="x")


def test_fetch_failure_raises_tool_error():
    def _fail(query):
        raise ToolError("web search request failed: timed out")

    tool = WebSearchTool(fetch=_fail)

    with pytest.raises(ToolError, match="timed out"):
        tool.run(query="x")


def test_missing_api_key_raises_tool_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = WebSearchTool()

    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        tool.run(query="x")


def test_default_fetch_builds_correct_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"answer": "ok", "results": []}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setattr(web_search.urllib.request, "urlopen", fake_urlopen)

    tool = WebSearchTool()
    result = tool.run(query="passau bavaria")

    assert result == "ok"
    request = captured["request"]
    assert request.full_url == "https://api.tavily.com/search"
    assert request.get_header("Authorization") == "Bearer tvly-test-key"
    body = json.loads(request.data)
    assert body["query"] == "passau bavaria"
    assert body["include_answer"] is True


def test_registered_via_registry_returns_error_string_not_raise():
    from agent.tools import ToolRegistry

    def _fail(query):
        raise ToolError("boom")

    registry = ToolRegistry()
    registry.register(WebSearchTool(fetch=_fail))

    result = registry.call("web_search", query="x")

    assert result == "Error: boom"
