"""LLM client for OpenAI-compatible chat APIs (developed against a vLLM-served Qwen model)."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from agent.observability import LifecycleEvent, LifecycleHook, NoOpHook, RunContext, notify_monitoring_failure, context_aware

load_dotenv()

# Used when neither the config's base_url nor LLM_BASE_URL / OPENAI_BASE_URL is set.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "qwen-agentworld-35b-a3b"


class ContextWindowExceededError(Exception):
    """Raised when the provider rejects a request because it exceeds the
    model's real context window - the ground-truth signal from the
    provider itself, as opposed to our own token-percentage estimate (which can
    still be wrong, e.g. a single large tool result jumping usage past
    100% between one proactive check and the next). ReactLoop catches
    this as the last-resort trigger for emergency compaction."""

# The system prompt teaches a Thought / Tool call / Answer structure. For a
# tool-call turn we deliberately split "Thought:" into its own field (see
# Case 1 below). When the model applies the same structure to a plain
# conversational reply instead, nothing did that splitting - the raw
# "Thought: ... Answer: ..." text would otherwise leak straight into the
# user-facing final answer. Strip it here so only the actual answer shows.
_THOUGHT_THEN_ANSWER = re.compile(r"^\s*Thought:\s*.*?\s*Answer:\s*", re.IGNORECASE | re.DOTALL)


def _strip_thought_scaffolding(content: str) -> str:
    match = _THOUGHT_THEN_ANSWER.match(content)
    return content[match.end():].strip() if match else content


@context_aware("chat", "chat_with_context")
class LLMClient:
    """Thin wrapper around the OpenAI SDK for an OpenAI-compatible endpoint.

    Usage::

        llm = LLMClient()
        response = llm.chat(messages, tools=registry.schemas())
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 60,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        context_window_tokens: int = 262144,
        hook: LifecycleHook | None = None,
        agent_id: str = "parent",
        id_provider: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens = 0
        # Prompt size of the most recent call, not a cumulative total -
        # this is what "how full is my context right now" actually means.
        # context_window_tokens defaults to the limit of the Qwen model this
        # was developed against, found empirically (the deployment's own
        # info endpoints rejected our scoped API key). Set it in the config
        # to match your model.
        self.last_prompt_tokens = 0
        self.context_window_tokens = context_window_tokens
        self.hook = hook if hook is not None else NoOpHook()
        self._monitoring_enabled = hook is not None
        self.agent_id = agent_id
        self._id_provider = id_provider or (lambda: uuid.uuid4().hex)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._diagnostic = diagnostic

        resolved_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ACCESS_TOKEN")
        )
        if not resolved_key:
            raise RuntimeError(
                "No API key found. Set LLM_API_KEY or ACCESS_TOKEN in .env"
            )

        resolved_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or _DEFAULT_BASE_URL
        )

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Send messages to the LLM and return a normalised dict.

        Returns one of two shapes so the Orchestrator can use simple
        ``response.get("tool_call")`` / ``response.get("final_answer")``
        checks.

        * Tool request  -> ``{"thought": ..., "tool_call": {"name": ..., "arguments": {...}, "id": ...}}``
        * Final answer  -> ``{"final_answer": "..."}``

        ``tool_choice`` defaults to "auto" (the model decides). ReactLoop
        passes "required" as a one-shot retry when the model narrates a tool
        call as text instead of invoking it — see react_loop.py.
        """
        return self.chat_with_context(messages, tools, tool_choice)

    def chat_with_context(
        self, messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None, tool_choice: str = "auto",
        *, context: RunContext | None = None,
    ) -> dict[str, Any]:
        parent = context
        context = self._monitor(
            lambda: (RunContext(
                parent.run_id if parent is not None else self._id_provider(),
                self._id_provider(), parent.span_id if parent is not None else None,
                parent.agent_id if parent is not None else self.agent_id,
            ), self._monotonic_clock())
        ) if self._monitoring_enabled else None
        self._emit_call_event("llm.call.start", context)
        try:
            result, completion = self._chat(messages, tools, tool_choice)
        except BaseException as error:
            self._emit_call_event("llm.call.error", context, error_type=type(error).__name__)
            raise
        self._emit_call_event("llm.call.finish", context, completion=completion)
        return result

    def _monitor(self, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except BaseException:
            notify_monitoring_failure(self._diagnostic)
            return None

    def _emit_call_event(
        self,
        event_name: str,
        context: tuple[RunContext, float] | None,
        completion: Any = None,
        error_type: str | None = None,
    ) -> None:
        if context is None:
            return

        def emit() -> None:
            attributes: dict[str, int | str] = {}
            if error_type is not None:
                attributes["error_type"] = error_type
            elif event_name == "llm.call.finish":
                usage = self._monitor(lambda: getattr(completion, "usage", None))
                for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                    value = self._monitor(lambda: getattr(usage, source, None))
                    if type(value) is int and 0 <= value <= 2**63 - 1:
                        attributes[target] = value
            self.hook.emit(LifecycleEvent(
                timestamp=self._clock(), event_name=event_name,
                run_id=context[0].run_id, span_id=context[0].span_id,
                parent_span_id=context[0].parent_span_id, agent_id=context[0].agent_id,
                component="llm", operation="call",
                outcome=None if event_name == "llm.call.start" else (
                    "failure" if error_type is not None else "success"
                ),
                duration_seconds=None if event_name == "llm.call.start" else (
                    self._monotonic_clock() - context[1]
                ),
                attributes=attributes,
            ))

        self._monitor(emit)

    def _chat(
        self, messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]], tool_choice: str,
    ) -> tuple[dict[str, Any], Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            message = str(exc)
            if "context" in message.lower() and ("exceed" in message.lower() or "maximum context length" in message.lower()):
                raise ContextWindowExceededError(message) from exc
            raise
        usage = getattr(completion, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            self.total_tokens += usage.total_tokens
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        except BaseException:
            prompt_tokens = None
        if type(prompt_tokens) is int and prompt_tokens >= 0:
            self.last_prompt_tokens = prompt_tokens
        choice = completion.choices[0]
        message = choice.message

        # --- Case 1: LLM requested a tool call ---
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            return {
                "thought": message.content or "",
                "tool_call": {
                    "name": tc.function.name,
                    "arguments": args,
                    "id": tc.id,
                },
            }, completion

        # --- Case 2: plain text answer ---
        return {"final_answer": _strip_thought_scaffolding(message.content or "")}, completion

    def context_usage_percent(self) -> float:
        """How full the context window was on the most recent call, as a
        percentage. 0 before any call has been made yet."""
        if not self.context_window_tokens:
            return 0.0
        return (self.last_prompt_tokens / self.context_window_tokens) * 100
