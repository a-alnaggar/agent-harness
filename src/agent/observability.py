from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, TypeAlias, TypeVar


SafeAttribute: TypeAlias = str | int | float | bool

MAX_ATTRIBUTES = 16
MAX_ATTRIBUTE_KEY_LENGTH = 64
MAX_ATTRIBUTE_VALUE_LENGTH = 128

_ATTRIBUTE_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", re.ASCII)


def _safe_identifier(value: Any) -> bool:
    return type(value) is str and 1 <= len(value) <= MAX_ATTRIBUTE_VALUE_LENGTH and bool(
        _IDENTIFIER_PATTERN.fullmatch(value)
    )


def context_aware(legacy: str, contextual: str) -> Callable[[type[Any]], type[Any]]:
    def declare(cls: type[Any]) -> type[Any]:
        capabilities = dict(cls.__dict__.get("_lifecycle_capabilities", {}))
        capabilities[legacy] = (getattr(cls, legacy), contextual, getattr(cls, contextual))
        cls._lifecycle_capabilities = MappingProxyType(capabilities)
        return cls
    return declare


def contextual_method(component: Any, legacy: str) -> Callable[..., Any] | None:
    for cls in type(component).__mro__:
        declaration = cls.__dict__.get("_lifecycle_capabilities", {}).get(legacy)
        if declaration is not None:
            original, name, contextual = declaration
            current = getattr(component, legacy)
            method = getattr(component, name)
            if getattr(current, "__func__", None) is original and getattr(method, "__func__", None) is contextual:
                return method
            return None
    return None
# Extend each event's policy deliberately; future token counts belong on LLM events.
_ATTRIBUTE_POLICIES = MappingProxyType({
    "agent.run.start": MappingProxyType({"max_steps": int}),
    "agent.run.finish": MappingProxyType({"max_steps": int}),
    "agent.run.error": MappingProxyType({"error_type": str}),
    "llm.call.start": MappingProxyType({}),
    "llm.call.finish": MappingProxyType({"input_tokens": int, "output_tokens": int}),
    "llm.call.error": MappingProxyType({"error_type": str}),
    "tool.call.start": MappingProxyType({"tool_name": str, "origin": str}),
    "tool.call.finish": MappingProxyType({"tool_name": str, "origin": str}),
    "tool.call.error": MappingProxyType({"tool_name": str, "origin": str, "error_type": str}),
    "memory.operation.start": MappingProxyType({}),
    "memory.operation.finish": MappingProxyType({}),
    "memory.operation.error": MappingProxyType({"error_type": str}),
    "permission.request.start": MappingProxyType({"tool_name": str, "origin": str, "tier": str}),
    "permission.request.finish": MappingProxyType({"tool_name": str, "origin": str, "tier": str, "decision": str}),
    "permission.request.error": MappingProxyType({"tool_name": str, "origin": str, "tier": str, "error_type": str}),
})


def notify_monitoring_failure(diagnostic: Callable[[str], None] | None) -> None:
    if diagnostic is not None:
        try:
            diagnostic("Monitoring failure; agent execution continues.")
        except BaseException:
            pass


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    span_id: str
    parent_span_id: str | None
    agent_id: str

    def __post_init__(self) -> None:
        for value in (self.run_id, self.span_id, self.agent_id):
            if not _safe_identifier(value):
                raise ValueError("context identifiers must be bounded non-empty strings")
        if self.parent_span_id is not None and (
            not _safe_identifier(self.parent_span_id)
        ):
            raise ValueError("parent_span_id must be a bounded non-empty string")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    timestamp: datetime
    event_name: str
    run_id: str
    span_id: str
    agent_id: str
    component: str
    operation: str
    outcome: str | None = None
    duration_seconds: float | None = None
    attributes: Mapping[str, SafeAttribute] = field(default_factory=dict)
    parent_span_id: str | None = None

    def __post_init__(self) -> None:
        RunContext(self.run_id, self.span_id, self.parent_span_id, self.agent_id)
        if not _safe_identifier(self.event_name):
            raise ValueError("event_name must be a bounded safe identifier")
        if self.event_name.startswith("memory.operation.") and self.operation not in {"read", "save", "forget"}:
            raise ValueError("memory operation must be read, save or forget")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        for name, value in (
            ("event_name", self.event_name),
            ("run_id", self.run_id),
            ("span_id", self.span_id),
            ("agent_id", self.agent_id),
            ("component", self.component),
            ("operation", self.operation),
        ):
            if not _safe_identifier(value):
                raise ValueError(f"{name} must contain 1-{MAX_ATTRIBUTE_VALUE_LENGTH} characters")
        if self.event_name.endswith(".start"):
            if self.outcome is not None:
                raise ValueError("start outcome must be None")
        elif type(self.outcome) is not str or self.outcome not in {"success", "failure"}:
            raise ValueError("terminal outcome must be success or failure")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be a finite non-negative number")

        attributes = dict(self.attributes)
        if self.event_name not in _ATTRIBUTE_POLICIES:
            raise ValueError("event_name has no safe attribute policy")
        if len(attributes) > MAX_ATTRIBUTES:
            raise ValueError(f"attributes must contain at most {MAX_ATTRIBUTES} entries")
        for key, value in attributes.items():
            self._validate_attribute(key, value)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))

    def _validate_attribute(self, key: str, value: SafeAttribute) -> None:
        if (
            not isinstance(key, str)
            or len(key) > MAX_ATTRIBUTE_KEY_LENGTH
            or not _ATTRIBUTE_KEY_PATTERN.fullmatch(key)
        ):
            raise ValueError("attribute keys must be short stable identifiers")
        expected_type = _ATTRIBUTE_POLICIES[self.event_name].get(key)
        if expected_type is None:
            raise ValueError("attribute key is not allowed for this event")
        if type(value) is not expected_type:
            raise TypeError("attribute value has the wrong type for this event")
        if expected_type is int and not 0 <= value <= 2**63 - 1:
            raise ValueError("numeric attributes must be bounded non-negative integers")
        if expected_type is str and (
            len(value) > MAX_ATTRIBUTE_VALUE_LENGTH or not _ATTRIBUTE_KEY_PATTERN.fullmatch(value)
        ):
            raise ValueError("error_type must be a bounded class identifier")
        if key == "origin" and value not in {"builtin", "mcp", "subagent"}:
            raise ValueError("origin must be builtin, mcp, or subagent")
        if key == "tier" and value not in {"regular", "confirm", "strict", "deny", "unknown"}:
            raise ValueError("tier must be regular, confirm, strict, deny, or unknown")
        if key == "decision" and value not in {"once", "always", "cached", "deny", "policy_deny"}:
            raise ValueError("decision must be once, always, cached, deny, or policy_deny")


class LifecycleHook(Protocol):
    def emit(self, event: LifecycleEvent) -> None: ...


class NoOpHook:
    def emit(self, event: LifecycleEvent) -> None:
        return None


T = TypeVar("T")


class LifecycleObserver:
    def __init__(
        self, hook: LifecycleHook | None = None, agent_id: str = "parent",
        id_provider: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        self.hook = hook
        self.agent_id = agent_id
        self.ids = id_provider or (lambda: uuid.uuid4().hex)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic_clock or time.monotonic
        self.diagnostic = diagnostic

    def protect(self, action: Callable[[], T]) -> T | None:
        try:
            return action()
        except BaseException:
            notify_monitoring_failure(self.diagnostic)
            return None

    def start(self, parent: RunContext | None) -> tuple[RunContext, float] | None:
        return self.protect(lambda: (RunContext(
            parent.run_id if parent is not None else self.ids(), self.ids(),
            parent.span_id if parent is not None else None,
            parent.agent_id if parent is not None else self.agent_id,
        ), self.monotonic())) if self.hook is not None else None

    def emit(
        self, name: str, span: tuple[RunContext, float] | None,
        component: str, operation: str, attributes: Mapping[str, SafeAttribute],
        error_type: str | None = None,
    ) -> None:
        if span is None:
            return

        def deliver() -> None:
            context, started = span
            terminal = not name.endswith(".start")
            self.hook.emit(LifecycleEvent(
                timestamp=self.clock(), event_name=name, run_id=context.run_id,
                span_id=context.span_id, parent_span_id=context.parent_span_id,
                agent_id=context.agent_id, component=component, operation=operation,
                outcome=("failure" if error_type is not None else "success") if terminal else None,
                duration_seconds=self.monotonic() - started if terminal else None,
                attributes={**attributes, **({"error_type": error_type} if error_type else {})},
            ))

        self.protect(deliver)

    def memory(self, operation: str, context: RunContext | None, action: Callable[[], T]) -> T:
        span = self.start(context)
        self.emit("memory.operation.start", span, "memory", operation, {})
        try:
            result = action()
        except BaseException as error:
            self.emit("memory.operation.error", span, "memory", operation, {}, type(error).__name__)
            raise
        self.emit("memory.operation.finish", span, "memory", operation, {})
        return result


def safe_tool_name(name: Any) -> str:
    return name if type(name) is str and len(name) <= MAX_ATTRIBUTE_VALUE_LENGTH and (
        _ATTRIBUTE_KEY_PATTERN.fullmatch(name)
    ) else "unknown_tool"


class CompositeHook:
    def __init__(
        self, hooks: Iterable[LifecycleHook], diagnostic: Callable[[str], None] | None = None
    ) -> None:
        self._hooks = tuple(hooks)
        self._diagnostic = diagnostic

    def emit(self, event: LifecycleEvent) -> None:
        for hook in self._hooks:
            try:
                hook.emit(event)
            except BaseException:
                notify_monitoring_failure(self._diagnostic)
