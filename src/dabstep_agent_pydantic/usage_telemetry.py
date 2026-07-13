from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class UsageBudgetExceeded(RuntimeError):
    """Raised when recording another model call would exceed the ledger budget."""


@dataclass(frozen=True)
class CallUsage:
    stage: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retries: int = 0
    model_fingerprint: str | None = None


class UsageLedger:
    def __init__(self, *, max_calls: int | None = None) -> None:
        if max_calls is not None and max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        self.max_calls = max_calls
        self._calls: list[CallUsage] = []

    def can_call(self, stage: str | None = None) -> bool:
        del stage
        return self.max_calls is None or len(self._calls) < self.max_calls

    def ensure_can_call(self, stage: str) -> None:
        if not self.can_call(stage):
            raise UsageBudgetExceeded(
                f"global model-call budget of {self.max_calls} exhausted before stage {stage!r}"
            )

    def record(self, usage: CallUsage) -> None:
        self.ensure_can_call(usage.stage)
        self._calls.append(usage)

    def summary(self) -> dict[str, dict[str, Any]]:
        by_stage: dict[str, dict[str, Any]] = {}
        for usage in self._calls:
            stage = by_stage.setdefault(
                usage.stage,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "retries": 0,
                    "model_fingerprints": [],
                },
            )
            stage["calls"] += 1
            stage["input_tokens"] += usage.input_tokens
            stage["output_tokens"] += usage.output_tokens
            stage["latency_ms"] += usage.latency_ms
            stage["retries"] += usage.retries
            fingerprints = stage["model_fingerprints"]
            if usage.model_fingerprint and usage.model_fingerprint not in fingerprints:
                fingerprints.append(usage.model_fingerprint)
        return by_stage


def call_usage_from_result(
    result: Any,
    *,
    stage: str,
    latency_ms: float,
    retries: int = 0,
) -> CallUsage:
    """Adapt current and older Pydantic AI result usage shapes."""
    raw_usage = _safe_getattr(result, "usage")
    if callable(raw_usage):
        try:
            raw_usage = raw_usage()
        except Exception:  # pragma: no cover - defensive compatibility path
            raw_usage = None
    return CallUsage(
        stage=stage,
        input_tokens=_token_count(raw_usage, "input_tokens", "prompt_tokens", "request_tokens"),
        output_tokens=_token_count(raw_usage, "output_tokens", "completion_tokens", "response_tokens"),
        latency_ms=max(0, round(latency_ms)),
        retries=max(0, retries),
        model_fingerprint=_model_fingerprint(result),
    )


def _token_count(usage: Any, *names: str) -> int:
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else _safe_getattr(usage, name)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return 0


def _model_fingerprint(result: Any) -> str | None:
    response = _safe_getattr(result, "response")
    details = _safe_getattr(response, "provider_details")
    if isinstance(details, Mapping):
        for key in ("system_fingerprint", "model_fingerprint", "fingerprint"):
            if value := details.get(key):
                return str(value)
    model_name = _safe_getattr(response, "model_name") or _safe_getattr(result, "model_name")
    provider_name = _safe_getattr(response, "provider_name")
    if model_name and provider_name:
        return f"{provider_name}:{model_name}"
    return str(model_name) if model_name else None


def _safe_getattr(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, name, None)
    except Exception:  # pragma: no cover - provider objects may expose raising properties
        return None
