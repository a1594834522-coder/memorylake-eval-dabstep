from __future__ import annotations

import re
import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.analysis_executor import ExecutionResult
from dabstep_agent_pydantic.analysis_executor import execute_analysis
from dabstep_agent_pydantic.candidate_builder import Candidate
from dabstep_agent_pydantic.candidate_builder import CandidateSet
from dabstep_agent_pydantic.candidate_builder import build_candidate_set
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.policy_registry import PolicyQuery
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.runtime_util import cached_load_dabstep_data
from dabstep_agent_pydantic.compact_semantic_planner import plan_compact_intent
from dabstep_agent_pydantic.compact_semantic_planner import route_intent_family
from dabstep_agent_pydantic.intent_compiler import IntentCompilationError
from dabstep_agent_pydantic.intent_compiler import compile_semantic_intent
from dabstep_agent_pydantic.semantic_intent import UnsupportedIntent
from dabstep_agent_pydantic.semantic_planner import AnalysisPlanProposal
from dabstep_agent_pydantic.semantic_planner import SemanticPlannerResult
from dabstep_agent_pydantic.semantic_planner import SemanticUncertainty
from dabstep_agent_pydantic.semantic_planner import plan_semantics
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_verifier import SemanticVerificationResult
from dabstep_agent_pydantic.semantic_verifier import verify_candidate_set
from dabstep_agent_pydantic.usage_telemetry import UsageLedger


class SemanticMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    PRIMARY = "primary"
    STRICT = "strict"


PlannerHook = Callable[..., Awaitable[SemanticPlannerResult]]
CandidateBuilderHook = Callable[..., CandidateSet]
ExecutorHook = Callable[..., ExecutionResult]
VerifierHook = Callable[..., Awaitable[SemanticVerificationResult]]
LegacyRunner = Callable[[], Awaitable[dict[str, object]]]


@dataclass(frozen=True)
class SemanticRuntimeHooks:
    registry: PolicyRegistry
    planner: PlannerHook
    candidate_builder: CandidateBuilderHook
    executor: ExecutorHook
    verifier: VerifierHook
    primary_judge: Any = None
    secondary_judge: Any = None
    require_second_judge: bool = False
    max_model_calls: int = 4


class SemanticWorkflowError(RuntimeError):
    def __init__(self, message: str, *, trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class _SemanticAttempt:
    record: dict[str, object] | None
    trace: dict[str, Any]
    fallback_reason: str | None = None


async def run_semantic_workflow(
    *,
    task: Task,
    mode: SemanticMode | str,
    data_dir: Path | None,
    workspace_dir: Path | None,
    file_summary: str,
    memory_context: str | None,
    legacy_runner: LegacyRunner,
    hooks: SemanticRuntimeHooks | None = None,
) -> dict[str, object]:
    resolved_mode = SemanticMode(mode)
    runtime = hooks or default_semantic_runtime_hooks()

    if resolved_mode is SemanticMode.LEGACY:
        legacy_record = await legacy_runner()
        trace = _empty_trace(resolved_mode)
        trace["legacy_trace"] = _legacy_trace(legacy_record)
        return _attach_trace(legacy_record, trace, selected_path="legacy")

    legacy_record: dict[str, object] | None = None
    if resolved_mode is SemanticMode.SHADOW:
        legacy_record = await legacy_runner()

    attempt = await _run_semantic_attempt(
        task=task,
        mode=resolved_mode,
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        file_summary=file_summary,
        memory_context=memory_context,
        hooks=runtime,
    )

    if resolved_mode is SemanticMode.SHADOW:
        assert legacy_record is not None
        trace = dict(attempt.trace)
        trace["legacy_trace"] = _legacy_trace(legacy_record)
        if attempt.record is not None:
            trace["semantic_candidate_answer"] = attempt.record.get("agent_answer")
        return _attach_trace(
            legacy_record,
            trace,
            selected_path="legacy",
            fallback_reason=attempt.fallback_reason,
        )

    if attempt.record is not None:
        return _attach_trace(attempt.record, attempt.trace, selected_path="semantic")

    if resolved_mode is SemanticMode.STRICT:
        trace = dict(attempt.trace)
        trace["selected_path"] = "strict_failure"
        trace["fallback_reason"] = attempt.fallback_reason
        raise SemanticWorkflowError(
            attempt.fallback_reason or "strict semantic workflow failed",
            trace=trace,
        )

    legacy_record = await legacy_runner()
    trace = dict(attempt.trace)
    trace["legacy_trace"] = _legacy_trace(legacy_record)
    return _attach_trace(
        legacy_record,
        trace,
        selected_path="legacy_fallback",
        fallback_reason=attempt.fallback_reason,
    )


async def _run_semantic_attempt(
    *,
    task: Task,
    mode: SemanticMode,
    data_dir: Path | None,
    workspace_dir: Path | None,
    file_summary: str,
    memory_context: str | None,
    hooks: SemanticRuntimeHooks,
) -> _SemanticAttempt:
    started = time.perf_counter()
    ledger = UsageLedger(max_calls=hooks.max_model_calls)
    trace = _empty_trace(mode)
    policy_query = _policy_query(task)
    policies = [match.policy for match in hooks.registry.search(policy_query)]

    try:
        planner_result = await hooks.planner(
            task=task,
            schema_summary=file_summary,
            policies=policies,
            registry=hooks.registry,
            memory_context=memory_context,
            usage_ledger=ledger,
        )
    except Exception as exc:  # noqa: BLE001 - semantic failures must preserve legacy availability.
        return _failed_attempt(
            trace,
            ledger,
            started,
            f"semantic planner failed: {type(exc).__name__}: {exc}",
        )

    proposal = planner_result.proposal
    trace["plan"] = proposal.model_dump(mode="json") if proposal is not None else None
    trace["intent"] = planner_result.intent
    trace["intent_family"] = planner_result.intent_family
    trace["schema_chars"] = planner_result.schema_chars
    trace["prompt_chars"] = planner_result.prompt_chars
    trace["compiled_plan"] = planner_result.compiled_plan
    if proposal is None:
        return _failed_attempt(
            trace,
            ledger,
            started,
            planner_result.fallback_reason or "semantic planner returned no valid proposal",
        )
    if proposal.unsupported_reason:
        return _failed_attempt(trace, ledger, started, proposal.unsupported_reason)
    if proposal.analysis_spec is None:
        return _failed_attempt(trace, ledger, started, "semantic planner returned no AnalysisSpec")
    if planner_result.compiled_plan is None:
        trace["compiled_plan"] = proposal.analysis_spec.model_dump(mode="json")

    try:
        candidate_set = hooks.candidate_builder(
            proposed_spec=proposal.analysis_spec,
            uncertainties=proposal.uncertainties,
            registry=hooks.registry,
            task=task,
            data_dir=data_dir,
        )
    except Exception as exc:  # noqa: BLE001 - malformed candidate expansion must fall back.
        return _failed_attempt(
            trace,
            ledger,
            started,
            f"candidate construction failed: {type(exc).__name__}: {exc}",
        )
    trace["candidates"] = candidate_set.model_dump(mode="json")
    if not candidate_set.candidates:
        return _failed_attempt(trace, ledger, started, "candidate construction produced no candidates")

    executions: dict[str, ExecutionResult] = {}
    for candidate in candidate_set.candidates:
        try:
            execution = hooks.executor(
                candidate=candidate,
                task=task,
                data_dir=data_dir,
            )
        except Exception as exc:  # noqa: BLE001 - partial execution is not safe to select from.
            trace["executions"][candidate.candidate_id] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return _failed_attempt(
                trace,
                ledger,
                started,
                f"execution failed for {candidate.candidate_id}: {type(exc).__name__}: {exc}",
            )
        executions[candidate.candidate_id] = execution
        trace["executions"][candidate.candidate_id] = asdict(execution)

    try:
        verification = await hooks.verifier(
            task=task,
            candidate_set=candidate_set,
            executions=executions,
            registry=hooks.registry,
            policy_query=policy_query,
            primary_judge=hooks.primary_judge,
            secondary_judge=hooks.secondary_judge,
            require_second_judge=hooks.require_second_judge,
            usage_ledger=ledger,
        )
    except Exception as exc:  # noqa: BLE001 - judge/verifier failures must not leak an unverified answer.
        return _failed_attempt(
            trace,
            ledger,
            started,
            f"semantic verifier failed: {type(exc).__name__}: {exc}",
        )
    trace["verifier"] = verification.model_dump(mode="json")
    if not verification.accepted or not verification.selected_candidate_id:
        return _failed_attempt(trace, ledger, started, verification.reason)

    selected_candidate = next(
        (
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == verification.selected_candidate_id
        ),
        None,
    )
    selected_execution = executions.get(verification.selected_candidate_id)
    if selected_candidate is None or selected_execution is None:
        return _failed_attempt(
            trace,
            ledger,
            started,
            "semantic verifier selected an unknown or unexecuted candidate",
        )
    if mode is SemanticMode.STRICT and not _is_strictly_certified(
        selected_candidate,
        hooks.registry,
    ):
        return _failed_attempt(
            trace,
            ledger,
            started,
            "strict mode requires a fully resolved AnalysisSpec backed by certified policies",
        )

    trace["usage"] = ledger.summary()
    trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return _SemanticAttempt(
        record={
            "task_id": task.task_id,
            "agent_answer": selected_execution.formatted_value,
            "reasoning": (
                "Deterministic AnalysisSpec execution selected by the semantic verifier: "
                f"{verification.reason}"
            ),
            "used_code": False,
            "elapsed_seconds": round(trace["elapsed_ms"] / 1000, 3),
            "code_path": str(workspace_dir) if workspace_dir is not None else None,
            "deterministic_route": f"semantic:{selected_candidate.candidate_id}",
            "usage_trace": ledger.summary(),
        },
        trace=trace,
    )


def default_semantic_runtime_hooks() -> SemanticRuntimeHooks:
    return SemanticRuntimeHooks(
        registry=PolicyRegistry([]),
        planner=_default_planner,
        candidate_builder=_default_candidate_builder,
        executor=_default_executor,
        verifier=_default_verifier,
    )


async def _default_planner(
    *,
    task: Task,
    schema_summary: str,
    policies,
    memory_context: str | None,
    usage_ledger: UsageLedger,
    **kwargs,
) -> SemanticPlannerResult:
    """Compact intent planner + deterministic compiler (default path)."""
    del kwargs
    family = route_intent_family(task.question, task.guidelines)
    compact = await plan_compact_intent(
        question=task.question,
        guidelines=task.guidelines or "",
        schema_summary=schema_summary,
        policies=list(policies),
        manual_excerpts=[memory_context] if memory_context else [],
        family=family,
        usage_ledger=usage_ledger,
    )
    usage = compact.usage_trace
    intent_dump = (
        compact.intent.model_dump(mode="json") if compact.intent is not None else None
    )
    if compact.intent is None:
        return SemanticPlannerResult(
            proposal=None,
            attempts=compact.attempts,
            fallback_reason=compact.fallback_reason or "compact planner returned no intent",
            usage_trace=usage,
            intent=intent_dump,
            intent_family=compact.family,
            schema_chars=compact.schema_chars,
            prompt_chars=compact.prompt_chars,
        )
    if isinstance(compact.intent, UnsupportedIntent):
        return SemanticPlannerResult(
            proposal=AnalysisPlanProposal(unsupported_reason=compact.intent.reason),
            attempts=compact.attempts,
            usage_trace=usage,
            intent=intent_dump,
            intent_family=compact.family,
            schema_chars=compact.schema_chars,
            prompt_chars=compact.prompt_chars,
        )
    try:
        compiled = compile_semantic_intent(
            compact.intent,
            guidelines=task.guidelines or "",
        )
    except IntentCompilationError as exc:
        return SemanticPlannerResult(
            proposal=None,
            attempts=compact.attempts,
            fallback_reason=f"intent compilation failed: {exc}",
            usage_trace=usage,
            intent=intent_dump,
            intent_family=compact.family,
            schema_chars=compact.schema_chars,
            prompt_chars=compact.prompt_chars,
        )
    uncertainties = [
        SemanticUncertainty(
            axis=axis,
            proposed_choice="intent_selected",
            rival_policy_ids=[],
            rationale="Surfaced by compact intent uncertainty_axes.",
            confidence=0.5,
        )
        for axis in getattr(compact.intent, "uncertainty_axes", []) or []
    ]
    return SemanticPlannerResult(
        proposal=AnalysisPlanProposal(
            analysis_spec=compiled,
            uncertainties=uncertainties,
        ),
        attempts=compact.attempts,
        usage_trace=usage,
        intent=intent_dump,
        intent_family=compact.family,
        schema_chars=compact.schema_chars,
        prompt_chars=compact.prompt_chars,
        compiled_plan=compiled.model_dump(mode="json"),
    )


async def full_spec_planner_hook(
    *,
    task: Task,
    schema_summary: str,
    policies,
    memory_context: str | None,
    usage_ledger: UsageLedger,
    **kwargs,
) -> SemanticPlannerResult:
    """Compatibility hook: model emits full AnalysisSpec (pre-compact path)."""
    del kwargs
    return await plan_semantics(
        question=task.question,
        guidelines=task.guidelines or "",
        schema_summary=schema_summary,
        policies=list(policies),
        manual_excerpts=[memory_context] if memory_context else [],
        usage_ledger=usage_ledger,
    )


def _default_candidate_builder(*, proposed_spec, uncertainties, registry, **kwargs) -> CandidateSet:
    del kwargs
    return build_candidate_set(
        proposed_spec,
        uncertainties=list(uncertainties),
        registry=registry,
    )


def _default_executor(*, candidate: Candidate, data_dir: Path | None, **kwargs) -> ExecutionResult:
    del kwargs
    if data_dir is None:
        raise ValueError("data_dir is required for deterministic semantic execution")
    return execute_analysis(cached_load_dabstep_data(data_dir), candidate.spec)


async def _default_verifier(
    *,
    task: Task,
    candidate_set: CandidateSet,
    executions: dict[str, ExecutionResult],
    registry: PolicyRegistry,
    policy_query: PolicyQuery,
    primary_judge,
    secondary_judge,
    require_second_judge: bool,
    usage_ledger: UsageLedger,
    **kwargs,
) -> SemanticVerificationResult:
    del kwargs
    return await verify_candidate_set(
        question=task.question,
        guidelines=task.guidelines or "",
        candidate_set=candidate_set,
        executions=executions,
        registry=registry,
        policy_query=policy_query,
        primary_judge=primary_judge,
        secondary_judge=secondary_judge,
        require_second_judge=require_second_judge,
        usage_ledger=usage_ledger,
    )


def _policy_query(task: Task) -> PolicyQuery:
    family = plan_task(
        question=task.question,
        guidelines=task.guidelines,
        route_cards=[],
    ).task_family
    terms = sorted(set(re.findall(r"[a-z0-9_]+", task.question.lower())))
    return PolicyQuery(family=family, terms=terms, top_k=5)


def _is_strictly_certified(candidate: Candidate, registry: PolicyRegistry) -> bool:
    if candidate.spec.unresolved_axes or not candidate.spec.policy_ids:
        return False
    return all(
        (policy := registry.get(policy_id)) is not None
        and policy.status in {PolicyStatus.CERTIFIED, PolicyStatus.ACTIVE}
        for policy_id in candidate.spec.policy_ids
    )


def _empty_trace(mode: SemanticMode) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "plan": None,
        "intent": None,
        "intent_family": None,
        "schema_chars": None,
        "prompt_chars": None,
        "compiled_plan": None,
        "candidates": None,
        "executions": {},
        "verifier": None,
        "usage": {},
        "fallback_reason": None,
        "selected_path": None,
        "legacy_trace": None,
        "semantic_candidate_answer": None,
        "elapsed_ms": 0,
    }


def _failed_attempt(
    trace: dict[str, Any],
    ledger: UsageLedger,
    started: float,
    reason: str,
) -> _SemanticAttempt:
    trace["usage"] = ledger.summary()
    trace["fallback_reason"] = reason
    trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return _SemanticAttempt(record=None, trace=trace, fallback_reason=reason)


def _legacy_trace(record: dict[str, object]) -> dict[str, object]:
    return {
        "workflow": record.get("workflow_trace"),
        "usage": record.get("usage_trace", {}),
    }


def _attach_trace(
    record: dict[str, object],
    trace: dict[str, Any],
    *,
    selected_path: str,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    result = dict(record)
    result_trace = dict(trace)
    result_trace["selected_path"] = selected_path
    result_trace["fallback_reason"] = fallback_reason
    result["semantic_trace"] = result_trace
    return result
