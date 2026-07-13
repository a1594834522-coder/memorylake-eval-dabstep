from __future__ import annotations

import asyncio

import pytest

from dabstep_agent_pydantic.analysis_executor import ExecutionResult
from dabstep_agent_pydantic.analysis_executor import analysis_plan_fingerprint
from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.candidate_builder import Candidate
from dabstep_agent_pydantic.candidate_builder import CandidateSet
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_planner import AnalysisPlanProposal
from dabstep_agent_pydantic.semantic_planner import SemanticPlannerResult
from dabstep_agent_pydantic.semantic_verifier import SemanticVerificationResult
from dabstep_agent_pydantic.semantic_workflow import SemanticMode
from dabstep_agent_pydantic.semantic_workflow import SemanticRuntimeHooks
from dabstep_agent_pydantic.semantic_workflow import SemanticWorkflowError
from dabstep_agent_pydantic.semantic_workflow import run_semantic_workflow


def _task() -> Task:
    return Task(
        task_id="task-1",
        question="How many payments are there?",
        guidelines="Return a single integer.",
    )


def _spec() -> AnalysisSpec:
    return AnalysisSpec(
        source=SourceSpec(table="payments"),
        measure=AggregateMeasure(kind="count"),
        output=AnalysisOutputContract(kind="integer"),
    )


def _candidate(spec: AnalysisSpec | None = None) -> Candidate:
    return Candidate(
        candidate_id="candidate-proposed",
        spec=spec or _spec(),
        origin="proposed",
    )


def _execution(spec: AnalysisSpec | None = None) -> ExecutionResult:
    active_spec = spec or _spec()
    return ExecutionResult(
        raw_value=42,
        formatted_value="42",
        row_counts={"source": 42, "measure": 42},
        intermediates={},
        policy_ids=list(active_spec.policy_ids),
        plan_fingerprint=analysis_plan_fingerprint(active_spec),
        execution_fingerprint="sha256:" + "e" * 64,
    )


def _planner_result(
    *,
    proposal: AnalysisPlanProposal | None = None,
    fallback_reason: str | None = None,
) -> SemanticPlannerResult:
    return SemanticPlannerResult(
        proposal=proposal,
        attempts=1,
        fallback_reason=fallback_reason,
    )


def _accepted_verification() -> SemanticVerificationResult:
    return SemanticVerificationResult(
        accepted=True,
        selected_candidate_id="candidate-proposed",
        level="identical_outputs",
        reason="single deterministic candidate",
    )


def _hooks(**overrides) -> SemanticRuntimeHooks:
    async def planner(**kwargs):
        del kwargs
        return _planner_result(proposal=AnalysisPlanProposal(analysis_spec=_spec()))

    def candidate_builder(proposed_spec, **kwargs):
        del kwargs
        return CandidateSet(candidates=[_candidate(proposed_spec)])

    def executor(candidate, **kwargs):
        del kwargs
        return _execution(candidate.spec)

    async def verifier(**kwargs):
        del kwargs
        return _accepted_verification()

    values = {
        "registry": PolicyRegistry([]),
        "planner": planner,
        "candidate_builder": candidate_builder,
        "executor": executor,
        "verifier": verifier,
    }
    values.update(overrides)
    return SemanticRuntimeHooks(**values)


def _legacy_runner(calls: list[str]):
    async def run():
        calls.append("legacy")
        return {
            "task_id": "task-1",
            "agent_answer": "legacy-answer",
            "reasoning": "legacy reasoning",
            "used_code": True,
            "workflow_trace": {"stages": ["plan", "solve", "verify", "finalize"]},
            "usage_trace": {"solver": {"calls": 1}},
        }

    return run


def _run(mode: SemanticMode, *, hooks: SemanticRuntimeHooks | None = None, calls=None):
    legacy_calls = calls if calls is not None else []
    return asyncio.run(
        run_semantic_workflow(
            task=_task(),
            mode=mode,
            data_dir=None,
            workspace_dir=None,
            file_summary="payments table",
            memory_context=None,
            legacy_runner=_legacy_runner(legacy_calls),
            hooks=hooks or _hooks(),
        )
    )


def test_legacy_mode_skips_semantic_pipeline():
    planner_called = False

    async def planner(**kwargs):
        nonlocal planner_called
        del kwargs
        planner_called = True
        raise AssertionError("legacy mode must not call the semantic planner")

    calls: list[str] = []
    record = _run(SemanticMode.LEGACY, hooks=_hooks(planner=planner), calls=calls)

    assert record["agent_answer"] == "legacy-answer"
    assert calls == ["legacy"]
    assert planner_called is False
    assert record["semantic_trace"]["selected_path"] == "legacy"


def test_shadow_returns_legacy_answer_and_retains_both_traces():
    calls: list[str] = []

    record = _run(SemanticMode.SHADOW, calls=calls)

    trace = record["semantic_trace"]
    assert record["agent_answer"] == "legacy-answer"
    assert calls == ["legacy"]
    assert trace["selected_path"] == "legacy"
    assert trace["legacy_trace"]["workflow"]["stages"] == [
        "plan",
        "solve",
        "verify",
        "finalize",
    ]
    assert trace["plan"]["analysis_spec"]["measure"]["kind"] == "count"
    assert trace["candidates"]["candidates"][0]["candidate_id"] == "candidate-proposed"
    assert trace["executions"]["candidate-proposed"]["formatted_value"] == "42"
    assert trace["verifier"]["accepted"] is True
    assert trace["semantic_candidate_answer"] == "42"
    assert "usage" in trace


@pytest.mark.parametrize("mode", [SemanticMode.CANDIDATE, SemanticMode.PRIMARY])
def test_candidate_and_primary_use_verified_semantic_answer_without_legacy(mode):
    calls: list[str] = []

    record = _run(mode, calls=calls)

    assert record["agent_answer"] == "42"
    assert record["deterministic_route"] == "semantic:candidate-proposed"
    assert record["semantic_trace"]["selected_path"] == "semantic"
    assert record["semantic_trace"]["fallback_reason"] is None
    assert calls == []


@pytest.mark.parametrize(
    ("hooks", "reason"),
    [
        (
            _hooks(
                planner=lambda **kwargs: _async_result(
                    _planner_result(
                        proposal=None,
                        fallback_reason="invalid semantic planner output after repair",
                    )
                )
            ),
            "invalid semantic planner output",
        ),
        (
            _hooks(
                planner=lambda **kwargs: _async_result(
                    _planner_result(
                        proposal=AnalysisPlanProposal(
                            unsupported_reason="operation cannot be represented"
                        )
                    )
                )
            ),
            "operation cannot be represented",
        ),
        (
            _hooks(
                planner=lambda **kwargs: _async_result(
                    _planner_result(
                        proposal=None,
                        fallback_reason="global model-call budget of 0 exhausted",
                    )
                )
            ),
            "budget of 0 exhausted",
        ),
        (
            _hooks(executor=lambda **kwargs: _raise(RuntimeError("missing column"))),
            "execution failed",
        ),
        (
            _hooks(
                verifier=lambda **kwargs: _async_result(
                    SemanticVerificationResult(
                        accepted=False,
                        level="unresolved",
                        reason="independent semantic judges did not agree",
                    )
                )
            ),
            "independent semantic judges did not agree",
        ),
    ],
)
@pytest.mark.parametrize("mode", [SemanticMode.CANDIDATE, SemanticMode.PRIMARY])
def test_candidate_and_primary_fall_back_to_legacy_for_hard_failures(hooks, reason, mode):
    calls: list[str] = []

    record = _run(mode, hooks=hooks, calls=calls)

    assert record["agent_answer"] == "legacy-answer"
    assert calls == ["legacy"]
    assert record["semantic_trace"]["selected_path"] == "legacy_fallback"
    assert reason in record["semantic_trace"]["fallback_reason"]


def test_strict_mode_fails_closed_instead_of_calling_legacy():
    calls: list[str] = []

    with pytest.raises(SemanticWorkflowError, match="certified") as exc_info:
        _run(SemanticMode.STRICT, calls=calls)

    assert calls == []
    assert exc_info.value.trace["selected_path"] == "strict_failure"


def test_compact_intent_trace_fields_and_compiled_plan():
    compiled = _spec()

    async def planner(**kwargs):
        del kwargs
        return SemanticPlannerResult(
            proposal=AnalysisPlanProposal(analysis_spec=compiled),
            attempts=1,
            intent={
                "operation": "aggregate",
                "aggregation": "count",
                "source": "payments",
                "output_kind": "integer",
            },
            intent_family="general",
            schema_chars=1200,
            prompt_chars=800,
            compiled_plan=compiled.model_dump(mode="json"),
        )

    record = _run(SemanticMode.CANDIDATE, hooks=_hooks(planner=planner))
    trace = record["semantic_trace"]
    assert trace["intent"]["operation"] == "aggregate"
    assert trace["intent_family"] == "general"
    assert trace["schema_chars"] == 1200
    assert trace["prompt_chars"] == 800
    assert trace["compiled_plan"]["measure"]["kind"] == "count"
    assert record["agent_answer"] == "42"


def test_invalid_intent_compilation_falls_back_in_candidate_mode():
    async def planner(**kwargs):
        del kwargs
        return SemanticPlannerResult(
            proposal=None,
            attempts=1,
            fallback_reason="intent compilation failed: no deterministic primitive",
            intent={"operation": "correlation", "source": "payments"},
            intent_family="general",
            schema_chars=1000,
            prompt_chars=500,
        )

    calls: list[str] = []
    record = _run(SemanticMode.CANDIDATE, hooks=_hooks(planner=planner), calls=calls)
    assert record["agent_answer"] == "legacy-answer"
    assert calls == ["legacy"]
    assert "intent compilation failed" in record["semantic_trace"]["fallback_reason"]
    assert record["semantic_trace"]["intent_family"] == "general"


def test_shadow_retains_compact_and_legacy_traces():
    compiled = _spec()

    async def planner(**kwargs):
        del kwargs
        return SemanticPlannerResult(
            proposal=AnalysisPlanProposal(analysis_spec=compiled),
            attempts=1,
            intent={"operation": "aggregate", "aggregation": "count", "source": "payments"},
            intent_family="general",
            schema_chars=1100,
            prompt_chars=700,
            compiled_plan=compiled.model_dump(mode="json"),
        )

    calls: list[str] = []
    record = _run(SemanticMode.SHADOW, hooks=_hooks(planner=planner), calls=calls)
    trace = record["semantic_trace"]
    assert record["agent_answer"] == "legacy-answer"
    assert calls == ["legacy"]
    assert trace["selected_path"] == "legacy"
    assert trace["intent_family"] == "general"
    assert trace["compiled_plan"] is not None
    assert trace["semantic_candidate_answer"] == "42"
    assert trace["legacy_trace"]["workflow"] is not None


def test_strict_fails_closed_on_unsupported_intent():
    async def planner(**kwargs):
        del kwargs
        return SemanticPlannerResult(
            proposal=AnalysisPlanProposal(unsupported_reason="no closed operation"),
            attempts=1,
            intent={"reason": "no closed operation"},
            intent_family="unsupported",
            schema_chars=200,
            prompt_chars=100,
        )

    calls: list[str] = []
    with pytest.raises(SemanticWorkflowError, match="no closed operation") as exc_info:
        _run(SemanticMode.STRICT, hooks=_hooks(planner=planner), calls=calls)
    assert calls == []
    assert exc_info.value.trace["selected_path"] == "strict_failure"
    assert exc_info.value.trace["intent_family"] == "unsupported"


async def _async_result(value):
    return value


def _raise(exc: Exception):
    raise exc
