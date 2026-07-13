from __future__ import annotations

import asyncio

import pytest

from dabstep_agent_pydantic.analysis_executor import ExecutionResult
from dabstep_agent_pydantic.analysis_executor import analysis_plan_fingerprint
from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import OrderSpec
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.candidate_builder import Candidate
from dabstep_agent_pydantic.candidate_builder import CandidateSet
from dabstep_agent_pydantic.policy_registry import PolicyQuery
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_policy import ApplicabilityPredicate
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy
from dabstep_agent_pydantic.semantic_verifier import JudgeDecision
from dabstep_agent_pydantic.semantic_verifier import create_semantic_judge_agent
from dabstep_agent_pydantic.semantic_verifier import verify_candidate_set


HASH = "a" * 64


def _policy(policy_id: str, *, axis: str, choice: str, status: PolicyStatus) -> SemanticPolicy:
    certified = status in {PolicyStatus.ACTIVE, PolicyStatus.CERTIFIED}
    return SemanticPolicy(
        policy_id=policy_id,
        version=1,
        family="test_family",
        axis=axis,
        choice=choice,
        convention=f"Use {choice} for {axis}.",
        citations=[PolicyCitation(
            document_name="manual.md",
            section="semantics",
            content_hash=HASH,
        )] if certified else [],
        certification_id=f"cert:{policy_id}" if certified else None,
        status=status,
    )


def _spec(policy_id: str, *, output=None, group_by=None, ordering=None) -> AnalysisSpec:
    return AnalysisSpec(
        source=SourceSpec(table="payments"),
        measure=AggregateMeasure(kind="sum", column="eur_amount"),
        group_by=group_by or [],
        ordering=ordering or [],
        output=output or AnalysisOutputContract(kind="decimal", decimals=4),
        policy_ids=[policy_id] if policy_id else [],
    )


def _candidate(candidate_id: str, spec: AnalysisSpec, *, origin="uncertainty_rival") -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        spec=spec,
        origin=origin,
        applied_policy_ids=list(spec.policy_ids),
    )


def _execution(spec: AnalysisSpec, value, formatted: str) -> ExecutionResult:
    return ExecutionResult(
        raw_value=value,
        formatted_value=formatted,
        row_counts={"source": 3, "measure": 3},
        intermediates={},
        policy_ids=list(spec.policy_ids),
        plan_fingerprint=analysis_plan_fingerprint(spec),
        execution_fingerprint="sha256:" + "b" * 64,
    )


@pytest.mark.parametrize(
    ("scenario", "axis", "right_choice", "wrong_choice", "right_output", "wrong_output"),
    [
        ("fraud-volume-vs-count", "fraud_rate_basis", "eur_volume", "transaction_count", "23.0769", "33.3333"),
        ("fee-mean-vs-sum", "fee_rule_reducer", "mean", "sum", "0.4200", "1.2600"),
        ("missing-email", "missing_email", "exclude", "include", "10.0000", "12.0000"),
        ("repeat-history", "repeat_customer_scope", "full_history", "period_only", "40.0000", "25.0000"),
        ("fee-wildcard", "empty_fee_field", "wildcard", "strict", "1.0000", "0.0000"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_certified_policy_vetoes_semantic_sabotage(
    scenario, axis, right_choice, wrong_choice, right_output, wrong_output,
):
    del scenario
    right_policy = _policy("right.v1", axis=axis, choice=right_choice, status=PolicyStatus.ACTIVE)
    wrong_policy = _policy("wrong.v1", axis=axis, choice=wrong_choice, status=PolicyStatus.PROPOSED)
    wrong_spec = _spec(wrong_policy.policy_id)
    right_spec = _spec(right_policy.policy_id)
    candidates = CandidateSet(candidates=[
        _candidate("wrong", wrong_spec, origin="proposed"),
        _candidate("right", right_spec),
    ])

    result = asyncio.run(verify_candidate_set(
        question="ambiguous benchmark question",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "wrong": _execution(wrong_spec, float(wrong_output), wrong_output),
            "right": _execution(right_spec, float(right_output), right_output),
        },
        registry=PolicyRegistry([wrong_policy, right_policy]),
    ))

    assert result.accepted is True
    assert result.selected_candidate_id == "right"
    assert result.level == "certified_policy"
    assert result.judge_votes == []


def test_structural_verifier_rejects_wrong_precision():
    policy = _policy("p.v1", axis="aggregation", choice="sum", status=PolicyStatus.ACTIVE)
    spec = _spec(policy.policy_id)
    candidates = CandidateSet(candidates=[_candidate("proposed", spec, origin="proposed")])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={"proposed": _execution(spec, 1.2, "1.2")},
        registry=PolicyRegistry([policy]),
    ))

    assert result.accepted is False
    assert "deterministic format" in result.reason


def test_structural_verifier_contains_malformed_execution_values():
    policy = _policy("p.v1", axis="aggregation", choice="sum", status=PolicyStatus.ACTIVE)
    spec = _spec(policy.policy_id)
    candidates = CandidateSet(candidates=[_candidate("proposed", spec, origin="proposed")])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={"proposed": _execution(spec, "not-a-number", "not-a-number")},
        registry=PolicyRegistry([policy]),
    ))

    assert result.accepted is False
    assert "could not be reconstructed" in result.reason


def test_structural_verifier_rejects_wrong_group_ordering():
    policy = _policy("p.v1", axis="ordering", choice="ascending", status=PolicyStatus.ACTIVE)
    spec = _spec(
        policy.policy_id,
        output=AnalysisOutputContract(kind="group_value_list", decimals=2),
        group_by=["channel"],
        ordering=[OrderSpec(by="value", direction="asc")],
    )
    raw = [
        {"group": {"channel": "A"}, "value": 2.0},
        {"group": {"channel": "B"}, "value": 1.0},
    ]
    candidates = CandidateSet(candidates=[_candidate("proposed", spec, origin="proposed")])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be a comma separated list of [group: value] entries.",
        candidate_set=candidates,
        executions={"proposed": _execution(spec, raw, "[A: 2.00, B: 1.00]")},
        registry=PolicyRegistry([policy]),
    ))

    assert result.accepted is False
    assert "ordering" in result.reason


def test_identical_outputs_are_accepted_without_judge():
    first_spec = _spec("")
    second_spec = _spec("")
    candidates = CandidateSet(candidates=[
        _candidate("first", first_spec, origin="proposed"),
        _candidate("second", second_spec),
    ])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "first": _execution(first_spec, 1.0, "1.0000"),
            "second": _execution(second_spec, 1.0, "1.0000"),
        },
        registry=PolicyRegistry([]),
    ))

    assert result.accepted is True
    assert result.selected_candidate_id == "first"
    assert result.level == "identical_outputs"
    assert result.judge_votes == []


def test_proposed_output_matching_rejected_candidate_requires_resolution():
    proposed_spec = _spec("")
    rejected_spec = _spec("")
    candidates = CandidateSet(candidates=[
        _candidate("proposed", proposed_spec, origin="proposed"),
        _candidate("known-rejected", rejected_spec),
    ])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "proposed": _execution(proposed_spec, 2.0, "2.0000"),
            "known-rejected": _execution(rejected_spec, 2.0, "2.0000"),
        },
        registry=PolicyRegistry([]),
        rejected_candidate_ids={"known-rejected"},
    ))

    assert result.accepted is False
    assert "known rejected candidate output" in result.reason


def test_conflicting_oracle_and_dev_evidence_cannot_be_overridden_by_policy():
    right_policy = _policy("right.v1", axis="basis", choice="right", status=PolicyStatus.ACTIVE)
    wrong_policy = _policy("wrong.v1", axis="basis", choice="wrong", status=PolicyStatus.PROPOSED)
    first_spec = _spec(wrong_policy.policy_id)
    second_spec = _spec(right_policy.policy_id)
    candidates = CandidateSet(candidates=[
        _candidate("first", first_spec, origin="proposed"),
        _candidate("second", second_spec),
    ])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "first": _execution(first_spec, 1.0, "1.0000"),
            "second": _execution(second_spec, 2.0, "2.0000"),
        },
        registry=PolicyRegistry([wrong_policy, right_policy]),
        oracle_candidate_id="first",
        dev_candidate_id="second",
    ))

    assert result.accepted is False
    assert "oracle and dev evidence conflict" in result.reason


def test_policy_applicability_is_checked_against_query_context():
    policy = _policy("fraud.v1", axis="basis", choice="volume", status=PolicyStatus.ACTIVE)
    policy = policy.model_copy(update={
        "applicability": [ApplicabilityPredicate(
            field="metric", operator="eq", value="fraud_rate",
        )],
    })
    spec = _spec(policy.policy_id)
    candidates = CandidateSet(candidates=[_candidate("candidate", spec, origin="proposed")])

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={"candidate": _execution(spec, 1.0, "1.0000")},
        registry=PolicyRegistry([policy]),
        policy_query=PolicyQuery(family="test_family", metric="merchant_count"),
    ))

    assert result.accepted is False
    assert "not applicable" in result.reason


class _JudgeResult:
    def __init__(self, decision, model_name):
        self.output = decision
        self.usage = {"input_tokens": 30, "output_tokens": 5}
        self.model_name = model_name


class _Judge:
    def __init__(self, decision, model_name):
        self.decision = decision
        self.model_name = model_name
        self.calls = []

    async def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return _JudgeResult(self.decision, self.model_name)


def test_unresolved_candidates_use_one_short_judge_without_tools():
    first_spec = _spec("")
    second_spec = _spec("")
    candidates = CandidateSet(candidates=[
        _candidate("first", first_spec, origin="proposed"),
        _candidate("second", second_spec),
    ])
    judge = _Judge(JudgeDecision(
        selected_candidate_id="second",
        confidence=0.9,
        rationale="The modifier selects the second interpretation.",
    ), "judge-a")

    result = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "first": _execution(first_spec, 1.0, "1.0000"),
            "second": _execution(second_spec, 2.0, "2.0000"),
        },
        registry=PolicyRegistry([]),
        primary_judge=judge,
    ))

    assert result.accepted is True
    assert result.selected_candidate_id == "second"
    assert result.level == "semantic_judge"
    assert len(judge.calls) == 1
    assert "tools" not in judge.calls[0][1] and "toolsets" not in judge.calls[0][1]
    assert result.usage_trace["semantic_judge"]["calls"] == 1


def test_low_confidence_requires_independent_second_judge():
    first_spec = _spec("")
    second_spec = _spec("")
    candidates = CandidateSet(candidates=[
        _candidate("first", first_spec, origin="proposed"),
        _candidate("second", second_spec),
    ])
    primary = _Judge(JudgeDecision(
        selected_candidate_id="second", confidence=0.6, rationale="uncertain",
    ), "judge-a")
    secondary = _Judge(JudgeDecision(
        selected_candidate_id="second", confidence=0.9, rationale="independent agreement",
    ), "judge-b")

    accepted = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "first": _execution(first_spec, 1.0, "1.0000"),
            "second": _execution(second_spec, 2.0, "2.0000"),
        },
        registry=PolicyRegistry([]),
        primary_judge=primary,
        secondary_judge=secondary,
    ))
    same_model = _Judge(JudgeDecision(
        selected_candidate_id="second", confidence=0.9, rationale="same model",
    ), "judge-a")
    rejected = asyncio.run(verify_candidate_set(
        question="q",
        guidelines="Answer must be rounded to 4 decimals.",
        candidate_set=candidates,
        executions={
            "first": _execution(first_spec, 1.0, "1.0000"),
            "second": _execution(second_spec, 2.0, "2.0000"),
        },
        registry=PolicyRegistry([]),
        primary_judge=primary,
        secondary_judge=same_model,
    ))

    assert accepted.accepted is True
    assert accepted.level == "independent_judges"
    assert len(accepted.judge_votes) == 2
    assert rejected.accepted is False
    assert "independent" in rejected.reason


def test_semantic_judge_factory_has_no_tools_or_internal_retries(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr("dabstep_agent_pydantic.semantic_verifier.Agent", FakeAgent)
    monkeypatch.delenv("DABSTEP_PLANNER_THINKING", raising=False)

    judge = create_semantic_judge_agent(model="judge-model")

    assert isinstance(judge, FakeAgent)
    assert captured["kwargs"]["output_type"] is JudgeDecision
    assert captured["kwargs"]["retries"] == 0
    assert captured["kwargs"]["model_settings"] == {"thinking": "low"}
    assert "tools" not in captured["kwargs"] and "toolsets" not in captured["kwargs"]
