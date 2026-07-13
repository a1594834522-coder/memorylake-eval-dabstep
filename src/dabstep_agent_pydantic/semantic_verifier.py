from __future__ import annotations

import json
import time
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic_ai import Agent

from dabstep_agent_pydantic.agent import build_semantic_planner_model_from_env
from dabstep_agent_pydantic.agent import semantic_planner_model_settings_from_env
from dabstep_agent_pydantic.analysis_executor import ExecutionResult
from dabstep_agent_pydantic.analysis_executor import analysis_plan_fingerprint
from dabstep_agent_pydantic.candidate_builder import Candidate
from dabstep_agent_pydantic.candidate_builder import CandidateSet
from dabstep_agent_pydantic.output_contract import format_analysis_output
from dabstep_agent_pydantic.output_contract import parse_guidelines
from dabstep_agent_pydantic.output_contract import validate_output_contract
from dabstep_agent_pydantic.policy_registry import PolicyQuery
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.usage_telemetry import CallUsage
from dabstep_agent_pydantic.usage_telemetry import UsageBudgetExceeded
from dabstep_agent_pydantic.usage_telemetry import UsageLedger
from dabstep_agent_pydantic.usage_telemetry import call_usage_from_result


JUDGE_CONFIDENCE_THRESHOLD = 0.75
SELECTABLE_POLICY_STATUSES = {
    PolicyStatus.PROPOSED,
    PolicyStatus.PROVISIONAL,
    PolicyStatus.SHADOW,
    PolicyStatus.CERTIFIED,
    PolicyStatus.ACTIVE,
}
CERTIFIED_POLICY_STATUSES = {PolicyStatus.CERTIFIED, PolicyStatus.ACTIVE}
JUDGE_INSTRUCTIONS = """\
Choose the candidate whose semantic interpretation best matches the question
wording and guidelines. Do not calculate values, use data tools, write code,
or return the final answer. Return only a JudgeDecision.
"""


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_candidate_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class SemanticJudgeVote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["semantic_judge", "semantic_judge_secondary"]
    selected_candidate_id: str
    confidence: float
    rationale: str
    model_fingerprint: str | None = None


class SemanticVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    selected_candidate_id: str | None = None
    level: Literal[
        "structural",
        "oracle",
        "dev",
        "identical_outputs",
        "certified_policy",
        "semantic_judge",
        "independent_judges",
        "unresolved",
    ]
    reason: str
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    judge_votes: list[SemanticJudgeVote] = Field(default_factory=list)
    usage_trace: dict[str, dict[str, Any]] = Field(default_factory=dict)


def create_semantic_judge_agent(model=None):
    return Agent(
        model or build_semantic_planner_model_from_env(),
        output_type=JudgeDecision,
        instructions=JUDGE_INSTRUCTIONS,
        model_settings=semantic_planner_model_settings_from_env(),
        retries=0,
        defer_model_check=True,
    )


async def verify_candidate_set(
    *,
    question: str,
    guidelines: str,
    candidate_set: CandidateSet,
    executions: dict[str, ExecutionResult],
    registry: PolicyRegistry,
    policy_query: PolicyQuery | None = None,
    rejected_candidate_ids: set[str] | None = None,
    oracle_candidate_id: str | None = None,
    dev_candidate_id: str | None = None,
    primary_judge=None,
    secondary_judge=None,
    require_second_judge: bool = False,
    usage_ledger: UsageLedger | None = None,
) -> SemanticVerificationResult:
    ledger = usage_ledger or UsageLedger(max_calls=2)
    known_rejected = set(rejected_candidate_ids or set())
    candidates = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    invalid: dict[str, str] = {}

    for candidate_id, candidate in candidates.items():
        execution = executions.get(candidate_id)
        if execution is None:
            invalid[candidate_id] = "missing deterministic execution"
            continue
        issue = _execution_issue(candidate, execution, guidelines=guidelines)
        if issue:
            invalid[candidate_id] = issue
            continue
        policy_issue = _policy_issue(candidate, registry, policy_query=policy_query)
        if policy_issue:
            invalid[candidate_id] = policy_issue

    selectable = [
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_id not in invalid and candidate.candidate_id not in known_rejected
    ]
    if not selectable:
        reason = next(iter(invalid.values()), "every candidate is marked as rejected")
        return _result(
            accepted=False,
            level="structural",
            reason=reason,
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )

    evidence_ids = {
        candidate_id
        for candidate_id in (oracle_candidate_id, dev_candidate_id)
        if candidate_id and any(candidate.candidate_id == candidate_id for candidate in selectable)
    }
    evidence_conflict = len(evidence_ids) > 1
    if len(evidence_ids) == 1:
        selected = next(iter(evidence_ids))
        level = "oracle" if oracle_candidate_id == selected else "dev"
        return _result(
            accepted=True,
            selected_candidate_id=selected,
            level=level,
            reason=f"{level} evidence selected candidate {selected}",
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )

    proposed = next(
        (candidate for candidate in candidate_set.candidates if candidate.origin == "proposed"),
        candidate_set.candidates[0],
    )
    proposed_execution = executions.get(proposed.candidate_id)
    rejected_outputs = {
        executions[candidate_id].formatted_value
        for candidate_id in known_rejected
        if candidate_id in executions
    }
    rejected_collision = (
        proposed_execution is not None
        and proposed_execution.formatted_value in rejected_outputs
    )

    outputs = {
        executions[candidate.candidate_id].formatted_value
        for candidate in selectable
    }
    if len(outputs) == 1 and not rejected_collision and not evidence_conflict:
        selected = proposed if proposed in selectable else selectable[0]
        return _result(
            accepted=True,
            selected_candidate_id=selected.candidate_id,
            level="identical_outputs",
            reason="all structurally valid candidates produced the same output",
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )

    certified = [candidate for candidate in selectable if _is_certified(candidate, registry)]
    if len(certified) == 1 and not evidence_conflict:
        return _result(
            accepted=True,
            selected_candidate_id=certified[0].candidate_id,
            level="certified_policy",
            reason="a unique candidate is backed only by certified active policies",
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )

    if primary_judge is None:
        if evidence_conflict:
            reason = "oracle and dev evidence conflict and no semantic judge is configured"
        elif rejected_collision:
            reason = "proposed output matches a known rejected candidate output and lacks independent resolution"
        else:
            reason = "candidate semantics remain unresolved and no semantic judge is configured"
        return _result(
            accepted=False,
            level="unresolved",
            reason=reason,
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )

    prompt = _judge_prompt(question, guidelines, selectable, registry)
    primary_vote, error = await _run_judge(
        primary_judge,
        stage="semantic_judge",
        prompt=prompt,
        allowed_candidate_ids={candidate.candidate_id for candidate in selectable},
        ledger=ledger,
    )
    if primary_vote is None:
        return _result(
            accepted=False,
            level="unresolved",
            reason=error or "primary semantic judge failed",
            rejected_ids={*known_rejected, *invalid},
            ledger=ledger,
        )
    votes = [primary_vote]
    if primary_vote.confidence >= JUDGE_CONFIDENCE_THRESHOLD and not require_second_judge:
        return _result(
            accepted=True,
            selected_candidate_id=primary_vote.selected_candidate_id,
            level="semantic_judge",
            reason="primary semantic judge resolved the candidate set with high confidence",
            rejected_ids={*known_rejected, *invalid},
            votes=votes,
            ledger=ledger,
        )

    if secondary_judge is None:
        return _result(
            accepted=False,
            level="unresolved",
            reason="primary judge confidence is low and no independent secondary judge is configured",
            rejected_ids={*known_rejected, *invalid},
            votes=votes,
            ledger=ledger,
        )

    secondary_vote, error = await _run_judge(
        secondary_judge,
        stage="semantic_judge_secondary",
        prompt=prompt,
        allowed_candidate_ids={candidate.candidate_id for candidate in selectable},
        ledger=ledger,
    )
    if secondary_vote is None:
        return _result(
            accepted=False,
            level="unresolved",
            reason=error or "secondary semantic judge failed",
            rejected_ids={*known_rejected, *invalid},
            votes=votes,
            ledger=ledger,
        )
    votes.append(secondary_vote)
    if (
        not primary_vote.model_fingerprint
        or not secondary_vote.model_fingerprint
        or primary_vote.model_fingerprint == secondary_vote.model_fingerprint
    ):
        return _result(
            accepted=False,
            level="unresolved",
            reason="secondary semantic judge is not independent from the primary model",
            rejected_ids={*known_rejected, *invalid},
            votes=votes,
            ledger=ledger,
        )
    if (
        primary_vote.selected_candidate_id != secondary_vote.selected_candidate_id
        or secondary_vote.confidence < JUDGE_CONFIDENCE_THRESHOLD
    ):
        return _result(
            accepted=False,
            level="unresolved",
            reason="independent semantic judges did not reach a high-confidence agreement",
            rejected_ids={*known_rejected, *invalid},
            votes=votes,
            ledger=ledger,
        )
    return _result(
        accepted=True,
        selected_candidate_id=primary_vote.selected_candidate_id,
        level="independent_judges",
        reason="independent semantic judges selected the same candidate",
        rejected_ids={*known_rejected, *invalid},
        votes=votes,
        ledger=ledger,
    )


def _execution_issue(
    candidate: Candidate,
    execution: ExecutionResult,
    *,
    guidelines: str,
) -> str | None:
    if execution.plan_fingerprint != analysis_plan_fingerprint(candidate.spec):
        return "execution plan fingerprint does not match the candidate spec"
    if set(execution.policy_ids) != set(candidate.spec.policy_ids):
        return "execution policy IDs do not match the candidate spec"
    try:
        expected = format_analysis_output(
            execution.raw_value,
            kind=candidate.spec.output.kind,
            decimals=candidate.spec.output.decimals,
            empty_string_allowed=candidate.spec.output.empty_string_allowed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return f"execution output could not be reconstructed deterministically: {exc}"
    if expected != execution.formatted_value:
        return "execution output violates deterministic format reconstruction"
    feedback = validate_output_contract(execution.formatted_value, parse_guidelines(guidelines))
    if feedback:
        return feedback
    try:
        ordering_valid = _ordering_is_valid(candidate, execution)
    except (KeyError, TypeError, ValueError):
        ordering_valid = False
    if not ordering_valid:
        return "grouped execution output violates candidate ordering"
    return None


def _ordering_is_valid(candidate: Candidate, execution: ExecutionResult) -> bool:
    spec = candidate.spec
    raw = execution.raw_value
    if not spec.group_by:
        return True
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        return False
    expected = list(raw)
    expected.sort(key=lambda item: tuple(str(item["group"][column]) for column in spec.group_by))
    for order in reversed(spec.ordering):
        if order.by == "value":
            key = lambda item: item["value"]
        else:
            key = lambda item, column=order.by: str(item["group"][column])
        expected.sort(key=key, reverse=order.direction == "desc")
    if spec.limit is not None:
        expected = expected[:spec.limit]
    return json.dumps(expected, sort_keys=True) == json.dumps(raw, sort_keys=True)


def _policy_issue(
    candidate: Candidate,
    registry: PolicyRegistry,
    *,
    policy_query: PolicyQuery | None,
) -> str | None:
    for policy_id in candidate.spec.policy_ids:
        policy = registry.get(policy_id)
        if policy is None:
            return f"candidate references unknown policy {policy_id}"
        if policy.status not in SELECTABLE_POLICY_STATUSES:
            return f"candidate policy {policy_id} is {policy.status.value}"
        if policy_query is not None:
            query = policy_query.model_copy(update={
                "statuses": sorted(SELECTABLE_POLICY_STATUSES, key=lambda status: status.value),
                "top_k": 20,
            })
            if not registry.matches(policy_id, query):
                return f"candidate policy {policy_id} is not applicable to the current query"
    return None


def _is_certified(candidate: Candidate, registry: PolicyRegistry) -> bool:
    if not candidate.spec.policy_ids:
        return False
    return all(
        (policy := registry.get(policy_id)) is not None
        and policy.status in CERTIFIED_POLICY_STATUSES
        for policy_id in candidate.spec.policy_ids
    )


def _judge_prompt(
    question: str,
    guidelines: str,
    candidates: list[Candidate],
    registry: PolicyRegistry,
) -> str:
    rows = []
    for candidate in candidates:
        policies = [registry.get(policy_id) for policy_id in candidate.spec.policy_ids]
        conventions = [
            f"{policy.axis}={policy.choice}: {policy.convention}"
            for policy in policies
            if policy is not None
        ]
        rows.append(
            f"- {candidate.candidate_id}: "
            + (" | ".join(conventions) or "no certified policy description")
            + f" | spec={_semantic_spec_summary(candidate)}"
        )
    return f"""\
Choose the candidate whose semantic interpretation best matches the wording.
Do not calculate values and do not use tools.

QUESTION: {question}
GUIDELINES: {guidelines or "N/A"}
CANDIDATES:
{chr(10).join(rows)}
"""


def _semantic_spec_summary(candidate: Candidate) -> str:
    payload = {
        "source": candidate.spec.source.model_dump(mode="json"),
        "time_scope": candidate.spec.time_scope.model_dump(mode="json"),
        "filters": [condition.model_dump(mode="json") for condition in candidate.spec.filters],
        "measure": candidate.spec.measure.model_dump(mode="json"),
        "group_by": candidate.spec.group_by,
        "ordering": [order.model_dump(mode="json") for order in candidate.spec.ordering],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))[:1200]


async def _run_judge(
    judge,
    *,
    stage: Literal["semantic_judge", "semantic_judge_secondary"],
    prompt: str,
    allowed_candidate_ids: set[str],
    ledger: UsageLedger,
) -> tuple[SemanticJudgeVote | None, str | None]:
    try:
        ledger.ensure_can_call(stage)
    except UsageBudgetExceeded as exc:
        return None, str(exc)
    started = time.perf_counter()
    try:
        result = await judge.run(prompt)
    except Exception as exc:  # noqa: BLE001 - verifier failure must fall back safely.
        ledger.record(CallUsage(
            stage=stage,
            input_tokens=0,
            output_tokens=0,
            latency_ms=round((time.perf_counter() - started) * 1000),
        ))
        return None, f"{stage} failed: {type(exc).__name__}: {exc}"
    usage = call_usage_from_result(
        result,
        stage=stage,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    ledger.record(usage)
    try:
        decision = JudgeDecision.model_validate(result.output)
    except ValidationError as exc:
        return None, f"{stage} returned invalid output: {exc.errors(include_url=False)[0]['msg']}"
    if decision.selected_candidate_id not in allowed_candidate_ids:
        return None, f"{stage} selected an unknown or rejected candidate"
    return SemanticJudgeVote(
        stage=stage,
        selected_candidate_id=decision.selected_candidate_id,
        confidence=decision.confidence,
        rationale=decision.rationale,
        model_fingerprint=usage.model_fingerprint,
    ), None


def _result(
    *,
    accepted: bool,
    level: str,
    reason: str,
    rejected_ids: set[str],
    ledger: UsageLedger,
    selected_candidate_id: str | None = None,
    votes: list[SemanticJudgeVote] | None = None,
) -> SemanticVerificationResult:
    return SemanticVerificationResult(
        accepted=accepted,
        selected_candidate_id=selected_candidate_id,
        level=level,
        reason=reason,
        rejected_candidate_ids=sorted(rejected_ids),
        judge_votes=votes or [],
        usage_trace=ledger.summary(),
    )
