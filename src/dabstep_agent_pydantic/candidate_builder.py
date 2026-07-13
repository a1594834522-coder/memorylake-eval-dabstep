from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import InterpretationMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_planner import SemanticUncertainty
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy


MAX_CANDIDATES = 4
CERTIFIED_STATUSES = {PolicyStatus.ACTIVE, PolicyStatus.CERTIFIED}
HIGH_RISK_AXIS_SCORES = {
    "fraud_rate_basis": 100,
    "empty_fee_field": 95,
    "wildcard_policy": 95,
    "fee_rule_reducer": 90,
    "missing_email": 85,
    "missing_value_policy": 80,
    "repeat_customer_scope": 75,
    "aggregation": 50,
}


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    spec: AnalysisSpec
    origin: Literal["proposed", "uncertainty_rival", "high_risk_rival"]
    applied_policy_ids: list[str] = Field(default_factory=list)
    uncertainty_axes: list[str] = Field(default_factory=list)
    risk_score: int = Field(default=0, ge=0)


class CandidateSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: list[Candidate]
    collapsed_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)
    skipped_policy_ids: list[str] = Field(default_factory=list)
    truncated: bool = False


class UnsupportedPolicyAxis(ValueError):
    pass


def build_candidate_set(
    proposed_spec: AnalysisSpec,
    *,
    uncertainties: list[SemanticUncertainty],
    registry: PolicyRegistry,
    equivalence_key: Callable[[AnalysisSpec], str] | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> CandidateSet:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    base = Candidate(
        candidate_id=_candidate_id(proposed_spec, origin="proposed", policy_id=None),
        spec=proposed_spec,
        origin="proposed",
        applied_policy_ids=list(proposed_spec.policy_ids),
    )
    requests: dict[str, tuple[str, str, int]] = {}
    skipped: set[str] = set()

    for uncertainty in sorted(uncertainties, key=lambda item: item.axis):
        risk_score = HIGH_RISK_AXIS_SCORES.get(uncertainty.axis, 40)
        for policy_id in sorted(uncertainty.rival_policy_ids):
            policy = registry.get(policy_id)
            if policy is None or policy.status not in CERTIFIED_STATUSES:
                skipped.add(policy_id)
                continue
            requests[policy_id] = ("uncertainty_rival", uncertainty.axis, risk_score)

    explicit_ids = set(requests)
    for base_policy_id in sorted(proposed_spec.policy_ids):
        base_policy = registry.get(base_policy_id)
        if base_policy is None or base_policy.axis not in HIGH_RISK_AXIS_SCORES:
            continue
        for policy_id in sorted(base_policy.rival_policy_ids):
            if policy_id in explicit_ids:
                continue
            policy = registry.get(policy_id)
            if policy is None or policy.status in {PolicyStatus.STALE, PolicyStatus.REVOKED}:
                skipped.add(policy_id)
                continue
            requests[policy_id] = (
                "high_risk_rival",
                base_policy.axis,
                HIGH_RISK_AXIS_SCORES[base_policy.axis],
            )

    variants: list[Candidate] = [base]
    ordered_requests = sorted(
        requests.items(),
        key=lambda item: (-item[1][2], item[1][0], item[1][1], item[0]),
    )
    for policy_id, (origin, axis, risk_score) in ordered_requests:
        policy = registry.get(policy_id)
        assert policy is not None
        try:
            spec = apply_policy_choice(proposed_spec, policy, registry=registry)
        except (UnsupportedPolicyAxis, ValueError):
            skipped.add(policy_id)
            continue
        variants.append(Candidate(
            candidate_id=_candidate_id(spec, origin=origin, policy_id=policy_id),
            spec=spec,
            origin=origin,
            applied_policy_ids=[policy_id],
            uncertainty_axes=[axis],
            risk_score=risk_score,
        ))

    collapsed: dict[str, list[str]] = {}
    unique: list[Candidate] = []
    structural_keys: dict[str, Candidate] = {}
    extensional_keys: dict[str, Candidate] = {}
    for candidate in variants:
        structural_key = _behavior_fingerprint(candidate.spec)
        kept = structural_keys.get(structural_key)
        if kept is None and equivalence_key is not None:
            output_key = str(equivalence_key(candidate.spec))
            kept = extensional_keys.get(output_key)
        else:
            output_key = None
        if kept is not None:
            collapsed.setdefault(kept.candidate_id, []).append(candidate.candidate_id)
            continue
        unique.append(candidate)
        structural_keys[structural_key] = candidate
        if equivalence_key is not None:
            if output_key is None:
                output_key = str(equivalence_key(candidate.spec))
            extensional_keys[output_key] = candidate

    truncated = len(unique) > max_candidates
    selected = unique[:max_candidates]
    selected_ids = {candidate.candidate_id for candidate in selected}
    return CandidateSet(
        candidates=selected,
        collapsed_candidate_ids={
            candidate_id: sorted(ids)
            for candidate_id, ids in sorted(collapsed.items())
            if candidate_id in selected_ids
        },
        skipped_policy_ids=sorted(skipped),
        truncated=truncated,
    )


def apply_policy_choice(
    spec: AnalysisSpec,
    policy: SemanticPolicy,
    *,
    registry: PolicyRegistry,
) -> AnalysisSpec:
    measure = _apply_choice_to_measure(spec.measure, policy)
    retained_policy_ids = [
        policy_id
        for policy_id in spec.policy_ids
        if (existing := registry.get(policy_id)) is None or existing.axis != policy.axis
    ]
    payload = spec.model_dump(mode="json")
    payload["measure"] = measure.model_dump(mode="json")
    payload["policy_ids"] = sorted({*retained_policy_ids, policy.policy_id})
    return AnalysisSpec.model_validate(payload)


def _apply_choice_to_measure(measure, policy: SemanticPolicy):
    axis = policy.axis
    choice = policy.choice

    if axis in {"missing_value_policy", "missing_email"}:
        if choice not in {"exclude", "include", "error"}:
            raise UnsupportedPolicyAxis(f"unsupported missing-value choice: {choice}")
        return _map_aggregates(measure, lambda aggregate: aggregate.model_copy(
            update={"missing": choice},
        ))

    if axis in {"aggregation", "reducer"} and isinstance(measure, AggregateMeasure):
        if choice not in {"sum", "mean", "count", "nunique"}:
            raise UnsupportedPolicyAxis(f"unsupported aggregate choice: {choice}")
        column = None if choice == "count" else measure.column
        return AggregateMeasure.model_validate({
            **measure.model_dump(mode="json"),
            "kind": choice,
            "column": column,
        })

    if axis == "fraud_rate_basis" and isinstance(measure, RatioMeasure):
        if choice == "eur_volume":
            return _ratio_basis(measure, kind="sum", column="eur_amount")
        if choice == "transaction_count":
            return _ratio_basis(measure, kind="count", column=None)
        raise UnsupportedPolicyAxis(f"unsupported fraud-rate choice: {choice}")

    if isinstance(measure, InterpretationMeasure):
        interpretation = _apply_interpretation_choice(measure.interpretation, policy)
        return InterpretationMeasure(
            interpretation=interpretation,
            params=dict(measure.params),
        )

    raise UnsupportedPolicyAxis(f"unsupported policy axis for measure: {axis}")


def _map_aggregates(measure, transform: Callable[[AggregateMeasure], AggregateMeasure]):
    if isinstance(measure, AggregateMeasure):
        return transform(measure)
    if isinstance(measure, RatioMeasure):
        return RatioMeasure.model_validate({
            **measure.model_dump(mode="json"),
            "numerator": transform(measure.numerator).model_dump(mode="json"),
            "denominator": transform(measure.denominator).model_dump(mode="json"),
        })
    raise UnsupportedPolicyAxis("missing-value policy requires aggregate measures")


def _ratio_basis(measure: RatioMeasure, *, kind: str, column: str | None) -> RatioMeasure:
    numerator = AggregateMeasure.model_validate({
        **measure.numerator.model_dump(mode="json"),
        "kind": kind,
        "column": column,
    })
    denominator = AggregateMeasure.model_validate({
        **measure.denominator.model_dump(mode="json"),
        "kind": kind,
        "column": column,
    })
    return RatioMeasure.model_validate({
        **measure.model_dump(mode="json"),
        "numerator": numerator.model_dump(mode="json"),
        "denominator": denominator.model_dump(mode="json"),
    })


def _apply_interpretation_choice(
    interpretation: InterpretationSpec,
    policy: SemanticPolicy,
) -> InterpretationSpec:
    axis = policy.axis
    choice = policy.choice
    payload = interpretation.model_dump(mode="json")
    if interpretation.fee_rules is not None:
        fee_rules = interpretation.fee_rules.model_dump(mode="json")
        if axis in {"empty_fee_field", "wildcard_policy"}:
            fee_rules["wildcard_policy"] = "manual" if choice in {"manual", "wildcard"} else choice
        elif axis in {"fee_rule_reducer", "aggregation", "reducer"}:
            fee_rules["reducer"] = choice
        else:
            raise UnsupportedPolicyAxis(f"unsupported fee-rule axis: {axis}")
        payload["fee_rules"] = fee_rules
    elif interpretation.payments is not None:
        payments = interpretation.payments.model_dump(mode="json")
        field_by_axis = {
            "delta_basis": "delta_basis",
            "tuple_scope": "tuple_scope",
            "aci_candidate_policy": "aci_candidate_policy",
            "affected_mode": "affected_mode",
            "payments_reducer": "reducer",
        }
        field = field_by_axis.get(axis)
        if field is None:
            raise UnsupportedPolicyAxis(f"unsupported payments axis: {axis}")
        payments[field] = choice
        payload["payments"] = payments
    return InterpretationSpec.model_validate(payload)


def _behavior_fingerprint(spec: AnalysisSpec) -> str:
    payload = spec.model_dump(mode="json")
    payload.pop("spec_id", None)
    payload.pop("policy_ids", None)
    payload.pop("unresolved_axes", None)
    return _stable_hash(payload)


def _candidate_id(spec: AnalysisSpec, *, origin: str, policy_id: str | None) -> str:
    spec_payload = spec.model_dump(mode="json")
    spec_payload["policy_ids"] = sorted(spec_payload.get("policy_ids") or [])
    spec_payload["unresolved_axes"] = sorted(spec_payload.get("unresolved_axes") or [])
    return "candidate_" + _stable_hash({
        "spec": spec_payload,
        "origin": origin,
        "policy_id": policy_id,
    })[:12]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
