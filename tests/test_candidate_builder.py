from __future__ import annotations

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.candidate_builder import build_candidate_set
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_planner import SemanticUncertainty
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy


HASH = "a" * 64


def _policy(
    policy_id: str,
    *,
    axis: str,
    choice: str,
    rivals=None,
    status: PolicyStatus = PolicyStatus.ACTIVE,
) -> SemanticPolicy:
    certified = status in {PolicyStatus.ACTIVE, PolicyStatus.CERTIFIED}
    return SemanticPolicy(
        policy_id=policy_id,
        version=1,
        family="customer_fraud_metrics",
        axis=axis,
        choice=choice,
        convention=f"Use {choice} for {axis}.",
        rival_policy_ids=rivals or [],
        citations=[PolicyCitation(
            document_name="manual.md",
            section="metrics",
            content_hash=HASH,
        )] if certified else [],
        certification_id=f"cert:{policy_id}" if certified else None,
        status=status,
    )


def _sum_spec(*, policy_ids=None) -> AnalysisSpec:
    return AnalysisSpec(
        source=SourceSpec(table="payments"),
        measure=AggregateMeasure(kind="sum", column="eur_amount"),
        output=AnalysisOutputContract(kind="decimal", decimals=2),
        policy_ids=policy_ids or [],
    )


def _uncertainty(axis: str, rivals: list[str]) -> SemanticUncertainty:
    return SemanticUncertainty(
        axis=axis,
        proposed_choice="base",
        rival_policy_ids=rivals,
        rationale="The wording leaves this semantic axis uncertain.",
        confidence=0.6,
    )


def test_candidate_builder_expands_uncertainty_through_certified_rivals():
    base = _policy(
        "aggregation.sum.v1",
        axis="aggregation",
        choice="sum",
        rivals=["aggregation.mean.v1"],
    )
    rival = _policy("aggregation.mean.v1", axis="aggregation", choice="mean")
    registry = PolicyRegistry([base, rival])

    candidates = build_candidate_set(
        _sum_spec(policy_ids=[base.policy_id]),
        uncertainties=[_uncertainty("aggregation", [rival.policy_id])],
        registry=registry,
    )

    assert len(candidates.candidates) == 2
    assert candidates.candidates[0].origin == "proposed"
    assert candidates.candidates[1].spec.measure.kind == "mean"
    assert candidates.candidates[1].applied_policy_ids == [rival.policy_id]


def test_explicit_uncertainty_ignores_uncertified_rivals():
    rival = _policy(
        "aggregation.mean.v1",
        axis="aggregation",
        choice="mean",
        status=PolicyStatus.PROPOSED,
    )
    registry = PolicyRegistry([rival])

    candidates = build_candidate_set(
        _sum_spec(),
        uncertainties=[_uncertainty("aggregation", [rival.policy_id])],
        registry=registry,
    )

    assert len(candidates.candidates) == 1
    assert candidates.skipped_policy_ids == [rival.policy_id]


def test_high_risk_axis_automatically_adds_known_rejected_rival():
    volume = _policy(
        "fraud_rate.basis.eur_volume.v1",
        axis="fraud_rate_basis",
        choice="eur_volume",
        rivals=["fraud_rate.basis.transaction_count.v1"],
    )
    count = _policy(
        "fraud_rate.basis.transaction_count.v1",
        axis="fraud_rate_basis",
        choice="transaction_count",
        status=PolicyStatus.PROPOSED,
    )
    spec = AnalysisSpec(
        source=SourceSpec(table="payments"),
        measure=RatioMeasure(
            numerator=AggregateMeasure(
                kind="sum",
                column="eur_amount",
                filters=[EqFilter(column="has_fraudulent_dispute", value=True)],
            ),
            denominator=AggregateMeasure(kind="sum", column="eur_amount"),
            scale=100,
        ),
        output=AnalysisOutputContract(kind="decimal", decimals=6),
        policy_ids=[volume.policy_id],
    )

    candidates = build_candidate_set(
        spec,
        uncertainties=[],
        registry=PolicyRegistry([volume, count]),
    )

    assert len(candidates.candidates) == 2
    sabotage = candidates.candidates[1]
    assert sabotage.origin == "high_risk_rival"
    assert sabotage.spec.measure.numerator.kind == "count"
    assert sabotage.spec.measure.denominator.kind == "count"


def test_structurally_and_extensionally_equivalent_candidates_collapse():
    exclude_a = _policy("missing.exclude-a.v1", axis="missing_value_policy", choice="exclude")
    exclude_b = _policy("missing.exclude-b.v1", axis="missing_value_policy", choice="exclude")
    mean = _policy("aggregation.mean.v1", axis="aggregation", choice="mean")
    registry = PolicyRegistry([exclude_a, exclude_b, mean])

    structural = build_candidate_set(
        _sum_spec(),
        uncertainties=[_uncertainty(
            "missing_value_policy",
            [exclude_b.policy_id, exclude_a.policy_id],
        )],
        registry=registry,
    )
    extensional = build_candidate_set(
        _sum_spec(),
        uncertainties=[_uncertainty("aggregation", [mean.policy_id])],
        registry=registry,
        equivalence_key=lambda spec: "same-output",
    )

    assert len(structural.candidates) == 1
    assert structural.collapsed_candidate_ids
    assert len(extensional.candidates) == 1
    assert extensional.collapsed_candidate_ids


def test_candidate_order_and_four_candidate_cap_are_deterministic():
    policies = [
        _policy("aggregation.mean.v1", axis="aggregation", choice="mean"),
        _policy("aggregation.count.v1", axis="aggregation", choice="count"),
        _policy("aggregation.nunique.v1", axis="aggregation", choice="nunique"),
        _policy("missing.include.v1", axis="missing_value_policy", choice="include"),
    ]
    registry = PolicyRegistry(list(reversed(policies)))
    uncertainties = [
        _uncertainty("aggregation", [policy.policy_id for policy in reversed(policies[:3])]),
        _uncertainty("missing_value_policy", [policies[3].policy_id]),
    ]

    first = build_candidate_set(
        _sum_spec(),
        uncertainties=uncertainties,
        registry=registry,
    )
    second = build_candidate_set(
        _sum_spec(),
        uncertainties=list(reversed(uncertainties)),
        registry=registry,
    )

    assert len(first.candidates) == 4
    assert first.truncated is True
    assert [candidate.candidate_id for candidate in first.candidates] == [
        candidate.candidate_id for candidate in second.candidates
    ]
