from __future__ import annotations

import json

import pytest

from dabstep_agent_pydantic.policy_registry import PolicyQuery
from dabstep_agent_pydantic.policy_registry import PolicyRegistry
from dabstep_agent_pydantic.semantic_policy import ApplicabilityPredicate
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy


HASH = "a" * 64


def _policy(
    policy_id: str,
    *,
    family: str = "customer_fraud_metrics",
    axis: str = "fraud_rate_basis",
    choice: str = "eur_volume",
    convention: str = "Use fraudulent EUR volume divided by total EUR volume.",
    applicability=None,
    exclusions=None,
    status: PolicyStatus = PolicyStatus.ACTIVE,
) -> SemanticPolicy:
    certified = status in {PolicyStatus.CERTIFIED, PolicyStatus.ACTIVE}
    return SemanticPolicy(
        policy_id=policy_id,
        version=1,
        family=family,
        axis=axis,
        choice=choice,
        convention=convention,
        applicability=applicability or [],
        exclusions=exclusions or [],
        rival_policy_ids=[],
        citations=[PolicyCitation(
            document_name="manual.md",
            section="metrics",
            content_hash=HASH,
        )] if certified else [],
        certification_id=f"cert:{policy_id}" if certified else None,
        status=status,
    )


def test_registry_loads_local_snapshot_and_rejects_duplicate_ids(tmp_path):
    policy = _policy("fraud_rate.basis.eur_volume.v1")
    path = tmp_path / "policies.json"
    path.write_text(json.dumps({"policies": [policy.model_dump(mode="json")]}), encoding="utf-8")

    registry = PolicyRegistry.from_path(path)

    assert registry.get(policy.policy_id) == policy

    with pytest.raises(ValueError, match="duplicate policy_id"):
        PolicyRegistry([policy, policy])


def test_search_ranks_family_axis_and_terms_deterministically():
    exact = _policy("fraud_rate.basis.eur_volume.v1")
    related = _policy(
        "fraud_rate.missing.exclude.v1",
        axis="missing_value_policy",
        choice="exclude",
        convention="Exclude missing payment values before computing fraud metrics.",
    )
    other = _policy(
        "fee_rule.empty_field.wildcard.v1",
        family="fee_matching",
        axis="empty_fee_field",
        choice="wildcard",
        convention="Treat null and empty fee fields as wildcard matches.",
    )
    registry = PolicyRegistry([related, other, exact])

    matches = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        axis="fraud_rate_basis",
        terms=["fraud", "EUR", "volume"],
        top_k=3,
    ))

    assert [match.policy.policy_id for match in matches] == [
        exact.policy_id,
        related.policy_id,
    ]
    assert matches[0].score > matches[1].score


def test_search_applies_parameter_schema_and_exclusions():
    email_policy = _policy(
        "email_metrics.missing_email.exclude.v1",
        axis="missing_email",
        choice="exclude",
        convention="Exclude null email addresses from unique-email metrics.",
        applicability=[ApplicabilityPredicate(
            field="parameter",
            operator="eq",
            value="email_address",
        )],
        exclusions=[ApplicabilityPredicate(
            field="metric",
            operator="eq",
            value="merchant_count",
        )],
    )
    registry = PolicyRegistry([email_policy])

    matched = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        metric="unique_email_count",
        parameters=["email_address"],
    ))
    missing_parameter = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        metric="unique_email_count",
        parameters=["merchant"],
    ))
    excluded = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        metric="merchant_count",
        parameters=["email_address"],
    ))

    assert [match.policy for match in matched] == [email_policy]
    assert missing_parameter == []
    assert excluded == []


def test_search_excludes_uncertified_and_stale_policies_by_default():
    active = _policy("active.v1")
    proposed = _policy("proposed.v1", status=PolicyStatus.PROPOSED)
    stale = _policy("stale.v1", status=PolicyStatus.STALE)
    registry = PolicyRegistry([stale, proposed, active])

    default_matches = registry.search(PolicyQuery(family="customer_fraud_metrics"))
    expanded_matches = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        statuses=[PolicyStatus.ACTIVE, PolicyStatus.PROPOSED, PolicyStatus.STALE],
    ))

    assert [match.policy.policy_id for match in default_matches] == [active.policy_id]
    assert {match.policy.policy_id for match in expanded_matches} == {
        active.policy_id,
        proposed.policy_id,
        stale.policy_id,
    }


def test_search_respects_top_k_and_policy_id_tie_break():
    policies = [
        _policy(f"policy-{suffix}.v1", axis="shared_axis", choice="same")
        for suffix in ("c", "a", "b")
    ]
    registry = PolicyRegistry(policies)

    matches = registry.search(PolicyQuery(
        family="customer_fraud_metrics",
        axis="shared_axis",
        top_k=2,
    ))

    assert [match.policy.policy_id for match in matches] == ["policy-a.v1", "policy-b.v1"]
