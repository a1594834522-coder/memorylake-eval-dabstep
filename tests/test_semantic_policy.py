from __future__ import annotations

import pytest
from pydantic import ValidationError

from dabstep_agent_pydantic.certification import CandidateEvidence
from dabstep_agent_pydantic.certification import CertificationRecord
from dabstep_agent_pydantic.certification import JudgeVote
from dabstep_agent_pydantic.certification import WitnessEvidence
from dabstep_agent_pydantic.semantic_policy import ApplicabilityPredicate
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _predicate() -> ApplicabilityPredicate:
    return ApplicabilityPredicate(field="metric", operator="eq", value="fraud_rate")


def _citation() -> PolicyCitation:
    return PolicyCitation(
        document_name="manual.md",
        section="Fraud metrics",
        content_hash=HASH_A,
    )


def _policy(**overrides) -> SemanticPolicy:
    values = {
        "policy_id": "fraud_rate.basis.eur_volume.v1",
        "version": 1,
        "family": "customer_fraud_metrics",
        "axis": "fraud_rate_basis",
        "choice": "eur_volume",
        "applicability": [_predicate()],
        "exclusions": [],
        "convention": "Fraud rate is fraudulent EUR volume divided by total EUR volume.",
        "rival_policy_ids": ["fraud_rate.basis.transaction_count.v1"],
        "citations": [],
        "status": PolicyStatus.PROPOSED,
    }
    values.update(overrides)
    return SemanticPolicy(**values)


def _candidate(candidate_id: str = "candidate-eur") -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        spec_fingerprint=HASH_B,
        judge_votes=[
            JudgeVote(
                judge_id="judge-primary",
                model_fingerprint="provider:model-a",
                selected_candidate_id=candidate_id,
                confidence=0.9,
                rationale="The manual defines a volume ratio.",
                input_tokens=100,
                output_tokens=20,
                latency_ms=50,
            )
        ],
        witnesses=[
            WitnessEvidence(
                witness_fingerprint=HASH_C,
                candidate_values={candidate_id: "0.2308", "candidate-count": "0.3333"},
                separating_candidate_ids=[candidate_id, "candidate-count"],
            )
        ],
        citation_hashes=[HASH_A],
    )


def test_policy_rejects_duplicate_and_self_rivals():
    with pytest.raises(ValidationError, match="rival_policy_ids"):
        _policy(rival_policy_ids=[
            "fraud_rate.basis.transaction_count.v1",
            "fraud_rate.basis.transaction_count.v1",
        ])

    with pytest.raises(ValidationError, match="cannot rival itself"):
        _policy(rival_policy_ids=["fraud_rate.basis.eur_volume.v1"])


def test_certified_policy_requires_certification_and_hashed_citation():
    with pytest.raises(ValidationError, match="certification_id"):
        _policy(status=PolicyStatus.CERTIFIED, citations=[_citation()])

    with pytest.raises(ValidationError, match="citations"):
        _policy(status=PolicyStatus.CERTIFIED, certification_id="cert-fraud-v1")

    policy = _policy(
        status=PolicyStatus.CERTIFIED,
        certification_id="cert-fraud-v1",
        citations=[_citation()],
    )
    assert policy.citations[0].content_hash == HASH_A

    with pytest.raises(ValidationError, match="content_hash"):
        PolicyCitation(document_name="manual.md", section="5", content_hash="not-a-hash")


def test_policy_lifecycle_transitions_are_explicit_and_non_mutating():
    proposed = _policy()
    provisional = proposed.transition_to(PolicyStatus.PROVISIONAL)

    assert proposed.status is PolicyStatus.PROPOSED
    assert provisional.status is PolicyStatus.PROVISIONAL
    assert provisional.transition_to(PolicyStatus.SHADOW).status is PolicyStatus.SHADOW

    with pytest.raises(ValueError, match="invalid policy transition"):
        proposed.transition_to(PolicyStatus.ACTIVE)


def test_candidate_evidence_aggregates_model_costs():
    evidence = _candidate()

    assert evidence.cost_summary.calls == 1
    assert evidence.cost_summary.input_tokens == 100
    assert evidence.cost_summary.output_tokens == 20
    assert evidence.cost_summary.latency_ms == 50


def test_certification_requires_unique_evidence_and_passing_gates():
    candidate = _candidate()
    with pytest.raises(ValidationError, match="candidate_id values must be unique"):
        CertificationRecord(
            certification_id="cert-fraud-v1",
            policy_ids=["fraud_rate.basis.eur_volume.v1"],
            decision="certified",
            selected_candidate_id=candidate.candidate_id,
            gate_outcomes={"citations": True, "coverage": True},
            candidate_evidence=[candidate, candidate],
            document_fingerprints={"manual.md": HASH_A},
            mechanism_fingerprint=HASH_B,
        )

    with pytest.raises(ValidationError, match="all certification gates"):
        CertificationRecord(
            certification_id="cert-fraud-v1",
            policy_ids=["fraud_rate.basis.eur_volume.v1"],
            decision="certified",
            selected_candidate_id=candidate.candidate_id,
            gate_outcomes={"citations": True, "coverage": False},
            candidate_evidence=[candidate],
            document_fingerprints={"manual.md": HASH_A},
            mechanism_fingerprint=HASH_B,
        )


def test_certification_cost_summary_and_selected_candidate_validation():
    candidate = _candidate()
    record = CertificationRecord(
        certification_id="cert-fraud-v1",
        policy_ids=["fraud_rate.basis.eur_volume.v1"],
        decision="certified",
        selected_candidate_id=candidate.candidate_id,
        gate_outcomes={"citations": True, "coverage": True},
        candidate_evidence=[candidate],
        document_fingerprints={"manual.md": HASH_A},
        mechanism_fingerprint=HASH_B,
    )

    assert record.cost_summary == candidate.cost_summary

    with pytest.raises(ValidationError, match="selected_candidate_id"):
        record.model_copy(update={"selected_candidate_id": "missing"}).model_validate(
            {**record.model_dump(), "selected_candidate_id": "missing"}
        )


def test_policy_and_certification_forbid_task_answers_and_unknown_fields():
    with pytest.raises(ValidationError, match="task_id"):
        _policy(task_id="123")

    candidate = _candidate()
    with pytest.raises(ValidationError, match="answer"):
        CertificationRecord(
            certification_id="cert-fraud-v1",
            policy_ids=["fraud_rate.basis.eur_volume.v1"],
            decision="certified",
            selected_candidate_id=candidate.candidate_id,
            gate_outcomes={"citations": True},
            candidate_evidence=[candidate],
            document_fingerprints={"manual.md": HASH_A},
            mechanism_fingerprint=HASH_B,
            answer="0.2308",
        )
