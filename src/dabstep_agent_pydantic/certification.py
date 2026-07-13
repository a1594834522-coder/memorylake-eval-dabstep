from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class CostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class JudgeVote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)
    selected_candidate_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class WitnessEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    witness_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_values: dict[str, str] = Field(min_length=2)
    separating_candidate_ids: list[str] = Field(min_length=2)

    @field_validator("separating_candidate_ids")
    @classmethod
    def _separating_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("separating_candidate_ids must be unique")
        return values

    @model_validator(mode="after")
    def _separating_ids_have_values(self) -> "WitnessEvidence":
        missing = set(self.separating_candidate_ids) - set(self.candidate_values)
        if missing:
            raise ValueError(f"missing candidate_values for {sorted(missing)}")
        return self


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_votes: list[JudgeVote] = Field(default_factory=list)
    witnesses: list[WitnessEvidence] = Field(default_factory=list)
    citation_hashes: list[str] = Field(default_factory=list)
    sibling_policy_ids: list[str] = Field(default_factory=list)

    @field_validator("citation_hashes")
    @classmethod
    def _citation_hashes_are_valid(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("citation_hashes must be unique")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
            raise ValueError("citation_hashes must contain lowercase sha256 values")
        return values

    @property
    def cost_summary(self) -> CostSummary:
        return CostSummary(
            calls=len(self.judge_votes),
            input_tokens=sum(vote.input_tokens for vote in self.judge_votes),
            output_tokens=sum(vote.output_tokens for vote in self.judge_votes),
            latency_ms=sum(vote.latency_ms for vote in self.judge_votes),
        )


class CertificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    certification_id: str = Field(min_length=1)
    policy_ids: list[str] = Field(min_length=1)
    decision: Literal["certified", "rejected"]
    selected_candidate_id: str | None = None
    gate_outcomes: dict[str, bool] = Field(default_factory=dict)
    candidate_evidence: list[CandidateEvidence] = Field(default_factory=list)
    document_fingerprints: dict[str, str] = Field(default_factory=dict)
    mechanism_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    rejection_reasons: list[str] = Field(default_factory=list)

    @field_validator("policy_ids")
    @classmethod
    def _policy_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("policy_ids must be unique")
        return values

    @field_validator("document_fingerprints")
    @classmethod
    def _document_hashes_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        invalid = [
            name
            for name, value in values.items()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ]
        if invalid:
            raise ValueError(f"invalid document fingerprints: {sorted(invalid)}")
        return values

    @model_validator(mode="after")
    def _certification_is_evidence_backed(self) -> "CertificationRecord":
        candidate_ids = [evidence.candidate_id for evidence in self.candidate_evidence]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        if self.decision == "certified":
            if not self.selected_candidate_id or self.selected_candidate_id not in candidate_ids:
                raise ValueError("selected_candidate_id must identify candidate_evidence")
            if not self.gate_outcomes or not all(self.gate_outcomes.values()):
                raise ValueError("all certification gates must pass")
        return self

    @property
    def cost_summary(self) -> CostSummary:
        costs = [evidence.cost_summary for evidence in self.candidate_evidence]
        return CostSummary(
            calls=sum(cost.calls for cost in costs),
            input_tokens=sum(cost.input_tokens for cost in costs),
            output_tokens=sum(cost.output_tokens for cost in costs),
            latency_ms=sum(cost.latency_ms for cost in costs),
        )
