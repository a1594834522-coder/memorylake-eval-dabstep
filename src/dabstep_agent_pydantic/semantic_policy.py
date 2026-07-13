from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


HashValue = str
PredicateScalar = str | int | float | bool
PredicateValue = PredicateScalar | list[PredicateScalar]


class PolicyStatus(str, Enum):
    PROPOSED = "proposed"
    PROVISIONAL = "provisional"
    SHADOW = "shadow"
    CERTIFIED = "certified"
    ACTIVE = "active"
    STALE = "stale"
    REVOKED = "revoked"


class ApplicabilityPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: Literal[
        "family",
        "metric",
        "population",
        "parameter",
        "output_kind",
        "question_term",
    ]
    operator: Literal["eq", "neq", "in", "not_in", "contains", "exists", "gte", "lte"]
    value: PredicateValue

    @model_validator(mode="after")
    def _operator_matches_value(self) -> "ApplicabilityPredicate":
        if self.operator in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"operator {self.operator!r} requires a list value")
        if self.operator in {"eq", "neq", "contains", "gte", "lte"} and isinstance(self.value, list):
            raise ValueError(f"operator {self.operator!r} requires a scalar value")
        if self.operator == "exists" and not isinstance(self.value, bool):
            raise ValueError("operator 'exists' requires a boolean value")
        return self


class PolicyCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_name: str = Field(min_length=1)
    section: str = Field(min_length=1)
    content_hash: HashValue = Field(pattern=r"^[0-9a-f]{64}$")


_ALLOWED_TRANSITIONS: dict[PolicyStatus, frozenset[PolicyStatus]] = {
    PolicyStatus.PROPOSED: frozenset({PolicyStatus.PROVISIONAL, PolicyStatus.REVOKED}),
    PolicyStatus.PROVISIONAL: frozenset({PolicyStatus.SHADOW, PolicyStatus.REVOKED}),
    PolicyStatus.SHADOW: frozenset({PolicyStatus.CERTIFIED, PolicyStatus.REVOKED}),
    PolicyStatus.CERTIFIED: frozenset({PolicyStatus.ACTIVE, PolicyStatus.STALE, PolicyStatus.REVOKED}),
    PolicyStatus.ACTIVE: frozenset({PolicyStatus.STALE, PolicyStatus.REVOKED}),
    PolicyStatus.STALE: frozenset({PolicyStatus.PROVISIONAL, PolicyStatus.REVOKED}),
    PolicyStatus.REVOKED: frozenset(),
}


class SemanticPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    family: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    choice: str = Field(min_length=1)
    applicability: list[ApplicabilityPredicate] = Field(default_factory=list)
    exclusions: list[ApplicabilityPredicate] = Field(default_factory=list)
    convention: str = Field(min_length=1)
    rival_policy_ids: list[str] = Field(default_factory=list)
    citations: list[PolicyCitation] = Field(default_factory=list)
    certification_id: str | None = None
    status: PolicyStatus = PolicyStatus.PROPOSED

    @field_validator("rival_policy_ids")
    @classmethod
    def _rival_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("rival_policy_ids must be unique")
        return values

    @field_validator("citations")
    @classmethod
    def _citation_hashes_are_unique(cls, values: list[PolicyCitation]) -> list[PolicyCitation]:
        hashes = [citation.content_hash for citation in values]
        if len(hashes) != len(set(hashes)):
            raise ValueError("citation content_hash values must be unique")
        return values

    @model_validator(mode="after")
    def _certification_requirements(self) -> "SemanticPolicy":
        if self.policy_id in self.rival_policy_ids:
            raise ValueError("a policy cannot rival itself")
        if self.status in {PolicyStatus.CERTIFIED, PolicyStatus.ACTIVE}:
            if not self.certification_id:
                raise ValueError("certification_id is required for certified or active policies")
            if not self.citations:
                raise ValueError("citations are required for certified or active policies")
        return self

    def transition_to(self, status: PolicyStatus) -> "SemanticPolicy":
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid policy transition: {self.status.value} -> {status.value}")
        return type(self).model_validate({**self.model_dump(), "status": status})
