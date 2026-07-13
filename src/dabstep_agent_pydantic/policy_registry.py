from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from dabstep_agent_pydantic.semantic_policy import ApplicabilityPredicate
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy


class PolicyQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str = Field(min_length=1)
    axis: str | None = None
    metric: str | None = None
    population: str | None = None
    output_kind: str | None = None
    terms: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    statuses: list[PolicyStatus] = Field(
        default_factory=lambda: [PolicyStatus.ACTIVE, PolicyStatus.CERTIFIED]
    )
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("terms", "parameters", "statuses")
    @classmethod
    def _values_are_unique(cls, values, info):
        if len(values) != len(set(values)):
            raise ValueError(f"{info.field_name} must contain unique values")
        return values


class PolicyMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: SemanticPolicy
    score: int = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)


class PolicyRegistry:
    def __init__(self, policies: list[SemanticPolicy]):
        policy_ids = [policy.policy_id for policy in policies]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("duplicate policy_id in policy registry")
        self._policies = tuple(sorted(policies, key=lambda policy: policy.policy_id))
        self._by_id = {policy.policy_id: policy for policy in self._policies}

    @classmethod
    def from_path(cls, path: Path) -> "PolicyRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("policies") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("policy snapshot must be a list or contain a policies list")
        return cls([SemanticPolicy.model_validate(row) for row in rows])

    def get(self, policy_id: str) -> SemanticPolicy | None:
        return self._by_id.get(policy_id)

    def search(self, query: PolicyQuery) -> list[PolicyMatch]:
        query_terms = _normalized_terms(query.terms)
        matches: list[PolicyMatch] = []
        for policy in self._policies:
            if not self.matches(policy.policy_id, query):
                continue

            policy_terms = _policy_terms(policy)
            matched_terms = sorted(query_terms & policy_terms)
            score = 100
            if query.axis and policy.axis == query.axis:
                score += 30
            score += 3 * len(matched_terms)
            score += sum(
                2
                for predicate in policy.applicability
                if predicate.field == "parameter"
            )
            matches.append(PolicyMatch(
                policy=policy,
                score=score,
                matched_terms=matched_terms,
            ))
        matches.sort(key=lambda match: (-match.score, match.policy.policy_id))
        return matches[:query.top_k]

    def matches(self, policy_id: str, query: PolicyQuery) -> bool:
        policy = self.get(policy_id)
        if policy is None or policy.family != query.family or policy.status not in query.statuses:
            return False
        context = {
            "family": query.family,
            "metric": query.metric,
            "population": query.population,
            "output_kind": query.output_kind,
            "parameter": query.parameters,
            "question_term": query.terms,
        }
        if not all(_predicate_matches(predicate, context) for predicate in policy.applicability):
            return False
        return not any(_predicate_matches(predicate, context) for predicate in policy.exclusions)


def _normalized_terms(terms: list[str]) -> set[str]:
    return {
        token
        for term in terms
        for token in re.findall(r"[a-z0-9_]+", term.lower())
    }


def _policy_terms(policy: SemanticPolicy) -> set[str]:
    return _normalized_terms([
        policy.policy_id,
        policy.axis,
        policy.choice,
        policy.convention,
    ])


def _predicate_matches(predicate: ApplicabilityPredicate, context: dict[str, Any]) -> bool:
    actual = context.get(predicate.field)
    expected = predicate.value
    if predicate.operator == "exists":
        return bool(actual) is bool(expected)

    if predicate.operator in {"eq", "neq"}:
        matched = _equals(actual, expected)
        return matched if predicate.operator == "eq" else not matched

    if predicate.operator in {"in", "not_in"}:
        assert isinstance(expected, list)
        if isinstance(actual, (list, tuple, set)):
            matched = any(item in expected for item in actual)
        else:
            matched = actual in expected
        return matched if predicate.operator == "in" else not matched

    if predicate.operator == "contains":
        if isinstance(actual, str):
            return str(expected).lower() in actual.lower()
        if isinstance(actual, (list, tuple, set)):
            return expected in actual
        return False

    if actual is None or isinstance(actual, (list, tuple, set)):
        return False
    try:
        if predicate.operator == "gte":
            return actual >= expected
        if predicate.operator == "lte":
            return actual <= expected
    except TypeError:
        return False
    raise ValueError(f"unsupported predicate operator: {predicate.operator}")


def _equals(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (list, tuple, set)):
        return expected in actual
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.lower() == expected.lower()
    return actual == expected
