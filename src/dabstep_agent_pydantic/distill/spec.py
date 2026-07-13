"""Restricted DSL for interpretation candidates.

A spec is the pipeline's intermediate representation between "teacher-proposed
hypothesis" and "executable skill". The vocabulary is deliberately small:

- The *fee-rules family* (`population="fee_rules"`) is fully combinatorial:
  a match context built from question parameters, a per-rule value, and a
  reducer. It expresses fee-ID collection, hypothetical-fee averaging, and
  grouped extremes.
- The *payments family* delegates to domain primitives (period totals,
  counterfactual deltas, affected merchants, applicable IDs, ACI steering)
  whose interpretation axes surface as primitive parameters, so rejected
  candidates are expressible too (e.g. ``reducer="min_match"``).

Anything outside this vocabulary is not a valid candidate: the pipeline then
leaves the template to the LLM path instead of generating a skill.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


# Dimensions a question parameter may bind into the fee-rule match context.
ContextDim = Literal[
    "account_type",
    "aci",
    "card_scheme",
    "is_credit",
    "merchant_category_code",
    "capture_delay",
    "monthly_volume",
    "monthly_fraud_rate_pct",
    "intracountry",
]

RuleValue = Literal[
    "fee_at_amount",  # calculate_fee(fixed, rate, params["amount"])
    "rule_id",
]

RuleReducer = Literal[
    "mean",
    "sum",
    "min",
    "max",
    "collect_ids",
]

GroupExtreme = Literal["argmax", "argmin"]

PaymentsPrimitive = Literal[
    "period_total_fees",          # total fees paid by merchant over period
    "period_fee_rate_delta",      # counterfactual: rule rate changed to X
    "mcc_change_fee_delta",       # counterfactual: merchant MCC changed
    "affected_merchants",         # merchants matched by a fee rule over a year
    "applicable_fee_ids_period",  # fee IDs applicable to merchant in period
    "steer_optimal_aci",          # lowest-fee ACI for fraudulent transactions
    "field_domain_values",
    "unique_merchant_count",
    "repeat_customer_percentage",
]

PaymentsReducer = Literal[
    "sum_all_matching",  # canonical: every matching rule contributes
    "min_match",         # rejected candidate: cheapest matching rule only
    "first_match",       # rejected candidate: first matching rule only
]

AffectedMode = Literal["losers_only", "symmetric_difference", "baseline_members"]
AciCandidatePolicy = Literal["exclude_current", "include_all"]
DeltaBasis = Literal["rate", "fixed_component"]

OutputKind = Literal["decimal", "id_list", "string_list", "single_string", "integer"]


class OutputSpec(BaseModel):
    kind: OutputKind
    # decimal places: None -> take from guidelines via _decimal_places(default)
    decimals_default: int | None = None
    tie_policy: Literal["list_all_sorted", "first_alphabetical"] | None = None


class FeeRulesSpec(BaseModel):
    """Fully combinatorial family over fees.json rows."""

    context_dims: list[ContextDim] = Field(default_factory=list)
    extra_context: dict[str, object] = Field(default_factory=dict)
    value: RuleValue = "fee_at_amount"
    reducer: RuleReducer = "mean"
    # "manual": null / empty-list rule fields are wildcards (manual §5).
    # "strict": rejected reading — a rule matches only by explicit membership.
    wildcard_policy: Literal["manual", "strict"] = "manual"
    # Optional grouped-extreme stage: group rules by a dimension (wildcard
    # rules join every group), reduce within groups, then pick the extreme.
    group_by: Literal["merchant_category_code", "aci", "card_scheme"] | None = None
    group_wildcard_expansion: bool = False
    group_extreme: GroupExtreme | None = None

    @model_validator(mode="after")
    def _grouping_is_consistent(self) -> "FeeRulesSpec":
        if (self.group_by is None) != (self.group_extreme is None):
            raise ValueError("group_by and group_extreme must be set together")
        if self.group_by is not None and self.reducer == "collect_ids":
            raise ValueError("collect_ids cannot be combined with grouping")
        return self


class PaymentsSpec(BaseModel):
    """Domain-primitive family over payments/merchant context."""

    primitive: PaymentsPrimitive
    reducer: PaymentsReducer = "sum_all_matching"
    affected_mode: AffectedMode | None = None
    aci_candidate_policy: AciCandidatePolicy | None = None
    delta_basis: DeltaBasis | None = None
    tuple_scope: Literal["full_period", "sampled_first_day"] | None = None


class InterpretationSpec(BaseModel):
    name: str = Field(min_length=1)
    population: Literal["fee_rules", "payments"]
    fee_rules: FeeRulesSpec | None = None
    payments: PaymentsSpec | None = None
    output: OutputSpec
    manual_citation: str = Field(min_length=1)
    contradicts_manual: bool = False

    @model_validator(mode="after")
    def _family_payload_present(self) -> "InterpretationSpec":
        if self.population == "fee_rules" and self.fee_rules is None:
            raise ValueError("population=fee_rules requires fee_rules payload")
        if self.population == "payments" and self.payments is None:
            raise ValueError("population=payments requires payments payload")
        return self

    @property
    def axis_summary(self) -> str:
        if self.population == "fee_rules":
            assert self.fee_rules is not None
            parts = [f"reducer={self.fee_rules.reducer}", f"dims={','.join(self.fee_rules.context_dims) or '-'}"]
            if self.fee_rules.wildcard_policy != "manual":
                parts.append(f"wildcard={self.fee_rules.wildcard_policy}")
            if self.fee_rules.group_by:
                parts.append(f"group={self.fee_rules.group_by}:{self.fee_rules.group_extreme}")
            return " ".join(parts)
        assert self.payments is not None
        p = self.payments
        parts = [f"primitive={p.primitive}", f"reducer={p.reducer}"]
        for field in ("affected_mode", "aci_candidate_policy", "delta_basis", "tuple_scope"):
            value = getattr(p, field)
            if value:
                parts.append(f"{field}={value}")
        return " ".join(parts)
