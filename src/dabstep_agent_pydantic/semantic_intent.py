"""Compact family-specific semantic intents for the intent compiler.

Models capture semantic choices and scalar parameters only. They never carry
executable code, final answers, task IDs, or nested InterpretationSpec trees.
"""

from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import GetJsonSchemaHandler
from pydantic import StringConstraints
from pydantic import field_validator
from pydantic import model_validator
from pydantic.json_schema import JsonSchemaValue

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ScalarValue = str | int | float | bool

TableName = Literal[
    "payments",
    "fees",
    "merchants",
    "acquirer_countries",
    "merchant_category_codes",
]

OutputKind = Literal[
    "decimal",
    "integer",
    "single_string",
    "comma_list",
    "group_value_list",
]

GeneralOperation = Literal[
    "aggregate",
    "ratio",
    "grouped_aggregate",
    "domain",
    "missingness",
    "duplicate",
    "outlier",
    "correlation",
]

AggregationKind = Literal["sum", "count", "mean", "nunique"]

FraudOperation = Literal[
    "fraud_rate",
    "fraud_rate_extreme",
    "repeat_customer_percentage",
    "missing_identity",
    "outlier_fraud",
    "correlation",
]

FeeOperation = Literal[
    "period_total_fees",
    "applicable_fee_ids",
    "fee_at_amount",
    "card_scheme_extreme",
    "aci_extreme",
    "affected_merchants",
    "fee_rate_delta",
    "merchant_mcc_delta",
]

_FEE_PARAM_KEYS = frozenset({
    "account_type",
    "aci",
    "amount",
    "card_scheme",
    "day_of_year",
    "decimals",
    "fee_id",
    "field",
    "is_credit",
    "merchant",
    "merchant_category_code",
    "month",
    "new_mcc",
    "new_value",
    "objective",
    "percentile",
    "year",
})


class IntentFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column: Identifier
    value: ScalarValue


class IntentTimeScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["all", "year", "month", "month_range", "day"] = "all"
    year: int | None = Field(default=None, ge=1900, le=2200)
    month: int | None = Field(default=None, ge=1, le=12)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_month: int | None = Field(default=None, ge=1, le=12)
    day_of_year: int | None = Field(default=None, ge=1, le=366)

    @model_validator(mode="after")
    def _scope_fields_match_kind(self) -> "IntentTimeScope":
        if self.kind == "all":
            if any(v is not None for v in (
                self.year, self.month, self.start_month, self.end_month, self.day_of_year
            )):
                raise ValueError("all scope cannot set period fields")
        elif self.kind == "year":
            if self.year is None or any(v is not None for v in (
                self.month, self.start_month, self.end_month, self.day_of_year
            )):
                raise ValueError("year scope requires year only")
        elif self.kind == "month":
            if self.year is None or self.month is None or any(v is not None for v in (
                self.start_month, self.end_month, self.day_of_year
            )):
                raise ValueError("month scope requires year and month only")
        elif self.kind == "month_range":
            if (
                self.year is None
                or self.start_month is None
                or self.end_month is None
                or self.month is not None
                or self.day_of_year is not None
                or self.start_month > self.end_month
            ):
                raise ValueError("month_range requires ordered start_month and end_month in one year")
        elif self.kind == "day":
            if self.year is None or self.day_of_year is None or any(v is not None for v in (
                self.month, self.start_month, self.end_month
            )):
                raise ValueError("day scope requires year and day_of_year only")
        return self


class _IntentBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_kind: OutputKind = "decimal"
    decimals: int | None = Field(default=None, ge=0, le=18)
    uncertainty_axes: list[str] = Field(default_factory=list)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return _strip_schema_metadata(handler(core_schema))

    @model_validator(mode="after")
    def _decimals_match_output(self) -> "_IntentBase":
        accepts_decimals = self.output_kind in {"decimal", "group_value_list"}
        if not accepts_decimals and self.decimals is not None:
            raise ValueError(f"decimals are not allowed for {self.output_kind}")
        return self


def _strip_schema_metadata(value):
    if isinstance(value, dict):
        return {
            key: _strip_schema_metadata(item)
            for key, item in value.items()
            if key not in {"title", "description", "default"}
        }
    if isinstance(value, list):
        return [_strip_schema_metadata(item) for item in value]
    return value


class GeneralIntent(_IntentBase):
    """Compact intent for general analytics families."""

    operation: GeneralOperation
    source: TableName = "payments"
    aggregation: AggregationKind | None = None
    column: Identifier | None = None
    numerator_column: Identifier | None = None
    denominator_column: Identifier | None = None
    numerator_aggregation: AggregationKind | None = None
    denominator_aggregation: AggregationKind | None = None
    numerator_filters: list[IntentFilter] = Field(default_factory=list)
    denominator_filters: list[IntentFilter] = Field(default_factory=list)
    ratio_scale: Literal[1, 100] = 1
    filters: list[IntentFilter] = Field(default_factory=list)
    time_scope: IntentTimeScope = Field(default_factory=IntentTimeScope)
    group_by: list[Identifier] = Field(default_factory=list)
    order_by: Literal["value"] | Identifier | None = None
    order_direction: Literal["asc", "desc"] = "asc"
    limit: int | None = Field(default=None, ge=1)
    extreme: Literal["max", "min"] | None = None

    @model_validator(mode="after")
    def _operation_fields(self) -> "GeneralIntent":
        if self.operation in {"aggregate", "grouped_aggregate"}:
            if self.aggregation is None:
                raise ValueError("aggregation is required for aggregate operations")
            if self.aggregation in {"sum", "mean", "nunique"} and self.column is None:
                raise ValueError(f"column is required for {self.aggregation}")
        if self.operation == "ratio":
            if self.numerator_aggregation is None or self.denominator_aggregation is None:
                raise ValueError("ratio requires numerator and denominator aggregations")
            if self.numerator_aggregation != "count" and self.numerator_column is None:
                raise ValueError("numerator_column is required unless numerator aggregation is count")
            if self.denominator_aggregation != "count" and self.denominator_column is None:
                raise ValueError("denominator_column is required unless denominator aggregation is count")
        if self.operation == "grouped_aggregate" and not self.group_by:
            raise ValueError("grouped_aggregate requires group_by")
        if self.operation == "domain" and self.column is None:
            raise ValueError("domain requires column")
        if self.operation in {"outlier", "correlation"}:
            # Representable as intent for planner fallback signaling; compiler rejects.
            pass
        if self.limit is not None and not self.group_by and self.operation != "grouped_aggregate":
            raise ValueError("limit requires grouping")
        return self


class CustomerFraudIntent(_IntentBase):
    """Compact intent for customer / fraud metric families."""

    operation: FraudOperation
    fraud_rate_basis: Literal["eur_volume", "transaction_count"] | None = None
    repeat_scope: Literal["full_history", "period_only"] | None = None
    identity_missing: Literal["include", "exclude"] = "exclude"
    identity_field: Identifier = "email_address"
    group_by: Identifier | None = None
    extreme: Literal["max", "min"] | None = None
    filters: list[IntentFilter] = Field(default_factory=list)
    time_scope: IntentTimeScope = Field(default_factory=IntentTimeScope)
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _operation_fields(self) -> "CustomerFraudIntent":
        if self.operation in {"fraud_rate", "fraud_rate_extreme", "outlier_fraud"}:
            if self.fraud_rate_basis is None:
                raise ValueError("fraud_rate_basis is required for fraud rate operations")
        if self.operation == "fraud_rate_extreme" and self.group_by is None:
            raise ValueError("fraud_rate_extreme requires group_by")
        if self.operation == "repeat_customer_percentage" and self.repeat_scope is None:
            raise ValueError("repeat_scope is required for repeat_customer_percentage")
        if self.operation in {"outlier_fraud", "correlation"}:
            pass
        return self


class FeeIntent(_IntentBase):
    """Compact intent for fee-family operations. Never embeds InterpretationSpec."""

    operation: FeeOperation
    params: dict[str, ScalarValue] = Field(default_factory=dict)
    # Optional explicit axes when the planner surfaces uncertainty.
    fee_reducer: Literal["sum_all_matching", "min_match", "first_match"] = "sum_all_matching"
    wildcard_policy: Literal["manual", "strict"] = "manual"
    delta_basis: Literal["rate", "fixed_component"] | None = None
    affected_mode: Literal["losers_only", "symmetric_difference", "baseline_members"] | None = None
    objective: Literal["max", "min"] | None = None

    @field_validator("params", mode="before")
    @classmethod
    def _params_are_closed_scalars(cls, values):
        if not isinstance(values, dict):
            raise ValueError("params must be an object")
        unknown = set(values) - _FEE_PARAM_KEYS
        if unknown:
            raise ValueError(f"unsupported fee parameter(s): {sorted(unknown)}")
        if any(isinstance(value, (dict, list, tuple, set)) for value in values.values()):
            raise ValueError("fee parameters must be scalar values")
        return values

    @model_validator(mode="after")
    def _required_params(self) -> "FeeIntent":
        required: dict[str, set[str]] = {
            "period_total_fees": {"merchant", "year"},
            "applicable_fee_ids": {"merchant", "year"},
            "fee_at_amount": {"amount"},
            "card_scheme_extreme": {"amount"},
            "aci_extreme": {"amount"},
            "affected_merchants": {"fee_id", "year"},
            "fee_rate_delta": {"fee_id", "new_value", "year"},
            "merchant_mcc_delta": {"merchant", "new_mcc", "year"},
        }
        missing = required[self.operation] - set(self.params)
        if missing:
            raise ValueError(f"{self.operation} requires params: {sorted(missing)}")
        if self.operation in {"fee_rate_delta"} and self.delta_basis is None:
            # Default is applied by compiler; allow omission.
            pass
        return self


class UnsupportedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str = Field(min_length=1)


SemanticIntent = GeneralIntent | CustomerFraudIntent | FeeIntent | UnsupportedIntent
