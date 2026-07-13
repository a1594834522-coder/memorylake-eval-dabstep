from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints
from pydantic import field_validator
from pydantic import model_validator

from dabstep_agent_pydantic.distill.spec import InterpretationSpec


Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ScalarValue = str | int | float | bool
TableName = Literal[
    "payments",
    "fees",
    "merchants",
    "acquirer_countries",
    "merchant_category_codes",
]


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    table: TableName


class JoinKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_column: Identifier
    right_column: Identifier


class JoinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left: TableName
    right: TableName
    keys: list[JoinKey] = Field(min_length=1)
    how: Literal["inner", "left"] = "inner"


class EqFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["eq", "neq"] = "eq"
    column: Identifier
    value: ScalarValue


class InFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["in", "not_in"]
    column: Identifier
    values: list[ScalarValue] = Field(min_length=1)


class ComparisonFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["gt", "gte", "lt", "lte"]
    column: Identifier
    value: int | float


class NullFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["is_null", "not_null"]
    column: Identifier


FilterSpec = Annotated[
    EqFilter | InFilter | ComparisonFilter | NullFilter,
    Field(discriminator="op"),
]


class AllTimeScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["all"] = "all"


class YearScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["year"] = "year"
    year: int = Field(ge=1900, le=2200)
    year_column: Identifier = "year"


class MonthScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["month"] = "month"
    year: int = Field(ge=1900, le=2200)
    month: int = Field(ge=1, le=12)
    year_column: Identifier = "year"
    day_column: Identifier = "day_of_year"


class DayRangeScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["day_range"] = "day_range"
    year: int = Field(ge=1900, le=2200)
    start_day: int = Field(ge=1, le=366)
    end_day: int = Field(ge=1, le=366)
    year_column: Identifier = "year"
    day_column: Identifier = "day_of_year"

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "DayRangeScope":
        if self.end_day < self.start_day:
            raise ValueError("end_day must be greater than or equal to start_day")
        return self


TimeScope = Annotated[
    AllTimeScope | YearScope | MonthScope | DayRangeScope,
    Field(discriminator="kind"),
]


class AggregateMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["sum", "count", "mean", "nunique"]
    column: Identifier | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    missing: Literal["exclude", "include", "error"] = "exclude"

    @model_validator(mode="after")
    def _column_matches_aggregation(self) -> "AggregateMeasure":
        if self.kind in {"sum", "mean", "nunique"} and self.column is None:
            raise ValueError(f"column is required for {self.kind}")
        return self


class RatioMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ratio"] = "ratio"
    numerator: AggregateMeasure
    denominator: AggregateMeasure
    scale: Literal[1, 100] = 1
    zero_denominator: Literal["zero", "error"] = "zero"


_INTERPRETATION_PARAMS = frozenset({
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


class InterpretationMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["interpretation"] = "interpretation"
    interpretation: InterpretationSpec
    params: dict[str, ScalarValue]

    @field_validator("params", mode="before")
    @classmethod
    def _params_are_closed_scalars(cls, values):
        if not isinstance(values, dict):
            raise ValueError("interpretation params must be an object")
        unknown = set(values) - _INTERPRETATION_PARAMS
        if unknown:
            raise ValueError(f"unsupported interpretation parameter(s): {sorted(unknown)}")
        if any(isinstance(value, (dict, list, tuple, set)) for value in values.values()):
            raise ValueError("interpretation parameters must be scalar values")
        return values


MeasureSpec = Annotated[
    AggregateMeasure | RatioMeasure | InterpretationMeasure,
    Field(discriminator="kind"),
]


class OrderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    by: Identifier | Literal["value"]
    direction: Literal["asc", "desc"] = "asc"


class AnalysisOutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["decimal", "integer", "single_string", "comma_list", "group_value_list"]
    decimals: int | None = Field(default=None, ge=0, le=18)
    empty_string_allowed: bool = False
    not_applicable_allowed: bool = False

    @model_validator(mode="after")
    def _decimals_match_output(self) -> "AnalysisOutputContract":
        decimal_kinds = {"decimal", "group_value_list"}
        if self.kind in decimal_kinds and self.decimals is None:
            raise ValueError(f"decimals are required for {self.kind}")
        if self.kind not in decimal_kinds and self.decimals is not None:
            raise ValueError(f"decimals are not allowed for {self.kind}")
        return self


_ALLOWED_JOINS: dict[tuple[TableName, TableName], frozenset[tuple[str, str]]] = {
    ("payments", "merchants"): frozenset({("merchant", "merchant")}),
    ("payments", "acquirer_countries"): frozenset({("acquirer", "acquirer")}),
    ("merchants", "merchant_category_codes"): frozenset({
        ("merchant_category_code", "mcc"),
    }),
}


class AnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str | None = None
    source: SourceSpec
    joins: list[JoinSpec] = Field(default_factory=list)
    time_scope: TimeScope = Field(default_factory=AllTimeScope)
    filters: list[FilterSpec] = Field(default_factory=list)
    measure: MeasureSpec
    group_by: list[Identifier] = Field(default_factory=list)
    ordering: list[OrderSpec] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    output: AnalysisOutputContract
    policy_ids: list[str] = Field(default_factory=list)
    unresolved_axes: list[str] = Field(default_factory=list)

    @field_validator("group_by", "policy_ids", "unresolved_axes")
    @classmethod
    def _values_are_unique(cls, values: list[str], info) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError(f"{info.field_name} must contain unique values")
        return values

    @model_validator(mode="after")
    def _plan_is_compatible(self) -> "AnalysisSpec":
        if isinstance(self.measure, InterpretationMeasure):
            if (
                self.joins
                or not isinstance(self.time_scope, AllTimeScope)
                or self.filters
                or self.group_by
                or self.ordering
                or self.limit is not None
            ):
                raise ValueError("interpretation measure cannot combine with generic transforms")
            interpretation = self.measure.interpretation
            expected_source = "fees" if interpretation.population == "fee_rules" else "payments"
            if self.source.table != expected_source:
                raise ValueError(
                    f"interpretation source must be {expected_source!r}, got {self.source.table!r}"
                )
            output_kind = interpretation.output.kind
            expected_outputs = {
                "decimal": {"decimal"},
                "integer": {"integer"},
                "id_list": {"comma_list"},
                "string_list": {"comma_list"},
                "single_string": (
                    {"comma_list"}
                    if interpretation.output.tie_policy == "list_all_sorted"
                    else {"single_string"}
                ),
            }[output_kind]
            if self.output.kind not in expected_outputs:
                raise ValueError(
                    f"interpretation output {output_kind!r} requires one of {sorted(expected_outputs)}"
                )
        available_tables: set[TableName] = {self.source.table}
        for join in self.joins:
            allowed_keys = _ALLOWED_JOINS.get((join.left, join.right))
            requested_keys = frozenset(
                (key.left_column, key.right_column)
                for key in join.keys
            )
            if join.left not in available_tables or allowed_keys != requested_keys:
                raise ValueError(
                    f"join is not allowed: {join.left} -> {join.right} on {sorted(requested_keys)}"
                )
            available_tables.add(join.right)

        for order in self.ordering:
            if order.by != "value" and order.by not in self.group_by:
                raise ValueError(f"ordering column {order.by!r} is not in group_by")

        if self.output.kind == "group_value_list" and not self.group_by:
            raise ValueError("group_value_list output requires group_by")
        if self.group_by and self.output.kind not in {"group_value_list", "comma_list"}:
            raise ValueError("grouped analysis requires a grouped list output")
        if self.limit is not None and not self.group_by:
            raise ValueError("limit is only valid for grouped analysis")
        if self.output.kind == "integer":
            if isinstance(self.measure, (RatioMeasure,)) or (
                isinstance(self.measure, AggregateMeasure)
                and self.measure.kind not in {"count", "nunique"}
            ):
                raise ValueError("integer output requires count or nunique aggregation")
        return self
