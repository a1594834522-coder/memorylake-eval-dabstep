"""Closed compact semantic intent models and schema-size budgets."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dabstep_agent_pydantic.semantic_intent import (
    CustomerFraudIntent,
    FeeIntent,
    GeneralIntent,
    IntentFilter,
    IntentTimeScope,
    UnsupportedIntent,
)

FORBIDDEN_FIELDS = ("python_code", "sql", "final_answer", "task_id")
SCHEMA_BUDGET = 4000


def _schema_chars(model) -> int:
    return len(json.dumps(model.model_json_schema(), separators=(",", ":")))


@pytest.mark.parametrize(
    "model",
    [GeneralIntent, CustomerFraudIntent, FeeIntent, UnsupportedIntent],
)
def test_family_intent_schema_is_under_budget(model):
    assert _schema_chars(model) < SCHEMA_BUDGET


def test_general_intent_schema_has_runtime_cost_headroom():
    assert _schema_chars(GeneralIntent) < 3200


@pytest.mark.parametrize(
    "model",
    [GeneralIntent, CustomerFraudIntent, FeeIntent, UnsupportedIntent],
)
@pytest.mark.parametrize("forbidden", FORBIDDEN_FIELDS)
def test_intent_models_reject_code_and_answer_fields(model, forbidden):
    payload = _minimal_payload(model)
    model.model_validate(payload)
    payload[forbidden] = "x"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate(payload)


def test_general_intent_aggregate_requires_columns_for_sum():
    with pytest.raises(ValidationError):
        GeneralIntent(
            operation="aggregate",
            aggregation="sum",
            source="payments",
            output_kind="decimal",
            decimals=2,
        )


def test_general_intent_accepts_scalar_filters_and_time_scope():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        filters=[IntentFilter(column="card_scheme", value="NexPay")],
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="integer",
    )
    assert intent.operation == "aggregate"
    assert intent.filters[0].value == "NexPay"


def test_intent_time_scope_accepts_same_year_month_ranges():
    scope = IntentTimeScope(
        kind="month_range",
        year=2023,
        start_month=6,
        end_month=10,
    )

    assert scope.start_month == 6
    assert scope.end_month == 10

    with pytest.raises(ValidationError, match="month_range"):
        IntentTimeScope(kind="month_range", year=2023, start_month=10, end_month=6)


def test_general_count_ratio_uses_separate_numerator_filters_without_columns():
    intent = GeneralIntent(
        operation="ratio",
        numerator_aggregation="count",
        denominator_aggregation="count",
        numerator_filters=[IntentFilter(column="is_credit", value=True)],
        ratio_scale=100,
        output_kind="decimal",
        decimals=2,
    )

    assert intent.numerator_column is None
    assert intent.denominator_column is None
    assert intent.numerator_filters[0].column == "is_credit"


def test_general_intent_rejects_nested_params():
    with pytest.raises(ValidationError):
        GeneralIntent(
            operation="aggregate",
            aggregation="count",
            source="payments",
            output_kind="integer",
            filters=[{"column": "x", "value": {"nested": 1}}],
        )


def test_customer_fraud_intent_exposes_semantic_axes():
    intent = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=6,
    )
    assert intent.fraud_rate_basis == "eur_volume"


def test_customer_fraud_repeat_customer_axes():
    intent = CustomerFraudIntent(
        operation="repeat_customer_percentage",
        repeat_scope="full_history",
        identity_missing="exclude",
        output_kind="decimal",
        decimals=6,
    )
    assert intent.repeat_scope == "full_history"
    assert intent.identity_missing == "exclude"


def test_missing_identity_intent_names_the_identity_field():
    intent = CustomerFraudIntent(
        operation="missing_identity",
        identity_field="ip_address",
        output_kind="decimal",
        decimals=3,
    )

    assert intent.identity_field == "ip_address"


def test_fee_intent_schema_is_small_and_closed():
    schema = json.dumps(FeeIntent.model_json_schema(), separators=(",", ":"))
    assert len(schema) < SCHEMA_BUDGET
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FeeIntent(
            operation="period_total_fees",
            params={"merchant": "Merchant_A", "year": 2023},
            output_kind="decimal",
            decimals=2,
            final_answer="42",
        )


def test_decimal_intents_may_omit_precision_for_compiler_defaults():
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023},
    )

    assert intent.output_kind == "decimal"
    assert intent.decimals is None


def test_fee_intent_params_are_scalar_only():
    with pytest.raises(ValidationError):
        FeeIntent(
            operation="period_total_fees",
            params={"merchant": "Merchant_A", "year": 2023, "nested": {"x": 1}},
        )


def test_fee_intent_requires_params_object():
    intent = FeeIntent(
        operation="applicable_fee_ids",
        params={"merchant": "Merchant_A", "year": 2023, "month": 12},
        output_kind="comma_list",
    )
    assert intent.params["month"] == 12


def test_fee_intent_rejects_interpretation_spec_field():
    with pytest.raises(ValidationError):
        FeeIntent(
            operation="period_total_fees",
            params={"merchant": "Merchant_A", "year": 2023},
            interpretation={"name": "x"},
        )


def test_unsupported_intent_requires_reason():
    with pytest.raises(ValidationError):
        UnsupportedIntent(reason="")
    intent = UnsupportedIntent(reason="no deterministic primitive for correlation")
    assert "correlation" in intent.reason


def test_intent_filter_requires_identifier_column():
    with pytest.raises(ValidationError):
        IntentFilter(column="bad-column!", value=1)


def _minimal_payload(model) -> dict:
    if model is GeneralIntent:
        return {
            "operation": "aggregate",
            "aggregation": "count",
            "source": "payments",
            "output_kind": "integer",
        }
    if model is CustomerFraudIntent:
        return {
            "operation": "fraud_rate",
            "fraud_rate_basis": "eur_volume",
            "output_kind": "decimal",
            "decimals": 4,
        }
    if model is FeeIntent:
        return {
            "operation": "period_total_fees",
            "params": {"merchant": "Merchant_A", "year": 2023},
            "output_kind": "decimal",
            "decimals": 2,
        }
    if model is UnsupportedIntent:
        return {"reason": "unsupported"}
    raise AssertionError(model)
