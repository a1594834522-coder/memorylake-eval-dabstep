"""Deterministic compilation of compact analytics intents into AnalysisSpec."""

from __future__ import annotations

import pandas as pd
import pytest

from dabstep_agent_pydantic.analysis_executor import execute_analysis
from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import MonthScope
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import YearScope
from dabstep_agent_pydantic.intent_compiler import IntentCompilationError
from dabstep_agent_pydantic.intent_compiler import compile_semantic_intent
from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.semantic_intent import CustomerFraudIntent
from dabstep_agent_pydantic.semantic_intent import GeneralIntent
from dabstep_agent_pydantic.semantic_intent import IntentFilter
from dabstep_agent_pydantic.semantic_intent import IntentTimeScope
from dabstep_agent_pydantic.semantic_intent import UnsupportedIntent


def test_compile_count_aggregate():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        filters=[IntentFilter(column="merchant", value="Merchant_A")],
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="integer",
    )
    spec = compile_semantic_intent(intent)
    assert isinstance(spec.measure, AggregateMeasure)
    assert spec.measure.kind == "count"
    assert spec.source.table == "payments"
    assert isinstance(spec.time_scope, YearScope)
    assert spec.time_scope.year == 2023
    assert spec.filters == [EqFilter(column="merchant", value="Merchant_A")]
    assert spec.output.kind == "integer"


@pytest.mark.parametrize(
    ("aggregation", "column"),
    [("sum", "eur_amount"), ("mean", "eur_amount"), ("nunique", "email_address")],
)
def test_compile_column_aggregates(aggregation, column):
    intent = GeneralIntent(
        operation="aggregate",
        aggregation=aggregation,
        column=column,
        source="payments",
        output_kind="decimal" if aggregation != "nunique" else "integer",
        decimals=2 if aggregation != "nunique" else None,
    )
    spec = compile_semantic_intent(intent)
    assert spec.measure.kind == aggregation
    assert spec.measure.column == column


def test_compile_ratio_by_eur_volume_and_transaction_count():
    volume = GeneralIntent(
        operation="ratio",
        source="payments",
        numerator_column="eur_amount",
        denominator_column="eur_amount",
        numerator_aggregation="sum",
        denominator_aggregation="sum",
        ratio_scale=100,
        filters=[IntentFilter(column="has_fraudulent_dispute", value=True)],
        output_kind="decimal",
        decimals=4,
    )
    # Fraud-style ratio uses measure-level numerator filters via CustomerFraudIntent.
    fraud = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        filters=[IntentFilter(column="shopper_interaction", value="Ecommerce")],
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=4,
    )
    count_basis = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="transaction_count",
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=4,
    )

    volume_spec = compile_semantic_intent(volume)
    assert isinstance(volume_spec.measure, RatioMeasure)
    assert volume_spec.measure.scale == 100

    fraud_spec = compile_semantic_intent(fraud)
    assert fraud_spec.measure.kind == "ratio"
    assert fraud_spec.measure.numerator.kind == "sum"
    assert fraud_spec.measure.numerator.column == "eur_amount"
    assert fraud_spec.measure.numerator.filters == [
        EqFilter(column="has_fraudulent_dispute", value=True)
    ]
    assert fraud_spec.measure.denominator.kind == "sum"
    assert fraud_spec.measure.denominator.column == "eur_amount"
    assert fraud_spec.filters == [EqFilter(column="shopper_interaction", value="Ecommerce")]
    assert isinstance(fraud_spec.time_scope, YearScope)

    count_spec = compile_semantic_intent(count_basis)
    assert count_spec.measure.numerator.kind == "count"
    assert count_spec.measure.denominator.kind == "count"


def test_general_count_ratio_filters_only_the_numerator():
    intent = GeneralIntent(
        operation="ratio",
        numerator_aggregation="count",
        denominator_aggregation="count",
        numerator_filters=[IntentFilter(column="is_credit", value=True)],
        ratio_scale=100,
        output_kind="decimal",
        decimals=2,
    )
    data = DABStepData(
        payments=pd.DataFrame({"is_credit": [True, False, True, False]}),
        fees=pd.DataFrame(),
        merchants=pd.DataFrame(),
        acquirer_countries=pd.DataFrame(),
        merchant_category_codes=pd.DataFrame(),
    )

    result = execute_analysis(data, compile_semantic_intent(intent))

    assert result.formatted_value == "50.00"


def test_compile_grouped_extrema_with_ordering():
    intent = GeneralIntent(
        operation="grouped_aggregate",
        aggregation="mean",
        column="eur_amount",
        source="payments",
        group_by=["issuing_country"],
        order_by="value",
        order_direction="asc",
        limit=5,
        extreme="min",
        output_kind="group_value_list",
        decimals=2,
    )
    spec = compile_semantic_intent(intent)
    assert spec.group_by == ["issuing_country"]
    assert spec.ordering[0].by == "value"
    assert spec.ordering[0].direction == "asc"
    assert spec.limit == 5
    assert spec.output.kind == "group_value_list"


@pytest.mark.parametrize("output_kind", ["single_string", "integer"])
def test_grouped_top_one_returns_only_the_group_label(output_kind):
    intent = GeneralIntent(
        operation="grouped_aggregate",
        aggregation="count",
        group_by=["hour_of_day"],
        order_by="value",
        order_direction="desc",
        limit=1,
        output_kind=output_kind,
    )

    spec = compile_semantic_intent(intent)

    assert spec.output.kind == "comma_list"


def test_compile_month_and_day_scopes():
    month = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        time_scope=IntentTimeScope(kind="month", year=2023, month=3),
        output_kind="integer",
    )
    day = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        time_scope=IntentTimeScope(kind="day", year=2023, day_of_year=40),
        output_kind="integer",
    )
    month_spec = compile_semantic_intent(month)
    day_spec = compile_semantic_intent(day)
    assert isinstance(month_spec.time_scope, MonthScope)
    assert month_spec.time_scope.month == 3
    assert day_spec.time_scope.kind == "day_range"
    assert day_spec.time_scope.start_day == 40
    assert day_spec.time_scope.end_day == 40


def test_compile_multi_month_range_to_exact_day_range():
    intent = GeneralIntent(
        operation="grouped_aggregate",
        aggregation="mean",
        column="eur_amount",
        group_by=["ip_country"],
        time_scope=IntentTimeScope(
            kind="month_range",
            year=2023,
            start_month=6,
            end_month=10,
        ),
        output_kind="group_value_list",
        decimals=2,
    )

    spec = compile_semantic_intent(intent)

    assert spec.time_scope.kind == "day_range"
    assert spec.time_scope.start_day == 152
    assert spec.time_scope.end_day == 304


def test_compile_fraud_rate_by_eur_volume():
    intent = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        filters=[IntentFilter(column="shopper_interaction", value="Ecommerce")],
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=4,
    )
    spec = compile_semantic_intent(intent)
    assert spec.measure.kind == "ratio"
    assert spec.measure.numerator.kind == "sum"
    assert spec.measure.numerator.column == "eur_amount"


def test_customer_fraud_normalizes_public_in_person_schema_alias():
    intent = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        filters=[IntentFilter(column="device_type", value="in-person")],
        output_kind="decimal",
        decimals=4,
    )

    spec = compile_semantic_intent(intent)

    assert spec.filters == [EqFilter(column="shopper_interaction", value="POS")]


def test_general_normalizes_payment_method_to_card_scheme():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        filters=[IntentFilter(column="payment_method", value="TransactPlus")],
        output_kind="integer",
    )

    spec = compile_semantic_intent(intent)

    assert spec.filters == [EqFilter(column="card_scheme", value="TransactPlus")]


def test_conflicting_equality_filters_fall_back_instead_of_empty_and_population():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        filters=[
            IntentFilter(column="ip_country", value="NL"),
            IntentFilter(column="ip_country", value="BE"),
        ],
        output_kind="integer",
    )

    with pytest.raises(IntentCompilationError, match="multiple equality values"):
        compile_semantic_intent(intent)


def test_unspecified_decimal_precision_preserves_natural_numeric_string():
    intent = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        filters=[IntentFilter(column="shopper_interaction", value="POS")],
        output_kind="decimal",
    )
    data = DABStepData(
        payments=pd.DataFrame({
            "shopper_interaction": ["POS", "POS"],
            "has_fraudulent_dispute": [False, False],
            "eur_amount": [10.0, 20.0],
        }),
        fees=pd.DataFrame(),
        merchants=pd.DataFrame(),
        acquirer_countries=pd.DataFrame(),
        merchant_category_codes=pd.DataFrame(),
    )

    result = execute_analysis(data, compile_semantic_intent(intent))

    assert result.formatted_value == "0.0"


def test_missing_identity_decimal_output_is_a_percentage_for_the_named_field():
    intent = CustomerFraudIntent(
        operation="missing_identity",
        identity_field="ip_address",
        filters=[IntentFilter(column="has_fraudulent_dispute", value=True)],
        output_kind="decimal",
        decimals=3,
    )
    data = DABStepData(
        payments=pd.DataFrame({
            "ip_address": [None, "1.1.1.1", None, None],
            "has_fraudulent_dispute": [True, True, False, True],
        }),
        fees=pd.DataFrame(),
        merchants=pd.DataFrame(),
        acquirer_countries=pd.DataFrame(),
        merchant_category_codes=pd.DataFrame(),
    )

    result = execute_analysis(data, compile_semantic_intent(intent))

    assert result.formatted_value == "66.667"


def test_compile_fraud_rate_extreme_groups_by_column():
    intent = CustomerFraudIntent(
        operation="fraud_rate_extreme",
        fraud_rate_basis="eur_volume",
        group_by="merchant",
        extreme="min",
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="single_string",
    )
    spec = compile_semantic_intent(intent)
    assert spec.group_by == ["merchant"]
    assert spec.ordering[0].direction == "asc"
    assert spec.limit == 1
    assert spec.output.kind == "comma_list" or spec.output.kind == "single_string"


def test_compile_repeat_customer_full_history_vs_period_only():
    full = CustomerFraudIntent(
        operation="repeat_customer_percentage",
        repeat_scope="full_history",
        identity_missing="exclude",
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=6,
    )
    period = CustomerFraudIntent(
        operation="repeat_customer_percentage",
        repeat_scope="period_only",
        identity_missing="include",
        time_scope=IntentTimeScope(kind="year", year=2023),
        output_kind="decimal",
        decimals=6,
    )
    with pytest.raises(IntentCompilationError, match="repeat_customer_percentage"):
        compile_semantic_intent(full)
    with pytest.raises(IntentCompilationError, match="repeat_customer_percentage"):
        compile_semantic_intent(period)


def test_customer_fraud_compiler_does_not_invent_registry_policy_ids():
    intent = CustomerFraudIntent(
        operation="fraud_rate",
        fraud_rate_basis="eur_volume",
        output_kind="decimal",
        decimals=4,
    )

    spec = compile_semantic_intent(intent)

    assert spec.policy_ids == []


def test_yes_no_fraud_comparison_falls_back_without_comparison_primitive():
    intent = CustomerFraudIntent(
        operation="fraud_rate_extreme",
        fraud_rate_basis="eur_volume",
        group_by="is_credit",
        extreme="max",
        output_kind="single_string",
    )

    with pytest.raises(IntentCompilationError, match="yes/no"):
        compile_semantic_intent(intent, guidelines="Answer must be just either yes or no.")


def test_compile_missing_email_included_versus_excluded():
    included = GeneralIntent(
        operation="missingness",
        column="email_address",
        source="payments",
        output_kind="integer",
    )
    excluded = GeneralIntent(
        operation="aggregate",
        aggregation="nunique",
        column="email_address",
        source="payments",
        output_kind="integer",
    )
    missing_spec = compile_semantic_intent(included)
    nunique_spec = compile_semantic_intent(excluded)
    assert missing_spec.measure.kind == "count"
    assert any(getattr(f, "op", None) == "is_null" for f in missing_spec.measure.filters) or any(
        getattr(f, "op", None) == "is_null" for f in missing_spec.filters
    )
    assert nunique_spec.measure.missing == "exclude"


@pytest.mark.parametrize("operation", ["correlation", "outlier"])
def test_unsupported_general_operations_raise(operation):
    intent = GeneralIntent(
        operation=operation,
        source="payments",
        output_kind="decimal",
        decimals=2,
        column="eur_amount" if operation == "outlier" else None,
    )
    with pytest.raises(IntentCompilationError):
        compile_semantic_intent(intent)


def test_unsupported_intent_raises():
    with pytest.raises(IntentCompilationError, match="correlation"):
        compile_semantic_intent(UnsupportedIntent(reason="no deterministic primitive for correlation"))


def test_compiler_does_not_accept_task_id_kwargs():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        output_kind="integer",
    )
    with pytest.raises(TypeError):
        compile_semantic_intent(intent, task_id="123")  # type: ignore[call-arg]
