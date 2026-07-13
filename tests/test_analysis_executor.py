from __future__ import annotations

import pandas as pd
import pytest

from dabstep_agent_pydantic.analysis_executor import execute_analysis
from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import JoinKey
from dabstep_agent_pydantic.analysis_spec_v2 import JoinSpec
from dabstep_agent_pydantic.analysis_spec_v2 import MonthScope
from dabstep_agent_pydantic.analysis_spec_v2 import OrderSpec
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.dabstep_core import DABStepData


@pytest.fixture
def data() -> DABStepData:
    payments = pd.DataFrame([
        {
            "merchant": "Merchant_A",
            "year": 2023,
            "day_of_year": 5,
            "eur_amount": 100.0,
            "has_fraudulent_dispute": False,
            "email_address": "x@example.com",
            "shopper_interaction": "POS",
            "acquirer": "acq-1",
        },
        {
            "merchant": "Merchant_A",
            "year": 2023,
            "day_of_year": 10,
            "eur_amount": 30.0,
            "has_fraudulent_dispute": True,
            "email_address": "x@example.com",
            "shopper_interaction": "Ecommerce",
            "acquirer": "acq-1",
        },
        {
            "merchant": "Merchant_A",
            "year": 2023,
            "day_of_year": 40,
            "eur_amount": 50.0,
            "has_fraudulent_dispute": False,
            "email_address": None,
            "shopper_interaction": "POS",
            "acquirer": "acq-1",
        },
        {
            "merchant": "Merchant_B",
            "year": 2023,
            "day_of_year": 7,
            "eur_amount": 20.0,
            "has_fraudulent_dispute": True,
            "email_address": "y@example.com",
            "shopper_interaction": "POS",
            "acquirer": "acq-2",
        },
    ])
    return DABStepData(
        fees=pd.DataFrame(),
        payments=payments,
        merchants=pd.DataFrame([
            {"merchant": "Merchant_A", "account_type": "H"},
            {"merchant": "Merchant_B", "account_type": "R"},
        ]),
        acquirer_countries=pd.DataFrame([
            {"acquirer": "acq-1", "country_code": "NL"},
            {"acquirer": "acq-2", "country_code": "DE"},
        ]),
        merchant_category_codes=pd.DataFrame(),
    )


def _spec(*, measure, **overrides) -> AnalysisSpec:
    values = {
        "source": SourceSpec(table="payments"),
        "measure": measure,
        "output": AnalysisOutputContract(kind="decimal", decimals=2),
        "policy_ids": ["test.policy.v1"],
    }
    values.update(overrides)
    return AnalysisSpec(**values)


def test_fraud_rate_by_eur_volume_returns_intermediates_and_row_counts(data):
    spec = _spec(
        filters=[EqFilter(column="merchant", value="Merchant_A")],
        time_scope=MonthScope(year=2023, month=1),
        measure=RatioMeasure(
            numerator=AggregateMeasure(
                kind="sum",
                column="eur_amount",
                filters=[EqFilter(column="has_fraudulent_dispute", value=True)],
            ),
            denominator=AggregateMeasure(kind="sum", column="eur_amount"),
        ),
        output=AnalysisOutputContract(kind="decimal", decimals=4),
    )

    result = execute_analysis(data, spec)

    assert result.raw_value == pytest.approx(30 / 130)
    assert result.formatted_value == "0.2308"
    assert result.intermediates == {"numerator": 30.0, "denominator": 130.0}
    assert result.row_counts == {
        "source": 4,
        "time_scope": 3,
        "filters": 2,
        "measure:numerator": 1,
        "measure:denominator": 2,
    }


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        (AggregateMeasure(kind="sum", column="eur_amount"), 180.0),
        (AggregateMeasure(kind="count"), 3),
        (AggregateMeasure(kind="mean", column="eur_amount"), 60.0),
    ],
)
def test_filtered_scalar_aggregations(data, measure, expected):
    result = execute_analysis(
        data,
        _spec(
            filters=[EqFilter(column="merchant", value="Merchant_A")],
            measure=measure,
        ),
    )

    assert result.raw_value == expected


def test_unique_counts_apply_missing_value_policy(data):
    base = {
        "filters": [EqFilter(column="merchant", value="Merchant_A")],
        "output": AnalysisOutputContract(kind="integer"),
    }
    excluded = execute_analysis(
        data,
        _spec(measure=AggregateMeasure(kind="nunique", column="email_address"), **base),
    )
    included = execute_analysis(
        data,
        _spec(
            measure=AggregateMeasure(kind="nunique", column="email_address", missing="include"),
            **base,
        ),
    )

    assert excluded.raw_value == 1
    assert included.raw_value == 2
    with pytest.raises(ValueError, match="missing values"):
        execute_analysis(
            data,
            _spec(
                measure=AggregateMeasure(kind="nunique", column="email_address", missing="error"),
                **base,
            ),
        )


def test_allowed_join_can_drive_filters(data):
    result = execute_analysis(
        data,
        _spec(
            joins=[JoinSpec(
                left="payments",
                right="merchants",
                keys=[JoinKey(left_column="merchant", right_column="merchant")],
            )],
            filters=[EqFilter(column="account_type", value="H")],
            measure=AggregateMeasure(kind="sum", column="eur_amount"),
        ),
    )

    assert result.raw_value == 180.0
    assert result.row_counts["join:merchants"] == 4


def test_group_extrema_are_stably_ordered_and_formatted(data):
    result = execute_analysis(
        data,
        _spec(
            measure=AggregateMeasure(kind="mean", column="eur_amount"),
            group_by=["shopper_interaction"],
            ordering=[OrderSpec(by="value", direction="desc")],
            limit=1,
            output=AnalysisOutputContract(kind="group_value_list", decimals=2),
        ),
    )

    assert result.raw_value == [
        {"group": {"shopper_interaction": "POS"}, "value": pytest.approx(170 / 3)},
    ]
    assert result.formatted_value == "[POS: 56.67]"
    assert result.row_counts["groups"] == 2


def test_grouped_comma_list_formats_group_labels(data):
    result = execute_analysis(
        data,
        _spec(
            measure=AggregateMeasure(kind="mean", column="eur_amount"),
            group_by=["shopper_interaction"],
            ordering=[OrderSpec(by="value", direction="desc")],
            output=AnalysisOutputContract(kind="comma_list"),
        ),
    )

    assert result.formatted_value == "POS, Ecommerce"


def test_plan_and_execution_fingerprints_are_deterministic(data):
    spec = _spec(measure=AggregateMeasure(kind="sum", column="eur_amount"))

    first = execute_analysis(data, spec)
    second = execute_analysis(data, spec)

    assert first.policy_ids == ["test.policy.v1"]
    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.plan_fingerprint != first.execution_fingerprint
