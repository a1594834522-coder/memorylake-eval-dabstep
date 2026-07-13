from __future__ import annotations

import pytest
from pydantic import ValidationError

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import DayRangeScope
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import JoinKey
from dabstep_agent_pydantic.analysis_spec_v2 import JoinSpec
from dabstep_agent_pydantic.analysis_spec_v2 import MonthScope
from dabstep_agent_pydantic.analysis_spec_v2 import OrderSpec
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec


def _sum_amount(**overrides) -> AggregateMeasure:
    values = {"kind": "sum", "column": "eur_amount"}
    values.update(overrides)
    return AggregateMeasure(**values)


def _spec(**overrides) -> AnalysisSpec:
    values = {
        "source": SourceSpec(table="payments"),
        "filters": [EqFilter(column="merchant", value="Merchant_A")],
        "measure": _sum_amount(),
        "output": AnalysisOutputContract(kind="decimal", decimals=2),
        "policy_ids": ["fees.total.sum_all_matching.v1"],
    }
    values.update(overrides)
    return AnalysisSpec(**values)


def test_analysis_spec_accepts_closed_scalar_aggregation_plan():
    spec = _spec(time_scope=MonthScope(year=2023, month=4))

    assert spec.source.table == "payments"
    assert spec.measure.kind == "sum"
    assert spec.output.decimals == 2


def test_sources_and_joins_are_restricted():
    with pytest.raises(ValidationError, match="table"):
        SourceSpec(table="arbitrary_file")

    join = JoinSpec(
        left="payments",
        right="merchants",
        keys=[JoinKey(left_column="merchant", right_column="merchant")],
    )
    assert _spec(joins=[join]).joins == [join]

    forbidden = JoinSpec(
        left="payments",
        right="fees",
        keys=[JoinKey(left_column="merchant", right_column="ID")],
    )
    with pytest.raises(ValidationError, match="join is not allowed"):
        _spec(joins=[forbidden])


def test_filter_operators_have_typed_values_and_forbid_code():
    with pytest.raises(ValidationError, match="list"):
        AnalysisSpec.model_validate({
            **_spec().model_dump(),
            "filters": [{"op": "in", "column": "merchant", "values": "Merchant_A"}],
        })

    with pytest.raises(ValidationError, match="code"):
        EqFilter(column="merchant", value="Merchant_A", code="df.query('x')")


def test_time_scopes_validate_calendar_ranges():
    assert MonthScope(year=2024, month=2).month == 2

    with pytest.raises(ValidationError, match="month"):
        MonthScope(year=2023, month=13)

    with pytest.raises(ValidationError, match="end_day"):
        DayRangeScope(year=2023, start_day=40, end_day=20)


def test_measures_and_ratios_enforce_aggregation_compatibility():
    with pytest.raises(ValidationError, match="column"):
        AggregateMeasure(kind="mean")

    assert AggregateMeasure(kind="count").column is None

    ratio = RatioMeasure(
        numerator=_sum_amount(filters=[EqFilter(column="has_fraudulent_dispute", value=True)]),
        denominator=_sum_amount(),
        scale=100,
    )
    assert _spec(measure=ratio).measure.kind == "ratio"

    with pytest.raises(ValidationError, match="scale"):
        RatioMeasure(numerator=_sum_amount(), denominator=_sum_amount(), scale=10)


def test_grouping_ordering_and_output_contracts_are_consistent():
    grouped = _spec(
        group_by=["shopper_interaction"],
        ordering=[OrderSpec(by="value", direction="asc")],
        limit=1,
        output=AnalysisOutputContract(kind="group_value_list", decimals=2),
    )
    assert grouped.limit == 1

    with pytest.raises(ValidationError, match="group_value_list"):
        _spec(output=AnalysisOutputContract(kind="group_value_list", decimals=2))

    with pytest.raises(ValidationError, match="integer output"):
        _spec(measure=AggregateMeasure(kind="mean", column="eur_amount"),
              output=AnalysisOutputContract(kind="integer"))

    with pytest.raises(ValidationError, match="ordering column"):
        _spec(group_by=["merchant"], ordering=[OrderSpec(by="country_code")])


def test_policy_ids_and_unresolved_axes_are_unique():
    with pytest.raises(ValidationError, match="policy_ids"):
        _spec(policy_ids=["p.v1", "p.v1"])

    with pytest.raises(ValidationError, match="unresolved_axes"):
        _spec(unresolved_axes=["fraud_basis", "fraud_basis"])


def test_analysis_spec_rejects_arbitrary_code_and_final_answers():
    payload = _spec().model_dump()
    with pytest.raises(ValidationError, match="python_code"):
        AnalysisSpec.model_validate({**payload, "python_code": "df.sum()"})

    with pytest.raises(ValidationError, match="final_answer"):
        AnalysisSpec.model_validate({**payload, "final_answer": "42"})
