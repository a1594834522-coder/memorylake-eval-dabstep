import pandas as pd
import pytest

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.spec import (
    FeeRulesSpec,
    InterpretationSpec,
    OutputSpec,
    PaymentsSpec,
)
from dabstep_agent_pydantic.distill.combinators import SpecNotExecutable, compile_spec


def _fixture() -> DABStepData:
    return DABStepData(
        fees=pd.DataFrame(
            [
                {
                    "ID": 1, "card_scheme": "SwiftCharge", "account_type": [], "capture_delay": None,
                    "monthly_fraud_level": None, "monthly_volume": None,
                    "merchant_category_code": [7372], "is_credit": True, "aci": ["A"],
                    "fixed_amount": 0.10, "rate": 100, "intracountry": None,
                },
                {
                    "ID": 2, "card_scheme": "SwiftCharge", "account_type": ["H"], "capture_delay": None,
                    "monthly_fraud_level": None, "monthly_volume": None,
                    "merchant_category_code": [], "is_credit": None, "aci": [],
                    "fixed_amount": 0.50, "rate": 0, "intracountry": None,
                },
                {
                    "ID": 3, "card_scheme": "NexPay", "account_type": [], "capture_delay": None,
                    "monthly_fraud_level": None, "monthly_volume": None,
                    "merchant_category_code": [5812], "is_credit": False, "aci": ["B"],
                    "fixed_amount": 0.20, "rate": 50, "intracountry": None,
                },
            ]
        ),
        payments=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant", "year": 2023, "day_of_year": 92,
                    "card_scheme": "SwiftCharge", "is_credit": True, "aci": "A",
                    "eur_amount": 100.0, "issuing_country": "NL", "acquirer": "acq_nl",
                    "has_fraudulent_dispute": False,
                },
                {
                    "merchant": "Synthetic_Merchant", "year": 2023, "day_of_year": 93,
                    "card_scheme": "SwiftCharge", "is_credit": True, "aci": "A",
                    "eur_amount": 50.0, "issuing_country": "NL", "acquirer": "acq_nl",
                    "has_fraudulent_dispute": True,
                },
            ]
        ),
        merchants=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant", "account_type": "H", "capture_delay": "1",
                    "merchant_category_code": 7372, "acquirer": ["acq_nl"],
                }
            ]
        ),
        acquirer_countries=pd.DataFrame([{"acquirer": "acq_nl", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame(
            [{"mcc": 7372, "description": "Computer programming"}, {"mcc": 5812, "description": "Eating places"}]
        ),
    )


def _out(kind, **kw):
    return OutputSpec(kind=kind, **kw)


def test_fee_rules_collect_ids_wildcard_semantics():
    spec = InterpretationSpec(
        name="wildcard_aware", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], reducer="collect_ids", value="rule_id"),
        output=_out("id_list"), manual_citation="manual §5 null/[] wildcard",
    )
    fn = compile_spec(spec)
    # account_type=H, aci=A: rule 1 ([] account matches, A in aci) + rule 2 (H, [] aci) — not rule 3
    assert fn(_fixture(), {"account_type": "H", "aci": "A"}, "") == "1, 2"


def test_fee_rules_mean_vs_sum_discriminate_differently():
    base = dict(context_dims=["card_scheme"], value="fee_at_amount")
    mean_spec = InterpretationSpec(
        name="mean", population="fee_rules",
        fee_rules=FeeRulesSpec(**base, reducer="mean"),
        output=_out("decimal", decimals_default=6), manual_citation="manual §5",
    )
    sum_spec = InterpretationSpec(
        name="sum", population="fee_rules",
        fee_rules=FeeRulesSpec(**base, reducer="sum"),
        output=_out("decimal", decimals_default=6), manual_citation="manual §5",
    )
    params = {"card_scheme": "SwiftCharge", "amount": 100.0}
    # rule1: 0.10+100*100/10000=1.10 ; rule2: 0.50+0=0.50
    assert compile_spec(mean_spec)(_fixture(), params, "") == "0.800000"
    assert compile_spec(sum_spec)(_fixture(), params, "") == "1.600000"


def test_fee_rules_grouped_extreme_with_wildcard_groups():
    spec = InterpretationSpec(
        name="most_expensive_mcc", population="fee_rules",
        fee_rules=FeeRulesSpec(
            context_dims=[], reducer="mean", value="fee_at_amount",
            group_by="merchant_category_code", group_extreme="argmax",
        ),
        output=_out("single_string", tie_policy="list_all_sorted"),
        manual_citation="manual §5",
    )
    # amount=10: rule1=0.2, rule2=0.5 (wildcard joins both groups), rule3=0.25
    # group 7372: mean(0.2, 0.5)=0.35 ; group 5812: mean(0.25, 0.5)=0.375 -> 5812 wins
    assert compile_spec(spec)(_fixture(), {"amount": 10.0}, "") == "5812"


def test_payments_total_fees_sum_vs_min_match():
    sum_spec = InterpretationSpec(
        name="sum_all", population="payments",
        payments=PaymentsSpec(primitive="period_total_fees", reducer="sum_all_matching"),
        output=_out("decimal", decimals_default=2), manual_citation="manual §5",
    )
    min_spec = InterpretationSpec(
        name="min_match", population="payments",
        payments=PaymentsSpec(primitive="period_total_fees", reducer="min_match"),
        output=_out("decimal", decimals_default=2), manual_citation="manual §5",
    )
    params = {"merchant": "Synthetic_Merchant", "year": 2023, "month": 4}
    # txn 100: rule1=1.10, rule2=0.50 -> sum 1.60 min 0.50
    # txn 50:  rule1=0.60, rule2=0.50 -> sum 1.10 min 0.50
    data = _fixture()
    assert compile_spec(sum_spec)(data, params, "") == "2.70"
    assert compile_spec(min_spec)(data, params, "") == "1.00"


def test_payments_affected_merchants_modes_diverge():
    losers = InterpretationSpec(
        name="losers", population="payments",
        payments=PaymentsSpec(primitive="affected_merchants", affected_mode="losers_only"),
        output=_out("string_list"), manual_citation="manual §5",
    )
    sym = InterpretationSpec(
        name="sym", population="payments",
        payments=PaymentsSpec(primitive="affected_merchants", affected_mode="symmetric_difference"),
        output=_out("string_list"), manual_citation="manual §5",
    )
    data = _fixture()
    # fee 1 applies to merchant (account []) ; restricted to "X" -> merchant loses
    params = {"year": 2023, "fee_id": 1, "account_type": "X"}
    assert compile_spec(losers)(data, params, "") == "Synthetic_Merchant"
    assert compile_spec(sym)(data, params, "") == "Synthetic_Merchant"
    # restricted to merchant's own type "H" -> no change: losers empty, sym empty
    params_h = {"year": 2023, "fee_id": 1, "account_type": "H"}
    assert compile_spec(losers)(data, params_h, "") == ""
    assert compile_spec(sym)(data, params_h, "") == ""


def test_empty_match_raises_spec_not_executable():
    spec = InterpretationSpec(
        name="mean", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["card_scheme"], reducer="mean"),
        output=_out("decimal"), manual_citation="manual §5",
    )
    with pytest.raises(SpecNotExecutable):
        compile_spec(spec)(_fixture(), {"card_scheme": "NoSuchScheme", "amount": 1.0}, "")
