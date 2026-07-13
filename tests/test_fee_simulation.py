import os

import pandas as pd
import pytest

from dabstep_agent_pydantic import dabstep_core
from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.dabstep_core import add_intracountry_flag
from dabstep_agent_pydantic.dabstep_core import fee_affected_merchants_for_year
from dabstep_agent_pydantic.dabstep_core import fee_fixed_component_delta_for_month
from dabstep_agent_pydantic.dabstep_core import fee_rate_delta_for_period
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.dabstep_core import match_count_summary
from dabstep_agent_pydantic.dabstep_core import matches_capture_delay
from dabstep_agent_pydantic.dataset import Task
from calibration_solver import _cached_load_dabstep_data_by_path
from calibration_solver import try_solve_deterministic


def test_nan_capture_delay_matches_all_values():
    assert matches_capture_delay(float("nan"), "1") is True


def test_aci_optimization_treats_nan_capture_delay_as_wildcard():
    data = DABStepData(
        fees=pd.DataFrame(
            [
                {
                    "ID": 1,
                    "card_scheme": "SwiftCharge",
                    "account_type": [],
                    "capture_delay": float("nan"),
                    "monthly_fraud_level": None,
                    "monthly_volume": None,
                    "merchant_category_code": [],
                    "is_credit": True,
                    "aci": ["E"],
                    "fixed_amount": 0.01,
                    "rate": 1,
                    "intracountry": None,
                },
                {
                    "ID": 2,
                    "card_scheme": "SwiftCharge",
                    "account_type": [],
                    "capture_delay": None,
                    "monthly_fraud_level": None,
                    "monthly_volume": None,
                    "merchant_category_code": [],
                    "is_credit": True,
                    "aci": ["F"],
                    "fixed_amount": 0.20,
                    "rate": 1,
                    "intracountry": None,
                },
            ]
        ),
        payments=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant",
                    "year": 2023,
                    "day_of_year": 10,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "G",
                    "eur_amount": 100.0,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": True,
                }
            ]
        ),
        merchants=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant",
                    "account_type": "R",
                    "capture_delay": "1",
                    "merchant_category_code": 1234,
                    "acquirer": ["acq_nl"],
                }
            ]
        ),
        acquirer_countries=pd.DataFrame([{"acquirer": "acq_nl", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame(),
    )

    result = dabstep_core.optimize_aci_for_fraudulent_transactions(
        data,
        merchant="Synthetic_Merchant",
        year=2023,
        month=1,
        candidate_acis=["E", "F"],
    )

    assert result["aci"] == "E"


def test_fee_fixed_component_delta_counts_only_matching_transactions():
    data = _fee_fixture()

    value = fee_fixed_component_delta_for_month(
        data,
        merchant="Synthetic_Merchant",
        year=2023,
        month=4,
        fee_id=101,
        new_fixed_amount=0,
    )

    assert value == pytest.approx(-0.12)


def test_monthly_relative_fee_change_routes_to_rate_delta(monkeypatch, tmp_path):
    data = _fee_fixture()
    monkeypatch.setattr("calibration_solver.load_dabstep_data", lambda data_dir: data)
    task = Task(
        task_id="synthetic-relative-fee",
        question=(
            "In April 2023 what delta would Synthetic_Merchant pay if the relative fee of the fee "
            "with ID=101 changed to 1?"
        ),
        guidelines="Return the answer rounded to 14 decimals.",
    )

    answer = try_solve_deterministic(task, data_dir=tmp_path)

    assert answer is not None
    assert answer.route == "relative_fee_delta"
    assert answer.agent_answer == "-0.09015200000000"


def test_fee_rate_delta_for_period_accumulates_matching_months():
    data = _fee_fixture()

    value = fee_rate_delta_for_period(
        data,
        merchant="Synthetic_Merchant",
        year=2023,
        fee_id=101,
        new_rate=99,
    )

    assert value == pytest.approx(0.059592)


def test_annual_relative_fee_changed_to_rate_routes_to_rate_delta(monkeypatch, tmp_path):
    data = _fee_fixture()
    monkeypatch.setattr("calibration_solver.load_dabstep_data", lambda data_dir: data)
    task = Task(
        task_id="synthetic-annual-relative-fee",
        question=(
            "In the year 2023 what delta would Synthetic_Merchant pay if the relative fee of the fee "
            "with ID=101 changed to 99?"
        ),
        guidelines="Answer must be just a number rounded to 14 decimals.",
    )

    answer = try_solve_deterministic(task, data_dir=tmp_path)

    assert answer is not None
    assert answer.route == "relative_fee_rate_delta"
    assert answer.agent_answer == "0.05959200000000"


def test_fee_affected_merchants_for_year_compares_optional_account_type_change():
    data = _fee_fixture()

    original = fee_affected_merchants_for_year(data, year=2023, fee_id=101)
    changed = fee_affected_merchants_for_year(data, year=2023, fee_id=101, only_account_type="X")

    assert original == ["Synthetic_Merchant"]
    assert changed == ["Synthetic_Merchant"]


def test_fee_affected_merchants_for_year_honors_monthly_thresholds():
    data = DABStepData(
        fees=pd.DataFrame(
            [
                {
                    "ID": 202,
                    "card_scheme": "SwiftCharge",
                    "account_type": [],
                    "capture_delay": None,
                    "monthly_fraud_level": ">10%",
                    "monthly_volume": ">100",
                    "merchant_category_code": [],
                    "is_credit": True,
                    "aci": ["A"],
                    "fixed_amount": 0.0,
                    "rate": 1,
                    "intracountry": None,
                }
            ]
        ),
        payments=pd.DataFrame(
            [
                {
                    "merchant": "Alpha_Merchant",
                    "year": 2023,
                    "day_of_year": 10,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "A",
                    "eur_amount": 90.0,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": False,
                },
                {
                    "merchant": "Alpha_Merchant",
                    "year": 2023,
                    "day_of_year": 11,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "A",
                    "eur_amount": 30.0,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": True,
                },
                {
                    "merchant": "Beta_Merchant",
                    "year": 2023,
                    "day_of_year": 10,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "A",
                    "eur_amount": 120.0,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": False,
                },
            ]
        ),
        merchants=pd.DataFrame(
            [
                {
                    "merchant": "Alpha_Merchant",
                    "account_type": "D",
                    "capture_delay": "7",
                    "merchant_category_code": 7372,
                    "acquirer": ["acq_nl"],
                },
                {
                    "merchant": "Beta_Merchant",
                    "account_type": "D",
                    "capture_delay": "7",
                    "merchant_category_code": 7372,
                    "acquirer": ["acq_nl"],
                },
            ]
        ),
        acquirer_countries=pd.DataFrame([{"acquirer": "acq_nl", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 7372, "description": "Computer programming"}]),
    )

    assert fee_affected_merchants_for_year(data, year=2023, fee_id=202) == ["Alpha_Merchant"]


def test_fee_affected_merchants_questions_route_deterministically(monkeypatch, tmp_path):
    data = _fee_fixture()
    monkeypatch.setattr("calibration_solver.load_dabstep_data", lambda data_dir: data)

    affected = try_solve_deterministic(
        Task(
            task_id="synthetic-affected-fee",
            question="In 2023, which merchants were affected by the Fee with ID 101?",
            guidelines="Answer must be a list of values in comma separated list, eg: A, B, C.",
        ),
        data_dir=tmp_path,
    )
    changed = try_solve_deterministic(
        Task(
            task_id="synthetic-changed-fee",
            question=(
                "During 2023, imagine if the Fee with ID 101 was only applied to account type X, "
                "which merchants would have been affected by this change?"
            ),
            guidelines="Answer must be a list of values in comma separated list, eg: A, B, C.",
        ),
        data_dir=tmp_path,
    )

    assert affected is not None
    assert affected.route == "fee_affected_merchants"
    assert affected.agent_answer == "Synthetic_Merchant"
    assert changed is not None
    assert changed.route == "fee_account_type_change_affected_merchants"
    assert changed.agent_answer == "Synthetic_Merchant"


def test_deterministic_solver_reuses_loaded_data_for_same_context(monkeypatch, tmp_path):
    _cached_load_dabstep_data_by_path.cache_clear()
    data = _fee_fixture()
    calls = []

    def fake_load(data_dir):
        calls.append(data_dir)
        return data

    monkeypatch.setattr("calibration_solver.load_dabstep_data", fake_load)

    first = try_solve_deterministic(
        Task(
            task_id="synthetic-affected-fee",
            question="In 2023, which merchants were affected by the Fee with ID 101?",
            guidelines="Answer must be a list of values in comma separated list, eg: A, B, C.",
        ),
        data_dir=tmp_path,
    )
    second = try_solve_deterministic(
        Task(
            task_id="synthetic-total-fees",
            question="What were the total fees in EUR that Synthetic_Merchant paid in 2023?",
            guidelines="Answer must be just a number rounded to 2 decimals.",
        ),
        data_dir=tmp_path,
    )

    assert first is not None
    assert second is not None
    assert len(calls) == 1
    _cached_load_dabstep_data_by_path.cache_clear()


def test_match_count_summary_warns_when_all_rules_are_filtered_out():
    payments = _probe_payments()
    fees = [
        _probe_fee(1, card_scheme="OtherScheme"),
    ]

    summary = match_count_summary(payments, fees, context_base=_probe_context_base())

    assert summary == {
        "transactions": 2,
        "zero_match": 2,
        "single_match": 0,
        "multi_match": 0,
        "max_matches": 0,
        "warning": "2 transactions match no fee rule — check wildcard handling (null/[] means match-all)",
    }


def test_match_count_summary_detects_wildcard_matching_sabotage(monkeypatch):
    payments = _probe_payments()
    fees = [
        _probe_fee(1, account_type=["D"]),
        _probe_fee(2, account_type=[]),
    ]

    summary = match_count_summary(payments, fees, context_base=_probe_context_base())

    assert summary["zero_match"] == 0
    assert summary["multi_match"] == 2
    assert summary["max_matches"] == 2
    assert summary["warning"] is None

    original = dabstep_core._matches_fee_for_payment

    def strict_list_sabotage(fee, context):
        if fee.get("account_type") == []:
            return False
        return original(fee, context)

    monkeypatch.setattr(dabstep_core, "_matches_fee_for_payment", strict_list_sabotage)

    sabotaged = match_count_summary(payments, fees, context_base=_probe_context_base())

    assert sabotaged["multi_match"] == 0
    assert sabotaged["warning"] == "no transaction matches more than one rule — filters may be too strict"


def test_match_count_summary_normal_fixture_has_no_warning():
    payments = _probe_payments()
    fees = [
        _probe_fee(1, account_type=[]),
        _probe_fee(2, aci=[]),
    ]

    summary = match_count_summary(payments, fees, context_base=_probe_context_base())

    assert summary["transactions"] == 2
    assert summary["zero_match"] == 0
    assert summary["single_match"] == 0
    assert summary["multi_match"] == 2
    assert summary["warning"] is None


def test_match_count_summary_matches_sum_all_fee_hit_count():
    payments = _probe_payments()
    fees = [
        _probe_fee(1, fixed_amount=1.0, rate=0),
        _probe_fee(2, account_type=[], fixed_amount=1.0, rate=0),
    ]
    context_base = _probe_context_base()

    summary = match_count_summary(payments, fees, context_base=context_base)
    fee_total = dabstep_core._sum_matching_fees_for_payment_groups(
        payments,
        context_base=context_base,
        fee_rows=fees,
    )

    assert summary["single_match"] + 2 * summary["multi_match"] == int(fee_total)


def test_match_count_summary_real_data_spot_check_finds_unmatched_transactions():
    data_dir = os.getenv("DABSTEP_CONTEXT_DIR") or os.getenv("DABSTEP_DATA_DIR")
    if not data_dir:
        pytest.skip("set DABSTEP_CONTEXT_DIR to run the real-data match-count probe")
    data = load_dabstep_data(data_dir)
    payments = add_intracountry_flag(data.payments, data.acquirer_countries)
    payments = payments.copy()
    payments["month"] = payments.apply(
        lambda row: dabstep_core._month_from_day_of_year(int(row["year"]), int(row["day_of_year"])),
        axis=1,
    )
    candidate_keys = (
        payments[payments["aci"].astype(str).eq("G")]
        .groupby(["merchant", "year", "month"], dropna=False)
        .size()
        .reset_index()[["merchant", "year", "month"]]
        .to_dict(orient="records")
    )
    merchant_rows = data.merchants.set_index("merchant").to_dict(orient="index")
    for candidate in candidate_keys:
        merchant = str(candidate["merchant"])
        year = int(candidate["year"])
        month = int(candidate["month"])
        start, end = dabstep_core.get_month_day_range(year, month)
        month_payments = payments[
            (payments["merchant"].astype(str) == merchant)
            & (payments["year"].astype(int) == year)
            & (payments["day_of_year"].astype(int).between(start, end))
        ].copy()
        if month_payments.empty or merchant not in merchant_rows:
            continue
        context_base = dabstep_core._merchant_month_context(
            data,
            merchant_row=merchant_rows[merchant],
            payments=month_payments,
        )
        summary = match_count_summary(month_payments, data.fees, context_base=context_base)
        if summary["zero_match"] > 0:
            return
    pytest.fail("expected at least one real merchant-month with unmatched ACI transactions")


def _fee_fixture() -> DABStepData:
    return DABStepData(
        fees=pd.DataFrame(
            [
                {
                    "ID": 101,
                    "card_scheme": "SwiftCharge",
                    "account_type": [],
                    "capture_delay": ">5",
                    "monthly_fraud_level": None,
                    "monthly_volume": None,
                    "merchant_category_code": [],
                    "is_credit": True,
                    "aci": ["A"],
                    "fixed_amount": 0.12,
                    "rate": 60,
                    "intracountry": None,
                }
            ]
        ),
        payments=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant",
                    "year": 2023,
                    "day_of_year": 92,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "A",
                    "eur_amount": 15.28,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": False,
                },
                {
                    "merchant": "Synthetic_Merchant",
                    "year": 2023,
                    "day_of_year": 93,
                    "card_scheme": "SwiftCharge",
                    "is_credit": True,
                    "aci": "B",
                    "eur_amount": 20.0,
                    "issuing_country": "NL",
                    "acquirer": "acq_nl",
                    "has_fraudulent_dispute": False,
                },
            ]
        ),
        merchants=pd.DataFrame(
            [
                {
                    "merchant": "Synthetic_Merchant",
                    "account_type": "D",
                    "capture_delay": "7",
                    "merchant_category_code": 7372,
                    "acquirer": ["acq_nl"],
                }
            ]
        ),
        acquirer_countries=pd.DataFrame([{"acquirer": "acq_nl", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 7372, "description": "Computer programming"}]),
    )


def _probe_payments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "card_scheme": "SwiftCharge",
                "is_credit": True,
                "aci": "A",
                "intracountry": True,
                "eur_amount": 10.0,
            },
            {
                "card_scheme": "SwiftCharge",
                "is_credit": True,
                "aci": "A",
                "intracountry": True,
                "eur_amount": 20.0,
            },
        ]
    )


def _probe_context_base() -> dict[str, object]:
    return {
        "account_type": "D",
        "capture_delay": "7",
        "merchant_category_code": 1234,
        "monthly_volume": 30.0,
        "monthly_fraud_rate_pct": 0.0,
    }


def _probe_fee(
    fee_id: int,
    *,
    card_scheme: str = "SwiftCharge",
    account_type=None,
    aci=None,
    fixed_amount: float = 0.0,
    rate: int = 0,
) -> dict[str, object]:
    return {
        "ID": fee_id,
        "card_scheme": card_scheme,
        "account_type": ["D"] if account_type is None else account_type,
        "capture_delay": None,
        "monthly_fraud_level": None,
        "monthly_volume": None,
        "merchant_category_code": [],
        "is_credit": True,
        "aci": ["A"] if aci is None else aci,
        "fixed_amount": fixed_amount,
        "rate": rate,
        "intracountry": None,
    }


def test_most_expensive_mcc_treats_empty_mcc_lists_as_wildcards(monkeypatch, tmp_path):
    data = _fee_fixture()
    data.fees.at[0, "merchant_category_code"] = [7372]
    wildcard_rule = {
        "ID": 102,
        "card_scheme": "SwiftCharge",
        "account_type": [],
        "capture_delay": None,
        "monthly_fraud_level": None,
        "monthly_volume": None,
        "merchant_category_code": [],
        "is_credit": None,
        "aci": [],
        "fixed_amount": 0.05,
        "rate": 10,
        "intracountry": None,
    }
    cheap_rule = {**wildcard_rule, "ID": 103, "merchant_category_code": [8062], "fixed_amount": 0.01, "rate": 1}
    data.fees.loc[len(data.fees)] = wildcard_rule
    data.fees.loc[len(data.fees)] = cheap_rule
    monkeypatch.setattr("calibration_solver.load_dabstep_data", lambda data_dir: data)

    task = Task(
        task_id="synthetic-expensive-mcc",
        question="What is the most expensive MCC for a transaction of 10 euros, in general?",
        guidelines="Answer must be a list of values in comma separated list, eg: A, B, C.",
    )
    answer = try_solve_deterministic(task, data_dir=tmp_path)

    assert answer is not None
    assert answer.route == "most_expensive_mcc"
    # 7372: mean(0.12+60*10/1e4, 0.05+10*10/1e4) = mean(0.18, 0.06) = 0.12
    # 8062: mean(0.01+1*10/1e4, 0.06) = 0.0355 -> 7372 wins alone
    assert answer.agent_answer == "7372"


def test_fee_ids_by_attributes_routes_deterministically(monkeypatch, tmp_path):
    data = _fee_fixture()
    data.fees.at[0, "aci"] = ["A"]
    wildcard_rule = {
        "ID": 102,
        "card_scheme": "SwiftCharge",
        "account_type": [],
        "capture_delay": None,
        "monthly_fraud_level": None,
        "monthly_volume": None,
        "merchant_category_code": [],
        "is_credit": None,
        "aci": [],
        "fixed_amount": 0.05,
        "rate": 10,
        "intracountry": None,
    }
    other_rule = {**wildcard_rule, "ID": 103, "aci": ["B"], "account_type": ["X"]}
    data.fees.loc[len(data.fees)] = wildcard_rule
    data.fees.loc[len(data.fees)] = other_rule
    monkeypatch.setattr(
        "calibration_solver._cached_load_dabstep_data", lambda data_dir: data
    )

    answer = try_solve_deterministic(
        Task(
            task_id="synthetic-fee-ids-attrs",
            question="What is the fee ID or IDs that apply to account_type = H and aci = A?",
            guidelines="Answer must be a list of values in comma separated list, eg: A, B, C.",
        ),
        data_dir=tmp_path,
    )

    assert answer is not None
    assert answer.route == "fee_ids_by_attributes"
    # rule 101: aci=[A] account_type=[] wildcard -> match; 102: full wildcard -> match; 103: aci=[B] -> no
    assert answer.agent_answer == "101, 102"


def test_average_scheme_fee_routes_deterministically(monkeypatch, tmp_path):
    data = _fee_fixture()
    monkeypatch.setattr(
        "calibration_solver._cached_load_dabstep_data", lambda data_dir: data
    )

    answer = try_solve_deterministic(
        Task(
            task_id="synthetic-avg-scheme-fee",
            question=(
                "For credit transactions, what would be the average fee that the card scheme "
                "SwiftCharge would charge for a transaction value of 100 EUR?"
            ),
            guidelines="Answer must be just a number rounded to 6 decimals.",
        ),
        data_dir=tmp_path,
    )

    assert answer is not None
    assert answer.route == "average_scheme_fee"
    # single credit rule: 0.12 + 60*100/10000 = 0.72
    assert answer.agent_answer == "0.720000"
