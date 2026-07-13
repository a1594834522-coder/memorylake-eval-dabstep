"""Adopted-interpretation specs for the 15 hand-written routes (calibration set).

These transcribe each hand-written deterministic route's semantics into the
distill DSL. They are the pipeline's validity oracle: a compiled spec must
reproduce the hand-written route's answer byte-for-byte on real instances.
They also define the target shape for teacher-proposed hypotheses (Phase C).

Param extractors mirror the trigger regexes in deterministic_solver.py; the
signature layer (Phase B) replaces them with mechanical template-derived
parsers.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from dabstep_agent_pydantic.dabstep_core import mcc_code_for_description
from dabstep_agent_pydantic.runtime_util import MONTHS
from dabstep_agent_pydantic.distill.spec import (
    FeeRulesSpec,
    InterpretationSpec,
    OutputSpec,
    PaymentsSpec,
)

ParamFn = Callable[[str, Any], dict[str, Any] | None]


def _search(pattern: str, question: str) -> re.Match | None:
    return re.search(pattern, question, flags=re.IGNORECASE)


def _p_fee_ids_by_attributes(q: str, data) -> dict | None:
    m = _search(r"fee ID or IDs that apply to account_type = ([A-Za-z]) and aci = ([A-Za-z])", q)
    return {"account_type": m.group(1).upper(), "aci": m.group(2).upper()} if m else None


def _p_average_scheme_fee(q: str, data) -> dict | None:
    m = _search(
        r"(?:For (credit|debit) transactions, |For account type ([A-Za-z])(?: and the MCC description:? (.+?))?, )?"
        r"what would be the average fee that the card scheme (\w+) would charge for a transaction value of (\d+(?:\.\d+)?) EUR",
        q,
    )
    if not m:
        return None
    credit, account_type, mcc_desc, scheme, amount = m.groups()
    return {
        "card_scheme": scheme,
        "amount": float(amount),
        "is_credit": {"credit": True, "debit": False}.get((credit or "").lower()),
        "account_type": account_type,
        "merchant_category_code": mcc_code_for_description(data.merchant_category_codes, mcc_desc) if mcc_desc else None,
    }


def _p_most_expensive_mcc(q: str, data) -> dict | None:
    m = _search(r"most expensive MCC for a transaction of (\d+(?:\.\d+)?) euros, in general", q)
    return {"amount": float(m.group(1))} if m else None


def _p_total_fees(q: str, data) -> dict | None:
    m = _search(r"total fees .* that (.+?) paid in (?:([A-Za-z]+) )?(\d{4})", q)
    if not m:
        return None
    merchant, month, year = m.groups()
    return {"merchant": merchant, "year": int(year), "month": MONTHS.get((month or "").lower())}


def _p_mcc_fee_delta(q: str, data) -> dict | None:
    m = _search(r"merchant\s+(.+?)\s+had changed its MCC code to\s+(\d+)\s+before\s+(\d{4})\s+started", q)
    return {"merchant": m.group(1), "new_mcc": int(m.group(2)), "year": int(m.group(3))} if m else None


def _p_relative_fee_delta(q: str, data) -> dict | None:
    m = _search(
        r"In ([A-Za-z]+) (\d{4}) what delta would (.+?) pay if the relative fee of the fee with ID=(\d+) changed to (\d+(?:\.\d+)?)",
        q,
    )
    if not m:
        return None
    month, year, merchant, fee_id, value = m.groups()
    return {"merchant": merchant, "year": int(year), "month": MONTHS[month.lower()],
            "fee_id": int(fee_id), "new_value": float(value)}


def _p_relative_fee_rate_delta(q: str, data) -> dict | None:
    m = _search(
        r"In the year (\d{4}) what delta would (.+?) pay if the relative fee of the fee with ID=(\d+) changed to (\d+(?:\.\d+)?)",
        q,
    )
    if not m:
        return None
    year, merchant, fee_id, value = m.groups()
    return {"merchant": merchant, "year": int(year), "month": None, "fee_id": int(fee_id), "new_value": float(value)}


def _p_account_type_change(q: str, data) -> dict | None:
    m = _search(
        r"During (\d{4}), imagine if the Fee with ID (\d+) was only applied to account type ([A-Za-z]), which merchants",
        q,
    )
    return {"year": int(m.group(1)), "fee_id": int(m.group(2)), "account_type": m.group(3)} if m else None


def _p_fee_affected(q: str, data) -> dict | None:
    m = _search(r"In (\d{4}), which merchants were affected by the Fee with ID (\d+)", q)
    return {"year": int(m.group(1)), "fee_id": int(m.group(2)), "account_type": None} if m else None


def _p_applicable_fee_ids(q: str, data) -> dict | None:
    m = _search(r"applicable Fee IDs for (.+?) in ([A-Za-z]+) (\d{4})", q)
    return {"merchant": m.group(1), "month": MONTHS[m.group(2).lower()], "year": int(m.group(3))} if m else None


def _p_aci_optimization(q: str, data) -> dict | None:
    m = _search(r"For (.+?) in ([A-Za-z]+), if we were to move the fraudulent transactions", q)
    if m:
        return {"merchant": m.group(1), "month": MONTHS[m.group(2).lower()], "year": 2023}
    m = _search(r"year (\d{4}) and at the merchant (.+?), if we were to move the fraudulent transactions", q)
    if m:
        return {"merchant": m.group(2), "year": int(m.group(1)), "month": None}
    return None


def _p_repeat_pct(q: str, data) -> dict | None:
    m = _search(r"percentage of high-value transactions .*?above the (\d+)(?:st|nd|rd|th)? percentile.*?repeat customers", q)
    return {"percentile": float(m.group(1))} if m else None


def _p_none(q: str, data) -> dict | None:
    return {}


def _spec(name, population, *, fee_rules=None, payments=None, output, citation="manual §5") -> InterpretationSpec:
    return InterpretationSpec(
        name=name, population=population, fee_rules=fee_rules, payments=payments,
        output=output, manual_citation=citation,
    )


# route_id -> (adopted spec, param extractor)
CALIBRATION: dict[str, tuple[InterpretationSpec, ParamFn]] = {
    "fee_ids_by_attributes": (
        _spec("wildcard_aware", "fee_rules",
              fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], value="rule_id", reducer="collect_ids"),
              output=OutputSpec(kind="id_list")),
        _p_fee_ids_by_attributes,
    ),
    "average_scheme_fee": (
        _spec("unspecified_unfiltered_mean", "fee_rules",
              fee_rules=FeeRulesSpec(
                  context_dims=["card_scheme", "is_credit", "account_type", "merchant_category_code"],
                  value="fee_at_amount", reducer="mean"),
              output=OutputSpec(kind="decimal", decimals_default=6)),
        _p_average_scheme_fee,
    ),
    "most_expensive_mcc": (
        _spec("mean_aggregation", "fee_rules",
              fee_rules=FeeRulesSpec(context_dims=[], value="fee_at_amount", reducer="mean",
                                     group_by="merchant_category_code", group_extreme="argmax"),
              output=OutputSpec(kind="single_string", tie_policy="list_all_sorted")),
        _p_most_expensive_mcc,
    ),
    "total_fees": (
        _spec("sum_all_matching", "payments",
              payments=PaymentsSpec(primitive="period_total_fees", reducer="sum_all_matching"),
              output=OutputSpec(kind="decimal", decimals_default=2)),
        _p_total_fees,
    ),
    "mcc_fee_delta": (
        _spec("sum_all_matching", "payments",
              payments=PaymentsSpec(primitive="mcc_change_fee_delta"),
              output=OutputSpec(kind="decimal", decimals_default=6)),
        _p_mcc_fee_delta,
    ),
    "relative_fee_delta": (
        _spec("relative_rate_only", "payments",
              payments=PaymentsSpec(primitive="period_fee_rate_delta", delta_basis="rate"),
              output=OutputSpec(kind="decimal", decimals_default=14)),
        _p_relative_fee_delta,
    ),
    "relative_fee_rate_delta": (
        _spec("relative_rate_only", "payments",
              payments=PaymentsSpec(primitive="period_fee_rate_delta", delta_basis="rate"),
              output=OutputSpec(kind="decimal", decimals_default=14)),
        _p_relative_fee_rate_delta,
    ),
    "fee_account_type_change_affected_merchants": (
        _spec("losers_only", "payments",
              payments=PaymentsSpec(primitive="affected_merchants", affected_mode="losers_only"),
              output=OutputSpec(kind="string_list")),
        _p_account_type_change,
    ),
    "fee_affected_merchants": (
        _spec("baseline_members", "payments",
              payments=PaymentsSpec(primitive="affected_merchants", affected_mode="baseline_members"),
              output=OutputSpec(kind="string_list")),
        _p_fee_affected,
    ),
    "applicable_fee_ids": (
        _spec("full_period", "payments",
              payments=PaymentsSpec(primitive="applicable_fee_ids_period", tuple_scope="full_period"),
              output=OutputSpec(kind="id_list")),
        _p_applicable_fee_ids,
    ),
    "aci_fraud_optimization": (
        _spec("exclude_current_sum_all", "payments",
              payments=PaymentsSpec(primitive="steer_optimal_aci",
                                    aci_candidate_policy="exclude_current", reducer="sum_all_matching"),
              output=OutputSpec(kind="single_string")),
        _p_aci_optimization,
    ),
    "high_value_repeat_customer_percentage": (
        _spec("canonical", "payments",
              payments=PaymentsSpec(primitive="repeat_customer_percentage"),
              output=OutputSpec(kind="decimal", decimals_default=3)),
        _p_repeat_pct,
    ),
    "unique_merchant_count": (
        _spec("canonical", "payments",
              payments=PaymentsSpec(primitive="unique_merchant_count"),
              output=OutputSpec(kind="integer")),
        _p_none,
    ),
    "field_domain_values": (
        _spec("manual_union_observed", "payments",
              payments=PaymentsSpec(primitive="field_domain_values"),
              output=OutputSpec(kind="string_list")),
        _p_none,
    ),
}
