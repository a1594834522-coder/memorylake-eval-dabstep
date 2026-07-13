from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dabstep_agent_pydantic.runtime_util import MONTHS as _SHARED_MONTHS  # noqa: F401
from dabstep_agent_pydantic.runtime_util import cached_load_dabstep_data as _cached_load_dabstep_data_shared

from dabstep_agent_pydantic.dabstep_core import aci_fee_extreme_for_scheme_transaction
from dabstep_agent_pydantic.dabstep_core import applicable_fee_ids_for_merchant_period
from dabstep_agent_pydantic.dabstep_core import average_scheme_fee
from dabstep_agent_pydantic.dabstep_core import field_domain_values
from dabstep_agent_pydantic.dabstep_core import fee_affected_merchants_for_year
from dabstep_agent_pydantic.dabstep_core import fee_rate_delta_for_period
from dabstep_agent_pydantic.dabstep_core import format_decimal_places
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.dabstep_core import matching_fee_ids
from dabstep_agent_pydantic.dabstep_core import mcc_code_for_description
from dabstep_agent_pydantic.dabstep_core import merchant_mcc_fee_delta_for_year
from dabstep_agent_pydantic.dabstep_core import most_expensive_mccs_for_amount
from dabstep_agent_pydantic.dabstep_core import optimize_aci_for_fraudulent_transactions
from dabstep_agent_pydantic.dabstep_core import repeat_customer_percentage
from dabstep_agent_pydantic.dabstep_core import total_fees_for_merchant_period
from dabstep_agent_pydantic.dabstep_core import unique_merchant_count
from dabstep_agent_pydantic.dataset import Task


@dataclass(frozen=True)
class DeterministicAnswer:
    agent_answer: str
    reasoning: str
    route: str


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def try_solve_deterministic(task: Task, *, data_dir: Path) -> DeterministicAnswer | None:
    question = " ".join(task.question.split())
    guidelines = task.guidelines or ""
    data = None

    if _looks_like_account_type_domain(question):
        data = _cached_load_dabstep_data(data_dir)
        values = field_domain_values(
            "account_type",
            frames=[data.fees, data.merchants],
        )
        return DeterministicAnswer(
            agent_answer=", ".join(values),
            reasoning="Deterministically combined manual-defined and observed account_type domain values.",
            route="field_domain_values",
        )

    if _looks_like_unique_merchant_count(question):
        data = _cached_load_dabstep_data(data_dir)
        value = unique_merchant_count(data.payments, merchant_profiles=data.merchants)
        return DeterministicAnswer(
            agent_answer=str(value),
            reasoning="Deterministically counted distinct merchant values in the payments fact table.",
            route="unique_merchant_count",
        )

    high_value_repeat_match = re.search(
        r"percentage of high-value transactions .*?above the (?P<percentile>\d+)(?:st|nd|rd|th)? percentile.*?repeat customers",
        question,
        flags=re.IGNORECASE,
    )
    if high_value_repeat_match:
        data = _cached_load_dabstep_data(data_dir)
        quantile = int(high_value_repeat_match.group("percentile")) / 100
        value = repeat_customer_percentage(
            data.payments,
            identity_field="email_address",
            high_value_quantile=quantile,
        )
        places = _decimal_places(guidelines, default=3)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning="Deterministically computed repeat-customer percentage for high-value transactions using repeat status from the full payments table.",
            route="high_value_repeat_customer_percentage",
        )

    expensive_mcc_match = re.search(
        r"most expensive MCC for a transaction of (?P<amount>\d+(?:\.\d+)?) euros, in general",
        question,
        flags=re.IGNORECASE,
    )
    if expensive_mcc_match:
        data = _cached_load_dabstep_data(data_dir)
        mccs = most_expensive_mccs_for_amount(data, amount=float(expensive_mcc_match.group("amount")))
        return DeterministicAnswer(
            agent_answer=", ".join(str(mcc) for mcc in mccs),
            reasoning=(
                "Deterministically ranked MCCs by mean fee across all applicable rules, "
                "treating empty MCC lists as wildcards."
            ),
            route="most_expensive_mcc",
        )

    fee_ids_by_attrs_match = re.search(
        r"fee ID or IDs that apply to account_type = (?P<account_type>[A-Za-z]) and aci = (?P<aci>[A-Za-z])",
        question,
        flags=re.IGNORECASE,
    )
    if fee_ids_by_attrs_match:
        data = _cached_load_dabstep_data(data_dir)
        ids = matching_fee_ids(
            data.fees,
            {
                "account_type": fee_ids_by_attrs_match.group("account_type").upper(),
                "aci": fee_ids_by_attrs_match.group("aci").upper(),
            },
        )
        return DeterministicAnswer(
            agent_answer=", ".join(str(item) for item in ids),
            reasoning=(
                "Deterministically matched fee rules against the requested account_type and ACI "
                "with wildcard-aware semantics (null scalars and empty lists match any value)."
            ),
            route="fee_ids_by_attributes",
        )

    avg_scheme_fee_match = re.search(
        r"(?:For (?P<credit>credit|debit) transactions, |For account type (?P<account_type>[A-Za-z])(?: and the MCC description:? (?P<mcc_description>.+?))?, )?"
        r"what would be the average fee that the card scheme (?P<scheme>\w+) would charge for a transaction value of (?P<amount>\d+(?:\.\d+)?) EUR",
        question,
        flags=re.IGNORECASE,
    )
    if avg_scheme_fee_match:
        data = _cached_load_dabstep_data(data_dir)
        mcc_description = avg_scheme_fee_match.group("mcc_description")
        credit_text = (avg_scheme_fee_match.group("credit") or "").lower()
        value = average_scheme_fee(
            data.fees,
            card_scheme=avg_scheme_fee_match.group("scheme"),
            transaction_value=float(avg_scheme_fee_match.group("amount")),
            is_credit={"credit": True, "debit": False}.get(credit_text),
            account_type=avg_scheme_fee_match.group("account_type"),
            merchant_category_code=(
                mcc_code_for_description(data.merchant_category_codes, mcc_description)
                if mcc_description
                else None
            ),
        )
        places = _decimal_places(guidelines, default=6)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning=(
                "Deterministically averaged the fee over all rules matching the stated constraints "
                "with wildcard-aware semantics, leaving unspecified dimensions unfiltered."
            ),
            route="average_scheme_fee",
        )

    aci_extreme_match = re.search(
        r"For a (?P<credit>credit|debit)?\s*transaction of (?P<amount>\d+(?:\.\d+)?) euros on (?P<scheme>.+?), what would be the (?P<objective>most|least) expensive .*?ACI",
        question,
        flags=re.IGNORECASE,
    )
    if aci_extreme_match:
        data = _cached_load_dabstep_data(data_dir)
        credit_text = aci_extreme_match.group("credit")
        result = aci_fee_extreme_for_scheme_transaction(
            data.fees,
            card_scheme=aci_extreme_match.group("scheme"),
            transaction_value=float(aci_extreme_match.group("amount")),
            is_credit=True if credit_text and credit_text.lower() == "credit" else False if credit_text else None,
            objective="max" if aci_extreme_match.group("objective").lower() == "most" else "min",
        )
        return DeterministicAnswer(
            agent_answer=str(result["aci"]),
            reasoning="Deterministically selected the ACI with the extreme sum-all fee for the hypothetical transaction.",
            route="aci_fee_extreme",
        )

    mcc_match = re.search(
        r"merchant\s+(?P<merchant>.+?)\s+had changed its MCC code to\s+(?P<mcc>\d+)\s+before\s+(?P<year>\d{4})\s+started",
        question,
        flags=re.IGNORECASE,
    )
    if mcc_match:
        data = _cached_load_dabstep_data(data_dir)
        value = merchant_mcc_fee_delta_for_year(
            data,
            merchant=mcc_match.group("merchant"),
            year=int(mcc_match.group("year")),
            new_mcc=int(mcc_match.group("mcc")),
        )
        places = _decimal_places(guidelines, default=6)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning="Deterministically computed MCC-change fee delta with yearly total-fee matching semantics.",
            route="mcc_fee_delta",
        )

    total_match = re.search(
        r"total fees .* that (?P<merchant>.+?) paid in (?:(?P<month>[A-Za-z]+) )?(?P<year>\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if total_match:
        data = _cached_load_dabstep_data(data_dir)
        month_text = total_match.group("month")
        month = MONTHS.get(month_text.lower()) if month_text else None
        value = total_fees_for_merchant_period(
            data,
            merchant=total_match.group("merchant"),
            year=int(total_match.group("year")),
            month=month,
        )
        places = _decimal_places(guidelines, default=2)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning="Deterministically computed total fees with sum-all matching rules.",
            route="total_fees",
        )

    changed_fee_affected_match = re.search(
        r"During (?P<year>\d{4}), imagine if the Fee with ID (?P<fee_id>\d+) "
        r"was only applied to account type (?P<account_type>[A-Za-z]), which merchants "
        r"would have been affected by this change",
        question,
        flags=re.IGNORECASE,
    )
    if changed_fee_affected_match:
        data = _cached_load_dabstep_data(data_dir)
        merchants = fee_affected_merchants_for_year(
            data,
            year=int(changed_fee_affected_match.group("year")),
            fee_id=int(changed_fee_affected_match.group("fee_id")),
            only_account_type=changed_fee_affected_match.group("account_type"),
        )
        return DeterministicAnswer(
            agent_answer=", ".join(merchants),
            reasoning="Deterministically listed merchants that would lose the fee under the account-type restriction.",
            route="fee_account_type_change_affected_merchants",
        )

    fee_affected_match = re.search(
        r"In (?P<year>\d{4}), which merchants were affected by the Fee with ID (?P<fee_id>\d+)",
        question,
        flags=re.IGNORECASE,
    )
    if fee_affected_match:
        data = _cached_load_dabstep_data(data_dir)
        merchants = fee_affected_merchants_for_year(
            data,
            year=int(fee_affected_match.group("year")),
            fee_id=int(fee_affected_match.group("fee_id")),
        )
        return DeterministicAnswer(
            agent_answer=", ".join(merchants),
            reasoning="Deterministically listed merchants with at least one yearly transaction matching the fee rule.",
            route="fee_affected_merchants",
        )

    relative_fee_match = re.search(
        r"In (?P<month>[A-Za-z]+) (?P<year>\d{4}) what delta would (?P<merchant>.+?) pay "
        r"if the relative fee of the fee with ID=(?P<fee_id>\d+) changed to "
        r"(?P<value>\d+(?:\.\d+)?)",
        question,
        flags=re.IGNORECASE,
    )
    if relative_fee_match:
        data = _cached_load_dabstep_data(data_dir)
        value = fee_rate_delta_for_period(
            data,
            merchant=relative_fee_match.group("merchant"),
            year=int(relative_fee_match.group("year")),
            month=MONTHS[relative_fee_match.group("month").lower()],
            fee_id=int(relative_fee_match.group("fee_id")),
            new_rate=float(relative_fee_match.group("value")),
        )
        places = _decimal_places(guidelines, default=14)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning=(
                "Deterministically computed the monthly delta after changing the fee rule's "
                "relative rate over matching transactions."
            ),
            route="relative_fee_delta",
        )

    annual_relative_fee_match = re.search(
        r"In the year (?P<year>\d{4}) what delta would (?P<merchant>.+?) pay "
        r"if the relative fee of the fee with ID=(?P<fee_id>\d+) changed to "
        r"(?P<value>\d+(?:\.\d+)?)",
        question,
        flags=re.IGNORECASE,
    )
    if annual_relative_fee_match:
        data = _cached_load_dabstep_data(data_dir)
        value = fee_rate_delta_for_period(
            data,
            merchant=annual_relative_fee_match.group("merchant"),
            year=int(annual_relative_fee_match.group("year")),
            fee_id=int(annual_relative_fee_match.group("fee_id")),
            new_rate=float(annual_relative_fee_match.group("value")),
        )
        places = _decimal_places(guidelines, default=14)
        return DeterministicAnswer(
            agent_answer=format_decimal_places(value, places),
            reasoning=(
                "Deterministically computed the yearly delta after changing the fee rule's "
                "relative rate over matching transactions."
            ),
            route="relative_fee_rate_delta",
        )

    fee_ids_match = re.search(
        r"applicable Fee IDs for (?P<merchant>.+?) in (?P<month>[A-Za-z]+) (?P<year>\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if fee_ids_match:
        data = _cached_load_dabstep_data(data_dir)
        ids = applicable_fee_ids_for_merchant_period(
            data,
            merchant=fee_ids_match.group("merchant"),
            year=int(fee_ids_match.group("year")),
            month=MONTHS[fee_ids_match.group("month").lower()],
        )
        return DeterministicAnswer(
            agent_answer=", ".join(str(item) for item in ids),
            reasoning="Deterministically computed applicable fee IDs with wildcard-aware fee matching.",
            route="applicable_fee_ids",
        )

    aci_month_match = aci_year_match = None
    if _looks_like_aci_optimization(question):
        aci_month_match = re.search(
            r"For (?P<merchant>.+?) in (?P<month>[A-Za-z]+), if we were to move the fraudulent transactions",
            question,
            flags=re.IGNORECASE,
        )
        aci_year_match = re.search(
            r"year (?P<year>\d{4}) and at the merchant (?P<merchant>.+?), if we were to move the fraudulent transactions",
            question,
            flags=re.IGNORECASE,
        )
    if aci_month_match or aci_year_match:
        data = _cached_load_dabstep_data(data_dir)
        if aci_month_match:
            merchant = aci_month_match.group("merchant")
            year = _year_from_question(question, default=2023)
            month = MONTHS[aci_month_match.group("month").lower()]
        else:
            assert aci_year_match is not None
            merchant = aci_year_match.group("merchant")
            year = int(aci_year_match.group("year"))
            month = None
        result = optimize_aci_for_fraudulent_transactions(data, merchant=merchant, year=year, month=month)
        return DeterministicAnswer(
            agent_answer=str(result["aci"]),
            reasoning="Deterministically selected the lowest-fee ACI for fraudulent transactions.",
            route="aci_fraud_optimization",
        )

    return None


def _cached_load_dabstep_data(data_dir: Path):
    return _cached_load_dabstep_data_by_path(str(Path(data_dir).expanduser().resolve()))


@lru_cache(maxsize=4)
def _cached_load_dabstep_data_by_path(data_dir: str):
    return load_dabstep_data(Path(data_dir))


def _looks_like_account_type_domain(question: str) -> bool:
    text = question.lower()
    return "possible values" in text and "account_type" in text


def _looks_like_unique_merchant_count(question: str) -> bool:
    text = question.lower()
    return "how many" in text and "unique merchants" in text


def _looks_like_aci_optimization(question: str) -> bool:
    text = question.lower()
    return (
        "fraudulent transactions" in text
        and ("authorization characteristics indicator" in text or "aci" in text)
        and "lowest possible fees" in text
    )


def _decimal_places(guidelines: str, *, default: int) -> int:
    match = re.search(r"rounded to (?P<places>\d+) decimals?", guidelines, flags=re.IGNORECASE)
    return int(match.group("places")) if match else default


def _year_from_question(question: str, *, default: int) -> int:
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else default
