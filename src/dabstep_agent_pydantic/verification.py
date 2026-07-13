from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from dabstep_agent_pydantic.dabstep_core import fee_rate_delta_for_period
from dabstep_agent_pydantic.dabstep_core import format_decimal_places
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.dabstep_core import merchant_mcc_fee_delta_for_year
from dabstep_agent_pydantic.dabstep_core import optimize_aci_for_fraudulent_transactions
from dabstep_agent_pydantic.dabstep_core import total_fees_for_merchant_period
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.output_contract import parse_guidelines
from dabstep_agent_pydantic.output_contract import validate_output_contract

if TYPE_CHECKING:
    from dabstep_agent_pydantic.planning import PlanDecision


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

# Merchant identifiers in the public dataset are underscore-joined capitalized tokens.
MERCHANT_TOKEN = r"[A-Z][A-Za-z0-9']*(?:_[A-Za-z0-9']+)+"

FEE_VERIFICATION_FOCUS = {"fee_formula", "fee_rule_matching", "counterfactual_consistency"}


async def verify_semantic_candidates(**kwargs):
    """Public integration point for the candidate-level semantic verifier."""
    from dabstep_agent_pydantic.semantic_verifier import verify_candidate_set

    return await verify_candidate_set(**kwargs)


def verify_record(
    record: dict[str, object],
    *,
    task: Task,
    plan: "PlanDecision | None",
    data_dir: Path,
) -> str | None:
    """Return retry feedback when the answer violates a contract or an independent recomputation disagrees."""
    answer = str(record.get("agent_answer", "")).strip()
    guidelines = task.guidelines or ""

    feedback = validate_output_contract(answer, parse_guidelines(guidelines))
    if feedback:
        return feedback

    if record.get("deterministic_route"):
        return None
    focus = set(plan.verification_focus) if plan else set()
    if focus and not focus & FEE_VERIFICATION_FOCUS:
        return None
    return _recomputation_feedback(answer, task=task, data_dir=data_dir)


def _recomputation_feedback(answer: str, *, task: Task, data_dir: Path) -> str | None:
    if not (Path(data_dir) / "fees.json").exists():
        return None
    question = " ".join(task.question.split())
    guidelines = task.guidelines or ""
    try:
        for check in (
            _verify_relative_fee_delta,
            _verify_mcc_fee_delta,
            _verify_total_fees,
            _verify_aci_steering,
        ):
            feedback = check(answer, question=question, guidelines=guidelines, data_dir=data_dir)
            if feedback:
                return feedback
    except Exception:  # noqa: BLE001 - verification is best effort and must never fail the task.
        return None
    return None


def _verify_relative_fee_delta(answer: str, *, question: str, guidelines: str, data_dir: Path) -> str | None:
    match = re.search(
        r"relative fee of the fee with ID[= ]?(?P<fee_id>\d+) changed to (?P<value>\d+(?:\.\d+)?)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    merchant = _merchant_from_question(question)
    year = _year_from_question(question)
    if merchant is None or year is None:
        return None
    month = _month_from_question(question)
    new_value = float(match.group("value"))
    fee_id = int(match.group("fee_id"))

    data = load_dabstep_data(data_dir)
    expected = fee_rate_delta_for_period(
        data,
        merchant=merchant,
        year=year,
        month=month,
        fee_id=fee_id,
        new_rate=new_value,
    )
    method = "fee_rate_delta_for_period"
    return _numeric_mismatch_feedback(
        answer,
        expected=expected,
        places=_decimal_places(guidelines, default=14),
        description="the relative-fee change delta",
        method=method,
    )


def _verify_mcc_fee_delta(answer: str, *, question: str, guidelines: str, data_dir: Path) -> str | None:
    match = re.search(r"changed its MCC(?: code)? to (?P<mcc>\d+)", question, flags=re.IGNORECASE)
    if not match:
        return None
    merchant = _merchant_from_question(question)
    year = _year_from_question(question)
    if merchant is None or year is None:
        return None

    data = load_dabstep_data(data_dir)
    expected = merchant_mcc_fee_delta_for_year(
        data,
        merchant=merchant,
        year=year,
        new_mcc=int(match.group("mcc")),
    )
    return _numeric_mismatch_feedback(
        answer,
        expected=expected,
        places=_decimal_places(guidelines, default=6),
        description="the MCC-change yearly fee delta",
        method="merchant_mcc_fee_delta_for_year",
    )


def _verify_total_fees(answer: str, *, question: str, guidelines: str, data_dir: Path) -> str | None:
    if not re.search(r"total fees", question, flags=re.IGNORECASE):
        return None
    if re.search(r"delta|changed|imagine|steer", question, flags=re.IGNORECASE):
        return None
    # Conditioned totals (per scheme, account type, ACI, ...) have a narrower scope
    # than the plain period helper; skip rather than recompute the wrong population.
    if re.search(r"scheme|account type|\baci\b|fraud|shopper", question, flags=re.IGNORECASE):
        return None
    merchant = _merchant_from_question(question)
    year = _year_from_question(question)
    if merchant is None or year is None:
        return None

    data = load_dabstep_data(data_dir)
    expected = total_fees_for_merchant_period(
        data,
        merchant=merchant,
        year=year,
        month=_month_from_question(question),
        day_of_year=_day_of_year_from_question(question),
    )
    return _numeric_mismatch_feedback(
        answer,
        expected=expected,
        places=_decimal_places(guidelines, default=2),
        description="the total fees for the requested period",
        method="total_fees_for_merchant_period",
    )


def _verify_aci_steering(answer: str, *, question: str, guidelines: str, data_dir: Path) -> str | None:
    del guidelines
    text = question.lower()
    if "fraudulent transactions" not in text:
        return None
    if "aci" not in text and "authorization characteristics indicator" not in text:
        return None
    if not any(keyword in text for keyword in ("lowest", "minimize", "incentive")):
        return None
    merchant = _merchant_from_question(question)
    year = _year_from_question(question)
    if merchant is None or year is None:
        return None

    data = load_dabstep_data(data_dir)
    result = optimize_aci_for_fraudulent_transactions(
        data,
        merchant=merchant,
        year=year,
        month=_month_from_question(question),
    )
    expected_aci = str(result["aci"]).strip()
    answered_aci = answer.split(":", 1)[0].strip()
    if answered_aci.upper() == expected_aci.upper():
        return None
    return (
        f"Independent verification with optimize_aci_for_fraudulent_transactions selected ACI "
        f"{expected_aci} as the lowest-fee option; reconcile your fee-matching semantics or correct the answer."
    )


def _numeric_mismatch_feedback(
    answer: str,
    *,
    expected: float,
    places: int,
    description: str,
    method: str,
) -> str | None:
    try:
        answered_value = float(answer.replace(",", ""))
    except ValueError:
        return None
    if abs(answered_value - expected) <= 0.5 * 10**-places:
        return None
    if format_decimal_places(answered_value, places) == format_decimal_places(expected, places):
        return None
    return (
        f"Independent verification recomputed {description} as {format_decimal_places(expected, places)} "
        f"using {method}; reconcile your filters and fee-matching semantics or correct the answer."
    )


def _merchant_from_question(question: str) -> str | None:
    for pattern in (
        rf"would (?P<merchant>{MERCHANT_TOKEN}) pay",
        rf"merchant (?P<merchant>{MERCHANT_TOKEN})",
        rf"(?:that|for) (?P<merchant>{MERCHANT_TOKEN})",
        rf"(?P<merchant>{MERCHANT_TOKEN})",
    ):
        match = re.search(pattern, question)
        if match:
            return match.group("merchant")
    return None


def _year_from_question(question: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else None


def _day_of_year_from_question(question: str) -> int | None:
    match = re.search(
        r"(?:for|on) the (?P<day>\d{1,3})(?:st|nd|rd|th)? (?:day )?of (?:the year )?(?:20\d{2})",
        question,
        flags=re.IGNORECASE,
    ) or re.search(r"\bday (?P<day>\d{1,3}) of\b", question, flags=re.IGNORECASE)
    return int(match.group("day")) if match else None


def _month_from_question(question: str) -> int | None:
    for name, number in MONTHS.items():
        if re.search(rf"\b{name}\b", question, flags=re.IGNORECASE):
            return number
    return None


def _decimal_places(guidelines: str, *, default: int | None = None) -> int | None:
    match = re.search(r"rounded to (?P<places>\d+) decimals?", guidelines, flags=re.IGNORECASE)
    return int(match.group("places")) if match else default
