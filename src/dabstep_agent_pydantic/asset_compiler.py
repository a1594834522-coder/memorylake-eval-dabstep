from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class AnalysisLayer(str, Enum):
    SCHEMA_SEMANTICS = "schema_semantics"
    FEE_MATCHING = "fee_matching"
    FEE_SIMULATION = "fee_simulation"
    CUSTOMER_FRAUD_METRICS = "customer_fraud_metrics"
    OUTPUT_CONTRACT = "output_contract"


class RouteCard(BaseModel):
    route_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    analysis_layer: AnalysisLayer = AnalysisLayer.SCHEMA_SEMANTICS
    keywords: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    helper_functions: list[str] = Field(default_factory=list)
    verification_checks: list[str] = Field(default_factory=list)


ROUTE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "fee_matching": {
        "title": "Fee matching and applicability",
        "analysis_layer": AnalysisLayer.FEE_MATCHING,
        "keywords": ("fee", "fees.json", "matching_fee", "applicable fee", "wildcard"),
        "triggers": ["fee id", "applicable fee", "fees.json", "wildcard matching"],
        "helper_functions": [
            "matching_fee_ids",
            "match_count_summary",
            "matches_fee_rule",
            "applicable_fee_ids_for_merchant_period",
            "fee_affected_merchants_for_year",
            "total_fees_for_merchant_period",
            "calculate_fee_monthly_metrics",
            "calculate_monthly_metrics",
        ],
        "verification_checks": [
            "Treat null and empty-list fee fields as wildcards.",
            "Default fee matching uses EUR-volume fraud buckets; compare count-based buckets only as an explicit diagnostic if a task says count-based.",
            "Check merchant-level filters before payment-level filters.",
        ],
    },
    "fee_simulation": {
        "title": "Fee simulation and counterfactual changes",
        "analysis_layer": AnalysisLayer.FEE_SIMULATION,
        "keywords": ("simulate", "simulation", "changed", "delta", "steer", "minimum fees", "hypothetical"),
        "triggers": ["changed fee rule", "MCC change", "traffic steering", "fee delta"],
        "helper_functions": [
            "calculate_fee",
            "match_count_summary",
            "total_fees_for_merchant_period",
            "merchant_mcc_fee_delta_for_year",
            "fee_rate_delta_for_month",
            "fee_rate_delta_for_period",
            "fee_fixed_component_delta_for_month",
            "optimize_aci_for_fraudulent_transactions",
            "calculate_fee_monthly_metrics",
            "add_intracountry_flag",
        ],
        "verification_checks": [
            "Sum every matching fee rule per transaction unless the task explicitly asks for IDs.",
            "For total-fee and MCC-change deltas, prefer EUR-volume fraud buckets unless the task explicitly states count-based buckets.",
            "Compare original and counterfactual sets or totals with the same matching semantics.",
        ],
    },
    "fraud_and_customer_semantics": {
        "title": "Fraud, shopper, and customer metrics",
        "analysis_layer": AnalysisLayer.CUSTOMER_FRAUD_METRICS,
        "keywords": ("fraud", "customer", "email", "shopper", "repeat", "unique"),
        "triggers": ["fraud rate", "unique email", "repeat customer", "shopper metric"],
        "helper_functions": [
            "fraud_rate_by_volume",
            "fraud_rate_by_group",
            "average_transactions_per_unique_email",
            "average_transaction_amount_per_unique_email",
            "repeat_customer_percentage",
        ],
        "verification_checks": [
            "Use fraudulent EUR volume divided by total EUR volume for DABStep fraud-rate semantics.",
            "Ignore missing email addresses for unique-email metrics unless the question asks about missing values.",
        ],
    },
    "schema_domain_semantics": {
        "title": "Schema domain and manual-defined values",
        "analysis_layer": AnalysisLayer.SCHEMA_SEMANTICS,
        "keywords": ("manual", "schema", "possible values", "domain", "field_domain_values", "category"),
        "triggers": ["possible values", "field domain", "manual-defined category"],
        "helper_functions": [
            "field_domain_values",
            "mcc_code_for_description",
            "most_expensive_mccs_for_amount",
            "fee_factor_monotonicity",
        ],
        "verification_checks": [
            "Combine manual-defined values with observed values.",
            "Do not assume unobserved manual categories are impossible.",
        ],
    },
    "output_contracts": {
        "title": "DABStep answer formatting contracts",
        "analysis_layer": AnalysisLayer.OUTPUT_CONTRACT,
        "keywords": ("output", "answer", "format", "comma", "rounded", "decimal"),
        "triggers": ["comma list", "rounded number", "empty list", "Not Applicable"],
        "helper_functions": ["format_decimal_places"],
        "verification_checks": [
            "Return only the requested answer shape in agent_answer.",
            "For empty-list answers, use an empty string when the guideline asks for an empty string.",
        ],
    },
}


def load_route_cards(path: Path) -> list[RouteCard]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RouteCard.model_validate(item) for item in payload.get("cards", [])]


def format_route_cards(cards: list[RouteCard], max_chars: int = 8000) -> str:
    if not cards:
        return ""
    lines = ["Compiled official-safe route cards:"]
    for card in cards:
        lines.append(f"- {card.route_id}: {card.title} [{card.analysis_layer.value}]")
        if card.keywords:
            lines.append(f"  Keywords: {', '.join(card.keywords)}")
        lines.append(f"  Triggers: {', '.join(card.triggers)}")
        if card.helper_functions:
            lines.append(f"  Helpers: {', '.join(card.helper_functions)}")
        for instruction in card.instructions[:4]:
            lines.append(f"  - {instruction}")
        for check in card.verification_checks:
            lines.append(f"  Verify: {check}")
    return "\n".join(lines)[:max_chars]


def select_route_cards(
    cards: list[RouteCard],
    *,
    question: str,
    guidelines: str | None = None,
    max_cards: int = 4,
) -> list[RouteCard]:
    if not cards:
        return []
    text = f"{question}\n{guidelines or ''}".lower()
    by_id = {card.route_id: card for card in cards}
    scores: dict[str, int] = {}
    for index, card in enumerate(cards):
        score = 0
        for trigger in card.triggers:
            if trigger.lower() in text:
                score += 4
        for helper in card.helper_functions:
            if helper.lower() in text:
                score += 2
        score += _route_keyword_score(card.route_id, text)
        if score:
            # Preserve input order as deterministic tie breaker.
            scores[card.route_id] = score * 1000 - index

    if _looks_like_fee_counterfactual(text):
        for route_id in ("fee_simulation", "fee_matching"):
            if route_id in by_id:
                scores[route_id] = max(scores.get(route_id, 0), 9000)
    if _looks_like_fee_matching(text) and "fee_matching" in by_id:
        scores["fee_matching"] = max(scores.get("fee_matching", 0), 8000)
    if _looks_like_customer_or_fraud_metric(text) and "fraud_and_customer_semantics" in by_id:
        scores["fraud_and_customer_semantics"] = max(scores.get("fraud_and_customer_semantics", 0), 8000)
    if _looks_like_schema_domain_question(text) and "schema_domain_semantics" in by_id:
        scores["schema_domain_semantics"] = max(scores.get("schema_domain_semantics", 0), 8000)

    selected_ids = [
        route_id
        for route_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if route_id != "output_contracts"
    ][: max(0, max_cards - 1)]
    if "output_contracts" in by_id:
        selected_ids.append("output_contracts")
    if not selected_ids:
        selected_ids = [cards[0].route_id]
    return [by_id[route_id] for route_id in selected_ids if route_id in by_id]


def _matches_route(text: str, definition: dict[str, Any]) -> bool:
    return any(keyword in text for keyword in definition["keywords"])


def _route_keyword_score(route_id: str, text: str) -> int:
    route_keywords = {
        "fee_matching": ("fee id", "applicable fee", "fees.json", "matching fee"),
        "fee_simulation": ("imagine", "changed", "delta", "steer", "minimum fees", "counterfactual"),
        "fraud_and_customer_semantics": ("fraud", "customer", "email", "shopper", "repeat"),
        "schema_domain_semantics": ("possible", "domain", "schema", "manual", "values"),
        "output_contracts": ("answer must", "rounded", "comma", "empty string", "not applicable"),
    }
    return sum(3 for keyword in route_keywords.get(route_id, ()) if keyword in text)


def _looks_like_fee_counterfactual(text: str) -> bool:
    fee_or_amount = any(keyword in text for keyword in ("fee", "amount", "pay", "cost"))
    counterfactual = any(keyword in text for keyword in ("imagine", "changed", "delta", "steer", "minimum", "mcc"))
    return fee_or_amount and counterfactual


def _looks_like_fee_matching(text: str) -> bool:
    return "fee" in text or "fees.json" in text


def _looks_like_customer_or_fraud_metric(text: str) -> bool:
    return any(keyword in text for keyword in ("fraud", "customer", "email", "shopper", "repeat"))


def _looks_like_schema_domain_question(text: str) -> bool:
    return any(keyword in text for keyword in ("possible value", "domain", "schema", "manual-defined", "which values"))
