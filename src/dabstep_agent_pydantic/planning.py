from __future__ import annotations

import re

from pydantic import BaseModel
from pydantic import Field

from dabstep_agent_pydantic.analysis_plan import AnalysisPlan
from dabstep_agent_pydantic.analysis_plan import build_analysis_plan
from dabstep_agent_pydantic.asset_compiler import RouteCard
from dabstep_agent_pydantic.asset_compiler import select_route_cards


class PlanDecision(BaseModel):
    task_family: str = Field(min_length=1)
    selected_route_ids: list[str] = Field(default_factory=list)
    toolset_ids: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    ambiguity_axes: list[str] = Field(default_factory=list)
    max_solver_retries: int = Field(default=1, ge=0, le=3)
    analysis_plan: AnalysisPlan | None = None


def plan_task(
    *,
    question: str,
    guidelines: str | None,
    route_cards: list[RouteCard],
) -> PlanDecision:
    selected_cards = select_route_cards(route_cards, question=question, guidelines=guidelines)
    analysis_plan = build_analysis_plan(
        question=question,
        guidelines=guidelines,
        route_cards=selected_cards,
    )
    selected_route_ids = [card.route_id for card in selected_cards]
    return PlanDecision(
        task_family=analysis_plan.task_family,
        selected_route_ids=selected_route_ids,
        toolset_ids=_toolset_ids_for_routes(selected_route_ids),
        verification_focus=_verification_focus(question=question, guidelines=guidelines, route_ids=selected_route_ids),
        ambiguity_axes=_ambiguity_axes(question=question, guidelines=guidelines),
        max_solver_retries=1,
        analysis_plan=analysis_plan,
    )


def _toolset_ids_for_routes(route_ids: list[str]) -> list[str]:
    toolsets: list[str] = []
    for route_id in route_ids:
        if route_id == "fee_matching":
            toolsets.append("fee_matching")
        elif route_id == "fee_simulation":
            toolsets.append("fee_simulation")
        elif route_id == "fraud_and_customer_semantics":
            toolsets.append("fraud_and_customer_semantics")
        elif route_id == "schema_domain_semantics":
            toolsets.append("schema_domain_semantics")
        elif route_id == "output_contracts":
            toolsets.append("output_contracts")
    if "output_contracts" not in toolsets:
        toolsets.append("output_contracts")
    return _dedupe(toolsets)


def _verification_focus(*, question: str, guidelines: str | None, route_ids: list[str]) -> list[str]:
    text = f"{question}\n{guidelines or ''}".lower()
    focus: list[str] = ["answer_format"]
    if "fee_matching" in route_ids or "fee_simulation" in route_ids:
        focus.extend(["fee_formula", "fee_rule_matching"])
    if "fee_simulation" in route_ids:
        focus.append("counterfactual_consistency")
    if "fraud_and_customer_semantics" in route_ids:
        focus.append("metric_denominator")
    if "schema_domain_semantics" in route_ids:
        focus.append("schema_vs_observed_values")
    if "round" in text or "decimal" in text:
        focus.append("rounding")
    return _dedupe(focus)


def _ambiguity_axes(*, question: str, guidelines: str | None) -> list[str]:
    text = f"{question}\n{guidelines or ''}".lower()
    axes: list[str] = []
    if "percentage of" in text:
        axes.append("percentage_scope")
    if re.search(r"\bfraud\s*rate\b|\bfraudrate\b", text) and not re.search(
        r"\b(volume|count|transaction-count|transaction count|fraudulent volume|eur volume)\b",
        text,
    ):
        axes.append("fraud_rate_basis")
    if "average fee" in text:
        axes.append("average_fee_basis")
    if re.search(r"\b(most|least)\s+expensive\b", text) and "in general" in text:
        axes.append("fee_extreme_basis")
    return _dedupe(axes)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
