from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from dabstep_agent_pydantic.asset_compiler import AnalysisLayer
from dabstep_agent_pydantic.asset_compiler import RouteCard


class AnalysisStep(BaseModel):
    layer: str = Field(min_length=1)
    action: str = Field(min_length=1)
    verification: str = Field(min_length=1)


class AnalysisPlan(BaseModel):
    task_family: str = Field(min_length=1)
    selected_route_ids: list[str] = Field(default_factory=list)
    recommended_helpers: list[str] = Field(default_factory=list)
    steps: list[AnalysisStep] = Field(default_factory=list)
    output_contract: str = "Return the exact requested answer shape in agent_answer."


def build_analysis_plan(
    *,
    question: str | None,
    guidelines: str | None,
    route_cards: list[RouteCard],
) -> AnalysisPlan:
    task_family = _infer_task_family(question=question, route_cards=route_cards)
    helpers = _dedupe(
        helper
        for card in route_cards
        for helper in card.helper_functions
    )
    route_ids = [card.route_id for card in route_cards]
    output_contract = _output_contract(guidelines)
    steps = [
        AnalysisStep(
            layer="plan",
            action="Identify the target entity, time window, metric, and required answer format before writing code.",
            verification="Restate these filters in code variables and compare them with the question text.",
        ),
        AnalysisStep(
            layer="data",
            action="Load DABStep tables with load_dabstep_data(data_dir) and inspect the relevant columns.",
            verification="Check row counts after each merchant, date, card, fraud, or fee-rule filter.",
        ),
        AnalysisStep(
            layer="execute",
            action=_execution_action(task_family, helpers),
            verification="Prefer deterministic helpers for known DABStep semantics; use pandas for inspection and cross-checks.",
        ),
        AnalysisStep(
            layer="verify",
            action="Run an independent sanity check before finalizing the answer.",
            verification=_verification_rule(route_cards),
        ),
        AnalysisStep(
            layer="format",
            action="Format agent_answer exactly as requested.",
            verification=output_contract,
        ),
    ]
    return AnalysisPlan(
        task_family=task_family,
        selected_route_ids=route_ids,
        recommended_helpers=helpers,
        steps=steps,
        output_contract=output_contract,
    )


def format_analysis_plan(plan: AnalysisPlan | None) -> str:
    if plan is None:
        return "(no typed analysis plan)"
    lines = [
        "Typed analysis plan:",
        f"- task_family: {plan.task_family}",
    ]
    if plan.selected_route_ids:
        lines.append(f"- selected_routes: {', '.join(plan.selected_route_ids)}")
    if plan.recommended_helpers:
        lines.append(f"- recommended_helpers: {', '.join(plan.recommended_helpers)}")
    for step in plan.steps:
        lines.append(f"- {step.layer}: {step.action}")
        lines.append(f"  Verify: {step.verification}")
    lines.append(f"- output_contract: {plan.output_contract}")
    return "\n".join(lines)


def _infer_task_family(*, question: str | None, route_cards: list[RouteCard]) -> str:
    text = (question or "").lower()
    if ("mcc" in text or "changed" in text or "counterfactual" in text or "simulate" in text) and (
        "fee" in text or "delta" in text or "amount" in text
    ):
        return "fee_simulation"
    route_ids = {card.route_id for card in route_cards}
    if "fee_simulation" in route_ids:
        return "fee_simulation"
    if "fee_matching" in route_ids:
        return "fee_matching"
    if "fraud_and_customer_semantics" in route_ids:
        return "customer_fraud_metrics"
    if "schema_domain_semantics" in route_ids:
        return "schema_semantics"
    if "fee" in text:
        return "fee_analysis"
    if "fraud" in text or "customer" in text or "email" in text:
        return "customer_fraud_metrics"
    return "general_data_analysis"


def _execution_action(task_family: str, helpers: list[str]) -> str:
    if helpers:
        return f"Use the selected helper path first: {', '.join(helpers[:6])}."
    if task_family == "general_data_analysis":
        return "Use pandas groupby, filtering, joins, and aggregation to compute the requested metric."
    return "Use the relevant deterministic helper or a transparent pandas equivalent."


def _verification_rule(route_cards: list[RouteCard]) -> str:
    checks = _dedupe(
        check
        for card in route_cards
        for check in card.verification_checks
    )
    if checks:
        return " ".join(checks[:3])
    return "Confirm the final value by checking intermediate rows and aggregation denominators."


def _output_contract(guidelines: str | None) -> str:
    if guidelines and guidelines.strip():
        return f"Follow guidelines exactly: {guidelines.strip()}"
    return "Return the exact requested answer shape in agent_answer."


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
