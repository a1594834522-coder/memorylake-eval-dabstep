from __future__ import annotations

import os
from collections.abc import Sequence

from pydantic_ai import FunctionToolset
from pydantic_ai import RunContext

from dabstep_agent_pydantic.agent import ChoiceToolResult
from dabstep_agent_pydantic.agent import DABStepDeps
from dabstep_agent_pydantic.agent import NumericToolResult
from dabstep_agent_pydantic.dabstep_core import format_decimal_places
from dabstep_agent_pydantic.dabstep_core import fee_rate_delta_for_period
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.dabstep_core import merchant_mcc_fee_delta_for_year
from dabstep_agent_pydantic.dabstep_core import optimize_aci_for_fraudulent_transactions
from dabstep_agent_pydantic.dabstep_core import total_fees_for_merchant_period
from dabstep_agent_pydantic.planning import PlanDecision
from dabstep_agent_pydantic.python_tool import ToolResult
from dabstep_agent_pydantic.skill_tools import apply_learned_skill
from dabstep_agent_pydantic.skill_tools import learned_skill_tools_enabled
from dabstep_agent_pydantic.skill_tools import list_learned_skills
from dabstep_agent_pydantic.skill_tools import propose_skill_candidate


def execute_python_code(ctx: RunContext[DABStepDeps], code: str) -> ToolResult:
    """Execute Python code in a persistent workspace and return stdout or a structured error."""
    return ctx.deps.workspace.execute(code)


def compute_total_fees(
    ctx: RunContext[DABStepDeps],
    merchant: str,
    year: int,
    month: int | None = None,
    day_of_year: int | None = None,
    decimal_places: int = 2,
) -> NumericToolResult:
    """Compute total fees paid by a merchant with deterministic DABStep fee matching."""
    data = load_dabstep_data(ctx.deps.data_dir)
    value = total_fees_for_merchant_period(
        data,
        merchant=merchant,
        year=year,
        month=month,
        day_of_year=day_of_year,
    )
    return NumericToolResult(
        value=value,
        formatted=format_decimal_places(value, decimal_places),
        method="total_fees_for_merchant_period with public fee matching semantics",
    )


def compute_mcc_fee_delta(
    ctx: RunContext[DABStepDeps],
    merchant: str,
    year: int,
    new_mcc: int,
    decimal_places: int = 6,
) -> NumericToolResult:
    """Compute yearly fee delta if a merchant changed MCC before the year started."""
    data = load_dabstep_data(ctx.deps.data_dir)
    value = merchant_mcc_fee_delta_for_year(
        data,
        merchant=merchant,
        year=year,
        new_mcc=new_mcc,
    )
    return NumericToolResult(
        value=value,
        formatted=format_decimal_places(value, decimal_places),
        method="merchant_mcc_fee_delta_for_year with public fee matching semantics",
    )


def compute_relative_fee_delta(
    ctx: RunContext[DABStepDeps],
    merchant: str,
    year: int,
    month: int | None,
    fee_id: int,
    relative_fee_state: int = 1,
    decimal_places: int = 14,
) -> NumericToolResult:
    """Compute delta for relative fee simulations over a month or year."""
    data = load_dabstep_data(ctx.deps.data_dir)
    value = fee_rate_delta_for_period(
        data,
        merchant=merchant,
        year=year,
        month=month,
        fee_id=fee_id,
        new_rate=relative_fee_state,
    )
    method = "fee_rate_delta_for_period for relative-rate fee simulation"
    return NumericToolResult(
        value=value,
        formatted=format_decimal_places(value, decimal_places),
        method=method,
    )


def compute_best_aci_for_fraudulent_transactions(
    ctx: RunContext[DABStepDeps],
    merchant: str,
    year: int,
    month: int | None = None,
) -> ChoiceToolResult:
    """Find the lowest-fee ACI for fraudulent transactions in a month or full year."""
    data = load_dabstep_data(ctx.deps.data_dir)
    result = optimize_aci_for_fraudulent_transactions(
        data,
        merchant=merchant,
        year=year,
        month=month,
    )
    return ChoiceToolResult(
        choice=str(result["aci"]),
        cost=float(result["cost"]),
        formatted_cost=str(result["formatted"]),
        method="optimize_aci_for_fraudulent_transactions using public fee context",
    )


COMMON_TOOLSET = FunctionToolset(
    tools=[execute_python_code],
    id="common",
    instructions="Use Python for dataset inspection, joins, aggregation, and independent verification.",
)
FEE_MATCHING_TOOLSET = FunctionToolset(
    tools=[compute_total_fees],
    id="fee_matching",
    instructions="Use deterministic helpers for fee applicability and total-fee calculations.",
)
FEE_SIMULATION_TOOLSET = FunctionToolset(
    tools=[
        compute_total_fees,
        compute_mcc_fee_delta,
        compute_relative_fee_delta,
        compute_best_aci_for_fraudulent_transactions,
    ],
    id="fee_simulation",
    instructions="Use deterministic helpers for fee counterfactuals, steering, MCC changes, and ACI optimization.",
)
FRAUD_CUSTOMER_TOOLSET = FunctionToolset(
    tools=[],
    id="fraud_and_customer_semantics",
    instructions="Use the Python workspace and public helper module for fraud, shopper, email, and repeat-customer metrics.",
)
SCHEMA_DOMAIN_TOOLSET = FunctionToolset(
    tools=[],
    id="schema_domain_semantics",
    instructions="Use the Python workspace and public manual/schema files for field domains, MCC lookup, and date bounds.",
)
OUTPUT_CONTRACT_TOOLSET = FunctionToolset(
    tools=[],
    id="output_contracts",
    instructions="Format final answers exactly according to the task guideline.",
)

LEARNED_SKILLS_TOOLSET = FunctionToolset(
    tools=[list_learned_skills, apply_learned_skill, propose_skill_candidate],
    id="learned_skills",
    instructions=(
        "Learned deterministic skills cover recurring question templates with "
        "audited interpretations. When the task is semantically an instance of "
        "a learned template, apply the skill with structured parameters instead "
        "of re-deriving the computation. When you solve a recurring question "
        "shape manually because no skill covers it, propose it as a skill "
        "candidate afterwards."
    ),
)

TOOLSETS_BY_ID = {
    "common": COMMON_TOOLSET,
    "learned_skills": LEARNED_SKILLS_TOOLSET,
    "fee_matching": FEE_MATCHING_TOOLSET,
    "fee_simulation": FEE_SIMULATION_TOOLSET,
    "fraud_and_customer_semantics": FRAUD_CUSTOMER_TOOLSET,
    "schema_domain_semantics": SCHEMA_DOMAIN_TOOLSET,
    "output_contracts": OUTPUT_CONTRACT_TOOLSET,
}


def tool_selection_mode() -> str:
    """`planner` (default): the plan's route cards select the toolsets.
    `open`: the full library is exposed and the plan is advisory — the model
    decides which tools a task needs."""
    mode = os.getenv("DABSTEP_TOOL_SELECTION", "planner").strip().lower()
    return mode if mode in ("planner", "open") else "planner"


def toolset_names_for_plan(plan: PlanDecision) -> list[str]:
    names = ["common"]
    if learned_skill_tools_enabled():
        names.append("learned_skills")
    if tool_selection_mode() == "open":
        # fee_matching is a strict subset of fee_simulation (duplicate tool
        # names would collide), so the open library carries fee_simulation.
        names.extend(t for t in TOOLSETS_BY_ID
                     if t not in ("common", "learned_skills", "fee_matching"))
        return _dedupe(names)
    for toolset_id in plan.toolset_ids:
        if toolset_id not in TOOLSETS_BY_ID:
            continue
        if toolset_id == "fee_matching" and "fee_simulation" in plan.toolset_ids:
            continue
        names.append(toolset_id)
    return _dedupe(names)


def toolsets_for_plan(plan: PlanDecision) -> Sequence[FunctionToolset[DABStepDeps]]:
    return [TOOLSETS_BY_ID[name] for name in toolset_names_for_plan(plan)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
