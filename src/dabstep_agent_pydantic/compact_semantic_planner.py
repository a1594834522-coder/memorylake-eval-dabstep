"""Family-scoped compact semantic planner.

Routes to one small intent schema per call, then lets the model choose
semantic parameters only. No tools, no full AnalysisSpec generation.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic_ai import Agent

from dabstep_agent_pydantic.agent import build_semantic_planner_model_from_env
from dabstep_agent_pydantic.agent import semantic_planner_model_settings_from_env
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.semantic_intent import CustomerFraudIntent
from dabstep_agent_pydantic.semantic_intent import FeeIntent
from dabstep_agent_pydantic.semantic_intent import GeneralIntent
from dabstep_agent_pydantic.semantic_intent import SemanticIntent
from dabstep_agent_pydantic.semantic_intent import UnsupportedIntent
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy
from dabstep_agent_pydantic.usage_telemetry import CallUsage
from dabstep_agent_pydantic.usage_telemetry import UsageBudgetExceeded
from dabstep_agent_pydantic.usage_telemetry import UsageLedger
from dabstep_agent_pydantic.usage_telemetry import call_usage_from_result

IntentFamily = Literal["general", "customer_fraud", "fee", "unsupported"]

COMPACT_INTENT_MODELS: dict[str, type[BaseModel]] = {
    "general": GeneralIntent,
    "customer_fraud": CustomerFraudIntent,
    "fee": FeeIntent,
    "unsupported": UnsupportedIntent,
}

_TASK_FAMILY_TO_INTENT: dict[str, IntentFamily] = {
    "customer_fraud_metrics": "customer_fraud",
    "fee_matching": "fee",
    "fee_simulation": "fee",
    "fee_analysis": "fee",
    "schema_semantics": "general",
    "general_data_analysis": "general",
}

COMPACT_PLANNER_INSTRUCTIONS = """\
Produce one compact semantic intent for the payments analytics question.
Choose operation, semantic axes, filters, and scalar parameters only.
Do not compute or return the final answer. Do not emit Python, SQL, formulas,
tool calls, task IDs, or answer literals. Use only the closed output schema.
"""

_FAMILY_GUIDANCE = {
    "general": (
        "For ratio operations, put population-wide filters in filters and only numerator-specific "
        "conditions in numerator_filters. Count ratios do not need numerator_column or "
        "denominator_column. For ranges such as June through October, use time_scope.kind="
        "'month_range' with start_month and end_month. IP address means column ip_address; "
        "IP country means ip_country."
    ),
    "customer_fraud": (
        "The public fraud-rate convention is fraudulent EUR volume divided by total EUR volume "
        "unless the question explicitly requests transaction count. Map in-person or in-store to "
        "shopper_interaction='POS' and ecommerce to shopper_interaction='Ecommerce'; do not use "
        "device_type for those populations. Use repeat_customer_percentage only when the repeat "
        "population meaning is explicit. For missing-data questions, set identity_field to the "
        "named field, such as ip_address or email_address. Unsupported compilation safely falls back."
    ),
    "fee": (
        "Required params: period_total_fees/applicable_fee_ids need merchant and year; fee_at_amount "
        "needs amount; aci_extreme/card_scheme_extreme need amount; affected_merchants needs fee_id "
        "and year; fee_rate_delta needs fee_id, new_value, year; merchant_mcc_delta needs merchant, "
        "new_mcc, year. Put objective in the top-level objective field."
    ),
    "unsupported": "Return a concise reason only.",
}


class CompactPlannerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: SemanticIntent | None = None
    family: str = Field(min_length=1)
    attempts: int = Field(ge=0, le=2)
    fallback_reason: str | None = None
    schema_chars: int = Field(ge=0)
    prompt_chars: int = Field(ge=0)
    usage_trace: dict[str, dict[str, Any]] = Field(default_factory=dict)


def route_intent_family(question: str, guidelines: str | None = None) -> IntentFamily:
    """Deterministic family routing from the existing zero-LLM planner."""
    lowered = question.lower()
    if re.search(r"\baci\b", lowered) and any(
        term in lowered for term in ("transaction", "fee", "expensive", "cheapest")
    ):
        return "fee"
    decision = plan_task(question=question, guidelines=guidelines, route_cards=[])
    return _TASK_FAMILY_TO_INTENT.get(decision.task_family, "general")


def create_compact_planner_agent(*, family: str, model=None):
    if family not in COMPACT_INTENT_MODELS:
        raise ValueError(f"unknown intent family: {family}")
    return Agent(
        model or build_semantic_planner_model_from_env(),
        output_type=COMPACT_INTENT_MODELS[family],
        instructions=COMPACT_PLANNER_INSTRUCTIONS,
        model_settings=semantic_planner_model_settings_from_env(),
        retries=0,
        defer_model_check=True,
    )


async def plan_compact_intent(
    *,
    question: str,
    guidelines: str,
    schema_summary: str,
    policies: list[SemanticPolicy],
    manual_excerpts: list[str],
    family: str | None = None,
    agent=None,
    usage_ledger: UsageLedger | None = None,
) -> CompactPlannerResult:
    resolved_family = family or route_intent_family(question, guidelines)
    if resolved_family not in COMPACT_INTENT_MODELS:
        raise ValueError(f"unknown intent family: {resolved_family}")
    if agent is None and (fast_intent := _deterministic_fast_intent(question, resolved_family)):
        return CompactPlannerResult(
            intent=fast_intent,
            family=resolved_family,
            attempts=0,
            schema_chars=0,
            prompt_chars=0,
            usage_trace={},
        )
    model_cls = COMPACT_INTENT_MODELS[resolved_family]
    schema_chars = len(json.dumps(model_cls.model_json_schema(), separators=(",", ":")))
    planner = agent or create_compact_planner_agent(family=resolved_family)
    ledger = usage_ledger or UsageLedger(max_calls=2)
    base_prompt = build_compact_planner_prompt(
        question=question,
        guidelines=guidelines,
        schema_summary=schema_summary,
        policies=policies,
        manual_excerpts=manual_excerpts,
        family=resolved_family,
    )
    prompt = base_prompt
    prompt_chars = len(base_prompt)
    last_error = "unknown compact planner validation failure"
    attempts = 0

    for attempt in range(2):
        stage = "planner" if attempt == 0 else "planner_repair"
        try:
            ledger.ensure_can_call(stage)
        except UsageBudgetExceeded as exc:
            return CompactPlannerResult(
                intent=None,
                family=resolved_family,
                attempts=attempts,
                fallback_reason=str(exc),
                schema_chars=schema_chars,
                prompt_chars=prompt_chars,
                usage_trace=ledger.summary(),
            )

        attempts += 1
        started = time.perf_counter()
        try:
            result = await planner.run(prompt)
        except Exception as exc:  # noqa: BLE001 - planner failure must fall back.
            ledger.record(CallUsage(
                stage=stage,
                input_tokens=0,
                output_tokens=0,
                latency_ms=round((time.perf_counter() - started) * 1000),
                retries=attempt,
            ))
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            ledger.record(call_usage_from_result(
                result,
                stage=stage,
                latency_ms=(time.perf_counter() - started) * 1000,
                retries=attempt,
            ))
            try:
                intent = model_cls.model_validate(
                    result.output.model_dump(mode="json")
                    if isinstance(result.output, BaseModel)
                    else result.output
                )
            except ValidationError as exc:
                last_error = _validation_summary(exc)
            else:
                return CompactPlannerResult(
                    intent=intent,  # type: ignore[arg-type]
                    family=resolved_family,
                    attempts=attempts,
                    schema_chars=schema_chars,
                    prompt_chars=prompt_chars,
                    usage_trace=ledger.summary(),
                )

        if attempt == 0:
            prompt = (
                f"{base_prompt}\n\nREPAIR: The previous structured output was invalid: "
                f"{last_error}. Return one corrected {model_cls.__name__}."
            )
            prompt_chars = max(prompt_chars, len(prompt))

    return CompactPlannerResult(
        intent=None,
        family=resolved_family,
        attempts=attempts,
        fallback_reason=f"invalid compact planner output after repair: {last_error}",
        schema_chars=schema_chars,
        prompt_chars=prompt_chars,
        usage_trace=ledger.summary(),
    )


def build_compact_planner_prompt(
    *,
    question: str,
    guidelines: str,
    schema_summary: str,
    policies: list[SemanticPolicy],
    manual_excerpts: list[str],
    family: str,
) -> str:
    policy_text = "\n".join(
        f"- {policy.policy_id}: axis={policy.axis}; choice={policy.choice}; {policy.convention}"
        for policy in policies[:5]
    ) or "- none"
    excerpts = "\n---\n".join(excerpt[:1200] for excerpt in manual_excerpts[:4]) or "(none)"
    model_name = COMPACT_INTENT_MODELS[family].__name__
    return f"""\
INTENT FAMILY: {family}
OUTPUT MODEL: {model_name}

FAMILY CONTRACT:
{_FAMILY_GUIDANCE[family]}

QUESTION:
{question}

OUTPUT GUIDELINES:
{guidelines or "N/A"}

SCHEMA SUMMARY:
{schema_summary}

RELEVANT CERTIFIED POLICIES:
{policy_text}

MINIMAL MANUAL EXCERPTS:
{excerpts}

Do not compute or return the final answer. Return only {model_name}.
"""


def _validation_summary(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False)
    if not errors:
        return "schema validation failed"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "root"
    return f"{location}: {first.get('msg', 'invalid value')}"


def _deterministic_fast_intent(question: str, family: str) -> SemanticIntent | None:
    normalized = re.sub(r"\s+", " ", question.strip().lower()).strip(" .?!")
    if family != "general":
        return None
    count_cue = normalized.startswith("how many ") or any(
        cue in normalized for cue in ("total count", "total number", "overall count")
    )
    has_payment_entity = bool(re.search(r"\b(?:transactions|payments)\b", normalized))
    qualifiers = (
        " by ", " per ", " for ", " during ", " between ", " where ", " with ",
        "merchant", "country", "scheme", "fraud", "email", "device", "group",
    )
    if count_cue and has_payment_entity and not any(term in normalized for term in qualifiers):
        return GeneralIntent(
            operation="aggregate",
            aggregation="count",
            source="payments",
            output_kind="integer",
        )
    return None
