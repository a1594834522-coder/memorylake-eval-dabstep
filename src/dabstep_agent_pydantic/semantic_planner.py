from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator
from pydantic_ai import Agent

from dabstep_agent_pydantic.agent import build_semantic_planner_model_from_env
from dabstep_agent_pydantic.agent import semantic_planner_model_settings_from_env
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy
from dabstep_agent_pydantic.usage_telemetry import CallUsage
from dabstep_agent_pydantic.usage_telemetry import UsageBudgetExceeded
from dabstep_agent_pydantic.usage_telemetry import UsageLedger
from dabstep_agent_pydantic.usage_telemetry import call_usage_from_result


PLANNER_INSTRUCTIONS = """\
Produce a semantic AnalysisSpec for one payments analytics question.
You may choose semantics and identify uncertainty, but you must not compute or
return the final answer. Use only the closed AnalysisSpec schema. Never emit
Python, SQL, formulas as executable text, tool calls, or task-answer literals.
Return unsupported_reason when the operation cannot be represented safely.
"""


class SemanticUncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str = Field(min_length=1)
    proposed_choice: str = Field(min_length=1)
    rival_policy_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisPlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_spec: AnalysisSpec | None = None
    uncertainties: list[SemanticUncertainty] = Field(default_factory=list)
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def _has_plan_or_unsupported_reason(self) -> "AnalysisPlanProposal":
        has_spec = self.analysis_spec is not None
        has_reason = bool(self.unsupported_reason and self.unsupported_reason.strip())
        if has_spec == has_reason:
            raise ValueError("provide exactly one of analysis_spec or unsupported_reason")
        if has_reason and self.uncertainties:
            raise ValueError("unsupported proposals cannot contain uncertainties")
        return self


class SemanticPlannerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: AnalysisPlanProposal | None
    attempts: int = Field(ge=0, le=2)
    fallback_reason: str | None = None
    usage_trace: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Compact-intent path metadata (absent for the full-spec compatibility planner).
    intent: dict[str, Any] | None = None
    intent_family: str | None = None
    schema_chars: int | None = None
    prompt_chars: int | None = None
    compiled_plan: dict[str, Any] | None = None


def create_semantic_planner_agent(model=None):
    return Agent(
        model or build_semantic_planner_model_from_env(),
        output_type=AnalysisPlanProposal,
        instructions=PLANNER_INSTRUCTIONS,
        model_settings=semantic_planner_model_settings_from_env(),
        retries=0,
        defer_model_check=True,
    )


async def plan_semantics(
    *,
    question: str,
    guidelines: str,
    schema_summary: str,
    policies: list[SemanticPolicy],
    manual_excerpts: list[str],
    agent=None,
    usage_ledger: UsageLedger | None = None,
) -> SemanticPlannerResult:
    planner = agent or create_semantic_planner_agent()
    ledger = usage_ledger or UsageLedger(max_calls=2)
    base_prompt = build_semantic_planner_prompt(
        question=question,
        guidelines=guidelines,
        schema_summary=schema_summary,
        policies=policies,
        manual_excerpts=manual_excerpts,
    )
    prompt = base_prompt
    last_error = "unknown planner validation failure"
    attempts = 0

    for attempt in range(2):
        stage = "planner" if attempt == 0 else "planner_repair"
        try:
            ledger.ensure_can_call(stage)
        except UsageBudgetExceeded as exc:
            return SemanticPlannerResult(
                proposal=None,
                attempts=attempts,
                fallback_reason=str(exc),
                usage_trace=ledger.summary(),
            )

        attempts += 1
        started = time.perf_counter()
        try:
            result = await planner.run(prompt)
        except Exception as exc:  # noqa: BLE001 - planner failure must fall back to legacy.
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
                proposal = AnalysisPlanProposal.model_validate(result.output)
            except ValidationError as exc:
                last_error = _validation_summary(exc)
            else:
                return SemanticPlannerResult(
                    proposal=proposal,
                    attempts=attempts,
                    usage_trace=ledger.summary(),
                )

        if attempt == 0:
            prompt = (
                f"{base_prompt}\n\nREPAIR: The previous structured output was invalid: "
                f"{last_error}. Return one corrected AnalysisPlanProposal."
            )

    return SemanticPlannerResult(
        proposal=None,
        attempts=attempts,
        fallback_reason=f"invalid semantic planner output after repair: {last_error}",
        usage_trace=ledger.summary(),
    )


def build_semantic_planner_prompt(
    *,
    question: str,
    guidelines: str,
    schema_summary: str,
    policies: list[SemanticPolicy],
    manual_excerpts: list[str],
) -> str:
    policy_text = "\n".join(
        f"- {policy.policy_id}: axis={policy.axis}; choice={policy.choice}; {policy.convention}"
        for policy in policies[:5]
    ) or "- none"
    excerpts = "\n---\n".join(excerpt[:1200] for excerpt in manual_excerpts[:4]) or "(none)"
    return f"""\
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

Do not compute or return the final answer. Return only AnalysisPlanProposal.
"""


def _validation_summary(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False)
    if not errors:
        return "schema validation failed"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "root"
    return f"{location}: {first.get('msg', 'invalid value')}"
