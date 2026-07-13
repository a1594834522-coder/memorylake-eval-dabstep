from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext
from pydantic_ai.models import get_user_agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from dabstep_agent_pydantic.analysis_plan import AnalysisPlan
from dabstep_agent_pydantic.analysis_plan import format_analysis_plan
from dabstep_agent_pydantic.asset_compiler import RouteCard
from dabstep_agent_pydantic.asset_compiler import format_route_cards
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.python_tool import PythonWorkspace


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_THINKING = "high"
DEFAULT_PLANNER_THINKING = "low"
DEFAULT_MEMORYLAKE_ROUTER_BASE_URL = "https://app.memorylake.ai/v1"
DEFAULT_MEMORYLAKE_BYOK_BASE_URL = "https://app.memorylake.ai/v1/openai"
MODEL_GATEWAY_RETRY_STATUS_CODES = frozenset({502, 503, 504})
MODEL_GATEWAY_RETRY_DELAYS_SECONDS = (5.0, 15.0, 45.0)

logger = logging.getLogger(__name__)
_SleepFunc = Callable[[float], Awaitable[None]]
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


@dataclass
class ModelEndpoint:
    model_name: str
    api_key: str
    base_url: str | None
    headers: dict[str, str] | None = None


@dataclass
class DABStepDeps:
    data_dir: Path
    workspace: PythonWorkspace
    file_summary: str
    manual_excerpt: str | None = None
    runtime_patterns: str | None = None
    memory_context: str | None = None
    route_cards: list[RouteCard] = field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None
    evaluation_policy: EvaluationPolicy = field(default_factory=EvaluationPolicy.development)


class DABStepAnswer(BaseModel):
    agent_answer: str = Field(description="Final answer formatted according to the task guidelines.")
    reasoning: str = Field(description="Concise explanation of the calculation path.")
    used_code: bool = Field(description="Whether Python code was executed to compute or verify the answer.")


class NumericToolResult(BaseModel):
    value: float = Field(description="Unrounded numeric value.")
    formatted: str = Field(description="Value formatted with the requested decimal places.")
    method: str = Field(description="Deterministic helper and assumptions used.")


class ChoiceToolResult(BaseModel):
    choice: str = Field(description="Selected categorical answer.")
    cost: float = Field(description="Unrounded cost for the selected choice.")
    formatted_cost: str = Field(description="Choice and cost formatted as choice:cost.")
    method: str = Field(description="Deterministic helper and assumptions used.")


BASE_INSTRUCTIONS = """\
You are a DABStep benchmark specialist. Solve financial payments data questions exactly.

Rules:
- Read the documentation before relying on field names or business terms.
- Use execute_python_code for calculations, joins, filtering, fee matching, and verification.
- Prefer deterministic DABStep helper functions for fee rules, monthly metrics, and fee simulation. Use ad hoc pandas only for inspection or final verification.
- Treat MemoryLake context as reusable hypotheses, not as answers. Verify memory-derived guidance with documentation, deterministic helpers, or Python before using it.
- Prefer deterministic pandas code over mental arithmetic.
- For data-filtering and fee-matching questions, include self-checks in final code: validate categorical filter values with check_categorical(...), guard filtered frames with assert_nonempty(...), call match_count_summary(...) for fee-rule matching, and inspect row counts plus key intermediate magnitudes before finalizing.
- Return only the structured DABStepAnswer fields requested by the schema.
- Keep agent_answer concise and match the task guidelines exactly.
"""


HELPER_DOC_LINES: dict[str, str] = {
    "matching_fee_ids": "matching_fee_ids(fees, context) and matches_fee_rule(...) for wildcard-aware fee-rule matching.",
    "match_count_summary": "match_count_summary(payments, fees, context_base=...) before finalizing fee matching calculations; it summarizes zero/single/multi fee-rule matches per transaction and warns about overly strict filters.",
    "matches_fee_rule": "matches_fee_rule(rule, context) to test a single fee rule against a transaction context.",
    "calculate_fee": "calculate_fee(fixed_amount, rate, transaction_value) for the DABStep fee formula.",
    "average_scheme_fee": "average_scheme_fee(...) for per-scheme average fee comparisons.",
    "mcc_code_for_description": "mcc_code_for_description(mcc_table, description) to convert MCC descriptions to integer MCC codes before fee filtering.",
    "most_expensive_mccs_for_amount": "most_expensive_mccs_for_amount(data, amount=X) for 'most expensive MCC for a transaction of X euros, in general' questions; empty MCC lists are wildcards over every MCC in the fee schedule and the cost per MCC is the mean fee across applicable rules.",
    "applicable_fee_ids_for_merchant_period": "applicable_fee_ids_for_merchant_period(...) for merchant-specific applicable fee ID questions by day or month.",
    "fee_affected_merchants_for_year": "fee_affected_merchants_for_year(...) for questions asking which merchants are affected by a fee rule or a fee-rule applicability restriction.",
    "total_fees_for_merchant_period": "total_fees_for_merchant_period(...) for total fee questions by day, month, or year; it sums every matching fee rule per transaction and defaults to EUR-volume fraud-rate buckets for total-fee semantics.",
    "merchant_mcc_fee_delta_for_year": 'merchant_mcc_fee_delta_for_year(...) for "merchant changed MCC before year started" counterfactual delta tasks; it uses the same total-fee matching semantics and accepts fraud_metric="count" only when a task explicitly requires count-based fee buckets.',
    "fee_rate_delta_for_month": "fee_rate_delta_for_month(...) for explicit rate-change monthly delta simulations.",
    "fee_fixed_component_delta_for_month": "fee_fixed_component_delta_for_month(...) only for simulations that explicitly change a fee rule's fixed amount; when a task says the relative fee changed to some value, treat that value as the rule's rate and use fee_rate_delta_for_period instead.",
    "fee_rate_delta_for_period": "fee_rate_delta_for_period(...) for relative-fee change deltas over a month or year; treat 'relative fee changed to X' as setting the affected fee rule's rate to X.",
    "optimize_aci_for_fraudulent_transactions": "optimize_aci_for_fraudulent_transactions(...) for ACI incentive / lowest-fee simulations on fraudulent transactions.",
    "calculate_fee_monthly_metrics": "calculate_fee_monthly_metrics(...) for count-based monthly_volume/monthly_fraud_level diagnostics. For fee matching, total fees paid, and MCC-change total-fee deltas, prefer the helpers' default EUR-volume fraud bucket unless the question explicitly says count-based.",
    "calculate_monthly_metrics": "calculate_monthly_metrics(...), get_month_day_range(...), add_intracountry_flag(...), and top_fraud_ip_country_by_rate(...) for monthly and fraud tasks.",
    "add_intracountry_flag": "add_intracountry_flag(payments, acquirer_countries) to derive the intracountry flag used by fee rules.",
    "fraud_rate_by_volume": "fraud_rate_by_volume(...) and fraud_rate_by_group(...) for manual/schema-derived fraud-rate tasks; DABStep fraud rate is fraudulent EUR volume divided by total EUR volume, not fraudulent transaction count.",
    "fraud_rate_by_group": "fraud_rate_by_group(payments, group_by) for fraud rates split by a grouping column, using EUR-volume semantics.",
    "average_transactions_per_unique_email": "average_transactions_per_unique_email(...), average_transaction_amount_per_unique_email(...), and repeat_customer_percentage(...) for shopper/email/customer metrics; ignore missing email addresses unless the question explicitly asks about missing values.",
    "average_transaction_amount_per_unique_email": "average_transaction_amount_per_unique_email(payments) for per-unique-email amount metrics; ignore missing email addresses unless asked.",
    "repeat_customer_percentage": "repeat_customer_percentage(...) for repeat-customer metrics; determine repeat status from the full payments table.",
    "field_domain_values": "field_domain_values(...) for possible values / categorical domain questions; combine manual-defined values with observed data and do not assume unobserved manual values are impossible.",
    "fee_factor_monotonicity": "fee_factor_monotonicity(...) for questions about which fee factors become cheaper when decreased; distinguish tier/risk factors from formula operands.",
    "format_decimal_places": "format_decimal_places(value, places) for fixed decimal-place answer contracts after computing the value.",
}

CORE_HELPER_LINES = [
    "Import helpers from the preloaded `dabstep` module when needed, e.g. `from dabstep import load_dabstep_data`.",
    "Do not import or rely on `dabstep_helpers`; it is not the runtime helper module for this agent.",
    "load_dabstep_data(data_dir) to load fees, payments, merchants, and reference tables.",
    HELPER_DOC_LINES["format_decimal_places"],
]

TYPED_TOOL_DOC_LINES: dict[str, list[str]] = {
    "fee_matching": [
        "compute_total_fees(...) for total fees paid by merchant over a day, month, or year.",
    ],
    "fee_simulation": [
        "compute_total_fees(...) for total fees paid by merchant over a day, month, or year.",
        "compute_mcc_fee_delta(...) for merchant MCC-change yearly delta tasks.",
        'compute_relative_fee_delta(...) for monthly "relative fee changed to 1" simulations where the fee becomes rate-only.',
        "compute_best_aci_for_fraudulent_transactions(...) for ACI incentive / fraudulent transaction optimization; use the returned choice when the benchmark expects only the ACI letter.",
    ],
}


def build_helper_section(deps: DABStepDeps) -> str:
    plan = deps.analysis_plan
    if plan is not None and plan.recommended_helpers:
        helper_names = [name for name in plan.recommended_helpers if name in HELPER_DOC_LINES]
        selected_routes = plan.selected_route_ids
    else:
        helper_names = list(HELPER_DOC_LINES)
        selected_routes = list(TYPED_TOOL_DOC_LINES)

    typed_tool_lines: list[str] = []
    seen_tools: set[str] = set()
    for route_id in selected_routes:
        for line in TYPED_TOOL_DOC_LINES.get(route_id, []):
            if line not in seen_tools:
                seen_tools.add(line)
                typed_tool_lines.append(line)

    lines = ["The execute_python_code workspace preloads deterministic DABStep helpers. You can call:"]
    if typed_tool_lines:
        lines.append("- For common fee totals, prefer typed tools over free-form Python stdout copying:")
        lines.extend(f"  - {line}" for line in typed_tool_lines)
    lines.extend(f"- {line}" for line in CORE_HELPER_LINES)
    lines.extend(
        f"- {HELPER_DOC_LINES[name]}"
        for name in dict.fromkeys(helper_names)
        if HELPER_DOC_LINES[name] not in CORE_HELPER_LINES
    )
    return "\n".join(lines)


def build_instructions(deps: DABStepDeps) -> str:
    manual = deps.manual_excerpt or "(manual excerpt not loaded)"
    route_card_text = format_route_cards(deps.route_cards) if deps.route_cards else ""
    patterns = route_card_text or deps.runtime_patterns or "(no runtime route patterns loaded)"
    memory_context = deps.memory_context or "(no MemoryLake context retrieved)"
    analysis_plan = format_analysis_plan(deps.analysis_plan)
    helper_section = build_helper_section(deps)
    return f"""\
{BASE_INSTRUCTIONS}

Data directory: {deps.data_dir}

Available files:
{deps.file_summary}

Manual excerpt:
{manual}

Pydantic graph route-card orchestration:
{patterns}

{analysis_plan}

MemoryLake retrieved context:
{memory_context}

{helper_section}

Use the execute_python_code tool when the answer depends on the dataset. The final output must include agent_answer.
"""


def build_model_from_env() -> OpenAIChatModel:
    endpoint = resolve_model_endpoint_from_env()
    return _model_for_endpoint(endpoint)


def build_teacher_model_from_env() -> OpenAIChatModel:
    """Teacher/labeling model; DABSTEP_TEACHER_* overrides fall back to the solver endpoint.

    Keeping teacher and student separately configurable preserves the
    cross-model validity of discrimination: a student agreeing with its own
    teacher is weaker evidence than agreement across model families.
    """
    endpoint = resolve_model_endpoint_from_env()
    model_name = os.getenv("DABSTEP_TEACHER_MODEL", "").strip() or endpoint.model_name
    base_url = os.getenv("DABSTEP_TEACHER_BASE_URL", "").strip() or endpoint.base_url
    api_key = os.getenv("DABSTEP_TEACHER_API_KEY", "").strip() or endpoint.api_key
    return _model_for_endpoint(ModelEndpoint(
        model_name=model_name, api_key=api_key, base_url=base_url, headers=endpoint.headers,
    ))


def build_semantic_planner_model_from_env() -> OpenAIChatModel:
    """Short semantic-planning model; planner-specific overrides are optional."""
    endpoint = resolve_model_endpoint_from_env()
    model_name = os.getenv("DABSTEP_PLANNER_MODEL", "").strip() or endpoint.model_name
    base_url = os.getenv("DABSTEP_PLANNER_BASE_URL", "").strip() or endpoint.base_url
    api_key = os.getenv("DABSTEP_PLANNER_API_KEY", "").strip() or endpoint.api_key
    return _model_for_endpoint(ModelEndpoint(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        headers=endpoint.headers,
    ))


def _model_for_endpoint(endpoint: ModelEndpoint) -> OpenAIChatModel:
    http_client_factory = lambda: _build_model_http_client(headers=endpoint.headers)
    http_client = http_client_factory()
    provider = OpenAIProvider(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        http_client=http_client,
    )
    provider._own_http_client = http_client
    provider._http_client_factory = http_client_factory
    return OpenAIChatModel(endpoint.model_name, provider=provider)


def _build_model_http_client(headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    client_headers = {"User-Agent": get_user_agent()}
    if headers:
        client_headers.update(headers)
    return httpx.AsyncClient(
        # read=120: a half-closed gateway socket must raise, not block — a
        # 600s dead read outlives every solve timeout and turns task
        # cancellation into a process-wide hang (observed in learn runs).
        timeout=httpx.Timeout(timeout=600, connect=5, read=120),
        headers=client_headers,
        transport=_RetryingAsyncTransport(httpx.AsyncHTTPTransport()),
    )


class _RetryingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        *,
        status_codes: Iterable[int] = MODEL_GATEWAY_RETRY_STATUS_CODES,
        delays: Iterable[float] = MODEL_GATEWAY_RETRY_DELAYS_SECONDS,
        sleep: _SleepFunc = asyncio.sleep,
    ) -> None:
        self._transport = transport
        self._status_codes = frozenset(status_codes)
        self._delays = tuple(delays)
        self._sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        for attempt in range(len(self._delays) + 1):
            try:
                response = await self._transport.handle_async_request(request)
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                if attempt == len(self._delays):
                    raise
                await self._wait_before_retry(request, attempt, exc=exc)
                continue

            if response.status_code not in self._status_codes or attempt == len(self._delays):
                return response

            await response.aread()
            await response.aclose()
            await self._wait_before_retry(request, attempt, response=response)

        raise RuntimeError("unreachable retry loop state")

    async def _wait_before_retry(
        self,
        request: httpx.Request,
        attempt: int,
        *,
        response: httpx.Response | None = None,
        exc: Exception | None = None,
    ) -> None:
        delay = self._delays[attempt]
        if response is not None:
            logger.warning(
                "Model gateway returned HTTP %s for %s %s; retrying in %.0fs (attempt %s/%s)",
                response.status_code,
                request.method,
                request.url,
                delay,
                attempt + 1,
                len(self._delays),
            )
        else:
            logger.warning(
                "Model gateway transport error %s for %s %s; retrying in %.0fs (attempt %s/%s)",
                type(exc).__name__ if exc else "unknown",
                request.method,
                request.url,
                delay,
                attempt + 1,
                len(self._delays),
            )
        await self._sleep(delay)

    async def aclose(self) -> None:
        await self._transport.aclose()


def resolve_model_endpoint_from_env() -> ModelEndpoint:
    load_dotenv()
    router_mode = os.getenv("DABSTEP_MEMORY_ROUTER_MODE", "off")
    model_name = os.getenv("DABSTEP_MODEL", DEFAULT_MODEL)
    headers: dict[str, str] | None = None

    if router_mode == "hosted":
        api_key = _first_env("DABSTEP_MEMORY_ROUTER_API_KEY", "MEMORYLAKE_API_KEY")
        base_url = _first_env(
            "DABSTEP_MEMORY_ROUTER_BASE_URL",
            "MEMORYLAKE_OPENAI_BASE_URL",
            default=DEFAULT_MEMORYLAKE_ROUTER_BASE_URL,
        )
        required_key = "DABSTEP_MEMORY_ROUTER_API_KEY or MEMORYLAKE_API_KEY"
    elif router_mode == "byok":
        api_key = _first_env("DABSTEP_MEMORY_ROUTER_API_KEY", "DABSTEP_OPENAI_API_KEY", "OPENAI_API_KEY")
        base_url = _first_env(
            "DABSTEP_MEMORY_ROUTER_BASE_URL",
            "MEMORYLAKE_OPENAI_BASE_URL",
            default=DEFAULT_MEMORYLAKE_BYOK_BASE_URL,
        )
        required_key = "DABSTEP_MEMORY_ROUTER_API_KEY, DABSTEP_OPENAI_API_KEY, or OPENAI_API_KEY"
        memorylake_key = os.getenv("MEMORYLAKE_API_KEY")
        if not memorylake_key:
            raise RuntimeError("MEMORYLAKE_API_KEY is required for Memory Router BYOK mode")
        headers = {"x-memorylake-api-key": memorylake_key}
    else:
        api_key = _first_env("DABSTEP_OPENAI_API_KEY", "OPENAI_API_KEY")
        base_url = _first_env("DABSTEP_OPENAI_BASE_URL", "OPENAI_BASE_URL")
        required_key = "DABSTEP_OPENAI_API_KEY or OPENAI_API_KEY"

    if not api_key:
        raise RuntimeError(f"{required_key} is required")

    return ModelEndpoint(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        headers=headers,
    )


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def model_settings_from_env() -> dict[str, str]:
    thinking = os.getenv("DABSTEP_THINKING", DEFAULT_THINKING).strip()
    return {"thinking": thinking} if thinking else {}


def semantic_planner_model_settings_from_env() -> dict[str, str]:
    thinking = os.getenv("DABSTEP_PLANNER_THINKING", DEFAULT_PLANNER_THINKING).strip()
    return {"thinking": thinking} if thinking else {}


def create_agent(model: OpenAIChatModel | str | None = None) -> Agent[DABStepDeps, DABStepAnswer]:
    agent = Agent(
        model or build_model_from_env(),
        deps_type=DABStepDeps,
        output_type=DABStepAnswer,
        instructions=BASE_INSTRUCTIONS,
        model_settings=model_settings_from_env(),
        defer_model_check=True,
    )

    @agent.instructions
    def add_dabstep_context(ctx: RunContext[DABStepDeps]) -> str:
        return build_instructions(ctx.deps)

    @agent.output_validator
    def validate_answer_shape(ctx: RunContext[DABStepDeps], output: DABStepAnswer) -> DABStepAnswer:
        del ctx
        answer = output.agent_answer.strip()
        if not answer:
            raise ModelRetry("agent_answer must not be empty unless the task explicitly asks for an empty string.")
        lowered = answer.lower()
        if lowered.startswith(("answer:", "the answer is", "result:")):
            raise ModelRetry("Return only the requested value in agent_answer, without explanatory prefixes.")
        output.agent_answer = answer
        return output

    return agent
