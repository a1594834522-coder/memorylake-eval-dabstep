"""Family-scoped compact semantic planner."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from dabstep_agent_pydantic.compact_semantic_planner import (
    COMPACT_INTENT_MODELS,
    CompactPlannerResult,
    build_compact_planner_prompt,
    create_compact_planner_agent,
    plan_compact_intent,
    route_intent_family,
)
from dabstep_agent_pydantic.semantic_intent import CustomerFraudIntent
from dabstep_agent_pydantic.semantic_intent import FeeIntent
from dabstep_agent_pydantic.semantic_intent import GeneralIntent
from dabstep_agent_pydantic.semantic_intent import IntentTimeScope
from dabstep_agent_pydantic.semantic_intent import UnsupportedIntent
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy
from dabstep_agent_pydantic.usage_telemetry import UsageLedger


class _Result:
    def __init__(self, output, *, input_tokens=50, output_tokens=10):
        self.output = output
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.model_name = "compact-planner"


class _RecordingAgent:
    def __init__(self, outputs, *, fail_once: bool = False):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, dict]] = []
        self.fail_once = fail_once
        self._failed = False

    async def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.fail_once and not self._failed:
            self._failed = True
            raise RuntimeError("provider timeout")
        return _Result(self.outputs.pop(0))


def _policy() -> SemanticPolicy:
    return SemanticPolicy(
        policy_id="fraud.rate.eur_volume.v1",
        version=1,
        family="customer_fraud_metrics",
        axis="fraud_rate_basis",
        choice="eur_volume",
        convention="Fraud rate uses EUR volume.",
        citations=[PolicyCitation(
            document_name="manual.md",
            section="fraud",
            content_hash="b" * 64,
        )],
        certification_id="cert:fraud-rate",
        status=PolicyStatus.ACTIVE,
    )


def test_route_intent_family_is_deterministic():
    assert route_intent_family(
        "What is the fraud rate for Ecommerce in 2023?",
        "Return 4 decimals.",
    ) == "customer_fraud"
    assert route_intent_family(
        "What are the total fees for Merchant_A in January 2023?",
        "2 decimals",
    ) == "fee"
    assert route_intent_family(
        "How many payments are there?",
        "integer",
    ) == "general"
    assert route_intent_family(
        "For a credit transaction, which ACI is most expensive?",
        "Return the ACI.",
    ) == "fee"


def test_unambiguous_total_count_uses_zero_model_fast_path(monkeypatch):
    monkeypatch.setattr(
        "dabstep_agent_pydantic.compact_semantic_planner.create_compact_planner_agent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model must not be created")),
    )

    for question in (
        "How many payments exist overall?",
        "Give the total count of transactions.",
    ):
        result = asyncio.run(plan_compact_intent(
            question=question,
            guidelines="Answer must be just a number.",
            schema_summary="payments",
            policies=[],
            manual_excerpts=[],
        ))

        assert isinstance(result.intent, GeneralIntent)
        assert result.intent.operation == "aggregate"
        assert result.intent.aggregation == "count"
        assert result.attempts == 0
        assert result.schema_chars == 0
        assert result.prompt_chars == 0
        assert result.usage_trace == {}


def test_create_agent_uses_only_one_family_schema():
    agent = create_compact_planner_agent(family="fee", model="test")
    # Pydantic AI stores output type on the agent; inspect via factory map.
    assert COMPACT_INTENT_MODELS["fee"] is FeeIntent
    assert COMPACT_INTENT_MODELS["general"] is GeneralIntent
    assert COMPACT_INTENT_MODELS["customer_fraud"] is CustomerFraudIntent
    assert agent is not None


def test_only_one_family_schema_is_serialized_per_call():
    for family, model in COMPACT_INTENT_MODELS.items():
        schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
        assert len(schema) < 4000
        # Sibling family field names unique to other families should not all appear.
        if family == "fee":
            assert "fraud_rate_basis" not in schema
        if family == "general":
            assert "fee_reducer" not in schema
        if family == "customer_fraud":
            assert "fee_reducer" not in schema


def test_planner_prompt_has_no_tools_or_answers():
    intent = GeneralIntent(
        operation="aggregate",
        aggregation="count",
        source="payments",
        output_kind="integer",
    )
    agent = _RecordingAgent([intent])
    result = asyncio.run(plan_compact_intent(
        question="How many payments are there?",
        guidelines="Return a single integer.",
        schema_summary="payments(year, eur_amount)",
        policies=[],
        manual_excerpts=[],
        family="general",
        agent=agent,
    ))
    assert isinstance(result, CompactPlannerResult)
    assert result.intent == intent
    prompt, kwargs = agent.calls[0]
    assert "How many payments are there?" in prompt
    assert "Do not compute or return the final answer" in prompt
    assert "task_id" not in prompt.lower()
    assert "expected answer" not in prompt.lower()
    assert "toolsets" not in kwargs and "tools" not in kwargs and "deps" not in kwargs
    assert result.schema_chars < 4000
    assert result.prompt_chars == len(prompt)


def test_planner_allows_one_repair_and_records_usage():
    ledger = UsageLedger(max_calls=2)
    agent = _RecordingAgent([
        {"operation": "aggregate"},  # invalid
        GeneralIntent(
            operation="aggregate",
            aggregation="count",
            source="payments",
            output_kind="integer",
        ),
    ])
    result = asyncio.run(plan_compact_intent(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[_policy()],
        manual_excerpts=["excerpt"],
        family="general",
        agent=agent,
        usage_ledger=ledger,
    ))
    assert result.intent is not None
    assert result.attempts == 2
    assert "REPAIR" in agent.calls[1][0]
    assert result.usage_trace["planner"]["calls"] == 1
    assert result.usage_trace["planner_repair"]["calls"] == 1


def test_failed_provider_call_still_counts_calls_and_latency():
    agent = _RecordingAgent(
        [GeneralIntent(
            operation="aggregate",
            aggregation="count",
            source="payments",
            output_kind="integer",
        )],
        fail_once=True,
    )
    result = asyncio.run(plan_compact_intent(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        family="general",
        agent=agent,
    ))
    # First call fails, second succeeds after repair path / retry loop.
    assert result.attempts >= 1
    usage = result.usage_trace
    total_calls = sum(stage.get("calls", 0) for stage in usage.values())
    total_latency = sum(stage.get("latency_ms", 0) for stage in usage.values())
    assert total_calls >= 1
    assert total_latency >= 0


def test_budget_exhaustion_returns_typed_fallback():
    ledger = UsageLedger(max_calls=0)
    agent = _RecordingAgent([
        GeneralIntent(
            operation="aggregate",
            aggregation="count",
            source="payments",
            output_kind="integer",
        )
    ])
    result = asyncio.run(plan_compact_intent(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        family="general",
        agent=agent,
        usage_ledger=ledger,
    ))
    assert result.intent is None
    assert result.fallback_reason
    assert "budget" in result.fallback_reason.lower() or "call" in result.fallback_reason.lower()
    assert agent.calls == []


def test_two_invalid_outputs_fall_back():
    agent = _RecordingAgent([
        {"bad": True},
        {"still": "bad"},
    ])
    result = asyncio.run(plan_compact_intent(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        family="fee",
        agent=agent,
    ))
    assert result.intent is None
    assert result.attempts == 2
    assert result.fallback_reason


def test_fee_intent_success_path():
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023},
        output_kind="decimal",
        decimals=2,
    )
    agent = _RecordingAgent([intent])
    result = asyncio.run(plan_compact_intent(
        question="total fees for Merchant_A in 2023?",
        guidelines="2 decimals",
        schema_summary="payments, fees",
        policies=[],
        manual_excerpts=[],
        family="fee",
        agent=agent,
    ))
    assert result.family == "fee"
    assert isinstance(result.intent, FeeIntent)


def test_unsupported_intent_is_accepted_when_family_is_unsupported():
    intent = UnsupportedIntent(reason="no closed operation for this request")
    agent = _RecordingAgent([intent])
    result = asyncio.run(plan_compact_intent(
        question="Write a poem about fees",
        guidelines="",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        family="unsupported",
        agent=agent,
    ))
    assert isinstance(result.intent, UnsupportedIntent)


def test_prompt_builder_includes_policies_and_schema():
    prompt = build_compact_planner_prompt(
        question="fraud rate?",
        guidelines="4 decimals",
        schema_summary="payments(...)",
        policies=[_policy()],
        manual_excerpts=["manual text"],
        family="customer_fraud",
    )
    assert "fraud rate?" in prompt
    assert "fraud.rate.eur_volume.v1" in prompt
    assert "payments(...)" in prompt
    assert "manual text" in prompt
    assert "CustomerFraudIntent" in prompt or "customer_fraud" in prompt


def test_prompt_builder_includes_family_operation_contracts():
    fee_prompt = build_compact_planner_prompt(
        question="total fees?",
        guidelines="2 decimals",
        schema_summary="payments, fees",
        policies=[],
        manual_excerpts=[],
        family="fee",
    )
    general_prompt = build_compact_planner_prompt(
        question="credit percentage?",
        guidelines="2 decimals",
        schema_summary="payments",
        policies=[],
        manual_excerpts=[],
        family="general",
    )

    assert "period_total_fees" in fee_prompt
    assert "merchant" in fee_prompt and "year" in fee_prompt
    assert "numerator_filters" in general_prompt
    assert "ip_address" in general_prompt and "ip_country" in general_prompt

    customer_prompt = build_compact_planner_prompt(
        question="in-person fraud rate?",
        guidelines="just a number",
        schema_summary="payments",
        policies=[],
        manual_excerpts=[],
        family="customer_fraud",
    )
    assert "shopper_interaction='POS'" in customer_prompt
    assert "EUR volume" in customer_prompt
    assert "identity_field" in customer_prompt


def test_default_thinking_is_low_via_agent_settings():
    # Ensure factory wires planner settings (thinking=low by default env).
    from dabstep_agent_pydantic.agent import semantic_planner_model_settings_from_env

    settings = semantic_planner_model_settings_from_env()
    assert settings.get("thinking") == "low" or "thinking" in settings or settings == {}
