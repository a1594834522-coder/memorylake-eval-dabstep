from __future__ import annotations

import asyncio

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.semantic_planner import AnalysisPlanProposal
from dabstep_agent_pydantic.semantic_planner import SemanticPlannerResult
from dabstep_agent_pydantic.semantic_planner import SemanticUncertainty
from dabstep_agent_pydantic.semantic_planner import create_semantic_planner_agent
from dabstep_agent_pydantic.semantic_planner import plan_semantics
from dabstep_agent_pydantic.semantic_policy import PolicyCitation
from dabstep_agent_pydantic.semantic_policy import PolicyStatus
from dabstep_agent_pydantic.semantic_policy import SemanticPolicy
from dabstep_agent_pydantic.usage_telemetry import UsageLedger


def _spec() -> AnalysisSpec:
    return AnalysisSpec(
        source=SourceSpec(table="payments"),
        measure=AggregateMeasure(kind="sum", column="eur_amount"),
        output=AnalysisOutputContract(kind="decimal", decimals=2),
        policy_ids=["payments.total.sum.v1"],
    )


def _policy() -> SemanticPolicy:
    return SemanticPolicy(
        policy_id="payments.total.sum.v1",
        version=1,
        family="general_analytics",
        axis="aggregation",
        choice="sum",
        convention="Sum EUR amount over the explicitly filtered payment population.",
        citations=[PolicyCitation(
            document_name="payments-readme.md",
            section="eur_amount",
            content_hash="a" * 64,
        )],
        certification_id="cert:payments-total-sum",
        status=PolicyStatus.ACTIVE,
    )


class _Result:
    def __init__(self, output, *, input_tokens=100, output_tokens=20):
        self.output = output
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.model_name = "planner-model"


class _RecordingAgent:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, dict]] = []

    async def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return _Result(self.outputs.pop(0))


def test_planner_prompt_is_minimal_structured_and_has_no_tools():
    proposal = AnalysisPlanProposal(
        analysis_spec=_spec(),
        uncertainties=[SemanticUncertainty(
            axis="time_scope",
            proposed_choice="explicit_period",
            rival_policy_ids=[],
            rationale="The question names a period.",
            confidence=0.95,
        )],
    )
    agent = _RecordingAgent([proposal])

    result = asyncio.run(plan_semantics(
        question="What is the total EUR amount in 2023?",
        guidelines="Return a number rounded to 2 decimals.",
        schema_summary="payments(year, eur_amount)",
        policies=[_policy()],
        manual_excerpts=["eur_amount is the transaction value in EUR."],
        agent=agent,
    ))

    assert isinstance(result, SemanticPlannerResult)
    assert result.proposal == proposal
    prompt, kwargs = agent.calls[0]
    assert "What is the total EUR amount in 2023?" in prompt
    assert "payments(year, eur_amount)" in prompt
    assert "payments.total.sum.v1" in prompt
    assert "eur_amount is the transaction value" in prompt
    assert "Do not compute or return the final answer" in prompt
    assert "toolsets" not in kwargs and "tools" not in kwargs and "deps" not in kwargs


def test_planner_allows_exactly_one_structured_repair():
    agent = _RecordingAgent([
        {"analysis_spec": {"source": {"table": "unknown"}}},
        AnalysisPlanProposal(analysis_spec=_spec()),
    ])

    result = asyncio.run(plan_semantics(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        agent=agent,
    ))

    assert result.proposal is not None
    assert result.attempts == 2
    assert len(agent.calls) == 2
    assert "REPAIR" in agent.calls[1][0]


def test_planner_falls_back_after_second_invalid_output():
    agent = _RecordingAgent([
        {"analysis_spec": None, "unsupported_reason": None},
        {"analysis_spec": {"python_code": "df.sum()"}},
        AnalysisPlanProposal(analysis_spec=_spec()),
    ])

    result = asyncio.run(plan_semantics(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        agent=agent,
    ))

    assert result.proposal is None
    assert result.attempts == 2
    assert result.fallback_reason and "invalid semantic planner output" in result.fallback_reason
    assert len(agent.calls) == 2


def test_planner_records_usage_by_initial_and_repair_stage():
    ledger = UsageLedger(max_calls=2)
    agent = _RecordingAgent([
        {"bad": "shape"},
        AnalysisPlanProposal(unsupported_reason="The requested operation is outside the DSL."),
    ])

    result = asyncio.run(plan_semantics(
        question="q",
        guidelines="g",
        schema_summary="s",
        policies=[],
        manual_excerpts=[],
        agent=agent,
        usage_ledger=ledger,
    ))

    assert result.proposal is not None
    assert result.usage_trace["planner"]["input_tokens"] == 100
    assert result.usage_trace["planner_repair"]["output_tokens"] == 20


def test_semantic_planner_agent_is_created_without_tools(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("dabstep_agent_pydantic.semantic_planner.Agent", FakeAgent)
    monkeypatch.delenv("DABSTEP_PLANNER_THINKING", raising=False)

    created = create_semantic_planner_agent(model="test-model")

    assert isinstance(created, FakeAgent)
    assert captured["kwargs"]["output_type"] is AnalysisPlanProposal
    assert "tools" not in captured["kwargs"] and "toolsets" not in captured["kwargs"]
    assert captured["kwargs"]["retries"] == 0
    assert captured["kwargs"]["model_settings"] == {"thinking": "low"}
