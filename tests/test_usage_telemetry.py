import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic_ai.usage import RunUsage

from dabstep_agent_pydantic.agent import DABStepAnswer
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.distill.bootstrap import bootstrap_labels
from dabstep_agent_pydantic.distill.reference_gen import generate_references
from dabstep_agent_pydantic.usage_telemetry import CallUsage
from dabstep_agent_pydantic.usage_telemetry import UsageBudgetExceeded
from dabstep_agent_pydantic.usage_telemetry import UsageLedger
from dabstep_agent_pydantic.usage_telemetry import call_usage_from_result
from dabstep_agent_pydantic.workflow import run_task_workflow


def test_call_usage_and_ledger_aggregate_by_stage():
    ledger = UsageLedger()
    ledger.record(
        CallUsage(
            stage="planner",
            input_tokens=100,
            output_tokens=20,
            latency_ms=50,
            retries=1,
            model_fingerprint="provider:model-a",
        )
    )
    ledger.record(
        CallUsage(
            stage="planner",
            input_tokens=70,
            output_tokens=10,
            latency_ms=30,
            retries=0,
        )
    )

    assert ledger.summary()["planner"] == {
        "calls": 2,
        "input_tokens": 170,
        "output_tokens": 30,
        "latency_ms": 80,
        "retries": 1,
        "model_fingerprints": ["provider:model-a"],
    }


def test_usage_ledger_enforces_global_call_budget():
    ledger = UsageLedger(max_calls=1)
    ledger.record(CallUsage(stage="judge", input_tokens=10, output_tokens=2, latency_ms=1))

    assert ledger.can_call("judge") is False
    with pytest.raises(UsageBudgetExceeded, match="global model-call budget of 1"):
        ledger.record(CallUsage(stage="repair", input_tokens=5, output_tokens=1, latency_ms=1))


def test_pydantic_ai_adapter_handles_property_usage_and_model_fingerprint():
    result = SimpleNamespace(
        usage=RunUsage(requests=2, input_tokens=123, output_tokens=45),
        response=SimpleNamespace(
            model_name="model-a",
            provider_name="provider",
            provider_details={"system_fingerprint": "fp-123"},
        ),
    )

    usage = call_usage_from_result(result, stage="solver", latency_ms=12.6, retries=2)

    assert usage == CallUsage(
        stage="solver",
        input_tokens=123,
        output_tokens=45,
        latency_ms=13,
        retries=2,
        model_fingerprint="fp-123",
    )


def test_pydantic_ai_adapter_handles_method_and_legacy_token_names():
    class LegacyResult:
        response = SimpleNamespace(
            model_name="model-b",
            provider_name="provider",
            provider_details=None,
        )

        def usage(self):
            return {"prompt_tokens": 7, "completion_tokens": 3}

    usage = call_usage_from_result(LegacyResult(), stage="reference", latency_ms=4)

    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.model_fingerprint == "provider:model-b"


class _TelemetryResult:
    def __init__(self, answer: str, *, input_tokens: int = 11, output_tokens: int = 4):
        self.output = DABStepAnswer(agent_answer=answer, reasoning="computed", used_code=False)
        self.usage = RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        self.response = SimpleNamespace(
            model_name="model-a",
            provider_name="provider",
            provider_details=None,
        )


class _TelemetrySolver:
    async def run(self, prompt, **kwargs):
        return _TelemetryResult("42")


def test_workflow_records_solver_usage_without_changing_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "off")
    monkeypatch.setattr("dabstep_agent_pydantic.workflow.create_agent", _TelemetrySolver)
    task = Task(task_id="t1", question="What total fee did merchant A pay?", guidelines="Return a number.")

    record = asyncio.run(
        run_task_workflow(
            task,
            data_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            file_summary="Data directory: empty",
        )
    )

    assert record["agent_answer"] == "42"
    assert record["usage_trace"]["solver"]["calls"] == 1
    assert record["usage_trace"]["solver"]["input_tokens"] == 11
    assert record["usage_trace"]["solver"]["output_tokens"] == 4


def test_reference_output_persists_reference_usage_trace(tmp_path):
    persist_path = tmp_path / "references.jsonl"

    records = asyncio.run(
        generate_references(
            instances=[{"task_id": "1", "question": "q"}],
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            agent=_TelemetrySolver(),
            persist_path=persist_path,
        )
    )

    row = json.loads(persist_path.read_text().strip())
    assert records["1"].answer == "42"
    assert row["answer"] == "42"
    assert row["usage_trace"]["reference"]["input_tokens"] == 11


def test_bootstrap_output_persists_bootstrap_usage_trace(tmp_path):
    persist_path = tmp_path / "bootstrap.jsonl"

    labels = asyncio.run(
        bootstrap_labels(
            instances=[{"task_id": "1", "question": "q"}],
            label_tids=["1"],
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            samples=1,
            consensus=1,
            agent=_TelemetrySolver(),
            persist_path=persist_path,
        )
    )

    row = json.loads(persist_path.read_text().strip())
    assert labels["1"].answer == "42"
    assert row["answer"] == "42"
    assert row["usage_trace"]["bootstrap"]["output_tokens"] == 4


def test_failed_reference_calls_are_kept_in_external_learn_ledger(tmp_path):
    class FailingSolver:
        async def run(self, prompt, **kwargs):
            raise ConnectionError("gateway unavailable")

    ledger = UsageLedger()
    records = asyncio.run(
        generate_references(
            instances=[{"task_id": "1", "question": "q"}],
            data_dir=tmp_path,
            workspace_dir=tmp_path,
            agent=FailingSolver(),
            persist_path=tmp_path / "references.jsonl",
            usage_ledger=ledger,
        )
    )

    assert records == {}
    # Initial attempt plus the self-healing retry must both remain visible
    # even though neither produced a reference row.
    assert ledger.summary()["reference"]["calls"] == 2
    assert ledger.summary()["reference"]["retries"] == 1
