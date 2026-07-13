from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dabstep_agent_pydantic.dataset import Task
from semantic_shadow_eval import REQUIRED_CATEGORY_COUNTS  # noqa: E402
from semantic_shadow_eval import classify_shadow_category  # noqa: E402
from semantic_shadow_eval import flatten_selected_ids  # noqa: E402
from semantic_shadow_eval import load_representative_ids  # noqa: E402
from semantic_shadow_eval import select_representative_ids  # noqa: E402
from semantic_shadow_eval import summarize_shadow_records  # noqa: E402


def _trace(
    *,
    semantic_answer=None,
    fallback_reason=None,
    verifier=None,
    elapsed_ms=100,
    usage=None,
):
    return {
        "mode": "shadow",
        "plan": {"analysis_spec": {"measure": {"kind": "count"}}},
        "candidates": {
            "candidates": [
                {"candidate_id": "proposed", "origin": "proposed"},
                {"candidate_id": "rival", "origin": "uncertainty_rival"},
            ]
        },
        "executions": {},
        "verifier": verifier,
        "usage": usage or {},
        "fallback_reason": fallback_reason,
        "selected_path": "legacy",
        "semantic_candidate_answer": semantic_answer,
        "elapsed_ms": elapsed_ms,
    }


def _synthetic_tasks() -> list[Task]:
    tasks: list[Task] = []
    # customer / fraud
    for i in range(15):
        tasks.append(Task(
            task_id=f"fraud-{i:02d}",
            question=f"What is the fraud rate for merchant group {i} in 2023?",
            guidelines="Return 4 decimals.",
        ))
    # general
    for i in range(15):
        tasks.append(Task(
            task_id=f"general-{i:02d}",
            question=f"How many payments are there for cohort {i}?",
            guidelines="Return an integer.",
        ))
    # fee
    for i in range(10):
        tasks.append(Task(
            task_id=f"fee-{i:02d}",
            question=f"What are the total fees for Merchant_{i} in January 2023?",
            guidelines="Return 2 decimals.",
        ))
    # schema / counterfactual
    for i in range(10):
        tasks.append(Task(
            task_id=f"schema-{i:02d}",
            question=(
                f"If Merchant_{i} changed its MCC to 5411 in 2023, "
                "what would the fee delta be?"
            ),
            guidelines="Return 6 decimals.",
        ))
    return tasks


def test_select_representative_ids_is_stratified_and_order_invariant():
    tasks = _synthetic_tasks()
    selected = select_representative_ids(tasks)
    assert {key: len(value) for key, value in selected.items()} == REQUIRED_CATEGORY_COUNTS
    flat = flatten_selected_ids(selected)
    assert len(flat) == 30
    assert len(flat) == len(set(flat))

    reversed_tasks = list(reversed(tasks))
    selected_rev = select_representative_ids(reversed_tasks)
    assert selected == selected_rev
    assert flatten_selected_ids(selected_rev) == flat


def test_select_representative_ids_uses_generic_classification_only():
    assert classify_shadow_category(
        "What is the fraud rate by card scheme?",
        None,
    ) == "customer_fraud"
    assert classify_shadow_category(
        "How many payments are there?",
        None,
    ) == "general"
    assert classify_shadow_category(
        "What are the total fees for Merchant_A in 2023?",
        None,
    ) == "fee"
    assert classify_shadow_category(
        "If the merchant changed its MCC, what is the fee delta?",
        None,
    ) == "schema_counterfactual"


def test_select_representative_ids_raises_when_bucket_too_small():
    tasks = [
        Task(task_id="g1", question="How many payments?", guidelines="int"),
    ]
    with pytest.raises(ValueError, match="not enough tasks"):
        select_representative_ids(tasks)


def test_external_fixture_loader_still_works(tmp_path: Path):
    payload = {
        "customer_fraud": [f"cf-{i}" for i in range(10)],
        "general": [f"g-{i}" for i in range(10)],
        "fee": [f"f-{i}" for i in range(5)],
        "schema_counterfactual": [f"s-{i}" for i in range(5)],
    }
    path = tmp_path / "external_ids.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ids = load_representative_ids(path)
    assert len(ids) == 30
    assert len(set(ids)) == 30


def test_committed_fixture_is_removed():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "semantic_representative_ids.json"
    assert not fixture.exists(), "fixed benchmark ID fixture must not remain in the repo"


def test_repo_rejects_committed_answer_mapping_artifacts():
    root = Path(__file__).resolve().parents[1]
    forbidden_names = {
        "task_answers.json",
        "task_answer_map.json",
        "benchmark_answers.json",
        "semantic_representative_ids.json",
    }
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        if path.name in forbidden_names:
            offenders.append(str(path.relative_to(root)))
        if path.suffix == ".json" and "answer" in path.name.lower() and "fixture" in path.parts:
            # Allow model_reference shape fixtures that are not answer maps.
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "task_id" in text and "agent_answer" in text and "reference_answer" in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_shadow_report_aggregates_required_metrics():
    records = [
        {
            "task_id": "agree",
            "agent_answer": "42",
            "semantic_trace": _trace(
                semantic_answer="42",
                verifier={
                    "accepted": True,
                    "selected_candidate_id": "proposed",
                    "level": "semantic_judge",
                    "reason": "resolved",
                    "rejected_candidate_ids": [],
                    "judge_votes": [{"stage": "semantic_judge"}],
                },
                elapsed_ms=100,
                usage={
                    "planner": {"calls": 1, "input_tokens": 100, "output_tokens": 20, "latency_ms": 50},
                    "semantic_judge": {"calls": 1, "input_tokens": 40, "output_tokens": 5, "latency_ms": 20},
                },
            ),
        },
        {
            "task_id": "fallback",
            "agent_answer": "legacy",
            "semantic_trace": _trace(
                fallback_reason="global model-call budget exhausted",
                elapsed_ms=200,
            ),
        },
        {
            "task_id": "disagree",
            "agent_answer": "legacy",
            "semantic_trace": _trace(
                semantic_answer="semantic",
                verifier={
                    "accepted": True,
                    "selected_candidate_id": "rival",
                    "level": "certified_policy",
                    "reason": "rival selected",
                    "rejected_candidate_ids": ["bad"],
                    "judge_votes": [],
                },
                elapsed_ms=300,
                usage={
                    "planner": {"calls": 1, "input_tokens": 80, "output_tokens": 10, "latency_ms": 40}
                },
            ),
        },
        {
            "task_id": "legacy-only",
            "agent_answer": "legacy",
            "semantic_trace": {
                "mode": "legacy",
                "plan": None,
                "candidates": None,
                "executions": {},
                "verifier": None,
                "usage": {},
                "fallback_reason": None,
                "selected_path": "legacy",
                "semantic_candidate_answer": None,
                "elapsed_ms": 0,
            },
        },
    ]

    report = summarize_shadow_records(records)

    assert report["tasks"] == 4
    assert report["path_coverage"]["selected_paths"] == {"legacy": 4}
    assert report["path_coverage"]["semantic_outcomes"] == {
        "resolved": 2,
        "fallback": 1,
        "not_attempted": 1,
    }
    assert report["agreement"] == {
        "comparable": 2,
        "agreed": 1,
        "disagreed": 1,
        "rate": 0.5,
        "disagreement_task_ids": ["disagree"],
    }
    assert report["fallback"]["count"] == 1
    assert report["fallback"]["rate"] == 0.25
    assert report["fallback"]["reasons"] == {"global model-call budget exhausted": 1}
    assert report["judges"]["tasks"] == 1
    assert report["judges"]["rate"] == 0.25
    assert report["verifier_interventions"]["count"] == 1
    assert report["verifier_interventions"]["task_ids"] == ["disagree"]
    assert report["latency_ms"] == {
        "count": 3,
        "average": 200.0,
        "p50": 200.0,
        "p95": 290.0,
        "maximum": 300.0,
    }
    assert report["tokens_by_stage"]["planner"] == {
        "calls": 2,
        "input_tokens": 180,
        "output_tokens": 30,
        "latency_ms": 90,
    }
    assert report["tokens_by_stage"]["semantic_judge"]["output_tokens"] == 5


def test_shadow_report_distinguishes_exact_and_numeric_agreement():
    records = [
        {
            "task_id": "format-only",
            "agent_answer": "0.00",
            "semantic_trace": _trace(semantic_answer="0.0"),
        },
        {
            "task_id": "semantic-difference",
            "agent_answer": "12.5",
            "semantic_trace": _trace(semantic_answer="13.5"),
        },
    ]

    report = summarize_shadow_records(records)

    assert report["agreement"]["agreed"] == 0
    assert report["numeric_agreement"] == {
        "comparable": 2,
        "equivalent": 1,
        "different": 1,
        "rate": 0.5,
        "difference_task_ids": ["semantic-difference"],
    }


def test_shadow_harness_cli_starts_without_editable_install():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "scripts/semantic_shadow_eval.py", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "30-task semantic shadow harness" in completed.stdout
    assert "--fixture" in completed.stdout
