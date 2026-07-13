from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.dataset import load_tasks
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.runner import run_benchmark
from dabstep_agent_pydantic.semantic_workflow import SemanticMode


REQUIRED_CATEGORY_COUNTS = {
    "customer_fraud": 10,
    "general": 10,
    "fee": 5,
    "schema_counterfactual": 5,
}


def classify_shadow_category(question: str, guidelines: str | None = None) -> str:
    """Generic family classification for stratified shadow selection.

    Uses the zero-LLM planner family and coarse question features only — never
    task IDs, exact benchmark question tables, or answers.
    """
    text = f"{question}\n{guidelines or ''}".lower()
    family = plan_task(question=question, guidelines=guidelines, route_cards=[]).task_family
    if family == "customer_fraud_metrics":
        return "customer_fraud"
    if family == "fee_simulation":
        return "schema_counterfactual"
    if family in {"fee_matching", "fee_analysis"}:
        return "fee"
    if family == "schema_semantics":
        return "schema_counterfactual"
    # Keyword backstop when route cards are empty.
    if "fraud" in text or "customer" in text or "email" in text or "shopper" in text:
        return "customer_fraud"
    if any(token in text for token in ("fee", "mcc", "aci", "card scheme")):
        if any(token in text for token in ("changed", "delta", "instead", "counterfactual", "would be")):
            return "schema_counterfactual"
        return "fee"
    if "possible values" in text or "domain" in text:
        return "schema_counterfactual"
    return "general"


def select_representative_ids(tasks: Sequence[Task]) -> dict[str, list[str]]:
    """Deterministic stratified selection via stable SHA-256 ordering."""
    buckets: dict[str, list[Task]] = {category: [] for category in REQUIRED_CATEGORY_COUNTS}
    for task in tasks:
        category = classify_shadow_category(task.question, task.guidelines)
        buckets[category].append(task)

    selected: dict[str, list[str]] = {}
    for category, required in REQUIRED_CATEGORY_COUNTS.items():
        ordered = sorted(
            buckets[category],
            key=lambda task: (
                hashlib.sha256(f"{category}\0{task.task_id}".encode("utf-8")).hexdigest(),
                task.task_id,
            ),
        )
        if len(ordered) < required:
            raise ValueError(
                f"not enough tasks for category {category!r}: need {required}, found {len(ordered)}"
            )
        selected[category] = [task.task_id for task in ordered[:required]]
    return selected


def flatten_selected_ids(selected: dict[str, list[str]]) -> list[str]:
    task_ids: list[str] = []
    for category in REQUIRED_CATEGORY_COUNTS:
        task_ids.extend(selected[category])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("selected representative task IDs must be unique")
    return task_ids


def load_representative_ids(path: Path) -> list[str]:
    """Load an operator-supplied external fixture (never a committed default)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != set(REQUIRED_CATEGORY_COUNTS):
        raise ValueError(
            "representative fixture must contain exactly: "
            + ", ".join(REQUIRED_CATEGORY_COUNTS)
        )

    task_ids: list[str] = []
    for category, required_count in REQUIRED_CATEGORY_COUNTS.items():
        values = payload.get(category)
        if not isinstance(values, list) or len(values) != required_count:
            raise ValueError(f"{category} must contain exactly {required_count} task IDs")
        if any(not isinstance(task_id, str) or not task_id.strip() for task_id in values):
            raise ValueError(f"{category} contains an invalid task ID")
        task_ids.extend(values)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("representative fixture task IDs must be unique")
    return task_ids


def summarize_shadow_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_paths: Counter[str] = Counter()
    semantic_outcomes: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    disagreements: list[str] = []
    verifier_interventions: list[str] = []
    latencies: list[float] = []
    tokens_by_stage: dict[str, dict[str, int]] = {}
    comparable = 0
    agreed = 0
    numeric_comparable = 0
    numeric_equivalent = 0
    numeric_differences: list[str] = []
    judge_tasks = 0
    errors = 0

    for record in records:
        task_id = str(record.get("task_id", ""))
        if record.get("error"):
            errors += 1
        trace = record.get("semantic_trace")
        if not isinstance(trace, dict):
            selected_paths["missing_trace"] += 1
            semantic_outcomes["not_attempted"] += 1
            continue

        selected_paths[str(trace.get("selected_path") or "unknown")] += 1
        fallback_reason = trace.get("fallback_reason")
        semantic_answer = trace.get("semantic_candidate_answer")
        if semantic_answer is not None:
            semantic_outcomes["resolved"] += 1
            comparable += 1
            legacy_answer = str(record.get("agent_answer", ""))
            semantic_text = str(semantic_answer)
            if legacy_answer == semantic_text:
                agreed += 1
            else:
                disagreements.append(task_id)
            numeric_match = _numeric_equivalent(legacy_answer, semantic_text)
            if numeric_match is not None:
                numeric_comparable += 1
                if numeric_match:
                    numeric_equivalent += 1
                else:
                    numeric_differences.append(task_id)
        elif fallback_reason:
            semantic_outcomes["fallback"] += 1
            fallback_reasons[str(fallback_reason)] += 1
        else:
            semantic_outcomes["not_attempted"] += 1

        verifier = trace.get("verifier")
        if isinstance(verifier, dict):
            votes = verifier.get("judge_votes")
            if isinstance(votes, list) and votes:
                judge_tasks += 1
            if _verifier_intervened(trace, verifier):
                verifier_interventions.append(task_id)

        elapsed_ms = trace.get("elapsed_ms")
        if isinstance(elapsed_ms, (int, float)) and elapsed_ms > 0:
            latencies.append(float(elapsed_ms))
        _merge_usage(tokens_by_stage, trace.get("usage"))

    total = len(records)
    fallback_count = sum(fallback_reasons.values())
    return {
        "tasks": total,
        "errors": errors,
        "path_coverage": {
            "selected_paths": dict(sorted(selected_paths.items())),
            "semantic_outcomes": {
                key: semantic_outcomes.get(key, 0)
                for key in ("resolved", "fallback", "not_attempted")
            },
        },
        "agreement": {
            "comparable": comparable,
            "agreed": agreed,
            "disagreed": comparable - agreed,
            "rate": _rate(agreed, comparable),
            "disagreement_task_ids": sorted(disagreements),
        },
        "numeric_agreement": {
            "comparable": numeric_comparable,
            "equivalent": numeric_equivalent,
            "different": numeric_comparable - numeric_equivalent,
            "rate": _rate(numeric_equivalent, numeric_comparable),
            "difference_task_ids": sorted(numeric_differences),
        },
        "fallback": {
            "count": fallback_count,
            "rate": _rate(fallback_count, total),
            "reasons": dict(sorted(fallback_reasons.items())),
        },
        "judges": {
            "tasks": judge_tasks,
            "rate": _rate(judge_tasks, total),
        },
        "verifier_interventions": {
            "count": len(verifier_interventions),
            "rate": _rate(len(verifier_interventions), total),
            "task_ids": sorted(verifier_interventions),
        },
        "latency_ms": _latency_summary(latencies),
        "tokens_by_stage": dict(sorted(tokens_by_stage.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 30-task semantic shadow harness")
    parser.add_argument("--input", required=True, type=Path, help="DABStep task JSON")
    parser.add_argument("--data-dir", required=True, type=Path, help="DABStep context directory")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional external ID-only fixture; default is runtime stratified selection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/semantic_shadow_30.jsonl"),
        help="Per-task shadow records",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/semantic_shadow_30.report.json"),
        help="Aggregate report path",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("workspace/semantic-shadow-30"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "default",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.fixture is not None:
        task_ids = load_representative_ids(args.fixture)
    else:
        tasks = load_tasks(args.input)
        task_ids = flatten_selected_ids(select_representative_ids(tasks))
    asyncio.run(
        run_benchmark(
            input_path=args.input,
            data_dir=args.data_dir,
            output_path=args.output,
            workspace_dir=args.workspace_dir,
            assets_dir=args.assets_dir,
            task_ids=",".join(task_ids),
            memory_config=MemoryLakeConfig(),
            evaluation_policy=EvaluationPolicy.official(),
            concurrency=max(1, args.concurrency),
            resume=args.resume,
            semantic_mode=SemanticMode.SHADOW,
        )
    )
    records = _load_jsonl(args.output)
    report = summarize_shadow_records(records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _verifier_intervened(trace: dict[str, Any], verifier: dict[str, Any]) -> bool:
    candidates = trace.get("candidates")
    proposed_id = None
    if isinstance(candidates, dict):
        rows = candidates.get("candidates")
        if isinstance(rows, list):
            proposed_id = next(
                (
                    row.get("candidate_id")
                    for row in rows
                    if isinstance(row, dict) and row.get("origin") == "proposed"
                ),
                None,
            )
    selected_id = verifier.get("selected_candidate_id")
    rejected_ids = verifier.get("rejected_candidate_ids")
    return bool(
        verifier.get("accepted") is False
        or (proposed_id and selected_id and proposed_id != selected_id)
        or (isinstance(rejected_ids, list) and rejected_ids)
    )


def _merge_usage(target: dict[str, dict[str, int]], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    fields = ("calls", "input_tokens", "output_tokens", "latency_ms")
    for stage, values in usage.items():
        if not isinstance(values, dict):
            continue
        aggregate = target.setdefault(str(stage), {field: 0 for field in fields})
        for field in fields:
            value = values.get(field, 0)
            if isinstance(value, (int, float)):
                aggregate[field] += int(value)


def _latency_summary(values: list[float]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "average": 0.0, "p50": 0.0, "p95": 0.0, "maximum": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "average": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "maximum": round(ordered[-1], 3),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _numeric_equivalent(left: str, right: str) -> bool | None:
    try:
        return Decimal(left.strip()) == Decimal(right.strip())
    except (InvalidOperation, ValueError):
        return None


if __name__ == "__main__":
    main()
