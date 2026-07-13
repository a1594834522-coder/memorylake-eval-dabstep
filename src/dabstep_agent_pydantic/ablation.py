"""Ablation tooling: stratified task sampling and clean-vs-memory run comparison.

`sample` draws a family-stratified subset of task ids so a paired ablation run
(`--run-mode clean` vs `--run-mode memory-assisted`) stays affordable while
covering every task family. `compare` diffs two runtime JSONLs and reports
agreement, per-run behavior stats, and — when an external reference answers
file is supplied — accuracy. Reference answers are an input, never part of
this repository.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.cli import default_assets_dir
from dabstep_agent_pydantic.dataset import load_tasks
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.runner import allowed_curriculum_documents_for_family
from dabstep_agent_pydantic.runtime_assets import load_runtime_assets


def sample_task_ids(
    *,
    input_path: Path,
    per_family: int,
    seed: int,
    assets_dir: Path | None = None,
) -> dict[str, list[str]]:
    route_cards = load_runtime_assets(assets_dir or default_assets_dir()).route_cards
    by_family: dict[str, list[str]] = defaultdict(list)
    for task in load_tasks(input_path):
        plan = plan_task(question=task.question, guidelines=task.guidelines, route_cards=route_cards)
        by_family[plan.task_family].append(task.task_id)

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for family in sorted(by_family):
        task_ids = sorted(by_family[family])
        rng.shuffle(task_ids)
        selected[family] = sorted(task_ids[:per_family], key=_task_sort_key)
    return selected


def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (int(task_id), task_id) if task_id.isdigit() else (10**9, task_id)


def compare_runs(
    *,
    run_a: Path,
    run_b: Path,
    label_a: str = "run_a",
    label_b: str = "run_b",
    reference_path: Path | None = None,
) -> dict[str, Any]:
    records_a = _latest_success_records(run_a)
    records_b = _latest_success_records(run_b)
    shared = sorted(set(records_a) & set(records_b), key=_task_sort_key)
    reference = _load_reference(reference_path) if reference_path else None

    disagreements = []
    agree = 0
    for task_id in shared:
        answer_a = str(records_a[task_id].get("agent_answer", ""))
        answer_b = str(records_b[task_id].get("agent_answer", ""))
        if answers_match(answer_a, answer_b):
            agree += 1
            continue
        item: dict[str, Any] = {"task_id": task_id, label_a: answer_a, label_b: answer_b}
        if reference is not None and task_id in reference:
            item["reference_match"] = {
                label_a: answers_match(answer_a, reference[task_id]),
                label_b: answers_match(answer_b, reference[task_id]),
            }
        disagreements.append(item)

    report: dict[str, Any] = {
        "shared_tasks": len(shared),
        "only_in": {label_a: sorted(set(records_a) - set(records_b), key=_task_sort_key),
                    label_b: sorted(set(records_b) - set(records_a), key=_task_sort_key)},
        "agreement": agree,
        "agreement_rate": round(agree / len(shared), 4) if shared else None,
        "disagreements": disagreements,
        label_a: _run_stats(records_a, shared),
        label_b: _run_stats(records_b, shared),
    }
    if reference is not None:
        report["accuracy"] = {
            label_a: _accuracy(records_a, shared, reference),
            label_b: _accuracy(records_b, shared, reference),
        }
    return report


def answers_match(left: str, right: str) -> bool:
    left_text = left.strip()
    right_text = right.strip()
    if left_text.lower() == right_text.lower():
        return True
    try:
        left_value = float(left_text.replace(",", ""))
        right_value = float(right_text.replace(",", ""))
    except ValueError:
        return False
    return abs(left_value - right_value) <= 1e-6 * max(1.0, abs(left_value), abs(right_value))


def _accuracy(records: dict[str, dict], shared: list[str], reference: dict[str, str]) -> dict[str, Any]:
    scored = [task_id for task_id in shared if task_id in reference]
    correct = sum(
        1
        for task_id in scored
        if answers_match(str(records[task_id].get("agent_answer", "")), reference[task_id])
    )
    return {
        "scored_tasks": len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
    }


def _run_stats(records: dict[str, dict], shared: list[str]) -> dict[str, Any]:
    rows = [records[task_id] for task_id in shared]
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
    deterministic = sum(1 for row in rows if row.get("deterministic_route"))
    retries = sum(
        1
        for row in rows
        if int((row.get("workflow_trace") or {}).get("solver_attempts") or 1) > 1
    )
    retrieved = [
        int((row.get("memory_trace") or {}).get("retrieved_count") or 0)
        for row in rows
    ]
    documents = [
        int((row.get("memory_trace") or {}).get("document_retrieved_count") or 0)
        for row in rows
    ]
    document_hits = sum(1 for count in documents if count > 0)
    curriculum_hits = sum(1 for row in rows if _has_curriculum_rules_hit(row.get("memory_trace") or {}))
    family_aligned_hits = sum(1 for row in rows if _has_family_aligned_curriculum_hit(row.get("memory_trace") or {}))
    retrieval_failures = sum(1 for row in rows if _has_retrieval_failure(row.get("memory_trace") or {}))
    truncated = sum(1 for row in rows if bool((row.get("memory_trace") or {}).get("document_context_truncated")))
    chunks_dropped = [
        int((row.get("memory_trace") or {}).get("document_chunks_dropped") or 0)
        for row in rows
    ]
    return {
        "avg_elapsed_seconds": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
        "deterministic_route_count": deterministic,
        "verifier_retry_count": retries,
        "avg_memories_retrieved": round(sum(retrieved) / len(retrieved), 2) if retrieved else None,
        "avg_documents_retrieved": round(sum(documents) / len(documents), 2) if documents else None,
        "document_hit_rate": round(document_hits / len(rows), 4) if rows else None,
        "curriculum_rules_hit_rate": round(curriculum_hits / len(rows), 4) if rows else None,
        "family_aligned_hit_rate": round(family_aligned_hits / len(rows), 4) if rows else None,
        "retrieval_failure_count": retrieval_failures,
        "document_context_truncated_count": truncated,
        "avg_document_chunks_dropped": round(sum(chunks_dropped) / len(chunks_dropped), 2) if chunks_dropped else None,
    }


def _has_curriculum_rules_hit(memory_trace: dict[str, Any]) -> bool:
    return any(
        str(document.get("document_name") or document.get("document_id") or "") == "curriculum_rules.md"
        for document in memory_trace.get("retrieved_documents") or []
        if isinstance(document, dict)
    )


def _has_family_aligned_curriculum_hit(memory_trace: dict[str, Any]) -> bool:
    task_family = str((memory_trace.get("analysis_plan") or {}).get("task_family") or "")
    allowed_docs = allowed_curriculum_documents_for_family(task_family)
    if not allowed_docs:
        return False
    return any(
        str(document.get("document_name") or document.get("document_id") or "") in allowed_docs
        for document in memory_trace.get("retrieved_documents") or []
        if isinstance(document, dict)
    )


def _has_retrieval_failure(memory_trace: dict[str, Any]) -> bool:
    for decision in memory_trace.get("policy_decisions") or []:
        if not isinstance(decision, dict):
            continue
        if decision.get("action") != "search_memory" or decision.get("allowed") is not False:
            continue
        if str(decision.get("reason", "")).startswith("retrieval failed"):
            return True
    return False


def _latest_success_records(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record.get("task_id") or "")
        if not task_id or record.get("error") or "agent_answer" not in record:
            continue
        latest[task_id] = record
    return latest


def _load_reference(path: Path) -> dict[str, str]:
    reference: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row.get("task_id") or "")
        if task_id:
            reference[task_id] = str(row.get("answer", row.get("agent_answer", "")))
    return reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation sampling and run comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Draw a family-stratified task id sample")
    sample.add_argument("--input", required=True, type=Path, help="DABStep tasks JSON file")
    sample.add_argument("--per-family", type=int, default=20)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--assets-dir", type=Path, default=None)

    compare = subparsers.add_parser("compare", help="Compare two runtime JSONL runs")
    compare.add_argument("--run-a", required=True, type=Path)
    compare.add_argument("--run-b", required=True, type=Path)
    compare.add_argument("--label-a", default="clean")
    compare.add_argument("--label-b", default="memory_assisted")
    compare.add_argument(
        "--reference-answers",
        type=Path,
        default=None,
        help="Optional external JSONL of {task_id, answer} for offline accuracy monitoring; never committed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "sample":
        selected = sample_task_ids(
            input_path=args.input,
            per_family=args.per_family,
            seed=args.seed,
            assets_dir=args.assets_dir,
        )
        all_ids = sorted({task_id for ids in selected.values() for task_id in ids}, key=_task_sort_key)
        print(json.dumps({"families": selected, "task_ids": ",".join(all_ids)}, indent=2, ensure_ascii=False))
    else:
        report = compare_runs(
            run_a=args.run_a,
            run_b=args.run_b,
            label_a=args.label_a,
            label_b=args.label_b,
            reference_path=args.reference_answers,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
