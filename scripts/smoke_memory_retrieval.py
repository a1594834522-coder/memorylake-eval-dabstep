#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from dabstep_agent_pydantic.cli import default_assets_dir
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.dataset import load_tasks
from dabstep_agent_pydantic.memorylake import DEFAULT_MEMORYLAKE_BASE_URL
from dabstep_agent_pydantic.memorylake import MemoryLakeClient
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.runner import build_memory_context
from dabstep_agent_pydantic.runtime_assets import load_runtime_assets


def main() -> None:
    args = _parser().parse_args()
    load_dotenv()
    tasks = _load_sample_tasks(sample_path=args.sample, input_path=args.input, limit=args.limit)
    assets = load_runtime_assets(args.assets_dir)
    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        memory_write_enabled=False,
        project_id=args.project_id or os.getenv("MEMORYLAKE_PROJECT_ID"),
        user_id=args.user_id or os.getenv("MEMORYLAKE_USER_ID"),
        top_k=args.top_k,
        threshold=args.threshold,
        rerank=args.rerank,
    )
    api_key = os.getenv("MEMORYLAKE_API_KEY")
    if not api_key:
        raise RuntimeError("MEMORYLAKE_API_KEY is required")
    client = MemoryLakeClient(
        api_key=api_key,
        base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
    )

    latencies: list[float] = []
    curriculum_hits = 0
    for task in tasks:
        started = time.perf_counter()
        _context, trace = build_memory_context(
            question=task.question,
            guidelines=task.guidelines,
            file_summary="",
            config=config,
            memory_client=client,
            route_cards=assets.route_cards,
            asset_fingerprint=assets.asset_fingerprint,
        )
        latency = time.perf_counter() - started
        latencies.append(latency)
        document_names = [str(item.get("document_name", "")) for item in trace.retrieved_documents]
        curriculum_hit_count = sum(1 for name in document_names if name == "curriculum_rules.md")
        if curriculum_hit_count:
            curriculum_hits += 1
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "doc_hit_count": trace.document_retrieved_count,
                    "curriculum_rules_hits": curriculum_hit_count,
                    "document_scores": [
                        item.get("score")
                        for item in trace.retrieved_documents
                        if item.get("score") is not None
                    ],
                    "latency_seconds": round(latency, 3),
                    "search_queries": trace.search_queries,
                    "retrieval_failed": any(
                        decision.get("action") == "search_memory" and decision.get("allowed") is False
                        for decision in trace.policy_decisions
                    ),
                },
                ensure_ascii=False,
            )
        )

    p95 = _percentile(latencies, 95)
    summary = {
        "tasks": len(tasks),
        "curriculum_rules_hit_rate": round(curriculum_hits / len(tasks), 4) if tasks else 0.0,
        "latency_p95_seconds": round(p95, 3) if p95 is not None else None,
        "acceptance": {
            "curriculum_rules_hit_rate_min": 0.8,
            "latency_p95_seconds_max": 3.0,
        },
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test MemoryLake document retrieval without running the model")
    parser.add_argument("--sample", type=Path, default=_default_sample_path(), help="JSON sample of tasks or task IDs")
    parser.add_argument("--input", type=Path, default=None, help="Full DABStep tasks JSON, required when sample has IDs only")
    parser.add_argument("--limit", type=int, default=30, help="Maximum tasks to query")
    parser.add_argument("--assets-dir", type=Path, default=default_assets_dir(), help="Runtime route-card assets directory")
    parser.add_argument("--project-id", default=None, help="MemoryLake project ID; defaults to MEMORYLAKE_PROJECT_ID")
    parser.add_argument("--user-id", default=None, help="MemoryLake user ID; defaults to MEMORYLAKE_USER_ID")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--rerank", action="store_true")
    return parser


def _default_sample_path() -> Path | None:
    job_dir = os.getenv("CLAUDE_JOB_DIR")
    if not job_dir:
        return None
    candidate = Path(job_dir) / "tmp" / "ablation_sample.json"
    return candidate if candidate.exists() else None


def _load_sample_tasks(*, sample_path: Path | None, input_path: Path | None, limit: int) -> list[Task]:
    if sample_path and sample_path.exists():
        raw = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected sample JSON list: {sample_path}")
        if raw and isinstance(raw[0], dict) and "question" in raw[0]:
            return [Task.model_validate(item) for item in raw[:limit]]
        task_ids = [
            str(item.get("task_id") if isinstance(item, dict) else item)
            for item in raw
        ][:limit]
        if not input_path:
            raise ValueError("--input is required when --sample contains task IDs only")
        by_id = {task.task_id: task for task in load_tasks(input_path)}
        return [by_id[task_id] for task_id in task_ids if task_id in by_id]
    if not input_path:
        raise ValueError("Provide --sample with full task objects, or provide --input")
    return load_tasks(input_path)[:limit]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1]


if __name__ == "__main__":
    main()
