"""Full self-reference generation: the verifier's model solves every planned
instance once, and the discrimination layer aggregates across instances.

Single-shot references are noisy (model accuracy ~0.85-0.9); correctness does
not come from per-instance voting but from cross-instance aggregation: the
right interpretation candidate agrees with a noisy reference on most of a
template's instances, rivals only by coincidence. Discrimination therefore
uses a relative gate (margin over the runner-up) for this reference source,
and adaptive escalation re-solves only the instances that separate the top
two candidates when a template stays ambiguous.

The reference file is a task-answer mapping: it lives in the learn working
directory only (persist path next to the artifacts, gitignored) and must
never be shipped, exported to MemoryLake, or read at runtime.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.distill.bootstrap import (
    _data_dir_summary,
    _majority_cluster,
    _representative,
    _solve_once,
    build_bootstrap_agent,
)
from dabstep_agent_pydantic.distill.discriminate import ReferenceRecord
from dabstep_agent_pydantic.usage_telemetry import UsageLedger


async def generate_references(
    *,
    instances: list[dict[str, Any]],
    data_dir,
    workspace_dir,
    samples: int = 1,
    concurrency: int = 12,
    agent=None,
    persist_path=None,
    tag_prefix: str = "ref",
    usage_ledger: UsageLedger | None = None,
) -> dict[str, ReferenceRecord]:
    """Solve each instance ``samples`` times; persist confirmed answers (JSONL).

    samples=1 records the single answer. samples>1 requires a majority
    cluster (> samples/2) and keeps its most precise member. Already-persisted
    task_ids are never re-solved, so interrupted runs resume for free and
    escalation rounds only pay for the new samples.
    """
    solver = agent or build_bootstrap_agent()
    file_summary = _data_dir_summary(data_dir)
    records: dict[str, ReferenceRecord] = {}
    if persist_path is not None and Path(persist_path).exists():
        for line in Path(persist_path).read_text().splitlines():
            row = json.loads(line)
            records[str(row["task_id"])] = ReferenceRecord(
                task_id=str(row["task_id"]), answer=str(row["answer"]),
                high_confidence=True, source="self_reference",
            )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    todo = [i for i in instances if str(i["task_id"]) not in records]
    usage_by_tid = {str(instance["task_id"]): UsageLedger() for instance in todo}
    learn_usage = usage_ledger

    async def solve_instance(instance: dict[str, Any], *, retries: int = 0) -> None:
        tid = str(instance["task_id"])
        async with semaphore:
            results = await asyncio.gather(*(
                _solve_once(solver, instance, data_dir, workspace_dir,
                            f"{tag_prefix}_{tid}_{i}", file_summary=file_summary,
                            stage="reference", retries=retries)
                for i in range(samples)
            ))
        for result in results:
            usage_by_tid[tid].record(result.usage)
            if learn_usage is not None:
                learn_usage.record(result.usage)
        answers = [result.answer for result in results if result.answer is not None]
        if not answers:
            return
        if samples == 1:
            answer = answers[0]
        else:
            cluster = _majority_cluster(answers)
            if len(cluster) * 2 <= samples:
                return
            answer = _representative(cluster)
        # Persist as soon as each instance resolves: an interrupted phase
        # resumes from what it already paid for.
        records[tid] = ReferenceRecord(
            task_id=tid, answer=answer, high_confidence=True, source="self_reference",
        )
        if persist_path is not None:
            with Path(persist_path).open("a") as fh:
                fh.write(json.dumps({
                    "task_id": tid,
                    "answer": answer,
                    "usage_trace": usage_by_tid[tid].summary(),
                }) + "\n")

    await asyncio.gather(*(solve_instance(i) for i in todo))
    # Self-healing sweep: instances lost to transient network trouble get one
    # automatic second pass (the gateway is usually healthy again by the time
    # the first pass finishes) instead of requiring a manual --resume restart.
    missing = [i for i in todo if str(i["task_id"]) not in records]
    if missing:
        await asyncio.gather(*(solve_instance(i, retries=1) for i in missing))
    return records
