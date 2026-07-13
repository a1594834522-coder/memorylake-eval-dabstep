"""Self-bootstrapped reference labeling, disagreement-driven.

The verifier generates every reference answer with their own model; nothing is
shipped. Token spend is aligned with information gain:

- Step A (zero token): compute the full candidate x instance answer matrix
  locally — candidates are compiled specs, not model calls.
- Step B (zero token): instances where all candidates agree carry no
  discriminative information and are never labeled; unanimous templates are
  adoptable with zero labels.
- Step C (zero token): greedily pick the labeling set that separates the most
  candidate pairs per label, diversified over instances.
- Step D: sequentially label disagreement points with the verifier's model
  (N solves, majority cluster under precision-aware tolerance); stop as soon
  as one candidate survives. Samples that compute the same value at different
  printed precision (23.834676 vs 23.8347) must agree — adjudication only
  needs samples to support the same candidate, not to match byte-for-byte.

Bootstrap solving NEVER consults deterministic routes or generated skills —
that would be circular. It is a plain LLM + Python-workspace solve.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from dabstep_agent_pydantic.ablation import answers_match
from dabstep_agent_pydantic.agent import DABStepAnswer, DABStepDeps, create_agent
from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.combinators import SpecNotExecutable, compile_spec
from dabstep_agent_pydantic.distill.discriminate import ReferenceRecord, reference_match
from dabstep_agent_pydantic.distill.signatures import TemplateSignature
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.python_tool import PythonWorkspace
from dabstep_agent_pydantic.usage_telemetry import CallUsage, UsageLedger, call_usage_from_result


def answer_matrix(
    *,
    data: DABStepData,
    candidates: list[InterpretationSpec],
    signature: TemplateSignature,
    instances: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """candidate name -> {task_id -> answer}; zero model calls."""
    compiled = [(spec.name, compile_spec(spec)) for spec in candidates]
    matrix: dict[str, dict[str, str]] = {name: {} for name, _ in compiled}
    for instance in instances:
        tid = str(instance["task_id"])
        params = signature.parse(str(instance["question"]), data)
        if params is None:
            continue
        guidelines = str(instance.get("guidelines") or "")
        for name, fn in compiled:
            try:
                matrix[name][tid] = fn(data, params, guidelines)
            except (SpecNotExecutable, KeyError, ValueError, TypeError):
                continue
    return matrix


def disagreement_instances(matrix: dict[str, dict[str, str]]) -> dict[str, set[tuple[str, str]]]:
    """task_id -> set of candidate pairs that instance separates."""
    names = sorted(matrix)
    separations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    tids = set().union(*(set(v) for v in matrix.values())) if matrix else set()
    for tid in tids:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = matrix[a].get(tid), matrix[b].get(tid)
                if va is not None and vb is not None and not answers_match(va, vb):
                    separations[tid].add((a, b))
    return dict(separations)


def is_unanimous(matrix: dict[str, dict[str, str]]) -> bool:
    return not disagreement_instances(matrix)


def select_labeling_instances(
    matrix: dict[str, dict[str, str]],
    *,
    max_labels: int = 6,
) -> list[str]:
    """Greedy max pair-coverage selection over disagreement points."""
    separations = disagreement_instances(matrix)
    remaining: set[tuple[str, str]] = set().union(*separations.values()) if separations else set()
    chosen: list[str] = []
    while remaining and len(chosen) < max_labels:
        tid = max(separations, key=lambda t: (len(separations[t] & remaining), t))
        gain = separations[tid] & remaining
        if not gain:
            break
        chosen.append(tid)
        remaining -= gain
    # Add extra confirmations of already-covered pairs for statistical strength.
    for tid in sorted(separations, key=lambda t: -len(separations[t])):
        if len(chosen) >= max_labels:
            break
        if tid not in chosen:
            chosen.append(tid)
    return chosen


def lenient_match(left: str, right: str) -> bool:
    """Symmetric precision-aware comparison: tolerance from the coarser side."""
    return reference_match(left, right) or reference_match(right, left)


def _majority_cluster(answers: list[str]) -> list[str]:
    """Largest group of answers that all lenient-match a common seed."""
    best: list[str] = []
    for seed in answers:
        cluster = [a for a in answers if lenient_match(seed, a)]
        if len(cluster) > len(best):
            best = cluster
    return best


def _representative(cluster: list[str]) -> str:
    """Prefer the most precise form so downstream tolerance stays tight."""
    def decimals(text: str) -> int:
        text = text.strip()
        return len(text.split(".")[1]) if "." in text else 0
    return max(cluster, key=decimals)


BOOTSTRAP_INSTRUCTIONS = """\
You solve one payments-analytics question exactly. FIRST read the manual
sections relevant to the question - the manual's definitions (wildcard
semantics, the fee formula, field meanings) are the authoritative
interpretation; do not substitute common-sense readings for them. Use the
Python tool for all computation over the data files, and return only the
answer in the requested format. Do not guess: compute.
All data files live in the data directory given in the prompt; work only
with those files. Never scan the filesystem (no find/os.walk outside it).
The final answer must be plain text following the guidelines exactly:
no Python reprs (write A, not ['A']; write 1, 2, 3 for lists), and match
the requested decimal precision.
"""


def _data_dir_summary(data_dir) -> str:
    files = sorted(p.name for p in Path(data_dir).iterdir() if p.is_file())
    return f"Data directory: {data_dir}\nFiles: {', '.join(files)}"


def build_bootstrap_agent() -> Agent:
    from dabstep_agent_pydantic.agent import build_teacher_model_from_env
    from dabstep_agent_pydantic.toolsets import COMMON_TOOLSET

    return Agent(
        build_teacher_model_from_env(),
        output_type=DABStepAnswer,
        instructions=BOOTSTRAP_INSTRUCTIONS,
        toolsets=[COMMON_TOOLSET],
    )


# Hard questions legitimately need long multi-round computation; timing out
# at 5 minutes discards reasoning that was already paid for. Overridable for
# constrained environments.
SOLVE_TIMEOUT_SECONDS = float(os.getenv("DABSTEP_SOLVE_TIMEOUT_SECONDS", "600"))


@dataclass(frozen=True)
class _SolveOutcome:
    answer: str | None
    usage: CallUsage


async def _solve_once(solver: Agent, instance: dict[str, Any], data_dir, workspace_dir, tag: str,
                      file_summary: str | None = None, *, stage: str = "bootstrap",
                      retries: int = 0) -> _SolveOutcome:
    deps = DABStepDeps(
        data_dir=data_dir,
        workspace=PythonWorkspace(workspace_dir / tag),
        file_summary=file_summary or _data_dir_summary(data_dir),
    )
    model_start = time.perf_counter()
    try:
        task = asyncio.ensure_future(
            solver.run(
                # The static agent instructions cannot see deps, so the data
                # location must ride in the prompt: without it the model
                # scanned the whole filesystem for the data files and most
                # solves hit the timeout (observed 8/159 survivors).
                f"{deps.file_summary}\n\n"
                f"QUESTION: {instance['question']}\n\n"
                f"GUIDELINES: {instance.get('guidelines') or 'N/A'}",
                deps=deps,
                usage_limits=UsageLimits(request_limit=None),
            )
        )
        # Abandon on timeout instead of wait_for: awaiting cancellation of a
        # task stuck on a poisoned connection hung a whole learn run.
        done, _pending = await asyncio.wait({task}, timeout=SOLVE_TIMEOUT_SECONDS)
        if not done:
            task.cancel()
            print(f"solve {tag} failed: TimeoutError", file=sys.stderr)
            return _SolveOutcome(
                answer=None,
                usage=CallUsage(
                    stage=stage,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=round((time.perf_counter() - model_start) * 1000),
                    retries=retries,
                ),
            )
        result = task.result()
    except Exception as exc:  # noqa: BLE001 - a flaky solve drops one sample, not the run.
        print(f"solve {tag} failed: {type(exc).__name__}", file=sys.stderr)
        return _SolveOutcome(
            answer=None,
            usage=CallUsage(
                stage=stage,
                input_tokens=0,
                output_tokens=0,
                latency_ms=round((time.perf_counter() - model_start) * 1000),
                retries=retries,
            ),
        )
    return _SolveOutcome(
        answer=str(result.output.agent_answer),
        usage=call_usage_from_result(
            result,
            stage=stage,
            latency_ms=(time.perf_counter() - model_start) * 1000,
            retries=retries,
        ),
    )


async def bootstrap_labels(
    *,
    instances: list[dict[str, Any]],
    label_tids: list[str],
    data_dir,
    workspace_dir,
    samples: int = 5,
    consensus: int = 3,
    agent: Agent | None = None,
    matrix: dict[str, dict[str, str]] | None = None,
    survivors_needed: int = 1,
    persist_path=None,
    usage_ledger: UsageLedger | None = None,
) -> dict[str, ReferenceRecord]:
    """Label disagreement points sequentially (early stop); samples run concurrently.

    A label is confirmed when >= ``consensus`` of ``samples`` answers form one
    cluster under precision-aware tolerance (candidate-vote semantics: samples
    that would pick the same candidate agree even when printed precision
    differs). Each confirmed label is appended to ``persist_path`` (JSONL) so
    an interrupted bootstrap resumes without relabeling.
    """
    solver = agent or build_bootstrap_agent()
    file_summary = _data_dir_summary(data_dir)
    by_tid = {str(i["task_id"]): i for i in instances}
    labels: dict[str, ReferenceRecord] = {}
    if persist_path is not None and Path(persist_path).exists():
        for line in Path(persist_path).read_text().splitlines():
            row = json.loads(line)
            labels[str(row["task_id"])] = ReferenceRecord(
                task_id=str(row["task_id"]), answer=str(row["answer"]),
                high_confidence=True, source="self_bootstrap",
            )
    alive: set[str] | None = set(matrix) if matrix else None

    for tid in label_tids:
        if tid in labels:
            continue
        instance = by_tid.get(tid)
        if instance is None:
            continue
        results = await asyncio.gather(*(
            _solve_once(solver, instance, data_dir, workspace_dir, f"bootstrap_{tid}_{i}",
                        file_summary=file_summary, stage="bootstrap")
            for i in range(samples)
        ))
        batch_usage = UsageLedger()
        for result in results:
            batch_usage.record(result.usage)
            if usage_ledger is not None:
                usage_ledger.record(result.usage)
        answers = [result.answer for result in results if result.answer is not None]
        if not answers:
            continue
        cluster = _majority_cluster(answers)
        if len(cluster) < consensus and persist_path is not None:
            # Post-mortem material for consensus failures: raw sample answers
            # (working file next to the labels, never a shipped artifact).
            diag = Path(persist_path).with_name("_bootstrap_failed_batches.jsonl")
            with diag.open("a") as fh:
                fh.write(json.dumps({
                    "task_id": tid, "samples": len(results),
                    "answers": answers, "cluster_size": len(cluster),
                    "usage_trace": batch_usage.summary(),
                }) + "\n")
        if len(cluster) >= consensus:
            answer = _representative(cluster)
            labels[tid] = ReferenceRecord(
                task_id=tid, answer=answer, high_confidence=True, source="self_bootstrap",
            )
            if persist_path is not None:
                with Path(persist_path).open("a") as fh:
                    fh.write(json.dumps({
                        "task_id": tid,
                        "answer": answer,
                        "usage_trace": batch_usage.summary(),
                    }) + "\n")
            if alive is not None and matrix is not None:
                alive = {
                    name for name in alive
                    if (v := matrix[name].get(tid)) is not None and lenient_match(v, answer)
                }
                if len(alive) <= survivors_needed:
                    break
        # no majority cluster: instance dropped (no low-confidence labels)
    return labels
