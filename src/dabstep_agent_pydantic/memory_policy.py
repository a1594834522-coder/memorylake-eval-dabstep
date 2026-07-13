from __future__ import annotations

from pydantic import BaseModel

from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import RunMode


class MemoryCandidate(BaseModel):
    category: str
    content: str


def filter_memory_candidates(
    candidates: list[MemoryCandidate],
    config: MemoryLakeConfig,
    *,
    task_id: str,
    answer: str,
) -> tuple[list[MemoryCandidate], list[dict[str, object]]]:
    decisions: list[dict[str, object]] = []
    allowed: list[MemoryCandidate] = []
    for candidate in candidates:
        decision = _decide(candidate, config, task_id=task_id, answer=answer)
        decisions.append(decision)
        if decision["allowed"]:
            allowed.append(candidate)
    return allowed, decisions


def _decide(
    candidate: MemoryCandidate,
    config: MemoryLakeConfig,
    *,
    task_id: str,
    answer: str,
) -> dict[str, object]:
    if config.run_mode is RunMode.CLEAN or not config.memory_enabled:
        return {
            "action": "write_memory",
            "category": candidate.category,
            "allowed": False,
            "reason": "memory disabled",
        }
    text = candidate.content.lower()
    if f"task {task_id}" in text:
        return {
            "action": "write_memory",
            "category": candidate.category,
            "allowed": False,
            "reason": "task-specific memory blocked",
        }
    if answer and answer.lower() in text and len(answer.strip()) <= 20:
        return {
            "action": "write_memory",
            "category": candidate.category,
            "allowed": False,
            "reason": "answer leakage blocked",
        }
    return {
        "action": "write_memory",
        "category": candidate.category,
        "allowed": True,
        "reason": "policy allows reusable memory",
    }
