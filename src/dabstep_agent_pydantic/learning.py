from __future__ import annotations

from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.memory_policy import MemoryCandidate


def extract_memory_candidates(record: dict[str, object], *, run_mode: RunMode) -> list[MemoryCandidate]:
    del run_mode
    candidates: list[MemoryCandidate] = []
    generated_code = str(record.get("generated_code", "") or "")
    if record.get("used_code") and ("value_counts" in generated_code or "groupby" in generated_code):
        candidates.append(
            MemoryCandidate(
                category="calculation_recipe",
                content=f"Reusable DABStep calculation recipe: {_safe_code_summary(generated_code)}",
            )
        )
    return candidates


def _safe_code_summary(code: str) -> str:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    selected = [line for line in lines if "value_counts" in line or "groupby" in line]
    return selected[0][:500] if selected else "Use deterministic pandas aggregation and verify the result."
