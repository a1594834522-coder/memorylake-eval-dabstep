"""Discriminate interpretation candidates against model-generated references.

Every candidate is a compiled spec executed on real data with parameters
parsed from each instance's question text. High-confidence filtering: an
instance participates when its reference is resolved or its top candidate
vote ratio is >= ``min_vote_ratio``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.ablation import answers_match


def reference_match(answer: str, reference: str) -> bool:
    """Scorer-aligned comparison: numeric tolerance scales with the
    reference's printed precision (a 2-decimal reference matches any value
    within 0.005), otherwise falls back to strict matching."""
    if answers_match(answer, reference):
        return True
    try:
        a = float(str(answer).replace(",", ""))
        r = float(str(reference).replace(",", ""))
    except ValueError:
        return False
    text = str(reference).strip()
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(a - r) <= 0.5 * 10 ** -decimals
from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.combinators import SpecNotExecutable, compile_spec
from dabstep_agent_pydantic.distill.signatures import TemplateSignature
from dabstep_agent_pydantic.distill.spec import InterpretationSpec


@dataclass(frozen=True)
class ReferenceRecord:
    task_id: str
    answer: str
    high_confidence: bool
    source: str = "model_consensus"  # or "official_dev"


@dataclass
class CandidateResult:
    spec: InterpretationSpec
    agree: int = 0
    total: int = 0
    errors: int = 0
    instance_matches: dict[str, bool] = field(default_factory=dict)

    @property
    def rate(self) -> float | None:
        return self.agree / self.total if self.total else None


@dataclass
class DiscriminationReport:
    template: str
    funnel: dict[str, int] = field(default_factory=dict)
    candidates: list[CandidateResult] = field(default_factory=list)
    adopted: str | None = None
    adoption_basis: str | None = None
    official_dev: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "funnel": self.funnel,
            "candidates": [
                {
                    "name": c.spec.name,
                    "axis": c.spec.axis_summary,
                    "agree": c.agree,
                    "total": c.total,
                    "errors": c.errors,
                    "rate": round(c.rate, 4) if c.rate is not None else None,
                }
                for c in self.candidates
            ],
            "adopted": self.adopted,
            "adoption_basis": self.adoption_basis,
            "official_dev": self.official_dev,
        }


def load_reference(
    path: Path,
    *,
    min_vote_ratio: float = 0.70,
    official_dev_path: Path | None = None,
) -> dict[str, ReferenceRecord]:
    """Load a model-generated reference file: {primary, candidates, ambiguous, ...}.

    When ``official_dev_path`` points to the benchmark's published dev sample
    (tasks with official answers), those records override the model consensus:
    they are always high-confidence and tagged source="official_dev".
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    primary: dict[str, Any] = raw.get("primary") or {}
    votes: dict[str, Any] = raw.get("candidates") or {}
    ambiguous = {str(t) for t in _iter_ids(raw.get("ambiguous") or [])}
    records: dict[str, ReferenceRecord] = {}
    for task_id, answer in primary.items():
        tid = str(task_id)
        resolved = tid not in ambiguous
        high = resolved or _vote_ratio(votes.get(tid) or {}) >= min_vote_ratio
        records[tid] = ReferenceRecord(task_id=tid, answer=str(answer), high_confidence=high)
    if official_dev_path is not None:
        for row in json.loads(Path(official_dev_path).read_text(encoding="utf-8")):
            tid = str(row["task_id"])
            records[tid] = ReferenceRecord(
                task_id=tid, answer=str(row["answer"]), high_confidence=True, source="official_dev",
            )
    return records


def _iter_ids(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.keys())
    return list(value)


def _vote_ratio(votes: dict[str, Any]) -> float:
    counts = [int(v) for v in votes.values() if isinstance(v, (int, float))]
    total = sum(counts)
    return max(counts) / total if total else 0.0


def sample_for_discrimination(
    instances: list[dict[str, Any]],
    reference: dict[str, ReferenceRecord],
    *,
    max_instances: int | None = 12,
) -> list[dict[str, Any]]:
    """The instance subset discrimination actually evaluates.

    Instances with a high-confidence reference are the evidence already paid
    for (bootstrap labels, official dev): they always participate. The
    remaining slots spread across the instance list (parameter diversity)
    instead of taking a contiguous prefix. Exposed separately so reference
    generation (Phase R) can solve exactly this subset instead of every raw
    instance of a template.
    """
    if max_instances is None or len(instances) <= max_instances:
        return instances
    keyed = [i for i in instances
             if (r := reference.get(str(i["task_id"]))) is not None and r.high_confidence]
    keyed = keyed[:max_instances]
    keyed_tids = {str(i["task_id"]) for i in keyed}
    rest = [i for i in instances if str(i["task_id"]) not in keyed_tids]
    need = max_instances - len(keyed)
    step = len(rest) / need if need else 0
    return keyed + [rest[int(i * step)] for i in range(need)]


def discriminate_template(
    *,
    data: DABStepData,
    template: str,
    instances: list[dict[str, Any]],  # {task_id, question, guidelines}
    candidates: list[InterpretationSpec],
    signature: TemplateSignature,
    reference: dict[str, ReferenceRecord],
    max_instances: int | None = 12,
) -> DiscriminationReport:
    instances = sample_for_discrimination(instances, reference, max_instances=max_instances)
    report = DiscriminationReport(template=template)
    report.candidates = [CandidateResult(spec=spec) for spec in candidates]
    compiled = [compile_spec(spec) for spec in candidates]

    total = len(instances)
    high_confidence = 0
    participated = 0
    for instance in instances:
        tid = str(instance["task_id"])
        record = reference.get(tid)
        if record is None or not record.high_confidence:
            continue
        high_confidence += 1
        params = signature.parse(str(instance["question"]), data)
        if params is None:
            continue
        participated += 1
        guidelines = str(instance.get("guidelines") or "")
        for result, fn in zip(report.candidates, compiled):
            try:
                answer = fn(data, params, guidelines)
            except (SpecNotExecutable, KeyError, ValueError, TypeError):
                result.errors += 1
                continue
            result.total += 1
            matched = reference_match(answer, record.answer)
            result.instance_matches[tid] = matched
            if matched:
                result.agree += 1

    report.funnel = {
        "instances": total,
        "high_confidence": high_confidence,
        "participated": participated,
    }
    official_ids = {
        str(i["task_id"]) for i in instances
        if (r := reference.get(str(i["task_id"]))) is not None and r.source == "official_dev"
    }
    if official_ids:
        report.official_dev = {
            c.spec.name: {
                "agree": sum(1 for tid in official_ids if c.instance_matches.get(tid)),
                "total": sum(1 for tid in official_ids if tid in c.instance_matches),
            }
            for c in report.candidates
        }
    viable = [c for c in report.candidates if c.rate is not None and not c.spec.contradicts_manual]
    if viable:
        best = max(viable, key=lambda c: (c.rate, c.agree))
        report.adopted = best.spec.name
        report.adoption_basis = (
            f"highest agreement {best.agree}/{best.total} among manual-consistent candidates"
        )
    return report
