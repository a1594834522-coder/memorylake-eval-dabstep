"""Post-adoption hardening: invariants and distinguishability evidence.

Positioning is deliberate: these checks establish that the adopted spec's
*implementation* behaves structurally (monotonicity) and that the adoption
decision had discriminative power (the winner disagrees with each rejected
candidate on at least one participating instance). Interpretation correctness
itself comes from discrimination, not from these checks.
"""

from __future__ import annotations

from typing import Any

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.combinators import SpecNotExecutable, compile_spec
from dabstep_agent_pydantic.distill.discriminate import DiscriminationReport
from dabstep_agent_pydantic.distill.signatures import TemplateSignature
from dabstep_agent_pydantic.distill.spec import InterpretationSpec


def harden(
    *,
    data: DABStepData,
    report: DiscriminationReport,
    signature: TemplateSignature,
    sample_instance: dict[str, Any],
) -> dict[str, Any]:
    adopted = next(c for c in report.candidates if c.spec.name == report.adopted)
    checks: dict[str, Any] = {}

    checks["amount_monotonicity"] = _check_amount_monotonicity(
        data, adopted.spec, signature, sample_instance
    )

    distinguishable: dict[str, bool] = {}
    for candidate in report.candidates:
        if candidate.spec.name == adopted.spec.name:
            continue
        shared = set(adopted.instance_matches) & set(candidate.instance_matches)
        distinguishable[candidate.spec.name] = any(
            adopted.instance_matches[tid] != candidate.instance_matches[tid] for tid in shared
        )
    checks["distinguishable_from_rejected"] = distinguishable

    checks["regression_instances"] = [
        {"task_id": tid, "matched": matched}
        for tid, matched in sorted(adopted.instance_matches.items(), key=lambda kv: kv[0])
    ]
    return checks


def _check_amount_monotonicity(
    data: DABStepData,
    spec: InterpretationSpec,
    signature: TemplateSignature,
    sample_instance: dict[str, Any],
) -> bool | None:
    """fee = fixed + rate*amount/10000 is non-decreasing in amount for mean/sum/min/max."""
    if spec.population != "fee_rules" or spec.fee_rules is None:
        return None
    if spec.fee_rules.value != "fee_at_amount" or spec.fee_rules.group_by is not None:
        return None
    params = signature.parse(str(sample_instance["question"]), data)
    if params is None or params.get("amount") is None:
        return None
    fn = compile_spec(spec)
    try:
        low = float(fn(data, {**params, "amount": float(params["amount"])}, ""))
        high = float(fn(data, {**params, "amount": float(params["amount"]) * 2 + 1}, ""))
    except (SpecNotExecutable, ValueError):
        return None
    return high >= low
