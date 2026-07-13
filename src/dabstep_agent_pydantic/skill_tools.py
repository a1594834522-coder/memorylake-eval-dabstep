"""Learned skills exposed as solver tools: AI-judged applicability,
deterministic execution, and agent-initiated skill proposals.

The signature layer matches learned skills by canonical template text, which
is exact but brittle: a paraphrase of a learned template misses the
deterministic path entirely. These tools hand the applicability judgement to
the solver model — it recognizes that a task is semantically an instance of a
learned template and invokes the skill with structured parameters. Execution
stays inside the audited spec/mechanism layer; the model never influences the
computed value, only whether the skill is consulted.

Proposals close the agency loop in the other direction: the model can flag a
recurring, uncovered question shape as a skill candidate. A proposal is only
a queue entry for the offline certification pipeline (discrimination +
calibration audit via `dabstep-learn --from-proposals`); it never affects the
answer path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from dabstep_agent_pydantic.agent import DABStepDeps
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.distill.emit import adoption_summary, render_convention
from dabstep_agent_pydantic.distill.shadow import (
    _output_shape_violation,
    _skills,
    generated_skills_mode,
    stale_documents,
)
from dabstep_agent_pydantic.distill.templates import normalize_question
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy

PROPOSALS_FILENAME = "_proposals.jsonl"


class LearnedSkillInfo(BaseModel):
    skill_id: str = Field(description="Identifier to pass to apply_learned_skill.")
    template: str = Field(description="Canonical question template with <PLACEHOLDER> slots.")
    params: list[str] = Field(description="Parameter names apply_learned_skill expects.")
    interpretation: str = Field(description="Plain-language statement of the audited interpretation this skill computes.")
    adoption: str = Field(description="Evidence basis on which the interpretation was adopted.")


class LearnedSkillResult(BaseModel):
    skill_id: str
    answer: str | None = Field(description="Deterministic answer, or null when the skill could not run.")
    detail: str = Field(description="How the result was produced, or why execution failed.")


def list_learned_skills(ctx: RunContext[DABStepDeps]) -> list[LearnedSkillInfo]:
    """List learned deterministic skills: template, parameters, and the audited interpretation.

    Each skill encodes an interpretation that won statistical discrimination
    against rival readings and passed an independent calibration audit. If the
    current task is semantically an instance of one of these templates — even
    phrased differently — prefer apply_learned_skill over re-deriving the
    logic yourself."""
    return [
        LearnedSkillInfo(
            skill_id=s.skill_id,
            template=s.template,
            params=list(s.signature.group_params),
            interpretation=render_convention(s.spec),
            adoption=adoption_summary(s.evidence),
        )
        for s in _skills()
    ]


def apply_learned_skill(
    ctx: RunContext[DABStepDeps],
    skill_id: str,
    params: dict[str, str],
    guidelines: str = "",
) -> LearnedSkillResult:
    """Execute a learned skill with structured parameters extracted from the task.

    Supply each parameter value as it appears in the task (merchant name,
    month name, fee ID, ...); values are converted by the same typed parsers
    the canonical template uses. Execution is fully deterministic — the answer
    comes from the audited interpretation spec, not from generation. Pass the
    task's answer guidelines so output formatting follows them."""
    skill = next((s for s in _skills() if s.skill_id == skill_id), None)
    if skill is None:
        return LearnedSkillResult(skill_id=skill_id, answer=None,
                                  detail="unknown skill_id; call list_learned_skills first")
    stale = stale_documents(skill, ctx.deps.data_dir)
    if stale:
        return LearnedSkillResult(
            skill_id=skill_id, answer=None,
            detail=f"knowledge docs changed since this skill was learned ({', '.join(stale)}); "
                   "the skill is stale — solve without it",
        )
    data = load_dabstep_data(ctx.deps.data_dir)
    try:
        answer = skill.solve_with_params(data, params, guidelines)
    except ValueError as exc:
        return LearnedSkillResult(skill_id=skill_id, answer=None, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - a broken artifact must never break solving.
        return LearnedSkillResult(skill_id=skill_id, answer=None,
                                  detail=f"skill execution failed: {type(exc).__name__}")
    if answer is None:
        return LearnedSkillResult(
            skill_id=skill_id, answer=None,
            detail="the spec could not produce an answer for these parameters; "
                   "check parameter values or solve without the skill",
        )
    violation = _output_shape_violation(skill, answer)
    if violation is not None:
        return LearnedSkillResult(
            skill_id=skill_id, answer=None,
            detail=f"answer failed the output-shape invariant ({violation}); "
                   "solve without the skill",
        )
    return LearnedSkillResult(skill_id=skill_id, answer=answer,
                              detail="deterministic audited interpretation spec")


def propose_skill_candidate(
    ctx: RunContext[DABStepDeps],
    question: str,
    rationale: str,
) -> str:
    """Propose a recurring question shape as a candidate for a learned skill.

    Use this after solving a task manually when (1) no learned skill covered
    it, and (2) the question is clearly a parameterized instance of a
    recurring template. Pass the task question verbatim plus a one-sentence
    rationale naming the interpretation choice you had to make. The proposal
    only queues the template for the offline certification pipeline
    (discrimination against references + calibration audit); it changes
    nothing about the current task."""
    policy = getattr(ctx.deps, "evaluation_policy", EvaluationPolicy.development())
    if not policy.local_proposal_writes:
        return "skill proposals are disabled by the evaluation policy"

    directory = Path(os.getenv("DABSTEP_GENERATED_SKILLS_DIR", "artifacts/skills").strip()
                     or "artifacts/skills")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROPOSALS_FILENAME
    normalized = normalize_question(" ".join(str(question).split()))
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if json.loads(line).get("template") == normalized:
                return "already queued: an equivalent template proposal exists"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "template": normalized,
            "question": str(question),
            "rationale": str(rationale),
        }, ensure_ascii=False) + "\n")
    return "queued for the offline certification pipeline (dabstep-learn --from-proposals)"


def learned_skill_tools_enabled() -> bool:
    return generated_skills_mode() != "off" and bool(_skills())
