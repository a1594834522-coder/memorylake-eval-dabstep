"""Teacher-proposed interpretation hypotheses + mechanical grid completion.

The only model-calling stage of the learn pipeline. The teacher reads a
template, a few instance questions, and relevant manual excerpts, and returns
candidate `InterpretationSpec` objects (structured output — never code).
A deterministic grid pass then completes combinations the teacher did not
propose, so candidate coverage does not depend on teacher recall. Candidates
without a manual citation are dropped.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from dabstep_agent_pydantic.agent import build_teacher_model_from_env
from dabstep_agent_pydantic.distill.spec import (
    FeeRulesSpec,
    InterpretationSpec,
    OutputSpec,
    PaymentsSpec,
)

MAX_CANDIDATES_PER_TEMPLATE = 8

# A half-closed gateway connection otherwise hangs the whole learn run
# (observed: 22 min stalled on a CLOSE_WAIT socket with no timeout).
TEACHER_TIMEOUT_SECONDS = 300.0
TEACHER_ATTEMPTS = 2

TEACHER_INSTRUCTIONS = """\
You analyze one question template from a payments-analytics benchmark and
propose every plausible interpretation of its metric as a structured spec.

Rules:
- Propose interpretations ONLY as InterpretationSpec objects in the given
  schema. Never write code and never answer the questions themselves.
- Every spec MUST cite a phrase from the provided documentation excerpts in
  manual_citation. If the documentation does not support an interpretation
  but it is still a plausible reading, set contradicts_manual=true and cite
  the closest passage it deviates from.
- Cover rival readings deliberately: aggregation basis (mean vs sum vs
  min/max), whether unspecified dimensions are filtered, wildcard vs strict
  rule matching, inclusion/exclusion of current members in steering
  questions.
- Prefer the fee_rules family for hypothetical-transaction questions and the
  payments family primitives for merchant/period questions.

Example of a well-formed candidate (fee_rules family, mean reading):
{"name": "unfiltered_mean", "population": "fee_rules",
 "fee_rules": {"context_dims": ["card_scheme", "is_credit"], "value": "fee_at_amount",
               "reducer": "mean"},
 "output": {"kind": "decimal", "decimals_default": 6},
 "manual_citation": "fee = fixed_amount + rate * transaction_value / 10000"}

The instance GUIDELINES dictate the output contract - set output.kind,
decimals and tie policy from what the guidelines ask for, not from guesswork.

Payments-family primitives and when they apply:
- period_total_fees: total fees a merchant paid over a month/year/day.
- period_fee_rate_delta: "what delta would X pay if the relative fee of the
  fee with ID=N changed to V" — the rival delta_basis values are "rate"
  (change the rule's rate to V) and "fixed_component" (zero the fixed fee).
- mcc_change_fee_delta: merchant changed MCC before the year started.
- affected_merchants: which merchants are affected by a fee rule or its
  restriction to one account type.
- applicable_fee_ids_period: fee IDs applicable to a merchant in a period.
- steer_optimal_aci: move fraudulent transactions to the lowest-fee ACI.
- field_domain_values / unique_merchant_count / repeat_customer_percentage.
"""


class TeacherProposal(BaseModel):
    candidates: list[InterpretationSpec]


def build_teacher_agent() -> Agent:
    return Agent(
        build_teacher_model_from_env(),
        output_type=TeacherProposal,
        instructions=TEACHER_INSTRUCTIONS,
    )


def build_teacher_prompt(*, template: str, instance_questions: list[str], manual_excerpts: list[str],
                         guidelines: str = "") -> str:
    instances = "\n".join(f"- {q}" for q in instance_questions[:3])
    excerpts = "\n---\n".join(manual_excerpts[:8])
    guide = f"\nINSTANCE GUIDELINES (dictate the output contract):\n{guidelines}\n" if guidelines else ""
    return f"""\
TEMPLATE:
{template}

INSTANCE QUESTIONS:
{instances}
{guide}
DOCUMENTATION EXCERPTS:
{excerpts}

Propose the candidate interpretations for this template.
"""


def _pinned_core_excerpts(docs_dir: Path) -> list[str]:
    """The fee formula and wildcard-semantics paragraphs are load-bearing for
    almost every template but rarely share the template's surface keywords,
    so keyword retrieval misses them - pin them unconditionally. Verbatim
    public-doc content only: injecting distilled conventions here would let
    the reference pre-agree with the candidates (circular adjudication)."""
    pins: list[str] = []
    patterns = (re.compile(r"fee\s*=|/\s*10000|transaction_value", re.I),
                re.compile(r"null|empty", re.I))
    for doc in sorted(docs_dir.glob("*.md")):
        for paragraph in doc.read_text(encoding="utf-8").split("\n\n"):
            text = paragraph.strip()
            if len(text) < 40 or len(pins) >= 3:
                continue
            if any(pat.search(text) for pat in patterns) and \
                    ("fee" in text.lower() or "wildcard" in text.lower() or "applies" in text.lower()):
                pins.append(text[:800])
    return pins


def manual_excerpts_for_template(template: str, docs_dir: Path, *, max_excerpts: int = 6) -> list[str]:
    """Pinned core semantics + keyword-scored paragraph retrieval (local, offline)."""
    keywords = {w.lower() for w in re.findall(r"[a-zA-Z_]{4,}", template)} - {
        "what", "would", "which", "that", "with", "this", "were", "have", "been",
    }
    scored: list[tuple[int, str]] = []
    for doc in sorted(docs_dir.glob("*.md")):
        for paragraph in doc.read_text(encoding="utf-8").split("\n\n"):
            text = paragraph.strip()
            if len(text) < 60:
                continue
            score = sum(1 for word in keywords if word in text.lower())
            if score >= 2:
                scored.append((score, text[:800]))
    scored.sort(key=lambda item: -item[0])
    pinned = _pinned_core_excerpts(docs_dir)
    keyword_hits = [text for _, text in scored if text not in pinned]
    return pinned + keyword_hits[: max(0, max_excerpts - len(pinned))]


async def propose_candidates(
    *, template: str, instance_questions: list[str], docs_dir: Path,
    agent: Agent | None = None, guidelines: str = "",
) -> list[InterpretationSpec]:
    teacher = agent or build_teacher_agent()
    prompt = build_teacher_prompt(
        template=template,
        instance_questions=instance_questions,
        manual_excerpts=manual_excerpts_for_template(template, docs_dir),
        guidelines=guidelines,
    )
    result = None
    for attempt in range(TEACHER_ATTEMPTS):
        # wait_for awaits the cancellation of the timed-out task; a task stuck
        # on a poisoned pooled connection never completes cancellation and
        # hangs the whole learn run (same failure fixed in bootstrap._solve_once,
        # aaa6f3c). Abandon instead: one timed-out attempt, never the pipeline.
        task = asyncio.ensure_future(teacher.run(prompt))
        done, _pending = await asyncio.wait({task}, timeout=TEACHER_TIMEOUT_SECONDS)
        if done:
            result = task.result()
            break
        task.cancel()
        if attempt == TEACHER_ATTEMPTS - 1:
            raise asyncio.TimeoutError(f"teacher call timed out after {TEACHER_ATTEMPTS} attempts")
    assert result is not None
    proposed = [c for c in result.output.candidates if c.manual_citation.strip()]
    return complete_grid(proposed)[:MAX_CANDIDATES_PER_TEMPLATE]


def seed_grid(template: str, signature) -> list[InterpretationSpec]:
    """Zero-model candidate seeding from template keywords + signature parameters.

    The standard interpretation grid covers the rival readings for every known
    question family; the teacher is consulted only when no seeded candidate
    survives coverage discipline ("expert consultation" mode).
    """
    text = template.lower()
    params = set(signature.group_params) | {name for name, _ in signature.constant_params}
    dims = [d for d in ("card_scheme", "is_credit", "account_type", "aci", "merchant_category_code") if d in params]
    cite = "grid seed - manual fee-rule semantics (null/[] wildcard, fee = fixed + rate*value/10000)"
    out: list[InterpretationSpec] = []

    def fee_rules(name, **kw):
        out.append(InterpretationSpec(
            name=name, population="fee_rules", fee_rules=FeeRulesSpec(**kw),
            output=OutputSpec(kind=kw.pop("_out_kind", "decimal") if "_out_kind" in kw else "decimal",
                              decimals_default=6),
            manual_citation=cite))

    def payments(name, out_kind, **kw):
        decimals = kw.pop("_decimals", None)
        out.append(InterpretationSpec(
            name=name, population="payments", payments=PaymentsSpec(**kw),
            output=OutputSpec(kind=out_kind, decimals_default=decimals),
            manual_citation=cite))

    if "average fee" in text and "card scheme" in text and "amount" in params:
        for reducer in ("mean", "sum", "min"):
            fee_rules(f"seed_avg_{reducer}", context_dims=dims, reducer=reducer)
        out.append(InterpretationSpec(
            name="seed_avg_strict", population="fee_rules",
            fee_rules=FeeRulesSpec(context_dims=dims, reducer="mean", wildcard_policy="strict"),
            output=OutputSpec(kind="decimal", decimals_default=6),
            manual_citation=cite, contradicts_manual=True))
    elif "most expensive mcc" in text:
        for reducer in ("mean", "max", "sum"):
            out.append(InterpretationSpec(
                name=f"seed_mcc_{reducer}", population="fee_rules",
                fee_rules=FeeRulesSpec(context_dims=[], reducer=reducer, value="fee_at_amount",
                                       group_by="merchant_category_code", group_extreme="argmax"),
                output=OutputSpec(kind="single_string", tie_policy="list_all_sorted"),
                manual_citation=cite))
    elif "fee id" in text and "merchant" not in params:
        out.append(InterpretationSpec(
            name="seed_ids_wildcard", population="fee_rules",
            fee_rules=FeeRulesSpec(context_dims=dims, value="rule_id", reducer="collect_ids"),
            output=OutputSpec(kind="id_list"), manual_citation=cite))
        out.append(InterpretationSpec(
            name="seed_ids_strict", population="fee_rules",
            fee_rules=FeeRulesSpec(context_dims=dims, value="rule_id", reducer="collect_ids",
                                   wildcard_policy="strict"),
            output=OutputSpec(kind="id_list"), manual_citation=cite, contradicts_manual=True))
    elif ("fee id" in text or "applicable" in text) and "merchant" in params:
        for scope in ("full_period", "sampled_first_day"):
            payments(f"seed_applicable_{scope}", "id_list",
                     primitive="applicable_fee_ids_period", tuple_scope=scope)
    elif "changed its mcc" in text:
        payments("seed_mcc_delta", "decimal", primitive="mcc_change_fee_delta", _decimals=6)
    elif "delta" in text and "new_value" in params:
        for basis in ("rate", "fixed_component"):
            payments(f"seed_delta_{basis}", "decimal",
                     primitive="period_fee_rate_delta", delta_basis=basis, _decimals=14)
    elif "affected" in text and "fee_id" in params:
        modes = ("losers_only", "symmetric_difference") if "account_type" in params else ("baseline_members",)
        for mode in modes:
            payments(f"seed_affected_{mode}", "string_list",
                     primitive="affected_merchants", affected_mode=mode)
    elif "fraudulent transactions" in text and ("move" in text or "lowest" in text):
        for policy in ("exclude_current", "include_all"):
            for reducer in ("sum_all_matching", "min_match"):
                payments(f"seed_steer_{policy}_{reducer}", "single_string",
                         primitive="steer_optimal_aci", aci_candidate_policy=policy, reducer=reducer)
    elif "total fees" in text and "merchant" in params:
        for reducer in ("sum_all_matching", "min_match", "first_match"):
            payments(f"seed_total_{reducer}", "decimal",
                     primitive="period_total_fees", reducer=reducer, _decimals=2)
    elif "possible values" in text:
        payments("seed_domain", "string_list", primitive="field_domain_values")
    elif "unique merchants" in text:
        payments("seed_unique", "integer", primitive="unique_merchant_count")
    elif "percentile" in params or "percentile" in text:
        payments("seed_repeat", "decimal", primitive="repeat_customer_percentage", _decimals=3)
    return out


def complete_grid(candidates: list[InterpretationSpec]) -> list[InterpretationSpec]:
    """Mechanically add rival combinations the teacher did not propose."""
    completed: list[InterpretationSpec] = []
    seen: set[str] = set()

    def add(spec: InterpretationSpec) -> None:
        key = _axis_key(spec)
        if key not in seen:
            seen.add(key)
            completed.append(spec)

    for candidate in candidates:
        add(candidate)

    for candidate in candidates:
        if candidate.population == "fee_rules" and candidate.fee_rules is not None:
            fr = candidate.fee_rules
            for reducer in ("mean", "sum", "min"):
                if fr.reducer == "collect_ids":
                    break
                add(_variant(candidate, fee_rules=fr.model_copy(update={"reducer": reducer}),
                             suffix=f"grid_{reducer}"))
            add(_variant(candidate, fee_rules=fr.model_copy(update={"wildcard_policy": "strict"}),
                         suffix="grid_strict", contradicts_manual=True))
        elif candidate.payments is not None:
            p = candidate.payments
            if p.primitive == "period_total_fees":
                for reducer in ("sum_all_matching", "min_match", "first_match"):
                    add(_variant(candidate, payments=p.model_copy(update={"reducer": reducer}),
                                 suffix=f"grid_{reducer}"))
            if p.primitive == "affected_merchants":
                for mode in ("losers_only", "symmetric_difference"):
                    add(_variant(candidate, payments=p.model_copy(update={"affected_mode": mode}),
                                 suffix=f"grid_{mode}"))
            if p.primitive == "steer_optimal_aci":
                for policy in ("exclude_current", "include_all"):
                    add(_variant(candidate, payments=p.model_copy(update={"aci_candidate_policy": policy}),
                                 suffix=f"grid_{policy}"))
            if p.primitive == "period_fee_rate_delta":
                for basis in ("rate", "fixed_component"):
                    add(_variant(candidate, payments=p.model_copy(update={"delta_basis": basis}),
                                 suffix=f"grid_{basis}"))
            if p.primitive == "applicable_fee_ids_period":
                for scope in ("full_period", "sampled_first_day"):
                    add(_variant(candidate, payments=p.model_copy(update={"tuple_scope": scope}),
                                 suffix=f"grid_{scope}"))
    return completed


def _variant(base: InterpretationSpec, *, suffix: str,
             fee_rules: FeeRulesSpec | None = None,
             payments: PaymentsSpec | None = None,
             contradicts_manual: bool | None = None) -> InterpretationSpec:
    return base.model_copy(update={
        "name": f"{base.name}__{suffix}",
        "fee_rules": fee_rules if fee_rules is not None else base.fee_rules,
        "payments": payments if payments is not None else base.payments,
        "contradicts_manual": base.contradicts_manual if contradicts_manual is None else contradicts_manual,
    })


def _axis_key(spec: InterpretationSpec) -> str:
    return f"{spec.population}|{spec.axis_summary}"
