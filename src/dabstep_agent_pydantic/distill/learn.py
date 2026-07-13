"""Learn pipeline orchestration: templates -> hypotheses -> discrimination -> skills.

    dabstep-agent learn \
        --tasks tasks.json --data-dir <context> --reference <model-generated refs> \
        --output artifacts/skills

Coverage discipline: a template yields a skill only when the adopted candidate's
agreement rate is >= --min-adoption-rate over >= --min-participation
high-confidence instances, and it wins over every rejected candidate with
discriminative evidence. Everything else stays on the LLM path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.local_paths import resolve_standard_data_path
from dabstep_agent_pydantic.distill.bootstrap import (
    answer_matrix,
    bootstrap_labels,
    disagreement_instances,
    is_unanimous,
    select_labeling_instances,
)
from dabstep_agent_pydantic.distill.discriminate import (
    discriminate_template,
    load_reference,
    sample_for_discrimination,
)
from dabstep_agent_pydantic.distill.emit import (
    template_note_id,
    template_skill_id,
    write_note_artifact,
    write_skill_artifact,
)
from dabstep_agent_pydantic.distill.harden import harden
from dabstep_agent_pydantic.distill.hypotheses import propose_candidates, seed_grid
from dabstep_agent_pydantic.distill.memo import learn_memo
from dabstep_agent_pydantic.distill.matrix_cache import MatrixCache
from dabstep_agent_pydantic.distill.matrix_cache import MatrixCacheKey
from dabstep_agent_pydantic.distill.matrix_cache import fingerprint_files
from dabstep_agent_pydantic.distill.reference_gen import generate_references
from dabstep_agent_pydantic.distill.signatures import SignatureError, compile_signature
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.distill.stats import passes_binomial_gate
from dabstep_agent_pydantic.distill.templates import group_templates
from dabstep_agent_pydantic.usage_telemetry import UsageLedger

# Bounds the escalation disagreement scan: payments-family specs recompute an
# expensive merchant-month slice per call, memoized only under learn_memo();
# scanning every raw instance unmemoized pegged a core for 10+ minutes on
# templates with many instances and rare candidate splits.
ESCALATE_SCAN_CAP = 30

# Family evidence transfer: interpretation generalizes across parameterized
# variants of one metric (the project's core thesis), so a candidate that was
# decisively adopted on a sibling template is strong prior evidence for the
# same candidate elsewhere. Transfer requires the sibling adoption to be clean
# (rate >= SIBLING_RATE at full participation) and the local evidence to point
# the same way (the identical candidate leads with rate >= LOCAL_FLOOR).
FAMILY_TRANSFER_SIBLING_RATE = 0.9
FAMILY_TRANSFER_LOCAL_FLOOR = 0.5


async def learn(
    *,
    tasks_path: Path,
    data_dir: Path,
    output_dir: Path,
    reference_path: Path | None = None,
    official_dev_path: Path | None = None,
    min_instances: int = 5,
    min_participation: int = 4,
    min_adoption_rate: float = 0.90,
    max_templates: int | None = None,
    proposals_path: Path | None = None,
    teacher_agent=None,
    resume: bool = False,
    concurrency: int = 4,
    bootstrap_samples: int = 5,
    bootstrap_consensus: int = 3,
    reference_mode: str = "full",
    reference_samples: int = 1,
    reference_concurrency: int = 12,
    adoption_floor: float = 0.60,
    adoption_margin: float = 0.25,
    adoption_null_rate: float = 0.5,
    adoption_alpha: float = 0.05,
    escalation_rounds: int = 1,
    solver_agent=None,
) -> dict[str, Any]:
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    data = load_dabstep_data(data_dir)
    if reference_path is not None:
        reference_mode = "file"
    if reference_mode not in ("full", "budget", "file"):
        raise ValueError(f"unknown reference mode: {reference_mode!r}")
    reference = (
        load_reference(reference_path, official_dev_path=official_dev_path)
        if reference_path is not None
        else load_reference_official_only(official_dev_path)
    )
    self_bootstrap = reference_mode == "budget"
    # Fingerprint the knowledge documents this learn run reads: skills record
    # the exact doc versions they were learned from, and `freeze` cross-checks
    # these against what it uploads — learn stays offline-capable while
    # knowledge-version consistency is mechanically enforced.
    doc_fingerprints = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(data_dir).glob("*.md"))
    }
    data_fingerprint = fingerprint_files(
        Path(data_dir),
        (
            path
            for path in Path(data_dir).rglob("*")
            if path.is_file() and path.suffix.lower() != ".md"
        ),
    )
    # Single-shot self references are individually noisy; adoption switches to
    # a relative gate (margin over the runner-up) and adaptive escalation.
    noisy_reference = reference_mode == "full"
    templates = group_templates(tasks)

    summary: dict[str, Any] = {
        "skills": [],
        "skipped": [],
        "matrix_cache": {"hits": 0, "misses": 0},
    }
    usage_ledger = UsageLedger()
    matrix_cache = MatrixCache(output_dir / "_matrix_cache")

    def load_candidate_matrix(template, candidates, signature, instances):
        key = MatrixCacheKey.from_inputs(
            template=template,
            candidates=candidates,
            instances=instances,
            data_hash=data_fingerprint,
            document_fingerprints=doc_fingerprints,
        )
        cached = matrix_cache.get(key)
        if cached is not None and isinstance(cached.get("matrix"), dict):
            summary["matrix_cache"]["hits"] += 1
            return key, cached["matrix"], cached
        summary["matrix_cache"]["misses"] += 1
        matrix = answer_matrix(
            data=data,
            candidates=candidates,
            signature=signature,
            instances=instances,
        )
        return key, matrix, {}

    if proposals_path is not None:
        # Agent-initiated learning: solver-filed proposals select which
        # templates to certify. The proposal grants no trust — every selected
        # template goes through the same discrimination gate and audit.
        proposed = select_proposed_templates(templates, proposals_path)
        summary["proposals"] = {
            "requested": len(proposed["requested"]),
            "matched": len(proposed["templates"]),
            "unmatched": proposed["unmatched"],
        }
        templates = proposed["templates"]
    # Cheap sequential pre-pass keeps --max-templates semantics; the expensive
    # per-template work below (labeling solves, teacher calls) is network-bound
    # and runs concurrently.
    planned: list[tuple[str, list[dict[str, Any]], Any]] = []
    resumed_adoptions: list[tuple[Any, float | None, dict[str, Any]]] = []
    for template, instances in templates.items():
        if max_templates is not None and len(planned) >= max_templates:
            break
        if len(instances) < min_instances:
            summary["skipped"].append({"template": template[:80], "reason": "too few instances"})
            continue
        if resume and (output_dir / f"{template_skill_id(template)}.json").exists():
            summary["skills"].append({
                "skill_id": template_skill_id(template),
                "template": template[:80],
                "adopted": "(resumed from existing artifact)",
                "rate": None,
                "teacher_consulted": False,
            })
            # Resumed artifacts still contribute sibling evidence for the
            # family-transfer post-pass.
            try:
                artifact = json.loads((output_dir / f"{template_skill_id(template)}.json").read_text())
                spec = InterpretationSpec.model_validate(artifact["spec"])
                disc = artifact.get("evidence", {}).get("discrimination", {})
                adopted_c = next((c for c in disc.get("candidates", [])
                                  if c.get("name") == disc.get("adopted")), {})
                rate = adopted_c.get("rate")
                funnel = disc.get("funnel", {})
                funnel = {**funnel, "_agree": adopted_c.get("agree", 0),
                          "_total": adopted_c.get("total", 0)}
                resumed_adoptions.append((spec, rate, funnel))
            except Exception:  # noqa: BLE001 - a malformed artifact must not kill planning.
                pass
            continue
        try:
            signature = compile_signature(template)
        except SignatureError as exc:
            summary["skipped"].append({"template": template[:80], "reason": f"unsignable: {exc}"})
            continue
        planned.append((template, instances, signature))

    # Per-template reference targeting, computed from the zero-token answer
    # matrix BEFORE any model call: instances where every candidate agrees
    # carry no discriminative information (a reference there cannot change the
    # ranking), and a template whose candidates agree everywhere needs no
    # references at all — the candidates are extensionally equal on the
    # sample, so any of them is adoptable ("unanimous"). References are bought
    # only on disagreement instances; single-shot noise still cancels through
    # cross-instance aggregation at discrimination time.
    targeting: dict[str, dict[str, Any]] = {}
    if noisy_reference and planned:
        covered = {t for t, r in reference.items() if r.high_confidence}
        # The matrix loop (CPU, pandas over merchant-month slices) runs in a
        # worker thread and streams each template's disagreement batch to the
        # event loop, which buys references for template N while the thread
        # computes template N+1. Batches are consumed one at a time, so
        # --reference-concurrency remains the global cap on in-flight solves.
        # learn_memo stays thread-confined: the per-template process() phase
        # only starts after the producer thread has exited its context.
        loop = asyncio.get_running_loop()
        batches: asyncio.Queue[list[dict[str, Any]] | None] = asyncio.Queue()

        def compute_targeting() -> None:
            try:
                with learn_memo():
                    for idx, (template, instances, signature) in enumerate(planned, 1):
                        sampled = sample_for_discrimination(instances, reference)
                        grid = seed_grid(template, signature)
                        info: dict[str, Any] = {"unanimous": False, "sep_tids": None}
                        batch: list[dict[str, Any]] = []
                        if grid:
                            key, matrix, cached = load_candidate_matrix(
                                template, grid, signature, sampled,
                            )
                            cached_targeting = cached.get("targeting")
                            cached_unanimous = (
                                cached_targeting.get("unanimous")
                                if isinstance(cached_targeting, dict) else None
                            )
                            cached_sep_tids = (
                                cached_targeting.get("sep_tids")
                                if isinstance(cached_targeting, dict) else None
                            )
                            cached_targeting_valid = (
                                isinstance(cached_targeting, dict)
                                and cached_targeting.get("min_participation") == min_participation
                                and isinstance(cached_unanimous, bool)
                                and (
                                    (cached_unanimous and cached_sep_tids is None)
                                    or (
                                        not cached_unanimous
                                        and isinstance(cached_sep_tids, list)
                                        and all(isinstance(task_id, str) for task_id in cached_sep_tids)
                                    )
                                )
                            )
                            if cached_targeting_valid:
                                info["unanimous"] = cached_unanimous
                                info["sep_tids"] = (
                                    set(cached_sep_tids) if isinstance(cached_sep_tids, list) else None
                                )
                            else:
                                executable = max((len(v) for v in matrix.values()), default=0)
                                sep = disagreement_instances(matrix)
                                if not sep and executable >= min(min_participation, len(sampled)):
                                    info["unanimous"] = True
                                else:
                                    info["sep_tids"] = set(sep)
                                matrix_cache.put(key, {
                                    "matrix": matrix,
                                    "targeting": {
                                        "min_participation": min_participation,
                                        "unanimous": info["unanimous"],
                                        "sep_tids": (
                                            sorted(info["sep_tids"])
                                            if info["sep_tids"] is not None else None
                                        ),
                                    },
                                })
                            if not info["unanimous"]:
                                batch = [i for i in sampled
                                         if str(i["task_id"]) in (info["sep_tids"] or set())
                                         and str(i["task_id"]) not in covered]
                        else:
                            # No grid candidates: the teacher will propose some, and
                            # discrimination needs references — buy the full sample.
                            batch = [i for i in sampled if str(i["task_id"]) not in covered]
                        targeting[template] = info
                        status = "unanimous" if info["unanimous"] else f"buy {len(batch)}"
                        print(f"[targeting] {idx}/{len(planned)} {status} | {template[:60]}",
                              flush=True)
                        loop.call_soon_threadsafe(batches.put_nowait, batch)
            finally:
                # Always deliver the sentinel: a crash mid-loop must fail the
                # awaiting producer future, not deadlock the consumer.
                loop.call_soon_threadsafe(batches.put_nowait, None)

        producer = loop.run_in_executor(None, compute_targeting)
        generated: dict[str, Any] = {}
        pending_count = 0
        buys = 0
        while True:
            batch = await batches.get()
            if batch is None:
                break
            if not batch:
                continue
            pending_count += len(batch)
            buys += 1
            generated.update(await generate_references(
                instances=batch, data_dir=data_dir,
                workspace_dir=output_dir / "_reference_ws",
                samples=reference_samples, concurrency=reference_concurrency,
                agent=solver_agent,
                persist_path=output_dir / "_self_reference.jsonl",
                usage_ledger=usage_ledger,
            ))
        await producer
        if not buys:
            # No template needed new references, but references persisted by a
            # previous interrupted run still flow into discrimination (the
            # unconditional call used to pick them up as a side effect).
            generated.update(await generate_references(
                instances=[], data_dir=data_dir,
                workspace_dir=output_dir / "_reference_ws",
                samples=reference_samples, concurrency=reference_concurrency,
                agent=solver_agent,
                persist_path=output_dir / "_self_reference.jsonl",
                usage_ledger=usage_ledger,
            ))
        reference = {**generated, **reference}
        summary["reference"] = {
            "mode": reference_mode,
            "generated": len(generated),
            "pending": pending_count,
        }

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def process(template: str, instances: list[dict[str, Any]], signature) -> tuple[str, dict[str, Any]]:
        info = targeting.get(template) or {}
        # A template whose disagreement set is smaller than the participation
        # gate can never satisfy the raw gate no matter how many references we
        # buy - complete evidence over every disagreement point is the most
        # this template can offer, so the gate caps there.
        sep_tids = info.get("sep_tids")
        required_participation = (
            max(1, min(min_participation, len(sep_tids))) if sep_tids is not None
            else min_participation
        )

        def run_discrimination(candidates, ref):
            with learn_memo():
                return discriminate_template(
                    data=data, template=template, instances=instances,
                    candidates=candidates, signature=signature, reference=ref,
                )

        def adoption_ok(report):
            adopted = next((c for c in report.candidates if c.spec.name == report.adopted), None)
            needed = required_participation if noisy_reference else min_participation
            if adopted is None or adopted.rate is None or \
                    report.funnel["participated"] < needed:
                return False, adopted
            if not noisy_reference:
                return adopted.rate >= min_adoption_rate, adopted
            # Relative gate for noisy references. Rivals with the identical
            # per-instance match pattern are extensionally indistinguishable
            # on this evidence and cannot be asked to lose by a margin.
            rival_rates = [
                c.rate for c in report.candidates
                if c is not adopted and c.rate is not None
                and c.instance_matches != adopted.instance_matches
            ]
            runner_up = max(rival_rates, default=0.0)
            dev = (report.official_dev or {}).get(adopted.spec.name)
            dev_ok = dev is None or dev["total"] == 0 or dev["agree"] == dev["total"]
            # Size-aware evidence gate: the observed agreement must be
            # significant against a coin-flip null, so a high rate on a thin
            # sample (3/3) no longer earns a skill while a decisive larger
            # sample does. The relative margin and dev check are unchanged.
            evidence_ok = passes_binomial_gate(
                adopted.agree, adopted.total,
                null_rate=adoption_null_rate, alpha=adoption_alpha,
            )
            return (evidence_ok
                    and adopted.rate >= adoption_floor
                    and adopted.rate - runner_up >= adoption_margin
                    and dev_ok), adopted

        async def escalate(report, specs, ref):
            """Re-solve (3-sample majority) up to 3 instances that separate the
            top two candidates; returns the updated reference dict."""
            from dabstep_agent_pydantic.ablation import answers_match
            from dabstep_agent_pydantic.distill.combinators import SpecNotExecutable, compile_spec

            ranked = sorted((c for c in report.candidates if c.rate is not None),
                            key=lambda c: -c.rate)
            if len(ranked) < 2:
                return ref
            by_name = {s.name: s for s in specs}
            fns = [compile_spec(by_name[c.spec.name]) for c in ranked[:2]]
            disputed = []
            with learn_memo():
                for instance in instances[:ESCALATE_SCAN_CAP]:
                    params = signature.parse(str(instance["question"]), data)
                    if params is None:
                        continue
                    guidelines = str(instance.get("guidelines") or "")
                    try:
                        va, vb = (fn(data, params, guidelines) for fn in fns)
                    except (SpecNotExecutable, KeyError, ValueError, TypeError):
                        continue
                    if not answers_match(va, vb):
                        disputed.append(instance)
                    if len(disputed) >= 3:
                        break
            if not disputed:
                return ref
            fresh = await generate_references(
                instances=disputed, data_dir=data_dir,
                workspace_dir=output_dir / "_reference_ws",
                samples=3, concurrency=reference_concurrency,
                agent=solver_agent, tag_prefix="esc",
                persist_path=output_dir / "_self_reference_escalated.jsonl",
                usage_ledger=usage_ledger,
            )
            return {**ref, **fresh}

        ref = reference
        # Grid-first: the standard interpretation grid needs no model calls.
        candidates = seed_grid(template, signature)
        if info.get("unanimous") and candidates:
            # Every candidate produced identical answers on the sampled
            # instances: they are extensionally equal on this template's
            # observed parameter range, so adoption needs no references.
            # Deterministic pick: manual-consistent first, then by name.
            spec = sorted(candidates, key=lambda c: (c.contradicts_manual, c.name))[0]
            from dabstep_agent_pydantic.distill.discriminate import CandidateResult, DiscriminationReport
            report = DiscriminationReport(template=template)
            report.candidates = [CandidateResult(spec=spec)]
            report.adopted = spec.name
            report.adoption_basis = "unanimous over sampled instances"
            checks = harden(data=data, report=report, signature=signature,
                            sample_instance=instances[0])
            write_skill_artifact(
                out_dir=output_dir, template=template, spec=spec,
                evidence={"discrimination": report.summary(), "hardening": checks},
                provenance={
                    "reference_kind": "model-generated reference answers",
                    "adoption_basis": "unanimous over sampled instances",
                    "doc_fingerprints": doc_fingerprints,
                },
            )
            return "skills", {
                "skill_id": template_skill_id(template),
                "template": template[:80],
                "adopted": spec.name,
                "rate": None,
                "teacher_consulted": False,
                "unanimous": True,
            }, {"spec": spec, "rate": None, "funnel": {}}
        if candidates and self_bootstrap:
            # Disagreement-driven self-labeling: only label instances that
            # separate candidates, sequentially, with early stopping.
            key, matrix, cached = load_candidate_matrix(
                template, candidates, signature, instances,
            )
            if not cached:
                matrix_cache.put(key, {"matrix": matrix})
            covered = {t for t in (ref or {}) if ref[t].high_confidence}
            if not is_unanimous(matrix):
                label_tids = [t for t in select_labeling_instances(matrix, max_labels=6)
                              if t not in covered]
                labels = await bootstrap_labels(
                    instances=instances, label_tids=label_tids, data_dir=data_dir,
                    workspace_dir=output_dir / "_bootstrap_ws", matrix=matrix,
                    persist_path=output_dir / "_bootstrap_labels.jsonl",
                    samples=bootstrap_samples, consensus=bootstrap_consensus,
                    usage_ledger=usage_ledger,
                )
                ref = {**ref, **labels}
        specs = candidates
        report = run_discrimination(candidates, ref) if candidates else None
        ok, adopted = adoption_ok(report) if report else (False, None)
        teacher_consulted = False
        escalated = False

        # Failure triage: match the remedy to the failure shape. Evidence-
        # shaped failures (participation shortfall, close margin) get cheap
        # targeted evidence first; the teacher is the LAST resort, consulted
        # only when no candidate explains the references at all - adding
        # candidates enlarges the rival pool and dilutes the margin, so
        # consulting it for an evidence problem can turn a would-pass
        # template into a fail.
        async def buy_participation() -> None:
            nonlocal ref, report, ok, adopted
            covered_tids = {t for t, r in ref.items() if r.high_confidence}
            topup = [i for i in instances if str(i["task_id"]) not in covered_tids]
            topup = topup[:2 * min_participation]
            if not topup:
                return
            fresh = await generate_references(
                instances=topup, data_dir=data_dir,
                workspace_dir=output_dir / "_reference_ws",
                samples=reference_samples, concurrency=reference_concurrency,
                agent=solver_agent, tag_prefix="topup",
                persist_path=output_dir / "_self_reference.jsonl",
                usage_ledger=usage_ledger,
            )
            if fresh:
                ref = {**ref, **fresh}
                report = run_discrimination(specs, ref)
                ok, adopted = adoption_ok(report)

        async def buy_margin() -> None:
            nonlocal ref, report, ok, adopted, escalated
            for _ in range(escalation_rounds):
                new_ref = await escalate(report, [s for s in specs], ref)
                if new_ref is ref:
                    break
                ref = new_ref
                escalated = True
                report = run_discrimination(specs, ref)
                ok, adopted = adoption_ok(report)
                if ok:
                    break

        def evidence_shaped() -> bool:
            return (report is not None and adopted is not None
                    and adopted.rate is not None and adopted.rate >= adoption_floor)

        if not ok and noisy_reference and specs and evidence_shaped():
            if report.funnel["participated"] < required_participation:
                await buy_participation()
            if not ok and evidence_shaped():
                await buy_margin()

        if not ok and (not noisy_reference or not evidence_shaped()):
            # Candidate-shaped failure (or non-noisy modes, where evidence
            # remedies do not apply): consult the teacher for new hypotheses.
            teacher_consulted = True
            try:
                teacher_candidates = await propose_candidates(
                    template=template,
                    instance_questions=[str(t["question"]) for t in instances[:3]],
                    docs_dir=data_dir,
                    agent=teacher_agent,
                    guidelines=str(instances[0].get("guidelines") or ""),
                )
            except Exception as exc:  # noqa: BLE001 - a flaky gateway must not kill the run.
                return "skipped", {"template": template[:80],
                                   "reason": f"teacher unavailable: {type(exc).__name__}"}, None
            merged = candidates + [
                c for c in teacher_candidates
                if all(c.axis_summary != s.axis_summary or c.population != s.population for s in candidates)
            ]
            if not merged:
                return "skipped", {"template": template[:80], "reason": "no cited candidates"}, None
            specs = merged
            report = run_discrimination(merged, ref)
            ok, adopted = adoption_ok(report)
            if not ok and noisy_reference and evidence_shaped():
                # The teacher's pool may need evidence of its own.
                await buy_margin()
        if not ok:
            entry = {
                "template": template[:80],
                "reason": "coverage discipline",
                "funnel": report.funnel,
                "best_rate": adopted.rate if adopted else None,
                "escalated": escalated,
            }
            # Third-tier output: when the leading candidate clears the
            # adoption floor but the template still fails purely for want of
            # evidence (participation), the interpretation itself is well
            # supported even though it does not earn a deterministic skill.
            # Distil it into a schema-level note that steers the LLM path.
            # Emitted ONLY from this strong-signal shortfall - candidate-
            # shaped failures (no candidate explains the references) and true
            # ambiguity (no clear leader) produce no note. The note is still
            # subject to the calibration audit like any other artifact.
            note_emitted = False
            if noisy_reference and report is not None and adopted is not None \
                    and adopted.rate is not None and adopted.rate >= adoption_floor \
                    and report.funnel["participated"] < required_participation:
                write_note_artifact(
                    out_dir=output_dir, template=template, spec=adopted.spec,
                    evidence={"discrimination": report.summary()},
                    provenance={
                        "reference_kind": "model-generated reference answers",
                        "basis": "leading candidate below participation gate",
                        "leading_rate": adopted.rate,
                        "doc_fingerprints": doc_fingerprints,
                    },
                )
                note_emitted = True
                entry["note_emitted"] = True
            # Carry the evidence forward: the family-transfer post-pass may
            # still adopt this template's leading candidate on the strength
            # of a decisive sibling adoption.
            context = {
                "report": report, "signature": signature,
                "instances": instances, "template": template,
                "note_emitted": note_emitted,
            }
            return "skipped", entry, context
        checks = harden(data=data, report=report, signature=signature, sample_instance=instances[0])
        write_skill_artifact(
            out_dir=output_dir,
            template=template,
            spec=adopted.spec,
            evidence={
                "discrimination": report.summary(),
                "hardening": checks,
            },
            provenance={
                "reference_kind": "model-generated reference answers",
                "instances": report.funnel,
                "doc_fingerprints": doc_fingerprints,
            },
        )
        return "skills", {
            "skill_id": template_skill_id(template),
            "template": template[:80],
            "adopted": report.adopted,
            "rate": adopted.rate,
            "teacher_consulted": teacher_consulted,
            "escalated": escalated,
        }, {"spec": adopted.spec, "rate": adopted.rate, "funnel": report.funnel,
            "agree": adopted.agree, "total": adopted.total}

    done_count = 0

    async def guarded(item) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        nonlocal done_count
        async with semaphore:
            kind, entry, context = await process(*item)
            done_count += 1
            outcome = entry.get("adopted") if kind == "skills" else \
                f"skipped ({entry.get('reason')})"
            print(f"[discriminate] {done_count}/{len(planned)} {outcome} | "
                  f"{entry.get('template', '')[:60]}", flush=True)
            return kind, entry, context

    results = await asyncio.gather(*(guarded(item) for item in planned))

    # Family evidence transfer post-pass: a candidate decisively adopted on a
    # sibling template (same population + interpretation axes) is prior
    # evidence for the identical candidate on templates that fell short of
    # the gate, provided the local evidence points the same way. Transferred
    # skills carry their provenance and remain subject to the calibration
    # audit like any other artifact.
    sibling_adoptions: dict[tuple[str, str], float] = {}
    # Pooled family evidence: summed agree/total across decisively adopted
    # siblings sharing the interpretation axes. A template too thin to clear
    # the binomial gate alone can clear it on the family's pooled weight.
    pooled_evidence: dict[tuple[str, str], list[int]] = {}

    def _pool(key, agree, total):
        slot = pooled_evidence.setdefault(key, [0, 0])
        slot[0] += int(agree or 0)
        slot[1] += int(total or 0)

    for spec, rate, funnel in resumed_adoptions:
        if rate is not None and rate >= FAMILY_TRANSFER_SIBLING_RATE \
                and funnel.get("participated", 0) >= min_participation:
            key = (spec.population, spec.axis_summary)
            sibling_adoptions[key] = max(sibling_adoptions.get(key, 0.0), rate)
            _pool(key, funnel.get("_agree", 0), funnel.get("_total", 0))
    for kind, _entry, context in results:
        if kind == "skills" and context and context.get("rate") is not None \
                and context["funnel"]["participated"] >= min_participation \
                and context["rate"] >= FAMILY_TRANSFER_SIBLING_RATE:
            spec = context["spec"]
            key = (spec.population, spec.axis_summary)
            sibling_adoptions[key] = max(sibling_adoptions.get(key, 0.0), context["rate"])
            _pool(key, context.get("agree", 0), context.get("total", 0))

    for kind, entry, context in results:
        if kind != "skipped" or not context:
            summary[kind].append(entry)
            continue
        report = context["report"]
        best = next((c for c in report.candidates if c.spec.name == report.adopted), None)
        key = (best.spec.population, best.spec.axis_summary) if best else None
        if best is None or best.rate is None or best.rate < FAMILY_TRANSFER_LOCAL_FLOOR \
                or key not in sibling_adoptions:
            summary["skipped"].append(entry)
            continue
        # Pool this template's local evidence with the family and require the
        # combined counts to pass the same binomial gate the direct path uses.
        pooled_agree = pooled_evidence.get(key, [0, 0])[0] + best.agree
        pooled_total = pooled_evidence.get(key, [0, 0])[1] + best.total
        if not passes_binomial_gate(pooled_agree, pooled_total,
                                    null_rate=adoption_null_rate, alpha=adoption_alpha):
            summary["skipped"].append(entry)
            continue
        template = context["template"]
        checks = harden(data=data, report=report, signature=context["signature"],
                        sample_instance=context["instances"][0])
        write_skill_artifact(
            out_dir=output_dir,
            template=template,
            spec=best.spec,
            evidence={"discrimination": report.summary(), "hardening": checks},
            provenance={
                "reference_kind": "model-generated reference answers",
                "instances": report.funnel,
                "adoption_basis": "family transfer",
                "sibling_rate": sibling_adoptions[key],
                "pooled_evidence": f"{pooled_agree}/{pooled_total}",
                "doc_fingerprints": doc_fingerprints,
            },
        )
        summary["skills"].append({
            "skill_id": template_skill_id(template),
            "template": template[:80],
            "adopted": report.adopted,
            "rate": best.rate,
            "teacher_consulted": False,
            "family_transfer": True,
        })
        # A deterministic skill supersedes any note this template emitted:
        # the note steers the LLM path, which is now bypassed for it.
        if context.get("note_emitted"):
            (output_dir / f"{template_note_id(template)}.json").unlink(missing_ok=True)
    summary["usage_trace"] = usage_ledger.summary()
    return summary


def select_proposed_templates(
    templates: dict[str, list[dict[str, Any]]], proposals_path: Path,
) -> dict[str, Any]:
    """Match solver-filed proposals against the induced template set.

    Proposals carry the normalized template of the question they were filed
    for; matching is exact against group_templates keys, so a proposal can
    only ever select templates the public task set actually contains."""
    from dabstep_agent_pydantic.distill.templates import normalize_question

    requested: list[str] = []
    for line in Path(proposals_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        template = str(row.get("template") or "").strip() or normalize_question(
            " ".join(str(row.get("question") or "").split())
        )
        if template and template not in requested:
            requested.append(template)
    matched = {t: instances for t, instances in templates.items() if t in requested}
    return {
        "requested": requested,
        "templates": matched,
        "unmatched": [t[:80] for t in requested if t not in matched],
    }


def load_reference_official_only(official_dev_path: Path | None):
    from dabstep_agent_pydantic.distill.discriminate import ReferenceRecord
    records: dict[str, ReferenceRecord] = {}
    if official_dev_path is not None:
        for row in json.loads(Path(official_dev_path).read_text(encoding="utf-8")):
            records[str(row["task_id"])] = ReferenceRecord(
                task_id=str(row["task_id"]), answer=str(row["answer"]),
                high_confidence=True, source="official_dev",
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill deterministic skills from public templates")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional pre-generated model reference answers; omitted -> "
                             "self-bootstrap labels with your own model (disagreement-driven)")
    parser.add_argument("--official-dev", type=Path, default=None,
                        help="Benchmark-published dev sample with official answers (overrides consensus)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-instances", type=int, default=5)
    parser.add_argument("--min-participation", type=int, default=4)
    parser.add_argument("--min-adoption-rate", type=float, default=0.90)
    parser.add_argument("--reference-mode", choices=["full", "budget"], default="full",
                        help="full: solve every planned instance once and aggregate "
                             "(default); budget: disagreement-driven bootstrap labeling. "
                             "Passing --reference switches to file mode automatically.")
    parser.add_argument("--reference-samples", type=int, default=1)
    parser.add_argument("--reference-concurrency", type=int, default=12)
    parser.add_argument("--adoption-floor", type=float, default=0.60,
                        help="full mode: minimum agreement rate for adoption")
    parser.add_argument("--adoption-margin", type=float, default=0.25,
                        help="full mode: required lead over the best distinguishable rival")
    parser.add_argument("--adoption-null-rate", type=float, default=0.5,
                        help="full mode: binomial null agreement rate for the evidence gate")
    parser.add_argument("--adoption-alpha", type=float, default=0.05,
                        help="full mode: binomial significance level for the evidence gate")
    parser.add_argument("--escalation-rounds", type=int, default=1)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--from-proposals", type=Path, default=None,
                        help="Solver-filed proposal queue (_proposals.jsonl); learn only "
                             "the templates it selects — same gates, same audit")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Templates processed concurrently (network-bound stages overlap)")
    parser.add_argument("--bootstrap-samples", type=int, default=5)
    parser.add_argument("--bootstrap-consensus", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="Skip templates whose skill artifact already exists in --output")
    return parser


def _run_builtin_audit(output_dir: Path, data_dir: Path, tasks_path: Path) -> dict[str, Any]:
    """A model cannot certify itself: after every learn run, cross-check the
    emitted skills against the independent calibration oracle and remove any
    that contradict it. Runs whenever the repo's audit tooling is present;
    packaged environments without the calibration set skip with a notice."""
    import subprocess
    import sys as _sys

    script = Path("scripts/skill_audit.py")
    if not script.exists():
        return {"status": "skipped: scripts/skill_audit.py not found (run it manually)"}
    proc = subprocess.run(
        [_sys.executable, str(script), "--skills-dir", str(output_dir),
         "--context-dir", str(data_dir), "--tasks", str(tasks_path), "--remove"],
        capture_output=True, text=True,
    )
    removed = [line.split("removed ", 1)[1] for line in proc.stdout.splitlines()
               if line.startswith("removed ")]
    return {
        "status": "clean" if proc.returncode == 0 else "mismatches found",
        "removed": removed,
        "detail": proc.stdout.strip().splitlines()[-12:],
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.tasks = resolve_standard_data_path(args.tasks)
    args.data_dir = resolve_standard_data_path(args.data_dir)
    if args.official_dev is not None:
        args.official_dev = resolve_standard_data_path(args.official_dev)
    summary = asyncio.run(
        learn(
            tasks_path=args.tasks,
            data_dir=args.data_dir,
            reference_path=args.reference,
            output_dir=args.output,
            official_dev_path=args.official_dev,
            min_instances=args.min_instances,
            min_participation=args.min_participation,
            min_adoption_rate=args.min_adoption_rate,
            max_templates=args.max_templates,
            proposals_path=args.from_proposals,
            resume=args.resume,
            concurrency=args.concurrency,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_consensus=args.bootstrap_consensus,
            reference_mode=args.reference_mode,
            reference_samples=args.reference_samples,
            reference_concurrency=args.reference_concurrency,
            adoption_floor=args.adoption_floor,
            adoption_null_rate=args.adoption_null_rate,
            adoption_alpha=args.adoption_alpha,
            adoption_margin=args.adoption_margin,
            escalation_rounds=args.escalation_rounds,
        )
    )
    # stdout can be polluted by subprocesses spawned from LLM-written code
    # (they inherit the real fd, bypassing tool-level capture), so the summary
    # is also persisted next to the artifacts.
    summary["audit"] = _run_builtin_audit(args.output, args.data_dir, args.tasks)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    (args.output / "_learn_summary.json").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
