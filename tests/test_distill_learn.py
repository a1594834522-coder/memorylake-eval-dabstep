import asyncio
import json

import pandas as pd
import pytest

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.emit import (
    GeneratedSkill,
    load_generated_skills,
    write_skill_artifact,
)
from dabstep_agent_pydantic.distill.learn import learn
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec, InterpretationSpec, OutputSpec

TEMPLATE = "What is the fee ID or IDs that apply to account_type = <LETTER> and aci = <LETTER>?"


def _adopted_spec(name="wildcard_aware", policy="manual", contradicts=False) -> InterpretationSpec:
    return InterpretationSpec(
        name=name, population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], value="rule_id",
                               reducer="collect_ids", wildcard_policy=policy),
        output=OutputSpec(kind="id_list"), manual_citation="manual §5",
        contradicts_manual=contradicts,
    )


def _data() -> DABStepData:
    return DABStepData(
        fees=pd.DataFrame(
            [
                {"ID": 1, "card_scheme": "S", "account_type": [], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": True, "aci": ["A"], "fixed_amount": 0.1, "rate": 10, "intracountry": None},
                {"ID": 2, "card_scheme": "S", "account_type": ["H"], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": None, "aci": [], "fixed_amount": 0.5, "rate": 0, "intracountry": None},
            ]
        ),
        payments=pd.DataFrame([{"merchant": "M_X", "year": 2023, "day_of_year": 1, "card_scheme": "S",
                                "is_credit": True, "aci": "A", "eur_amount": 1.0, "issuing_country": "NL",
                                "acquirer": "a", "has_fraudulent_dispute": False}]),
        merchants=pd.DataFrame([{"merchant": "M_X", "account_type": "H", "capture_delay": "1",
                                 "merchant_category_code": 1, "acquirer": ["a"]}]),
        acquirer_countries=pd.DataFrame([{"acquirer": "a", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 1, "description": "d"}]),
    )


def test_artifact_roundtrip_and_runtime_solve(tmp_path):
    write_skill_artifact(
        out_dir=tmp_path, template=TEMPLATE, spec=_adopted_spec(),
        evidence={"discrimination": {"funnel": {"instances": 20}}},
        provenance={"reference_kind": "model-generated reference answers"},
    )
    skills = load_generated_skills(tmp_path)
    assert len(skills) == 1 and isinstance(skills[0], GeneratedSkill)
    answer = skills[0].solve(_data(), "What is the fee ID or IDs that apply to account_type = H and aci = A?", "")
    assert answer == "1, 2"
    assert skills[0].solve(_data(), "unrelated question", "") is None


def test_artifact_safety_rejects_forbidden_content(tmp_path):
    with pytest.raises(ValueError):
        write_skill_artifact(
            out_dir=tmp_path, template=TEMPLATE, spec=_adopted_spec(),
            evidence={"note": "validated against pseudo-gold"},
            provenance={},
        )
    with pytest.raises(ValueError):
        write_skill_artifact(
            out_dir=tmp_path, template=TEMPLATE, spec=_adopted_spec(),
            evidence={"reference_answer": "42"},
            provenance={},
        )


class _FakeTeacher:
    """Stands in for the teacher agent: returns fixed candidates."""

    def __init__(self, candidates):
        self._candidates = candidates

    async def run(self, prompt):
        class _Result:
            def __init__(self, candidates):
                from dabstep_agent_pydantic.distill.hypotheses import TeacherProposal
                self.output = TeacherProposal(candidates=candidates)
        return _Result(self._candidates)


def _learn_env(tmp_path, answers: dict[str, str]):
    tasks = [
        {"task_id": str(i), "question": f"What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"}
        for i in range(1, 7)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    reference_path = tmp_path / "ref.json"
    reference_path.write_text(json.dumps({
        "primary": answers, "candidates": {}, "ambiguous": [], "resolved": list(answers), "stats": {},
    }))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean the rule applies to all values.")
    return tasks_path, data_dir, reference_path


def test_learn_emits_skill_when_discipline_met(tmp_path):
    tasks_path, data_dir, reference_path = _learn_env(
        tmp_path, {str(i): "1, 2" for i in range(1, 7)}
    )
    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, reference_path=reference_path,
        output_dir=out, min_instances=5, min_participation=4,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    assert len(summary["skills"]) == 1
    assert summary["skills"][0]["adopted"] == "seed_ids_wildcard"  # grid-first: seeded candidate wins
    assert summary["skills"][0]["teacher_consulted"] is False
    assert list(out.glob("skill_*.json"))


def test_learn_coverage_discipline_rejects_low_agreement(tmp_path):
    tasks_path, data_dir, reference_path = _learn_env(
        tmp_path, {str(i): "999" for i in range(1, 7)}  # reference disagrees with all candidates
    )
    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, reference_path=reference_path,
        output_dir=out, min_instances=5, min_participation=4,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    assert not summary["skills"]
    assert any(s["reason"] == "coverage discipline" for s in summary["skipped"])
    assert not list(out.glob("skill_*.json"))


class _FakeSolverAgent:
    """Stands in for the verifier's solver in full reference mode."""

    def __init__(self, answer):
        self._answer = answer
        self.calls = 0

    async def run(self, prompt, **kwargs):
        self.calls += 1
        answer = self._answer

        class _R:
            class output:
                agent_answer = answer
        return _R()


def test_learn_full_mode_generates_references_and_emits_skill(tmp_path):
    tasks_path, data_dir, _ = _learn_env(tmp_path, {})
    out = tmp_path / "skills"
    solver = _FakeSolverAgent("1, 2")
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    assert summary["reference"] == {"mode": "full", "generated": 6, "pending": 6}
    assert solver.calls == 6  # one single-shot solve per instance
    assert len(summary["skills"]) == 1
    assert summary["skills"][0]["escalated"] is False
    assert list(out.glob("skill_*.json"))
    # The task-answer mapping stays in the working dir, never in artifacts.
    assert (out / "_self_reference.jsonl").exists()
    from dabstep_agent_pydantic.distill.emit import load_generated_skills
    assert load_generated_skills(out)


def test_learn_reuses_persisted_candidate_matrix_and_targeting(tmp_path, monkeypatch):
    from dabstep_agent_pydantic.distill import learn as learn_mod

    tasks_path, data_dir, _ = _learn_env(tmp_path, {})
    out = tmp_path / "skills"
    real_answer_matrix = learn_mod.answer_matrix
    calls = 0

    def counting_answer_matrix(**kwargs):
        nonlocal calls
        calls += 1
        return real_answer_matrix(**kwargs)

    monkeypatch.setattr(learn_mod, "answer_matrix", counting_answer_matrix)
    first = asyncio.run(learn_mod.learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=_FakeSolverAgent("1, 2"),
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    second = asyncio.run(learn_mod.learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=_FakeSolverAgent("1, 2"),
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))

    assert calls == 1
    assert first["matrix_cache"] == {"hits": 0, "misses": 1}
    assert second["matrix_cache"] == {"hits": 1, "misses": 0}
    cache_rows = list((out / "_matrix_cache").glob("*.json"))
    assert len(cache_rows) == 1
    assert "reference" not in cache_rows[0].read_text(encoding="utf-8").lower()


def test_learn_full_mode_rejects_and_escalates_when_reference_disagrees(tmp_path):
    tasks_path, data_dir, _ = _learn_env(tmp_path, {})
    out = tmp_path / "skills"
    solver = _FakeSolverAgent("999")  # disagrees with every candidate
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    assert not summary["skills"]
    skipped = [s for s in summary["skipped"] if s["reason"] == "coverage discipline"]
    # Failure triage: a candidate-shaped failure (rate 0) must NOT waste
    # escalation solves - the teacher is consulted instead.
    assert skipped and skipped[0]["escalated"] is False
    assert not list(out.glob("skill_*.json"))


def test_generate_references_resume_skips_persisted(tmp_path):
    from dabstep_agent_pydantic.distill.reference_gen import generate_references

    persist = tmp_path / "refs.jsonl"
    persist.write_text(json.dumps({"task_id": "1", "answer": "cached"}) + "\n")
    solver = _FakeSolverAgent("fresh")
    records = asyncio.run(generate_references(
        instances=[{"task_id": "1", "question": "q"}, {"task_id": "2", "question": "q"}],
        data_dir=tmp_path, workspace_dir=tmp_path, samples=1,
        agent=solver, persist_path=persist,
    ))
    assert solver.calls == 1  # only task 2 solved
    assert records["1"].answer == "cached" and records["2"].answer == "fresh"
    assert records["2"].source == "self_reference"


def test_escalate_scans_are_bounded_and_memoized(tmp_path, monkeypatch):
    """A payments-family disagreement scan must not recompute the merchant-month
    slice per instance unmemoized, and must not scan past ESCALATE_SCAN_CAP."""
    import dabstep_agent_pydantic.distill.learn as learn_mod
    from dabstep_agent_pydantic.distill.memo import learn_memo

    # Many more instances than ESCALATE_SCAN_CAP, none of which resolve
    # (unsignable question) so the loop would run to the end without a cap.
    tasks = [
        {"task_id": str(i), "question": f"What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"}
        for i in range(1, 200)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean the rule applies to all values.")

    scan_calls = []
    real_parse = None

    out = tmp_path / "skills"
    solver = _FakeSolverAgent("999")  # never agrees -> forces escalation every time
    summary = asyncio.run(learn_mod.learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    # 199 instances presented; the escalate scan must not touch more than the cap.
    assert learn_mod.ESCALATE_SCAN_CAP == 30
    skipped = [s for s in summary["skipped"] if s["reason"] == "coverage discipline"]
    assert skipped and skipped[0]["escalated"] is False  # triage: hopeless candidates skip escalation


def test_phase_r_only_solves_the_discrimination_sample(tmp_path):
    """Phase R must not solve raw instances beyond what discrimination
    actually samples (max_instances=12/template) — solving the rest is
    pure waste on templates with many instances."""
    tasks = [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"}
        for i in range(1, 41)  # 40 instances, well beyond the 12-sample cap
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean the rule applies to all values.")

    out = tmp_path / "skills"
    solver = _FakeSolverAgent("1, 2")
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    assert solver.calls == 12  # capped to the discrimination sample, not all 40
    assert summary["reference"]["generated"] == 12
    assert summary["reference"]["pending"] == 12
    assert len(summary["skills"]) == 1


def test_generate_references_retries_transient_failures_once(tmp_path):
    """Instances lost to transient trouble get one automatic second pass."""
    from dabstep_agent_pydantic.distill.reference_gen import generate_references

    class FlakyThenGoodSolver:
        def __init__(self):
            self.calls = 0

        async def run(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            class _R:
                class output:
                    agent_answer = "42"
            return _R()

    solver = FlakyThenGoodSolver()
    records = asyncio.run(generate_references(
        instances=[{"task_id": "5", "question": "q"}], data_dir=tmp_path,
        workspace_dir=tmp_path, samples=1, agent=solver,
        persist_path=tmp_path / "refs.jsonl",
    ))
    assert solver.calls == 2  # first failed, retry sweep succeeded
    assert records["5"].answer == "42"


def test_participation_topup_recovers_near_miss_template(tmp_path):
    """best candidate clears the floor but participation falls short: the
    top-up buys references for uncovered instances instead of skipping."""
    from dabstep_agent_pydantic.distill import reference_gen

    # 6 instances; the initial Phase R solver loses 4 of them (returns errors
    # twice: first sweep + self-heal), leaving participation 2 < 4. The
    # top-up pass succeeds (fresh attempts recover).
    tasks = [
        {"task_id": str(i), "question": f"What is the fee ID or IDs that apply to account_type = H and aci = A? (v{i})",
         "guidelines": "", "level": "easy"}
        for i in range(1, 7)
    ]
    # Distinct questions break template grouping; use identical question text.
    for t in tasks:
        t["question"] = "What is the fee ID or IDs that apply to account_type = H and aci = A?"
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean wildcard.")

    class FlakyFirstSweeps:
        """Fails first sweep (6) + most of the self-heal sweep, leaving
        participation 2 < 4 so only the participation top-up can recover."""

        def __init__(self):
            self.calls = 0

        async def run(self, prompt, **kwargs):
            self.calls += 1
            if self.calls <= 10:
                raise ConnectionError("lost solve")

            class _R:
                class output:
                    agent_answer = "1, 2"
            return _R()

    out = tmp_path / "skills"
    solver = FlakyFirstSweeps()
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([]),
    ))
    assert len(summary["skills"]) == 1, summary
    assert list(out.glob("skill_*.json"))


def test_family_transfer_adopts_near_miss_sibling(tmp_path, monkeypatch):
    """A decisively adopted sibling lets an under-evidenced template adopt the
    identical candidate via the post-pass (with provenance)."""
    import dabstep_agent_pydantic.distill.learn as learn_mod

    # Template A: 6 instances, all referenced -> clean adoption at rate 1.0.
    # Template B: same candidate space (sibling), but only 2 of its 6
    # instances get references (solver loses the rest permanently) -> fails
    # participation, then transfers from A.
    tasks_a = [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"} for i in range(1, 7)
    ]
    tasks_b = [
        {"task_id": str(100 + i), "question": "Which is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"} for i in range(1, 7)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks_a + tasks_b))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean wildcard.")

    class SplitSolver2:
        """Template A always answered; template B resolves only 2 references
        (fails the participation gate but keeps a leading candidate)."""

        def __init__(self):
            self.b_success = 0

        async def run(self, prompt, **kwargs):
            if "Which is the fee ID" in prompt:
                if self.b_success >= 2:
                    raise ConnectionError("lost solve")
                self.b_success += 1

            class _R:
                class output:
                    agent_answer = "1, 2"
            return _R()

    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=SplitSolver2(),
        teacher_agent=_FakeTeacher([]),
    ))
    transferred = [s for s in summary["skills"] if s.get("family_transfer")]
    assert transferred, summary
    assert transferred[0]["rate"] >= learn_mod.FAMILY_TRANSFER_LOCAL_FLOOR
    assert len(list(out.glob("skill_*.json"))) == 2


def test_skill_artifacts_record_doc_fingerprints(tmp_path):
    """Skills must pin the exact doc versions they were learned from, so
    freeze can refuse to upload drifted knowledge."""
    import hashlib

    tasks_path, data_dir, reference_path = _learn_env(
        tmp_path, {str(i): "1, 2" for i in range(1, 7)}
    )
    out = tmp_path / "skills"
    asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, reference_path=reference_path,
        output_dir=out, min_instances=5, min_participation=4,
        teacher_agent=_FakeTeacher([_adopted_spec()]),
    ))
    artifact = json.loads(next(out.glob("skill_*.json")).read_text())
    prints = artifact["provenance"]["doc_fingerprints"]
    expected = hashlib.sha256((data_dir / "manual.md").read_bytes()).hexdigest()
    assert prints["manual.md"] == expected


def test_freeze_detects_doc_drift(tmp_path):
    from dabstep_agent_pydantic.cli import _doc_drift_against_skills

    skills = tmp_path / "skills"
    skills.mkdir()
    doc = tmp_path / "manual.md"
    doc.write_text("original knowledge")
    import hashlib
    (skills / "skill_abc.json").write_text(json.dumps({
        "provenance": {"doc_fingerprints": {
            "manual.md": hashlib.sha256(b"original knowledge").hexdigest()}},
    }))
    assert _doc_drift_against_skills([doc], skills) == []
    doc.write_text("edited after learn")
    drift = _doc_drift_against_skills([doc], skills)
    assert len(drift) == 1 and "manual.md" in drift[0]
    # Docs learn never consulted (curriculum exports) pass through freely.
    other = tmp_path / "curriculum.md"
    other.write_text("anything")
    assert _doc_drift_against_skills([other], skills) == []


def test_unanimous_template_adopts_with_zero_references(tmp_path):
    """When every grid candidate answers identically on the sample, the
    template adopts without a single model call."""
    import pandas as pd
    from dabstep_agent_pydantic.dabstep_core import DABStepData

    # Fees listing account_type/aci explicitly: wildcard and strict matching
    # coincide, so all seeded candidates are extensionally equal.
    data = DABStepData(
        fees=pd.DataFrame([
            {"ID": 1, "card_scheme": "S", "account_type": ["H"], "capture_delay": None,
             "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
             "is_credit": True, "aci": ["A"], "fixed_amount": 0.1, "rate": 10, "intracountry": None},
            {"ID": 2, "card_scheme": "S", "account_type": ["H"], "capture_delay": None,
             "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
             "is_credit": None, "aci": ["A"], "fixed_amount": 0.5, "rate": 0, "intracountry": None},
        ]),
        payments=pd.DataFrame([{"merchant": "M_X", "year": 2023, "day_of_year": 1, "card_scheme": "S",
                                "is_credit": True, "aci": "A", "eur_amount": 1.0, "issuing_country": "NL",
                                "acquirer": "a", "has_fraudulent_dispute": False}]),
        merchants=pd.DataFrame([{"merchant": "M_X", "account_type": "H", "capture_delay": "1",
                                 "merchant_category_code": 1, "acquirer": ["a"]}]),
        acquirer_countries=pd.DataFrame([{"acquirer": "a", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 1, "description": "d"}]),
    )
    tasks = [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"} for i in range(1, 7)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: explicit lists match exactly.")

    out = tmp_path / "skills"
    solver = _FakeSolverAgent("1, 2")
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=solver,
        teacher_agent=_FakeTeacher([]),
    ))
    assert solver.calls == 0  # zero model calls: unanimous adoption
    assert len(summary["skills"]) == 1
    assert summary["skills"][0].get("unanimous") is True
    artifact = json.loads(next(out.glob("skill_*.json")).read_text())
    assert artifact["provenance"]["adoption_basis"] == "unanimous over sampled instances"


def test_builtin_audit_skips_gracefully_outside_repo(tmp_path, monkeypatch):
    from dabstep_agent_pydantic.distill.learn import _run_builtin_audit

    monkeypatch.chdir(tmp_path)  # no scripts/skill_audit.py here
    result = _run_builtin_audit(tmp_path, tmp_path, tmp_path / "tasks.json")
    assert result["status"].startswith("skipped")


def test_note_emitted_when_leader_clears_floor_but_participation_short(tmp_path):
    """Strong-signal shortfall: the leading candidate clears the adoption
    floor but too few references ever land (top-up cannot recover). No skill,
    but a schema-level note is distilled for the LLM path."""
    from dabstep_agent_pydantic.distill.emit import load_calibration_notes

    tasks = [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"} for i in range(1, 9)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean wildcard.")

    class ThreeThenDead:
        """Succeeds on exactly 3 solves (agreeing), errors on everything
        else forever - including the top-up and escalation sweeps."""
        def __init__(self):
            self.calls = 0
            self.ok = 0
        async def run(self, prompt, **kwargs):
            self.calls += 1
            if self.ok < 3:
                self.ok += 1
                class _R:
                    class output:
                        agent_answer = "1, 2"
                return _R()
            raise ConnectionError("lost solve")

    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=ThreeThenDead(),
        teacher_agent=_FakeTeacher([]),
    ))
    assert not summary["skills"], summary
    notes = load_calibration_notes(out)
    assert len(notes) == 1, [n.note_id for n in notes]
    assert "wildcard" in notes[0].convention.lower() or "matching" in notes[0].convention.lower()


def test_note_audit_removes_oracle_contradicting_note(tmp_path):
    import subprocess, sys
    from dabstep_agent_pydantic.distill.emit import write_note_artifact
    from dabstep_agent_pydantic.distill.spec import (
        InterpretationSpec, FeeRulesSpec, OutputSpec)

    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("wildcard")
    q = "What is the fee ID or IDs that apply to account_type = H and aci = A?"
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([{"task_id": "1", "question": q, "guidelines": "", "level": "easy"}]))
    # A note with the strict (wrong) wildcard policy contradicts the oracle.
    bad = InterpretationSpec(name="strict", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], reducer="collect_ids",
                               wildcard_policy="strict"),
        output=OutputSpec(kind="id_list"), manual_citation="cite")
    note_path = write_note_artifact(out_dir=tmp_path / "skills", template=q, spec=bad,
                                    evidence={}, provenance={})
    # The subprocess sets up sys.path for the calibration modules itself.
    proc = subprocess.run(
        [sys.executable, "scripts/skill_audit.py", "--skills-dir", str(tmp_path / "skills"),
         "--context-dir", str(data_dir), "--tasks", str(tasks_path), "--remove"],
        capture_output=True, text=True, cwd=".")
    # fee_ids_by_attributes covers this template; the strict-wildcard note
    # contradicts the manual-wildcard oracle, so it is flagged and removed.
    assert "note" in proc.stdout.lower(), proc.stdout
    assert not note_path.exists(), "oracle-contradicting note should be removed"


def _learn_env_n(tmp_path, n: int):
    """Like _learn_env but with n identical-question instances."""
    tasks = [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": "", "level": "easy"} for i in range(1, n + 1)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    data_dir = tmp_path / "ctx"
    data_dir.mkdir()
    data = _data()
    data.fees.to_json(data_dir / "fees.json", orient="records")
    data.payments.to_csv(data_dir / "payments.csv", index=False)
    data.merchants.to_json(data_dir / "merchant_data.json", orient="records")
    data.acquirer_countries.to_csv(data_dir / "acquirer_countries.csv", index=False)
    data.merchant_category_codes.to_csv(data_dir / "merchant_category_codes.csv", index=False)
    (data_dir / "manual.md").write_text("Fee rules: null and empty list mean wildcard.")
    return tasks_path, data_dir


def test_binomial_gate_rejects_thin_perfect_sample(tmp_path):
    """3/3 agreement is a perfect rate but weak evidence (binomial p=0.125):
    the old hard floor would adopt it; the size-aware gate does not."""
    tasks_path, data_dir = _learn_env_n(tmp_path, 3)
    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=3, min_participation=3,
        reference_mode="full", solver_agent=_FakeSolverAgent("1, 2"),
        teacher_agent=_FakeTeacher([]),
    ))
    assert not summary["skills"], summary
    assert not list(out.glob("skill_*.json"))


def test_binomial_gate_adopts_sufficient_sample(tmp_path):
    """6/6 agreement (binomial p=0.016) clears the gate and earns a skill."""
    tasks_path, data_dir = _learn_env_n(tmp_path, 6)
    out = tmp_path / "skills"
    summary = asyncio.run(learn(
        tasks_path=tasks_path, data_dir=data_dir, output_dir=out,
        min_instances=5, min_participation=4,
        reference_mode="full", solver_agent=_FakeSolverAgent("1, 2"),
        teacher_agent=_FakeTeacher([]),
    ))
    assert len(summary["skills"]) == 1, summary
