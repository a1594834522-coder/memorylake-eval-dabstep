"""Learned-skill tools: AI-judged applicability, structured invocation,
and agent-initiated skill proposals."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dabstep_agent_pydantic import skill_tools
from dabstep_agent_pydantic.distill.shadow import clear_skill_cache
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec, InterpretationSpec, OutputSpec
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.planning import PlanDecision
from dabstep_agent_pydantic.toolsets import toolset_names_for_plan

_SPEC = InterpretationSpec(
    name="stub_mean_manual",
    population="fee_rules",
    fee_rules=FeeRulesSpec(context_dims=["account_type"]),
    output=OutputSpec(kind="decimal"),
    manual_citation="manual §5",
)


class _StubSkill:
    skill_id = "skill_stub000000"
    template = "For account type <LETTER>, what would be the average fee?"
    spec = _SPEC
    evidence = {
        "discrimination": {
            "adopted": "stub_mean_manual",
            "candidates": [{"name": "stub_mean_manual", "rate": 1.0, "agree": 6, "total": 6}],
        }
    }
    signature = SimpleNamespace(group_params=("account_type",), constant_params=())

    def doc_fingerprints(self):
        return {}

    def solve_with_params(self, data, raw_params, guidelines):
        unknown = set(raw_params) - set(self.signature.group_params)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}")
        if raw_params.get("account_type") == "H":
            return "0.42"
        return None


class _BrokenSkill(_StubSkill):
    skill_id = "skill_broken0000"

    def solve_with_params(self, data, raw_params, guidelines):
        raise KeyError("boom")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    clear_skill_cache()
    monkeypatch.setattr(skill_tools, "load_dabstep_data", lambda data_dir: object())
    yield
    clear_skill_cache()


def _ctx():
    return SimpleNamespace(deps=SimpleNamespace(data_dir="unused"))


def _install(monkeypatch, skills):
    monkeypatch.setattr(skill_tools, "_skills", lambda: skills)


def test_tools_disabled_without_artifacts(monkeypatch):
    _install(monkeypatch, [])
    assert not skill_tools.learned_skill_tools_enabled()


def test_tools_disabled_when_mode_off(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "off")
    assert not skill_tools.learned_skill_tools_enabled()


def test_plan_exposes_learned_skills_toolset_when_enabled(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "primary")
    plan = PlanDecision(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation"],
        toolset_ids=["fee_simulation"],
        verification_focus=["counterfactual"],
    )
    assert toolset_names_for_plan(plan)[:2] == ["common", "learned_skills"]


def test_list_learned_skills_exposes_params_and_interpretation(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    infos = skill_tools.list_learned_skills(_ctx())
    assert [i.skill_id for i in infos] == ["skill_stub000000"]
    assert infos[0].params == ["account_type"]
    assert infos[0].interpretation  # render_convention output, non-empty
    assert "6/6" in infos[0].adoption


def test_apply_learned_skill_executes_with_structured_params(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    result = skill_tools.apply_learned_skill(
        _ctx(), "skill_stub000000", {"account_type": "H"})
    assert result.answer == "0.42"


def test_apply_learned_skill_rejects_unknown_params(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    result = skill_tools.apply_learned_skill(
        _ctx(), "skill_stub000000", {"merchant": "Rafa_AI"})
    assert result.answer is None
    assert "unknown parameter" in result.detail


def test_apply_learned_skill_reports_no_answer(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    result = skill_tools.apply_learned_skill(
        _ctx(), "skill_stub000000", {"account_type": "Z"})
    assert result.answer is None
    assert "could not produce an answer" in result.detail


def test_apply_learned_skill_unknown_id(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    result = skill_tools.apply_learned_skill(_ctx(), "skill_nope", {})
    assert result.answer is None
    assert "unknown skill_id" in result.detail


def test_apply_learned_skill_contains_artifact_errors(monkeypatch):
    _install(monkeypatch, [_BrokenSkill()])
    result = skill_tools.apply_learned_skill(
        _ctx(), "skill_broken0000", {"account_type": "H"})
    assert result.answer is None
    assert "execution failed" in result.detail


def test_propose_skill_candidate_queues_and_dedupes(monkeypatch, tmp_path):
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS_DIR", str(tmp_path))
    q = "What is the average fee for account type H in 2023?"
    first = skill_tools.propose_skill_candidate(_ctx(), q, "mean vs sum over matching rules")
    assert "queued" in first and "already" not in first
    # A different instantiation of the same template dedupes.
    second = skill_tools.propose_skill_candidate(
        _ctx(), "What is the average fee for account type R in 2019?", "same shape")
    assert "already queued" in second
    rows = [json.loads(line) for line in
            (tmp_path / skill_tools.PROPOSALS_FILENAME).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["question"] == q
    assert rows[0]["template"]


def test_propose_skill_candidate_refuses_official_run_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS_DIR", str(tmp_path))
    ctx = SimpleNamespace(deps=SimpleNamespace(
        data_dir="unused",
        evaluation_policy=EvaluationPolicy.official(),
    ))

    result = skill_tools.propose_skill_candidate(
        ctx,
        "What is the average fee for account type H in 2023?",
        "recurring ambiguity",
    )

    assert "disabled" in result
    assert not (tmp_path / skill_tools.PROPOSALS_FILENAME).exists()


def test_select_proposed_templates_matches_normalized_questions(tmp_path):
    from dabstep_agent_pydantic.distill.learn import select_proposed_templates
    from dabstep_agent_pydantic.distill.templates import group_templates

    tasks = [
        {"task_id": "1", "question": "What is the average fee for account type H in 2023?"},
        {"task_id": "2", "question": "What is the average fee for account type R in 2019?"},
        {"task_id": "3", "question": "How many merchants are there?"},
    ]
    templates = group_templates(tasks)
    proposals = tmp_path / "_proposals.jsonl"
    proposals.write_text(json.dumps({
        "question": "What is the average fee for account type D in 2021?",
        "rationale": "recurring",
    }) + "\n" + json.dumps({
        "template": "no such template",
        "rationale": "stale",
    }) + "\n")

    selected = select_proposed_templates(templates, proposals)
    assert len(selected["templates"]) == 1
    only = next(iter(selected["templates"].values()))
    assert {t["task_id"] for t in only} == {"1", "2"}
    assert selected["unmatched"] == ["no such template"]


def test_open_mode_exposes_full_library_with_dedup(monkeypatch):
    _install(monkeypatch, [_StubSkill()])
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "primary")
    monkeypatch.setenv("DABSTEP_TOOL_SELECTION", "open")
    plan = PlanDecision(
        task_family="customer_fraud_metrics",
        selected_route_ids=["fraud_and_customer_semantics"],
        toolset_ids=["fraud_and_customer_semantics"],
        verification_focus=["denominator"],
    )
    names = toolset_names_for_plan(plan)
    assert names[:2] == ["common", "learned_skills"]
    assert "fee_simulation" in names and "fee_matching" not in names
    assert len(names) == len(set(names))


def test_open_mode_injects_toolset_hint(monkeypatch):
    monkeypatch.setenv("DABSTEP_TOOL_SELECTION", "open")
    from dabstep_agent_pydantic.workflow import _toolset_hint_prompt

    plan = PlanDecision(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation"],
        toolset_ids=["fee_simulation"],
        verification_focus=["counterfactual"],
    )
    hint = _toolset_hint_prompt(plan)
    assert "fee_simulation" in hint and "not a restriction" in hint
    monkeypatch.setenv("DABSTEP_TOOL_SELECTION", "planner")
    assert _toolset_hint_prompt(plan) == ""


class _StaleSkill(_StubSkill):
    skill_id = "skill_stale00000"

    def doc_fingerprints(self):
        return {"manual.md": "0" * 64}


class _MisshapenSkill(_StubSkill):
    skill_id = "skill_shape00000"

    def solve_with_params(self, data, raw_params, guidelines):
        return "not-a-decimal"

    def solve(self, data, question, guidelines):
        return "not-a-decimal"

    def match(self, question):
        return {"matched": True}


def test_apply_learned_skill_refuses_stale_documents(monkeypatch, tmp_path):
    (tmp_path / "manual.md").write_text("a different manual version")
    _install(monkeypatch, [_StaleSkill()])
    from dabstep_agent_pydantic.distill.shadow import clear_skill_cache
    clear_skill_cache()
    ctx = SimpleNamespace(deps=SimpleNamespace(data_dir=str(tmp_path)))
    result = skill_tools.apply_learned_skill(ctx, "skill_stale00000", {"account_type": "H"})
    assert result.answer is None
    assert "stale" in result.detail and "manual.md" in result.detail


def test_apply_learned_skill_enforces_output_shape(monkeypatch):
    _install(monkeypatch, [_MisshapenSkill()])
    result = skill_tools.apply_learned_skill(
        _ctx(), "skill_shape00000", {"account_type": "H"})
    assert result.answer is None
    assert "output-shape invariant" in result.detail


def test_try_solve_generated_guards_fire(monkeypatch, tmp_path):
    from dabstep_agent_pydantic.distill import shadow

    # Sabotage 1: stale docs → the deterministic path refuses to answer.
    (tmp_path / "manual.md").write_text("drifted")
    shadow.clear_skill_cache()
    monkeypatch.setattr(shadow, "_skills", lambda: [_StaleSkill()])
    monkeypatch.setattr(_StaleSkill, "match", lambda self, q: {"matched": True}, raising=False)
    assert shadow.try_solve_generated("q", "", object(), data_dir=str(tmp_path)) is None

    # Sabotage 2: wrong output shape → rejected, falls back to the LLM path.
    shadow.clear_skill_cache()
    monkeypatch.setattr(shadow, "_skills", lambda: [_MisshapenSkill()])
    assert shadow.try_solve_generated("q", "", object(), data_dir=None) is None

    # Control: a healthy skill still answers with the guards active.
    class _Healthy(_StubSkill):
        def match(self, question):
            return {"matched": True}

        def solve(self, data, question, guidelines):
            return "0.42"

    shadow.clear_skill_cache()
    monkeypatch.setattr(shadow, "_skills", lambda: [_Healthy()])
    got = shadow.try_solve_generated("q", "", object(), data_dir=str(tmp_path))
    assert got is not None and got.agent_answer == "0.42"


def test_render_skills_digest_is_schema_level(tmp_path):
    from dabstep_agent_pydantic.distill.emit import (
        render_skills_digest, write_note_artifact, write_skill_artifact,
    )

    template = "For account type <LETTER>, what is the average fee for <MERCHANT> in <YEAR>?"
    write_skill_artifact(
        out_dir=tmp_path, template=template, spec=_SPEC,
        evidence={"discrimination": {
            "adopted": _SPEC.name,
            "candidates": [{"name": _SPEC.name, "rate": 1.0, "agree": 6, "total": 6}],
        }},
        provenance={"doc_fingerprints": {}},
    )
    write_note_artifact(
        out_dir=tmp_path, template="What is the <X> ratio for <MERCHANT>?", spec=_SPEC,
        evidence={}, provenance={},
    )
    digest = render_skills_digest(tmp_path)
    assert "For account type <LETTER>" in digest
    assert "average the per-rule fee" in digest        # rendered convention
    assert "agreement 1.00 (6/6" in digest             # evidence basis
    assert "advisory" in digest                        # note section
    # Schema-level only: placeholders survive, no task references leak.
    assert "<MERCHANT>" in digest
    assert "task " not in digest.lower()
