from pathlib import Path

from dabstep_agent_pydantic.distill.hypotheses import (
    build_teacher_prompt,
    complete_grid,
    manual_excerpts_for_template,
)
from dabstep_agent_pydantic.distill.spec import (
    FeeRulesSpec,
    InterpretationSpec,
    OutputSpec,
    PaymentsSpec,
)


def _fee_rules_candidate() -> InterpretationSpec:
    return InterpretationSpec(
        name="teacher_mean", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["card_scheme"], reducer="mean"),
        output=OutputSpec(kind="decimal", decimals_default=6),
        manual_citation="manual §5 fee = fixed + rate * value / 10000",
    )


def test_grid_completes_reducers_and_strict_policy():
    completed = complete_grid([_fee_rules_candidate()])
    names = {c.name for c in completed}
    assert "teacher_mean" in names
    assert any(n.endswith("grid_sum") for n in names)
    assert any(n.endswith("grid_min") for n in names)
    strict = [c for c in completed if c.name.endswith("grid_strict")]
    assert strict and strict[0].contradicts_manual and strict[0].fee_rules.wildcard_policy == "strict"


def test_grid_is_deduplicating_by_axis():
    base = _fee_rules_candidate()
    dup = base.model_copy(update={"name": "same_axis_other_name"})
    completed = complete_grid([base, dup])
    axes = [f"{c.population}|{c.axis_summary}" for c in completed]
    assert len(axes) == len(set(axes))


def test_grid_completes_payments_primitive_axes():
    candidate = InterpretationSpec(
        name="teacher_steer", population="payments",
        payments=PaymentsSpec(primitive="steer_optimal_aci", aci_candidate_policy="exclude_current"),
        output=OutputSpec(kind="single_string"), manual_citation="manual §5",
    )
    names = {c.name for c in complete_grid([candidate])}
    assert any("grid_include_all" in n for n in names)


def test_manual_excerpts_are_keyword_scored(tmp_path):
    (tmp_path / "manual.md").write_text(
        "Irrelevant intro paragraph about nothing in particular, padded to length.\n\n"
        "The fee is computed as fixed_amount plus rate times the transaction value divided by 10000; "
        "card scheme fees vary by transaction value in this fee schedule.\n\n"
        "Contact support for details unrelated to anything else whatsoever here.",
    )
    excerpts = manual_excerpts_for_template(
        "average fee that the card scheme <SCHEME> would charge for a transaction value of <N> EUR",
        tmp_path,
    )
    assert excerpts and "fixed_amount" in excerpts[0]


def test_teacher_prompt_contains_template_instances_and_excerpts():
    prompt = build_teacher_prompt(
        template="total fees for <MERCHANT> in <YEAR>",
        instance_questions=["q1", "q2", "q3", "q4"],
        manual_excerpts=["excerpt-one"],
    )
    assert "total fees for <MERCHANT>" in prompt
    assert "q3" in prompt and "q4" not in prompt
    assert "excerpt-one" in prompt


def test_propose_candidates_times_out_on_hung_teacher(tmp_path, monkeypatch):
    import asyncio

    import pytest

    from dabstep_agent_pydantic.distill import hypotheses

    class HungAgent:
        async def run(self, prompt):
            await asyncio.sleep(3600)

    monkeypatch.setattr(hypotheses, "TEACHER_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(hypotheses.propose_candidates(
            template="test <N>", instance_questions=["test 1"],
            docs_dir=tmp_path, agent=HungAgent(),
        ))


def test_propose_candidates_abandons_and_recovers_on_retry(tmp_path, monkeypatch):
    """First attempt hangs on a poisoned connection; the abandoned task must
    never be awaited for cancellation (that's the deadlock), and the retry
    must get a fresh attempt that can succeed."""
    import asyncio

    from dabstep_agent_pydantic.distill import hypotheses
    from dabstep_agent_pydantic.distill.hypotheses import TeacherProposal

    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        async def run(self, prompt):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(3600)  # simulates a stuck pooled connection
            class _R:
                output = TeacherProposal(candidates=[])
            return _R()

    monkeypatch.setattr(hypotheses, "TEACHER_TIMEOUT_SECONDS", 0.05)
    agent = FlakyAgent()
    result = asyncio.run(hypotheses.propose_candidates(
        template="test <N>", instance_questions=["test 1"],
        docs_dir=tmp_path, agent=agent,
    ))
    assert result == []  # empty candidates, but no hang and no crash
    assert agent.calls == 2


def test_core_semantics_excerpts_are_pinned(tmp_path):
    """The fee formula and wildcard paragraphs must reach the teacher even
    when the template shares no keywords with them."""
    from dabstep_agent_pydantic.distill.hypotheses import manual_excerpts_for_template

    (tmp_path / "manual.md").write_text(
        "Payment processing overview paragraph that is long enough to pass filters.\n\n"
        "The fee is computed as fee = fixed_amount + rate * transaction_value / 10000 "
        "for every matching rule in the fee schedule.\n\n"
        "A null or empty list field means the rule applies to all possible values "
        "of that fee dimension (wildcard semantics).\n\n"
        "Contact support for further questions about your account setup process."
    )
    excerpts = manual_excerpts_for_template(
        "What is the most expensive MCC for a transaction of <N> euros?", tmp_path)
    joined = "\n".join(excerpts)
    assert "10000" in joined          # fee formula pinned
    assert "applies to all" in joined  # wildcard semantics pinned
