
import pytest
def test_note_artifact_roundtrip_and_convention(tmp_path):
    from dabstep_agent_pydantic.distill.emit import (
        write_note_artifact, load_calibration_notes, template_note_id)
    from dabstep_agent_pydantic.distill.spec import (
        InterpretationSpec, FeeRulesSpec, OutputSpec)

    template = "What is the fee ID or IDs that apply to account_type = H and aci = A?"
    spec = InterpretationSpec(
        name="wildcard_mean", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], reducer="mean",
                               wildcard_policy="manual"),
        output=OutputSpec(kind="decimal"), manual_citation="fee = fixed + rate*x/10000")
    path = write_note_artifact(out_dir=tmp_path, template=template, spec=spec,
                               evidence={"discrimination": {}},
                               provenance={"basis": "leading candidate below participation gate"})
    assert path.name == f"{template_note_id(template)}.json"
    notes = load_calibration_notes(tmp_path)
    assert len(notes) == 1
    assert "average the per-rule fee" in notes[0].convention
    assert notes[0].match("What is the fee ID or IDs that apply to account_type = H and aci = A?")


def test_note_artifact_rejects_leaked_answer(tmp_path):
    from dabstep_agent_pydantic.distill.emit import write_note_artifact
    from dabstep_agent_pydantic.distill.spec import (
        InterpretationSpec, FeeRulesSpec, OutputSpec)
    spec = InterpretationSpec(name="m", population="fee_rules",
        fee_rules=FeeRulesSpec(reducer="mean"), output=OutputSpec(kind="decimal"),
        manual_citation="cite")
    with pytest.raises(ValueError):
        write_note_artifact(out_dir=tmp_path, template="t", spec=spec,
                            evidence={"agent_answer": "1, 2"}, provenance={})


def test_calibration_note_injected_into_llm_prompt(tmp_path, monkeypatch):
    from dabstep_agent_pydantic.distill.emit import write_note_artifact
    from dabstep_agent_pydantic.distill.spec import (
        InterpretationSpec, FeeRulesSpec, OutputSpec)
    from dabstep_agent_pydantic.distill import shadow
    from dabstep_agent_pydantic.workflow import build_task_prompt
    from dabstep_agent_pydantic.dataset import Task

    q = "What is the fee ID or IDs that apply to account_type = H and aci = A?"
    spec = InterpretationSpec(name="wm", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], reducer="mean"),
        output=OutputSpec(kind="decimal"), manual_citation="fee = fixed + rate*x/10000")
    write_note_artifact(out_dir=tmp_path, template=q, spec=spec, evidence={}, provenance={})
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "primary")
    shadow.clear_skill_cache()

    task = Task(task_id="t1", question=q, guidelines="", level="easy")
    prompt = build_task_prompt(task)
    assert "INTERPRETATION NOTE" in prompt
    assert "average the per-rule fee" in prompt
    # A non-matching question gets no note.
    other = Task(task_id="t2", question="How many unique merchants are there?",
                 guidelines="", level="easy")
    assert "INTERPRETATION NOTE" not in build_task_prompt(other)
    shadow.clear_skill_cache()
