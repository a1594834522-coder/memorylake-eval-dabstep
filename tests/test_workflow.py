import asyncio

from dabstep_agent_pydantic.agent import DABStepAnswer
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.planning import PlanDecision
from dabstep_agent_pydantic.semantic_workflow import SemanticMode
from dabstep_agent_pydantic.workflow import build_task_prompt
from dabstep_agent_pydantic.workflow import run_task_workflow


def test_workflow_records_native_graph_stages(tmp_path):
    async def solve_hook(*args, **kwargs):
        return {
            "task_id": "t1",
            "agent_answer": "42",
            "reasoning": "computed",
            "used_code": True,
        }

    task = Task(task_id="t1", question="What total fee did merchant A pay?", guidelines="Return a number.")

    record = asyncio.run(
        run_task_workflow(
            task,
            data_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            file_summary="Data directory: empty",
            assets_dir=None,
            memory_context=None,
            solve_hook=solve_hook,
        )
    )

    assert record["agent_answer"] == "42"
    assert record["workflow_trace"]["stages"] == [
        "plan",
        "solve",
        "verify",
        "finalize",
    ]
    assert record["workflow_trace"]["plan"]["toolset_ids"]


def test_workflow_retries_once_when_verifier_requests_retry(tmp_path):
    calls = []

    async def solve_hook(*args, **kwargs):
        calls.append(kwargs["feedback"])
        return {
            "task_id": "t1",
            "agent_answer": "bad" if len(calls) == 1 else "42",
            "reasoning": "computed",
            "used_code": True,
        }

    task = Task(task_id="t1", question="What total fee did merchant A pay?", guidelines="Return a number.")

    record = asyncio.run(
        run_task_workflow(
            task,
            data_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            file_summary="Data directory: empty",
            assets_dir=None,
            memory_context=None,
            solve_hook=solve_hook,
            verify_hook=lambda record, state: "retry with numeric answer" if record["agent_answer"] == "bad" else None,
        )
    )

    assert calls == [None, "retry with numeric answer"]
    assert record["agent_answer"] == "42"
    assert record["workflow_trace"]["solver_attempts"] == 2


def test_default_solve_disables_pydantic_ai_request_limit(monkeypatch, tmp_path):
    captured = {}

    class FakeResult:
        output = DABStepAnswer(agent_answer="42", reasoning="computed", used_code=False)

    class FakeAgent:
        async def run(self, *args, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "off")
    monkeypatch.setattr("dabstep_agent_pydantic.workflow.create_agent", lambda: FakeAgent())

    task = Task(task_id="t1", question="What total fee did merchant A pay?", guidelines="Return a number.")

    record = asyncio.run(
        run_task_workflow(
            task,
            data_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            file_summary="Data directory: empty",
            assets_dir=None,
            memory_context=None,
        )
    )

    assert record["agent_answer"] == "42"
    assert captured["usage_limits"].request_limit is None


def test_task_prompt_injects_ambiguity_axis_instructions_only_when_needed():
    task = Task(task_id="t1", question="What is the average fee?", guidelines="Return a number.")
    plain = build_task_prompt(task)
    plan = PlanDecision(
        task_family="fee_analysis",
        ambiguity_axes=["average_fee_basis"],
    )
    with_axes = build_task_prompt(task, plan=plan)

    assert "AMBIGUITY AXES" not in plain
    assert "AMBIGUITY AXES" in with_axes
    assert "average_fee_basis" in with_axes
    assert "question wording" in with_axes
    assert "reasoning" in with_axes


def test_workflow_dispatches_nonlegacy_modes_to_semantic_runtime(monkeypatch, tmp_path):
    captured = {}

    async def fake_semantic_workflow(**kwargs):
        captured.update(kwargs)
        return {"task_id": "t1", "agent_answer": "semantic"}

    monkeypatch.setattr(
        "dabstep_agent_pydantic.workflow.run_semantic_workflow",
        fake_semantic_workflow,
    )
    task = Task(task_id="t1", question="How many payments?", guidelines="Return an integer.")

    record = asyncio.run(
        run_task_workflow(
            task,
            data_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            file_summary="payments table",
            semantic_mode=SemanticMode.SHADOW,
        )
    )

    assert record["agent_answer"] == "semantic"
    assert captured["mode"] is SemanticMode.SHADOW
    assert callable(captured["legacy_runner"])
