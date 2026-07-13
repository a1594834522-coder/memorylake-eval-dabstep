import asyncio
import json

from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.runner import run_benchmark
from dabstep_agent_pydantic.semantic_workflow import SemanticMode
from dabstep_agent_pydantic.semantic_workflow import SemanticWorkflowError


def test_run_benchmark_resume_skips_completed_tasks(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "t1", "question": "Question 1", "guidelines": "Return a number."},
                {"task_id": "t2", "question": "Question 2", "guidelines": "Return a number."},
            ]
        )
    )
    output_path.write_text(json.dumps({"task_id": "t1", "agent_answer": "42"}) + "\n")
    calls = []

    async def fake_solve_task(task, **kwargs):
        calls.append(task.task_id)
        return {
            "task_id": task.task_id,
            "agent_answer": "24",
            "reasoning": "computed",
            "used_code": True,
        }

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    records = asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(),
            resume=True,
        )
    )

    assert calls == ["t2"]
    assert [record["task_id"] for record in records] == ["t2"]
    output_records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["task_id"] for record in output_records] == ["t1", "t2"]


def test_run_benchmark_resume_skips_successful_empty_answers(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(json.dumps([{"task_id": "t1", "question": "Question", "guidelines": "Return empty."}]))
    output_path.write_text(json.dumps({"task_id": "t1", "agent_answer": ""}) + "\n")
    calls = []

    async def fake_solve_task(task, **kwargs):
        calls.append(task.task_id)
        return {
            "task_id": task.task_id,
            "agent_answer": "unexpected",
            "reasoning": "computed",
            "used_code": True,
        }

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    records = asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(),
            resume=True,
        )
    )

    assert calls == []
    assert records == []


def test_run_benchmark_retries_failed_task_once(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(json.dumps([{"task_id": "t1", "question": "Question", "guidelines": "Return a number."}]))
    calls = []

    async def fake_solve_task(task, **kwargs):
        calls.append(task.task_id)
        if len(calls) == 1:
            raise RuntimeError("temporary model timeout")
        return {
            "task_id": task.task_id,
            "agent_answer": "24",
            "reasoning": "computed",
            "used_code": True,
        }

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    records = asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(),
            task_retries=1,
            retry_delay_seconds=0,
        )
    )

    assert calls == ["t1", "t1"]
    assert records[0]["agent_answer"] == "24"
    assert "error" not in records[0]


def test_run_benchmark_continues_when_memory_retrieval_fails(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(json.dumps([{"task_id": "t1", "question": "Question", "guidelines": "Return a number."}]))
    calls = []

    class FailingMemoryClient:
        def search_project(self, *args, **kwargs):
            raise TimeoutError("retrieval timed out")

    async def fake_solve_task(task, **kwargs):
        calls.append((task.task_id, kwargs["memory_context"]))
        return {
            "task_id": task.task_id,
            "agent_answer": "24",
            "reasoning": "computed without memory",
            "used_code": True,
        }

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    records = asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(
                run_mode=RunMode.MEMORY_ASSISTED,
                memory_enabled=True,
                project_id="project",
                user_id="user",
            ),
            memory_client=FailingMemoryClient(),
        )
    )

    assert calls == [("t1", None)]
    assert records[0]["agent_answer"] == "24"
    assert "error" not in records[0]
    decisions = records[0]["memory_trace"]["policy_decisions"]
    assert decisions[0] == {
        "action": "search_memory",
        "allowed": False,
        "reason": "retrieval failed: TimeoutError",
    }


def test_run_benchmark_forwards_semantic_mode(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(
        json.dumps([{"task_id": "t1", "question": "Question", "guidelines": "Return a number."}])
    )
    captured = []

    async def fake_solve_task(task, **kwargs):
        captured.append(kwargs["semantic_mode"])
        return {
            "task_id": task.task_id,
            "agent_answer": "24",
            "reasoning": "computed",
            "used_code": True,
        }

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(),
            semantic_mode=SemanticMode.SHADOW,
        )
    )

    assert captured == [SemanticMode.SHADOW]


def test_run_benchmark_preserves_strict_semantic_failure_trace(monkeypatch, tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.jsonl"
    workspace_dir = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path.write_text(
        json.dumps([{"task_id": "t1", "question": "Question", "guidelines": "Return a number."}])
    )
    trace = {
        "mode": "strict",
        "plan": None,
        "candidates": None,
        "executions": {},
        "verifier": None,
        "usage": {},
        "fallback_reason": "certified policy required",
        "selected_path": "strict_failure",
    }

    async def fake_solve_task(task, **kwargs):
        del task, kwargs
        raise SemanticWorkflowError("certified policy required", trace=trace)

    monkeypatch.setattr("dabstep_agent_pydantic.runner.solve_task", fake_solve_task)

    records = asyncio.run(
        run_benchmark(
            input_path=input_path,
            data_dir=data_dir,
            output_path=output_path,
            workspace_dir=workspace_dir,
            memory_config=MemoryLakeConfig(),
            semantic_mode=SemanticMode.STRICT,
        )
    )

    assert records[0]["error"]["type"] == "SemanticWorkflowError"
    assert records[0]["semantic_trace"] == trace
