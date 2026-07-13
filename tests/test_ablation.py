import json

from dabstep_agent_pydantic.ablation import answers_match
from dabstep_agent_pydantic.ablation import compare_runs
from dabstep_agent_pydantic.ablation import sample_task_ids


def test_answers_match_semantics():
    assert answers_match("42.00", "42.0")
    assert answers_match(" yes ", "Yes")
    assert answers_match("", "")
    assert not answers_match("42.00", "43.00")
    assert not answers_match("A, B", "A,B ,C")


def test_sample_task_ids_is_stratified_and_deterministic(tmp_path):
    tasks = []
    for index in range(30):
        tasks.append({"task_id": str(index), "question": f"What were the total fees that a merchant paid in 2023? v{index}", "guidelines": "Rounded to 2 decimals."})
    for index in range(30, 60):
        tasks.append({"task_id": str(index), "question": f"What is the fraud rate for repeat customers? v{index}", "guidelines": "Rounded to 3 decimals."})
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps(tasks), encoding="utf-8")

    first = sample_task_ids(input_path=input_path, per_family=5, seed=7)
    second = sample_task_ids(input_path=input_path, per_family=5, seed=7)
    different_seed = sample_task_ids(input_path=input_path, per_family=5, seed=8)

    assert first == second
    assert first != different_seed
    assert len(first) >= 2
    assert all(len(ids) == 5 for ids in first.values())


def _write_run(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_runs_reports_agreement_and_accuracy(tmp_path):
    run_a = tmp_path / "clean.jsonl"
    run_b = tmp_path / "memory.jsonl"
    reference = tmp_path / "reference.jsonl"
    _write_run(
        run_a,
        [
            {"task_id": "1", "agent_answer": "42.00", "elapsed_seconds": 10,
             "workflow_trace": {"solver_attempts": 1}, "memory_trace": {
                "retrieved_count": 0,
                "document_retrieved_count": 0,
                "analysis_plan": {"task_family": "fee_matching"},
                "document_chunks_dropped": 0,
             }},
            {"task_id": "2", "agent_answer": "wrong", "elapsed_seconds": 30,
             "workflow_trace": {"solver_attempts": 2}, "memory_trace": {
                "retrieved_count": 0,
                "document_retrieved_count": 1,
                "retrieved_documents": [{"document_name": "manual.md"}],
                "analysis_plan": {"task_family": "customer_fraud_metrics"},
                "policy_decisions": [
                    {"action": "search_memory", "allowed": False, "reason": "retrieval failed: TimeoutError"}
                ],
                "document_context_truncated": True,
                "document_chunks_dropped": 3,
             }},
            {"task_id": "3", "agent_answer": "", "error": {"type": "Timeout"}},
        ],
    )
    _write_run(
        run_b,
        [
            {"task_id": "1", "agent_answer": "42.0", "elapsed_seconds": 8,
             "workflow_trace": {"solver_attempts": 1}, "memory_trace": {
                "retrieved_count": 3,
                "document_retrieved_count": 2,
                "retrieved_documents": [
                    {"document_name": "curriculum_rules.md"},
                    {"document_name": "curriculum_fee_matching.md"},
                    {"document_name": "manual.md"},
                ],
                "analysis_plan": {"task_family": "fee_matching"},
                "document_chunks_dropped": 1,
             }},
            {"task_id": "2", "agent_answer": "right", "elapsed_seconds": 12,
             "workflow_trace": {"solver_attempts": 1}, "memory_trace": {
                "retrieved_count": 4,
                "document_retrieved_count": 0,
                "retrieved_documents": [],
                "analysis_plan": {"task_family": "customer_fraud_metrics"},
                "document_chunks_dropped": 0,
             }},
        ],
    )
    _write_run(reference, [{"task_id": "1", "answer": "42.00"}, {"task_id": "2", "answer": "right"}])

    report = compare_runs(
        run_a=run_a,
        run_b=run_b,
        label_a="clean",
        label_b="memory",
        reference_path=reference,
    )

    assert report["shared_tasks"] == 2
    assert report["agreement"] == 1
    assert report["disagreements"][0]["task_id"] == "2"
    assert report["disagreements"][0]["reference_match"] == {"clean": False, "memory": True}
    assert report["accuracy"]["clean"]["correct"] == 1
    assert report["accuracy"]["memory"]["correct"] == 2
    assert report["clean"]["verifier_retry_count"] == 1
    assert report["memory"]["avg_memories_retrieved"] == 3.5
    assert report["clean"]["document_hit_rate"] == 0.5
    assert report["clean"]["curriculum_rules_hit_rate"] == 0.0
    assert report["clean"]["retrieval_failure_count"] == 1
    assert report["clean"]["document_context_truncated_count"] == 1
    assert report["clean"]["family_aligned_hit_rate"] == 0.0
    assert report["clean"]["avg_document_chunks_dropped"] == 1.5
    assert report["memory"]["document_hit_rate"] == 0.5
    assert report["memory"]["curriculum_rules_hit_rate"] == 0.5
    assert report["memory"]["retrieval_failure_count"] == 0
    assert report["memory"]["document_context_truncated_count"] == 0
    assert report["memory"]["family_aligned_hit_rate"] == 0.5
    assert report["memory"]["avg_document_chunks_dropped"] == 0.5
    # Error records never count as run results.
    assert report["only_in"] == {"clean": [], "memory": []}
