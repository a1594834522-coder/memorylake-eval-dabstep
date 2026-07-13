import json

from dabstep_agent_pydantic.submission import export_submission


def test_export_submission_keeps_latest_successful_answer_per_task(tmp_path):
    input_path = tmp_path / "runtime.jsonl"
    output_path = tmp_path / "submission.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "alpha", "agent_answer": "", "error": "timeout"}),
                json.dumps({"task_id": "beta", "agent_answer": "first"}),
                json.dumps({"task_id": "alpha", "agent_answer": "recovered"}),
                json.dumps({"task_id": "beta", "agent_answer": "second"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = export_submission(input_path, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report["line_count"] == 2
    assert rows == [
        {"task_id": "alpha", "agent_answer": "recovered"},
        {"task_id": "beta", "agent_answer": "second"},
    ]


def test_export_submission_keeps_successful_empty_answers(tmp_path):
    input_path = tmp_path / "runtime.jsonl"
    output_path = tmp_path / "submission.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "empty-list", "agent_answer": ""}) + "\n",
        encoding="utf-8",
    )

    report = export_submission(input_path, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report["line_count"] == 1
    assert rows == [{"task_id": "empty-list", "agent_answer": ""}]
