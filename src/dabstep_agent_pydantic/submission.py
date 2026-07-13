from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_FIELDS = ("task_id", "agent_answer")


def export_submission(input_path: Path, output_path: Path) -> dict[str, Any]:
    records = _load_jsonl(input_path)
    task_order: list[str] = []
    latest_success: dict[str, dict[str, str]] = {}
    skipped_count = 0
    duplicate_count = 0

    for line_number, record in enumerate(records, 1):
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"line {line_number}: missing task_id")
        if task_id not in task_order:
            task_order.append(task_id)

        answer = record.get("agent_answer")
        if answer is None:
            raise ValueError(f"line {line_number}: missing agent_answer")
        answer_text = str(answer).strip()
        if record.get("error"):
            skipped_count += 1
            continue

        if task_id in latest_success:
            duplicate_count += 1
        latest_success[task_id] = {"task_id": task_id, "agent_answer": answer_text}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task_id in task_order:
            record = latest_success.get(task_id)
            if record is None:
                continue
            handle.write(json.dumps({field: record[field] for field in ALLOWED_FIELDS}, ensure_ascii=False) + "\n")
    return {
        "ok": True,
        "path": str(output_path),
        "line_count": len(latest_success),
        "input_line_count": len(records),
        "skipped_count": skipped_count,
        "duplicate_success_count": duplicate_count,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        rows.append(item)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DABStep submission JSONL")
    parser.add_argument("--input", required=True, type=Path, help="Agent runtime JSONL output")
    parser.add_argument("--output", required=True, type=Path, help="Submission JSONL output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_submission(args.input, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
