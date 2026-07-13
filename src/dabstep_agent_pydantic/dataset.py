from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    question: str
    guidelines: str | None = None
    level: str | None = None


def load_tasks(path: Path) -> list[Task]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected task file to contain a list: {path}")

    tasks: list[Task] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Task at index {index} is not an object")
        task_id = str(item.get("task_id", index))
        task_data = {**item, "task_id": task_id}
        tasks.append(Task(**task_data))
    return tasks


def filter_tasks(tasks: list[Task], task_ids: str | None) -> list[Task]:
    if not task_ids:
        return tasks
    wanted = {part.strip() for part in task_ids.split(",") if part.strip()}
    return [task for task in tasks if task.task_id in wanted]


def load_hf_tasks(split: str = "default", dataset_name: str = "adyen/DABstep") -> list[Task]:
    if split not in {"default", "dev"}:
        raise ValueError("DABStep tasks split must be 'default' or 'dev'")
    rows = _load_hf_dataset(dataset_name, "tasks", split)
    return [Task(**dict(row)) for row in rows]


def _load_hf_dataset(dataset_name: str, config_name: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, config_name, split=split)


def summarize_context_dir(path: Path) -> str:
    lines = [f"Data directory: {path.resolve()}"]
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() == ".csv":
            lines.append(_summarize_csv(file_path))
        elif file_path.suffix.lower() == ".json":
            lines.append(_summarize_json(file_path))
    return "\n".join(lines)


def _summarize_csv(path: Path) -> str:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    columns = ", ".join(header[:12])
    extra = f" (+{len(header) - 12} more)" if len(header) > 12 else ""
    return f"- {path.name} (CSV): columns: {columns}{extra}"


def _summarize_json(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(raw, list):
        sample = raw[0] if raw else {}
        shape = "JSON list"
    elif isinstance(raw, dict):
        sample = raw
        shape = "JSON object"
    else:
        sample = {}
        shape = f"JSON {type(raw).__name__}"

    keys = list(sample.keys()) if isinstance(sample, dict) else []
    key_text = ", ".join(str(key) for key in keys[:12])
    extra = f" (+{len(keys) - 12} more)" if len(keys) > 12 else ""
    return f"- {path.name} ({shape}): keys: {key_text}{extra}"
