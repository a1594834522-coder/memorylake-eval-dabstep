"""Calibration: compiled adopted-specs must reproduce hand-written routes byte-for-byte.

Requires real context data and the sanitized task file; skipped otherwise:
  DABSTEP_CONTEXT_DIR=<context dir> DABSTEP_TASKS_PATH=<tasks.json> pytest tests/test_distill_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibration_specs import CALIBRATION  # noqa: E402

from dabstep_agent_pydantic.dabstep_core import load_dabstep_data  # noqa: E402
from dabstep_agent_pydantic.dataset import Task  # noqa: E402
from calibration_solver import try_solve_deterministic  # noqa: E402
from dabstep_agent_pydantic.distill.combinators import compile_spec  # noqa: E402

CONTEXT_DIR = os.environ.get("DABSTEP_CONTEXT_DIR")
TASKS_PATH = os.environ.get("DABSTEP_TASKS_PATH")

pytestmark = pytest.mark.skipif(
    not (CONTEXT_DIR and TASKS_PATH and Path(CONTEXT_DIR).exists() and Path(TASKS_PATH).exists()),
    reason="requires DABSTEP_CONTEXT_DIR and DABSTEP_TASKS_PATH",
)

MAX_INSTANCES_PER_ROUTE = 3


def _route_instances():
    from calibration_routes import match_route

    tasks = json.loads(Path(TASKS_PATH).read_text())
    by_route: dict[str, list[dict]] = {}
    for task in tasks:
        route = match_route(" ".join(str(task["question"]).split()))
        if route:
            by_route.setdefault(route, []).append(task)
    return by_route


def test_calibration_specs_match_handwritten_routes():
    data = load_dabstep_data(Path(CONTEXT_DIR))
    by_route = _route_instances()
    checked = mismatches = 0
    report: list[str] = []
    for route_id, (spec, extract) in CALIBRATION.items():
        instances = by_route.get(route_id, [])[:MAX_INSTANCES_PER_ROUTE]
        assert instances, f"no instances found for calibration route {route_id}"
        fn = compile_spec(spec)
        for task in instances:
            question = " ".join(str(task["question"]).split())
            guidelines = str(task.get("guidelines") or "")
            params = extract(question, data)
            expected = try_solve_deterministic(
                Task(task_id=str(task["task_id"]), question=question, guidelines=guidelines),
                data_dir=Path(CONTEXT_DIR),
            )
            solver_handles = expected is not None and expected.route == route_id
            if params is None and not solver_handles:
                # Template variant outside both the extractor and the hand-written
                # route (match_route patterns are broader than solver triggers).
                continue
            assert params is not None, f"{route_id}: extractor missed solver-handled {question[:60]}"
            assert solver_handles, f"{route_id}: solver route mismatch on task {task['task_id']}"
            got = fn(data, params, guidelines)
            checked += 1
            if got != expected.agent_answer:
                mismatches += 1
                report.append(f"{route_id} task={task['task_id']}: spec={got[:50]!r} solver={expected.agent_answer[:50]!r}")
    assert mismatches == 0, f"{mismatches}/{checked} mismatches:\n" + "\n".join(report)
    assert checked >= len(CALIBRATION) * 1
