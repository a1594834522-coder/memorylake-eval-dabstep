"""Audit: learned skills must not silently contradict the calibration set.

This is the safety net for exactly the failure mode discovered running the
first full-scale self-reference learn: single-shot self-reference and a
seed-grid candidate can share the same systematic misunderstanding (e.g.
"most expensive" read as reducer=max instead of manual-mandated reducer=mean),
so discrimination sees high agreement while the answer is wrong. Reference
agreement alone cannot catch a correlated model bias; only an independent
oracle can. The 14 hand-written calibration routes are that oracle for the
templates they cover.

The comparison logic lives in scripts/skill_audit.py (shared with the
post-learn --remove CLI); this test exercises it on a real learn output.

Requires real context data, a sanitized task file, and a learn() output
directory; skipped otherwise:
  DABSTEP_CONTEXT_DIR=<context dir> DABSTEP_TASKS_PATH=<tasks.json> \\
  DABSTEP_SKILLS_DIR=<learn output dir> pytest tests/test_distill_skill_audit.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from skill_audit import audit_skills  # noqa: E402

CONTEXT_DIR = os.environ.get("DABSTEP_CONTEXT_DIR")
TASKS_PATH = os.environ.get("DABSTEP_TASKS_PATH")
SKILLS_DIR = os.environ.get("DABSTEP_SKILLS_DIR")

pytestmark = pytest.mark.skipif(
    not (CONTEXT_DIR and TASKS_PATH and SKILLS_DIR
         and Path(CONTEXT_DIR).exists() and Path(TASKS_PATH).exists() and Path(SKILLS_DIR).exists()),
    reason="requires DABSTEP_CONTEXT_DIR, DABSTEP_TASKS_PATH, DABSTEP_SKILLS_DIR",
)


def test_learned_skills_agree_with_calibration_on_covered_routes():
    result = audit_skills(
        skills_dir=Path(SKILLS_DIR), context_dir=Path(CONTEXT_DIR), tasks_path=Path(TASKS_PATH),
    )
    assert result.checked > 0, "no calibration-covered instances matched a learned skill"
    assert not result.mismatches, (
        f"{len(result.mismatches)}/{result.checked} learned skills contradict the calibration "
        f"oracle (same-model reference/candidate collusion — do not ship):\n"
        + "\n".join(
            f"  route={m.route_id} skill={m.skill_id} task={m.task_id}: "
            f"learned={m.learned!r} calibration={m.expected!r}"
            for m in result.mismatches
        )
    )
