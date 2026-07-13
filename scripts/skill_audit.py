"""Audit learned skills against the hand-written calibration oracle.

Single-shot self-reference (learn --reference-mode full) can adopt a wrong
interpretation when the model that generated the reference and the seed-grid
candidate share the same systematic misreading: discrimination then sees high
self-agreement while the answer is wrong on every instance. Reference
agreement cannot detect a correlated model bias; an independent oracle can.

The 14 hand-written calibration routes are that oracle for the templates they
cover. This module recompiles every learned skill that matches a calibration
route and requires byte-identical output on real instances. It is the single
source of truth shared by tests/test_distill_skill_audit.py (verification) and
the `--remove` CLI (post-learn cleanup).

Internal tooling: depends on the calibration set (tests/), which never ships.

    PYTHONPATH=src python scripts/skill_audit.py \\
        --skills-dir <learn output> --context-dir <ctx> --tasks <tasks.json> [--remove]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from calibration_routes import match_route  # noqa: E402
from calibration_specs import CALIBRATION  # noqa: E402

from dabstep_agent_pydantic.dabstep_core import load_dabstep_data  # noqa: E402
from dabstep_agent_pydantic.distill.combinators import compile_spec  # noqa: E402
from dabstep_agent_pydantic.distill.emit import (  # noqa: E402
    load_calibration_notes,
    load_generated_skills,
)


@dataclass(frozen=True)
class Mismatch:
    skill_id: str
    route_id: str
    task_id: str
    learned: str
    expected: str


@dataclass
class AuditResult:
    checked: int
    mismatches: list[Mismatch]

    @property
    def offending_skill_ids(self) -> set[str]:
        return {m.skill_id for m in self.mismatches}


def audit_skills(
    *,
    skills_dir: Path,
    context_dir: Path,
    tasks_path: Path,
    max_per_route: int = 5,
) -> AuditResult:
    """Compare every learned skill covered by a calibration route against the
    calibration spec on real instances; return byte-level mismatches."""
    data = load_dabstep_data(context_dir)
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    skills = load_generated_skills(skills_dir)

    checked = 0
    seen_per_route: dict[str, int] = {}
    mismatches: list[Mismatch] = []
    for task in tasks:
        question = " ".join(str(task["question"]).split())
        route_id = match_route(question)
        if route_id is None or route_id not in CALIBRATION:
            continue
        if seen_per_route.get(route_id, 0) >= max_per_route:
            continue
        skill = next((s for s in skills if s.signature.parse(question, data) is not None), None)
        if skill is None:
            continue
        cal_spec, cal_extract = CALIBRATION[route_id]
        cal_params = cal_extract(question, data)
        if cal_params is None:
            continue
        guidelines = str(task.get("guidelines") or "")
        expected = compile_spec(cal_spec)(data, cal_params, guidelines)
        got = skill.solve(data, question, guidelines)
        checked += 1
        seen_per_route[route_id] = seen_per_route.get(route_id, 0) + 1
        if got != expected:
            mismatches.append(Mismatch(
                skill_id=skill.skill_id, route_id=route_id, task_id=str(task["task_id"]),
                learned=str(got), expected=str(expected),
            ))
    return AuditResult(checked=checked, mismatches=mismatches)


def audit_notes(
    *, skills_dir: Path, context_dir: Path, tasks_path: Path, max_per_route: int = 5,
) -> set[str]:
    """A calibration note steers the LLM toward an interpretation; a note
    whose spec contradicts the oracle would steer it wrong on covered
    templates, so it earns the same removal discipline as a skill. Returns
    note_ids to remove."""
    data = load_dabstep_data(context_dir)
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    notes = load_calibration_notes(skills_dir)

    seen_per_route: dict[str, int] = {}
    offenders: set[str] = set()
    for task in tasks:
        question = " ".join(str(task["question"]).split())
        route_id = match_route(question)
        if route_id is None or route_id not in CALIBRATION:
            continue
        if seen_per_route.get(route_id, 0) >= max_per_route:
            continue
        note = next((n for n in notes if n.signature.parse(question, data) is not None), None)
        if note is None:
            continue
        cal_spec, cal_extract = CALIBRATION[route_id]
        cal_params = cal_extract(question, data)
        if cal_params is None:
            continue
        guidelines = str(task.get("guidelines") or "")
        expected = compile_spec(cal_spec)(data, cal_params, guidelines)
        note_params = note.signature.parse(question, data)
        got = compile_spec(note.spec)(data, note_params, guidelines) if note_params else None
        seen_per_route[route_id] = seen_per_route.get(route_id, 0) + 1
        if got != expected:
            offenders.add(note.note_id)
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit learned skills against the calibration oracle")
    parser.add_argument("--skills-dir", required=True, type=Path)
    parser.add_argument("--context-dir", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--max-per-route", type=int, default=5)
    parser.add_argument("--remove", action="store_true",
                        help="Delete skill artifacts that contradict the calibration oracle")
    args = parser.parse_args(argv)

    result = audit_skills(
        skills_dir=args.skills_dir, context_dir=args.context_dir,
        tasks_path=args.tasks, max_per_route=args.max_per_route,
    )
    note_offenders = audit_notes(
        skills_dir=args.skills_dir, context_dir=args.context_dir,
        tasks_path=args.tasks, max_per_route=args.max_per_route,
    )
    print(f"checked {result.checked} calibration-covered instances")
    if not result.mismatches and not note_offenders:
        print("clean: every calibration-covered skill and note agrees with the oracle")
        return 0

    offenders = sorted(result.offending_skill_ids)
    if result.mismatches:
        print(f"\n{len(result.mismatches)} mismatches across {len(offenders)} skill(s) "
              f"(same-model reference/candidate collusion — do not ship):")
        for m in result.mismatches:
            print(f"  route={m.route_id} skill={m.skill_id} task={m.task_id}: "
                  f"learned={m.learned!r} calibration={m.expected!r}")
    if note_offenders:
        print(f"\n{len(note_offenders)} note(s) contradict the oracle (would misdirect the LLM):")
        for note_id in sorted(note_offenders):
            print(f"  note={note_id}")

    if args.remove:
        for skill_id in offenders:
            path = args.skills_dir / f"{skill_id}.json"
            if path.exists():
                path.unlink()
                print(f"removed {path.name}")
        for note_id in sorted(note_offenders):
            path = args.skills_dir / f"{note_id}.json"
            if path.exists():
                path.unlink()
                print(f"removed {path.name}")
    else:
        print("\nre-run with --remove to delete the offending artifacts")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
