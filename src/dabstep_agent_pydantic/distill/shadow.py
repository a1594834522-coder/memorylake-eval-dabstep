"""Shadow integration of generated skills into the solving path.

Modes via ``DABSTEP_GENERATED_SKILLS``:
- ``off`` (default): generated skills are not consulted.
- ``shadow``: generated skills run alongside the existing solver; the outcome
  is recorded in the trace and never changes the answer.
- ``primary``: a matching generated skill answers the task; the hand-written
  deterministic route is the fallback.

Artifacts directory via ``DABSTEP_GENERATED_SKILLS_DIR``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dabstep_agent_pydantic.distill.emit import (
    CalibrationNote,
    GeneratedSkill,
    load_calibration_notes,
    load_generated_skills,
)

_MODES = ("off", "primary")
_cache_lock = threading.Lock()
_cache: dict[str, list[GeneratedSkill]] = {}
_note_cache: dict[str, list[CalibrationNote]] = {}


def generated_skills_mode() -> str:
    mode = os.getenv("DABSTEP_GENERATED_SKILLS", "primary").strip().lower()
    return mode if mode in _MODES else "primary"


def _skills() -> list[GeneratedSkill]:
    directory = os.getenv("DABSTEP_GENERATED_SKILLS_DIR", "artifacts/skills").strip()
    if not directory:
        return []
    with _cache_lock:
        if directory not in _cache:
            path = Path(directory)
            _cache[directory] = load_generated_skills(path) if path.exists() else []
        return _cache[directory]


def clear_skill_cache() -> None:
    with _cache_lock:
        _cache.clear()
        _note_cache.clear()
        _fingerprint_cache.clear()
        _warned.clear()


def _notes() -> list[CalibrationNote]:
    directory = os.getenv("DABSTEP_GENERATED_SKILLS_DIR", "artifacts/skills").strip()
    if not directory:
        return []
    with _cache_lock:
        if directory not in _note_cache:
            path = Path(directory)
            _note_cache[directory] = load_calibration_notes(path) if path.exists() else []
        return _note_cache[directory]


def calibration_note_for(question: str) -> str | None:
    """The schema-level convention for the first note whose signature matches,
    or None. Consulted on the LLM path only when no skill answered."""
    if generated_skills_mode() == "off":
        return None
    for note in _notes():
        if note.match(question):
            return note.convention
    return None


@dataclass(frozen=True)
class GeneratedAnswer:
    skill_id: str
    agent_answer: str
    primitive: str | None = None


_fingerprint_cache: dict[str, dict[str, str]] = {}
_warned: set[str] = set()


def _current_doc_fingerprints(data_dir) -> dict[str, str]:
    import hashlib
    from pathlib import Path

    key = str(data_dir)
    with _cache_lock:
        if key not in _fingerprint_cache:
            _fingerprint_cache[key] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(data_dir).glob("*.md"))
            }
        return _fingerprint_cache[key]


def stale_documents(skill, data_dir) -> list[str]:
    """Knowledge documents that changed since this skill was learned.

    A skill's interpretation was discriminated against a specific manual
    version (pinned by sha256 in its provenance); executing it against a
    different version silently shifts meaning. Non-empty result means the
    skill must not answer deterministically."""
    learned = skill.doc_fingerprints()
    if not learned:
        return []  # pre-fingerprint artifact: nothing to compare against
    current = _current_doc_fingerprints(data_dir)
    return sorted(name for name, digest in learned.items()
                  if name in current and current[name] != digest)


def _output_shape_violation(skill, answer: str) -> str | None:
    """Cheap per-answer invariant from the spec's output contract. A learned
    spec that stops producing its own declared shape signals artifact or
    mechanism drift; the answer must fall back to the LLM path."""
    if not answer.strip():
        return "empty answer"
    kind = skill.spec.output.kind
    try:
        if kind == "decimal":
            float(answer.replace("%", "").strip())
        elif kind == "integer":
            int(answer.strip())
        elif kind == "id_list":
            tokens = [t.strip() for t in answer.split(",")]
            if not all(t.lstrip("-").isdigit() for t in tokens if t):
                return "id_list contains a non-integer token"
    except ValueError:
        return f"answer does not parse as {kind}"
    return None


def _warn_once(skill_id: str, message: str) -> None:
    if skill_id not in _warned:
        _warned.add(skill_id)
        print(message, flush=True)


def try_solve_generated(question: str, guidelines: str, data, data_dir=None) -> GeneratedAnswer | None:
    for skill in _skills():
        if skill.match(question) is None:
            continue
        if data_dir is not None:
            stale = stale_documents(skill, data_dir)
            if stale:
                _warn_once(skill.skill_id,
                           f"[skills] {skill.skill_id} skipped: knowledge docs changed "
                           f"since learning ({', '.join(stale)}); re-learn or re-audit")
                return None
        try:
            answer = skill.solve(data, question, guidelines)
        except Exception:  # noqa: BLE001 - a broken artifact must never break solving.
            return None
        if answer is None:
            return None
        violation = _output_shape_violation(skill, answer)
        if violation is not None:
            _warn_once(skill.skill_id,
                       f"[skills] {skill.skill_id} answer rejected ({violation}); "
                       f"falling back to the LLM path")
            return None
        payments = skill.spec.payments
        return GeneratedAnswer(
            skill_id=skill.skill_id,
            agent_answer=answer,
            primitive=payments.primitive if payments is not None else None,
        )
    return None


def shadow_trace(
    question: str,
    guidelines: str,
    data,
    *,
    route_answer: str | None,
    route_id: str | None,
) -> dict[str, Any] | None:
    """Shadow-mode comparison record; returns None when no skill matches."""
    generated = try_solve_generated(question, guidelines, data)
    if generated is None:
        return None
    return {
        "skill_id": generated.skill_id,
        "route_id": route_id,
        "agrees_with_route": (
            generated.agent_answer == route_answer if route_answer is not None else None
        ),
    }
