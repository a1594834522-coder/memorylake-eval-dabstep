"""Skill artifacts: spec-as-code persistence and runtime loading.

An emitted skill is a JSON artifact containing the template, the adopted
spec, and its evidence summary (agreement statistics — never per-task
answers). At runtime the artifact is recompiled on the fly by
`compile_spec`; no generated Python code ever exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.combinators import CandidateFn, compile_spec
from dabstep_agent_pydantic.distill.signatures import TemplateSignature, compile_signature, parse_raw_param
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec, InterpretationSpec, PaymentsSpec

if TYPE_CHECKING:
    from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec

ARTIFACT_VERSION = 1

_FORBIDDEN_ARTIFACT_KEYS = {"reference_answer", "agent_answer", "answer"}


def template_skill_id(template: str) -> str:
    return "skill_" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class GeneratedSkill:
    skill_id: str
    template: str
    signature: TemplateSignature
    spec: InterpretationSpec
    evidence: dict[str, Any]
    provenance: dict[str, Any] = None  # type: ignore[assignment]

    def doc_fingerprints(self) -> dict[str, str]:
        return dict((self.provenance or {}).get("doc_fingerprints") or {})

    def match(self, question: str) -> dict[str, Any] | None:
        return {"matched": True} if self.signature.regex.search(" ".join(question.split())) else None

    def solve(self, data: DABStepData, question: str, guidelines: str) -> str | None:
        params = self.signature.parse(question, data)
        if params is None:
            return None
        fn: CandidateFn = compile_spec(self.spec)
        return fn(data, params, guidelines)

    def solve_with_params(
        self, data: DABStepData, raw_params: dict[str, Any], guidelines: str,
    ) -> str | None:
        """Execute the spec from structured parameters instead of template
        text. Values go through the same typed parsers as regex captures, so
        this path cannot diverge from textual matching."""
        expected = set(self.signature.group_params)
        unknown = set(raw_params) - expected
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {sorted(unknown)}; expected {sorted(expected)}"
            )
        params: dict[str, Any] = dict(self.signature.constant_params)
        for name, raw in raw_params.items():
            params[name] = parse_raw_param(name, str(raw), data)
        fn: CandidateFn = compile_spec(self.spec)
        return fn(data, params, guidelines)

    def to_analysis_spec(
        self,
        data: DABStepData,
        question: str,
        guidelines: str,
    ) -> "AnalysisSpec | None":
        """Compile a matched generated skill into the AnalysisSpec v2 bridge."""
        params = self.signature.parse(question, data)
        if params is None:
            return None
        from dabstep_agent_pydantic.fee_spec_adapter import adapt_interpretation_spec

        return adapt_interpretation_spec(
            self.spec,
            params=params,
            guidelines=guidelines,
        )


_FEE_REDUCER_PHRASE = {
    "mean": "average the per-rule fee across every matching rule",
    "sum": "sum the per-rule fee across every matching rule",
    "min": "take the smallest per-rule fee among matching rules",
    "max": "take the largest per-rule fee among matching rules",
    "collect_ids": "collect the IDs of all matching rules (do not compute a fee)",
}


def _render_fee_rules(fr: "FeeRulesSpec") -> str:
    lines = [f"Aggregation: {_FEE_REDUCER_PHRASE[fr.reducer]}."]
    if fr.wildcard_policy == "manual":
        lines.append("Rule matching: a null or empty-list rule field is a wildcard that "
                     "matches every value of that dimension (manual convention).")
    else:
        lines.append("Rule matching: a rule matches only by explicit membership.")
    if fr.context_dims:
        lines.append(f"Match on these dimensions from the question: {', '.join(fr.context_dims)}.")
    if fr.group_by:
        lines.append(f"Grouping: group rules by {fr.group_by} (wildcard rules join every group: "
                     f"{fr.group_wildcard_expansion}); reduce within each group, then take the "
                     f"{fr.group_extreme}.")
    return " ".join(lines)


def _render_payments(p: "PaymentsSpec") -> str:
    lines = [f"Primitive: {p.primitive}; reducer: {p.reducer}."]
    for field in ("affected_mode", "aci_candidate_policy", "delta_basis", "tuple_scope"):
        value = getattr(p, field)
        if value:
            lines.append(f"{field.replace('_', ' ')}: {value}.")
    return " ".join(lines)


def render_convention(spec: InterpretationSpec) -> str:
    """Deterministic schema-level guidance rendered from a spec: how this
    metric family should be interpreted (aggregation, matching, grouping) —
    never an answer and never a task-specific value."""
    if spec.population == "fee_rules" and spec.fee_rules is not None:
        return _render_fee_rules(spec.fee_rules)
    if spec.population == "payments" and spec.payments is not None:
        return _render_payments(spec.payments)
    return spec.axis_summary


def write_skill_artifact(
    *,
    out_dir: Path,
    template: str,
    spec: InterpretationSpec,
    evidence: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    artifact = {
        "version": ARTIFACT_VERSION,
        "skill_id": template_skill_id(template),
        "template": template,
        "spec": spec.model_dump(mode="json"),
        "evidence": evidence,
        "provenance": provenance,
    }
    _assert_artifact_safe(artifact)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact['skill_id']}.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_generated_skills(artifacts_dir: Path) -> list[GeneratedSkill]:
    skills: list[GeneratedSkill] = []
    for path in sorted(Path(artifacts_dir).glob("skill_*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        template = str(artifact["template"])
        skills.append(
            GeneratedSkill(
                skill_id=str(artifact["skill_id"]),
                template=template,
                signature=compile_signature(template),
                spec=InterpretationSpec.model_validate(artifact["spec"]),
                evidence=dict(artifact.get("evidence") or {}),
                provenance=dict(artifact.get("provenance") or {}),
            )
        )
    # Larger (more specific) templates first for deterministic precedence.
    skills.sort(key=lambda s: (-len(s.template), s.skill_id))
    return skills


def adoption_summary(evidence: dict[str, Any]) -> str:
    """One-line evidence basis for an adopted interpretation."""
    disc = evidence.get("discrimination") or {}
    basis = disc.get("adoption_basis")
    adopted = next((c for c in disc.get("candidates") or []
                    if c.get("name") == disc.get("adopted")), {})
    rate, agree, total = adopted.get("rate"), adopted.get("agree"), adopted.get("total")
    if rate is not None and total:
        return f"agreement {rate:.2f} ({agree}/{total} reference instances)"
    return str(basis or "adopted")


def render_skills_digest(artifacts_dir: Path) -> str:
    """Schema-level digest of every learned artifact, written for LLM
    consumption through the knowledge plane: canonical template, adopted
    convention, evidence basis. Contains only templates (placeholders, no
    entities) and interpretation rules — no task references, no answer
    values — so it satisfies the freeze sanitizer's constraints by
    construction."""
    lines = [
        "# Learned interpretation conventions",
        "",
        "Conventions distilled by the semantics pipeline from the public "
        "documentation and discriminated on real data. Skill entries passed "
        "the adoption gate and the independent calibration audit; note "
        "entries led their template's discrimination but fell short of the "
        "evidence gate — apply them unless the question wording overrides.",
        "",
    ]
    for skill in load_generated_skills(artifacts_dir):
        lines.append(f"## Question shape: {skill.template}")
        lines.append(f"- Convention: {render_convention(skill.spec)}")
        lines.append(f"- Evidence: {adoption_summary(skill.evidence)}")
        lines.append("")
    for note in load_calibration_notes(artifacts_dir):
        lines.append(f"## Question shape (advisory): {note.template}")
        lines.append(f"- Convention: {note.convention}")
        lines.append("")
    return "\n".join(lines)


def template_note_id(template: str) -> str:
    return "note_" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class CalibrationNote:
    """A schema-level interpretation note for a template that led the
    discrimination but fell short of adoption. It steers the LLM path (never
    answers), carries its spec for auditing, and matches by signature."""

    note_id: str
    template: str
    signature: TemplateSignature
    spec: InterpretationSpec
    convention: str

    def match(self, question: str) -> bool:
        return self.signature.regex.search(" ".join(question.split())) is not None


def write_note_artifact(
    *, out_dir: Path, template: str, spec: InterpretationSpec,
    evidence: dict[str, Any], provenance: dict[str, Any],
) -> Path:
    artifact = {
        "version": ARTIFACT_VERSION,
        "note_id": template_note_id(template),
        "template": template,
        "spec": spec.model_dump(mode="json"),
        "convention": render_convention(spec),
        "evidence": evidence,
        "provenance": provenance,
    }
    _assert_artifact_safe(artifact)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact['note_id']}.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_calibration_notes(artifacts_dir: Path) -> list[CalibrationNote]:
    notes: list[CalibrationNote] = []
    for path in sorted(Path(artifacts_dir).glob("note_*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        template = str(artifact["template"])
        notes.append(CalibrationNote(
            note_id=str(artifact["note_id"]),
            template=template,
            signature=compile_signature(template),
            spec=InterpretationSpec.model_validate(artifact["spec"]),
            convention=str(artifact.get("convention") or ""),
        ))
    notes.sort(key=lambda n: (-len(n.template), n.note_id))
    return notes


def _assert_artifact_safe(artifact: dict[str, Any]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in _FORBIDDEN_ARTIFACT_KEYS:
                    raise ValueError(f"artifact contains forbidden key {key!r} at {path}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            if re.search(r"\bground.?truth\b|\bpseudo.?gold\b", value, flags=re.IGNORECASE):
                raise ValueError(f"artifact contains forbidden label at {path}")

    walk(artifact, "$")
