"""Official-submission safety suite.

Guards the official run boundary: no answer leakage in the shipped code or
assets, submission exports preserve every successful task (including empty
answers), and the memory-write kill switch actually disables writes.
"""

import json
import os
import re
from pathlib import Path

import pytest

from dabstep_agent_pydantic.cli import build_parser
from dabstep_agent_pydantic.cli import memory_config_from_args
from dabstep_agent_pydantic.curriculum import ANSWER_LIKE_NUMBER
from dabstep_agent_pydantic.curriculum import CurriculumMemory
from dabstep_agent_pydantic.curriculum import TASK_REFERENCE
from dabstep_agent_pydantic.curriculum import load_forbidden_entity_terms
from dabstep_agent_pydantic.curriculum import sanitize_curriculum_memories
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import MemoryTrace
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.memory_policy import MemoryCandidate
from dabstep_agent_pydantic.memory_policy import filter_memory_candidates
from dabstep_agent_pydantic.runner import build_family_query_terms
from dabstep_agent_pydantic.runner import write_memory_learnings
from dabstep_agent_pydantic.runtime_assets import load_runtime_assets
from dabstep_agent_pydantic.submission import export_submission


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION_LINES = 450

LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"pseudo[_-]?gold",
        r"gold_answers?",
        r"task_scores",
        r"dev_labels",
        r"answer_map|known_answers|answer_lookup",
        r"task_id\s*==\s*[\"']?\d",  # hardcoded per-task branches
    )
]


def _shipped_files():
    files = list((REPO_ROOT / "src").rglob("*.py"))
    assets_dir = REPO_ROOT / "assets"
    if assets_dir.exists():
        files.extend(path for suffix in ("*.json", "*.md") for path in assets_dir.rglob(suffix))
    return files


def test_shipped_code_and_assets_have_no_answer_leakage_markers():
    offenders = []
    for path in _shipped_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LEAKAGE_PATTERNS:
            match = pattern.search(text)
            if match:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)!r}")
    assert not offenders, "answer-leakage markers found:\n" + "\n".join(offenders)


def test_shipped_code_has_no_real_benchmark_entities():
    data_dir = os.getenv("DABSTEP_CONTEXT_DIR") or os.getenv("DABSTEP_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "merchant_data.json").exists():
        pytest.skip("set DABSTEP_CONTEXT_DIR to the DABStep context directory to run the entity scan")
    merchants = {
        str(row.get("merchant", "")).strip()
        for row in json.loads((Path(data_dir) / "merchant_data.json").read_text(encoding="utf-8"))
    }
    merchants.discard("")
    offenders = []
    for path in _shipped_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for merchant in merchants:
            if merchant in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {merchant}")
    assert not offenders, "real merchant identifiers found in shipped code:\n" + "\n".join(offenders)


MEMORY_ARTIFACTS = (
    REPO_ROOT / "results" / "curriculum_memories.jsonl",
    REPO_ROOT / "results" / "memory_export.jsonl",
)
CLEANED_CURRICULUM_CANDIDATES = REPO_ROOT / "results" / "curriculum_memories.cleaned.jsonl"


def _memory_artifact_contents():
    contents = []
    for path in MEMORY_ARTIFACTS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "rejected":
                continue
            contents.append((path.name, str(row.get("content", ""))))
    return contents


def test_memory_artifacts_are_answer_free_when_present():
    contents = _memory_artifact_contents()
    if not contents:
        pytest.skip("no memory artifacts exported yet")
    offenders = []
    for source, content in contents:
        if TASK_REFERENCE.search(content):
            offenders.append(f"{source}: task reference in {content[:80]!r}")
        if ANSWER_LIKE_NUMBER.search(content):
            offenders.append(f"{source}: answer-like value in {content[:80]!r}")
    assert not offenders, "memory artifacts contain answer leakage:\n" + "\n".join(offenders)


def test_memory_artifacts_have_no_real_benchmark_entities():
    contents = _memory_artifact_contents()
    if not contents:
        pytest.skip("no memory artifacts exported yet")
    data_dir = os.getenv("DABSTEP_CONTEXT_DIR") or os.getenv("DABSTEP_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "merchant_data.json").exists():
        pytest.skip("set DABSTEP_CONTEXT_DIR to the DABStep context directory to run the entity scan")
    merchants = {
        str(row.get("merchant", "")).strip()
        for row in json.loads((Path(data_dir) / "merchant_data.json").read_text(encoding="utf-8"))
    }
    merchants.discard("")
    offenders = [
        f"{source}: {merchant}"
        for source, content in contents
        for merchant in merchants
        if merchant in content
    ]
    assert not offenders, "memory artifacts reference real merchants:\n" + "\n".join(offenders)


def test_cleaned_curriculum_candidates_pass_sanitizer_when_context_available():
    if not CLEANED_CURRICULUM_CANDIDATES.exists():
        pytest.skip("no cleaned curriculum candidate artifact to validate")
    data_dir = os.getenv("DABSTEP_CONTEXT_DIR") or os.getenv("DABSTEP_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "merchant_data.json").exists():
        pytest.skip("set DABSTEP_CONTEXT_DIR to the DABStep context directory to run the candidate scan")

    memories = []
    for line in CLEANED_CURRICULUM_CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "rejected":
            continue
        memories.append(
            CurriculumMemory(
                title=row["title"],
                content=row["content"],
                category=row["category"],
                evidence=row["evidence"],
            )
        )

    allowed, rejections = sanitize_curriculum_memories(
        memories,
        forbidden_terms=load_forbidden_entity_terms(Path(data_dir)),
        data_dir=Path(data_dir),
    )

    assert len(allowed) >= 28
    assert not rejections


def test_memory_family_query_terms_do_not_contain_real_entities():
    data_dir = os.getenv("DABSTEP_CONTEXT_DIR") or os.getenv("DABSTEP_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "merchant_data.json").exists():
        pytest.skip("set DABSTEP_CONTEXT_DIR to the DABStep context directory to run the entity scan")
    route_cards = load_runtime_assets(REPO_ROOT / "assets" / "default").route_cards
    merchants = load_forbidden_entity_terms(Path(data_dir))
    family_terms = [
        term
        for family in ("fee_matching", "fee_simulation", "customer_fraud_metrics", "schema_semantics")
        for term in build_family_query_terms(family, route_cards)
    ]

    offenders = [
        f"{term}: {merchant}"
        for term in family_terms
        for merchant in merchants
        if merchant and merchant.lower() in term.lower()
    ]
    assert not offenders


def test_official_submission_artifact_shape_when_present():
    submission_path = REPO_ROOT / "results" / "official_submission.jsonl"
    if not submission_path.exists():
        pytest.skip("no local official submission artifact to validate")
    rows = [json.loads(line) for line in submission_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == EXPECTED_SUBMISSION_LINES
    task_ids = [row["task_id"] for row in rows]
    assert len(set(task_ids)) == EXPECTED_SUBMISSION_LINES
    for row in rows:
        assert set(row) == {"task_id", "agent_answer"}
        assert isinstance(row["agent_answer"], str)


def test_export_submission_preserves_every_successful_task(tmp_path):
    input_path = tmp_path / "runtime.jsonl"
    output_path = tmp_path / "submission.jsonl"
    lines = []
    for index in range(EXPECTED_SUBMISSION_LINES):
        answer = "" if index % 50 == 0 else str(index)
        lines.append(json.dumps({"task_id": f"task-{index}", "agent_answer": answer}))
    # A failed attempt followed by a successful retry must still yield one row.
    lines.insert(0, json.dumps({"task_id": "task-0", "agent_answer": "", "error": {"type": "Timeout"}}))
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = export_submission(input_path, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report["line_count"] == EXPECTED_SUBMISSION_LINES
    assert len(rows) == EXPECTED_SUBMISSION_LINES
    assert rows[0] == {"task_id": "task-0", "agent_answer": ""}


def test_cli_disable_memory_writes_flag_disables_writes():
    args = build_parser().parse_args(
        [
            "--input", "tasks.json",
            "--data-dir", "context",
            "--run-mode", "memory-assisted",
            "--memorylake-project-id", "project",
            "--memorylake-user-id", "user",
            "--disable-memory-writes",
        ]
    )
    config = memory_config_from_args(args)
    assert config.memory_enabled is True
    assert config.memory_write_enabled is False


def test_cli_clean_mode_disables_memory_entirely():
    args = build_parser().parse_args(["--input", "tasks.json", "--data-dir", "context"])
    config = memory_config_from_args(args)
    assert config.run_mode is RunMode.CLEAN
    assert config.memory_enabled is False


def test_disabled_writes_never_reach_the_memory_client():
    class ExplodingClient:
        def add_memory(self, *args, **kwargs):
            raise AssertionError("memory writes must not happen with writes disabled")

    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        memory_write_enabled=False,
        project_id="project",
        user_id="user",
    )
    trace = MemoryTrace()
    write_memory_learnings(
        {"task_id": "t1", "agent_answer": "42", "used_code": True, "generated_code": "df.groupby('a')"},
        config=config,
        memory_client=ExplodingClient(),
        trace=trace,
    )
    assert trace.created_count == 0
    assert trace.policy_decisions[-1]["reason"] == "memory writes disabled"


def test_memory_policy_blocks_task_specific_and_answer_bearing_memories():
    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        project_id="project",
        user_id="user",
    )
    candidates = [
        MemoryCandidate(category="recipe", content="For task 1234 the trick is to filter by account type."),
        MemoryCandidate(category="recipe", content="The result 87.5 comes from EUR-volume fraud buckets."),
        MemoryCandidate(category="recipe", content="Use EUR-volume fraud buckets for fee matching."),
    ]
    allowed, decisions = filter_memory_candidates(candidates, config, task_id="1234", answer="87.5")
    assert [candidate.content for candidate in allowed] == [
        "Use EUR-volume fraud buckets for fee matching."
    ]
    assert {decision["reason"] for decision in decisions} == {
        "task-specific memory blocked",
        "answer leakage blocked",
        "policy allows reusable memory",
    }
