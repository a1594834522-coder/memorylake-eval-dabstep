"""Offline curriculum pass: distill official-safe domain memories from the public docs.

The agent studies the public context documentation (manual.md, payments-readme.md),
verifies its hypotheses with Python experiments against the dataset, and produces
schema-level semantic memories. No benchmark tasks or answers are ever shown to it,
so the resulting memory set is answer-free by construction; a sanitizer additionally
rejects anything entity-specific before it can be written to MemoryLake.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from dabstep_agent_pydantic.agent import DABStepDeps
from dabstep_agent_pydantic.agent import build_model_from_env
from dabstep_agent_pydantic.agent import model_settings_from_env
from dabstep_agent_pydantic.dataset import summarize_context_dir
from dabstep_agent_pydantic.memorylake import DEFAULT_MEMORYLAKE_BASE_URL
from dabstep_agent_pydantic.memorylake import MemoryLakeClient
from dabstep_agent_pydantic.python_tool import PythonWorkspace
from dabstep_agent_pydantic.toolsets import COMMON_TOOLSET


CURRICULUM_CATEGORIES = (
    "schema_semantics",
    "metric_definitions",
    "fee_matching_semantics",
    "output_contracts",
    "data_quality",
)

PUBLIC_DOC_NAMES = ("manual.md", "payments-readme.md")

MIN_CONTENT_CHARS = 40
MAX_CONTENT_CHARS = 600

# Decimal values and counted subsets read as computed benchmark results, not semantics.
ANSWER_LIKE_NUMBER = re.compile(r"(?<![\w-])-?\d+\.\d+(?![\w-])")
NUMERIC_TOKEN = re.compile(r"(?<![\w.-])-?\d[\d,]*(?:\.\d+)?(?:[kKmM])?(?![\w.-])")
COUNTED_NUMBER = r"-?\d[\d,]*(?:\.\d+)?(?:[kKmM])?"
COUNTED_UNIT = r"(?:rows?|records?|payments?|transactions?|merchants?|fees?|matches?|results?|items?|entries?)"
COUNTED_RESULT = re.compile(
    rf"(?:(?:returned|found|equals?|result|total|count|computed|filtered|after filtering|there (?:are|were|is|was))"
    rf".{{0,60}}{COUNTED_NUMBER}\s*{COUNTED_UNIT}\b|{COUNTED_NUMBER}\s+{COUNTED_UNIT}\b)",
    re.IGNORECASE,
)
TASK_REFERENCE = re.compile(r"\btask[\s_#-]*\d+", re.IGNORECASE)


class CurriculumMemory(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1)
    category: str = Field(min_length=1)
    evidence: str = Field(
        min_length=1,
        description="How the rule was verified: the doc statement and/or the Python check that confirmed it.",
    )


class CurriculumReport(BaseModel):
    memories: list[CurriculumMemory] = Field(default_factory=list)


CURRICULUM_INSTRUCTIONS = """\
You are preparing reusable domain knowledge for future data analysts working on a payments dataset.

You are given the public documentation and a Python workspace with the dataset loaded via
load_dabstep_data(data_dir). Study the documentation, then verify every non-obvious claim with a
short Python experiment before recording it.

Record memories that state generalizable semantics, for example:
- how domain metrics are defined (which numerator/denominator, which currency or count basis),
- how matching or applicability rules treat null, empty-list, or wildcard fields,
- which schema fields interact and common misreadings of them,
- answer-formatting conventions the documentation prescribes,
- data-quality caveats (missing values, encodings, ranges) that change how metrics must be computed.

Hard rules for memory content:
- NEVER mention a specific merchant, shopper, email address, or other business entity by name.
- NEVER record a computed result value; record the METHOD and SEMANTICS, not numbers from the data.
- NEVER reference benchmark tasks, questions, or answers.
- Each memory must be a self-contained rule understandable without this conversation.
- Write 15-40 memories, each with category one of: schema_semantics, metric_definitions,
  fee_matching_semantics, output_contracts, data_quality.
"""


def build_curriculum_agent(model=None) -> Agent[DABStepDeps, CurriculumReport]:
    return Agent(
        model or build_model_from_env(),
        deps_type=DABStepDeps,
        output_type=CurriculumReport,
        instructions=CURRICULUM_INSTRUCTIONS,
        model_settings=model_settings_from_env(),
        toolsets=[COMMON_TOOLSET],
        defer_model_check=True,
    )


def build_curriculum_prompt(data_dir: Path) -> str:
    sections = [f"Data directory: {data_dir}", "", "Available files:", summarize_context_dir(data_dir), ""]
    for name in PUBLIC_DOC_NAMES:
        path = data_dir / name
        if path.exists():
            sections.append(f"===== {name} =====")
            sections.append(path.read_text(encoding="utf-8"))
            sections.append("")
    sections.append(
        "Study the documentation above, verify the non-obvious semantics with Python experiments, "
        "and return the CurriculumReport."
    )
    return "\n".join(sections)


def load_forbidden_entity_terms(data_dir: Path) -> list[str]:
    merchant_file = data_dir / "merchant_data.json"
    if not merchant_file.exists():
        return []
    terms = {
        str(row.get("merchant", "")).strip()
        for row in json.loads(merchant_file.read_text(encoding="utf-8"))
        if isinstance(row, dict)
    }
    terms.discard("")
    return sorted(terms)


def sanitize_curriculum_memories(
    memories: list[CurriculumMemory],
    *,
    forbidden_terms: list[str],
    data_dir: Path | None = None,
) -> tuple[list[CurriculumMemory], list[dict[str, str]]]:
    allowed: list[CurriculumMemory] = []
    rejections: list[dict[str, str]] = []
    seen_contents: set[str] = set()
    public_doc_numeric_tokens = load_public_doc_numeric_tokens(data_dir) if data_dir is not None else set()
    for memory in memories:
        reason = _rejection_reason(
            memory,
            forbidden_terms=forbidden_terms,
            seen_contents=seen_contents,
            public_doc_numeric_tokens=public_doc_numeric_tokens,
        )
        if reason:
            rejections.append({"title": memory.title, "reason": reason})
            continue
        seen_contents.add(memory.content.strip().lower())
        allowed.append(memory)
    return allowed, rejections


def _rejection_reason(
    memory: CurriculumMemory,
    *,
    forbidden_terms: list[str],
    seen_contents: set[str],
    public_doc_numeric_tokens: set[str],
) -> str | None:
    text = f"{memory.title}\n{memory.content}\n{memory.evidence}"
    if memory.category not in CURRICULUM_CATEGORIES:
        return f"unknown category {memory.category!r}"
    if len(memory.content) < MIN_CONTENT_CHARS:
        return "content too short to be a self-contained rule"
    if len(memory.content) > MAX_CONTENT_CHARS:
        return "content too long; memories must stay focused"
    for term in forbidden_terms:
        if term and term.lower() in text.lower():
            return "references a real business entity"
    if TASK_REFERENCE.search(text):
        return "references a benchmark task"
    if _contains_answer_like_value(text, public_doc_numeric_tokens=public_doc_numeric_tokens):
        return "contains an answer-like computed value"
    if memory.content.strip().lower() in seen_contents:
        return "duplicate content"
    return None


def load_public_doc_numeric_tokens(data_dir: Path | None) -> set[str]:
    if data_dir is None:
        return set()
    tokens: set[str] = set()
    for name in PUBLIC_DOC_NAMES:
        path = data_dir / name
        if not path.exists():
            continue
        tokens.update(match.group(0) for match in NUMERIC_TOKEN.finditer(path.read_text(encoding="utf-8")))
    return tokens


def _contains_answer_like_value(text: str, *, public_doc_numeric_tokens: set[str] | None = None) -> bool:
    allowed = public_doc_numeric_tokens or set()
    for match in ANSWER_LIKE_NUMBER.finditer(text):
        if match.group(0) not in allowed:
            return True
    return bool(COUNTED_RESULT.search(text))


def write_curriculum_memories(
    memories: list[CurriculumMemory],
    *,
    client: MemoryLakeClient,
    project_id: str,
    user_id: str,
    session_prefix: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, memory in enumerate(memories):
        event_ids = client.add_memory(
            project_id,
            messages=[
                {"role": "user", "content": f"Store this reusable domain rule: {memory.title}"},
                {"role": "assistant", "content": memory.content},
            ],
            user_id=user_id,
            chat_session_id=f"{session_prefix}-{index:03d}",
            metadata={
                "asset_type": "curriculum",
                "source": "public_docs",
                "official_safe": "true",
                "contains_answer": "false",
                "category": memory.category,
                "domain": "dabstep",
            },
            infer=False,
        )
        results.append({"title": memory.title, "event_ids": event_ids})
    return results


async def run_curriculum(
    *,
    data_dir: Path,
    workspace_dir: Path,
    output_path: Path,
    client: MemoryLakeClient | None,
    project_id: str | None,
    user_id: str | None,
    session_prefix: str,
    dry_run: bool,
) -> dict[str, object]:
    agent = build_curriculum_agent()
    deps = DABStepDeps(
        data_dir=data_dir,
        workspace=PythonWorkspace(workspace_dir),
        file_summary=summarize_context_dir(data_dir),
    )
    result = await agent.run(
        build_curriculum_prompt(data_dir),
        deps=deps,
        usage_limits=UsageLimits(request_limit=None),
    )
    candidates = result.output.memories
    allowed, rejections = sanitize_curriculum_memories(
        candidates,
        forbidden_terms=load_forbidden_entity_terms(data_dir),
        data_dir=data_dir,
    )

    written: list[dict[str, object]] = []
    if not dry_run and allowed:
        if client is None or not project_id or not user_id:
            raise RuntimeError("MemoryLake client, project_id, and user_id are required unless --dry-run is set")
        written = write_curriculum_memories(
            allowed,
            client=client,
            project_id=project_id,
            user_id=user_id,
            session_prefix=session_prefix,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for memory in allowed:
            handle.write(json.dumps({"status": "accepted", **memory.model_dump()}, ensure_ascii=False) + "\n")
        for rejection in rejections:
            handle.write(json.dumps({"status": "rejected", **rejection}, ensure_ascii=False) + "\n")
    return {
        "candidates": len(candidates),
        "accepted": len(allowed),
        "rejected": len(rejections),
        "written": len(written),
        "dry_run": dry_run,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill official-safe curriculum memories from the public docs")
    parser.add_argument("--data-dir", required=True, type=Path, help="DABStep context directory")
    parser.add_argument("--workspace-dir", type=Path, default=Path("workspace/curriculum"))
    parser.add_argument("--output", type=Path, default=Path("results/curriculum_memories.jsonl"))
    parser.add_argument("--memorylake-project-id", default=None)
    parser.add_argument("--memorylake-user-id", default=None)
    parser.add_argument("--session-prefix", default="dabstep-curriculum")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Distill and sanitize memories without writing anything to MemoryLake",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()
    client = None
    if not args.dry_run:
        api_key = os.getenv("MEMORYLAKE_API_KEY")
        if not api_key:
            raise RuntimeError("MEMORYLAKE_API_KEY is required unless --dry-run is set")
        client = MemoryLakeClient(
            api_key=api_key,
            base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
        )
    report = asyncio.run(
        run_curriculum(
            data_dir=args.data_dir,
            workspace_dir=args.workspace_dir,
            output_path=args.output,
            client=client,
            project_id=args.memorylake_project_id,
            user_id=args.memorylake_user_id,
            session_prefix=args.session_prefix,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
