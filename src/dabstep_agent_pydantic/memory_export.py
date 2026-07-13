"""Export a MemoryLake project's memories to an auditable, hashable JSONL snapshot.

The snapshot serves two purposes: freezing the official memory set before a
benchmark run (the content hash pins exactly what the agent could retrieve), and
feeding the official-safety leakage scan, which runs over the exported content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from dabstep_agent_pydantic.memorylake import DEFAULT_MEMORYLAKE_BASE_URL
from dabstep_agent_pydantic.memorylake import MemoryLakeClient


EXPORT_FIELDS = ("id", "content", "user_id", "expired", "created_at", "updated_at")


def export_memories(
    client: MemoryLakeClient,
    *,
    project_id: str,
    user_id: str | None,
    output_path: Path,
) -> dict[str, object]:
    memories = client.list_memories(project_id, user_id=user_id)
    rows = sorted(
        ({field: item.get(field) for field in EXPORT_FIELDS} for item in memories),
        key=lambda row: str(row.get("id") or ""),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "ok": True,
        "path": str(output_path),
        "memory_count": len(rows),
        "content_sha256": content_hash(rows),
    }


def content_hash(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("content") or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MemoryLake project memories to an auditable JSONL snapshot")
    parser.add_argument("--memorylake-project-id", required=True)
    parser.add_argument("--memorylake-user-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/memory_export.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()
    api_key = os.getenv("MEMORYLAKE_API_KEY")
    if not api_key:
        raise RuntimeError("MEMORYLAKE_API_KEY is required")
    client = MemoryLakeClient(
        api_key=api_key,
        base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
    )
    report = export_memories(
        client,
        project_id=args.memorylake_project_id,
        user_id=args.memorylake_user_id,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
