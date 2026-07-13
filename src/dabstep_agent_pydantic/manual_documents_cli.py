from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from dabstep_agent_pydantic.memorylake import DEFAULT_MEMORYLAKE_BASE_URL
from dabstep_agent_pydantic.memorylake import MemoryLakeClient


def upload_manual_document(
    *,
    client: MemoryLakeClient,
    project_id: str,
    manual_path: Path,
    parent_item_id: str,
    conflict_strategy: str,
) -> dict[str, object]:
    library_item_id = client.upload_file_to_library(
        manual_path,
        parent_item_id=parent_item_id,
        conflict_strategy=conflict_strategy,
    )
    import_result = client.add_documents(project_id, drive_item_ids=[library_item_id])
    return {
        "ok": import_result.get("success_count", 0) > 0 and import_result.get("failure_count", 0) == 0,
        "manual_path": str(manual_path),
        "library_item_id": library_item_id,
        "project_id": project_id,
        "import_result": import_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the official DABStep manual into MemoryLake project documents")
    parser.add_argument("--manual-path", type=Path, default=Path("data/context/manual.md"))
    parser.add_argument(
        "--skills-digest",
        type=Path,
        default=None,
        metavar="SKILLS_DIR",
        help="Render the learned-conventions digest from this artifacts directory "
             "and upload it as learned_conventions.md instead of --manual-path",
    )
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--parent-item-id", default="MY_SPACE")
    parser.add_argument(
        "--conflict-strategy",
        choices=["rename", "deny", "overwrite", "replace"],
        default="replace",
        help="Library file name conflict strategy",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    project_id = args.project_id or os.getenv("MEMORYLAKE_PROJECT_ID")
    api_key = os.getenv("MEMORYLAKE_API_KEY")
    if not project_id:
        raise RuntimeError("MEMORYLAKE_PROJECT_ID or --project-id is required")
    if not api_key:
        raise RuntimeError("MEMORYLAKE_API_KEY is required")
    manual_path = args.manual_path
    if args.skills_digest is not None:
        from dabstep_agent_pydantic.distill.emit import render_skills_digest

        digest = render_skills_digest(args.skills_digest)
        manual_path = args.skills_digest / "learned_conventions.md"
        manual_path.write_text(digest, encoding="utf-8")
    report = upload_manual_document(
        client=MemoryLakeClient(
            api_key=api_key,
            base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
        ),
        project_id=project_id,
        manual_path=manual_path,
        parent_item_id=args.parent_item_id,
        conflict_strategy=args.conflict_strategy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
