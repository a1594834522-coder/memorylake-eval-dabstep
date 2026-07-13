from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.local_paths import resolve_standard_data_path
from dabstep_agent_pydantic.memorylake import DEFAULT_MEMORYLAKE_BASE_URL
from dabstep_agent_pydantic.memorylake import MemoryLakeClient
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import MemoryRouterMode
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.runner import run_benchmark
from dabstep_agent_pydantic.semantic_workflow import SemanticMode


def default_freeze_state_path() -> Path:
    return Path(os.getenv("DABSTEP_FREEZE_STATE_PATH") or "artifacts/freeze_state.json")


def write_freeze_state(
    path: Path,
    *,
    project_id: str,
    document_digests: Mapping[str, str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": 1,
                "project_id": project_id,
                "document_digests": dict(sorted(document_digests.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_freeze_project_id(path: Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    project_id = str(payload.get("project_id") or "").strip() if isinstance(payload, dict) else ""
    return project_id or None


def resolve_memorylake_project_id(
    *,
    explicit: str | None,
    environment: Mapping[str, str] = os.environ,
    state_path: Path | None = None,
) -> str | None:
    return (
        explicit
        or environment.get("MEMORYLAKE_PROJECT_ID")
        or read_freeze_project_id(state_path or default_freeze_state_path())
    )


def default_assets_dir(root: Path | None = None) -> Path | None:
    project_root = root or Path(__file__).resolve().parents[2]
    candidate = project_root / "assets" / "default"
    return candidate if candidate.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pydantic AI DABStep agent")
    parser.add_argument("--input", required=True, type=Path, help="DABStep tasks JSON file")
    parser.add_argument("--data-dir", required=True, type=Path, help="DABStep context directory")
    parser.add_argument("--task-id", default=None, help="Comma-separated task IDs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks after filtering")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"), help="Output JSONL path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output file and skip task IDs that already have non-empty answers",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=default_assets_dir(),
        help="Optional runtime route-card directory; defaults to bundled route cards when present",
    )
    parser.add_argument(
        "--run-mode",
        choices=[mode.value for mode in RunMode],
        default=RunMode.CLEAN.value,
        help="Evaluation/memory policy mode",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=[mode.value for mode in SemanticMode],
        default=SemanticMode.LEGACY.value,
        help="Semantic compiler promotion mode; legacy preserves the current runtime",
    )
    parser.add_argument(
        "--memory-router-mode",
        choices=[mode.value for mode in MemoryRouterMode],
        default=MemoryRouterMode.OFF.value,
        help="Memory Router model endpoint mode",
    )
    parser.add_argument("--memorylake-project-id", default=None, help="MemoryLake project ID")
    parser.add_argument(
        "--freeze-state",
        type=Path,
        default=default_freeze_state_path(),
        help="Local freeze state written by `dabstep-agent freeze`",
    )
    parser.add_argument("--memorylake-user-id", default=os.getenv("MEMORYLAKE_USER_ID") or "dabstep-agent", help="MemoryLake user ID filter/owner")
    parser.add_argument(
        "--memorylake-session-prefix",
        default="dabstep",
        help="Prefix for MemoryLake chat_session_id values",
    )
    parser.add_argument("--memory-top-k", type=int, default=5, help="Memory search top_k")
    parser.add_argument("--memory-threshold", type=float, default=0.3, help="Memory search threshold")
    parser.add_argument("--memory-rerank", action="store_true", help="Enable MemoryLake memory reranking")
    parser.add_argument(
        "--disable-memory-writes",
        action="store_true",
        help="Retrieve MemoryLake context without writing new memories during this run",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("workspace"),
        help="Directory for generated code and intermediate files",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to solve concurrently; output and memory persistence remain serialized",
    )
    parser.add_argument("--task-retries", type=int, default=0, help="Retries per task after transient failures")
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=0.0,
        help="Delay between per-task retries",
    )
    return parser


def build_memory_client(config: MemoryLakeConfig):
    if not config.memory_enabled:
        return None
    load_dotenv()
    api_key = os.getenv("MEMORYLAKE_API_KEY")
    if not api_key:
        raise RuntimeError("MEMORYLAKE_API_KEY is required when memory is enabled")
    return MemoryLakeClient(
        api_key=api_key,
        base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
    )


def configure_model_endpoint_env(memory_router_mode: str) -> None:
    os.environ["DABSTEP_MEMORY_ROUTER_MODE"] = memory_router_mode


def memory_config_from_args(args: argparse.Namespace) -> MemoryLakeConfig:
    memory_enabled = args.run_mode != RunMode.CLEAN.value
    if memory_enabled and not args.memorylake_project_id:
        raise SystemExit(
            "run-mode is memory-assisted but MEMORYLAKE_PROJECT_ID is not set - "
            "run `dabstep-agent freeze` first (it auto-creates the project and prints the id), "
            "or pass --memorylake-project-id / use --run-mode clean."
        )
    return MemoryLakeConfig(
        run_mode=RunMode(args.run_mode),
        memory_router_mode=MemoryRouterMode(args.memory_router_mode),
        memory_enabled=memory_enabled and bool(args.memorylake_project_id),
        memory_write_enabled=not args.disable_memory_writes,
        project_id=args.memorylake_project_id,
        user_id=args.memorylake_user_id,
        session_prefix=args.memorylake_session_prefix,
        top_k=args.memory_top_k,
        threshold=args.memory_threshold,
        rerank=args.memory_rerank,
    )


def evaluation_policy_from_args(args: argparse.Namespace) -> EvaluationPolicy:
    if args.disable_memory_writes:
        return EvaluationPolicy.official()
    return EvaluationPolicy.development()


SUBCOMMANDS = ("learn", "freeze", "run", "report")


def main() -> None:
    import sys
    load_dotenv()
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        command, rest = argv[0], argv[1:]
        if command == "learn":
            from dabstep_agent_pydantic.distill.learn import main as learn_main
            learn_main(rest)
            return
        if command == "freeze":
            _freeze_main(rest)
            return
        if command == "report":
            _report_main(rest)
            return
        argv = rest  # "run"
    args = build_parser().parse_args(argv)
    args.input = resolve_standard_data_path(args.input)
    args.data_dir = resolve_standard_data_path(args.data_dir)
    args.memorylake_project_id = resolve_memorylake_project_id(
        explicit=args.memorylake_project_id,
        state_path=args.freeze_state,
    )
    args.memorylake_user_id = args.memorylake_user_id or os.getenv("MEMORYLAKE_USER_ID") or "dabstep-agent"
    configure_model_endpoint_env(args.memory_router_mode)
    memory_config = memory_config_from_args(args)
    records = asyncio.run(
        run_benchmark(
            input_path=args.input,
            data_dir=args.data_dir,
            output_path=args.output,
            workspace_dir=args.workspace_dir,
            assets_dir=args.assets_dir,
            task_ids=args.task_id,
            limit=args.limit,
            memory_config=memory_config,
            memory_client=build_memory_client(memory_config),
            evaluation_policy=evaluation_policy_from_args(args),
            concurrency=args.concurrency,
            resume=args.resume,
            task_retries=args.task_retries,
            retry_delay_seconds=args.retry_delay_seconds,
            semantic_mode=SemanticMode(args.semantic_mode),
        )
    )
    print(f"Completed {len(records)} task(s).")


if __name__ == "__main__":
    main()


def _doc_drift_against_skills(docs: list[Path], skills_dir: Path) -> list[str]:
    """Documents whose content differs from the version learn recorded.

    Learned skill artifacts carry provenance.doc_fingerprints (sha256 of every
    knowledge doc the learn run read); freeze must not upload a different
    version, or answers stop being traceable to the knowledge they came from.
    """
    import hashlib
    import json as _json

    recorded: dict[str, set[str]] = {}
    if skills_dir and Path(skills_dir).is_dir():
        for artifact_path in sorted(Path(skills_dir).glob("skill_*.json")):
            prints = (_json.loads(artifact_path.read_text())
                      .get("provenance", {}).get("doc_fingerprints") or {})
            for name, digest in prints.items():
                recorded.setdefault(name, set()).add(str(digest))
    drift = []
    for path in docs:
        expected = recorded.get(path.name)
        if not expected:
            continue  # doc not consulted by learn (e.g. curriculum exports)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual not in expected:
            drift.append(f"{path.name}: uploading {actual[:12]}..., "
                         f"skills were learned from {sorted(d[:12] for d in expected)}")
    return drift


def _freeze_main(argv: list[str]) -> None:
    """Upload knowledge documents to MemoryLake and print the frozen fingerprint."""
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(prog="dabstep-agent freeze")
    parser.add_argument("--docs", nargs="+", type=Path, required=True,
                        help="Documents to upload (public docs + curriculum/skill exports)")
    parser.add_argument("--project-id", default=None)
    parser.add_argument(
        "--freeze-state",
        type=Path,
        default=default_freeze_state_path(),
        help="Persist the created/reused project ID for subsequent run commands",
    )
    parser.add_argument("--skills-dir", type=Path, default=Path("artifacts/skills"),
                        help="Cross-check doc fingerprints recorded by learn (skipped if absent)")
    parser.add_argument("--allow-doc-drift", action="store_true",
                        help="Upload even when a document differs from the version learn read")
    args = parser.parse_args(argv)
    load_dotenv()
    drift = _doc_drift_against_skills(args.docs, args.skills_dir)
    if drift:
        for line in drift:
            print(f"DOC DRIFT: {line}")
        if not args.allow_doc_drift:
            raise SystemExit(
                "freeze aborted: these documents differ from the versions the learned "
                "skills were distilled from - re-run learn, or pass --allow-doc-drift "
                "if the change is intentional")
    client = MemoryLakeClient(
        api_key=os.environ["MEMORYLAKE_API_KEY"],
        base_url=os.getenv("MEMORYLAKE_BASE_URL", DEFAULT_MEMORYLAKE_BASE_URL),
    )
    project_id = resolve_memorylake_project_id(
        explicit=args.project_id,
        state_path=args.freeze_state,
    )
    if not project_id:
        import datetime

        name = f"dabstep-agent-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d-%H%M}"
        project_id = client.create_project(name, description="Auto-created by dabstep-agent freeze")
        print(f"created MemoryLake project {project_id!r} ({name})")
        print(f"add to your .env to reuse it:  MEMORYLAKE_PROJECT_ID={project_id}")
    item_ids: list[str] = []
    document_digests: dict[str, str] = {}
    for path in args.docs:
        item_ids.append(client.upload_file_to_library(path))
        document_digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = client.add_documents(project_id, drive_item_ids=item_ids)
    print("\n".join(f"{digest}  {name}" for name, digest in document_digests.items()))
    print(f"uploaded={result['success_count']} failed={result['failure_count']} project={project_id}")
    if result["failure_count"]:
        raise SystemExit("freeze failed; local project state was not updated")
    write_freeze_state(
        args.freeze_state,
        project_id=project_id,
        document_digests=document_digests,
    )
    print(f"freeze state written to {args.freeze_state}")


def _report_main(argv: list[str]) -> None:
    """Export the leaderboard submission and optionally compare two runs."""
    import argparse

    from dabstep_agent_pydantic.ablation import compare_runs
    from dabstep_agent_pydantic.submission import export_submission

    parser = argparse.ArgumentParser(prog="dabstep-agent report")
    parser.add_argument("--run", required=True, type=Path, help="Runtime results JSONL")
    parser.add_argument("--submission", type=Path, default=None, help="Write leaderboard JSONL here")
    parser.add_argument("--compare-with", type=Path, default=None, help="Optional second run to diff")
    parser.add_argument("--reference-answers", type=Path, default=None,
                        help="Optional self-generated reference answers for offline accuracy")
    args = parser.parse_args(argv)
    if args.submission:
        report = export_submission(args.run, args.submission)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.compare_with:
        report = compare_runs(run_a=args.run, run_b=args.compare_with,
                              reference_path=args.reference_answers)
        print(json.dumps(report, indent=2, ensure_ascii=False))
