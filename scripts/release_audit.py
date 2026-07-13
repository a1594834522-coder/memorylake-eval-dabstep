from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


FORBIDDEN_ROOTS = {
    "artifacts",
    "build",
    "data",
    "dist",
    "outputs",
    "results",
    "runs",
    "workspace",
}

FORBIDDEN_PREFIXES = {
    ("docs", "plans"),
}

FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}

FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".whl"}
ENV_SECRET_NAMES = {
    "DABSTEP_TEACHER_API_KEY",
    "MEMORYLAKE_API_KEY",
    "OPENAI_API_KEY",
}
PLACEHOLDER_VALUES = {
    "changeme",
    "dummy",
    "example",
    "replace-me",
    "test",
    "your-api-key",
}

OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")
PRIVATE_KEY_MARKER = "-----BEGIN " + "PRIVATE KEY-----"
RSA_PRIVATE_KEY_MARKER = "-----BEGIN RSA " + "PRIVATE KEY-----"
OPENSSH_PRIVATE_KEY_MARKER = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    path: str
    message: str


def _is_forbidden_path(path: str) -> bool:
    normalized = PurePosixPath(path)
    parts = normalized.parts
    if not parts:
        return False
    if parts[0] in FORBIDDEN_ROOTS:
        return True
    if any(parts[: len(prefix)] == prefix for prefix in FORBIDDEN_PREFIXES):
        return True
    if any(part in FORBIDDEN_NAMES for part in parts):
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if normalized.name.startswith(".env") and normalized.name != ".env.example":
        return True
    return normalized.suffix.lower() in FORBIDDEN_SUFFIXES


def _contains_secret(text: str) -> bool:
    if OPENAI_KEY_PATTERN.search(text):
        return True
    if any(
        marker in text
        for marker in (
            PRIVATE_KEY_MARKER,
            RSA_PRIVATE_KEY_MARKER,
            OPENSSH_PRIVATE_KEY_MARKER,
        )
    ):
        return True
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() not in ENV_SECRET_NAMES:
            continue
        candidate = value.strip().strip('"\'')
        if not candidate:
            continue
        if candidate.startswith("<") and candidate.endswith(">"):
            continue
        if candidate.lower() in PLACEHOLDER_VALUES:
            continue
        return True
    return False


def audit_paths(repo_root: Path, tracked_paths: list[str]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for relative_path in tracked_paths:
        path = relative_path.replace("\\", "/")
        if _is_forbidden_path(path):
            issues.append(
                AuditIssue(
                    kind="forbidden_path",
                    path=path,
                    message="generated or local-only path is tracked",
                )
            )
            continue
        absolute_path = repo_root / path
        if not absolute_path.is_file() or absolute_path.stat().st_size > 5_000_000:
            continue
        data = absolute_path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if _contains_secret(text):
            issues.append(
                AuditIssue(
                    kind="potential_secret",
                    path=path,
                    message="tracked text contains a likely non-placeholder secret",
                )
            )
    return issues


def tracked_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\x00") if path]


def commit_count(repo_root: Path) -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the tracked repository tree before public release"
    )
    parser.add_argument(
        "--require-root-commit",
        action="store_true",
        help="Require HEAD to contain exactly one commit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    issues = audit_paths(repo_root, tracked_paths(repo_root))
    if args.require_root_commit and commit_count(repo_root) != 1:
        issues.append(
            AuditIssue(
                kind="public_history",
                path=".git",
                message="public branch must contain exactly one root commit",
            )
        )
    if issues:
        for issue in issues:
            print(f"{issue.kind}: {issue.path}: {issue.message}")
        raise SystemExit(1)
    print("release audit passed")


if __name__ == "__main__":
    main()
