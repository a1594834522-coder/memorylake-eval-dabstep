from pathlib import Path

from scripts.release_audit import audit_paths


def test_audit_rejects_generated_and_local_paths(tmp_path: Path):
    tracked = [
        "data/tasks.json",
        "artifacts/skills/skill_example.json",
        "results/run.jsonl",
        "workspace/task/solution.py",
        "docs/plans/internal-implementation.md",
        ".env",
        ".DS_Store",
        "src/package/__pycache__/module.pyc",
        "src/package.egg-info/PKG-INFO",
        "dist/package.whl",
    ]

    issues = audit_paths(tmp_path, tracked)

    assert {issue.path for issue in issues} == set(tracked)
    assert all(issue.kind == "forbidden_path" for issue in issues)


def test_audit_allows_source_docs_and_empty_env_example(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "docs" / "design.md").write_text("# Design\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=\nMEMORYLAKE_API_KEY=\n",
        encoding="utf-8",
    )

    issues = audit_paths(
        tmp_path,
        ["src/module.py", "docs/design.md", ".env.example"],
    )

    assert issues == []


def test_audit_detects_non_placeholder_api_key_without_exposing_value(tmp_path: Path):
    path = tmp_path / "config.txt"
    secret = "OPENAI_API_KEY" + "=live-secret-value-123456"
    path.write_text(secret + "\n", encoding="utf-8")

    issues = audit_paths(tmp_path, ["config.txt"])

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("potential_secret", "config.txt")
    ]
    assert secret not in issues[0].message


def test_audit_detects_private_key_blocks(tmp_path: Path):
    path = tmp_path / "key.txt"
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    path.write_text(marker + "\nredacted\n", encoding="utf-8")

    issues = audit_paths(tmp_path, ["key.txt"])

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("potential_secret", "key.txt")
    ]
