from pathlib import Path

from dabstep_agent_pydantic.cli import build_parser
from dabstep_agent_pydantic.cli import read_freeze_project_id
from dabstep_agent_pydantic.cli import resolve_memorylake_project_id
from dabstep_agent_pydantic.cli import write_freeze_state
from dabstep_agent_pydantic.local_paths import resolve_standard_data_path
from dabstep_agent_pydantic.semantic_workflow import SemanticMode


def test_semantic_mode_defaults_to_legacy():
    args = build_parser().parse_args(["--input", "tasks.json", "--data-dir", "context"])

    assert args.semantic_mode == SemanticMode.LEGACY.value


def test_semantic_mode_accepts_all_runtime_modes():
    parser = build_parser()

    for mode in SemanticMode:
        args = parser.parse_args(
            [
                "--input",
                "tasks.json",
                "--data-dir",
                "context",
                "--semantic-mode",
                mode.value,
            ]
        )
        assert args.semantic_mode == mode.value


def test_standard_data_path_falls_back_to_download_layout(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    expected = data_dir / "tasks.json"
    expected.write_text("[]", encoding="utf-8")

    assert resolve_standard_data_path(Path("tasks.json")) == Path("data/tasks.json")


def test_standard_data_path_preserves_explicit_existing_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / "custom.json"
    explicit.write_text("[]", encoding="utf-8")

    assert resolve_standard_data_path(explicit) == explicit


def test_freeze_state_round_trip_and_project_precedence(tmp_path: Path):
    state_path = tmp_path / "artifacts" / "freeze_state.json"
    write_freeze_state(state_path, project_id="state-project", document_digests={"manual.md": "abc"})

    assert read_freeze_project_id(state_path) == "state-project"
    assert resolve_memorylake_project_id(
        explicit="cli-project",
        environment={"MEMORYLAKE_PROJECT_ID": "env-project"},
        state_path=state_path,
    ) == "cli-project"
    assert resolve_memorylake_project_id(
        explicit=None,
        environment={"MEMORYLAKE_PROJECT_ID": "env-project"},
        state_path=state_path,
    ) == "env-project"
    assert resolve_memorylake_project_id(
        explicit=None,
        environment={},
        state_path=state_path,
    ) == "state-project"


def test_invalid_freeze_state_is_ignored(tmp_path: Path):
    state_path = tmp_path / "freeze_state.json"
    state_path.write_text('{"project_id": ""}', encoding="utf-8")

    assert read_freeze_project_id(state_path) is None
