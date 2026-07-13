from types import SimpleNamespace

from dabstep_agent_pydantic.cli import build_parser, evaluation_policy_from_args
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig, MemoryTrace, RunMode
from dabstep_agent_pydantic.runner import write_memory_learnings


def test_official_policy_disables_all_mutating_channels():
    policy = EvaluationPolicy.official()
    assert policy.official_run is True
    assert policy.remote_memory_writes is False
    assert policy.local_proposal_writes is False
    assert policy.same_run_learning is False
    assert policy.snapshot_mutation is False


def test_cli_disable_memory_writes_selects_official_policy():
    args = build_parser().parse_args([
        "--input", "tasks.json",
        "--data-dir", "context",
        "--disable-memory-writes",
    ])
    assert evaluation_policy_from_args(args) == EvaluationPolicy.official()


def test_official_policy_blocks_remote_write_even_if_memory_config_allows_it():
    class ExplodingClient:
        def add_memory(self, *args, **kwargs):
            raise AssertionError("official policy must veto remote writes")

    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        memory_write_enabled=True,
        project_id="project",
        user_id="user",
    )
    trace = MemoryTrace()
    write_memory_learnings(
        {"task_id": "t1", "agent_answer": "42", "used_code": True},
        config=config,
        memory_client=ExplodingClient(),
        trace=trace,
        evaluation_policy=EvaluationPolicy.official(),
    )
    assert trace.created_count == 0
    assert trace.policy_decisions[-1]["reason"] == "official evaluation policy"

