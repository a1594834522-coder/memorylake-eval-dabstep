from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class EvaluationPolicy(BaseModel):
    """Controls mutations that must be disabled during official evaluation."""

    model_config = ConfigDict(frozen=True)

    official_run: bool = False
    remote_memory_writes: bool = True
    local_proposal_writes: bool = True
    same_run_learning: bool = False
    snapshot_mutation: bool = False

    @classmethod
    def development(cls) -> "EvaluationPolicy":
        return cls()

    @classmethod
    def official(cls) -> "EvaluationPolicy":
        return cls(
            official_run=True,
            remote_memory_writes=False,
            local_proposal_writes=False,
            same_run_learning=False,
            snapshot_mutation=False,
        )
