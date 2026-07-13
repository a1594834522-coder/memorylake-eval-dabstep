from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class RunMode(str, Enum):
    CLEAN = "clean"
    MEMORY_ASSISTED = "memory-assisted"


class MemoryRouterMode(str, Enum):
    OFF = "off"
    HOSTED = "hosted"
    BYOK = "byok"


class MemorySearchHit(BaseModel):
    id: str
    content: str
    score: float | None = None
    user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MemoryTrace(BaseModel):
    search_query: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    retrieval_params: dict[str, object] = Field(default_factory=dict)
    asset_fingerprint: str | None = None
    retrieved_count: int = 0
    retrieved: list[dict[str, object]] = Field(default_factory=list)
    document_retrieved_count: int = 0
    retrieved_documents: list[dict[str, object]] = Field(default_factory=list)
    document_chunks_dropped: int = 0
    document_context_truncated: bool = False
    created_count: int = 0
    created_event_ids: list[str] = Field(default_factory=list)
    policy_decisions: list[dict[str, object]] = Field(default_factory=list)
    helper_functions_used: list[str] = Field(default_factory=list)
    skipped_memory_writes: list[dict[str, object]] = Field(default_factory=list)
    route_card_ids: list[str] = Field(default_factory=list)
    route_traces: list[dict[str, object]] = Field(default_factory=list)
    analysis_plan: dict[str, object] | None = None


class MemoryLakeConfig(BaseModel):
    run_mode: RunMode = RunMode.CLEAN
    memory_router_mode: MemoryRouterMode = MemoryRouterMode.OFF
    memory_enabled: bool = False
    memory_write_enabled: bool = True
    api_key: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    session_prefix: str = "dabstep"
    top_k: int = 5
    threshold: float = 0.3
    rerank: bool = False

    @model_validator(mode="after")
    def validate_memory_requirements(self) -> "MemoryLakeConfig":
        if self.run_mode is RunMode.CLEAN:
            self.memory_enabled = False
            self.memory_router_mode = MemoryRouterMode.OFF
        if self.memory_enabled and not self.project_id:
            raise ValueError("project_id is required when memory is enabled")
        if self.memory_enabled and not self.user_id:
            raise ValueError("user_id is required when memory is enabled")
        return self

    def memory_metadata(
        self,
        *,
        asset_type: str = "runtime_learning",
        source: str = "runtime",
        official_safe: bool | None = None,
        contains_answer: bool = False,
    ) -> dict[str, str]:
        official = True if official_safe is None else official_safe
        return {
            "run_mode": self.run_mode.value,
            "memory_policy": self.run_mode.value.replace("-", "_"),
            "asset_type": asset_type,
            "source": source,
            "official_safe": str(official).lower(),
            "contains_answer": str(contains_answer).lower(),
            "domain": "dabstep",
        }
