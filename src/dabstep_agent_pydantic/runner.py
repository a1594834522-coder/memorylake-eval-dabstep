from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.dataset import filter_tasks
from dabstep_agent_pydantic.dataset import load_tasks
from dabstep_agent_pydantic.dataset import summarize_context_dir
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.learning import extract_memory_candidates
from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.memory_models import MemoryTrace
from dabstep_agent_pydantic.memory_policy import filter_memory_candidates
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.runtime_assets import load_runtime_assets
from dabstep_agent_pydantic.semantic_workflow import SemanticMode
from dabstep_agent_pydantic.semantic_workflow import SemanticWorkflowError
from dabstep_agent_pydantic.workflow import run_task_workflow


_FAMILY_SEARCH_CACHE: dict[tuple[object, ...], dict[str, list]] = {}
_FAMILY_SEARCH_INFLIGHT: dict[tuple[object, ...], threading.Event] = {}
_FAMILY_SEARCH_LOCK = threading.Lock()
_DOCUMENT_CONTEXT_CHUNKS = 4
_MAX_DOCUMENT_CHUNK_CHARS = 2500


def clear_family_search_cache() -> None:
    with _FAMILY_SEARCH_LOCK:
        _FAMILY_SEARCH_CACHE.clear()
        _FAMILY_SEARCH_INFLIGHT.clear()


def build_memory_context(
    *,
    question: str,
    guidelines: str | None,
    file_summary: str,
    config: MemoryLakeConfig,
    memory_client,
    route_cards: list | None = None,
    asset_fingerprint: str | None = None,
) -> tuple[str | None, MemoryTrace]:
    trace = MemoryTrace()
    trace.asset_fingerprint = asset_fingerprint
    trace.retrieval_params = {"top_k": config.top_k, "threshold": config.threshold, "rerank": config.rerank}
    if not config.memory_enabled or not memory_client or not config.project_id or not config.user_id:
        trace.policy_decisions.append(
            {"action": "search_memory", "allowed": False, "reason": "memory disabled"}
        )
        return None, trace

    route_cards = route_cards or []
    plan = plan_task(question=question, guidelines=guidelines, route_cards=route_cards)
    queries = build_memory_search_queries(question=question, task_family=plan.task_family, route_cards=route_cards)
    trace.search_query = queries[0] if queries else question
    trace.search_queries = queries
    trace.analysis_plan = plan.analysis_plan.model_dump(mode="json") if plan.analysis_plan else None
    try:
        if hasattr(memory_client, "search_project"):
            combined_results = _search_project_queries(
                memory_client=memory_client,
                config=config,
                queries=queries,
                asset_fingerprint=asset_fingerprint,
                trace=trace,
            )
            hits = _dedupe_memory_hits(
                hit
                for combined in combined_results
                for hit in combined.get("memories", [])
            )
            documents = _dedupe_documents(
                document
                for combined in combined_results
                for document in combined.get("documents", [])
            )
            documents = _filter_documents_for_family(documents, plan.task_family, trace)
            hits = hits[: config.top_k]
            documents = documents[: config.top_k]
            trace.document_retrieved_count = len(documents)
            trace.retrieved_documents = [_document_trace_item(document) for document in documents]
            context = _format_memory_and_document_context(hits=hits, documents=documents, trace=trace)
        else:
            hits = memory_client.search_memories(
                config.project_id,
                user_id=config.user_id,
                query=queries[0],
                top_k=config.top_k,
                threshold=config.threshold,
                rerank=config.rerank,
            )
            context = _format_memory_and_document_context(hits=hits, documents=[], trace=trace)
    except Exception as exc:  # noqa: BLE001 - retrieval must not fail the benchmark task.
        trace.policy_decisions.append(
            {"action": "search_memory", "allowed": False, "reason": f"retrieval failed: {type(exc).__name__}"}
        )
        return None, trace
    trace.retrieved_count = len(hits)
    trace.retrieved = [_memory_trace_item(hit) for hit in hits]
    return context or None, trace


def _search_project_queries(
    *,
    memory_client,
    config: MemoryLakeConfig,
    queries: list[str],
    asset_fingerprint: str | None,
    trace: MemoryTrace,
) -> list[dict[str, list]]:
    if not queries:
        return []
    results: list[dict[str, list] | None] = [None] * len(queries)
    jobs = []
    with ThreadPoolExecutor(max_workers=min(2, len(queries))) as executor:
        for index, query in enumerate(queries):
            if index > 0:
                cache_key = _family_search_cache_key(config, query=query, asset_fingerprint=asset_fingerprint)
                cached = _get_family_search_cache(cache_key)
                if cached is not None:
                    results[index] = cached
                    trace.policy_decisions.append(
                        {
                            "action": "search_memory_family_cache",
                            "allowed": True,
                            "reason": "family query cache hit",
                            "query": query,
                        }
                    )
                    continue
                if _claim_family_search(cache_key):
                    future = executor.submit(_search_project_uncached, memory_client, config, query)
                    jobs.append((index, query, cache_key, True, future))
                else:
                    future = executor.submit(_wait_for_family_search_cache, cache_key)
                    jobs.append((index, query, cache_key, False, future))
            else:
                future = executor.submit(_search_project_uncached, memory_client, config, query)
                jobs.append((index, query, None, False, future))

        for index, query, cache_key, owns_cache_fill, future in jobs:
            del query
            try:
                result = future.result()
            except Exception:
                if cache_key is not None and owns_cache_fill:
                    _finish_family_search(cache_key, None)
                raise
            if cache_key is not None and owns_cache_fill:
                _finish_family_search(cache_key, result)
            results[index] = result

    return [result or {"memories": [], "documents": []} for result in results]


def _search_project_uncached(memory_client, config: MemoryLakeConfig, query: str) -> dict[str, list]:
    return memory_client.search_project(
        config.project_id,
        user_id=config.user_id,
        query=query,
        top_n=config.top_k,
        threshold=config.threshold,
    )


def _family_search_cache_key(
    config: MemoryLakeConfig,
    *,
    query: str,
    asset_fingerprint: str | None,
) -> tuple[object, ...]:
    return (
        config.project_id,
        config.user_id,
        query,
        config.top_k,
        config.threshold,
        asset_fingerprint,
    )


def _get_family_search_cache(cache_key: tuple[object, ...]) -> dict[str, list] | None:
    with _FAMILY_SEARCH_LOCK:
        cached = _FAMILY_SEARCH_CACHE.get(cache_key)
    return copy.deepcopy(cached) if cached is not None else None


def _claim_family_search(cache_key: tuple[object, ...]) -> bool:
    with _FAMILY_SEARCH_LOCK:
        if cache_key in _FAMILY_SEARCH_CACHE:
            return False
        if cache_key in _FAMILY_SEARCH_INFLIGHT:
            return False
        _FAMILY_SEARCH_INFLIGHT[cache_key] = threading.Event()
        return True


def _wait_for_family_search_cache(cache_key: tuple[object, ...]) -> dict[str, list]:
    with _FAMILY_SEARCH_LOCK:
        event = _FAMILY_SEARCH_INFLIGHT.get(cache_key)
        cached = _FAMILY_SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    if event is not None:
        event.wait()
    cached = _get_family_search_cache(cache_key)
    if cached is None:
        return {"memories": [], "documents": []}
    return cached


def _finish_family_search(cache_key: tuple[object, ...], result: dict[str, list] | None) -> None:
    with _FAMILY_SEARCH_LOCK:
        if result is not None:
            _FAMILY_SEARCH_CACHE[cache_key] = copy.deepcopy(result)
        event = _FAMILY_SEARCH_INFLIGHT.pop(cache_key, None)
        if event is not None:
            event.set()


def build_memory_search_queries(*, question: str, task_family: str, route_cards: list) -> list[str]:
    queries = [question.strip()]
    family_terms = build_family_query_terms(task_family, route_cards)
    family_query = " ".join([task_family, *family_terms]).strip()
    if family_query and family_query not in queries:
        queries.append(family_query)
    return [query for query in queries if query]


def build_family_query_terms(task_family: str, route_cards: list) -> list[str]:
    route_ids = _route_ids_for_task_family(task_family)
    terms: list[str] = []
    for card in route_cards:
        if getattr(card, "route_id", None) not in route_ids:
            continue
        terms.extend(str(term).strip() for term in getattr(card, "keywords", []) if str(term).strip())
        terms.extend(str(term).strip() for term in getattr(card, "triggers", []) if str(term).strip())
    return _dedupe_strings(terms)


def _route_ids_for_task_family(task_family: str) -> set[str]:
    mapping = {
        "fee_matching": {"fee_matching"},
        "fee_simulation": {"fee_simulation", "fee_matching"},
        "customer_fraud_metrics": {"fraud_and_customer_semantics"},
        "schema_semantics": {"schema_domain_semantics"},
        "fee_analysis": {"fee_matching"},
        "general_data_analysis": {"output_contracts"},
    }
    return mapping.get(task_family, {task_family})


_FAMILY_CURRICULUM_DOCS = {
    "fee_matching": {"curriculum_fee_matching.md"},
    "fee_simulation": {"curriculum_fee_matching.md"},
    "fee_analysis": {"curriculum_fee_matching.md"},
    "customer_fraud_metrics": {"curriculum_fraud_customer.md"},
    "schema_semantics": {"curriculum_schema.md"},
    "general_data_analysis": {"curriculum_schema.md"},
}
_LEGACY_CURRICULUM_DOCS = {"curriculum_rules.md"}
_UNRESTRICTED_DOCS = {"manual.md", "payments-readme.md"}


def allowed_curriculum_documents_for_family(task_family: str) -> set[str] | None:
    return _FAMILY_CURRICULUM_DOCS.get(task_family)


def _filter_documents_for_family(
    documents: list[dict[str, object]],
    task_family: str,
    trace: MemoryTrace,
) -> list[dict[str, object]]:
    allowed_curriculum_docs = allowed_curriculum_documents_for_family(task_family)
    filtered: list[dict[str, object]] = []
    for document in documents:
        raw_document_name = _document_name(document)
        document_name = _canonical_document_name(raw_document_name)
        if document_name in _LEGACY_CURRICULUM_DOCS:
            trace.policy_decisions.append(
                {"action": "drop_document", "reason": "legacy curriculum doc", "document_name": document_name}
            )
            continue
        if (
            document_name.startswith("curriculum_")
            and allowed_curriculum_docs is not None
            and document_name not in allowed_curriculum_docs
        ):
            trace.policy_decisions.append(
                {"action": "drop_document", "reason": "family mismatch", "document_name": document_name}
            )
            continue
        if document_name in _UNRESTRICTED_DOCS or document_name:
            filtered_document = dict(document)
            filtered_document["document_name"] = document_name
            if raw_document_name and raw_document_name != document_name:
                filtered_document["document_name_raw"] = raw_document_name
            filtered.append(filtered_document)
    return filtered


def _dedupe_memory_hits(hits) -> list:
    seen: set[str] = set()
    result = []
    for hit in hits:
        hit_id = str(getattr(hit, "id", ""))
        if hit_id in seen:
            continue
        seen.add(hit_id)
        result.append(hit)
    return result


def _dedupe_documents(documents) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, object]] = []
    for document in documents:
        key = (str(document.get("document_id", "")), _document_snippet(document))
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _format_memory_and_document_context(
    *,
    hits: list,
    documents: list[dict[str, object]],
    trace: MemoryTrace,
) -> str:
    sections: list[str] = []
    if documents:
        lines = ["Verified domain rules and documentation excerpts (verify before relying on them):"]
        renderable_chunks: list[tuple[str, str]] = []
        for document in documents:
            name = _document_name(document) or "document"
            for chunk in _document_chunks(document):
                if len(chunk) <= _MAX_DOCUMENT_CHUNK_CHARS:
                    renderable_chunks.append((name, chunk))
                    continue
                trace.document_chunks_dropped += 1
                trace.document_context_truncated = True
                trace.policy_decisions.append(
                    {
                        "action": "drop_document_chunk",
                        "reason": "chunk too long",
                        "document_name": name,
                        "length": len(chunk),
                    }
                )
        if len(renderable_chunks) > _DOCUMENT_CONTEXT_CHUNKS:
            dropped = len(renderable_chunks) - _DOCUMENT_CONTEXT_CHUNKS
            trace.document_chunks_dropped += dropped
            trace.document_context_truncated = True
            trace.policy_decisions.append(
                {"action": "drop_document_chunk", "reason": "chunk budget exceeded", "dropped": dropped}
            )
        for name, chunk in renderable_chunks[:_DOCUMENT_CONTEXT_CHUNKS]:
            lines.append(f"- [{name}] {chunk}")
        if len(lines) > 1:
            sections.append("\n".join(lines))
    if hits:
        hits = _filter_injectable_memory_hits(hits, trace=trace)
    if hits:
        lines = ["MemoryLake reusable memory context:"]
        lines.extend(f"- {hit.content}" for hit in hits)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _filter_injectable_memory_hits(hits: list, *, trace: MemoryTrace) -> list:
    result = []
    for hit in hits:
        content = str(getattr(hit, "content", ""))
        if content.startswith("Wants to store"):
            trace.policy_decisions.append(
                {
                    "action": "inject_memory",
                    "allowed": False,
                    "reason": "dropped service merged summary memory",
                    "memory_id": str(getattr(hit, "id", "")),
                }
            )
            continue
        result.append(hit)
    return result


def _memory_trace_item(hit) -> dict[str, object]:
    return {
        "id": str(getattr(hit, "id", "")),
        "content_preview": str(getattr(hit, "content", ""))[:240],
        "score": _score_value(getattr(hit, "score", None)),
    }


def _document_trace_item(document: dict[str, object]) -> dict[str, object]:
    return {
        "document_id": str(document.get("document_id", "")),
        "document_name": _document_name(document),
        "type": str(document.get("type", "")),
        "content_preview": _document_snippet(document)[:240],
        "score": _score_value(document.get("score")),
    }


def _score_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _document_snippet(document: dict[str, object]) -> str:
    chunks = _document_chunks(document)
    if chunks:
        return " ".join(chunks).strip()
    highlight = document.get("highlight")
    if not isinstance(highlight, dict):
        return ""
    inner_tables = highlight.get("inner_tables")
    if isinstance(inner_tables, list) and inner_tables:
        return " ".join(
            str(column.get("name", ""))
            for table in inner_tables
            if isinstance(table, dict)
            for column in table.get("columns", [])
            if isinstance(column, dict)
        ).strip()
    return ""


def _document_name(document: dict[str, object]) -> str:
    return str(document.get("document_name") or document.get("document_id") or "")


def _canonical_document_name(document_name: str) -> str:
    return re.sub(r"_(\d+)(\.md)$", r"\2", document_name)


def _document_chunks(document: dict[str, object]) -> list[str]:
    highlight = document.get("highlight")
    if not isinstance(highlight, dict):
        return []
    chunks = highlight.get("chunks")
    if isinstance(chunks, list):
        return [
            str(chunk.get("text", "")).strip()
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
        ]
    return []


def write_memory_learnings(
    record: dict[str, object],
    *,
    config: MemoryLakeConfig,
    memory_client,
    trace: MemoryTrace,
    evaluation_policy: EvaluationPolicy | None = None,
) -> None:
    evaluation_policy = evaluation_policy or EvaluationPolicy.development()
    if not evaluation_policy.remote_memory_writes:
        trace.policy_decisions.append(
            {"action": "write_memory", "allowed": False, "reason": "official evaluation policy"}
        )
        return
    if not config.memory_enabled or not memory_client or not config.project_id or not config.user_id:
        trace.policy_decisions.append(
            {"action": "write_memory", "allowed": False, "reason": "memory disabled"}
        )
        return
    if not config.memory_write_enabled:
        trace.policy_decisions.append(
            {"action": "write_memory", "allowed": False, "reason": "memory writes disabled"}
        )
        return

    candidates = extract_memory_candidates(record, run_mode=config.run_mode)
    allowed, decisions = filter_memory_candidates(
        candidates,
        config,
        task_id=str(record.get("task_id", "")),
        answer=str(record.get("agent_answer", "")),
    )
    trace.policy_decisions.extend(decisions)

    for candidate in allowed:
        events = memory_client.add_memory(
            config.project_id,
            messages=[
                {"role": "user", "content": f"Store reusable DABStep {candidate.category}."},
                {"role": "assistant", "content": candidate.content},
            ],
            user_id=config.user_id,
            chat_session_id=f"{config.session_prefix}-{record.get('task_id', 'unknown')}",
            metadata={**config.memory_metadata(), "category": candidate.category},
            infer=True,
        )
        trace.created_event_ids.extend(events)
    trace.created_count = len(trace.created_event_ids)


def attach_memory_trace(
    record: dict[str, object],
    *,
    run_mode: RunMode,
    memory_router_mode: str,
    trace: MemoryTrace,
) -> dict[str, object]:
    enriched = dict(record)
    if "reasoning_trace" not in enriched and enriched.get("reasoning") is not None:
        enriched["reasoning_trace"] = [str(enriched["reasoning"])]
    enriched["run_mode"] = run_mode.value
    enriched["memory_router_mode"] = memory_router_mode
    enriched["memory_trace"] = trace.model_dump()
    return enriched


async def solve_task(
    task: Task,
    *,
    data_dir: Path,
    workspace_dir: Path,
    file_summary: str,
    assets_dir: Path | None = None,
    memory_context: str | None = None,
    evaluation_policy: EvaluationPolicy | None = None,
    semantic_mode: SemanticMode | str = SemanticMode.LEGACY,
) -> dict[str, object]:
    return await run_task_workflow(
        task,
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        file_summary=file_summary,
        assets_dir=assets_dir,
        memory_context=memory_context,
        evaluation_policy=evaluation_policy,
        semantic_mode=semantic_mode,
    )


async def run_benchmark(
    *,
    input_path: Path,
    data_dir: Path,
    output_path: Path,
    workspace_dir: Path,
    assets_dir: Path | None = None,
    task_ids: str | None = None,
    limit: int | None = None,
    memory_config: MemoryLakeConfig | None = None,
    memory_client=None,
    evaluation_policy: EvaluationPolicy | None = None,
    concurrency: int = 1,
    resume: bool = False,
    task_retries: int = 0,
    retry_delay_seconds: float = 0.0,
    semantic_mode: SemanticMode | str = SemanticMode.LEGACY,
) -> list[dict[str, object]]:
    memory_config = memory_config or MemoryLakeConfig()
    evaluation_policy = evaluation_policy or EvaluationPolicy.development()

    tasks = filter_tasks(load_tasks(input_path), task_ids)
    if resume:
        completed_task_ids = _load_completed_task_ids(output_path)
        tasks = [task for task in tasks if task.task_id not in completed_task_ids]
    if limit is not None:
        tasks = tasks[:limit]

    file_summary = summarize_context_dir(data_dir)
    runtime_assets = load_runtime_assets(assets_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    concurrency = max(1, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def solve_with_trace(task: Task) -> tuple[dict[str, object], MemoryTrace]:
        async with semaphore:
            started = time.time()
            memory_trace = MemoryTrace()
            last_exc: Exception | None = None
            for attempt in range(task_retries + 1):
                try:
                    memory_context, memory_trace = build_memory_context(
                        question=task.question,
                        guidelines=task.guidelines,
                        file_summary=file_summary,
                        config=memory_config,
                        memory_client=memory_client,
                        route_cards=runtime_assets.route_cards,
                        asset_fingerprint=runtime_assets.asset_fingerprint,
                    )
                    record = await solve_task(
                        task,
                        data_dir=data_dir,
                        workspace_dir=workspace_dir / task.task_id,
                        file_summary=file_summary,
                        assets_dir=assets_dir,
                        memory_context=memory_context,
                        evaluation_policy=evaluation_policy,
                        semantic_mode=semantic_mode,
                    )
                    if attempt:
                        record["retry_attempts"] = attempt
                    return record, memory_trace
                except Exception as exc:  # noqa: BLE001 - preserve batch progress and record the failed task.
                    last_exc = exc
                    memory_trace.policy_decisions.append(
                        {
                            "action": "solve_task",
                            "allowed": False,
                            "reason": "task failed",
                            "error_type": exc.__class__.__name__,
                            "attempt": attempt + 1,
                        }
                    )
                    if attempt < task_retries and retry_delay_seconds > 0:
                        await asyncio.sleep(retry_delay_seconds)
            assert last_exc is not None
            failure_record = {
                "task_id": task.task_id,
                "agent_answer": "",
                "reasoning": None,
                "used_code": False,
                "elapsed_seconds": round(time.time() - started, 3),
                "code_path": str(workspace_dir / task.task_id),
                "error": {
                    "type": last_exc.__class__.__name__,
                    "message": str(last_exc),
                },
            }
            if isinstance(last_exc, SemanticWorkflowError):
                failure_record["semantic_trace"] = last_exc.trace
            return failure_record, memory_trace

    records = []
    output_mode = "a" if resume else "w"
    with output_path.open(output_mode) as handle:
        pending = [asyncio.create_task(solve_with_trace(task)) for task in tasks]
        for completed in asyncio.as_completed(pending):
            record, memory_trace = await completed
            write_memory_learnings(
                record,
                config=memory_config,
                memory_client=memory_client,
                trace=memory_trace,
                evaluation_policy=evaluation_policy,
            )
            experience_routes = record.pop("experience_routes", None)
            if experience_routes:
                memory_trace.route_traces = list(experience_routes)  # type: ignore[arg-type]
            analysis_plan = record.pop("analysis_plan", None)
            if isinstance(analysis_plan, dict):
                memory_trace.analysis_plan = analysis_plan
            workflow_trace = record.get("workflow_trace")
            if isinstance(workflow_trace, dict):
                plan = workflow_trace.get("plan")
                if isinstance(plan, dict) and isinstance(plan.get("analysis_plan"), dict):
                    memory_trace.analysis_plan = plan["analysis_plan"]
                selected_routes = workflow_trace.get("selected_route_ids")
                selected_toolsets = workflow_trace.get("selected_toolsets")
                memory_trace.route_card_ids = [str(route_id) for route_id in selected_routes or []]
                memory_trace.route_traces = [
                    {
                        "route_id": str(route_id),
                        "analysis_layer": "pydantic_graph_route",
                        "helper_functions": [str(toolset) for toolset in selected_toolsets or []],
                    }
                    for route_id in selected_routes or []
                ]
            record = attach_memory_trace(
                record,
                run_mode=memory_config.run_mode,
                memory_router_mode=memory_config.memory_router_mode.value,
                trace=memory_trace,
            )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return records


def _load_completed_task_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = str(record.get("task_id", ""))
        if task_id and "agent_answer" in record and not record.get("error"):
            completed.add(task_id)
    return completed
