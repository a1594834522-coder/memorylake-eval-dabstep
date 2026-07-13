import threading
import time

from dabstep_agent_pydantic.memory_models import MemoryLakeConfig
from dabstep_agent_pydantic.memory_models import MemorySearchHit
from dabstep_agent_pydantic.memory_models import MemoryTrace
from dabstep_agent_pydantic.memory_models import RunMode
from dabstep_agent_pydantic.runner import build_family_query_terms
from dabstep_agent_pydantic.runner import build_memory_context
from dabstep_agent_pydantic.runner import clear_family_search_cache
from dabstep_agent_pydantic.runner import _filter_documents_for_family
from dabstep_agent_pydantic.runner import write_memory_learnings


class RecordingMemoryClient:
    def __init__(self):
        self.calls = []

    def add_memory(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ["event-1"]


class _RouteCard:
    def __init__(self, route_id, keywords=None, triggers=None):
        self.route_id = route_id
        self.keywords = keywords or []
        self.triggers = triggers or []
        self.helper_functions = []
        self.verification_checks = []


class _MemoryHit:
    def __init__(self, hit_id, content):
        self.id = hit_id
        self.content = content


class _StaticSearchClient:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def search_project(self, project_id, *, user_id, query, top_n, threshold):
        self.queries.append(query)
        return {"memories": [], "documents": self.documents}


def _doc(name, text, document_id=None):
    return {
        "document_id": document_id or name,
        "document_name": name,
        "highlight": {"chunks": [{"text": text}]},
    }


def _multi_chunk_doc(name, chunks):
    return {
        "document_id": name,
        "document_name": name,
        "highlight": {"chunks": [{"text": chunk} for chunk in chunks]},
    }


def _memory_config():
    return MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        project_id="project",
        user_id="user",
        top_k=10,
    )


def test_memory_assisted_run_can_disable_memory_writes():
    client = RecordingMemoryClient()
    trace = MemoryTrace()
    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        memory_write_enabled=False,
        project_id="project",
        user_id="user",
    )
    record = {
        "task_id": "synthetic",
        "agent_answer": "42",
        "reasoning": "Use deterministic pandas aggregation and verify the result.",
        "used_code": True,
    }

    write_memory_learnings(record, config=config, memory_client=client, trace=trace)

    assert client.calls == []
    assert trace.created_count == 0
    assert trace.policy_decisions[-1]["reason"] == "memory writes disabled"


def test_build_memory_context_uses_dual_queries_and_dedupes_results():
    clear_family_search_cache()

    class RouteCard:
        route_id = "fee_matching"
        keywords = ["fee rule", "wildcard", "account_type"]
        triggers = ["applicable fee"]
        helper_functions = []
        verification_checks = []

    class MemoryHit:
        def __init__(self, hit_id, content):
            self.id = hit_id
            self.content = content

    class RecordingSearchClient:
        def __init__(self):
            self.queries = []

        def search_project(self, project_id, *, user_id, query, top_n, threshold):
            self.queries.append(query)
            return {
                "memories": [
                    MemoryHit(f"mem-{len(self.queries)}", f"memory from {query}"),
                    MemoryHit("mem-shared", "shared memory"),
                ],
                "documents": [
                    {
                        "document_id": "doc-1",
                        "document_name": "curriculum_fee_matching.md",
                        "highlight": {"chunks": [{"text": "shared doc chunk"}]},
                    }
                ],
            }

    client = RecordingSearchClient()
    context, trace = build_memory_context(
        question="Which fee rules apply to this transaction?",
        guidelines="This formatting guidance must not enter the query.",
        file_summary="Constant file list noise must not enter the query.",
        config=MemoryLakeConfig(
            run_mode=RunMode.MEMORY_ASSISTED,
            memory_enabled=True,
            project_id="project",
            user_id="user",
            top_k=5,
        ),
        memory_client=client,
        route_cards=[RouteCard()],
    )

    assert client.queries == [
        "Which fee rules apply to this transaction?",
        "fee_matching fee rule wildcard account_type applicable fee",
    ]
    assert "formatting guidance" not in "\n".join(client.queries)
    assert "file list noise" not in "\n".join(client.queries)
    assert trace.retrieved_count == 3
    assert trace.document_retrieved_count == 1
    assert context is not None
    assert context.count("shared doc chunk") == 1


def test_build_memory_context_caches_family_query_results():
    clear_family_search_cache()

    class RouteCard:
        route_id = "fee_matching"
        keywords = ["fee rule"]
        triggers = ["applicable fee"]
        helper_functions = []
        verification_checks = []

    class MemoryHit:
        def __init__(self, hit_id, content):
            self.id = hit_id
            self.content = content

    class RecordingSearchClient:
        def __init__(self):
            self.queries = []

        def search_project(self, project_id, *, user_id, query, top_n, threshold):
            self.queries.append(query)
            return {
                "memories": [MemoryHit(query, f"memory from {query}")],
                "documents": [
                    {
                        "document_id": query,
                        "document_name": "curriculum_fee_matching.md",
                        "highlight": {"chunks": [{"text": f"doc from {query}"}]},
                    }
                ],
            }

    config = MemoryLakeConfig(
        run_mode=RunMode.MEMORY_ASSISTED,
        memory_enabled=True,
        project_id="project",
        user_id="user",
        top_k=5,
    )
    client = RecordingSearchClient()

    _first_context, first_trace = build_memory_context(
        question="Which fee rules apply to this transaction?",
        guidelines=None,
        file_summary="",
        config=config,
        memory_client=client,
        route_cards=[RouteCard()],
    )
    second_context, second_trace = build_memory_context(
        question="What total fee should this transaction pay?",
        guidelines=None,
        file_summary="",
        config=config,
        memory_client=client,
        route_cards=[RouteCard()],
    )

    assert client.queries == [
        "Which fee rules apply to this transaction?",
        "fee_matching fee rule applicable fee",
        "What total fee should this transaction pay?",
    ]
    assert second_context is not None
    assert "doc from fee_matching fee rule applicable fee" in second_context
    assert not any(decision.get("reason") == "family query cache hit" for decision in first_trace.policy_decisions)
    assert {
        "action": "search_memory_family_cache",
        "allowed": True,
        "reason": "family query cache hit",
        "query": "fee_matching fee rule applicable fee",
    } in second_trace.policy_decisions


def test_build_memory_context_runs_dual_queries_concurrently():
    clear_family_search_cache()

    class RouteCard:
        route_id = "fee_matching"
        keywords = ["fee rule"]
        triggers = ["applicable fee"]
        helper_functions = []
        verification_checks = []

    class BarrierSearchClient:
        def __init__(self):
            self.barrier = threading.Barrier(2, timeout=0.5)
            self.queries = []

        def search_project(self, project_id, *, user_id, query, top_n, threshold):
            self.queries.append(query)
            self.barrier.wait()
            return {
                "memories": [],
                "documents": [
                    {
                        "document_id": query,
                        "document_name": "curriculum_fee_matching.md",
                        "highlight": {"chunks": [{"text": query}]},
                    }
                ],
            }

    client = BarrierSearchClient()
    started = time.perf_counter()
    context, trace = build_memory_context(
        question="Which fee rules apply to this transaction?",
        guidelines=None,
        file_summary="",
        config=MemoryLakeConfig(
            run_mode=RunMode.MEMORY_ASSISTED,
            memory_enabled=True,
            project_id="project",
            user_id="user",
            top_k=5,
        ),
        memory_client=client,
        route_cards=[RouteCard()],
    )
    elapsed = time.perf_counter() - started

    assert context is not None
    assert trace.document_retrieved_count == 2
    assert set(client.queries) == {
        "Which fee rules apply to this transaction?",
        "fee_matching fee rule applicable fee",
    }
    assert elapsed < 0.5


def test_build_memory_context_filters_curriculum_documents_by_family():
    clear_family_search_cache()
    documents = [
        _doc("curriculum_fee_matching.md", "fee-only rule"),
        _doc("curriculum_fraud_customer.md", "fraud-customer rule"),
        _doc("manual.md", "public manual rule"),
        _doc("curriculum_rules.md", "legacy combined rules"),
    ]

    context, trace = build_memory_context(
        question="What is the fraud rate for repeat customers?",
        guidelines=None,
        file_summary="",
        config=_memory_config(),
        memory_client=_StaticSearchClient(documents),
        route_cards=[_RouteCard("fraud_and_customer_semantics", keywords=["fraud"], triggers=["fraud rate"])],
    )

    assert context is not None
    assert "fraud-customer rule" in context
    assert "public manual rule" in context
    assert "fee-only rule" not in context
    assert "legacy combined rules" not in context
    assert {document["document_name"] for document in trace.retrieved_documents} == {
        "curriculum_fraud_customer.md",
        "manual.md",
    }
    assert {
        "action": "drop_document",
        "reason": "family mismatch",
        "document_name": "curriculum_fee_matching.md",
    } in trace.policy_decisions
    assert {
        "action": "drop_document",
        "reason": "legacy curriculum doc",
        "document_name": "curriculum_rules.md",
    } in trace.policy_decisions


def test_filter_documents_for_family_sabotage_wrong_family_drops_curriculum():
    trace = MemoryTrace()
    documents = [
        _doc("curriculum_fee_matching.md", "fee-only rule"),
        _doc("curriculum_fraud_customer.md", "fraud-customer rule"),
        _doc("manual.md", "manual rule"),
    ]

    filtered = _filter_documents_for_family(documents, "schema_semantics", trace)

    assert [document["document_name"] for document in filtered] == ["manual.md"]
    assert [decision["reason"] for decision in trace.policy_decisions] == [
        "family mismatch",
        "family mismatch",
    ]


def test_filter_documents_for_family_normalizes_drive_renamed_suffixes():
    trace = MemoryTrace()
    documents = [
        _doc("manual_3.md", "manual rule"),
        _doc("curriculum_fee_matching_2.md", "fee rule"),
        _doc("curriculum_fraud_customer_9.md", "fraud rule"),
        _doc("curriculum_rules_4.md", "legacy rule"),
    ]

    filtered = _filter_documents_for_family(documents, "fee_matching", trace)

    assert [document["document_name"] for document in filtered] == [
        "manual.md",
        "curriculum_fee_matching.md",
    ]
    assert [document["document_name_raw"] for document in filtered] == [
        "manual_3.md",
        "curriculum_fee_matching_2.md",
    ]
    assert {
        "action": "drop_document",
        "reason": "family mismatch",
        "document_name": "curriculum_fraud_customer.md",
    } in trace.policy_decisions
    assert {
        "action": "drop_document",
        "reason": "legacy curriculum doc",
        "document_name": "curriculum_rules.md",
    } in trace.policy_decisions


def test_document_filtering_does_not_mutate_cached_results_between_families():
    cached_documents = [
        _doc("curriculum_fee_matching.md", "fee-only rule"),
        _doc("curriculum_fraud_customer.md", "fraud-customer rule"),
        _doc("payments-readme.md", "payments readme rule"),
    ]

    fee_docs = _filter_documents_for_family(cached_documents, "fee_matching", MemoryTrace())
    fraud_docs = _filter_documents_for_family(cached_documents, "customer_fraud_metrics", MemoryTrace())

    assert [document["document_name"] for document in fee_docs] == [
        "curriculum_fee_matching.md",
        "payments-readme.md",
    ]
    assert [document["document_name"] for document in fraud_docs] == [
        "curriculum_fraud_customer.md",
        "payments-readme.md",
    ]
    assert [document["document_name"] for document in cached_documents] == [
        "curriculum_fee_matching.md",
        "curriculum_fraud_customer.md",
        "payments-readme.md",
    ]


def test_build_memory_context_uses_complete_document_chunk_budget():
    clear_family_search_cache()
    chunks = [f"chunk {index} complete-tail-{index}" for index in range(1, 7)]
    context, trace = build_memory_context(
        question="Which fee rules apply?",
        guidelines=None,
        file_summary="",
        config=_memory_config(),
        memory_client=_StaticSearchClient([_multi_chunk_doc("curriculum_fee_matching.md", chunks)]),
        route_cards=[_RouteCard("fee_matching", keywords=["fee"], triggers=["fee"])],
    )

    assert context is not None
    for index in range(1, 5):
        assert f"chunk {index} complete-tail-{index}" in context
    assert "chunk 5" not in context
    assert "chunk 6" not in context
    assert "complete-tail-4" in context
    assert trace.document_chunks_dropped == 2
    assert trace.document_context_truncated is True
    assert {
        "action": "drop_document_chunk",
        "reason": "chunk budget exceeded",
        "dropped": 2,
    } in trace.policy_decisions


def test_build_memory_context_keeps_chunks_between_1200_and_2500_chars():
    clear_family_search_cache()
    medium_chunk = "M" * 1800
    context, trace = build_memory_context(
        question="Which fee rules apply?",
        guidelines=None,
        file_summary="",
        config=_memory_config(),
        memory_client=_StaticSearchClient(
            [_multi_chunk_doc("curriculum_fee_matching.md", [medium_chunk, "normal complete chunk"])]
        ),
        route_cards=[_RouteCard("fee_matching", keywords=["fee"], triggers=["fee"])],
    )

    assert context is not None
    assert medium_chunk in context
    assert "normal complete chunk" in context
    assert trace.document_chunks_dropped == 0
    assert trace.document_context_truncated is False


def test_build_memory_context_drops_oversized_document_chunks_without_truncating():
    clear_family_search_cache()
    oversized = "X" * 2501
    context, trace = build_memory_context(
        question="Which fee rules apply?",
        guidelines=None,
        file_summary="",
        config=_memory_config(),
        memory_client=_StaticSearchClient(
            [_multi_chunk_doc("curriculum_fee_matching.md", [oversized, "normal complete chunk"])]
        ),
        route_cards=[_RouteCard("fee_matching", keywords=["fee"], triggers=["fee"])],
    )

    assert context is not None
    assert "normal complete chunk" in context
    assert "X" * 200 not in context
    assert trace.document_chunks_dropped == 1
    assert trace.document_context_truncated is True
    assert {
        "action": "drop_document_chunk",
        "reason": "chunk too long",
        "document_name": "curriculum_fee_matching.md",
        "length": 2501,
    } in trace.policy_decisions


def test_build_memory_context_records_retrieval_observability():
    class RouteCard:
        route_id = "fee_matching"
        keywords = ["fee rule"]
        triggers = ["applicable fee"]
        helper_functions = []
        verification_checks = []

    class RecordingSearchClient:
        def search_project(self, *args, **kwargs):
            return {
                "memories": [
                    MemorySearchHit(
                        id="mem-1",
                        content="Use generic wildcard matching semantics.",
                        score=0.87,
                    )
                ],
                "documents": [
                    {
                        "document_id": "doc-1",
                        "document_name": "curriculum_fee_matching.md",
                        "score": 0.91,
                        "highlight": {"chunks": [{"text": "Public document excerpt."}]},
                    }
                ],
            }

    context, trace = build_memory_context(
        question="Which fee rules apply?",
        guidelines="Return one value.",
        file_summary="",
        config=MemoryLakeConfig(
            run_mode=RunMode.MEMORY_ASSISTED,
            memory_enabled=True,
            project_id="project",
            user_id="user",
            top_k=7,
            threshold=0.42,
            rerank=True,
        ),
        memory_client=RecordingSearchClient(),
        route_cards=[RouteCard()],
        asset_fingerprint="sha256:test-fingerprint",
    )

    assert context is not None
    assert trace.search_query == "Which fee rules apply?"
    assert trace.search_queries == [
        "Which fee rules apply?",
        "fee_matching fee rule applicable fee",
    ]
    assert trace.retrieval_params == {"top_k": 7, "threshold": 0.42, "rerank": True}
    assert trace.asset_fingerprint == "sha256:test-fingerprint"
    assert trace.retrieved == [
        {
            "id": "mem-1",
            "content_preview": "Use generic wildcard matching semantics.",
            "score": 0.87,
        }
    ]
    assert trace.retrieved_documents[0]["score"] == 0.91


def test_build_memory_context_formats_document_chunks_with_verified_heading():
    class MemoryHit:
        def __init__(self, hit_id, content):
            self.id = hit_id
            self.content = content

    class SearchClient:
        def search_project(self, *args, **kwargs):
            return {
                "memories": [MemoryHit("mem-1", "Reusable generic rule.")],
                "documents": [
                    {
                        "document_id": "doc-1",
                        "document_name": "manual.md",
                        "highlight": {
                            "chunks": [
                                {"text": "First public manual excerpt."},
                                {"text": "Second public manual excerpt."},
                            ]
                        },
                    }
                ],
            }

    context, trace = build_memory_context(
        question="How should fee rules be matched?",
        guidelines=None,
        file_summary="",
        config=MemoryLakeConfig(
            run_mode=RunMode.MEMORY_ASSISTED,
            memory_enabled=True,
            project_id="project",
            user_id="user",
        ),
        memory_client=SearchClient(),
        route_cards=[],
    )

    assert context is not None
    assert "Verified domain rules and documentation excerpts (verify before relying on them):" in context
    assert "- [manual.md] First public manual excerpt." in context
    assert "- [manual.md] Second public manual excerpt." in context
    assert "MemoryLake document context:" not in context
    assert trace.document_context_truncated is False


def test_build_memory_context_drops_service_merged_summary_memories():
    class MemoryHit:
        def __init__(self, hit_id, content):
            self.id = hit_id
            self.content = content

    class SearchClient:
        def search_project(self, *args, **kwargs):
            return {
                "memories": [
                    MemoryHit("merged", "Wants to store a title-only server summary."),
                    MemoryHit("normal", "Use schema-level fee wildcard semantics."),
                ],
                "documents": [],
            }

    context, trace = build_memory_context(
        question="Which fee rules apply?",
        guidelines=None,
        file_summary="",
        config=MemoryLakeConfig(
            run_mode=RunMode.MEMORY_ASSISTED,
            memory_enabled=True,
            project_id="project",
            user_id="user",
        ),
        memory_client=SearchClient(),
        route_cards=[],
    )

    assert context is not None
    assert "Use schema-level fee wildcard semantics." in context
    assert "Wants to store" not in context
    assert {
        "action": "inject_memory",
        "allowed": False,
        "reason": "dropped service merged summary memory",
        "memory_id": "merged",
    } in trace.policy_decisions


def test_family_query_terms_come_only_from_route_card_keywords_and_triggers():
    class RouteCard:
        route_id = "fee_matching"
        keywords = ["fee rule", "wildcard"]
        triggers = ["applicable fee"]
        helper_functions = ["real_helper_name_must_not_be_used"]
        instructions = ["instruction text must not be used"]

    terms = build_family_query_terms("fee_matching", [RouteCard()])

    assert terms == ["fee rule", "wildcard", "applicable fee"]
