import json

from dabstep_agent_pydantic.curriculum import CurriculumMemory
from dabstep_agent_pydantic.curriculum import build_curriculum_prompt
from dabstep_agent_pydantic.curriculum import load_forbidden_entity_terms
from dabstep_agent_pydantic.curriculum import sanitize_curriculum_memories
from dabstep_agent_pydantic.curriculum import write_curriculum_memories


def _memory(
    content: str,
    *,
    title: str = "rule",
    category: str = "metric_definitions",
    evidence: str = "verified against the docs",
) -> CurriculumMemory:
    return CurriculumMemory(title=title, content=content, category=category, evidence=evidence)


GOOD_RULE = (
    "The domain fraud rate is defined as fraudulent EUR volume divided by total EUR volume, "
    "not as a count of fraudulent transactions over total transactions."
)


def test_sanitizer_accepts_generic_semantic_rules():
    allowed, rejections = sanitize_curriculum_memories([_memory(GOOD_RULE)], forbidden_terms=["Some_Merchant"])
    assert [memory.content for memory in allowed] == [GOOD_RULE]
    assert rejections == []


def test_sanitizer_rejects_entity_references():
    memory = _memory("For Some_Merchant the fee rules always match account type wildcards in this dataset schema.")
    allowed, rejections = sanitize_curriculum_memories([memory], forbidden_terms=["Some_Merchant"])
    assert allowed == []
    assert rejections[0]["reason"] == "references a real business entity"


def test_sanitizer_rejects_task_references_and_answer_like_values():
    task_ref = _memory("As seen in task 1234, null fee-rule fields must be treated as wildcards when matching.")
    answer_like = _memory("The total for the relevant period equals 6764.6088 when every matching rule is summed up.")
    allowed, rejections = sanitize_curriculum_memories([task_ref, answer_like], forbidden_terms=[])
    assert allowed == []
    assert {rejection["reason"] for rejection in rejections} == {
        "references a benchmark task",
        "contains an answer-like computed value",
    }


def test_sanitizer_rejects_answer_like_values_in_title_content_and_evidence():
    title_value = _memory(
        GOOD_RULE,
        title="12.34 EUR result rule",
    )
    content_value = _memory(
        "After filtering the payments table, there are 72 rows in the computed subset.",
    )
    evidence_value = _memory(
        GOOD_RULE,
        evidence="Python check returned 19 merchants after filtering.",
    )
    allowed, rejections = sanitize_curriculum_memories(
        [title_value, content_value, evidence_value],
        forbidden_terms=[],
    )

    assert allowed == []
    assert [rejection["reason"] for rejection in rejections] == [
        "contains an answer-like computed value",
        "contains an answer-like computed value",
        "contains an answer-like computed value",
    ]


def test_sanitizer_allows_numeric_tokens_quoted_from_public_docs(tmp_path):
    (tmp_path / "manual.md").write_text(
        "The monthly_fraud_level example '7.7%-8.3%' means the ratio is between 7.7 and 8.3 percent.\n"
        "The fee then is provided by fee = fixed_amount + rate * transaction_value / 10000.",
        encoding="utf-8",
    )
    (tmp_path / "payments-readme.md").write_text("Column descriptions only.", encoding="utf-8")
    memory = _memory(
        "For monthly_fraud_level fee matching, the documented '7.7%-8.3%' range means the "
        "merchant-month fraud ratio must fall between 7.7 and 8.3 percent.",
        category="fee_matching_semantics",
        evidence="Manual wording was checked directly; no dataset result was recorded.",
    )

    allowed, rejections = sanitize_curriculum_memories([memory], forbidden_terms=[], data_dir=tmp_path)

    assert [item.content for item in allowed] == [memory.content]
    assert rejections == []


def test_sanitizer_still_rejects_computed_values_not_quoted_from_public_docs(tmp_path):
    (tmp_path / "manual.md").write_text("The fee formula divides by 10000.", encoding="utf-8")
    computed_decimal = _memory(
        "The documentation says null fee fields are wildcard fields, and a Python check produced 0.14.",
        category="fee_matching_semantics",
    )
    computed_count = _memory(
        GOOD_RULE,
        evidence="Python check returned 138,236 payments after filtering.",
    )

    allowed, rejections = sanitize_curriculum_memories(
        [computed_decimal, computed_count],
        forbidden_terms=[],
        data_dir=tmp_path,
    )

    assert allowed == []
    assert [rejection["reason"] for rejection in rejections] == [
        "contains an answer-like computed value",
        "contains an answer-like computed value",
    ]


def test_sanitizer_allows_schema_constants_in_general_rules():
    memory = _memory(
        "Fee rate fields use basis points, so the rate component is rate multiplied by EUR amount divided by 10000.",
        category="fee_matching_semantics",
    )

    allowed, rejections = sanitize_curriculum_memories([memory], forbidden_terms=[])

    assert [item.content for item in allowed] == [memory.content]
    assert rejections == []


def test_sanitizer_rejects_duplicates_and_fragments():
    fragment = _memory("Null means wildcard here always anyway.")
    duplicate = _memory(GOOD_RULE)
    allowed, rejections = sanitize_curriculum_memories(
        [_memory(GOOD_RULE), duplicate, fragment], forbidden_terms=[]
    )
    assert len(allowed) == 1
    assert {rejection["reason"] for rejection in rejections} == {
        "duplicate content",
        "content too short to be a self-contained rule",
    }


def test_sanitizer_rejects_unknown_category():
    memory = _memory(GOOD_RULE, category="benchmark_answers")
    allowed, rejections = sanitize_curriculum_memories([memory], forbidden_terms=[])
    assert allowed == []
    assert "unknown category" in rejections[0]["reason"]


def test_load_forbidden_entity_terms_reads_merchant_names(tmp_path):
    (tmp_path / "merchant_data.json").write_text(
        json.dumps([{"merchant": "Alpha_Shop"}, {"merchant": "Beta_Store"}, {"merchant": ""}]),
        encoding="utf-8",
    )
    assert load_forbidden_entity_terms(tmp_path) == ["Alpha_Shop", "Beta_Store"]


def test_curriculum_prompt_embeds_public_docs_only(tmp_path):
    (tmp_path / "manual.md").write_text("# Manual\nNull fee fields are wildcards.", encoding="utf-8")
    (tmp_path / "payments-readme.md").write_text("# Readme\nColumn semantics.", encoding="utf-8")
    (tmp_path / "other.md").write_text("should not be embedded", encoding="utf-8")
    prompt = build_curriculum_prompt(tmp_path)
    assert "Null fee fields are wildcards." in prompt
    assert "Column semantics." in prompt
    assert "should not be embedded" not in prompt


def test_write_curriculum_memories_marks_official_safe_metadata():
    calls = []

    class FakeClient:
        def add_memory(self, project_id, **kwargs):
            calls.append((project_id, kwargs))
            return [f"event-{len(calls)}"]

    results = write_curriculum_memories(
        [_memory(GOOD_RULE)],
        client=FakeClient(),
        project_id="project",
        user_id="user",
        session_prefix="curriculum-test",
    )
    assert results == [{"title": "rule", "event_ids": ["event-1"]}]
    project_id, kwargs = calls[0]
    assert project_id == "project"
    assert kwargs["infer"] is False
    assert kwargs["metadata"]["asset_type"] == "curriculum"
    assert kwargs["metadata"]["source"] == "public_docs"
    assert kwargs["metadata"]["official_safe"] == "true"
    assert kwargs["metadata"]["contains_answer"] == "false"
    assert kwargs["messages"][1]["content"] == GOOD_RULE
