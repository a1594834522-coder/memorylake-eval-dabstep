import asyncio

import pandas as pd

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.bootstrap import (
    answer_matrix,
    bootstrap_labels,
    disagreement_instances,
    is_unanimous,
    select_labeling_instances,
)
from dabstep_agent_pydantic.distill.signatures import compile_signature
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec, InterpretationSpec, OutputSpec

TEMPLATE = "What is the fee ID or IDs that apply to account_type = <LETTER> and aci = <LETTER>?"


def _data() -> DABStepData:
    return DABStepData(
        fees=pd.DataFrame(
            [
                {"ID": 1, "card_scheme": "S", "account_type": [], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": True, "aci": ["A"], "fixed_amount": 0.1, "rate": 10, "intracountry": None},
                {"ID": 2, "card_scheme": "S", "account_type": ["H"], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": None, "aci": [], "fixed_amount": 0.5, "rate": 0, "intracountry": None},
            ]
        ),
        payments=pd.DataFrame([{"merchant": "M_X", "year": 2023, "day_of_year": 1, "card_scheme": "S",
                                "is_credit": True, "aci": "A", "eur_amount": 1.0, "issuing_country": "NL",
                                "acquirer": "a", "has_fraudulent_dispute": False}]),
        merchants=pd.DataFrame([{"merchant": "M_X", "account_type": "H", "capture_delay": "1",
                                 "merchant_category_code": 1, "acquirer": ["a"]}]),
        acquirer_countries=pd.DataFrame([{"acquirer": "a", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 1, "description": "d"}]),
    )


def _candidates():
    wildcard = InterpretationSpec(
        name="wildcard", population="fee_rules",
        fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"], value="rule_id", reducer="collect_ids"),
        output=OutputSpec(kind="id_list"), manual_citation="manual",
    )
    strict = wildcard.model_copy(update={
        "name": "strict",
        "fee_rules": wildcard.fee_rules.model_copy(update={"wildcard_policy": "strict"}),
        "contradicts_manual": True,
    })
    return [wildcard, strict]


def _instances(n=4):
    return [
        {"task_id": str(i), "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?",
         "guidelines": ""}
        for i in range(1, n + 1)
    ]


def test_matrix_and_disagreement_zero_token():
    matrix = answer_matrix(data=_data(), candidates=_candidates(),
                           signature=compile_signature(TEMPLATE), instances=_instances())
    assert matrix["wildcard"]["1"] == "1, 2"
    assert matrix["strict"]["1"] == ""     # strict matches nothing here
    seps = disagreement_instances(matrix)
    assert all(("strict", "wildcard") in {tuple(sorted(p)) for p in pairs} for pairs in seps.values())
    assert not is_unanimous(matrix)


def test_unanimous_matrix_needs_no_labels():
    matrix = {"a": {"1": "5", "2": "6"}, "b": {"1": "5.0", "2": "6.00"}}  # numerically equal
    assert is_unanimous(matrix)
    assert select_labeling_instances(matrix) == []


def test_greedy_selection_prefers_max_pair_coverage():
    matrix = {
        "a": {"1": "x", "2": "x", "3": "x"},
        "b": {"1": "y", "2": "x", "3": "x"},   # separated from a only on tid 1
        "c": {"1": "z", "2": "z", "3": "x"},   # separated on 1 and 2
    }
    chosen = select_labeling_instances(matrix, max_labels=1)
    assert chosen == ["1"]  # tid 1 separates all three pairs at once


class _FakeSolver:
    def __init__(self, answer):
        self._answer = answer
        self.calls = 0

    async def run(self, prompt, **kwargs):
        self.calls += 1
        answer = self._answer

        class _R:
            class output:
                agent_answer = answer
        return _R()


def test_sequential_bootstrap_stops_at_single_survivor(tmp_path):
    data = _data()
    candidates = _candidates()
    sig = compile_signature(TEMPLATE)
    instances = _instances(4)
    matrix = answer_matrix(data=data, candidates=candidates, signature=sig, instances=instances)
    label_tids = select_labeling_instances(matrix, max_labels=4)
    solver = _FakeSolver("1, 2")  # model agrees with wildcard reading
    labels = asyncio.run(bootstrap_labels(
        instances=instances, label_tids=label_tids, data_dir=tmp_path,
        workspace_dir=tmp_path, samples=3, consensus=3, agent=solver, matrix=matrix,
    ))
    # first labeled instance already eliminates 'strict' -> early stop after one tid
    assert len(labels) == 1
    assert solver.calls == 3  # 3 samples for exactly one instance
    record = next(iter(labels.values()))
    assert record.source == "self_bootstrap" and record.high_confidence


def test_majority_cluster_tolerates_precision_variance():
    from dabstep_agent_pydantic.distill.bootstrap import _majority_cluster, _representative

    # Same computed value at different printed precision must form one cluster.
    answers = ["23.834676", "23.8347", "23.83", "99.0", "2172.02"]
    cluster = _majority_cluster(answers)
    assert set(cluster) == {"23.834676", "23.8347", "23.83"}
    # The most precise form is kept so downstream tolerance stays tight.
    assert _representative(cluster) == "23.834676"


class _VaryingSolver:
    """Same value, different printed precision per sample."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    async def run(self, prompt, **kwargs):
        answer = self._answers[self.calls % len(self._answers)]
        self.calls += 1

        class _R:
            class output:
                agent_answer = answer
        return _R()


def test_bootstrap_consensus_uses_precision_aware_clustering(tmp_path):
    from dabstep_agent_pydantic.distill.bootstrap import bootstrap_labels

    instances = [{"task_id": "7", "question": "q", "guidelines": ""}]
    solver = _VaryingSolver(["14.49", "14.490000", "14.5"])
    labels = asyncio.run(bootstrap_labels(
        instances=instances, label_tids=["7"], data_dir=tmp_path,
        workspace_dir=tmp_path, samples=3, consensus=3, agent=solver,
        persist_path=tmp_path / "labels.jsonl",
    ))
    # Old exact-match consensus rejected this batch; clustering accepts it.
    assert "7" in labels
    assert labels["7"].answer == "14.490000"


def test_discrimination_sample_always_includes_referenced_instances():
    """Labeled instances (paid evidence) must survive max_instances sampling."""
    from dabstep_agent_pydantic.distill.discriminate import ReferenceRecord, discriminate_template

    data = _data()
    candidates = _candidates()
    sig = compile_signature(TEMPLATE)
    instances = _instances(30)
    # Put the only reference on an instance the old stride sampling skipped.
    labeled_tid = str(instances[17]["task_id"])
    reference = {labeled_tid: ReferenceRecord(
        task_id=labeled_tid,
        answer="1, 2",
        high_confidence=True,
        source="self_bootstrap",
    )}
    report = discriminate_template(
        data=data, template=TEMPLATE, instances=instances,
        candidates=candidates, signature=sig, reference=reference,
        max_instances=12,
    )
    assert report.funnel["high_confidence"] == 1
    assert report.funnel["participated"] == 1


def test_failed_consensus_batches_are_logged_for_postmortem(tmp_path):
    import json as _json

    from dabstep_agent_pydantic.distill.bootstrap import bootstrap_labels

    instances = [{"task_id": "9", "question": "q", "guidelines": ""}]
    solver = _VaryingSolver(["1.0", "250.0", "999.0"])  # no majority cluster
    labels = asyncio.run(bootstrap_labels(
        instances=instances, label_tids=["9"], data_dir=tmp_path,
        workspace_dir=tmp_path, samples=3, consensus=3, agent=solver,
        persist_path=tmp_path / "labels.jsonl",
    ))
    assert labels == {}
    rows = [_json.loads(line) for line in
            (tmp_path / "_bootstrap_failed_batches.jsonl").read_text().splitlines()]
    assert rows[0]["task_id"] == "9"
    assert set(rows[0]["answers"]) == {"1.0", "250.0", "999.0"}
