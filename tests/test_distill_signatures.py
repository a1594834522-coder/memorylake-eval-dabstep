import pandas as pd
import pytest

from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.distill.discriminate import (
    ReferenceRecord,
    discriminate_template,
    load_reference,
)
from dabstep_agent_pydantic.distill.signatures import SignatureError, compile_signature
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec, InterpretationSpec, OutputSpec


def _data() -> DABStepData:
    return DABStepData(
        fees=pd.DataFrame(
            [
                {"ID": 1, "card_scheme": "SwiftCharge", "account_type": [], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": True, "aci": ["A"], "fixed_amount": 0.10, "rate": 100, "intracountry": None},
                {"ID": 2, "card_scheme": "SwiftCharge", "account_type": ["H"], "capture_delay": None,
                 "monthly_fraud_level": None, "monthly_volume": None, "merchant_category_code": [],
                 "is_credit": None, "aci": [], "fixed_amount": 0.50, "rate": 0, "intracountry": None},
            ]
        ),
        payments=pd.DataFrame([{"merchant": "M_X", "year": 2023, "day_of_year": 1, "card_scheme": "SwiftCharge",
                                "is_credit": True, "aci": "A", "eur_amount": 1.0, "issuing_country": "NL",
                                "acquirer": "a", "has_fraudulent_dispute": False}]),
        merchants=pd.DataFrame([{"merchant": "M_X", "account_type": "H", "capture_delay": "1",
                                 "merchant_category_code": 7372, "acquirer": ["a"]}]),
        acquirer_countries=pd.DataFrame([{"acquirer": "a", "country_code": "NL"}]),
        merchant_category_codes=pd.DataFrame([{"mcc": 7372, "description": "Computer programming"}]),
    )


def test_signature_binds_letter_placeholders_by_context():
    sig = compile_signature(
        "What is the fee ID or IDs that apply to account_type = <LETTER> and aci = <LETTER>?"
    )
    params = sig.parse("What is the fee ID or IDs that apply to account_type = h and aci = a?", _data())
    assert params == {"account_type": "H", "aci": "A"}


def test_signature_binds_amount_scheme_and_credit_constant():
    sig = compile_signature(
        "For credit transactions, what would be the average fee that the card scheme <SCHEME> "
        "would charge for a transaction value of <N> EUR?"
    )
    params = sig.parse(
        "For credit transactions, what would be the average fee that the card scheme SwiftCharge "
        "would charge for a transaction value of 100 EUR?",
        _data(),
    )
    assert params == {"is_credit": True, "card_scheme": "SwiftCharge", "amount": 100.0}


def test_signature_rejects_unbindable_placeholder():
    with pytest.raises(SignatureError):
        compile_signature("A mystery quantity <N> with no binding context at all?")


def test_signature_no_match_returns_none():
    sig = compile_signature("total fees for <MERCHANT> in <YEAR>")
    assert sig.parse("completely different question", _data()) is None


def test_load_reference_high_confidence_rules(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(
        '{"primary": {"1": "A", "2": "B", "3": "C"},'
        ' "candidates": {"2": {"B": 90, "X": 10}, "3": {"C": 5, "Y": 5}},'
        ' "ambiguous": ["2", "3"], "resolved": [], "stats": {}}'
    )
    records = load_reference(path)
    assert records["1"].high_confidence          # resolved
    assert records["2"].high_confidence          # ambiguous but 90% vote ratio
    assert not records["3"].high_confidence      # ambiguous at 50%


def test_discriminate_adopts_manual_consistent_winner():
    data = _data()
    sig = compile_signature(
        "What is the fee ID or IDs that apply to account_type = <LETTER> and aci = <LETTER>?"
    )
    candidates = [
        InterpretationSpec(name="wildcard_aware", population="fee_rules",
                           fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"],
                                                  value="rule_id", reducer="collect_ids"),
                           output=OutputSpec(kind="id_list"), manual_citation="manual §5"),
        InterpretationSpec(name="explicit_only", population="fee_rules",
                           fee_rules=FeeRulesSpec(context_dims=["account_type", "aci"],
                                                  value="rule_id", reducer="collect_ids",
                                                  wildcard_policy="strict"),
                           output=OutputSpec(kind="id_list"), manual_citation="strict reading",
                           contradicts_manual=True),
    ]
    instances = [{"task_id": "1", "question": "What is the fee ID or IDs that apply to account_type = H and aci = A?", "guidelines": ""}]
    reference = {"1": ReferenceRecord(task_id="1", answer="1, 2", high_confidence=True)}
    report = discriminate_template(data=data, template=sig.template, instances=instances,
                                   candidates=candidates, signature=sig, reference=reference)
    assert report.funnel == {"instances": 1, "high_confidence": 1, "participated": 1}
    assert report.adopted == "wildcard_aware"
    rates = {c.spec.name: c.rate for c in report.candidates}
    assert rates == {"wildcard_aware": 1.0, "explicit_only": 0.0}
