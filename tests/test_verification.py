import json

from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.verification import verify_record
from dabstep_agent_pydantic import verification


def _make_data_dir(tmp_path):
    (tmp_path / "fees.json").write_text(json.dumps([]), encoding="utf-8")
    return tmp_path


def test_verify_flags_empty_answer(tmp_path):
    task = Task(task_id="t1", question="How many merchants?", guidelines="Answer with a number.")
    feedback = verify_record({"agent_answer": ""}, task=task, plan=None, data_dir=tmp_path)
    assert feedback is not None and "non-empty" in feedback


def test_verify_allows_empty_answer_when_guideline_asks_for_empty_string(tmp_path):
    task = Task(task_id="t1", question="List applicable IDs.", guidelines="If none, reply with an empty string.")
    assert verify_record({"agent_answer": ""}, task=task, plan=None, data_dir=tmp_path) is None


def test_verify_flags_decimal_place_contract_violation(tmp_path):
    task = Task(task_id="t1", question="How much?", guidelines="Answer must be rounded to 6 decimals.")
    feedback = verify_record({"agent_answer": "12.34"}, task=task, plan=None, data_dir=tmp_path)
    assert feedback is not None and "6 decimal" in feedback
    assert verify_record({"agent_answer": "12.340000"}, task=task, plan=None, data_dir=tmp_path) is None


def test_verify_allows_not_applicable_for_decimal_contract(tmp_path):
    task = Task(task_id="t1", question="How much?", guidelines="Rounded to 2 decimals or Not Applicable.")
    assert verify_record({"agent_answer": "Not Applicable"}, task=task, plan=None, data_dir=tmp_path) is None


def test_verify_flags_non_comma_list(tmp_path):
    task = Task(task_id="t1", question="Which merchants?", guidelines="Answer with a comma separated list.")
    feedback = verify_record({"agent_answer": "Alpha_One and Beta_Two"}, task=task, plan=None, data_dir=tmp_path)
    assert feedback is not None and "comma" in feedback


def test_verify_recomputes_total_fees_and_flags_mismatch(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch.setattr(verification, "load_dabstep_data", lambda path: object())
    monkeypatch.setattr(
        verification,
        "total_fees_for_merchant_period",
        lambda data, *, merchant, year, month, day_of_year=None: 1234.56,
    )
    task = Task(
        task_id="t1",
        question="What are the total fees that Example_Merchant paid in March 2023?",
        guidelines="Answer must be rounded to 2 decimals.",
    )
    feedback = verify_record({"agent_answer": "1000.00"}, task=task, plan=None, data_dir=data_dir)
    assert feedback is not None
    assert "1234.56" in feedback
    assert "total_fees_for_merchant_period" in feedback

    assert verify_record({"agent_answer": "1234.56"}, task=task, plan=None, data_dir=data_dir) is None


def test_verify_total_fees_passes_day_of_year_scope(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)
    captured = {}

    def fake_total(data, *, merchant, year, month, day_of_year=None):
        captured.update(merchant=merchant, year=year, month=month, day_of_year=day_of_year)
        return 29.93

    monkeypatch.setattr(verification, "load_dabstep_data", lambda path: object())
    monkeypatch.setattr(verification, "total_fees_for_merchant_period", fake_total)
    task = Task(
        task_id="t1",
        question="For the 10th of the year 2023, what is the total fees (in euros) that Example_Merchant should pay?",
        guidelines="Answer must be rounded to 2 decimals.",
    )
    assert verify_record({"agent_answer": "29.93"}, task=task, plan=None, data_dir=data_dir) is None
    assert captured == {"merchant": "Example_Merchant", "year": 2023, "month": None, "day_of_year": 10}


def test_verify_total_fees_skips_conditioned_scopes(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)

    def _fail(*args, **kwargs):
        raise AssertionError("conditioned total-fee questions must not be recomputed")

    monkeypatch.setattr(verification, "load_dabstep_data", _fail)
    task = Task(
        task_id="t1",
        question="In 2023, what were the total fees that Example_Merchant paid on the card scheme SchemeName?",
        guidelines="Answer must be rounded to 2 decimals.",
    )
    assert verify_record({"agent_answer": "12.00"}, task=task, plan=None, data_dir=data_dir) is None


def test_verify_recomputes_relative_fee_delta(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch.setattr(verification, "load_dabstep_data", lambda path: object())
    monkeypatch.setattr(
        verification,
        "fee_rate_delta_for_period",
        lambda data, *, merchant, year, month, fee_id, new_rate: -0.5,
    )
    task = Task(
        task_id="t1",
        question="In January 2023 what delta would Example_Merchant pay if the relative fee of the fee with ID=64 changed to 1?",
        guidelines="Answer must be rounded to 14 decimals.",
    )
    feedback = verify_record(
        {"agent_answer": "0.10000000000000"},
        task=task,
        plan=None,
        data_dir=data_dir,
    )
    assert feedback is not None and "fee_rate_delta_for_period" in feedback


def test_verify_flags_aci_steering_mismatch(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch.setattr(verification, "load_dabstep_data", lambda path: object())
    monkeypatch.setattr(
        verification,
        "optimize_aci_for_fraudulent_transactions",
        lambda data, *, merchant, year, month: {"aci": "E", "cost": 1.0, "formatted": "E:1.00"},
    )
    task = Task(
        task_id="t1",
        question=(
            "For the year 2023 and at the merchant Example_Merchant, if we were to move the fraudulent "
            "transactions to a different ACI to incentivize the lowest possible fees, which ACI is preferred?"
        ),
        guidelines="Answer with the ACI letter.",
    )
    feedback = verify_record({"agent_answer": "C:12.00"}, task=task, plan=None, data_dir=data_dir)
    assert feedback is not None and "ACI E" in feedback

    assert verify_record({"agent_answer": "E:1.00"}, task=task, plan=None, data_dir=data_dir) is None


def test_verify_skips_recomputation_for_deterministic_route(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)

    def _fail(*args, **kwargs):
        raise AssertionError("recomputation should not run for deterministic answers")

    monkeypatch.setattr(verification, "load_dabstep_data", _fail)
    task = Task(
        task_id="t1",
        question="What are the total fees that Example_Merchant paid in March 2023?",
        guidelines="Answer must be rounded to 2 decimals.",
    )
    record = {"agent_answer": "1000.00", "deterministic_route": "total_fees"}
    assert verify_record(record, task=task, plan=None, data_dir=data_dir) is None


def test_verify_recomputation_errors_never_fail_the_task(monkeypatch, tmp_path):
    data_dir = _make_data_dir(tmp_path)

    def _boom(path):
        raise RuntimeError("corrupted data")

    monkeypatch.setattr(verification, "load_dabstep_data", _boom)
    task = Task(
        task_id="t1",
        question="What are the total fees that Example_Merchant paid in March 2023?",
        guidelines="Answer must be rounded to 2 decimals.",
    )
    assert verify_record({"agent_answer": "1000.00"}, task=task, plan=None, data_dir=data_dir) is None
