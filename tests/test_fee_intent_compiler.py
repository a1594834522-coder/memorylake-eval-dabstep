"""Fee intent compilation and mechanism-level real-data parity."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dabstep_agent_pydantic.analysis_executor import execute_analysis
from dabstep_agent_pydantic.analysis_spec_v2 import InterpretationMeasure
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.distill.spec import OutputSpec
from dabstep_agent_pydantic.distill.spec import PaymentsSpec
from dabstep_agent_pydantic.fee_spec_adapter import adapt_interpretation_spec
from dabstep_agent_pydantic.intent_compiler import IntentCompilationError
from dabstep_agent_pydantic.intent_compiler import compile_semantic_intent
from dabstep_agent_pydantic.semantic_intent import FeeIntent


def test_compile_period_total_fees():
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023, "month": 1},
        output_kind="decimal",
        decimals=2,
        fee_reducer="sum_all_matching",
    )
    spec = compile_semantic_intent(intent)
    assert isinstance(spec.measure, InterpretationMeasure)
    assert spec.source.table == "payments"
    assert spec.measure.interpretation.payments.primitive == "period_total_fees"
    assert spec.measure.interpretation.payments.reducer == "sum_all_matching"
    assert spec.filters == []
    assert spec.group_by == []
    assert spec.measure.params["merchant"] == "Merchant_A"


def test_compile_applicable_fee_ids():
    intent = FeeIntent(
        operation="applicable_fee_ids",
        params={"merchant": "Merchant_A", "year": 2023, "month": 12},
        output_kind="comma_list",
    )
    spec = compile_semantic_intent(intent)
    assert spec.measure.interpretation.payments.primitive == "applicable_fee_ids_period"
    assert spec.output.kind == "comma_list"


def test_compile_fee_at_amount():
    intent = FeeIntent(
        operation="fee_at_amount",
        params={"amount": 100.0, "card_scheme": "NexPay", "account_type": "H"},
        output_kind="decimal",
        decimals=6,
    )
    spec = compile_semantic_intent(intent)
    fr = spec.measure.interpretation.fee_rules
    assert fr is not None
    assert fr.value == "fee_at_amount"
    assert fr.reducer == "mean"
    assert fr.wildcard_policy == "manual"
    assert set(fr.context_dims) >= {"card_scheme", "account_type"}
    assert spec.source.table == "fees"


def test_compile_aci_extreme():
    intent = FeeIntent(
        operation="aci_extreme",
        params={"amount": 50.0, "card_scheme": "NexPay"},
        objective="max",
        output_kind="comma_list",
    )
    spec = compile_semantic_intent(intent)
    fr = spec.measure.interpretation.fee_rules
    assert fr.group_by == "aci"
    assert fr.group_extreme == "argmax"
    assert fr.reducer == "sum"


def test_merchant_aci_steering_falls_back_from_static_fee_extreme():
    intent = FeeIntent(
        operation="aci_extreme",
        params={"amount": 0, "merchant": "Merchant_A", "year": 2023},
        objective="min",
        output_kind="single_string",
    )

    with pytest.raises(IntentCompilationError, match="steering"):
        compile_semantic_intent(intent)


def test_compile_card_scheme_extreme_across_schemes():
    intent = FeeIntent(
        operation="card_scheme_extreme",
        params={"amount": 10.0},
        objective="max",
        output_kind="comma_list",
    )

    spec = compile_semantic_intent(intent)

    fr = spec.measure.interpretation.fee_rules
    assert fr.group_by == "card_scheme"
    assert fr.group_extreme == "argmax"
    assert fr.reducer == "mean"


def test_card_scheme_extreme_with_named_scheme_compiles_fee_at_amount():
    intent = FeeIntent(
        operation="card_scheme_extreme",
        params={"amount": 100.0, "card_scheme": "GlobalCard", "is_credit": True},
        objective="min",
        output_kind="decimal",
        decimals=6,
    )

    spec = compile_semantic_intent(intent)

    fr = spec.measure.interpretation.fee_rules
    assert fr.group_by is None
    assert "card_scheme" in fr.context_dims
    assert spec.output.kind == "decimal"


def test_compile_affected_merchants():
    intent = FeeIntent(
        operation="affected_merchants",
        params={"fee_id": 7, "year": 2023},
        affected_mode="losers_only",
        output_kind="comma_list",
    )
    spec = compile_semantic_intent(intent)
    pay = spec.measure.interpretation.payments
    assert pay.primitive == "affected_merchants"
    assert pay.affected_mode == "losers_only"


def test_compile_fee_rate_delta():
    intent = FeeIntent(
        operation="fee_rate_delta",
        params={"fee_id": 7, "new_value": 1.5, "year": 2023, "merchant": "Merchant_A"},
        delta_basis="rate",
        output_kind="decimal",
        decimals=14,
    )
    spec = compile_semantic_intent(intent)
    pay = spec.measure.interpretation.payments
    assert pay.primitive == "period_fee_rate_delta"
    assert pay.delta_basis == "rate"


def test_compile_merchant_mcc_delta():
    intent = FeeIntent(
        operation="merchant_mcc_delta",
        params={"merchant": "Merchant_A", "new_mcc": 5411, "year": 2023},
        output_kind="decimal",
        decimals=6,
    )
    spec = compile_semantic_intent(intent)
    assert spec.measure.interpretation.payments.primitive == "mcc_change_fee_delta"


def test_fee_compiler_preserves_zero_decimal_precision():
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023},
        output_kind="decimal",
        decimals=0,
    )

    spec = compile_semantic_intent(intent)

    assert spec.output.decimals == 0


def test_compile_rejects_conflicting_missing_params():
    with pytest.raises(Exception):
        FeeIntent(operation="period_total_fees", params={"year": 2023})
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023},
        output_kind="decimal",
        decimals=2,
    )
    # Irrelevant nested structures already rejected at intent layer; compiler
    # rejects unknown operation axes by construction.
    assert compile_semantic_intent(intent).measure.params["year"] == 2023


def test_fee_compiler_does_not_accept_task_id():
    intent = FeeIntent(
        operation="period_total_fees",
        params={"merchant": "Merchant_A", "year": 2023},
        output_kind="decimal",
        decimals=2,
    )
    with pytest.raises(TypeError):
        compile_semantic_intent(intent, task_id="1")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("operation", "params", "kwargs", "legacy"),
    [
        (
            "period_total_fees",
            {"merchant": "Merchant_A", "year": 2023, "month": 1},
            {"fee_reducer": "sum_all_matching", "output_kind": "decimal", "decimals": 2},
            InterpretationSpec(
                name="parity_total",
                population="payments",
                payments=PaymentsSpec(primitive="period_total_fees", reducer="sum_all_matching"),
                output=OutputSpec(kind="decimal", decimals_default=2),
                manual_citation="manual fee formula",
            ),
        ),
        (
            "fee_at_amount",
            {"amount": 100.0, "card_scheme": "NexPay", "account_type": "H"},
            {"output_kind": "decimal", "decimals": 6},
            InterpretationSpec(
                name="parity_avg",
                population="fee_rules",
                fee_rules=FeeRulesSpec(
                    context_dims=["card_scheme", "account_type"],
                    value="fee_at_amount",
                    reducer="mean",
                    wildcard_policy="manual",
                ),
                output=OutputSpec(kind="decimal", decimals_default=6),
                manual_citation="manual fee formula",
            ),
        ),
        (
            "fee_rate_delta",
            {"merchant": "Merchant_A", "year": 2023, "fee_id": 1, "new_value": 1.0},
            {"delta_basis": "rate", "output_kind": "decimal", "decimals": 14},
            InterpretationSpec(
                name="parity_delta",
                population="payments",
                payments=PaymentsSpec(
                    primitive="period_fee_rate_delta",
                    reducer="sum_all_matching",
                    delta_basis="rate",
                ),
                output=OutputSpec(kind="decimal", decimals_default=14),
                manual_citation="manual fee formula",
            ),
        ),
    ],
)
def test_compiled_fee_intent_matches_legacy_adapter_structure(operation, params, kwargs, legacy):
    intent = FeeIntent(operation=operation, params=params, **kwargs)
    compiled = compile_semantic_intent(intent, guidelines="")
    adapted = adapt_interpretation_spec(legacy, params=params, guidelines="")
    assert compiled.measure.interpretation.population == adapted.measure.interpretation.population
    assert compiled.source.table == adapted.source.table
    assert compiled.measure.params == adapted.measure.params
    if legacy.population == "payments":
        assert compiled.measure.interpretation.payments.primitive == adapted.measure.interpretation.payments.primitive
        assert compiled.measure.interpretation.payments.reducer == adapted.measure.interpretation.payments.reducer
    else:
        assert compiled.measure.interpretation.fee_rules.reducer == adapted.measure.interpretation.fee_rules.reducer
        assert set(compiled.measure.interpretation.fee_rules.context_dims) == set(
            adapted.measure.interpretation.fee_rules.context_dims
        )


def test_real_data_parity_fee_at_amount_and_period_total():
    data_dir = Path(os.getenv(
        "DABSTEP_CONTEXT_DIR",
        "/Users/abruzz1/code/dabstep-memorylake-agent-official/data/context",
    ))
    if not data_dir.exists():
        pytest.skip("real DABStep context data required for fee parity")

    data = load_dabstep_data(data_dir)

    # Mechanism parity: compact intent vs hand-built InterpretationSpec on the
    # same scalar params. No benchmark answers are stored or compared.
    avg_params = {"amount": 100.0, "card_scheme": "NexPay", "account_type": "H"}
    avg_intent = FeeIntent(
        operation="fee_at_amount",
        params=avg_params,
        output_kind="decimal",
        decimals=6,
    )
    avg_legacy = InterpretationSpec(
        name="parity_avg",
        population="fee_rules",
        fee_rules=FeeRulesSpec(
            context_dims=["card_scheme", "account_type"],
            value="fee_at_amount",
            reducer="mean",
        ),
        output=OutputSpec(kind="decimal", decimals_default=6),
        manual_citation="manual fee formula",
    )
    avg_compiled = compile_semantic_intent(avg_intent, guidelines="Answer rounded to 6 decimals.")
    avg_adapted = adapt_interpretation_spec(
        avg_legacy, params=avg_params, guidelines="Answer rounded to 6 decimals."
    )
    assert execute_analysis(data, avg_compiled).formatted_value == execute_analysis(
        data, avg_adapted
    ).formatted_value

    # Period total for a merchant that exists in the public sample.
    merchants = data.merchants["merchant"].astype(str).tolist()
    assert merchants
    merchant = merchants[0]
    total_params = {"merchant": merchant, "year": 2023}
    total_intent = FeeIntent(
        operation="period_total_fees",
        params=total_params,
        fee_reducer="sum_all_matching",
        output_kind="decimal",
        decimals=2,
    )
    total_legacy = InterpretationSpec(
        name="parity_total",
        population="payments",
        payments=PaymentsSpec(primitive="period_total_fees", reducer="sum_all_matching"),
        output=OutputSpec(kind="decimal", decimals_default=2),
        manual_citation="manual fee formula",
    )
    total_compiled = compile_semantic_intent(total_intent, guidelines="")
    total_adapted = adapt_interpretation_spec(total_legacy, params=total_params, guidelines="")
    left = execute_analysis(data, total_compiled).formatted_value
    right = execute_analysis(data, total_adapted).formatted_value
    assert left == right
