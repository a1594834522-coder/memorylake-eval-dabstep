from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dabstep_agent_pydantic.analysis_executor import execute_analysis
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import InterpretationMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.dabstep_core import load_dabstep_data
from dabstep_agent_pydantic.distill.emit import load_generated_skills
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.distill.spec import OutputSpec
from dabstep_agent_pydantic.distill.spec import PaymentsSpec
from dabstep_agent_pydantic.fee_spec_adapter import adapt_interpretation_spec


def test_adapter_preserves_fee_rule_axes_and_typed_params():
    legacy = InterpretationSpec(
        name="fee-mean",
        population="fee_rules",
        fee_rules=FeeRulesSpec(
            context_dims=["card_scheme", "account_type"],
            reducer="mean",
            wildcard_policy="manual",
        ),
        output=OutputSpec(kind="decimal", decimals_default=6),
        manual_citation="manual fee formula",
    )

    adapted = adapt_interpretation_spec(
        legacy,
        params={"card_scheme": "TransactPlus", "account_type": "H", "amount": 100.0},
        guidelines="Answer must be rounded to 4 decimals.",
    )

    assert isinstance(adapted.measure, InterpretationMeasure)
    assert adapted.measure.interpretation == legacy
    assert adapted.measure.params == {
        "card_scheme": "TransactPlus",
        "account_type": "H",
        "amount": 100.0,
    }
    assert adapted.output.decimals == 4
    assert adapted.source.table == "fees"


def test_adapter_preserves_payment_counterfactual_axes():
    legacy = InterpretationSpec(
        name="rate-delta",
        population="payments",
        payments=PaymentsSpec(
            primitive="period_fee_rate_delta",
            reducer="sum_all_matching",
            delta_basis="rate",
        ),
        output=OutputSpec(kind="decimal", decimals_default=14),
        manual_citation="manual fee formula",
    )

    adapted = adapt_interpretation_spec(
        legacy,
        params={
            "merchant": "Merchant_A",
            "year": 2023,
            "fee_id": 7,
            "new_value": 12.5,
        },
        guidelines="",
    )

    assert adapted.source.table == "payments"
    assert adapted.measure.interpretation.payments.delta_basis == "rate"
    assert adapted.output.decimals == 14


def test_adapter_rejects_unknown_or_nested_parameters():
    legacy = InterpretationSpec(
        name="domain",
        population="payments",
        payments=PaymentsSpec(primitive="field_domain_values"),
        output=OutputSpec(kind="string_list"),
        manual_citation="manual schema",
    )

    with pytest.raises(ValueError, match="unsupported interpretation parameter"):
        adapt_interpretation_spec(legacy, params={"python_code": "df.sum()"}, guidelines="")

    with pytest.raises(ValueError, match="scalar"):
        adapt_interpretation_spec(legacy, params={"field": {"nested": "value"}}, guidelines="")


def test_interpretation_measure_requires_matching_source_and_output():
    legacy = InterpretationSpec(
        name="rate-delta",
        population="payments",
        payments=PaymentsSpec(primitive="period_fee_rate_delta", delta_basis="rate"),
        output=OutputSpec(kind="decimal", decimals_default=14),
        manual_citation="manual fee formula",
    )
    measure = InterpretationMeasure(
        interpretation=legacy,
        params={"merchant": "Merchant_A", "year": 2023, "fee_id": 7, "new_value": 1.0},
    )

    with pytest.raises(ValidationError, match="interpretation source"):
        AnalysisSpec(
            source=SourceSpec(table="fees"),
            measure=measure,
            output=AnalysisOutputContract(kind="decimal", decimals=14),
        )

    with pytest.raises(ValidationError, match="interpretation output"):
        AnalysisSpec(
            source=SourceSpec(table="payments"),
            measure=measure,
            output=AnalysisOutputContract(kind="integer"),
        )


def test_all_persisted_generated_skills_match_new_executor_byte_for_byte():
    data_dir = Path(os.getenv("DABSTEP_CONTEXT_DIR", "data/context"))
    tasks_path = Path(os.getenv("DABSTEP_TASKS_PATH", "data/tasks.json"))
    skills_dir = Path(os.getenv("DABSTEP_GENERATED_SKILLS_DIR", "artifacts/skills"))
    if not data_dir.exists() or not tasks_path.exists() or not skills_dir.exists():
        pytest.skip("real DABStep data and generated skills are required for adapter parity")

    data = load_dabstep_data(data_dir)
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    skills = load_generated_skills(skills_dir)
    assert skills

    comparisons: dict[str, int] = {}
    for skill in skills:
        comparisons[skill.skill_id] = 0
        for task in tasks:
            question = str(task["question"])
            guidelines = str(task.get("guidelines") or "")
            analysis_spec = skill.to_analysis_spec(data, question, guidelines)
            if analysis_spec is None:
                continue
            legacy_answer = skill.solve(data, question, guidelines)
            new_answer = execute_analysis(data, analysis_spec).formatted_value
            assert new_answer == legacy_answer, f"{skill.skill_id} diverged for {task['task_id']}"
            comparisons[skill.skill_id] += 1
            break

    assert all(count == 1 for count in comparisons.values()), comparisons
