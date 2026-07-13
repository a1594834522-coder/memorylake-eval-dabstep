from pathlib import Path

from dabstep_agent_pydantic.agent import DABStepDeps
from dabstep_agent_pydantic.agent import build_helper_section
from dabstep_agent_pydantic.agent import build_instructions
from dabstep_agent_pydantic.analysis_plan import AnalysisPlan
from dabstep_agent_pydantic.python_tool import PythonWorkspace


def _deps(tmp_path: Path, plan: AnalysisPlan | None) -> DABStepDeps:
    return DABStepDeps(
        data_dir=tmp_path,
        workspace=PythonWorkspace(tmp_path / "workspace"),
        file_summary="(empty)",
        analysis_plan=plan,
    )


def test_helper_section_is_scoped_to_plan_helpers(tmp_path):
    plan = AnalysisPlan(
        task_family="customer_fraud_metrics",
        selected_route_ids=["fraud_and_customer_semantics", "output_contracts"],
        recommended_helpers=["fraud_rate_by_volume", "repeat_customer_percentage", "format_decimal_places"],
    )
    section = build_helper_section(_deps(tmp_path, plan))

    assert "fraud_rate_by_volume" in section
    assert "repeat_customer_percentage" in section
    # Fee-simulation helpers and typed tools stay out of non-fee prompts.
    assert "merchant_mcc_fee_delta_for_year" not in section
    assert "compute_mcc_fee_delta" not in section
    assert "compute_total_fees" not in section
    # Core lines are always present.
    assert "load_dabstep_data" in section
    assert "format_decimal_places" in section


def test_helper_section_includes_typed_tools_for_fee_simulation_routes(tmp_path):
    plan = AnalysisPlan(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation", "output_contracts"],
        recommended_helpers=["merchant_mcc_fee_delta_for_year", "total_fees_for_merchant_period"],
    )
    section = build_helper_section(_deps(tmp_path, plan))

    assert "compute_total_fees" in section
    assert "compute_mcc_fee_delta" in section
    assert "merchant_mcc_fee_delta_for_year" in section
    assert "fraud_rate_by_volume" not in section


def test_helper_section_falls_back_to_full_catalog_without_plan(tmp_path):
    section = build_helper_section(_deps(tmp_path, None))
    assert "compute_total_fees" in section
    assert "fraud_rate_by_volume" in section
    assert "field_domain_values" in section
    assert "match_count_summary" in section


def test_helper_section_documents_match_count_summary_for_fee_routes(tmp_path):
    plan = AnalysisPlan(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation", "fee_matching", "output_contracts"],
        recommended_helpers=["match_count_summary", "total_fees_for_merchant_period"],
    )
    section = build_helper_section(_deps(tmp_path, plan))

    assert "match_count_summary" in section
    assert "fee matching" in section


def test_build_instructions_embeds_scoped_helper_section(tmp_path):
    plan = AnalysisPlan(
        task_family="customer_fraud_metrics",
        selected_route_ids=["fraud_and_customer_semantics"],
        recommended_helpers=["fraud_rate_by_volume"],
    )
    instructions = build_instructions(_deps(tmp_path, plan))
    assert "fraud_rate_by_volume" in instructions
    assert "compute_mcc_fee_delta" not in instructions


def test_base_instructions_require_filter_self_checks():
    from dabstep_agent_pydantic.agent import BASE_INSTRUCTIONS

    assert "assert_nonempty" in BASE_INSTRUCTIONS
    assert "check_categorical" in BASE_INSTRUCTIONS
    assert "match_count_summary" in BASE_INSTRUCTIONS
    assert "row counts" in BASE_INSTRUCTIONS
