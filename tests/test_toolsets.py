import pytest

from dabstep_agent_pydantic.planning import PlanDecision
from dabstep_agent_pydantic.toolsets import toolset_names_for_plan


@pytest.fixture(autouse=True)
def _no_learned_skills(monkeypatch):
    # These tests pin the hand-written toolset selection; learned-skill
    # artifacts present in the working tree must not leak in.
    monkeypatch.setenv("DABSTEP_GENERATED_SKILLS", "off")


def test_fee_simulation_plan_selects_common_and_fee_toolsets():
    plan = PlanDecision(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation", "output_contracts"],
        toolset_ids=["fee_simulation", "output_contracts"],
        verification_focus=["counterfactual"],
    )

    assert toolset_names_for_plan(plan) == [
        "common",
        "fee_simulation",
        "output_contracts",
    ]


def test_fraud_plan_does_not_expose_fee_simulation_toolset():
    plan = PlanDecision(
        task_family="customer_fraud_metrics",
        selected_route_ids=["fraud_and_customer_semantics", "output_contracts"],
        toolset_ids=["fraud_and_customer_semantics", "output_contracts"],
        verification_focus=["denominator"],
    )

    assert "fee_simulation" not in toolset_names_for_plan(plan)


def test_fee_simulation_suppresses_fee_matching_to_avoid_duplicate_tool_names():
    plan = PlanDecision(
        task_family="fee_simulation",
        selected_route_ids=["fee_simulation", "fee_matching", "output_contracts"],
        toolset_ids=["fee_simulation", "fee_matching", "output_contracts"],
        verification_focus=["counterfactual"],
    )

    assert toolset_names_for_plan(plan) == [
        "common",
        "fee_simulation",
        "output_contracts",
    ]
