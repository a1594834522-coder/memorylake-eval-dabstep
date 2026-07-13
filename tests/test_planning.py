from dabstep_agent_pydantic.planning import plan_task


def _plan(question: str, guidelines: str | None = "Answer with a number."):
    return plan_task(question=question, guidelines=guidelines, route_cards=[])


def test_plan_flags_percentage_of_ambiguity_axis():
    plan = _plan("What percentage of high-value transactions came from repeat customers?")

    assert "percentage_scope" in plan.ambiguity_axes


def test_plan_flags_unqualified_fraud_rate_axis():
    plan = _plan("What is the fraud rate for online transactions?")

    assert "fraud_rate_basis" in plan.ambiguity_axes


def test_plan_does_not_flag_explicit_volume_fraud_rate_axis():
    plan = _plan("What is the fraud rate by fraudulent EUR volume over total EUR volume?")

    assert "fraud_rate_basis" not in plan.ambiguity_axes


def test_plan_flags_average_fee_axis():
    plan = _plan("What is the average fee for this card scheme?")

    assert "average_fee_basis" in plan.ambiguity_axes


def test_plan_flags_general_expensive_fee_axis():
    plan = _plan("What is the most expensive category in general for a transaction of 100 euros?")

    assert "fee_extreme_basis" in plan.ambiguity_axes
