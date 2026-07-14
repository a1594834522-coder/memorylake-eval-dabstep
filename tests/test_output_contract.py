from dabstep_agent_pydantic.output_contract import parse_guidelines
from dabstep_agent_pydantic.output_contract import validate_output_contract


def _feedback(answer: str, guidelines: str) -> str | None:
    return validate_output_contract(answer, parse_guidelines(guidelines))


def test_parse_guidelines_detects_decimal_contract():
    contract = parse_guidelines("Answer must be just a number rounded to 6 decimals.")

    assert contract.decimals == 6
    assert _feedback("12.34", contract.raw_guidelines) is not None
    assert _feedback("12.340000", contract.raw_guidelines) is None


def test_contract_allows_not_applicable_when_guideline_says_so():
    guidelines = "Answer must be just a number rounded to 2 decimals. If not relevant, return Not Applicable."

    assert parse_guidelines(guidelines).not_applicable_allowed is True
    assert _feedback("Not Applicable", guidelines) is None


def test_contract_allows_empty_string_only_when_guideline_says_so():
    assert _feedback("", "If the answer is an empty list, reply with an empty string.") is None
    feedback = _feedback("", "Answer must be a comma separated list.")
    assert feedback is not None and "non-empty" in feedback


def test_contract_flags_non_comma_list_separators():
    guidelines = "Answer must be a list of values in comma separated list, eg: A, B, C."

    assert parse_guidelines(guidelines).list_shape == "comma"
    feedback = _feedback("A and B", guidelines)
    assert feedback is not None and "comma-separated" in feedback
    assert _feedback("A, B", guidelines) is None


def test_contract_validates_bracketed_group_value_list_shape():
    guidelines = "Answer must be a comma separated list of [group: value] entries."

    contract = parse_guidelines(guidelines)

    assert contract.list_shape == "bracketed_group_value"
    assert validate_output_contract("[A: 1], [B: 2]", contract) is None
    feedback = validate_output_contract("A: 1, B: 2", contract)
    assert feedback is not None and "[group: value]" in feedback


def test_contract_accepts_multiple_group_values_inside_one_bracket():
    guidelines = "Answer must be a comma separated list of [group: value] entries."

    assert _feedback("[A: 1, B: 2]", guidelines) is None


def test_contract_preserves_case_sensitive_literal_samples():
    guidelines = "Answer must be exactly yes or NO."

    contract = parse_guidelines(guidelines)

    assert contract.case_sensitive_choices == ("yes", "NO")
    assert validate_output_contract("yes", contract) is None
    assert validate_output_contract("NO", contract) is None
    feedback = validate_output_contract("Yes", contract)
    assert feedback is not None and "case-sensitive" in feedback


def test_scorer_aligned_precision_coarsens_cents_scale_answers():
    from dabstep_agent_pydantic.output_contract import scorer_aligned_precision

    # cents-scale primitive: two decimals regardless of magnitude
    assert scorer_aligned_precision("-2.87721200000000", force_cents=True) == "-2.88"
    assert scorer_aligned_precision("0.05853600000000", force_cents=True) == "0.06"
    # generic numeric answers: coarsen only at |value| >= 1
    assert scorer_aligned_precision("9.767557") == "9.77"
    assert scorer_aligned_precision("-5.24362600000000") == "-5.24"
    # sub-1 ratio answers keep contract precision (scorer abs-tolerance branch)
    assert scorer_aligned_precision("0.315937") == "0.315937"
    assert scorer_aligned_precision("-0.090152") == "-0.090152"


def test_scorer_aligned_precision_leaves_non_bare_decimals_alone():
    from dabstep_agent_pydantic.output_contract import scorer_aligned_precision

    assert scorer_aligned_precision("42") == "42"
    assert scorer_aligned_precision("12.34") == "12.34"
    assert scorer_aligned_precision("Not Applicable") == "Not Applicable"
    assert scorer_aligned_precision("NL") == "NL"
    assert scorer_aligned_precision("36, 51, 65") == "36, 51, 65"
    assert scorer_aligned_precision("TransactPlus:2528.31") == "TransactPlus:2528.31"
    assert scorer_aligned_precision("9.767557%") == "9.767557%"
    assert scorer_aligned_precision("[POS: 89.34, Ecommerce: 92.70]") == "[POS: 89.34, Ecommerce: 92.70]"
