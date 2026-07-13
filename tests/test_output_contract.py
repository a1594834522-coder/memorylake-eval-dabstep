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
