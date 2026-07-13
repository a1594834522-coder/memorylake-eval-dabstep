from dabstep_agent_pydantic.python_tool import PythonWorkspace


def test_workspace_assert_nonempty_raises_with_label_for_empty_frame(tmp_path):
    workspace = PythonWorkspace(tmp_path / "workspace")

    result = workspace.execute(
        """
import pandas as pd
filtered = pd.DataFrame({"category": []})
assert_nonempty(filtered, "filtered payments")
"""
    )

    assert result.ok is False
    assert result.error_type == "ValueError"
    assert "filtered payments" in result.error
    assert "empty" in result.error


def test_workspace_check_categorical_reports_available_values(tmp_path):
    workspace = PythonWorkspace(tmp_path / "workspace")

    result = workspace.execute(
        """
import pandas as pd
frame = pd.DataFrame({"category": ["A", "B"]})
check_categorical(frame, "category", ["A", "C"])
"""
    )

    assert result.ok is False
    assert result.error_type == "ValueError"
    assert "category" in result.error
    assert "C" in result.error
    assert "A" in result.error and "B" in result.error


def test_workspace_invariant_helpers_pass_for_valid_inputs(tmp_path):
    workspace = PythonWorkspace(tmp_path / "workspace")

    result = workspace.execute(
        """
import pandas as pd
frame = pd.DataFrame({"category": ["A", "B"]})
assert_nonempty(frame, "filtered payments")
check_categorical(frame, "category", ["A", "B"])
print("ok")
"""
    )

    assert result.ok is True
    assert result.output.strip() == "ok"


def test_workspace_exposes_match_count_summary(tmp_path):
    workspace = PythonWorkspace(tmp_path / "workspace")

    result = workspace.execute(
        """
import pandas as pd
from dabstep import match_count_summary
payments = pd.DataFrame([{
    "card_scheme": "SwiftCharge",
    "is_credit": True,
    "aci": "A",
    "intracountry": True,
    "eur_amount": 10.0,
}])
fees = [{
    "ID": 1,
    "card_scheme": "SwiftCharge",
    "account_type": [],
    "capture_delay": None,
    "monthly_fraud_level": None,
    "monthly_volume": None,
    "merchant_category_code": [],
    "is_credit": True,
    "aci": [],
    "fixed_amount": 0,
    "rate": 0,
    "intracountry": None,
}]
summary = match_count_summary(
    payments,
    fees,
    context_base={
        "account_type": "D",
        "capture_delay": "7",
        "merchant_category_code": 1234,
        "monthly_volume": 10.0,
        "monthly_fraud_rate_pct": 0.0,
    },
)
print(summary["warning"])
"""
    )

    assert result.ok is True
    assert result.output.strip() == "no transaction matches more than one rule — filters may be too strict"


def test_workspace_systemexit_is_contained(tmp_path):
    workspace = PythonWorkspace(tmp_path / "workspace")

    for snippet in ("import sys; sys.exit(0)", "exit(0)", "raise SystemExit(2)"):
        result = workspace.execute(snippet)
        assert result.ok is False, snippet
        assert result.error_type == "SystemExit", snippet
        assert "do not call exit()" in result.error

    # The workspace stays usable afterwards.
    assert workspace.execute("print('alive')").output.strip() == "alive"
