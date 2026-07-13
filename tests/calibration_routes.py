"""Route trigger patterns for calibration tests (retired from runtime)."""

import re

ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("field_domain_values", re.compile(r"possible values.*account_type|account_type.*possible values", re.IGNORECASE)),
    ("unique_merchant_count", re.compile(r"how many.*unique merchants", re.IGNORECASE)),
    (
        "high_value_repeat_customer_percentage",
        re.compile(r"percentage of high-value transactions.*repeat customers", re.IGNORECASE),
    ),
    ("most_expensive_mcc", re.compile(r"most expensive MCC.*in general", re.IGNORECASE)),
    ("fee_ids_by_attributes", re.compile(r"fee ID or IDs.*account_type\s*=.*aci\s*=", re.IGNORECASE)),
    ("average_scheme_fee", re.compile(r"average fee.*card scheme.*transaction value", re.IGNORECASE)),
    ("aci_fee_extreme", re.compile(r"(most|least) expensive.*ACI|ACI.*(most|least) expensive", re.IGNORECASE)),
    ("mcc_fee_delta", re.compile(r"changed its MCC code", re.IGNORECASE)),
    ("total_fees", re.compile(r"total fees .* paid in", re.IGNORECASE)),
    ("fee_account_type_change_affected_merchants", re.compile(r"only applied to account type.*affected", re.IGNORECASE)),
    ("fee_affected_merchants", re.compile(r"which merchants were affected by the Fee", re.IGNORECASE)),
    ("relative_fee_delta", re.compile(r"In [A-Za-z]+ 20\d{2} what delta.*relative fee", re.IGNORECASE)),
    ("relative_fee_rate_delta", re.compile(r"In the year 20\d{2} what delta.*relative fee", re.IGNORECASE)),
    ("applicable_fee_ids", re.compile(r"applicable Fee IDs", re.IGNORECASE)),
    ("aci_fraud_optimization", re.compile(r"fraudulent transactions.*lowest possible fees", re.IGNORECASE)),
)




def match_route(question: str) -> str | None:
    for route_id, pattern in ROUTE_PATTERNS:
        if pattern.search(question):
            return route_id
    return None
