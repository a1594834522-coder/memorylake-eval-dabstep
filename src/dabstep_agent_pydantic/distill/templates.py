"""Template induction: normalize public question text into parameterized templates.

Migrated from the scripts scaffolding into the shipping package; the scripts
re-export from here.
"""

from __future__ import annotations

import re
from typing import Any

MONTH_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z<])\b\d+(?:\.\d+)?\b")
MERCHANT_PATTERN = re.compile(r"(?<!<)\b[A-Z][A-Za-z]+(?:_[A-Za-z]+)+\b")
MCC_DESC_PATTERN = re.compile(r'"[^"\n]{2,80}"')
MCC_DESC_CONTEXT_PATTERN = re.compile(
    r"(?P<prefix>\bMCC description:?\s*)(?P<value>[^,?]+?)(?=,\s*what\b|\s+what\b|[,.?])",
    flags=re.IGNORECASE,
)
ORDINAL_PATTERN = re.compile(r"\b\d+(?:st|nd|rd|th)\b", flags=re.IGNORECASE)
Z_SCORE_PATTERN = re.compile(r"\bZ-Score\s*>\s*\d+(?:\.\d+)?", flags=re.IGNORECASE)
SCHEME_PATTERN = re.compile(
    r"\b(Visa|Mastercard|MasterCard|Amex|American Express|SwiftCharge|QuickCard|GlobalCard|NexPay|TransactPlus)\b",
    re.IGNORECASE,
)
SCHEME_CONTEXT_PATTERN = re.compile(r"(?P<prefix>\bcard scheme\s+)(?P<value>[A-Z][A-Za-z0-9_+-]*)", re.IGNORECASE)
LETTER_CONTEXT_PATTERNS = (
    re.compile(r"\b(account_type\s*=\s*)([A-Za-z])\b", re.IGNORECASE),
    re.compile(r"\b(aci\s*=\s*)([A-Za-z])\b", re.IGNORECASE),
    re.compile(r"\b(account type\s+)([A-Za-z])\b", re.IGNORECASE),
)


def normalize_question(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = MCC_DESC_PATTERN.sub("<MCC_DESC>", normalized)
    normalized = MCC_DESC_CONTEXT_PATTERN.sub(lambda match: f"{match.group('prefix')}<MCC_DESC>", normalized)
    normalized = Z_SCORE_PATTERN.sub("Z-Score > <Z_SCORE_THRESHOLD>", normalized)
    normalized = ORDINAL_PATTERN.sub("<ORDINAL>", normalized)
    normalized = MONTH_PATTERN.sub("<MONTH>", normalized)
    normalized = YEAR_PATTERN.sub("<YEAR>", normalized)
    normalized = MERCHANT_PATTERN.sub("<MERCHANT>", normalized)
    normalized = SCHEME_PATTERN.sub("<SCHEME>", normalized)
    normalized = SCHEME_CONTEXT_PATTERN.sub(lambda match: f"{match.group('prefix')}<SCHEME>", normalized)
    for pattern in LETTER_CONTEXT_PATTERNS:
        normalized = pattern.sub(lambda match: f"{match.group(1)}<LETTER>", normalized)
    normalized = NUMBER_PATTERN.sub("<N>", normalized)
    return normalized


def group_templates(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group task rows by normalized template, largest templates first."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        template = normalize_question(" ".join(str(task["question"]).split()))
        grouped.setdefault(template, []).append(task)
    return dict(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])))
